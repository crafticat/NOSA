"""CPU tests of rung I4DC (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the arms
L4PTFDGUDCD and L4PTFDGUDKWCD (the I4CC arms + the draft passes' integer / mask / index / bit-copy helpers from
torch.compile / Inductor), benchmarks/Efficiency/r2_i4dc_compile.py; the cell r2_i4dc_cell.py; the step script
r2_i4dc_step.sh. The world: tests_cpu/i4dc_world.py over i4cc_world (FakeGraphsDCD: the compiled calls as kernel nodes of
the joint and of the split draft captures).
CUDA GRAPHS DO NOT RUN ON THE CPU AND INDUCTOR'S CPU BACKEND IS NOT ITS GPU BACKEND: these tests establish the plumbing,
the bit equality on the CPU backend (the same graphs and fusion decisions, another code generator), no graph break, the
compile-event record and the gates; only the GPU miniature i4dm2 establishes Triton's bits and capture legality inside
the draft graphs, and the timing cells the saving.

GATE 0    test_i4dc_gate0_files: the tree = I4CC d75a946 + the I4DC files + ONE module-level declaration block in each
          older Gate-0 test (test_wsa.py, test_wsb.py, test_wsc.py, test_i2.py, test_i3.py, test_i4a.py, test_i4g.py,
          test_i4d.py, test_i4k.py, test_i4u.py, test_i4h.py, test_i4s.py, test_i4cc.py) and ONE line of test_i4cc's
          name scan; nosi/ and every other file are d75a946's bytes; no kernel source; the fork is fia6's blob; the compile
          module imports torch, time and r2_i4cc_compile only, no hand-written kernel in the I4DC modules; nobody else
          names them.
HELPERS   test_draft_helpers_equal_lean: every compiled draft helper (Inductor, CPU) = THE LEAN ROW'S OWN OPS (the
          64-row torch.where write, the expanded masks: r2_i4a_lean.py:154-185, copied op for op here) bit for bit, on
          adversarial inputs (bf16 NaN payloads in the buffer, the row, the live bias and the buffer bias; rows 0 and 63;
          inactive rows; past and not past; pool and no pool, i.e. a non-contiguous scratch); in place (the same
          addresses), the scratch's rows past the window untouched. REPORT: the generated kernels per helper.
EQUAL     test_draft_compiled_equals (timing B 8, check B 8, timing B 2 rowgroup, both D arms): the run = L4PTFD BIT FOR
          BIT and = its C arm (L4PTFDGUDC / L4PTFDGUDKWC) in everything but the compile record; the cell's gates pass on
          the CPU evidence (EQF, LEAN, HOIST, COUNTS, GRAPH, DGRAPH, the keep gates, COMPILE). Preconditions (L34): every
          one-row layer through the compiled draft helpers, draft replays, every compile event before the arm's first
          capture.
PLANTS    test_draft_plants_caught: lost_dwrite (EQF), dlate (COMPILE: a compile inside a draft capture), a draft fallback
          (COMPILE).
WIRING    test_draft_refusals.
CELL      test_i4dc_cell_arguments, test_i4dc_cell_pure_parts, test_i4dc_registration_state, test_step_script_i4dc.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rfEs benchmarks/Efficiency/tests_cpu/test_i4dc.py
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
BASE_SHA = "d75a946dbea718062cd6e226e0ad441930b79062"         # nosi-r2-i4cc: I4CC with the lead's registration
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i4dc"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
E_ = "benchmarks/Efficiency/"
I4DC_NEW = (E_ + "r2_i4dc_compile.py", E_ + "r2_i4dc_cell.py", E_ + "r2_i4dc_step.sh", T_ + "i4dc_world.py",
            T_ + "test_i4dc.py")
I4DC_DECL = tuple(T_ + "test_%s.py" % n for n in ("wsa", "wsb", "wsc", "i2", "i3", "i4a", "i4g", "i4d", "i4k", "i4u",
                                                     "i4h", "i4s", "i4cc"))
I4DC_EDITED = {T_ + "test_i4cc.py": "test_i4cc_gate0_files"}
I4CC_FILES = (E_ + "r2_i4cc_compile.py", E_ + "r2_i4cc_cell.py", E_ + "r2_i4cc_step.sh", T_ + "i4cc_world.py")
COMPILE_IMPORTS = {"torch", "time", "r2_i4cc_compile"}
HAND_KERNEL = ("import triton", "@triton.jit", "triton.jit", "load_inline", "cpp_extension", "tl.load", "tl.store")
# RECON7 (nosi-r2-recon7: I4DC 2e9e497 + the measurement cells of the reconciliation): its files and its byte copies
# (its own Gate 0: tests_cpu/test_recon7.py); I4DC_LATER: its modules import the I4DC cell (scan (e) skips them)
I4DC_NEW = I4DC_NEW + (E_ + "r2_recon7_protocol.py", E_ + "r2_recon7_probe.py", E_ + "r2_recon7_cell.py",
                       E_ + "r2_recon7_streams.py", E_ + "r2_recon7_mem.py", E_ + "r2_recon7_step.sh",
                       E_ + "r2_i4c_cell.py", E_ + "r2_i4m_cell.py", E_ + "r2_wsm_cell.py", T_ + "test_recon7.py",
                       "docs/NOTES_recon7.md")                                                    # RECON7
I4DC_LATER = ("r2_recon7_protocol.py", "r2_recon7_probe.py", "r2_recon7_cell.py", "r2_recon7_streams.py",
              "r2_recon7_mem.py", "r2_i4c_cell.py", "r2_i4m_cell.py", "r2_wsm_cell.py")           # RECON7
# BF1 (nosi-r2-bf1: RECON7 cd0f0c5 + the draft-issued background fetch): its files (its own Gate 0: tests_cpu/test_bf1.py);
# I4DC_LATER: its modules import the I4DC modules (scan (e) skips them)
I4DC_NEW = I4DC_NEW + (E_ + "r2_bf1_fetch.py", E_ + "r2_bf1_cell.py", E_ + "r2_bf1_step.sh", T_ + "test_bf1.py")  # BF1
I4DC_LATER = I4DC_LATER + ("r2_bf1_fetch.py", "r2_bf1_cell.py")                                   # BF1


class Evidence(UserWarning):
    """An evidence line (the CPU template runs pytest without -s: the warnings summary carries these lines)."""


def _ev(msg: str) -> None:
    print(msg)
    warnings.warn(Evidence(msg), stacklevel=2)


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X = _load("i4dc_world")
TS = _load("test_i4s")
CD, CC, L, N0 = X.CD, X.CC, X.L, X.N0


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i4dc_cell
    return r2_i4dc_cell


# ------------------------------------------------------------------ GATE 0
def test_i4dc_gate0_files():
    if not _have(BASE_SHA):
        pytest.skip("no git or no commit %s in %s" % (BASE_SHA, ROOT))
    changed = {x for x in _git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    changed |= {x for x in _git("ls-files", "--others", "--exclude-standard", "--", ".",
                                ":(exclude,glob)**/__pycache__/**").stdout.split("\n") if x}
    assert changed <= set(I4DC_NEW) | set(I4DC_DECL), sorted(changed - set(I4DC_NEW) - set(I4DC_DECL))   # (a)
    assert all((ROOT / f).is_file() for f in I4DC_NEW), [f for f in I4DC_NEW if not (ROOT / f).is_file()]
    assert _git("diff", "--name-only", BASE_SHA, "--", "nosi").stdout.strip() == ""        # (b) nosi: d75a946's
    assert _git("diff", "--name-only", BASE_SHA, "--", *I4CC_FILES).stdout.strip() == ""   # the I4CC files: its bytes
    for f in I4DC_DECL:                                                   # (c) one I4DC block each (+ one line)
        old, new = TS._fn_units(_git("show", "%s:%s" % (BASE_SHA, f)).stdout), TS._fn_units((ROOT / f).read_text())
        allowed = {I4DC_EDITED[f]} if f in I4DC_EDITED else set()
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= allowed, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
        d = _git("diff", "-U0", BASE_SHA, "--", f).stdout.splitlines()
        body = [x for x in d if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        minus = [x for x in body if x.startswith("-")]
        assert len(minus) <= (1 if f in I4DC_EDITED else 0), (f, minus)
        assert any("r2_i4dc_compile.py" in x for x in body) and any("I4DC" in x for x in body), (f, body[:5])
    assert _git("diff", "--name-only", BASE_SHA, "--", "*.cu", "*.cuh", "*.cpp").stdout.strip() == ""   # no kernel
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(EFF / "fork_vllm_unified_attention.py")).stdout.strip() == FORK_BLOB
    tree = ast.parse((EFF / "r2_i4dc_compile.py").read_text())           # (d) the compile module
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert {m for m in mods if not m.startswith("torch.")} == COMPILE_IMPORTS, mods
    for name in ("r2_i4dc_compile.py", "r2_i4dc_cell.py"):              # NO HAND-WRITTEN KERNEL
        text = (EFF / name).read_text()
        assert not [w for w in HAND_KERNEL if w in text], (name, [w for w in HAND_KERNEL if w in text])
    for p in list((ROOT / "nosi/nosi").glob("*.py")) + list(EFF.glob("*.py")):     # (e) nobody else names it
        if p.name.startswith("r2_i4dc_") or p.name in I4DC_LATER:
            continue
        assert "r2_i4dc" not in p.read_text(), p.name


# ------------------------------------------------------------------ the draft helpers, alone
def _dinputs(seed: int, pool: bool):
    """Adversarial helper inputs at world-like shapes: H 2, B 5, T 8 (topk; tail 7), 64-row blocks, D 4, a cis row
    count of 70, 12 slots per engine with the pool (the attended view: the first T x 64 rows)."""
    g = torch.Generator().manual_seed(seed)
    H, B, T, bs, D, NC, C = 2, 5, 8, 64, 4, 70, 12
    ri = lambda lo, hi, shape: torch.randint(lo, hi, shape, generator=g)   # noqa: E731
    bf = lambda *shape: torch.randn(*shape, generator=g).to(torch.bfloat16)   # noqa: E731
    qb = ri(3, 9, (B,))
    sel = ri(-1, 14, (H, B, T))
    sel[0, 1, :] = -1                                                    # a head with no valid id
    sel[1, 2, 3] = int(qb[2])                                            # a head that selects block qb
    p = torch.tensor([0, 63, 64 * 5 + 17, 64 * 2 + 63, 129])            # rows 0 and 63 of a block
    act = torch.tensor([True, True, False, True, True])
    kb, vb, bb = bf(B, bs, H, D), bf(B, bs, H, D), bf(B, bs, H)
    kb.view(torch.int16)[1, 63, 0, 0] = 0x7FC1                          # NaN payloads: must survive bit for bit
    bb.view(torch.int16)[0, 5, 1] = 0x7FC3
    k_r, v_r = bf(B, H, D), bf(B, H, D)
    k_r.view(torch.int16)[3, 1, 2] = 0x7FC5                             # a NaN payload in an active row
    rows = C * bs if pool else T * bs
    live_full = bf(B, rows, H)
    live_full.view(torch.int16)[2, 7, 0] = 0x7FC7
    scratch = bf(B, rows, H)
    bias = bf(B, bs, H)
    bias.view(torch.int16)[4, 9, 1] = 0x7FC9
    return dict(H=H, B=B, T=T, bs=bs, NC=NC, qb3=qb.view(1, B, 1), sel=sel, p=p, act=act, kb=kb, vb=vb, bb=bb,
                cis=bf(B, NC, H), k_r=k_r, v_r=v_r, live_full=live_full, scratch=scratch, bias=bias,
                skip=torch.rand(H, B, T, generator=g) > 0.6, past3=(torch.rand(1, B, 1, generator=g) > 0.5),
                okv=torch.rand(B, bs, 1, generator=g) > 0.3, ncmt=torch.rand(B, bs, 1, generator=g) > 0.4,
                maskb=torch.tensor(-30000.0).to(torch.bfloat16), tail=T - 1, pool=pool)


def _lean(I):
    """THE LEAN ROW'S OPS (r2_i4a_lean.py:154-159, 167-185, 231-235; lp from _lean_pass / _lean_mask), op for op, on
    fresh copies. Returns every output and the written tensors."""
    B, T, bs, H = I["B"], I["T"], I["bs"], I["H"]
    sel, qb3 = I["sel"], I["qb3"]
    kb, vb, bb, scratch = I["kb"].clone(), I["vb"].clone(), I["bb"].clone(), I["scratch"].clone()
    att = scratch[:, :T * bs] if I["pool"] else scratch
    live = I["live_full"][:, :T * bs] if I["pool"] else I["live_full"]
    hasq = (sel == qb3).any(-1, keepdim=True)
    nhq = ~hasq
    inj = (sel > qb3) & nhq
    ti = torch.where(inj, qb3.expand_as(sel), sel).contiguous()
    i = I["p"] % bs
    oh = (torch.arange(bs).view(1, bs) == i.view(B, 1)) & I["act"].view(B, 1)
    prow = I["p"].clamp(max=I["NC"] - 1)
    torch.where(oh.view(B, bs, 1, 1), I["k_r"].unsqueeze(1), kb, out=kb)
    torch.where(oh.view(B, bs, 1, 1), I["v_r"].unsqueeze(1), vb, out=vb)
    torch.where(oh.view(B, bs, 1), I["cis"][torch.arange(B), prow].unsqueeze(1), bb, out=bb)
    m63 = nhq & I["past3"]
    skip = I["skip"]
    mslot = torch.cat((skip[:, :, :I["tail"]], skip[:, :, I["tail"]:] | m63), dim=2)
    torch.where(mslot.permute(1, 2, 0).unsqueeze(2), I["maskb"], live.view(B, T, bs, H), out=att.view(B, T, bs, H))
    vis = I["okv"] & (I["ncmt"] | hasq.permute(1, 2, 0))
    bias_b = torch.where(vis, I["bias"], I["maskb"])
    return dict(hasq=hasq, nhq=nhq, inj=inj, ti=ti, kb=kb, vb=vb, bb=bb, scratch=scratch, bias_b=bias_b)


def _helpers(I, fns):
    """The three draft helpers in the row's order on fresh copies of I (fns: name -> callable), on integer views as the
    D arm calls them. Returns every output, the written tensors and whether every write was in place."""
    B, T, bs, H = I["B"], I["T"], I["bs"], I["H"]
    bits = CD.bits
    kb, vb, bb, scratch = I["kb"].clone(), I["vb"].clone(), I["bb"].clone(), I["scratch"].clone()
    att = scratch[:, :T * bs] if I["pool"] else scratch
    live = I["live_full"][:, :T * bs] if I["pool"] else I["live_full"]
    ptrs = [t.data_ptr() for t in (kb, vb, bb, scratch)]
    hasq, nhq, inj, ti = fns["dpre"](I["sel"], I["qb3"])
    fns["dwrite"](bits(kb), bits(vb), bits(bb), bits(I["cis"]), torch.arange(B), I["p"] % bs,
                  I["p"].clamp(max=I["NC"] - 1), I["act"], bits(I["k_r"]), bits(I["v_r"]))
    bias_b = fns["dmask"](I["skip"], nhq, I["past3"], bits(I["maskb"]), bits(live).view(B, T, bs, H),
                          bits(att).view(B, T, bs, H), I["okv"], I["ncmt"], hasq, bits(I["bias"]),
                          I["tail"]).view(torch.bfloat16)
    same = ptrs == [t.data_ptr() for t in (kb, vb, bb, scratch)]
    return dict(hasq=hasq, nhq=nhq, inj=inj, ti=ti, kb=kb, vb=vb, bb=bb, scratch=scratch, bias_b=bias_b), same


def _kernels(fn, *args):
    """REPORT: the number of C++ kernels Inductor generates for fn on args. A COPY of fn's code object: dynamo keeps
    its cache on the code object, so the copy compiles fresh and leaves the helpers' caches as they are."""
    import types
    from torch._inductor.utils import run_and_get_code
    co = fn.__code__.replace(co_name=fn.__code__.co_name + "_kcount")
    fresh = types.FunctionType(co, fn.__globals__, co.co_name, fn.__defaults__, fn.__closure__)
    _, code = run_and_get_code(torch.compile(fresh, fullgraph=True, dynamic=False), *args)
    return sum(c.count("async_compile.cpp_pybinding") for c in code)


def test_draft_helpers_equal_lean():
    CC.configure()
    comp = {n: CD.compiled_d(n, "inductor") for n in CD.DHELPERS}
    rows = []
    with torch.inference_mode():
        for seed in range(3):
            for pool in (True, False):
                I = _dinputs(seed, pool)
                ref = _lean(I)
                e, se = _helpers(I, CD.DFNS)
                c, sc = _helpers(I, comp)
                assert se and sc                                       # every write in place (the same addresses)
                for got in (e, c):
                    assert set(got) == set(ref)
                    bad = [k for k in ref if not (ref[k].shape == got[k].shape and ref[k].dtype == got[k].dtype
                                                  and torch.equal(CD.bits(ref[k]), CD.bits(got[k])))]
                    assert bad == [], (seed, pool, bad)                # = the lean row's bytes (NaN payloads too)
                if pool:                                               # the scratch past the window: untouched
                    assert torch.equal(CD.bits(c["scratch"][:, I["T"] * I["bs"]:]),
                                       CD.bits(I["scratch"][:, I["T"] * I["bs"]:]))
                rows.append((seed, pool))
    I = _dinputs(0, True)
    B, T, bs, H = I["B"], I["T"], I["bs"], I["H"]
    with torch.inference_mode():
        n = dict(dpre=_kernels(CD.dpre_fn, I["sel"], I["qb3"]),
                 dwrite=_kernels(CD.dwrite_fn, CD.bits(I["kb"].clone()), CD.bits(I["vb"].clone()),
                                 CD.bits(I["bb"].clone()), CD.bits(I["cis"]), torch.arange(B), I["p"] % bs,
                                 I["p"].clamp(max=I["NC"] - 1), I["act"], CD.bits(I["k_r"]), CD.bits(I["v_r"])),
                 dmask=_kernels(CD.dmask_fn, I["skip"], I["skip"][:, :, :1], I["past3"], CD.bits(I["maskb"]),
                                CD.bits(I["live_full"][:, :T * bs]).view(B, T, bs, H),
                                CD.bits(I["scratch"].clone()[:, :T * bs]).view(B, T, bs, H), I["okv"], I["ncmt"],
                                I["skip"][:, :, :1], CD.bits(I["bias"]), I["tail"]))
    _ev("I4DC HELPERS: %d variants (seed, pool) bit-equal to the lean row's ops; generated C++ kernels per helper %s "
        "(the lean row's eager launches: dpre 6, dwrite 4, dmask 7); dynamo %s" % (len(rows), n, CC.dynamo_counts()))


# ------------------------------------------------------------------ the arms
@functools.lru_cache(maxsize=None)
def _run(arm, scen="timing", B=8, gemm="natural"):
    return X.run_arm(arm, scen, B=B, gemm=gemm)


CASES = dict(gudcd_timing=("GUDCD", "timing", 8, "natural"), gudcd_check=("GUDCD", "check", 8, "natural"),
             gudcd_b2=("GUDCD", "timing", 2, "rowgroup"), gudkcd_timing=("GUDKCD", "timing", 8, "natural"),
             gudkcd_check=("GUDKCD", "check", 8, "natural"), gudkcd_b2=("GUDKCD", "timing", 2, "rowgroup"))
C_OF = dict(GUDCD="GUDC", GUDKCD="GUDKC")


@pytest.mark.parametrize("case", list(CASES))
def test_draft_compiled_equals(case):
    arm, scen, B, gemm = CASES[case]
    keep = arm == "GUDKCD"
    name = X.DARMS[arm]
    d = _run("D", scen, B, gemm)
    b = _run(C_OF[arm], scen, B, gemm)                                 # the C arm first: it compiles the joint helpers
    x = _run(arm, scen, B, gemm)
    C = _cell()
    rd = x["rd"]
    c, gr, dr = rd["compile"], rd["graph"], rd["dgraph"]
    want_d = (dr["warm"] + dr["capture_passes"] + dr["eager"]) * L
    _ev("I4DC %s %s: rounds %d; compile calls %s, events %s, sections %s, draft sections %s (want %s), fallback %s / %s, "
        "captures begun %s, first draft capture %s, dynamo %s; joint graphs %s; draft warm %s captures %s replay passes "
        "%s eager %s" % (
            case, name, rd["rounds"], c["calls"], [(e["name"], e.get("kind"), e.get("dpass"), e.get("after_captures"),
                                                    e["capture"]) for e in c["events"]], c["sections"],
            c["dsections"], want_d, c["fallback"], c["dfallback"], c["captures_begun"], c["first_dcapture"],
            c["dynamo"], (gr["captures"], gr["warm"], gr["replays"], gr["eager"]), dr["warm"], dr["captures"],
            dr["replay_passes"], dr["eager"]))
    assert c["dsections"] == want_d > 0 and not c["dfallback"] and dr["replay_passes"] >= 10, c   # L34 preconditions
    assert all(c["calls"].get(n, 0) > 0 for n in CD.DHELPERS) and c["first_dcapture"] is not None
    assert all(not e["capture"] and e["after_captures"] == 0 for e in c["events"]), c["events"]
    skip = ("hoist", "keep", "round_blocks", "compile") if keep else ("hoist", "compile")
    ts = ("kid", "kage", "blocks") if keep else ()
    bad = TS._diff(d, x, skip=skip, tensors_skip=ts)
    assert bad == [], bad[:10]                                         # THE CLAIM: L4PTFD's run, bit for bit
    bad = TS._diff(b, x, skip=("compile",))
    assert bad == [], bad[:10]                                         # = the C arm in everything but the compile
    assert rd["hoist"] == b["rd"]["hoist"] and (rd.get("keep") == b["rd"].get("keep")) and rd["lean"] == b["rd"]["lean"]
    fd, fx = X.cell_inputs(d, C), X.cell_inputs(x, C)
    keepd = dict(L4PTFD=fd)
    if keep:
        keepd["L4PTFDGUDKWC"] = X.cell_inputs(b, C)
    rec = dict(graph=gr, dgraph=dr, gpu_peak_alloc_GB=1.0, mem_arena=dict(prefill_peak_GB=1.0))
    gs = C.arm_gates(name, rd, rec, fx, keepd, scen == "check", B, 10, N0, L, 1.0)
    names = [g[0] for g in gs]
    want = ["MEM", "GRAPH", "COUNTS", "DGRAPH", "HOIST"] + (["KEEP"] + (["CAP"] if scen == "check" else []) +
                                                            ["KEEP-EQ", "KEEP-REF"] if keep else []) + ["COMPILE", "EQF"]
    assert names == ["%s-%s" % (name, w) for w in want], names
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    if keep:
        kq = next(g for g in gs if g[0].endswith("-KEEP-EQ"))
        assert kq[1] is True and "L4PTFDGUDKWC" in kq[3], kq           # the keep reference: the C keep arm
    assert C.IA.lean_gate(name, dict(rd, K=X.K0), L)[2] is True


# ------------------------------------------------------------------ PLANTS
def test_draft_plants_caught():
    C = _cell()
    d = _run("D")
    _run("GUDC")                                                       # the joint helpers compiled before the plants
    fd = X.cell_inputs(d, C)
    p = X.run_arm("GUDCD", "timing", cdneg="lost_dwrite")              # the K rows into a copy of the buffer
    e = C.S4.eq_gate("L4PTFDGUDCD", X.cell_inputs(p, C), fd, N0)
    df = TS._diff(d, p, skip=("hoist", "compile"))
    _ev("I4DC PLANT lost_dwrite: differs %s; EQF %s" % (df[:4], e[2]))
    assert df != [] and e[2] is False                                  # not a no-op (L34); CAUGHT by EQF
    p = X.run_arm("GUDCD", "timing", cdneg="dlate")                    # a first compile inside a draft capture
    g = CD.compile_gate_cd("L4PTFDGUDCD", p["rd"], L, C.NAMES["L4PTFDGUDCD"])
    ev = p["rd"]["compile"]["events"]
    _ev("I4DC PLANT dlate: events %s; COMPILE %s" % ([(x["name"], x.get("kind"), x["capture"], x.get("after_captures"))
                                                      for x in ev], g[2]))
    assert any(x["capture"] and x.get("kind") == "draft" for x in ev) and g[2] is False

    def no_dtype(run):
        run.T._cd_ok = False
    p = X.run_arm("GUDCD", "timing", on_runner=no_dtype)               # the draft fallback (LN's eager row)
    g = CD.compile_gate_cd("L4PTFDGUDCD", p["rd"], L, C.NAMES["L4PTFDGUDCD"])
    df = TS._diff(d, p, skip=("hoist", "compile"))
    _ev("I4DC PLANT fallback (dtype): dfallback %s; COMPILE %s; = L4PTFD %s" % (p["rd"]["compile"]["dfallback"], g[2],
                                                                                df == []))
    assert p["rd"]["compile"]["dfallback"].get("dtype", 0) > 0 and g[2] is False and df == []


def test_draft_refusals():
    for kw in (dict(cdneg="nope"), dict(ccneg="nope"), dict(cc_backend="cudagraphs")):
        with pytest.raises(X.SR.SpecError):
            X.d_builder("GUDCD", **kw)(*X.world(2, 8)[:5])
    with pytest.raises(ValueError):
        CD.compiled_d("dpre", "nope")
    run = X.d_builder("GUDKCD")(*X.world(2, 8)[:5])
    assert run.meta["compile"]["arm"] == "L4PTFDGUDKWCD" and run.T.cdneg is None and run.T._cd_ok
    assert type(run.T).__name__ == "TentStateFDGUDKCD" and run.T.kwin and run.meta["compile"]["dhelpers"] == CD.DHELPERS
    run = X.d_builder("GUDCD")(*X.world(2, 8)[:5])
    assert type(run.T).__name__ == "TentStateFDGUDCD" and run.meta["compile"]["arm"] == "L4PTFDGUDCD"


# ------------------------------------------------------------------ CELL
def test_i4dc_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base)
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.check_every, a.probe_every) == (
        ["L4PTFD", "L4PTFDGUDC", "L4PTFDGUDCD"], 216, 512, 65, 3, 0, 16)
    for bad in (["--arms", "L4PTFD,L4PTFDGUDCD"], ["--arms", "L4PTFD,L4PTFDGUDKWCD"], ["--arms", "L4PTFD,L4PTFDGUDKWC,"
                "L4PTFDGUDCD"], ["--arms", "L4PTFDGUDC,L4PTFD"], ["--arms", "L4PTFD,L4PTFDGUDCD,L4PTFDGUDC"],
                ["--arms", "L4PTFD,L4PTFDGUDX"], ["--spec", "4"],
                ["--arms", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD", "--check-every", "1", "--batch", "216"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    for arms in ("L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD",
                 "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD", "L4PTFD,L4PTFDGUDKWC"):
        assert C.parse_args(base + ["--arms", arms]).arms == arms.split(",")
    assert C.runner_choice("L4PTFDGUDCD", False) == ("SCD", dict(cc_backend="inductor"))
    assert C.runner_choice("L4PTFDGUDKWCD", True) == ("SKCD", dict(keep=True, kwin=True, kcheck=True,
                                                                   cc_backend="inductor"))
    assert C.runner_choice("L4PTFDGUDC", False) == ("SC", dict(cc_backend="inductor"))
    assert C.runner_class("SCD", X.STF) is X.DCL["L4PTFDGUDCD"][1]
    assert C.runner_class("SKCD", X.STF) is X.DCL["L4PTFDGUDKWCD"][1]
    assert C.runner_class("SC", X.STF) is X.XC.CCL["L4PTFDGUDC"][1]
    assert C.main(["--out", "x", "--workload", "w", "--model", ""]) == 20     # no model: refused
    assert C.keep_ref_arm("L4PTFDGUDKWCD", {"L4PTFD": 1, "L4PTFDGUDKWC": 1}) == "L4PTFDGUDKWC"
    assert C.keep_ref_arm("L4PTFDGUDKWCD", {"L4PTFD": 1, "L4PTFKW": 1, "L4PTFDGUDKWC": 1}) == "L4PTFKW"
    assert C.keep_ref_arm("L4PTFDGUDKWC", {"L4PTFD": 1, "L4PTFDGUDKWC": 1}) is None
    assert C.keep_ref_arm("L4PTFDGUDKWC", {"L4PTFD": 1, "L4PTFDGUDKW": 1}) == "L4PTFDGUDKW"
    assert C.ROOT_ARM["L4PTFDGUDCD"] == "L4PTFDGUD" and C.ROOT_ARM["L4PTFDGUDKWCD"] == "L4PTFDGUDKW"
    assert C.NAMES["L4PTFDGUDKWCD"] == ("pre", "write", "loc1", "loc2", "kphys", "lists", "dpre", "dwrite", "dmask")


def test_i4dc_cell_pure_parts():
    C = _cell()
    d, b, x = _run("D"), _run("GUDKC"), _run("GUDKCD")
    fd, fb = X.cell_inputs(d, C), X.cell_inputs(b, C)
    rec = dict(graph=b["rd"]["graph"], dgraph=b["rd"]["dgraph"], gpu_peak_alloc_GB=1.0,
               mem_arena=dict(prefill_peak_GB=1.0))
    gs = C.arm_gates("L4PTFDGUDKWC", b["rd"], rec, fb, dict(L4PTFD=fd), False, 8, 10, N0, L, 1.0)
    kq = next(g for g in gs if g[0] == "L4PTFDGUDKWC-KEEP-EQ")
    assert kq[1] is False and "REPORT" in kq[3]                        # no keep reference in i4dcb: REPORT only
    assert all(g[2] for g in gs if g[1]), [g for g in gs if g[1] and not g[2]]
    assert [g[0] for g in gs][-2:] == ["L4PTFDGUDKWC-COMPILE", "L4PTFDGUDKWC-EQF"]   # the C arm: the I4CC gate
    rd = x["rd"]
    good = CD.compile_gate_cd("L4PTFDGUDKWCD", rd, L, C.NAMES["L4PTFDGUDKWCD"])
    assert good[2] is True, good
    ev0 = dict(name="dpre", graphs=1, s=1.0, layer=0, joint_pass=0, capture=False, kind="draft", dpass=1)
    for plant in (dict(events=rd["compile"]["events"] + [dict(ev0, after_captures=3)]),   # a compile after a capture
                  dict(events=rd["compile"]["events"] + [dict(ev0, after_captures=0, capture=True)]),   # inside one
                  dict(events=rd["compile"]["events"] + [dict(ev0, name="pre", after_captures=0, kind="joint")] * 2),
                  dict(events=rd["compile"]["events"] + [dict(ev0)]),   # no after_captures key: late
                  dict(fallback=dict(record=1)), dict(dfallback=dict(dtype=1)),
                  dict(sections=rd["compile"]["sections"] - 1), dict(dsections=rd["compile"]["dsections"] - 1),
                  dict(calls=dict(rd["compile"]["calls"], dmask=0)), dict(calls=dict(rd["compile"]["calls"], kphys=0)),
                  dict(dynamo=dict(graph_breaks=1)), dict(dtype_ok=False), dict(cd_ok=False),
                  dict(first_dcapture=None)):
        r2 = dict(rd, compile=dict(rd["compile"], **plant))
        assert CD.compile_gate_cd("L4PTFDGUDKWCD", r2, L, C.NAMES["L4PTFDGUDKWCD"])[2] is False, plant
    r2 = dict(rd, dgraph=dict(rd["dgraph"], warm=rd["dgraph"]["warm"] + 1))   # a pass that skipped the helpers
    assert CD.compile_gate_cd("L4PTFDGUDKWCD", r2, L, C.NAMES["L4PTFDGUDKWCD"])[2] is False
    sh2 = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    assert C.verdicts(sh2, {}, 216)[-1] == ("I4DC-RUN", "NOT_RUN", "no D arm ran")
    mk = lambda tok, dr_, jt, cs: dict(rates=dict(median_tok_s=tok), phases_median=dict(draft=dr_, joint=jt),  # noqa
                                       vs_strongest=0.5, gpu_peak_alloc_GB=82.0, mem_arena=dict(prefill_peak_GB=78.0),
                                       graph=dict(pool_bytes=50e6, capture_s=3.0), dgraph=dict(dcapture_s=4.0),
                                       cc=dict(compile_s=cs))
    sp = dict(L4PTFD=mk(400.0, 140.0, 200.0, None), L4PTFDGUDC=mk(430.0, 136.0, 181.0, 42.0),
              L4PTFDGUDCD=mk(440.0, 130.0, 181.0, 3.0))
    saved = {k: dict(v) for k, v in C.PRED_I4DC_BY_B.items()}
    try:
        for k in C.PRED_I4DC_BY_B:
            C.PRED_I4DC_BY_B[k].update({kk: None for kk in C.PRED_KEYS})
        v = {y[0]: y for y in C.verdicts(sh2, sp, 216, eqf=True)}
        C.PRED_I4DC_BY_B[216].update(I4DC_vs_ref=(1.03, 1.05), I4DC_draft_saving=(3, 8), I4DC_above_best=True)
        w = {y[0]: y for y in C.verdicts(sh2, sp, 216, eqf=True)}
    finally:
        for k in saved:
            C.PRED_I4DC_BY_B[k].clear()
            C.PRED_I4DC_BY_B[k].update(saved[k])
    assert all(y[1] == "REPORT" for y in v.values()), v
    assert "6.0000" in v["I4DC-DRAFT-SAVING"][2] and "3.0000" in v["I4DC-COMPILE-S"][2] and "1.0233" in v[
        "I4DC-VS-REF"][2] and "7.0000" in v["I4DC-CAPTURE-S"][2] and "1.0000" in v["I4DC-JOINT-RATIO"][2], v
    assert "X = L4PTFDGUDCD, REF = L4PTFDGUDC" in v["I4DC-ARM"][2] and "3.0 ms" in v["I4DC-DRAFT-STEP-SAVING"][2]
    assert w["I4DC-VS-REF"][1] == "REFUTED" and w["I4DC-DRAFT-SAVING"][1] == "CONFIRMED" and w["I4DC-ABOVE-BEST"][
        1] == "CONFIRMED"
    assert C.d_arm(dict(sp, L4PTFDGUDKWCD=1)) == "L4PTFDGUDKWCD" and C.d_arm(dict(L4PTFD=1)) is None


def test_i4dc_registration_state():
    C = _cell()
    if C.PRED_STAMP_I4DC is None:                                       # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I4DC_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I4DC)
    assert set(C.PRED_I4DC_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I4DC_BY_B.values())
    src = (EFF / "r2_i4dc_cell.py").read_text()
    assert ("\nPRED_STAMP_I4DC = None\n" in src) == (C.PRED_STAMP_I4DC is None)   # the step script's grep sees it


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    if shutil.which("srun") is None:     # the test runs on a compute node: the compute-side sha256sum wrapper (L55) is
        e["PATH"] = os.pathsep.join(p for p in e.get("PATH", "").split(os.pathsep)   # an srun, absent in the image
                                    if not p.rstrip("/").endswith("kstep-cpu-runs/bin-compute"))
    return subprocess.run(["bash", str(EFF / "r2_i4dc_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


CELLS = (("i4dm2", 10, 8, 72, 1, 1, 25, "100G", "L4PTFD,L4PTFKW,L4PTFDGUDC,L4PTFDGUDKWC,L4PTFDGUDCD,L4PTFDGUDKWCD"),
         ("i4dca", 65, 216, 512, 0, 0, 105, "240G", "L4PTFD,L4PTFDGUDC,L4PTFDGUDCD"),
         ("i4dcb", 10, 318, 512, 0, 0, 160, "340G", "L4PTFD,L4PTFDGUDKWC,L4PTFDGUDKWCD"))


def test_step_script_i4dc():
    src = (EFF / "r2_i4dc_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    assert "CCROOT=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i4dc_jit " in src          # the caches: never in HOME
    for env in ('TORCHINDUCTOR_CACHE_DIR="$CCJIT/inductor"', 'TRITON_CACHE_DIR="$CCJIT/triton"',
                'TRITON_HOME="$CCJIT/triton_home"'):
        assert env in src, env
    for f in ("r2_i4dc_compile.py", "r2_i4cc_compile.py", "r2_i4cc_cell.py", "r2_i4s_cell.py", "r2_i4s_stack.py",
              "r2_i4h_graph.py", "r2_i4h_cell.py", "r2_i4k_combo.py", "r2_i4d_keep.py", "r2_i4u_hoist.py",
              "r2_i4g_graph.py", "r2_i4a_lean.py"):
        assert "benchmarks/Efficiency/%s" % f in src, f
    assert "OUTROOT/sp_active_$JOB" in src
    for bad in ("i4cm2", "i4cca", "i4dc", "i4dm"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I4DC = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i4dc_cell.py" % sha).stdout
    ran = 0
    for cell, pool, batch, steps, check, stress, smin, mem, arms in CELLS:
        r = _step(cell, sha)
        if cell in ("i4dca", "i4dcb") and not registered:              # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I4DC is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        assert r.returncode == 0, (cell, r.stderr[-400:])
        ran += 1
        line = ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 fork-store 0 check %d stress %d STEP_MIN %d mem %s "
                "claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s compile caches "
                "/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i4dc_jit/i4dc_t1_%s_%s" % (
                    cell, 63 + pool, pool, batch, arms, steps, check, stress, smin, mem, HOLDER, HOLDER, cell))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--check-every %d" % check, "--require-stress %d" % stress,
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i4dc_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "TORCHINDUCTOR_CACHE_DIR=/mnt/beegfs/ojerbi/kstep-cpu-runs/tmp_sc/i4dc_jit/i4dc_t1_%s_%s/inductor"
                     % (HOLDER, cell), "i4dc_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        assert re.search(r"--arms %s " % arms.replace(",", r"\\?,"), r.stdout), (cell, r.stdout[-600:])
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", arms, "--check-every", str(check),
                                "--require-stress", str(stress)])
        assert a.arms == arms.split(",")                               # every cell's arguments parse
    assert ran >= 1
    _ev("I4DC DRYRUN: %d cell(s) printed their step command (registered %s)" % (ran, registered))
