"""CPU tests of WS-B (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs; departure D2): the
joint pass with ONE residency per layer for the UNION of the rows' selections (nosi/nosi/verify_rows.py mode "union";
nosi/nosi/spec_round.py SpecOps(mode="union") for L4 and L4P; benchmarks/Efficiency/r2_wsb_cell.py; r2_wsb_step.sh). The
world: tests_cpu/wsb_world.py over spec_world / verify_world / ragged_world (the SHIPPED engine from its own source;
every CUDA op a torch stand-in with the kernel's flat addresses, lesson L28). B 1 and B 8. No CUDA, no Triton launch.
CPU stand-ins do not establish GPU dispatch or ordering (the check cells wb1ca / wb1cb profile the kernels).

THE BINDING CORRECTION (coordinator, 2026-10-28): FAST is checked on FAST itself against SH (ordinary NOSA, the same
configuration), not against the EXACT serial pass. SH here = the SERIAL ragged step on a second world of the same
build (at equal lengths with natural ops it is bitwise the shipped step: test_ragged_engine.py).

GATE 0   test_gate0_files: the tree = 40ac348 except the declared WS-B files. test_gate0_verify_rows_unchanged (B 1, B 8):
         modes exact (gemm exact / rowgroup) and fast give the bits of the 40ac348 SOURCE (loaded from git), logits and
         every engine and twin table. test_gate0_spec_round_unchanged: L4 exact, L4 fast and L4P exact runs of this
         tree's round runner = the 40ac348 source's, bit for bit (tokens, logits rows, counters, records, state).
VS SH    test_union_vs_sh (B 1, B 8; row-invariant ops / natural ops at equal lengths): per pass, layer and row every
         row's selection SET = SH's at its position (MUST, no change), dropped 0, unserved 0, visible count = selection
         size (MUST), row 0 of the first pass BITWISE SH's logits, every row's argmax = SH's and |delta| <= 1e-4
         relative, ONE residency call per layer and pass (the module's host count and an independent count of the diff
         kernel), VRF-IDX on the window call and the first store call.
MATH     test_union_attention_is_sh_math (B 8): every full row's attention output = SH's attention math in float64,
         order-free (its own selection; the tail rows up to its position, F4 kept; past a block edge the block m and
         the next block's rows), within 2e-5. Preconditions: a row past a block edge, an F4 fill row, a store entry.
BYTES    test_every_selected_block_present (B 1, B 8, X 4: the union does not fit one store fill): every selected block
         of every full row is in the window or in a store slot of some fill with the host rows and cis rows of its
         block (or it is the tail / next block); the selections and logits still match SH. Precondition: spill > 0.
PLANTS   test_dropped_extras_plant_caught: "drop_spill" (one fill, no spill: the old 5.57 % FAST overflow) is caught by
         the presence recount (dropped > 0 = the byte check's missing count) and by the cell's PRESENT gate.
         test_union_attention_plant_caught: "union_attention" (every row sees the union) is caught by the visible
         count and by the cell's VIS gate, and the logits leave SH's. test_short_store_length_plant_caught:
         "short_store_len" (the b67cc05 defect: a store call's length one slot short) is caught by the visible count,
         which counts a store slot only where the length reads it.
COUNT    test_one_residency_call_per_layer: union = L diff calls per pass; the EXACT pass (negative control) = R L.
ROLES    test_resident_rows_single_pass: rows 0 .. K - 1 full, row K resident (L4XU's roles): the full rows match SH;
         the resident row's skipped blocks = the byte check's count; the role counters.
STAND-IN test_fake_attention_u: the per-entry bits of verify_world.fake_attention_idx; the kernel's early exit.
OPS      test_union_ops_and_bound: the refusals; the store fills C = ceil((R - 1) topk / (X - 1)) (cap >= the bound).
SYNC     test_no_host_sync_in_union_pass: verify_rows.py reads no device value during a union pass (check on and off).
RUNNER   test_l4xu_runner (B 1, B 8) and test_l4pu_runner (B 8): spec_round with mode "union": residency and host-row
         invariants after every round, one residency per layer and joint pass, both accept paths ran, the rung-L5 seam
         gets the draft row's misses (L4P); token agreement with the shipped decode >= 0.9 (REPORT of FAST; WS-G gives
         the accuracy gate).
CELL     test_cell_arguments, test_cell_pure_parts, test_step_script_wb1.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsb.py
"""
from __future__ import annotations

import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
BASE = "40ac348870e89b43ac5278458b660dea8d360508"            # nosi-r2-spec5: the tree WS-B starts from
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-wsb"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
DECLARED = ("nosi/nosi/verify_rows.py", "nosi/nosi/spec_round.py", "benchmarks/Efficiency/r2_wsb_cell.py",
            "benchmarks/Efficiency/r2_wsb_step.sh", "benchmarks/Efficiency/tests_cpu/wsb_world.py",
            "benchmarks/Efficiency/tests_cpu/test_wsb.py", "benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py",
            "docs/NOTES_wsb.md",
            # WS-C (nosi-r2-wsc, on top of 1a8ce68): its files; tests_cpu/test_wsc.py pins them (test_gate0_files)
            "benchmarks/Efficiency/fork_vllm_unified_attention.py", "benchmarks/Efficiency/r2_kv_arena.py",
            "benchmarks/Efficiency/r2_wsc_cell.py", "benchmarks/Efficiency/r2_wsc_step.sh",
            "benchmarks/Efficiency/tests_cpu/wsc_world.py", "benchmarks/Efficiency/tests_cpu/test_wsc.py",
            "benchmarks/Efficiency/tests_cpu/test_wsc_interp.py", "docs/NOTES_wsc.md")
# I2 (nosi-r2-i2: r2-wsareg 83dac96 + r2-wsb 1a8ce68 + the arm L4PTU): the WS-A files, WS-A's added Gate-0 lines in the
# older tests and the I2 files are declared too (their own Gate 0s: tests_cpu/test_wsa.py, tests_cpu/test_i2.py); in
# test_spec_prefetch.py the lines of WS-A's own diff (40ac348 -> 83dac96) are allowed besides WS-B's SPEC5_TIP lines
WSA_TIP = "83dac96becf599181913e207dae7174c7f12c147"
DECLARED = DECLARED + ("nosi/nosi/spec_tentative.py", "benchmarks/Efficiency/r2_wsa_cell.py",
                       "benchmarks/Efficiency/r2_wsa_step.sh", "benchmarks/Efficiency/tests_cpu/wsa_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_wsa.py", "docs/NOTES_wsa.md",               # WS-A
                       "benchmarks/Efficiency/tests_cpu/test_ragged_engine.py",
                       "benchmarks/Efficiency/tests_cpu/test_verify_rows.py",
                       "benchmarks/Efficiency/tests_cpu/test_rollback.py",
                       "benchmarks/Efficiency/tests_cpu/test_spec_round.py",                            # WS-A, I2 lines
                       "nosi/nosi/spec_tentative_union.py", "benchmarks/Efficiency/r2_i2_cell.py",
                       "benchmarks/Efficiency/r2_i2_step.sh", "benchmarks/Efficiency/tests_cpu/i2_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i2.py", "docs/NOTES_i2.md",                # I2
                       "benchmarks/Efficiency/r2_i2_check.py", "benchmarks/Efficiency/r2_wsg_check.py")  # I2
# I3 (nosi-r2-i3: I2 63da01d + WS-C2 db21536 + the arm L4PTF): its files (its own Gate 0: tests_cpu/test_i3.py)
DECLARED = DECLARED + ("nosi/nosi/spec_tentative_fork.py", "benchmarks/Efficiency/r2_i3_arena.py",
                       "benchmarks/Efficiency/r2_i3_cell.py", "benchmarks/Efficiency/r2_i3_step.sh",
                       "benchmarks/Efficiency/tests_cpu/i3_world.py", "benchmarks/Efficiency/tests_cpu/test_i3.py",
                       "benchmarks/Efficiency/tests_cpu/test_i3_interp.py", "docs/NOTES_i3.md")
# I4A (nosi-r2-i4a: I3 c4c64b2 + the arm L4PTFD): its files (its own Gate 0: tests_cpu/test_i4a.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4a_lean.py", "benchmarks/Efficiency/r2_i4a_cell.py",
                       "benchmarks/Efficiency/r2_i4a_step.sh", "benchmarks/Efficiency/tests_cpu/i4a_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4a.py", "docs/NOTES_i4a.md")                # I4A
# I4D (nosi-r2-i4d: I4A acbe433 + the accounting and the arm L4PTFK): its files (its own Gate 0: tests_cpu/test_i4d.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4d_keep.py", "benchmarks/Efficiency/r2_i4d_cell.py",
                       "benchmarks/Efficiency/r2_i4d_step.sh", "benchmarks/Efficiency/tests_cpu/test_i4d.py")  # I4D
# I4G (nosi-r2-i4g: I4A acbe433 + the arm L4PTFDG): its files (its own Gate 0: tests_cpu/test_i4g.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4g_graph.py", "benchmarks/Efficiency/r2_i4g_cell.py",
                       "benchmarks/Efficiency/r2_i4g_step.sh", "benchmarks/Efficiency/tests_cpu/i4g_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4g.py", "docs/NOTES_i4g.md")                # I4G
# I4K (nosi-r2-i4k: I4D 99c231e + I4G d7997e6 + the arm L4PTFDGKW): its files (its own Gate 0: tests_cpu/test_i4k.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4k_combo.py", "benchmarks/Efficiency/r2_i4k_cell.py",
                       "benchmarks/Efficiency/r2_i4k_step.sh", "benchmarks/Efficiency/tests_cpu/i4k_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4k.py")                                     # I4K
# I4U (nosi-r2-i4u: I4A acbe433 + the arm L4PTFDU): its files (its own Gate 0: tests_cpu/test_i4u.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4u_hoist.py", "benchmarks/Efficiency/r2_i4u_cell.py",
                       "benchmarks/Efficiency/r2_i4u_step.sh", "benchmarks/Efficiency/tests_cpu/i4u_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4u.py")                                     # I4U
# I4H (nosi-r2-i4h: I4G d7997e6 + the arm L4PTFDGD): its files (its own Gate 0: tests_cpu/test_i4h.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4h_graph.py", "benchmarks/Efficiency/r2_i4h_cell.py",
                       "benchmarks/Efficiency/r2_i4h_step.sh", "benchmarks/Efficiency/tests_cpu/i4h_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4h.py", "docs/NOTES_i4h.md")                # I4H
# I4S (nosi-r2-i4s: I4K 70059e5 + I4U 832dca5 + I4H fc6c680 + the stacked arms): its files (its own Gate 0:
# tests_cpu/test_i4s.py)
DECLARED = DECLARED + ("benchmarks/Efficiency/r2_i4s_stack.py", "benchmarks/Efficiency/r2_i4s_cell.py",
                       "benchmarks/Efficiency/r2_i4s_step.sh", "benchmarks/Efficiency/tests_cpu/i4s_world.py",
                       "benchmarks/Efficiency/tests_cpu/test_i4s.py")                                     # I4S


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


X_ = _load("wsb_world")
SW, V, W, RE, VR, SR, RW = X_.SW, X_.V, X_.W, X_.RE, X_.VR, X_.SR, X_.RW
K, R, L = 3, 4, X_.L
S1 = [4923]                                                      # 77 blocks
SF8 = [4800, 4857, 4923, 4860, 4795, 4811, 4830, 4891]          # 75-77 blocks; block edges at different rows
EQ8 = [4158] * 8                                                # equal lengths (natural ops: the shipped bits)
CASES = dict(B1=(S1, False, "exact"), B8=(SF8, False, "exact"), B1nat=([4158], True, "rowgroup"),
             B8nat=(EQ8, True, "rowgroup"))
FILTER = SW.corrupt(4, 1, kinds=("draft",), phase=2)
TIMING = ("wall_s", "phases", "round_ms", "probe_wait_ms", "prelude_ms")


@functools.lru_cache(maxsize=None)
def _ev(case, groups=3, extra=16, neg=None, draft_last=False):
    S_list, natural, gemm = CASES[case]
    return X_.run_union_vs_sh(S_list, groups, extra=extra, neg=neg, natural=natural, gemm=gemm, draft_last=draft_last)


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_wsb_cell
    return r2_wsb_cell


def _res_of(ev):
    """A check arm's result (r2_wsb_cell.check_gates' input) from a CPU run."""
    return dict(stats=ev["stats"], errors=ev["errors"][1], fa_check=ev["vs"].fa_check,
                sel=dict(compared=ev["compared"], changed=len(ev["changed"]), by_layer_nonzero={}, symdiff_hist={},
                         first=ev["changed"][:5]))


# ------------------------------------------------------------------ GATE 0
def _have(commit) -> bool:
    return X_.git("rev-parse", "-q", "--verify", commit + "^{commit}").returncode == 0


def test_gate0_files():
    if not _have(BASE):
        pytest.skip("no git or no commit %s in %s" % (BASE, ROOT))
    r = X_.git("diff", "--name-only", BASE, "--", ".", ":(exclude,glob)**/__pycache__/**")
    assert r.returncode == 0, r.stderr
    changed = [x for x in r.stdout.split("\n") if x]
    assert set(changed) <= set(DECLARED), sorted(set(changed) - set(DECLARED))
    r = X_.git("ls-files", "--others", "--exclude-standard", "--", "nosi", ":(exclude,glob)**/__pycache__/**")
    assert r.stdout.strip() == "", "untracked files in nosi/: %s" % r.stdout
    # test_spec_prefetch.py: its Gate 0 reads the rung-L5 tip (SPEC5_TIP) in place of the working tree; nothing else
    d = X_.git("diff", "-U0", BASE, "--", "benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py").stdout.splitlines()
    wsa = set(X_.git("diff", "-U0", BASE, WSA_TIP, "--", "benchmarks/Efficiency/tests_cpu/test_spec_prefetch.py")
              .stdout.splitlines()) if _have(WSA_TIP) else set()
    d = [x for x in d if x not in wsa]                        # I2: WS-A's own lines of this file (its Gate 0)
    added = [x[1:] for x in d if x.startswith("+") and not x.startswith("+++")]
    removed = [x for x in d if x.startswith("-") and not x.startswith("---")]
    assert added and all("SPEC5_TIP" in x or x.strip().startswith("#") for x in added), added
    assert len(removed) <= 8, removed
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}
    for n in ("spec_round", "ragged_rollback", "state_snapshot", "spec_prefetch"):   # the older Gate 0 name scans
        assert n not in src["verify_rows.py"], n
    assert sorted(n for n, t in src.items() if "union_list" in t) == ["spec_tentative_union.py",   # I2: the L4PTU
                                                                       "verify_rows.py"]          # pass calls it


@pytest.mark.parametrize("S_list", [S1, SF8], ids=["B1", "B8"])
def test_gate0_verify_rows_unchanged(S_list, tmp_path):
    if not _have(BASE):
        pytest.skip("no git or no commit %s in %s" % (BASE, ROOT))
    OLD = X_.old_module("verify_rows", BASE, tmp_path, patch_verify=True)
    for mode, gemm in (("exact", "exact"), ("exact", "rowgroup"), ("fast", "natural")):
        new, s_new = X_.joint_run(VR, S_list, 2, mode, gemm)
        old, s_old = X_.joint_run(OLD, S_list, 2, mode, gemm)
        for g, (a, b) in enumerate(zip(new, old)):
            assert torch.equal(W.bits(a), W.bits(b)), (mode, gemm, g, V.first_diff(a, b))
        bad = V.engines_equal(s_new.cache, s_old.cache, (s_new.st.twins, s_old.st.twins))
        assert bad == [], (mode, gemm, bad[:5])


def _same_records(a, b) -> list:
    bad = []
    if len(a) != len(b):
        return ["record count %d != %d" % (len(a), len(b))]
    for g, (x, y) in enumerate(zip(a, b)):
        for k in sorted(set(x) | set(y)):
            u, v = x.get(k), y.get(k)
            if torch.is_tensor(u) or torch.is_tensor(v):
                if not (torch.is_tensor(u) and torch.is_tensor(v) and torch.equal(W.bits(u), W.bits(v))):
                    bad.append((g, k))
            elif u != v:
                bad.append((g, k))
    return bad


@pytest.mark.parametrize("level,mode,gemm", [("L4", "exact", "rowgroup"), ("L4", "fast", "natural"),
                                             ("L4P", "exact", "rowgroup")], ids=["L4X", "L4F", "L4P"])
def test_gate0_spec_round_unchanged(level, mode, gemm, tmp_path):
    if not _have(BASE):
        pytest.skip("no git or no commit %s in %s" % (BASE, ROOT))
    OLD = X_.old_module("spec_round", BASE, tmp_path)
    new, old = [SW.run_vs_shipped(8, level=level, mode=mode, gemm=gemm, draft_filter=FILTER, sr=m) for m in (None, OLD)]
    rn, ro = new["runner"], old["runner"]
    for x, y in ((rn.out, ro.out), (rn.fmax, ro.fmax), (rn.fsum, ro.fsum), (rn.lg_out, ro.lg_out)):
        assert torch.equal(W.bits(x), W.bits(y)), (level, mode)
    dn = {k: v for k, v in new["rd"].items() if k not in TIMING}
    do = {k: v for k, v in old["rd"].items() if k not in TIMING}
    assert dn == do, (level, mode, [k for k in dn if dn[k] != do.get(k)])
    assert _same_records(new["records"], old["records"]) == [], (level, mode)
    assert RW.diff(RW.state(rn.rs), RW.state(ro.rs)) == [], (level, mode)
    assert new["facts"]["rejects"] > 0 and new["facts"]["full"] > 0, new["facts"]          # both paths ran (L34)


# ------------------------------------------------------------------ VS SH
@pytest.mark.parametrize("case", ["B1", "B8", "B1nat", "B8nat"])
def test_union_vs_sh(case):
    ev = _ev(case)
    st = ev["stats"]
    S_list = CASES[case][0]
    assert ev["compared"] == 3 * L * W.HKV * len(S_list) * R                 # every (head, request, row, layer) set
    assert ev["fills"], "precondition (L34): a block edge inside the 3 groups"
    if case == "B8":
        assert st["extras"] > 0 and st["extras_pool"] + st["extras_host"] > 0, st     # rows read store blocks
    assert ev["changed"] == [], ev["changed"][:8]                            # THE CLAIM: SH's selection, every row
    assert st["dropped"] == 0 and st["unserved"] == 0, st                    # every selected block present
    assert st["vis_bad"] == 0 and st["row_visible"] == st["row_selected"], st   # each row sees its selection only
    assert ev["row0_bits"] is True                                           # row 0, pass 0: SH's bits
    assert ev["argmax_bad"] == [] and ev["max_delta"] <= 1e-4, (ev["argmax_bad"][:5], ev["max_delta"])
    assert ev["diff_per_pass"] == [L] * 3 and st["calls"]["resid"] == L * 3, (ev["diff_per_pass"], st["calls"])
    assert st["calls"]["attn"] == L * 3 * (1 + ev["vs"].C), st["calls"]     # call A + one call per store fill
    assert ev["errors"] == ((0, [0] * L), (0, [0] * L)), ev["errors"]
    fa = ev["vs"].fa_check
    assert fa["A"]["entries"] and all(fa["A"]["equal"]), fa
    if st["extras"] > 0:
        assert fa["B"]["entries"] and all(fa["B"]["equal"]), fa


def test_union_attention_is_sh_math():
    ev = _ev("B8")
    recs = [r for r in ev["vs"].log if r["kind"] == "union"]
    past = any(bool((~r["in_tail"]).any()) for r in recs)
    f4 = any(bool(((r["p_rows"] % W.BS == W.BS - 1) & r["in_tail"]).any()) for r in recs)
    store = any(bool((f["lens"] > 0).any()) for r in recs for f in r["fills"])
    assert past and f4 and store, (past, f4, store)                          # preconditions (L34)
    worst, n = 0.0, 0
    for rec in recs:
        ref, done = X_.sh_attention_ref(rec, R)
        got = rec["out"].double()
        d = ((got - ref).abs() / (1 + ref.abs()))[done]
        n += int(done.sum())
        worst = max(worst, float(d.max()))
    assert n > 0 and worst <= 2e-5, (n, worst)


# ------------------------------------------------------------------ BYTES (forced overflow: X 4)
@pytest.mark.parametrize("case", ["B1", "B8"])
def test_every_selected_block_present(case):
    ev = _ev(case, groups=4, extra=4)
    st = ev["stats"]
    assert st["spill"] > 0, st                                               # precondition: the union did not fit
    u = X_.union_bytes(ev["vs"].log, 4)
    assert u["checked"] > 0 and u["fills"] == L * 4 * ev["vs"].C, u["fills"]
    assert u["missing_full"] == [] and u["contents_bad"] == [], (u["missing_full"][:5], u["contents_bad"][:5])
    assert st["dropped"] == 0 and st["unserved"] == 0 and st["vis_bad"] == 0, st
    assert ev["changed"] == [] and ev["argmax_bad"] == [] and ev["max_delta"] <= 1e-4, (ev["changed"][:5],
                                                                                         ev["max_delta"])


# ------------------------------------------------------------------ PLANTS
def test_dropped_extras_plant_caught():
    ev = _ev("B8", groups=4, extra=4, neg="drop_spill")
    st = ev["stats"]
    assert ev["vs"].C == 1 and st["unserved"] > 0, st                        # precondition: blocks past one fill
    u = X_.union_bytes(ev["vs"].log, 4)
    assert st["dropped"] > 0 and len(u["missing_full"]) == st["dropped"], (st["dropped"], len(u["missing_full"]))
    g = {n: ok for n, _, ok, _ in _cell().check_gates("UC", _res_of(ev), L=L)}
    assert g["UC-PRESENT"] is False, g                                       # the cell's MUST gate catches it
    clean = _ev("B8", groups=4, extra=4)
    assert clean["stats"]["dropped"] == 0 and {n: ok for n, _, ok, _ in _cell().check_gates(
        "UC", _res_of(clean), L=L)}["UC-PRESENT"] is True


def test_union_attention_plant_caught():
    ev = _ev("B8", neg="union_attention")
    st = ev["stats"]
    assert _ev("B8")["stats"]["extras"] > 0                                  # precondition: unions larger than rows
    assert st["vis_bad"] > 0, st
    assert ev["max_delta"] > 1e-4 or ev["argmax_bad"], ev["max_delta"]       # the logits leave SH's
    g = {n: ok for n, _, ok, _ in _cell().check_gates("UC", _res_of(ev), L=L)}
    assert g["UC-VIS"] is False, g


def test_short_store_length_plant_caught():
    """The defect of b67cc05 (review): a store call's length ended at the START of the highest visible slot, so that
    slot was never read; the mask-level counters (dropped, the old visible count) and VRF-IDX could not see it. The
    visible count now counts a store slot only where the length reads it."""
    ev = _ev("B8", neg="short_store_len")
    st = ev["stats"]
    assert _ev("B8")["stats"]["extras"] > 0                                  # precondition: rows read store slots
    assert st["dropped"] == 0, st                                            # the list holds every block ...
    assert st["vis_bad"] > 0, st                                             # ... but a row does not read one
    assert ev["max_delta"] > 1e-4 or ev["argmax_bad"] or ev["changed"], ev["max_delta"]
    g = {n: ok for n, _, ok, _ in _cell().check_gates("UC", _res_of(ev), L=L)}
    assert g["UC-VIS"] is False and g["UC-PRESENT"] is True, g


# ------------------------------------------------------------------ COUNT
def _built(S_list, steps):
    model = W.make_model(0)
    rows = W.prompt_rows(1, S_list)
    s = RE.build(model, W.new_cache(0), S_list, steps, V.exact_ragged_ops(), lambda b, l: rows[(b, l)], pin=False)
    W.poison_built(s, S_list)
    calls = []
    kd = s.ops.kernels["diff"]

    def counted(*a, **kw):
        calls.append(1)
        return kd(*a, **kw)
    s.ops.kernels["diff"] = counted
    return model, s, calls


def test_one_residency_call_per_layer():
    S_list = SF8[:4]
    B = len(S_list)
    pos = (torch.tensor(S_list).view(B, 1) + torch.arange(R).view(1, R)).to(torch.int32)
    ids = torch.zeros((B, R), dtype=torch.int64)
    counts = {}
    for mode in ("union", "exact"):
        model, s, calls = _built(S_list, R)
        vs = VR.attach(model, s, VR.VerifyOps(K=K, mode=mode, gemm="exact", linear=W.row_linear,
                                              attn=X_.fake_attention_u))
        VR.joint_pass(model, vs, ids, pos)
        counts[mode] = (len(calls), vs.calls["resid"])
    assert counts["union"] == (L, L), counts                                 # THE CLAIM: one per layer and pass
    assert R > 1 and counts["exact"] == (R * L, R * L), counts               # NEGATIVE CONTROL: the serial rows


# ------------------------------------------------------------------ ROLES
def test_resident_rows_single_pass():
    ev = _ev("B8", groups=1, draft_last=True)
    st = ev["stats"]
    B = len(SF8)
    assert st["rows_full"] == L * B * K and st["rows_res"] == L * B, st     # the roles of L4XU
    assert ev["changed"] == [] and ev["argmax_bad"] == [] and ev["max_delta"] <= 1e-4, ev["changed"][:5]
    assert st["dropped"] == 0 and st["vis_bad"] == 0, st
    u = X_.union_bytes(ev["vs"].log, 16)
    assert u["missing_full"] == [] and u["contents_bad"] == [] and u["missing_res"] == st["skipped"], (
        u["missing_res"], st["skipped"])
    assert st["row_skipped"][:K] == [0] * K, st["row_skipped"]


# ------------------------------------------------------------------ STAND-IN
def test_fake_attention_u():
    g = torch.Generator().manual_seed(5)
    Bc, rows = 3, 16 * W.BS
    E = Bc * R
    q = torch.randn(E, 1, W.HQ, W.D, generator=g)
    k = torch.randn(Bc, rows, W.HKV, W.D, generator=g)
    v = torch.randn(Bc, rows, W.HKV, W.D, generator=g)
    bias = torch.randn(E, rows, W.HKV, generator=g)
    lens = torch.randint(1, rows + 1, (E,), generator=g).to(torch.int32)
    cbi = (torch.arange(E) // R).to(torch.int32)
    kw = dict(cache_seqlens=lens, cache_batch_idx=cbi, num_splits=W.SPLITS, return_softmax_lse=True)
    o1, l1 = X_.fake_attention_u(q, k, v, bias[:Bc], **kw)
    o2, l2 = V.fake_attention_idx(q, k, v, bias[:Bc], **kw)
    assert torch.equal(W.bits(o1), W.bits(o2)) and torch.equal(W.bits(l1), W.bits(l2))
    z = lens.clone()
    z[::3] = 0
    o3, l3 = X_.fake_attention_u(q, k, v, bias[:Bc], **dict(kw, cache_seqlens=z))
    zero = z == 0
    assert bool((o3[zero] == 0).all()) and bool(torch.isinf(l3[zero]).all()) and bool((l3[zero] > 0).all())
    assert torch.equal(W.bits(o3[~zero]), W.bits(o1[~zero]))
    with pytest.raises(Exception):                                           # the old stand-in has no early exit
        V.fake_attention_idx(q, k, v, bias[:Bc], **dict(kw, cache_seqlens=z))


# ------------------------------------------------------------------ OPS
def test_union_ops_and_bound():
    for kw in (dict(mode="fast", spill=2), dict(mode="fast", check=True), dict(mode="exact", neg="drop_spill"),
               dict(mode="fast", neg="union_attention"), dict(mode="fast", neg="short_store_len"),
               dict(mode="union", neg="no_extras"), dict(mode="union", neg="no_row_mask"),
               dict(mode="union", neg="causal_leak"), dict(mode="union", spill=0), dict(mode="union", extra=1)):
        with pytest.raises(VR.VerifyError):
            VR.VerifyOps(**kw)
    with pytest.raises(SR.SpecError):
        SR.SpecOps(level="L3", mode="union")
    SR.SpecOps(level="L4P", K=3, mode="union")                               # accepted (WS-B)
    model, s, _ = _built([4100, 4157], 8)
    fa = X_.fake_attention_u
    for X in (2, 4, 16, 48):
        vs = VR.attach(model, s, VR.VerifyOps(K=K, mode="union", extra=X, attn=fa))
        bound = (R - 1) * W.TOPK
        assert vs.C == -(-bound // (X - 1)) and vs.cap >= bound and (vs.C - 1) * (X - 1) < bound, (X, vs.C)
        assert tuple(vs.ext_k.shape[:2]) == (2, X * W.BS)
    assert VR.attach(model, s, VR.VerifyOps(K=K, mode="union", extra=4, neg="drop_spill", attn=fa)).C == 1
    assert VR.attach(model, s, VR.VerifyOps(K=K, mode="union", extra=4, spill=2, attn=fa)).cap == 6


# ------------------------------------------------------------------ SYNC
def test_no_host_sync_in_union_pass():
    S_list = SF8[:4]
    B = len(S_list)
    for check in (False, True):
        model, s, _ = _built(S_list, 2 * R)
        vs = VR.attach(model, s, VR.VerifyOps(K=K, mode="union", gemm="exact", linear=W.row_linear, check=check,
                                              attn=X_.fake_attention_u))
        with RW.SyncTrap(VR.__file__) as trap:
            for g in range(2):
                pos = (torch.tensor(S_list).view(B, 1) + g * R + torch.arange(R).view(1, R)).to(torch.int32)
                VR.joint_pass(model, vs, torch.zeros((B, R), dtype=torch.int64), pos)
        assert vs.passes == 2
        # the only host read: joint_pass's activity check on the HOST mirror st.active_host (a CPU tensor, 40ac348)
        assert {(h[0], h[2]) for h in trap.hits} <= {("joint_pass", "__bool__")}, (check, trap.hits[:5])
    with RW.SyncTrap(VR.__file__) as trap:                                   # the trap is not blind: read_stats syncs
        VR.read_stats(vs)
    assert trap.hits


# ------------------------------------------------------------------ RUNNER
def _agreement(ev):
    cnt = torch.tensor(ev["rd"]["count"]).clamp(max=ev["N"])
    bad = {b: i for b, i, _, _ in ev["bad_tokens"]}
    good = sum(bad.get(b, int(cnt[b])) for b in range(ev["B"]))
    return good / max(1, int(cnt.sum()))


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_l4xu_runner(B):
    ev = X_.run_spec_union(B, level="L4", extra=16, draft_filter=FILTER)
    rd, f = ev["rd"], ev["facts"]
    v = rd["verify"]
    assert f["rejects"] > 0 and f["full"] > 0, f                            # both paths ran (L34)
    assert f["edge_crossed"], f
    assert ev["resid"] == [] and ev["host"] == [], (ev["resid"][:2], ev["host"][:2])   # residency, host rows
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert v["passes"] == rd["rounds"] and v["calls"]["resid"] == L * v["passes"], (v["passes"], v["calls"])
    assert v["unserved"] == 0 and v["union_samples"] > 0, v
    if B > 1:
        assert v["extras"] > 0, v
    a = _agreement(ev)
    print("L4XU B %d: committed tokens equal to the shipped decode: %.4f; first differences %s" % (
        B, a, ev["bad_tokens"][:4]))
    assert a >= 0.9, (a, ev["bad_tokens"][:5])


def test_l4pu_runner():
    box = {}

    def mk(rs):
        box["rec"] = SW.Recorder()
        return box["rec"]
    ev = X_.run_spec_union(8, level="L4P", extra=16, draft_filter=FILTER, make_prefetch=mk)
    rd, f = ev["rd"], ev["facts"]
    v = rd["verify"]
    assert f["rejects"] > 0 and f["full"] > 0, f
    assert ev["resid"] == [] and ev["host"] == [] and ev["host_c"] == [], (ev["resid"][:2], ev["host"][:2])
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert v["passes"] == rd["rounds"] and v["calls"]["resid"] == L * v["passes"], (v["passes"], v["calls"])
    assert v["rows_res"] > 0 and v["rows_full"] > 0, v                      # the roles were set per pass
    rec = box["rec"]
    joint = rec.drafts[K::K]                                                 # the prelude's K drafts, then per round:
    assert joint and any(bool((m >= 0).any()) for x in joint for m in x)    # the joint pass, K - 1 drafts
    a = _agreement(ev)
    print("L4P union B 8: committed tokens equal to the shipped decode: %.4f" % a)
    assert a >= 0.9, (a, ev["bad_tokens"][:5])


# ------------------------------------------------------------------ CELL
def test_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base + ["--joint-residency", "union"])
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.joint_extra, a.joint_spill, a.check_steps, a.spec_twin,
            a.spec_body, a.probe_every) == (["L4X", "L4XU"], 216, 512, 65, 3, 48, 0, 128, "fused", "fused", 16)
    assert C.parse_args(base + ["--arms", "L4X"]).joint_residency == "serial"
    for bad in ([], ["--joint-residency", "union", "--arms", "L4X"], ["--joint-residency", "union", "--arms", "UC,SHR"],
                ["--joint-residency", "union", "--arms", "UC"],
                ["--joint-residency", "union", "--arms", "SHR,UC", "--check-steps", "130"],
                ["--joint-residency", "union", "--arms", "SHR,UC", "--steps", "64"],
                ["--joint-residency", "union", "--pool", "66"], ["--joint-residency", "union", "--joint-extra", "1"],
                ["--joint-residency", "union", "--spec", "2"], ["--joint-residency", "union", "--arms", "L4XU,L4XU"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    assert C.parse_args(base + ["--joint-residency", "union", "--arms", "SHR,UC,UCR"]).arms == ["SHR", "UC", "UCR"]
    assert C.parse_args(base + ["--joint-residency", "union", "--steps", "64"]).check_steps == 128   # timing arms only


def test_cell_pure_parts(monkeypatch):
    C = _cell()
    ph = dict(pre=1.0, twin=2.0, select=3.0, resid=4.0, union=0.5, fetch=1.5, attn=6.0, spill=0.25, merge=0.75,
              check=0.0, end=0.1, post=2.0, head=0.4)
    g = C.group_phases(ph)
    assert g["dense"] == 3.4 and g["residency"] == 4.5 and g["attention"] == 6.75 and g["fetch"] == 1.5
    assert g["total"] == round(sum(ph.values()), 4)
    s = C.phase_summary([ph, dict(ph, attn=8.0)], [[1.0, 2.0], [3.0, 4.0]])
    assert s["n_probes"] == 2 and s["labels_median"]["attn"] == 7.0 and s["layer_ms_median"] == 2.5
    assert C.phase_summary([], []) == dict(n_probes=0)
    assert C.per_pass(dict(gemm=10, attn=4), 2) == dict(gemm=5.0, attn=2.0)
    assert C.quantiles([3, 1, 2])["median"] == 2.0 and C.quantiles([])["n"] == 0
    assert C.cap_key(65) == "C128" and C.cap_key(10) == "C73"
    ev = _ev("B8")
    g = {n: ok for n, _, ok, _ in C.check_gates("UC", _res_of(ev), L=L)}
    assert all(g.values()), g
    bad = _res_of(ev)
    bad["sel"] = dict(bad["sel"], changed=1)
    assert {n: ok for n, _, ok, _ in C.check_gates("UC", bad, L=L)}["UC-SEL"] is False
    assert "UC-STRESS" in {n for n, *_ in C.check_gates("UC", _res_of(ev), require_stress=True, L=L)}
    rd = dict(verify=dict(calls=dict(resid=2 * L), passes=2))
    assert C.timing_union_gates("L4XU", rd, L=L)[0][2] is True
    rd["verify"]["calls"]["resid"] = 2 * L * R
    assert C.timing_union_gates("L4XU", rd, L=L)[0][2] is False
    for key in ("C128", "C73"):                                              # the builder's state: unregistered
        monkeypatch.setitem(C.PRED_WSB, key, dict(joint_sh_steps=None, ratio_vs_sh=None, ratio_vs_l4x=None))
    sh = dict(rates=dict(steady_step_ms=64.0))
    sp = dict(L4XU=dict(phases_median=dict(joint=180.0), vs_sh=dict(median_tok_s=0.5), rates=dict(median_tok_s=1700.0)),
              L4X=dict(rates=dict(median_tok_s=860.0)))
    v = {n: x for n, x, _ in C.verdicts(sh, sp, 65)}
    assert v == {"JOINT-SH-STEPS": "REPORT", "RATIO-VS-SH": "REPORT", "RATIO-VS-L4X": "REPORT", "VS-STRONGEST": "REPORT"}
    monkeypatch.setitem(C.PRED_WSB, "C128", dict(joint_sh_steps=(2.5, 3.0), ratio_vs_sh=(0.6, 0.7),
                                                 ratio_vs_l4x=(1.5, 2.5)))
    v = {n: x for n, x, _ in C.verdicts(sh, sp, 65)}
    assert v["JOINT-SH-STEPS"] == "CONFIRMED" and v["RATIO-VS-SH"] == "REFUTED" and v["RATIO-VS-L4X"] == "CONFIRMED"
    assert C.verdicts(sh, {}, 65) == [("L4XU-RUN", "NOT_RUN", "the arm did not run")]
    assert C.PRED_STAMP_WSB is None or isinstance(C.PRED_STAMP_WSB, str)


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_wsb_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_wb1():
    for bad in ("wb2", "sp5a", "vr1"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    r = _step("wb1m", sha)
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
    line = ("cell wb1m: C 73 (pool 10) B 8 arms SH+L4X,L4XU,SHR,UC,UCR steps 64 K 3 X 4 check 64 stress 1 STEP_MIN 40 "
            "mem 100G")
    assert line in r.stdout, r.stdout[-600:]
    for want in ("NOSI_POOL_BLOCKS=10", "NOSI_ATTN_SPLITS=4", "--pool 10", "--batch 8", "--steps 64",
                 "--joint-residency union", "--joint-extra 4", "--check-steps 64", "--require-stress 1",
                 "--mem=100G", "--time=40", "r2_wsb_cell.py", "wb_t1_%s_wb1m" % HOLDER):
        assert want in r.stdout, want
    r = _step("wb1ca", sha)
    assert r.returncode == 0 and "cell wb1ca: C 128 (pool 65) B 216 arms SH+SHR,UC,UCR steps 128" in r.stdout, r.stdout
    pred = _cell().PRED_STAMP_WSB
    for cell in ("wb1a", "wb1b"):
        r = _step(cell, sha)
        if pred is None:
            assert r.returncode == 1 and "PRED_STAMP_WSB is None" in r.stderr, (cell, r.stderr[-300:])
        else:
            assert r.returncode == 0 and "cell %s:" % cell in r.stdout, (cell, r.stdout[-300:])
