"""CPU tests of rung I3 (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the arm L4PTF =
WS-A's tentative buffers on the schedule of contract 4 (I1) with WS-B's union residency (I2) and WS-C's ONE fork call
per layer over the union (nosi/nosi/spec_tentative_fork.py; the arena with the chain images benchmarks/Efficiency/
r2_i3_arena.py; the cell r2_i3_cell.py; the step script r2_i3_step.sh). The world: tests_cpu/i3_world.py over i2_world /
wsc_world / wsa_world (the SHIPPED engine from its own source; every CUDA op a torch stand-in with the kernel's flat
addresses, lesson L28). The fork here is wsc_world.fake_fork (the CPU world has 2 query heads per KV head: BLOCK_M 8,
below the real launcher's 16); tests_cpu/test_i3_interp.py runs the REAL fork under the Triton interpreter (its own
srun). CPU stand-ins do not establish GPU dispatch, ordering, timing or memory (the cells i3m / i3a / i3b / i3ca / i3cb /
i3g do). FAST is checked on itself against SH (ordinary NOSA at the same configuration), not certified by an EXACT arm.

GATE 0   test_i3_gate0_files: the tree = I2's 63da01d except WS-C2's files (their db21536 bytes; test_wsb.py the merge of
         both DECLARED tuples) and the I3 files; in the older Gate-0 tests (test_wsa.py, test_wsb.py, test_wsc.py,
         test_i2.py) only module-level declarations and their Gate-0 functions changed (AST); the fork is fia6's blob;
         the name scans. test_i3_gate0_arms (B 8): L4PT, L4PTU, L4PJ bit for bit against the 63da01d SOURCE loaded from
         git (verify_rows + spec_round + spec_tentative + spec_tentative_union as one chain); L4XU, L4XF against the
         db21536 source (verify_rows + spec_round).
ARENA    test_chain_arena_layout; test_nocopy_prefill_hook (the shipped prefill re-homed WITHOUT a copy = the plain
         prefill, bitwise; every engine row tensor IS the arena's view; no other allocation served).
TOKENS   test_l4ptf_commits_sh_tokens (B 1, B 8 rowgroup, B 8 natural): SH's tokens (MUST), exactly N per request; logits
         within TOL; the selection of every committed verify / target row = SH's; residency and host rows after every
         round; PRESENT (dropped 0, unserved 0), VIS, VB from the kernel's tables, TAIL 0 (the committed tail slot never
         read), CHAIN 0 (the chain images' bytes and positions), G_block0. Preconditions: rejects, full accepts, an edge
         crossed, rows past an edge, F4 rows, draft and forced skips, pool moves, store and pool reads, chain reads.
COUNT    test_l4ptf_one_residency_one_fork_call: one residency, ONE fork call (an INDEPENDENT count of the callable),
         ONE store fill and one chain build per layer and joint pass. CONTROL: I2X (L4PTU at X 193) makes 3 attention
         calls per layer; the cell's ATTN1 gate catches it.
D1 / D3  test_l4ptf_deferred_and_residency (B 1, B 8): as test_i2's (no deferred-row write, K + 1 rows, one replay per
         group, the J slots' writes = the trace, no residency undo); PLANT recompute_deferred (R 5: two fork sequences
         per request) CAUGHT by the counter, tokens still SH's. test_l4ptf_rollback_fixed_cost: reset / commit a fixed
         number of torch calls; a reset writes no engine table; PLANT undo_residency CAUGHT.
SCHEDULE test_l4ptf_schedule_trace: the contract-4 trace, exactly N, SH's tokens across rejects, refills, finals, targets.
STATE    test_l4ptf_committed_state (B 1, B 8): WS-A's state classes against SH fed the committed tokens.
LIFETIME test_l4ptf_buffer_lifetime: dead buffer entries poisoned change no token. PLANTS no_causal (CAUGHT: VB and the
         logits), no_lower (CAUGHT: CHAIN and the logits).
TEACHER  test_l4ptf_teacher_forced_vs_sh (B 1, B 8, B 4 warm 3): the CPU twin of the check cell: every row's selection SET
         = SH's (MUST), argmax, logits within TOL, PRESENT / VIS / VB / TAIL / CHAIN 0, one residency and ONE fork call
         per layer and pass (independent counts), "F" / "RA" probes bitwise; rows past an edge and F4 rows exist.
BYTES    test_l4ptf_chain_bytes: every (row, block, token) the kernel's tables show holds the right bytes (host rows;
         CB0 = the committed tail rows below qc % 64 and the chain's buffer rows above; CB1 = the chain's rows of block
         qb + 1, positions <= p), in a teacher run and a schedule run.
PLANTS   fork_union_rows (VB / VIS, logits, foreign bytes), fork_small_store (PRESENT / VB; refused without the plant),
         fork_probe_swap (IDX), window_tail (TAIL, VB, logits).
WIRING   test_l4ptf_refusals_and_no_host_sync.
WS-G     test_i3_wsg_arm_and_plant: WF every gate REPORT and every part passes on the stand-ins; WFP (fork_small_store at
         X 2) MUST PRE and CAUGHT.
CELL     test_i3_cell_arguments, test_i3_cell_pure_parts, test_i3_registration_state, test_step_script_i3, test_notes_i3.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -s -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i3.py
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
I2_TIP = "63da01d2f63ae7e694b15eb8c709d591a4adf305"         # nosi-r2-i2: I2X with the lead's registrations
WSC2 = "db21536c4e97f651cbac8cc9b95a376ff5dc76db"           # nosi-r2-wsc2: WS-C 22c27ee + the miniature's schedule
WSB_TIP = "1a8ce68d9d6852fecf45b916c9ef1e6a5a80e098"         # nosi-r2-wsb: the base of WS-C
FORK_BLOB = "f0dc0c183a388585e216505a6bbfc696531e8add"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-i3"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
T_ = "benchmarks/Efficiency/tests_cpu/"
I3_NEW = ("nosi/nosi/spec_tentative_fork.py", "benchmarks/Efficiency/r2_i3_arena.py", "benchmarks/Efficiency/r2_i3_cell.py",
          "benchmarks/Efficiency/r2_i3_step.sh", T_ + "i3_world.py", T_ + "test_i3.py", T_ + "test_i3_interp.py",
          "docs/NOTES_i3.md")
# the older Gate-0 tests that I3 edits: {file: (the commit whose version it edits, the functions that may change)}
I3_GATE0_EDITED = {T_ + "test_wsa.py": (I2_TIP, {"test_wsa_flag_off_is_spec5"}),
                   T_ + "test_wsb.py": (I2_TIP, set()),
                   T_ + "test_wsc.py": (WSC2, set()),
                   T_ + "test_i2.py": (I2_TIP, {"test_i2_gate0_files"})}
# I4A (nosi-r2-i4a: this tree c4c64b2 + the arm L4PTFD): its files are declared too (its own Gate 0:
# tests_cpu/test_i4a.py; it pins this tree's files to c4c64b2's bytes except these and the declaration lines)
I3_NEW = I3_NEW + ("benchmarks/Efficiency/r2_i4a_lean.py", "benchmarks/Efficiency/r2_i4a_cell.py",
                   "benchmarks/Efficiency/r2_i4a_step.sh", T_ + "i4a_world.py", T_ + "test_i4a.py",
                   "docs/NOTES_i4a.md")                                                          # I4A
# I4U (nosi-r2-i4u: I4A acbe433 + the arm L4PTFDU): its files (its own Gate 0: tests_cpu/test_i4u.py)
I3_NEW = I3_NEW + ("benchmarks/Efficiency/r2_i4u_hoist.py", "benchmarks/Efficiency/r2_i4u_cell.py",
                   "benchmarks/Efficiency/r2_i4u_step.sh", T_ + "i4u_world.py", T_ + "test_i4u.py")       # I4U


def _load(name, path=None):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(path or (HERE / (name + ".py"))))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


I3 = _load("i3_world")
WT = _load("wsa_tests_lib", HERE / "test_wsa.py")           # WS-A's test helpers (spec_trace, the row checks, the cost run)
I, Z, WW, WB, SW, W, RW, VW, RE, SR, BC, ST, STU, VR, STF, KA, IA = (
    I3.I, I3.Z, I3.WW, I3.WB, I3.SW, I3.W, I3.RW, I3.VW, I3.RE, I3.SR, I3.BC, I3.ST, I3.STU, I3.VR, I3.STF, I3.KA,
    I3.IA)
K, N, L, BS, TOL = I3.K0, I3.N0, I3.L, I3.BS, I3.TOL
R = K + 1
FILTER = WT.FILTER
S1 = [4658]


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True)


def _have(sha) -> bool:
    return _git("rev-parse", "-q", "--verify", sha + "^{commit}").returncode == 0


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_i3_cell
    return r2_i3_cell


def _units(src: str) -> dict:
    """{name: ast.dump}: top-level functions, class methods (Class.method), the other statements (#i)."""
    out, k = {}, 0
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.ClassDef):
            out[n.name + ".<bases>"] = ast.dump(ast.Tuple(elts=list(n.bases), ctx=ast.Load()))
            j = 0
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    out["%s.%s" % (n.name, m.name)] = ast.dump(m)
                else:
                    out["%s.#%d" % (n.name, j)] = ast.dump(m)
                    j += 1
        else:
            out["#%d" % k] = ast.dump(n)
            k += 1
    return out


# ------------------------------------------------------------------ GATE 0: the files
def test_i3_gate0_files():
    if not (_have(I2_TIP) and _have(WSC2) and _have(WSB_TIP)):
        pytest.skip("no git or no commit %s / %s / %s in %s" % (I2_TIP, WSC2, WSB_TIP, ROOT))
    wsc_files = {x for x in _git("diff", "--name-only", WSB_TIP, WSC2).stdout.split("\n") if x}
    changed = {x for x in _git("diff", "--name-only", I2_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**").stdout
               .split("\n") if x}
    allowed = wsc_files | set(I3_NEW) | set(I3_GATE0_EDITED)
    assert changed <= allowed, sorted(changed - allowed)                    # (a) nothing else moved
    assert all((ROOT / f).is_file() for f in I3_NEW), [f for f in I3_NEW if not (ROOT / f).is_file()]
    keep = sorted(f for f in wsc_files if f not in I3_GATE0_EDITED)        # (b) WS-C2's bytes
    assert _git("diff", "--name-only", WSC2, "--", *keep).stdout.strip() == ""
    for f, (sha, fns) in I3_GATE0_EDITED.items():                         # (c) only Gate-0 functions / declarations
        old = {k: v for k, v in _units(_git("show", "%s:%s" % (sha, f)).stdout).items() if not k.startswith("#")}
        new = {k: v for k, v in _units((ROOT / f).read_text()).items() if not k.startswith("#")}
        assert set(old) == set(new) and {k for k in old if old[k] != new[k]} <= fns, (f, sorted(
            k for k in set(old) | set(new) if old.get(k) != new.get(k)))
    fork = EFF / "fork_vllm_unified_attention.py"                          # (d) the fork: fia6's blob, NO NEW KERNEL
    assert _git("rev-parse", "HEAD:benchmarks/Efficiency/fork_vllm_unified_attention.py").stdout.strip() == FORK_BLOB
    assert _git("hash-object", str(fork)).stdout.strip() == FORK_BLOB
    r = _git("ls-files", "--others", "--exclude-standard", "--", "nosi", ":(exclude,glob)**/__pycache__/**")
    assert r.stdout.strip() == "", "untracked files in nosi/: %s" % r.stdout
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}        # (e) the name scans
    for n, t in src.items():                                     # the kernel and the arena: injected, not imported
        assert not re.search(r"(import|from)\s+[\w.]*(fork_vllm_unified_attention|r2_kv_arena|r2_i3_arena)\b", t), n
    assert [n for n, t in src.items() if "_fork_rows" in t] == ["verify_rows.py"]
    assert [n for n, t in src.items() if "spec_tentative_fork" in t] == []     # no nosi module names the I3 module
    for name in ("ragged_rollback", "state_snapshot", "spec_prefetch"):
        assert name not in src["spec_tentative_fork.py"], name
    assert "@triton.jit" not in src["spec_tentative_fork.py"] and "triton" not in (EFF / "r2_i3_arena.py").read_text()
    for c in ("r2_spec_cell.py", "r2_base_cell.py", "r2_wsa_cell.py", "r2_wsb_cell.py", "r2_wsc_cell.py", "r2_i2_cell.py",
              "r2_i2_check.py", "r2_wsg_check.py", "r2_kv_arena.py"):
        t = (EFF / c).read_text()
        assert "spec_tentative_fork" not in t and "r2_i3" not in t, c


# ------------------------------------------------------------------ GATE 0: the arms of both branches, bit for bit
CASES0 = [("L4PT", I2_TIP), ("L4PTU", I2_TIP), ("L4PJ", I2_TIP), ("L4XU", WSC2), ("L4XF", WSC2)]


@pytest.mark.parametrize("arm,sha", CASES0, ids=[c[0] for c in CASES0])
def test_i3_gate0_arms(arm, sha, tmp_path):
    if not _have(sha):
        pytest.skip("no git or no commit %s in %s" % (sha, ROOT))
    names = (["verify_rows", "spec_round", "spec_tentative", "spec_tentative_union"] if sha == I2_TIP
             else ["verify_rows", "spec_round"])
    mods = I.old_chain(sha, names, tmp_path, tag="i3" + arm.lower())
    if sha == I2_TIP:                                            # precondition: the chain is coherent
        assert mods["spec_tentative"].SR is mods["spec_round"] and mods["spec_tentative"].VR is mods["verify_rows"]
        assert mods["spec_tentative_union"].ST is mods["spec_tentative"]
    new, old = I3.arm_run(arm, 8), I3.arm_run(arm, 8, mods)
    bad = I.same_run(new, old)
    assert bad == [], (arm, bad)
    f = new["facts"]
    assert f["rejects"] > 0 and f["full"] > 0, f                      # both accept paths ran (L34)


# ------------------------------------------------------------------ ARENA
def test_chain_arena_layout():
    Bq, ns, X = 3, W.TOPK + W.P, 5
    a = IA.ChainArena(W.L, Bq, ns, W.HKV, W.D, X, W.DT, "cpu")
    assert a.store_base == W.L * Bq * ns and a.cb0_base == a.store_base + Bq * X and a.n_blocks == a.cb0_base + Bq
    assert tuple(a.k.shape) == (a.n_blocks, BS, W.HKV, W.D) and tuple(a.bias.shape) == (a.n_blocks, BS, W.HKV)
    assert tuple(a.store_k.shape) == (Bq, X * BS, W.HKV, W.D) and a.store_k.data_ptr() == a.k[a.store_base].data_ptr()
    assert tuple(a.cb0_k.shape) == (Bq, BS, W.HKV, W.D) and a.cb0_k.data_ptr() == a.k[a.cb0_base].data_ptr()
    ck, cv, cb = a.cb1_views()
    assert tuple(ck.shape) == (Bq, BS, W.HKV, W.D) and ck.data_ptr() == a.k[a.store_base].data_ptr()
    assert ck[1].data_ptr() == a.k[a.store_base + X].data_ptr()        # request 1's store slot 0
    for t in (a.store_k, a.store_v, a.store_bias, a.cb0_k, a.cb0_v, a.cb0_bias):
        assert bool((t == 0).all())
    d = a.describe()
    assert d["kind"] == "chain" and d["cb0_base"] == a.cb0_base and d["cb1"] == "store slot 0"
    assert a.chain_bytes() == Bq * BS * W.HKV * (2 * W.D + 1) * 4
    assert isinstance(a, KA.KVArena)                                   # WS-C's checks and views apply unchanged
    with pytest.raises(KA.ArenaError):
        IA.ChainArena(W.L, Bq, ns, W.HKV, W.D, 1, W.DT, "cpu")
    C = _cell()                                                        # the cell's memory model: the same formula
    assert C.cb0_gb(318) == C.store_gb(318, 1) == round(318 * 64 * 2 * 257 * 2 / 1e9, 4)
    assert C.layer_rows_gb(216, 65) == C.store_gb(216, 129) == 1.8332


def test_nocopy_prefill_hook():
    B, S = 2, 4200
    rows = W.prompt_rows(1, [S] * B)
    c1, c2, c3 = W.new_cache(40), W.new_cache(40), W.new_cache(40)
    W.shipped_prefill(c1, rows, S, B)
    arena = IA.for_engine(c2.layers[0].cache_engine, W.L, B, 5)
    with IA.NoCopyPrefillHook(c2, arena) as hook:
        W.shipped_prefill(c2, rows, S, B)
    assert hook.bound_at == list(range(W.L)) and KA.check(c2, arena) == []
    assert all(hook.served[l] == list(KA.ENGINE_ROWS) for l in range(W.L)), hook.served   # the views, no copy
    assert all("prefill_update" not in lay.cache_engine.__dict__ for lay in c2.layers)      # the wrappers are gone
    g = type(c2.layers[0].cache_engine).prefill_update.__globals__
    assert g["torch"] is W.TP                                          # the name is put back
    for l, lay in enumerate(c2.layers):                                # the engine's tensors ARE the arena's views
        for name, t in zip(KA.ENGINE_ROWS, arena.layer_views(l)):
            assert getattr(lay.cache_engine, name).data_ptr() == t.data_ptr(), (l, name)
    arena3 = KA.for_engine(c3.layers[0].cache_engine, W.L, B, 5)       # CONTROL: WS-C's copying hook, same bits
    with KA.PrefillHook(c3, arena3):
        W.shipped_prefill(c3, rows, S, B)
    for c in (c1, c2, c3):
        W.poison_shipped(c, S)
    assert VW.engines_equal(c1, c2) == [] and VW.engines_equal(c1, c3) == []    # the re-homed prefill = the plain one
    c4 = W.new_cache(40)                                               # a wrong arena: refused
    with pytest.raises(KA.ArenaError):
        with IA.NoCopyPrefillHook(c4, IA.for_engine(c4.layers[0].cache_engine, W.L, B + 1, 5)):
            W.shipped_prefill(c4, rows, S, B)


# ------------------------------------------------------------------ L4PTF on the stand-ins
@functools.lru_cache(maxsize=None)
def _evf(B, gemm="rowgroup", X=None, filt="periodic", tneg=None, poison=False, ref=True, vneg=None, fneg=None,
         max_rounds=0, vrecord=False):
    """One run_tent of L4PTF per parameter set; ev["diff"] / ev["fork"]: the independent residency / fork counts."""
    build = I3.tentf_builder(X=X, gemm=gemm, vneg=vneg, fneg=fneg, vrecord=vrecord)
    box = {}

    def on_runner(runner):
        box["diff"] = I.count_joint_diff(runner)
        box["fork"] = I3.count_fork_calls(runner)
    ev = I3.run_tentf(B, build=build, draft_filter=WT._filter(filt), tneg=tneg, poison=poison, with_reference=ref,
                      max_rounds=max_rounds, on_runner=on_runner)
    ev["diff"], ev["fork"] = box.get("diff"), box.get("fork")
    return ev


CASES_T = dict(B1=(1, "rowgroup"), B8=(8, "rowgroup"), B8nat=(8, "natural"))


@pytest.mark.parametrize("case", list(CASES_T))
def test_l4ptf_commits_sh_tokens(case):
    B, gemm = CASES_T[case]
    ev = _evf(B, gemm=gemm)
    f, rd = ev["facts"], ev["rd"]
    u = rd["union"]
    print("I3 L4PTF %s: max |logit - SH's| over %d committed rows = %.3e (TOL %.0e); selections compared %d; pool reads "
          "%d, store reads %d, chain reads %d, past-edge rows %d, F4 rows %d, chain rows checked %d" % (
              case, ev["rows_compared"], ev["max_logit_delta"], TOL, ev["sel_compared"], u["pool_reads"],
              u["store_reads"], u["chain_reads"], u["past_edge"], u["f4_rows"], u["chain_rows"]))
    assert f["rejects"] > 0 and f["full"] > 0, f                      # preconditions (L34)
    assert f["crossed"] > 0 and f["edge_rows"] > 0 and f["fill_rows"] > 0, f
    assert f["skipped"] > 0 and f["forced"] > 0 and f["move_out"] + f["swap"] > 0, f
    assert u["passes"] > 0 and u["union_samples"] > 0 and u["chain_reads"] > 0 and u["chain_rows"] > 0, u
    if B > 1:
        assert f["ragged"] and u["past_edge"] > 0 and u["f4_rows"] > 0, (f, u)
        assert u["pool_reads"] + u["store_reads"] > 0, u               # rows read blocks outside the window
    assert rd["count"] == [N] * B, rd["count"]                        # exactly N (contract 4)
    assert ev["ref_logits_bad"] == [], ev["ref_logits_bad"][:5]        # the reference IS SH
    assert ev["bad_tokens"] == [], ev["bad_tokens"][:5]                # THE CLAIM: SH's tokens
    assert ev["max_logit_delta"] <= TOL, ev["max_logit_delta"]
    assert ev["sel_compared"] > 0 and ev["sel_mismatch"] == [], ev["sel_mismatch"][:5]
    assert ev["resid"] == [] and ev["host"] == [], (ev["resid"][:2], ev["host"][:2])
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert u["dropped"] == 0 and u["unserved"] == 0 and u["vis_bad"] == 0, u      # present; each row its own blocks
    assert u["vb_bad"] == 0 and u["vb_rows"] > 0 and u["block0_bad"] == 0, u       # VB from the kernel's tables
    assert u["tail_reads"] == 0 and u["chain_bad"] == 0, u                          # TAIL, CHAIN
    assert u["row_dropped"] == [0] * R, u
    assert u["store_slots"] == KA.store_slots(W.TOPK, K) and u["chunks"] == [R] and u["fills"] == 1, u


def test_l4ptf_one_residency_one_fork_call():
    ev = _evf(8)
    c, u = ev["rd"]["tent"]["calls"], ev["rd"]["union"]
    jp = c["joint_passes"]
    assert jp > 0 and ev["diff"]["per_pass"] == [L] * jp, (jp, ev["diff"]["per_pass"][:5])   # INDEPENDENT counts
    assert ev["fork"]["per_pass"] == [L] * jp, ev["fork"]["per_pass"][:5]
    assert c["joint_resid"] == L * jp and c["union_layers"] == L * jp and c["chain"] == L * jp, c
    assert u["calls"]["resid"] == L * jp and u["calls"]["attn"] == L * jp and u["calls"]["fills"] == L * jp, u["calls"]
    assert u["passes"] == jp
    C = _cell()
    g = {n: ok for n, _, ok, _ in C.fork_gates("L4PTF", dict(read=ev["rd"], fa_check={}), L, check=False)}
    assert g == {"L4PTF-RESID": True, "L4PTF-ATTN1": True, "L4PTF-FILL1": True}, g
    x = I.run_tentu(8, build=I.tentu_builder(X=KA.store_slots(W.TOPK, R)), draft_filter=FILTER, with_reference=False)
    xu = x["rd"]["union"]                                              # CONTROL: I2X, X 193 (C = 1)
    assert xu["fills"] == 1 and xu["calls"]["attn"] == 3 * L * xu["passes"], xu["calls"]   # window + store + buffer
    g = {n: ok for n, _, ok, _ in C.fork_gates("L4PTU", dict(read=x["rd"], fa_check={}), L, check=False)}
    assert g["L4PTU-ATTN1"] is False and g["L4PTU-RESID"] is True, g   # the cell's gate catches it
    assert x["bad_tokens"] == []


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_l4ptf_deferred_and_residency(B):
    ev = _evf(B)
    tr, log = ev["rd"]["tent"], ev["runner"].T.rec
    c = tr["calls"]
    assert tr["deferred_writes"] == 0, tr["deferred_writes"]             # D3 (the cell's MUST gate)
    assert c["joint_rows"] == R * c["joint_passes"] and c["joint_passes"] > 0, c
    assert c["joint_passes"] + c["j1_passes"] == ev["rd"]["rounds"], c
    assert tr["replays"] == WT._groups(ev) > 0, (tr["replays"], WT._groups(ev))
    assert ev["rd"]["tokens_acct"]["remade_rows"] == 0
    assert WT._joint_row_errors(log, ev["rd"]["trace_all"], B) == []
    errs, seen = WT._replay_errors(log, B)
    assert errs == [] and seen == WT._groups(ev), (errs[:5], seen)
    assert WT._pos_errors(log) == []
    assert c["residency_undo"] == 0                                    # D1
    p = _evf(B, tneg="recompute_deferred", ref=False, max_rounds=12)   # PLANT: EXACT's K + 2 rows (two fork sequences)
    pt = p["rd"]["tent"]
    assert pt["deferred_writes"] > 0 and pt["calls"]["joint_rows"] == (K + 2) * pt["calls"]["joint_passes"], pt
    assert p["rd"]["union"]["chunks"] == [4, 1], p["rd"]["union"]
    assert p["bad_tokens"] == []                                       # a COST plant: the tokens stay SH's


def test_l4ptf_rollback_fixed_cost():
    with I._builder(I3.tentf_builder()):
        runs = [WT._cost_run(WT.FILTER), WT._cost_run(WT.FILTER0)]
        ev, _, facts = WT._cost_run(WT.FILTER, tneg="undo_residency")  # PLANT: a residency undo
    resets = {n for _, box, _ in runs for n in box["reset"]}
    commits = {n for _, box, _ in runs for n in box["commit"]}
    drops = [e["rd"]["tent"]["dropped"] for e, _, _ in runs]
    print("I3 COST: torch calls per reset %s, per commit %s; dropped entries %s" % (sorted(resets), sorted(commits),
                                                                                   drops))
    assert drops[0] != drops[1], drops                                  # precondition
    assert len(resets) == 1 and len(commits) == 1, (resets, commits)
    for e, _, fx in runs:
        assert fx["changed"] > 0 and fx["reset_wrote"] == [], fx["reset_wrote"][:3]
        assert e["rd"]["tent"]["dropped"] == WT._dropped_recount(e)
        assert e["rd"]["tent"]["calls"]["residency_undo"] == 0 and e["bad_tokens"] == []
        assert e["rd"]["tent"]["calls"]["chain"] > 0                   # the fork pass ran in these runs
    assert facts["runner"].T.calls["residency_undo"] > 0 and facts["reset_wrote"] != []   # CAUGHT


def test_l4ptf_schedule_trace():
    B = 8
    ev = _evf(B, filt="end", ref=False)
    rd = ev["rd"]
    acct = rd["tokens_acct"]
    assert acct["corrected"] > 0 and acct["fills"] > 0 and acct["finals"] == B and acct["targets"] > 0, acct
    assert rd["count"] == [N] * B, rd["count"]
    errs, remade = WT._trace_errors(ev, B, False)
    assert errs == [] and remade == [], (errs[:3], remade[:3])
    log, tr_all = ev["runner"].T.rec, rd["trace_all"]
    assert WT._joint_row_errors(log, tr_all, B) == []
    assert WT._slot_row_errors(log, tr_all, B) == []
    assert ev["bad_tokens"] == [] and ev["max_logit_delta"] <= TOL, (ev["bad_tokens"][:3], ev["max_logit_delta"])
    assert ev["resid"] == [] and ev["host"] == []
    u = rd["union"]
    assert u["vb_bad"] == 0 and u["chain_bad"] == 0 and u["tail_reads"] == 0 and u["dropped"] == 0, u
    for b in range(B):
        t = tr_all[b]
        k = max(i for i, x in enumerate(t) if x["role"] == "final")
        assert {x["role"] for x in t[k + 1:]} <= {"target", "idle"}
        assert sum(x["committed"] for x in t) == N - rd["warm"]


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_l4ptf_committed_state(B):
    ev = _evf(B)
    run = ev["runner"]
    assert ev["facts"]["rejects"] > 0 and ev["facts"]["crossed"] > 0
    run.T.reset()                                 # the twins to the committed counters (the chain is not committed)
    qc = run.T.qc.clone()
    assert torch.equal(qc.cpu(), ev["S"] + ev["cnt"])
    sc = WW.state_classes(run.rs, ev["ref_rs"], qc, ev["S"])
    print("I3 STATE B %d: EXACT differ %s; layer-0 not bitwise %s; beyond TOL %s; max |delta| %.3e; CACHE %s" % (
        B, sc["exact"], sc["l0"], sc["close"], sc["maxabs"], sc["cache"]))
    assert sc["exact"] == [] and sc["l0"] == [] and sc["close"] == [], (sc["exact"], sc["l0"], sc["close"][:5])
    assert WT._host_vs_shipped(ev) == []                              # the complete committed blocks: SH's host rows


def test_l4ptf_buffer_lifetime():
    ev = _evf(8, poison=True, ref=False)
    assert any(bool((t == ST.POISON_V).any()) for t in ev["runner"].T.v)    # precondition: the poison acted
    assert ev["bad_tokens"] == [] and ev["max_logit_delta"] <= TOL, (ev["bad_tokens"][:3], ev["max_logit_delta"])
    assert ev["rd"]["union"]["chain_bad"] == 0
    p = _evf(8, poison=True, tneg="no_causal", ref=False)              # PLANT: every row sees its own block whole
    assert p["bad_tokens"] != [] or p["max_logit_delta"] > TOL, "no_causal"
    assert p["rd"]["union"]["vb_bad"] > 0, p["rd"]["union"]            # CAUGHT by VB
    p = _evf(8, poison=True, tneg="no_lower", ref=False)               # PLANT: CB0 without the committed rows
    assert p["bad_tokens"] != [] or p["max_logit_delta"] > TOL, "no_lower"
    assert p["rd"]["union"]["chain_bad"] > 0, p["rd"]["union"]         # CAUGHT by CHAIN


# ------------------------------------------------------------------ the check cell on the CPU
@functools.lru_cache(maxsize=None)
def _tf(B, G=6, warm=1, vneg=None, fneg=None, vrecord=False, X=None, probe=True):
    return I3.run_teacher_f(B, G=G, warm=warm, vneg=vneg, fneg=fneg, vrecord=vrecord, X=X, probe=probe)


def _res_tf(ev):
    """A check arm's result (r2_i3_cell.check_gates_tf's input) from a CPU teacher run."""
    return dict(stats=ev["stats"], errors=ev["errors"], fa_check=ev["fa"],
                sel=dict(compared=ev["compared"], changed=len(ev["changed"]), by_layer_nonzero={}, symdiff_hist={},
                         first=ev["changed"][:5]))


CASES_TF = dict(B1=(1, 1), B8=(8, 1), B4w3=(4, 3))


@pytest.mark.parametrize("case", list(CASES_TF))
def test_l4ptf_teacher_forced_vs_sh(case):
    B, warm = CASES_TF[case]
    G = 6
    ev = _tf(B, G=G, warm=warm)
    st = ev["stats"]
    facts = STF.teacher_facts(ev["S"], warm, G, R)
    print("I3 TEACHER %s: max |logit - SH's| %.3e over %d rows; selection sets changed %d of %d; past-edge %d, F4 %d, "
          "pool reads %d, store reads %d, chain rows %d" % (
              case, ev["max_delta"], len(ev["deltas"]), len(ev["changed"]), ev["compared"] * W.HKV, st["past_edge"],
              st["f4_rows"], st["pool_reads"], st["store_reads"], st["chain_rows"]))
    assert ev["ref_logits_bad"] == [], ev["ref_logits_bad"][:5]
    assert ev["past_rows"] > 0 and facts["past_edge_rows"] > 0, (ev["past_rows"], facts)   # preconditions (L34)
    assert st["past_edge"] == facts["past_edge_rows"] * B * L and st["f4_rows"] == facts["f4_rows"] * B * L, (st, facts)
    assert facts == _cell().check_facts(ev["S"], warm, G, R)          # the cell's copy of the arithmetic
    assert ev["compared"] == G * L * R * B, ev["compared"]
    assert ev["changed"] == [], ev["changed"][:8]                     # THE CLAIM: SH's selections, every row
    assert ev["argmax_bad"] == [] and ev["max_delta"] <= TOL, (ev["argmax_bad"][:5], ev["max_delta"])
    assert st["dropped"] == 0 and st["unserved"] == 0 and st["vis_bad"] == 0 and st["vb_bad"] == 0, st
    assert st["tail_reads"] == 0 and st["chain_bad"] == 0 and st["chain_rows"] > 0 and st["block0_bad"] == 0, st
    assert st["rows_full"] == G * L * R * B and st["vb_rows"] == G * L * W.HKV * R * B, st
    assert ev["diffs"] == [L] * G and st["calls"]["resid"] == L * G and st["passes"] == G, (ev["diffs"], st["calls"])
    picks = len({0, B // 2, B - 1})
    assert ev["forks"] == [L + 2 * picks] + [L] * (G - 1), ev["forks"]     # ONE fork call per layer (+ the probes)
    assert st["calls"]["attn"] == L * G and st["calls"]["fills"] == L * G, st["calls"]
    fa = ev["fa"]
    assert fa["F"]["entries"] and all(fa["F"]["equal"]) and fa["RA"]["entries"] and all(fa["RA"]["equal"]), fa
    assert ev["errors"] == (0, [0] * L), ev["errors"]
    g = {n: ok for n, _, ok, _ in _cell().check_gates_tf("TFC", _res_tf(ev), L=L)}
    assert all(g.values()), g                                          # the check cell's gates on the CPU evidence
    bad = dict(_res_tf(ev), sel=dict(_res_tf(ev)["sel"], changed=1))
    assert {n: ok for n, _, ok, _ in _cell().check_gates_tf("TFC", bad, L=L)}["TFC-SEL"] is False


def test_l4ptf_chain_bytes():
    ev = _tf(8, G=6, vrecord=True, probe=False)
    u = I3.chain_bytes(ev["runner"].vs.log, R)
    print("I3 BYTES (teacher): checked %d, chain checked %d" % (u["checked"], u["chain_checked"]))
    assert u["checked"] > 0 and u["chain_checked"] > 0, u
    assert u["bad_contents"] == [] and u["foreign"] == [] and u["cross_request"] == [] and u["tail_slot"] == [], (
        u["bad_contents"][:5], u["foreign"][:5], u["cross_request"][:5], u["tail_slot"][:5])
    ev = _evf(8, vrecord=True, ref=False, max_rounds=8)                 # a schedule run: drafts, deferred rows
    u = I3.chain_bytes(ev["runner"].vs.log, R)
    print("I3 BYTES (schedule): checked %d, chain checked %d" % (u["checked"], u["chain_checked"]))
    assert u["checked"] > 0 and u["chain_checked"] > 0, u
    assert u["bad_contents"] == [] and u["foreign"] == [] and u["cross_request"] == [] and u["tail_slot"] == [], (
        u["bad_contents"][:5], u["foreign"][:5])


# ------------------------------------------------------------------ PLANTS
def test_fork_union_rows_plant_caught():
    clean = _tf(8)
    assert clean["stats"]["union_max"] > W.TOPK                        # precondition: unions larger than rows
    ev = _tf(8, vneg="fork_union_rows", vrecord=True, probe=False)
    st = ev["stats"]
    assert st["vb_bad"] > 0 and st["vis_bad"] > 0, st
    assert ev["max_delta"] > TOL or ev["argmax_bad"] or ev["changed"], ev["max_delta"]   # the logits leave SH's
    g = {n: ok for n, _, ok, _ in _cell().check_gates_tf("TFC", dict(_res_tf(ev), fa_check=clean["fa"]), L=L)}
    assert g["TFC-VB"] is False and g["TFC-VIS"] is False, g
    assert I3.chain_bytes(ev["runner"].vs.log, R)["foreign"], "the byte check sees the foreign blocks"


def test_fork_small_store_plant_caught():
    ev = _tf(8, vneg="fork_small_store", X=2)
    st = ev["stats"]
    assert st["extras_host"] + st["unserved"] > 1, st                 # precondition: more host blocks than X - 1
    assert st["unserved"] > 0 and st["dropped"] > 0 and st["vb_bad"] > 0, st
    g = {n: ok for n, _, ok, _ in _cell().check_gates_tf("TFC", _res_tf(ev), L=L)}
    assert g["TFC-PRESENT"] is False and g["TFC-VB"] is False, g
    with pytest.raises(SR.SpecError):                                  # without the plant the small store is refused
        _tf(8, X=2, G=1)


def test_fork_probe_swap_plant_caught():
    ev = _tf(8, G=3, vneg="fork_probe_swap")
    fa = ev["fa"]
    assert not all(fa["F"]["equal"]) and all(fa["RA"]["equal"]), fa   # the probe is not blind
    g = {n: ok for n, _, ok, _ in _cell().check_gates_tf("TFC", _res_tf(ev), L=L)}
    assert g["TFC-IDX"] is False and g["TFC-RA"] is True, g
    clean = _tf(8, G=3)
    assert ev["changed"] == clean["changed"] and ev["max_delta"] == clean["max_delta"]   # the plant is check-only


def test_window_tail_plant_caught():
    ev = _tf(8, fneg="window_tail", vrecord=True, probe=False)
    st = ev["stats"]
    assert st["tail_reads"] > 0 and st["vb_bad"] > 0, st               # CAUGHT: TAIL and VB
    assert ev["max_delta"] > TOL or ev["argmax_bad"] or ev["changed"], ev["max_delta"]   # block qb's tentative rows unseen
    assert I3.chain_bytes(ev["runner"].vs.log, R)["tail_slot"], "the byte check sees the tail-slot reads"
    g = {n: ok for n, _, ok, _ in _cell().check_gates_tf("TFC", dict(_res_tf(ev), fa_check=_tf(8)["fa"]), L=L)}
    assert g["TFC-TAIL"] is False and g["TFC-VB"] is False, g


# ------------------------------------------------------------------ WIRING
def _small(B=2, N_=8, **kw):
    S = I3.S0
    hz = SW.horizon_for(N_, K)
    model, _, c = SW.two_caches(B, S, hz + 64, 0)
    tok0 = torch.randint(0, I3.V, (B,), generator=torch.Generator().manual_seed(2))
    return I3.tentf_builder(**kw)(model, c, S, hz, tok0, level="L4P", hooks=None, check_every=0, record=False)


def test_l4ptf_refusals_and_no_host_sync():
    f = Z.fake_fork
    ops = lambda **k: SR.SpecOps(**dict(dict(level="L4P", K=K, attn=WB.fake_attention_u), **k))   # noqa: E731
    model, rs = RW.build(S1)
    for bad, kw in ((dict(level="L3"), {}), ({}, dict(fneg="nope")), ({}, dict(vneg="drop_spill")),
                    ({}, dict(tneg="nope")), ({}, dict(fork=None))):
        args = dict(dict(fork=f), **kw)
        with pytest.raises(SR.SpecError):
            STF.TentRunnerF(model, rs, ops(**bad), BC.HostClock(), **args)        # no arena: refused too
    model, rs = RW.build(S1)
    KA.attach(rs, I3.Z.arena_for(rs.cache, 1, KA.store_slots(W.TOPK, K)))         # WS-C's plain arena: no CB0
    with pytest.raises(SR.SpecError):
        STF.TentRunnerF(model, rs, ops(), BC.HostClock(), fork=f)
    model, rs = RW.build(S1)
    KA.attach(rs, I3.chain_arena(rs.cache, 1, KA.store_slots(W.TOPK, K) - 1))     # below the bound
    with pytest.raises(SR.SpecError):
        STF.TentRunnerF(model, rs, ops(), BC.HostClock(), fork=f)
    assert rs.ops.kernels["gather_k"] is W.KERNELS["gather_k"]         # every refusal before the meter attached
    run = STF.TentRunnerF(model, rs, ops(), BC.HostClock(), fork=f, vneg="fork_small_store")   # the plant accepts it
    assert run.vs.X == KA.store_slots(W.TOPK, K) - 1 and run.vs.meta["full_max"] == K
    with pytest.raises(SR.SpecError):                                  # run_teacher needs full_max R
        run.run_teacher(torch.zeros((1, 1 + R), dtype=torch.int64), 1)
    for path, allowed in ((Path(STF.__file__), set()), (Path(ST.__file__), {"_run_paper", "_ps_after"}),
                          (Path(VR.__file__), set())):
        run = _small(check=False, stats=False)
        with RW.SyncTrap(path) as trap:
            run.run(8, rounds=3)
        fns = {h[0] for h in trap.hits}
        assert fns <= allowed, (path.name, trap.hits[:10])             # only the per-round sync
        if allowed:
            assert fns, "the trap is not blind"
        assert run.T.calls["chain"] > 0                                # the fork pass ran


# ------------------------------------------------------------------ WS-G's check (cell i3g) on the stand-ins
def test_i3_wsg_arm_and_plant():
    C = _cell()
    ev = I3.run_wsg_cpu_f("WF", draft_filter=WT._filter("end"))
    assert ev["restored"] and ev["ref_restored"]                       # every wrapper is back after the run
    assert ev["rd"]["tokens_acct"]["targets"] > 0
    g, rec = C.wsg_gates_f("WF", ev["cap"], ev["ref"])
    print("I3 WS-G WF: %s; logit %s" % (rec["details"], rec["logit"]["detail"]))
    assert all(not m for _, m, _, _ in g)                              # every gate of the FAST arm is a REPORT
    assert ev["cap"]["miss_checked"] > 0 and ev["cap"]["sel_columns"] > 0, ev["cap"]
    assert all(rec["parts"].values()), rec["details"]                  # on the stand-ins FAST passes every part
    p = I3.run_wsg_cpu_f("WFP", X=2, vneg="fork_small_store", draft_filter=WT._filter("end"))
    assert p["rd"]["union"]["unserved"] > 0, p["rd"]["union"]          # precondition: the plant dropped blocks
    g, rec = C.wsg_gates_f("WFP", p["cap"], p["ref"])
    gd = {n: (m, ok) for n, m, ok, _ in g}
    print("I3 WS-G WFP (planted): failed parts %s; miss %d" % (rec["failed"], p["cap"]["miss"]))
    assert p["cap"]["miss"] > 0 and not rec["parts"]["MISS"] and not rec["ok"], rec["details"]
    assert gd["WFP-CHK"] == (False, False)                             # a REPORT
    assert gd["WFP-PLANT-CHK-PRE"] == (True, True) and gd["WFP-PLANT-CHK-CAUGHT"] == (True, True)   # MUST, CAUGHT


# ------------------------------------------------------------------ CELL
def test_i3_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base + ["--joint-attn", "fork"])
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.joint_extra, a.fork_store, a.prefill_hook, a.tu_gemm,
            a.check_passes, a.check_phase, a.check_every, a.require_stress, a.probe_every) == (
        ["L4PT", "L4PTU", "L4PTF"], 216, 512, 65, 3, 193, 0, "nocopy", "natural", 32, 0, 0, 0, 16)
    assert C.parse_args(base + ["--arms", "L4PT,L4PTU"]).joint_attn == "store"     # the flag is off by default
    fk = ["--joint-attn", "fork"]
    for bad in ([], fk + ["--arms", "L4PT"], ["--arms", "L4PTF"], fk + ["--arms", "TFC,SHR"], fk + ["--arms", "TFC"],
                fk + ["--spec", "4"], fk + ["--arms", "SHR,TFC", "--check-passes", "200"],
                fk + ["--arms", "SHR,TFC", "--steps", "129", "--check-phase", "1"],
                fk + ["--fork-store", "150"], fk + ["--arms", "L4PTF,L4PTF"], fk + ["--arms", "WFP"],
                fk + ["--check-phase", "64"], fk + ["--pool", "0"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    a = C.parse_args(base + fk + ["--arms", "SHR,TFC", "--steps", "129"])
    assert a.arms == ["SHR", "TFC"] and 1 + a.check_phase + a.check_passes * R == 129
    a = C.parse_args(base + fk + ["--arms", "L4PT,L4PTU,L4PTF,SHR,TFC", "--batch", "8", "--steps", "72",
                                  "--check-passes", "16", "--check-phase", "2", "--check-every", "1",
                                  "--require-stress", "1"])
    assert (a.check_passes, a.check_phase, a.check_every) == (16, 2, 1)
    assert C.parse_args(base + fk + ["--arms", "WJ,WT,WF,WFP"]).arms == ["WJ", "WT", "WF", "WFP"]
    assert C.main(["--out", "x", "--workload", "w", "--arms", "L4PT", "--model", ""]) == 20     # no model: refused


def test_i3_cell_pure_parts():
    C = _cell()
    assert C.fork_store_slots("L4PTF", 3) == 129 and C.fork_store_slots("WF", 3) == 129
    assert C.fork_store_slots("TFC", 3) == C.fork_store_slots("TFCR", 3) == 193
    assert C.fork_store_slots("WFP", 3, plant_x=4) == 4 and C.fork_store_slots("L4PTF", 3, 200) == 200
    # the memory model of docs/NOTES_i3.md section 6 (bf16, H 2, D 128)
    assert (C.store_gb(216, 129), C.store_gb(318, 129), C.store_gb(216, 193), C.store_gb(318, 193)) == (
        1.8332, 2.6989, 2.7427, 4.0379)
    assert (C.cb0_gb(216), C.cb0_gb(318)) == (0.0142, 0.0209)
    assert (C.layer_rows_gb(216, 65), C.layer_rows_gb(318, 10)) == (1.8332, 1.5482)
    f = C.check_facts(16128, 3, 16, R)                                 # i3m: S0 = 252 x 64, phase 2, 16 passes
    assert f["past_edge_rows"] == 3 and f["f4_rows"] == 1 and f["last"] == 16128 + 3 + 64 - 1, f
    assert C.check_facts(16128, 1, 15, R)["past_edge_rows"] == 0       # CONTROL: I2's i2m schedule (15 passes) has none
    assert C.check_facts(16128, 1, 32, R)["past_edge_rows"] > 0        # i3ca / i3cb: 32 passes cross the edge
    assert C.check_facts(16128, 3, 16, R) == STF.teacher_facts(16128, 3, 16, R)
    ph = dict(resid=1.0, write=0.5, chain=0.25, union=0.25, fetch=2.0, attn=3.0, check=0.0, post=4.0)
    gr = C.fork_groups(ph)
    assert gr["attention"] == 3.0 and gr["chain"] == 0.25 and gr["dense_next"] == 4.0 and gr["total"] == 11.0
    ev = _evf(8)
    rd = dict(ev["rd"])
    res = dict(read=rd, fa_check=dict(F=dict(entries=[0], equal=[True]), RA=dict(entries=[0], equal=[True])))
    g = {n: ok for n, _, ok, _ in C.fork_gates("L4PTF", res, L, check=True, require_stress=True)}
    assert all(g.values()), g
    assert set(g) == {"L4PTF-RESID", "L4PTF-ATTN1", "L4PTF-FILL1", "L4PTF-PRESENT", "L4PTF-VIS", "L4PTF-VB",
                      "L4PTF-TAIL", "L4PTF-CHAIN", "L4PTF-IDX", "L4PTF-RA", "L4PTF-FSTRESS"}
    rec = C.fork_record(rd)
    assert rec["store_slots"] == 129 and rec["counters"]["vb_bad"] == 0 and rec["chunks"] == [R], rec
    n, must, ok, d = C.mem_record("L4PTF", dict(prefill_peak_GB=78.13), 82.57)   # 78.13 GiB = 83.891 GB
    assert (n, must, ok) == ("L4PTF-MEM", False, True) and "peak allocated 83.891 GB" in d, d
    assert C.mem_record("TFC", dict(prefill_peak_GB=79.0), 80.0)[2] is False       # 84.826 GB > 84
    assert C.mem_record("SH", "prefill + decode", 82.043)[2] is True
    sh = dict(rates=dict(steady_step_ms=10.0, steady_tok_s=800.0, effective_tok_s=700.0), steady=dict(mean_ms=10.0))
    v = C.verdicts(sh, {}, 216)
    assert v == [("L4PTF-RUN", "NOT_RUN", "the arm did not run")]


def test_i3_registration_state():
    C = _cell()
    if C.PRED_STAMP_I3 is None:                                         # the builder's state: every value None
        assert all(v is None for b in (216, 318) for v in C.PRED_I3_BY_B[b].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_I3)
    assert set(C.PRED_I3_BY_B) == {216, 318} and all(set(x) == set(C.PRED_KEYS) for x in C.PRED_I3_BY_B.values())
    src = (EFF / "r2_i3_cell.py").read_text()
    assert ("\nPRED_STAMP_I3 = None\n" in src) == (C.PRED_STAMP_I3 is None)    # the step script's grep sees the state


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_i3_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _clean_head():
    if str(ROOT) != STEP_ROOT:
        return None
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


def test_step_script_i3():
    src = (EFF / "r2_i3_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                       # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    assert "FORK_BLOB=%s" % FORK_BLOB in src and "NOSI=%s " % STEP_ROOT in src
    for bad in ("i2m", "wc1m", "i3", "i3x"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    registered = "\nPRED_STAMP_I3 = None\n" not in _git("show", "%s:benchmarks/Efficiency/r2_i3_cell.py" % sha).stdout
    for cell, pool, batch, steps, arms, passes, phase, check, stress, smin, mem, slot in (
            ("i3m", 10, 8, 72, "L4PT,L4PTU,L4PTF,SHR,TFC", 16, 2, 1, 1, 55, "100G", 32),
            ("i3a", 65, 216, 512, "L4PT,L4PTU,L4PTF", 32, 0, 0, 0, 100, "240G", 32),
            ("i3b", 10, 318, 512, "L4PT,L4PTU,L4PTF", 32, 0, 0, 0, 135, "340G", 32),
            ("i3ca", 65, 216, 129, "SHR,TFC", 32, 0, 0, 0, 70, "240G", 32),
            ("i3cb", 10, 318, 129, "SHR,TFC", 32, 0, 0, 0, 95, "340G", 32),
            ("i3g", 10, 100, 64, "WJ,WT,WF,WFP", 32, 0, 0, 0, 90, "200G", 4)):
        r = _step(cell, sha)
        if cell in ("i3a", "i3b") and not registered:                 # the timing cells STOP until the registration
            assert r.returncode == 1 and "PRED_STAMP_I3 is None" in r.stderr, (cell, r.returncode, r.stderr[-300:])
            continue
        if r.returncode != 0:
            pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
        line = ("cell %s: C %d (pool %d) B %d arms SH+%s steps %d K 3 X 193 fork-store 0 passes %d phase %d check %d "
                "stress %d slot %d STEP_MIN %d mem %s claim /mnt/beegfs/ojerbi/r2-runs/spec/sp_active_%s" % (
                    cell, 63 + pool, pool, batch, arms, steps, passes, phase, check, stress, slot, smin, mem, HOLDER))
        assert line in r.stdout, (cell, r.stdout[-600:])
        for want in ("NOSI_POOL_BLOCKS=%d" % pool, "--pool %d" % pool, "--batch %d" % batch, "--steps %d" % steps,
                     "--joint-attn fork", "--joint-extra 193", "--fork-store 0", "--prefill-hook nocopy",
                     "--check-passes %d" % passes, "--check-phase %d" % phase, "--check-every %d" % check,
                     "--require-stress %d" % stress, "--chk-slot-every %d" % slot, "--chk-plant-extra 4",
                     STEP_ROOT + "/benchmarks/Efficiency/r2_i3_cell.py", "--mem=%s" % mem, "--time=%d" % smin,
                     "i3_t1_%s_%s" % (HOLDER, cell)):
            assert want in r.stdout, (cell, want)
        a = _cell().parse_args(["--out", "x", "--workload", "w", "--batch", str(batch), "--steps", str(steps),
                                "--pool", str(pool), "--arms", arms, "--joint-attn", "fork", "--check-passes",
                                str(passes), "--check-phase", str(phase), "--check-every", str(check),
                                "--require-stress", str(stress)])
        assert a.arms == arms.split(",")                                # every cell's arguments parse


def test_notes_i3():
    doc = (ROOT / "docs/NOTES_i3.md").read_text()
    for word in ("test_i3.py", "test_i3_interp.py", "its own srun", "PRED_I3_BY_B", "PRED_STAMP_I3", "INFERENCE",
                 "MEASURED", "i3m", "i3a", "i3b", "i3ca", "i3cb", "i3g", "CB0", "CB1", "84 GB", "NoCopyPrefillHook",
                 "fork :1060-1063", "spec_tentative_union.py:197-205"):
        assert word in doc, word
