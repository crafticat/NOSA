"""CPU tests of WS-C (KS/analysis/plan/PLAN_L4P_faithful.md, KS = /mnt/beegfs/ojerbi/kstep-cpu-runs; Q2 after the fia6
A/B): the union joint pass of WS-B with ONE call of the shared-KV fork per layer (nosi/nosi/verify_rows.py mode "union"
with joint_attn "fork"; nosi/nosi/spec_round.py SpecOps(joint_attn="fork"); benchmarks/Efficiency/r2_kv_arena.py,
r2_wsc_cell.py, r2_wsc_step.sh). The world: tests_cpu/wsc_world.py over wsb_world (the SHIPPED engine from its own
source; every CUDA op a torch stand-in with the kernel's flat addresses, lesson L28). The fork is its torch stand-in
fake_fork here (the CPU world has 2 query heads per KV head: BLOCK_M 8, below the real launcher's 16);
tests_cpu/test_wsc_interp.py runs the REAL fork under the Triton interpreter in a child process (its own srun).
CPU stand-ins do not establish GPU dispatch or ordering (the check cells profile the kernels).

FAST is checked on FAST itself against SH (the serial ragged step on a second world of the same build), not against
the EXACT pass (the coordinator's binding correction, 2026-10-28).

GATE 0   test_gate0_files: the tree = 1a8ce68 (WS-B's registration) except the declared WS-C files; the fork copy =
         fia6's file byte for byte (an existing component, NO NEW KERNEL); no nosi module imports the fork or the arena.
         test_gate0_union_store_unchanged (B 1, B 8): mode union with the default joint_attn "store" gives the bits of
         the 1a8ce68 source (logits, engine and twin tables, the statistics). test_gate0_spec_round_store_unchanged
         (L4 union, L4P union): the round runner = the 1a8ce68 source, bit for bit.
ARENA    test_arena_rehome: one allocation; every engine tensor a CONTIGUOUS view with the old strides; the bits kept;
         the block ids; serial steps on a re-homed world = on a plain world, bitwise; refusals.
         test_prefill_hook: the shipped prefill re-homed layer by layer = the plain prefill, bitwise.
STAND-IN test_fake_fork_semantics: fake_fork = fork_ref64 (float64, order-free) within 1e-5; its tables = union_ref;
         a row alone = the row in the full launch, bitwise (block 0 first); the launcher's refusals.
VS SH    test_fork_vs_sh (B 1, B 8, B 4 long, B 8 natural): every row's selection SET = SH's (MUST); dropped 0, unserved
         0; VIS and VB 0 from the kernel's tables; G_block0; the "F" and "RA" probes bitwise; argmax = SH's, delta <=
         1e-4; one residency, ONE fork call and ONE store fill per layer and pass.
MATH     test_fork_attention_is_sh_math (B 8): every full row's output = SH's attention math in float64 (2e-5).
BYTES    test_fork_bytes (B 8, B 4 long): every (row, block) of the kernel's tables holds the block's host rows.
PLANTS   fork_union_rows (VB / VIS fail, the logits leave SH's, the byte check sees foreign blocks);
         fork_small_store (unserved and dropped > 0: PRESENT and VB fail); fork_probe_swap (IDX fails).
COUNT    test_one_residency_one_fork_call_per_layer; the WS-B store attention as the negative control.
ROLES    test_resident_rows_fork (L4XF's roles: K full rows, the draft row resident).
CHUNKS   test_fork_r5_chunks (R 5: 4 + 1 rows per request, two fork sequences).
OPS      test_fork_ops_refusals. SYNC test_no_host_sync_in_fork_pass.
RUNNER   test_l4xf_runner (B 1, B 8), test_l4pf_runner (B 8).
CELL     test_cell_arguments, test_cell_pure_parts, test_step_script_wc1, test_notes.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsc.py
"""
from __future__ import annotations

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
WSB_TIP = "1a8ce68d9d6852fecf45b916c9ef1e6a5a80e098"         # nosi-r2-wsb: the tree WS-C starts from
FIA6 = "94d5bb0ef62119717975f355b81b7996551adc8f"            # nosi-r2-fia6: the fork as fia6 ran it
FORK_FIA6_PATH = "nosi/nosi/fork_vllm_unified_attention.py"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-wsc"
HOLDER = "2190876"                     # a lead-owned uuid file that exists (HOLDERS/2190876.gpu_uuid)
DECLARED = ("nosi/nosi/verify_rows.py", "nosi/nosi/spec_round.py", "benchmarks/Efficiency/fork_vllm_unified_attention.py",
            "benchmarks/Efficiency/r2_kv_arena.py", "benchmarks/Efficiency/r2_wsc_cell.py",
            "benchmarks/Efficiency/r2_wsc_step.sh", "benchmarks/Efficiency/tests_cpu/wsc_world.py",
            "benchmarks/Efficiency/tests_cpu/test_wsc.py", "benchmarks/Efficiency/tests_cpu/test_wsc_interp.py",
            "benchmarks/Efficiency/tests_cpu/test_wsb.py", "docs/NOTES_wsc.md")


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


Z = _load("wsc_world")
X_, SW, V, W, RE, VR, SR, RW, KA = Z.X_, Z.SW, Z.V, Z.W, Z.RE, Z.VR, Z.SR, Z.RW, Z.KA
K, R, L = 3, 4, W.L
S1 = [4923]                                                      # 77 blocks; a block edge at 4927 (F4 row)
SF8 = [4800, 4857, 4923, 4860, 4795, 4811, 4830, 4891]          # 75-77 blocks; block edges at different rows
SF4 = SF8[:4]
SL4 = [6100, 6143, 6205, 6139]                                   # 95-97 blocks: more blocks than window + pool (72)
EQ8 = [4158] * 8                                                # equal lengths (natural ops: the shipped bits)
CASES = dict(B1=(S1, False, "exact"), B8=(SF8, False, "exact"), B4L=(SL4, False, "exact"), B8nat=(EQ8, True, "rowgroup"))
FILTER = SW.corrupt(4, 1, kinds=("draft",), phase=2)
TIMING = ("wall_s", "phases", "round_ms", "probe_wait_ms", "prelude_ms")


@functools.lru_cache(maxsize=None)
def _ev(case, groups=3, neg=None, draft_last=False, X=None, Kq=3):
    S_list, natural, gemm = CASES[case]
    return Z.run_fork_vs_sh(S_list, groups, K=Kq, X=X, neg=neg, natural=natural, gemm=gemm, draft_last=draft_last)


def _cell():
    if str(EFF) not in sys.path:
        sys.path.insert(0, str(EFF))
    import r2_wsc_cell
    return r2_wsc_cell


def _res_of(ev):
    """A fork check arm's result (r2_wsc_cell.check_gates_fork's input) from a CPU run."""
    return dict(stats=ev["stats"], errors=ev["errors"][1], fa_check=ev["vs"].fa_check,
                sel=dict(compared=ev["compared"], changed=len(ev["changed"]), by_layer_nonzero={}, symdiff_hist={},
                         first=ev["changed"][:5]))


def _gates(ev, **kw):
    return {n: ok for n, _, ok, _ in _cell().check_gates_fork("UCF", _res_of(ev), L=L, **kw)}


def _have(commit) -> bool:
    return X_.git("rev-parse", "-q", "--verify", commit + "^{commit}").returncode == 0


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


def _fork_ops(**kw):
    base = dict(K=K, mode="union", gemm="exact", linear=W.row_linear, attn=X_.fake_attention_u, joint_attn="fork",
                fork=Z.fake_fork)
    base.update(kw)
    return VR.VerifyOps(**base)


# ------------------------------------------------------------------ GATE 0
def test_gate0_files():
    if not _have(WSB_TIP):
        pytest.skip("no git or no commit %s in %s" % (WSB_TIP, ROOT))
    r = X_.git("diff", "--name-only", WSB_TIP, "--", ".", ":(exclude,glob)**/__pycache__/**")
    assert r.returncode == 0, r.stderr
    changed = [x for x in r.stdout.split("\n") if x]
    assert set(changed) <= set(DECLARED), sorted(set(changed) - set(DECLARED))
    r = X_.git("ls-files", "--others", "--exclude-standard", "--", "nosi", ":(exclude,glob)**/__pycache__/**")
    assert r.stdout.strip() == "", "untracked files in nosi/: %s" % r.stdout
    fork = EFF / "fork_vllm_unified_attention.py"
    if _have(FIA6):                                              # the fork copy = fia6's file, byte for byte
        old = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (FIA6, FORK_FIA6_PATH)], capture_output=True)
        assert old.returncode == 0 and old.stdout == fork.read_bytes()
        blob = X_.git("rev-parse", "%s:%s" % (FIA6, FORK_FIA6_PATH)).stdout.strip()
        assert blob == _cell().FORK_BLOB and ("FORK_BLOB=%s" % blob) in (EFF / "r2_wsc_step.sh").read_text()
    src = {p.name: p.read_text() for p in (ROOT / "nosi/nosi").glob("*.py")}
    for n in ("spec_round", "ragged_rollback", "state_snapshot", "spec_prefetch"):   # the older Gate 0 name scans
        assert n not in src["verify_rows.py"], n
    for n, t in src.items():                                     # the kernel and the arena: injected, not imported
        assert not re.search(r"(import|from)\s+[\w.]*(fork_vllm_unified_attention|r2_kv_arena)\b", t), n
    assert [n for n, t in src.items() if "_fork_rows" in t] == ["verify_rows.py"]


@pytest.mark.parametrize("S_list", [S1, SF8], ids=["B1", "B8"])
def test_gate0_union_store_unchanged(S_list, tmp_path):
    if not _have(WSB_TIP):
        pytest.skip("no git or no commit %s in %s" % (WSB_TIP, ROOT))
    OLD = X_.old_module("verify_rows", WSB_TIP, tmp_path, patch_verify=True)
    new, s_new, st_new = Z.joint_run_union(VR, S_list, 2)
    old, s_old, st_old = Z.joint_run_union(OLD, S_list, 2)
    for g, (a, b) in enumerate(zip(new, old)):
        assert torch.equal(W.bits(a), W.bits(b)), (g, V.first_diff(a, b))
    assert V.engines_equal(s_new.cache, s_old.cache, (s_new.st.twins, s_old.st.twins)) == []
    assert st_new == st_old


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


@pytest.mark.parametrize("level", ["L4", "L4P"], ids=["L4XU", "L4PU"])
def test_gate0_spec_round_store_unchanged(level, tmp_path):
    if not _have(WSB_TIP):
        pytest.skip("no git or no commit %s in %s" % (WSB_TIP, ROOT))
    OLD = X_.old_module("spec_round", WSB_TIP, tmp_path)
    new, old = [X_.run_spec_union(8, level=level, extra=16, draft_filter=FILTER, sr=m) for m in (None, OLD)]
    rn, ro = new["runner"], old["runner"]
    for x, y in ((rn.out, ro.out), (rn.fmax, ro.fmax), (rn.fsum, ro.fsum), (rn.lg_out, ro.lg_out)):
        assert torch.equal(W.bits(x), W.bits(y)), level
    dn = {k: v for k, v in new["rd"].items() if k not in TIMING}
    do = {k: v for k, v in old["rd"].items() if k not in TIMING}
    assert dn == do, (level, [k for k in dn if dn[k] != do.get(k)])
    assert _same_records(new["records"], old["records"]) == [], level
    assert RW.diff(RW.state(rn.rs), RW.state(ro.rs)) == [], level
    assert new["facts"]["rejects"] > 0 and new["facts"]["full"] > 0, new["facts"]          # both paths ran (L34)


# ------------------------------------------------------------------ ARENA
def test_arena_rehome():
    S_list, B = SF4, 4
    model = W.make_model(0)
    rows = W.prompt_rows(1, S_list)
    worlds = [RE.build(model, W.new_cache(0), S_list, 8, V.exact_ragged_ops(), lambda b, l: rows[(b, l)], pin=False)
              for _ in range(2)]
    for s in worlds:
        W.poison_built(s, S_list)
    A, Bw = worlds
    eng0 = Bw.cache.layers[0].cache_engine
    ns = int(eng0._k_gpu.shape[1]) // W.BS
    assert ns == W.TOPK + W.P                                    # precondition: the pooled engine (window + pool)
    strides = lambda c: [tuple(getattr(lay.cache_engine, n).stride()) for lay in c.layers for n in KA.ENGINE_ROWS]  # noqa
    before = strides(Bw.cache)
    X = 7
    arena = Z.arena_for(Bw.cache, B, X)
    assert KA.check(Bw.cache, arena) == [] and strides(Bw.cache) == before              # the same strides (L28)
    for l, lay in enumerate(Bw.cache.layers):
        e = lay.cache_engine
        for name, t in zip(KA.ENGINE_ROWS, arena.layer_views(l)):
            x = getattr(e, name)
            assert x.is_contiguous() and x.data_ptr() == t.data_ptr(), (l, name)
        assert e._k_gpu_att.data_ptr() == e._k_gpu.data_ptr() and e._k_gpu_att.shape[1] == W.TOPK * W.BS
        b, s_ = 1, W.TOPK + 2                                    # a pool slot: arena block id l B NS + b NS + s
        blk = arena.layer_base(l) + b * ns + s_
        assert torch.equal(W.bits(arena.k[blk]), W.bits(e._k_gpu[b, s_ * W.BS:(s_ + 1) * W.BS]))
        assert torch.equal(W.bits(arena.bias[blk]), W.bits(e._kv_bias_gpu[b, s_ * W.BS:(s_ + 1) * W.BS]))
    assert V.engines_equal(A.cache, Bw.cache, (A.st.twins, Bw.st.twins)) == []        # every engine bit kept
    assert arena.store_base == W.L * B * ns and tuple(arena.store_k.shape) == (B, X * W.BS, W.HKV, W.D)
    assert bool((arena.store_k == 0).all()) and bool((arena.store_bias == 0).all())
    assert arena.store_k.data_ptr() == arena.k[arena.store_base].data_ptr()
    for t in range(3):                                           # the shipped kernels see the same tensors
        pos = (torch.tensor(S_list) + t).view(B, 1).to(torch.int32)
        ids = torch.full((B, 1), t + 1, dtype=torch.int64)
        la = RE.decode_step(model, A, ids, pos)
        lb = RE.decode_step(model, Bw, ids, pos)
        assert torch.equal(W.bits(la), W.bits(lb)), t
    assert V.engines_equal(A.cache, Bw.cache, (A.st.twins, Bw.st.twins)) == []
    KA.rehome(Bw.cache, arena)                                   # idempotent
    assert KA.check(Bw.cache, arena) == []
    with pytest.raises(KA.ArenaError):
        KA.rehome(Bw.cache, KA.KVArena(W.L, B + 1, ns, W.HKV, W.D, X, W.DT, "cpu"))
    with pytest.raises(KA.ArenaError):
        KA.attach(A, arena)                                      # A's engines are not the arena's views
    with pytest.raises(KA.ArenaError):
        KA.KVArena(W.L, B, ns, W.HKV, W.D, 1, W.DT, "cpu")      # X >= 2
    assert KA.store_slots(64, 3) == 129 and KA.store_slots(64, 4) == 193 and KA.store_slots(64, 1) == 1


def test_prefill_hook():
    B, S = 2, 4200
    rows = W.prompt_rows(1, [S] * B)
    c1, c2 = W.new_cache(40), W.new_cache(40)
    W.shipped_prefill(c1, rows, S, B)
    arena = KA.for_engine(c2.layers[0].cache_engine, W.L, B, 5)
    with KA.PrefillHook(c2, arena) as hook:
        W.shipped_prefill(c2, rows, S, B)
    assert hook.bound_at == list(range(W.L)) and KA.check(c2, arena) == []
    assert all("prefill_update" not in lay.cache_engine.__dict__ for lay in c2.layers)      # the wrappers are gone
    W.poison_shipped(c1, S)
    W.poison_shipped(c2, S)
    assert V.engines_equal(c1, c2) == []                         # the re-homed prefill = the plain one, bitwise


# ------------------------------------------------------------------ STAND-IN
def _case(seed, nseq, rows_of, H=2, Hq=4, D=8, nblk=40, P=24):
    """Synthetic fork inputs: nseq sequences of rows_of[s] rows; per (row, head) a sorted subset of ids 0 .. 31 that
    holds block 0, a physical block per id (a permutation; entry 0, slots_per_entry 1), last_len in 1 .. 64 (one row
    0: an F4 row)."""
    g = torch.Generator().manual_seed(seed)
    E = sum(rows_of)
    q = torch.randn(E, Hq, D, generator=g)
    k = torch.randn(nblk, 64, H, D, generator=g)
    v = torch.randn(nblk, 64, H, D, generator=g)
    bias = torch.randn(nblk, 64, H, generator=g) * 0.1
    key = torch.full((E, H, P), 2 ** 31 - 1, dtype=torch.int32)
    slot = torch.zeros((E, H, P), dtype=torch.int32)
    npg = torch.zeros((E, H), dtype=torch.int32)
    e = 0
    for s, n in enumerate(rows_of):
        perm = torch.randperm(nblk, generator=g)
        for _ in range(n):
            for h in range(H):
                m = int(torch.randint(2, P + 1, (1,), generator=g))
                ids = torch.cat([torch.zeros(1, dtype=torch.int64), 1 + torch.randperm(31, generator=g)[:m - 1]])
                ids = ids.sort().values
                key[e, h, :m] = ids.to(torch.int32)
                slot[e, h, :m] = perm[ids].to(torch.int32)
                npg[e, h] = m
            e += 1
    last = torch.randint(1, 65, (E,), generator=g).to(torch.int32)
    last[0] = 0
    cu = torch.tensor([0] + [sum(rows_of[:i + 1]) for i in range(len(rows_of))], dtype=torch.int32)
    build = dict(page_key=key, page_table=slot, num_pages=npg, last_len=last, seq_idx=torch.zeros(E, dtype=torch.int32),
                 slots_per_entry=1)
    return q, k, v, bias, build, cu


def _launch(fork, q, k, v, bias, build, cu, width=128):
    n = cu.numel() - 1
    H = k.shape[2]
    out = torch.zeros_like(q)
    tab = torch.zeros((H, n, width), dtype=torch.int32)
    ext = torch.zeros_like(tab)
    ln = torch.zeros((H, n), dtype=torch.int32)
    qmax = int((cu[1:] - cu[:-1]).max())
    fork(q, k, v, out, cu, qmax, torch.zeros(n, dtype=torch.int32), 64, q.shape[-1] ** -0.5, True, (-1, -1), tab, 0,
         None, None, None, union_ext=ext, union_len=ln, cis_bias=bias, build_union=build, rows_per_q_block=4,
         num_warps=4, num_stages=3, tile_size=64)
    return out, tab, ext, ln


def test_fake_fork_semantics():
    rows_of = [4, 1, 3, 4]
    q, k, v, bias, build, cu = _case(3, len(rows_of), rows_of)
    out, tab, ext, ln = _launch(Z.fake_fork, q, k, v, bias, build, cu)
    ref = Z.fork_ref64(q, k, v, bias, build, cu, q.shape[-1] ** -0.5)
    d = float(((out.double() - ref).abs() / (1 + ref.abs())).max())
    assert d <= 1e-5, d
    assert bool((out[0] != 0).any()) and bool(torch.isfinite(out).all())       # the F4 row still sees its blocks
    t2, x2, l2 = Z.union_ref(build, cu.tolist(), 2, 128)
    assert torch.equal(ln.to(torch.int64), l2)
    live = torch.arange(128).view(1, 1, -1) < l2.unsqueeze(-1)
    assert torch.equal(tab.to(torch.int64)[live], t2[live]) and torch.equal(ext.to(torch.int64)[live], x2[live])
    assert int(l2.max()) > int(build["num_pages"].max())                      # precondition: unions > one row
    for e in range(q.shape[0]):                                  # a row alone = its row in the full launch, bitwise
        b1 = {kk: (t[e:e + 1].contiguous() if torch.is_tensor(t) else t) for kk, t in build.items()}
        o1, *_ = _launch(Z.fake_fork, q[e:e + 1].contiguous(), k, v, bias, b1, torch.tensor([0, 1], dtype=torch.int32))
        assert torch.equal(W.bits(o1[0]), W.bits(out[e])), e
    bad = dict(build, page_key=build["page_key"].clone())        # the precondition: strictly ascending ids per row
    bad["page_key"][1, 0, :2] = torch.tensor([5, 3], dtype=torch.int32)
    with pytest.raises(AssertionError):
        _launch(Z.fake_fork, q, k, v, bias, bad, cu)
    with pytest.raises(AssertionError):
        _launch(Z.fake_fork, q, k, v, bias, build, cu, width=64)                 # W < BLOCK_Q x P_PAD
    with pytest.raises(AssertionError):
        _launch(Z.make_fake_fork(16), q, k, v, bias, build, cu)                  # BLOCK_M 8 < 16: the real launcher's


# ------------------------------------------------------------------ VS SH
@pytest.mark.parametrize("case", ["B1", "B8", "B4L", "B8nat"])
def test_fork_vs_sh(case):
    ev = _ev(case)
    st = ev["stats"]
    S_list = CASES[case][0]
    B = len(S_list)
    print("%s: store reads %d, pool reads %d, listed max %d, past-edge %d, F4 %d, loaded %d, visible %d" % (
        case, st["store_reads"], st["pool_reads"], st["extras_max"], st["past_edge"], st["f4_rows"],
        st["fork_loaded"], st["visible_blocks"]))
    assert ev["rehome_equal"]                                                # the re-home kept every engine bit
    assert ev["compared"] == 3 * L * W.HKV * B * R
    assert ev["fills"] and st["past_edge"] > 0 and st["f4_rows"] > 0, st    # preconditions (L34): edge, F4 rows
    if case == "B4L":
        assert st["store_reads"] > 0 and st["extras_host"] > 0, st          # rows read listed host blocks
    assert ev["changed"] == [], ev["changed"][:8]                            # THE CLAIM: SH's selection, every row
    assert st["dropped"] == 0 and st["unserved"] == 0, st                    # every selected block present
    assert st["vis_bad"] == 0 and st["vb_bad"] == 0, st                      # each row sees its own selection only
    assert st["vb_rows"] == 3 * L * W.HKV * B * R and st["block0_bad"] == 0, st
    assert ev["argmax_bad"] == [] and ev["max_delta"] <= 1e-4, (ev["argmax_bad"][:5], ev["max_delta"])
    assert ev["diff_per_pass"] == [L] * 3 and st["calls"]["resid"] == L * 3, (ev["diff_per_pass"], st["calls"])
    assert st["calls"]["attn"] == L * 3 and st["calls"]["fills"] == L * 3, st["calls"]   # ONE fork call, ONE fill
    picks = len({0, B // 2, B - 1})
    assert ev["fork_per_pass"] == [L + 2 * picks, L, L], ev["fork_per_pass"]          # + the probes (layer 0, once)
    assert ev["errors"] == ((0, [0] * L), (0, [0] * L)), ev["errors"]
    fa = ev["vs"].fa_check
    assert fa["F"]["entries"] and all(fa["F"]["equal"]), fa                 # IDX: requests alone, bitwise
    assert fa["RA"]["entries"] and all(fa["RA"]["equal"]), fa               # a row alone, bitwise
    assert all(_gates(ev).values()), _gates(ev)


def test_fork_attention_is_sh_math():
    ev = _ev("B8")
    recs = [r for r in ev["vs"].log if r["kind"] == "fork"]
    past = any(bool((~r["in_tail"]).any()) for r in recs)
    f4 = any(bool(((r["p_rows"] % W.BS == W.BS - 1) & r["in_tail"]).any()) for r in recs)
    assert past and f4, (past, f4)                                           # preconditions (L34)
    worst, n = 0.0, 0
    for rec in recs:
        ref, done = X_.sh_attention_ref(rec, R)
        got = rec["out"].double()
        d = ((got - ref).abs() / (1 + ref.abs()))[done]
        n += int(done.sum())
        worst = max(worst, float(d.max()))
    assert n > 0 and worst <= 2e-5, (n, worst)


@pytest.mark.parametrize("case", ["B8", "B4L"])
def test_fork_bytes(case):
    ev = _ev(case)
    u = Z.fork_bytes(ev["vs"].log, R)
    assert u["checked"] > 0, u
    assert u["bad_contents"] == [] and u["foreign"] == [] and u["cross_request"] == [], (
        u["bad_contents"][:5], u["foreign"][:5], u["cross_request"][:5])


# ------------------------------------------------------------------ PLANTS
def test_fork_union_rows_plant_caught():
    clean = _ev("B8")
    assert clean["stats"]["union_max"] > W.TOPK                              # precondition: unions larger than rows
    ev = _ev("B8", neg="fork_union_rows")
    st = ev["stats"]
    assert st["vb_bad"] > 0 and st["vis_bad"] > 0, st
    assert ev["max_delta"] > 1e-4 or ev["argmax_bad"] or ev["changed"], ev["max_delta"]   # the logits leave SH's
    g = _gates(ev)
    assert g["UCF-VB"] is False and g["UCF-VIS"] is False, g
    assert Z.fork_bytes(ev["vs"].log, R)["foreign"], "the byte check sees the foreign blocks"
    assert all(_gates(clean).values())


def test_fork_small_store_plant_caught():
    ev = _ev("B4L", groups=3, neg="fork_small_store", X=2)
    st = ev["stats"]
    assert st["unserved"] > 0, st                                            # precondition: more host blocks than X - 1
    assert st["dropped"] > 0 and st["vb_bad"] > 0, st
    g = _gates(ev)
    assert g["UCF-PRESENT"] is False and g["UCF-VB"] is False, g
    clean = _gates(_ev("B4L"))
    assert clean["UCF-PRESENT"] is True and clean["UCF-VB"] is True, clean
    model, s, _ = _built(SF4, R)
    KA.attach(s, Z.arena_for(s.cache, 4, 2))
    with pytest.raises(VR.VerifyError):                          # without the plant the small store is refused
        VR.attach(model, s, _fork_ops())


def test_fork_probe_swap_plant_caught():
    ev = _ev("B8", neg="fork_probe_swap")
    fa = ev["vs"].fa_check
    assert not all(fa["F"]["equal"]) and all(fa["RA"]["equal"]), fa         # the probe is not blind
    g = _gates(ev)
    assert g["UCF-IDX"] is False and g["UCF-RA"] is True, g
    clean = _ev("B8")
    assert ev["changed"] == clean["changed"] and ev["max_delta"] == clean["max_delta"]   # the plant is check-only


# ------------------------------------------------------------------ COUNT
def test_one_residency_one_fork_call_per_layer():
    S_list, B = SF4, 4
    pos = (torch.tensor(S_list).view(B, 1) + torch.arange(R).view(1, R)).to(torch.int32)
    ids = torch.zeros((B, R), dtype=torch.int64)
    model, s, calls = _built(S_list, R)
    KA.attach(s, Z.arena_for(s.cache, B, KA.store_slots(W.TOPK, R)))
    vs = VR.attach(model, s, _fork_ops())
    f0 = len(Z.FORK_CALLS)
    VR.joint_pass(model, vs, ids, pos)
    assert len(calls) == L and vs.calls["resid"] == L, (len(calls), vs.calls)           # one residency per layer
    assert len(Z.FORK_CALLS) - f0 == L and vs.calls["attn"] == L and vs.calls["fills"] == L, vs.calls
    model, s, calls = _built(S_list, R)                          # NEGATIVE CONTROL: WS-B's store attention
    vs = VR.attach(model, s, VR.VerifyOps(K=K, mode="union", gemm="exact", linear=W.row_linear,
                                          attn=X_.fake_attention_u))
    VR.joint_pass(model, vs, ids, pos)
    assert vs.calls["attn"] == L * (1 + vs.C) and vs.C > 1, (vs.calls, vs.C)


# ------------------------------------------------------------------ ROLES
def test_resident_rows_fork():
    ev = _ev("B8", groups=1, draft_last=True)
    st = ev["stats"]
    B = len(SF8)
    assert ev["vs"].X == KA.store_slots(W.TOPK, K) == 129                   # L4XF's bound: K full rows
    assert st["rows_full"] == L * B * K and st["rows_res"] == L * B, st     # the roles of L4XF
    assert ev["changed"] == [] and ev["argmax_bad"] == [] and ev["max_delta"] <= 1e-4, ev["changed"][:5]
    assert st["dropped"] == 0 and st["vis_bad"] == 0 and st["vb_bad"] == 0, st
    assert st["row_skipped"][:K] == [0] * K, st["row_skipped"]
    u = Z.fork_bytes(ev["vs"].log, R)
    assert u["foreign"] == [] and u["bad_contents"] == [], u


# ------------------------------------------------------------------ CHUNKS
def test_fork_r5_chunks():
    ev = _ev("B8", groups=2, Kq=4)
    vs, st = ev["vs"], ev["stats"]
    B = len(SF8)
    assert vs.fork_sizes == [4, 1] and vs.fork_ns == 2 * B, (vs.fork_sizes, vs.fork_ns)
    assert vs.fork_cu.tolist()[:5] == [0, 4, 5, 9, 10] and vs.X == KA.store_slots(W.TOPK, 5)
    assert ev["changed"] == [] and ev["argmax_bad"] == [] and ev["max_delta"] <= 1e-4, ev["changed"][:5]
    assert st["dropped"] == 0 and st["vis_bad"] == 0 and st["vb_bad"] == 0 and st["vb_rows"] > 0, st
    assert st["calls"]["attn"] == L * 2, st["calls"]                         # still ONE fork call per layer
    fa = vs.fa_check
    assert all(fa["F"]["equal"]) and all(fa["RA"]["equal"]), fa


# ------------------------------------------------------------------ OPS
def test_fork_ops_refusals():
    f = Z.fake_fork
    for kw in (dict(mode="fast", joint_attn="fork", fork=f), dict(mode="union", joint_attn="fork"),
               dict(mode="union", joint_attn="fork", fork=f, spill=2),
               dict(mode="union", joint_attn="fork", fork=f, neg="drop_spill"),
               dict(mode="union", joint_attn="fork", fork=f, neg="union_attention"),
               dict(mode="union", neg="fork_union_rows"), dict(mode="union", fork=f), dict(mode="union", full_max=2),
               dict(mode="union", joint_attn="bad"), dict(mode="union", joint_attn="fork", fork=f, full_max=9),
               dict(mode="union", joint_attn="fork", fork=1)):
        with pytest.raises(VR.VerifyError):
            VR.VerifyOps(**kw)
    for kw in (dict(level="L4", mode="exact", joint_attn="fork", fork=f), dict(level="L4", mode="union", joint_attn="fork"),
               dict(level="L4", mode="union", fork=f), dict(level="L4", mode="union", joint_attn="x")):
        with pytest.raises(SR.SpecError):
            SR.SpecOps(K=3, **kw)
    SR.SpecOps(level="L4", K=3, mode="union", joint_attn="fork", fork=f)               # accepted
    SR.SpecOps(level="L4P", K=3, mode="union", joint_attn="fork", fork=f)
    assert SR.SpecOps(level="L4", K=3, mode="union").describe().get("joint_attn") is None   # store: WS-B's record
    model, s, _ = _built(SF4, R)
    with pytest.raises(VR.VerifyError):                          # no arena on the state
        VR.attach(model, s, _fork_ops())
    s.kv_arena = KA.for_engine(s.cache.layers[0].cache_engine, W.L, 4, 193)   # an arena the engines do not use
    with pytest.raises(VR.VerifyError):
        VR.attach(model, s, _fork_ops())
    model, s, _ = _built(SF4, R)
    KA.attach(s, Z.arena_for(s.cache, 4, 129))                   # below the bound of 4 full rows (193)
    with pytest.raises(VR.VerifyError):
        VR.attach(model, s, _fork_ops())
    vs = VR.attach(model, s, _fork_ops(draft_last=True))         # 3 full rows: 129 is the bound
    assert (vs.X, vs.C, vs.cap, vs.bound) == (129, 1, 128, 128)
    assert vs.meta["ops"]["joint_attn"] == "fork" and vs.meta["fork_launch"] == VR.FORK_LAUNCH
    assert VR.FORK_LAUNCH == dict(num_warps=4, num_stages=3, tile_size=64)    # FV6 = FVW4S3T64B16 (SCORING_fia6.md)


# ------------------------------------------------------------------ SYNC
def test_no_host_sync_in_fork_pass():
    S_list, B = SF4, 4
    for check in (False, True):
        model, s, _ = _built(S_list, 2 * R)
        KA.attach(s, Z.arena_for(s.cache, B, KA.store_slots(W.TOPK, R)))
        vs = VR.attach(model, s, _fork_ops(check=check))
        with RW.SyncTrap(VR.__file__) as trap:
            for g in range(2):
                pos = (torch.tensor(S_list).view(B, 1) + g * R + torch.arange(R).view(1, R)).to(torch.int32)
                VR.joint_pass(model, vs, torch.zeros((B, R), dtype=torch.int64), pos)
        assert vs.passes == 2
        assert {(h[0], h[2]) for h in trap.hits} <= {("joint_pass", "__bool__")}, (check, trap.hits[:5])


# ------------------------------------------------------------------ RUNNER
def _agreement(ev):
    cnt = torch.tensor(ev["rd"]["count"]).clamp(max=ev["N"])
    bad = {b: i for b, i, _, _ in ev["bad_tokens"]}
    good = sum(bad.get(b, int(cnt[b])) for b in range(ev["B"]))
    return good / max(1, int(cnt.sum()))


@pytest.mark.parametrize("B", [1, 8], ids=["B1", "B8"])
def test_l4xf_runner(B):
    ev = Z.run_spec_fork(B, level="L4", draft_filter=FILTER)
    rd, f = ev["rd"], ev["facts"]
    v = rd["verify"]
    assert f["rejects"] > 0 and f["full"] > 0 and f["edge_crossed"], f    # both paths ran, a block edge (L34)
    assert ev["resid"] == [] and ev["host"] == [], (ev["resid"][:2], ev["host"][:2])   # residency, host rows
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    n = v["passes"]
    assert n == rd["rounds"] and v["calls"]["resid"] == L * n and v["calls"]["attn"] == L * n, (n, v["calls"])
    assert v["calls"]["fills"] == L * n and v["unserved"] == 0 and v["union_samples"] > 0, v
    assert v["joint_attn"] == "fork" and v["store_slots"] == KA.store_slots(W.TOPK, K) and v["chunks"] == [4], v
    a = _agreement(ev)
    print("L4XF B %d: committed tokens equal to the shipped decode: %.4f; first differences %s" % (
        B, a, ev["bad_tokens"][:4]))
    assert a >= 0.9, (a, ev["bad_tokens"][:5])


def test_l4pf_runner():
    box = {}

    def mk(rs):
        box["rec"] = SW.Recorder()
        return box["rec"]
    ev = Z.run_spec_fork(8, level="L4P", draft_filter=FILTER, make_prefetch=mk)
    rd, f = ev["rd"], ev["facts"]
    v = rd["verify"]
    assert f["rejects"] > 0 and f["full"] > 0, f
    assert ev["resid"] == [] and ev["host"] == [] and ev["host_c"] == [], (ev["resid"][:2], ev["host"][:2])
    assert rd["errors"] == (0, [0] * L), rd["errors"]
    assert v["passes"] == rd["rounds"] and v["calls"]["attn"] == L * v["passes"], (v["passes"], v["calls"])
    assert v["rows_res"] > 0 and v["rows_full"] > 0 and v["chunks"] == [4, 1] and v["unserved"] == 0, v
    rec = box["rec"]
    joint = rec.drafts[K::K]
    assert joint and any(bool((m >= 0).any()) for x in joint for m in x)    # the rung-L5 seam gets the draft misses
    a = _agreement(ev)
    print("L4P fork B 8: committed tokens equal to the shipped decode: %.4f" % a)
    assert a >= 0.9, (a, ev["bad_tokens"][:5])


# ------------------------------------------------------------------ CELL
def test_cell_arguments():
    C = _cell()
    base = ["--out", "x", "--workload", "w"]
    a = C.parse_args(base + ["--joint-residency", "union", "--joint-attn", "fork", "--arms", "L4X,L4XU,L4XF"])
    assert (a.arms, a.batch, a.steps, a.pool, a.spec, a.joint_extra, a.fork_store, a.check_steps, a.joint_attn,
            a.spec_twin, a.spec_body, a.probe_every) == (["L4X", "L4XU", "L4XF"], 216, 512, 65, 3, 48, 0, 128, "fork",
                                                         "fused", "fused", 16)
    d = C.parse_args(base + ["--arms", "L4X"])
    assert d.joint_residency == "serial" and d.joint_attn == "store"           # the defaults: unchanged
    u = ["--joint-residency", "union"]
    fk = u + ["--joint-attn", "fork"]
    for bad in ([], u + ["--arms", "L4XF"], fk + ["--arms", "L4XU"], ["--joint-attn", "fork", "--arms", "L4XF"],
                fk + ["--arms", "UCF"], fk + ["--arms", "SHR,UCF", "--check-steps", "130"],
                fk + ["--arms", "SHR,UCF", "--steps", "64"], fk + ["--arms", "L4XF", "--spec", "4"],
                fk + ["--arms", "L4XF", "--fork-store", "100"], fk + ["--arms", "L4XF,L4XF"],
                fk + ["--arms", "L4XF", "--pool", "66"]):
        with pytest.raises(SystemExit):
            C.parse_args(base + bad)
    assert C.parse_args(base + fk + ["--arms", "SHR,UCF,UCFR"]).arms == ["SHR", "UCF", "UCFR"]
    assert C.parse_args(base + fk + ["--arms", "L4XF", "--fork-store", "200"]).fork_store == 200


def test_cell_pure_parts(monkeypatch):
    C = _cell()
    assert C.cap_key(65) == "C128" and C.cap_key(10) == "C73"
    assert C.fork_store_slots("L4XF", 3) == 129 and C.fork_store_slots("UCF", 3) == 193
    assert C.fork_store_slots("UCFR", 3, 200) == 200
    # the store memory (bf16, H 2, D 128): B x X x 64 x 2 x 257 x 2 B
    assert (C.store_gb(216, 129), C.store_gb(318, 129), C.store_gb(216, 193), C.store_gb(318, 193)) == (
        1.8332, 2.6989, 2.7427, 4.0379)
    ev = _ev("B8")
    g = _gates(ev)
    assert all(g.values()), g
    assert set(g) == {"UCF-ERR", "UCF-SEL", "UCF-PRESENT", "UCF-VIS", "UCF-VB", "UCF-RESID", "UCF-ATTN1", "UCF-FILL1",
                      "UCF-IDX", "UCF-RA"}
    bad = _res_of(ev)
    bad["sel"] = dict(bad["sel"], changed=1)
    assert {n: ok for n, _, ok, _ in C.check_gates_fork("UCF", bad, L=L)}["UCF-SEL"] is False
    assert "UCF-STRESS" in _gates(ev, require_stress=True)
    rd = dict(verify=dict(calls=dict(resid=2 * L, attn=2 * L, fills=2 * L), passes=2))
    assert all(x[2] for x in C.timing_fork_gates("L4XF", rd, L=L))
    rd["verify"]["calls"]["attn"] = 2 * L * 6                     # WS-B's window call + 5 store calls per layer
    assert {x[0]: x[2] for x in C.timing_fork_gates("L4XF", rd, L=L)}["L4XF-ATTN1"] is False
    for key in ("C128", "C73"):                                   # the builder's state: unregistered
        monkeypatch.setitem(C.PRED_WSC, key, dict(joint_sh_steps=None, joint_saving_ms=None, ratio_vs_l4xu=None,
                                                  ratio_vs_sh=None))
    sh = dict(rates=dict(steady_step_ms=64.0))
    sp = dict(L4XF=dict(phases_median=dict(joint=150.0), vs_sh=dict(median_tok_s=0.34), rates=dict(median_tok_s=1150.0)),
              L4XU=dict(phases_median=dict(joint=184.0), rates=dict(median_tok_s=1080.0)))
    v = {n: (x, d) for n, x, d in C.verdicts(sh, sp, 65)}
    assert {n: x for n, (x, _) in v.items()} == {"JOINT-SH-STEPS": "REPORT", "JOINT-SAVING-MS": "REPORT",
                                                  "RATIO-VS-L4XU": "REPORT", "RATIO-VS-SH": "REPORT",
                                                  "PLAN-FALSIFIER": "REPORT", "VS-STRONGEST": "REPORT"}
    assert "not fired" in v["PLAN-FALSIFIER"][1]                  # 184 - 150 = 34 ms >= 15 ms
    monkeypatch.setitem(C.PRED_WSC, "C128", dict(joint_sh_steps=(2.0, 2.5), joint_saving_ms=(40.0, 60.0),
                                                 ratio_vs_l4xu=(1.0, 1.1), ratio_vs_sh=(0.3, 0.4)))
    v = {n: x for n, x, _ in C.verdicts(sh, sp, 65)}
    assert (v["JOINT-SH-STEPS"], v["JOINT-SAVING-MS"], v["RATIO-VS-L4XU"], v["RATIO-VS-SH"]) == (
        "CONFIRMED", "REFUTED", "CONFIRMED", "CONFIRMED")
    assert C.verdicts(sh, {}, 65) == [("L4XF-RUN", "NOT_RUN", "the arm did not run")]
    cmp_ = C.attention_compare(dict(joint_phases=dict(groups_median=dict(attention=70.0), layer_ms_median=5.0)),
                               dict(joint_phases=dict(groups_median=dict(attention=40.0), layer_ms_median=4.0)))
    assert cmp_["groups_L4XU_L4XF"]["attention"] == (70.0, 40.0) and cmp_["layer_ms_median"] == (5.0, 4.0)


def test_pred_wsc_registration_state():
    C = _cell()
    if C.PRED_STAMP_WSC is None:                                  # the builder's state: every value None
        assert all(v is None for key in ("C128", "C73") for v in C.PRED_WSC[key].values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", C.PRED_STAMP_WSC)
    assert set(C.PRED_WSC["C128"]) == set(C.PRED_WSC["C73"]) == {"joint_sh_steps", "joint_saving_ms", "ratio_vs_l4xu",
                                                                 "ratio_vs_sh"}


def _step(cell, sha="0" * 40):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    e["DRYRUN"] = "1"
    return subprocess.run(["bash", str(EFF / "r2_wsc_step.sh"), "--nosi", sha, HOLDER, "t1", cell], env=e,
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


def test_step_script_wc1():
    src = (EFF / "r2_wsc_step.sh").read_text()
    assert src.rstrip().endswith('main "$@"; exit $?') and "\nmain() {\n" in src
    for word in ("python", "perl", "awk", "sed"):                  # no interpreter word outside the cell launch
        lines = [x for x in src.splitlines() if re.search(r"\b%s\b" % word, x) and "/venv/nosa/bin/python" not in x
                 and not x.lstrip().startswith("#")]
        assert lines == [], (word, lines[:3])
    for bad in ("wc2", "wb1m", "sp5a"):
        r = _step(bad)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    r = _step("wc1m", sha)
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
    line = ("cell wc1m: C 73 (pool 10) B 8 arms SH+L4X,L4XU,L4XF,SHR,UCF steps 64 K 3 X 48 fork-store 0 check 64 "
            "stress 1 STEP_MIN 40 mem 100G")
    assert line in r.stdout, r.stdout[-600:]
    for want in ("NOSI_POOL_BLOCKS=10", "NOSI_ATTN_SPLITS=4", "--pool 10", "--batch 8", "--steps 64",
                 "--joint-residency union", "--joint-attn fork", "--fork-store 0", "--check-steps 64",
                 "--require-stress 1", "--mem=100G", "--time=40", "r2_wsc_cell.py", "wc_t1_%s_wc1m" % HOLDER):
        assert want in r.stdout, want
    r = _step("wc1ca", sha)
    assert r.returncode == 0 and "cell wc1ca: C 128 (pool 65) B 216 arms SH+SHR,UCF steps 128" in r.stdout, r.stdout
    pred = _cell().PRED_STAMP_WSC
    for cell in ("wc1a", "wc1b"):
        r = _step(cell, sha)
        if pred is None:
            assert r.returncode == 1 and "PRED_STAMP_WSC is None" in r.stderr, (cell, r.stderr[-300:])
        else:
            assert r.returncode == 0 and "cell %s:" % cell in r.stdout, (cell, r.stdout[-300:])


def test_notes():
    doc = (ROOT / "docs/NOTES_wsc.md").read_text()
    for word in ("test_wsc.py", "test_wsc_interp.py", "its own srun", "PRED_WSC", "PRED_STAMP_WSC", "INFERENCE",
                 "wc1m", "wc1a", "wc1b", "wc1ca", "wc1cb", "FVW4S3T64B16", "C = 1"):
        assert word in doc, word
