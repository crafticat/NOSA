"""CPU tests of the INTEGRATION branch r2-integ (docs/R2_INTEG_PLAN.md): every gap-closing build at once, each an opt-in
flag with its default unchanged. THE STACK's six components and their OFF / ON values (r2_tick_sg5.STACK_OFF / ON):

  twin     TickEngine(twin="torch" | "fused")              (r2_twin_fused, p6)
  rebuild  TickEngine(rebuild="per_call" | "batched")      (p6)
  resolve  OwnResidency(resolve="torch" | "fused")          (r2_fusedres, fusedres2c 740ea1d)
  layout   per-stream store | the shared X pool             (r2_xpool, feaa7d6; X_pool = X)
  attn     piece 3 | SharedKV split, union fused             (bi_paged_attention_shared, sharedkv4 1c5ff8a; the plan's
                                                              page keys: S-C / B2 want_keys)
  warmup   GilfreeTransport(warm_up=False | True)          (gfw1f, 000870e)

ST1  THE FULL STACK EQUALS THE DEFAULT STACK BITWISE on a teacher-forced round (forced_schedule) of b2 at the
     PRESSURE point (C 8, X 40, R_ver 4: the LRU goes over C, slots are released and taken again) with the fixed
     arrival lag 2, the gilfree transport (the pinned packer v2, the Python worker, synchronous) in every arm: per tick
     the rows, the input tokens, the logits (bits), the LOGICAL page tables (the service's decisions), memory() without
     slots_physical, the twins' counters; at the end the committed tokens and the journal. Each component's
     precondition is asserted on the full arm BEFORE the claim (L34): evictions > 0; a pool lane changed hands
     between two requests; the shared kernel ran (keyed calls with >= 2 rows) with the FUSED union build only; the
     fused resolve launched kernels with err 0; the twins are the fused ones; a batched rebuild call published >= 2
     banks; the warm-up ran once. SCOPE ([integ3], review-integ2 M4): the attn component runs the shared kernel's
     torch TWIN and the fused union's torch twin in this round (impl "reference"), the transport synchronously, B2
     checked, no inference mode; the production context is ST9's.
ST9  [integ3] THE PRODUCTION CONTEXT (review M4): the same forced pressure round with the shared Triton KERNEL and the
     fused union KERNEL (TRITON_INTERPRET) against piece 3's KERNEL, under torch.inference_mode, the gilfree FEEDER
     thread (background), B2 UNCHECKED: full == default bitwise, every component acted; and a pool-path fault that
     changes LOGITS (one translated page one position off at the 9th translate call; L34 precondition: the entry
     changed) is caught with its first difference in the logits.
ST4b [integ3] review M5: check mode compares the fused lane map with a SHADOW pool bound by S-C's rule from the torch
     path's decisions (and fills the reference bias rows on the shadow lanes); a planted fused-only bind fault (one
     bind one slot off; precondition: the map changed) is reported as a difference.
ST10 [integ3] the GPU witnesses (review M2): XPool's handoff counter and per-(layer, head) peak, the engine's batched
     counters, SharedKV's pages_saved; stack_acted / stack_acted_gates; the fused dispatch gate is REPORT in a stack
     cell (M1); stack_bands_verdict and write_stack_bands (M3).
ST2  THE 2^k-LITE MATRIX: full minus each one component (six arms) also equals the default, and in each arm the
     removed component is really OFF (its precondition does not hold) while the others stay ON.
ST3  NEGATIVE CONTROL of the matrix comparison: the full stack with the fused resolve's plant 2 (a draft grant takes
     the (q + 1)-th FREE slot): precondition stats d_grant > 0 (the plant changed an output, L34); the comparison
     reports a difference (the logical page tables name the planted slots).
ST4  THE FUSED RESOLVE IN CHECK MODE ON THE FULL STACK (pool + page keys + shared): at every layer the fused outputs
     equal the torch path's (decision fields, pages, the record, and the bias rows written through
     draft_bias_fill_flat on the lanes); compared == layers; err 0.
ST5  ZERO-STRIDE VIEWS (the xpool flat address mode) in the shared-KV KERNEL and piece 3's KERNEL under the Triton
     interpreter: a flat store (M * 64, H, D) as (N, M * 64, H, D) views with stride(0) == 0 and a page table of
     ABSOLUTE positions gives, row for row, the same bits as the per-request store with per-request slots; the fused
     union kernel builds the same union; negative control: the table shifted by one position differs.
ST6  STATIC: FusedResolver keeps S-C's [sharedkv2] page-key hooks (_check_tail_keys once per tick, _page_keys on the
     plan) and implements docs/R2_FUSEDRES_PLAN.md section 5 (release_free after sync_mirror behind the blocking-read
     rule, bind after V2 and after D2, draft_bias_fill_flat, no xpool refusal); the gilfree transport's flat checks
     live in _check (both workers), the C++ worker submits m' = Transfer.phys, issue_desc refuses the pool.
ST7  THE SG5 HARNESS: --stack presets (full / full-no-<c> / none), refusals (explicit component flags with --stack,
     a non-b2 backend, --attn shared without --sel-check, --stack-compare without a lag), the labels, --shipped-batch,
     and one_point_stacks' gates with a stand-in point (equal arms PASS; a differing stack arm FAILS with the tick and
     row named; a differing replica makes the bitwise gate VOID).
ST8  THE STEP CELLS stack1a / stack1t / stack1b (DRYRUN on a clean checkout of this branch).

The kernels run under TRITON_INTERPRET=1 in FRESH child processes (triton reads the flag at its first import, L36).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_integ_stack.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"


def _shared(name, file):
    """A sibling test module as a shared module (loaded once per session under its own name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


TT = _shared("test_r2_tick_shared_st", "test_r2_tick.py")              # the tiny model, the SG5 harness loader
p3 = TT.p3                                                            # the fixture, by name
GFT = _shared("test_r2_transport_gilfree_shared_st", "test_r2_transport_gilfree.py")
pinned = GFT.pinned                                                   # the pinned packer v2, built (fixture)
COMPONENTS = ("twin", "rebuild", "resolve", "layout", "attn", "warmup")


def _has_triton():
    return importlib.util.find_spec("triton") is not None


# =====================================================================================================================
# the child process (TRITON_INTERPRET=1 from its start): the stack matrix on the CPU tick
# =====================================================================================================================
ST_CHILD = r'''
import importlib.util, json, os, sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "ST child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "ST child: triton was imported before the interpreter flag was read"
import torch
HERE, ROOT, P3PATH, GP_PATH, GP_BUILD, PART = sys.argv[1:7]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


TT = load("test_r2_tick", os.path.join(HERE, "test_r2_tick.py"))
P3 = load("piece3_v2_st", P3PATH)
TK, RES, OWN2 = TT.TK, TT.RES, TT.OWN2
XPM = TT._load("r2_xpool")
GF = TT._load("r2_transport_gilfree")
FR = TT._load("r2_fusedres")
FU = TT._load("r2_twin_fused")
assert FR.triton is not None and FU.triton is not None, "ST child: triton did not import"
SKM = load("sharedkv_st", os.path.join(ROOT, "nosi", "nosi", "bi_paged_attention_shared.py"))
GP = load("gilfree_pack_pinned_st", GP_PATH)
EXT = GP.load(GP_BUILD)
BS = TT.BS
C, X, R_VER, LAG = 8, 40, 4, 2                       # the PRESSURE point (test_r2_xpool CFG), the fixed arrival lag
COMPONENTS = ("twin", "rebuild", "resolve", "layout", "attn", "warmup")


CTX = dict(impl="reference", background=False, checked=True)    # [integ3] the "real" part: kernel, feeder, unchecked


def build(on, plant=0, resolve_mode="fused"):
    """make_res_engine's / make_pool_engine's world with the components in `on` switched ON; every arm: b2 (checked
    unless CTX says otherwise), the gilfree transport (pinned v2, Python worker, synchronous unless CTX["background"]),
    the arrival lag LAG, TickEngine(check=True). CTX["impl"] "kernel" ([integ3] review M4): piece 3's Triton KERNEL in
    the default arms and the shared Triton kernel + the fused union KERNEL in the attn arms (the interpreter)."""
    base_ops = TT.make_ops(P3)
    if CTX["impl"] == "kernel":
        base_ops = SimpleNamespace(linear=base_ops.linear, mean=base_ops.mean,
                                   attn=SimpleNamespace(bi_paged_decode=P3.bi_paged_decode,
                                                        build_page_table=P3.build_page_table))
    sk = None
    ops = base_ops
    if "attn" in on:
        sk = SKM.SharedKV(P3, mode="split", impl=CTX["impl"], union="fused")
        sk.count_saved = True
        ops = SimpleNamespace(linear=base_ops.linear, attn=sk, mean=base_ops.mean)
    ref = TT.make_engine(TT.make_ops(P3), S_list=TT.S_TINY, steps=TT.STEPS, seed=1)
    S_list = TT.S_TINY
    B, NBr = len(S_list), C + X
    n_host = (ref.NB + 1) * BS
    xp = XPM.XPool(TT.NL_LAYERS, B, TT.HKV, NBr, mode="pool", C=C, X_pool=X) if "layout" in on else None
    host_k = [torch.zeros(B, n_host, TT.HKV, TT.D) for _ in range(TT.NL_LAYERS)]
    host_v = [torch.zeros(B, n_host, TT.HKV, TT.D) for _ in range(TT.NL_LAYERS)]
    sk_, sv_, sb_ = [], [], []
    for l in range(TT.NL_LAYERS):
        if xp is not None:
            k, v, bias = xp.new_store(TT.D)
        else:
            k = torch.zeros(B, (NBr + 4) * BS, TT.HKV, TT.D)
            v = torch.zeros_like(k)
            bias = torch.zeros(B, (NBr + 4) * BS, TT.HKV)
        for b, S in enumerate(S_list):
            full = (S // BS) * BS
            host_k[l][b, :full] = ref.store_k[l][b, :full]
            host_v[l][b, :full] = ref.store_v[l][b, :full]
            for j in range(4):                                          # the four tail pages
                e_ref = (ref.NB + j) * BS
                if xp is not None:
                    p0 = xp.tail_pos(b, j) * BS
                    k[p0:p0 + BS] = ref.store_k[l][b, e_ref:e_ref + BS]
                    v[p0:p0 + BS] = ref.store_v[l][b, e_ref:e_ref + BS]
                    bias[p0:p0 + BS] = ref.store_bias[l][b, e_ref:e_ref + BS]
                else:
                    e_new = (NBr + j) * BS
                    k[b, e_new:e_new + BS] = ref.store_k[l][b, e_ref:e_ref + BS]
                    v[b, e_new:e_new + BS] = ref.store_v[l][b, e_ref:e_ref + BS]
                    bias[b, e_new:e_new + BS] = ref.store_bias[l][b, e_ref:e_ref + BS]
            if S > full:
                r0 = (ref.NB + (S // BS) % 2) * BS
                host_k[l][b, full:S] = ref.store_k[l][b, r0:r0 + S - full]
                host_v[l][b, full:S] = ref.store_v[l][b, r0:r0 + S - full]
        sk_.append(k)
        sv_.append(v)
        sb_.append(bias)
    eng = TK.TickEngine(ref.model, ops, ref.twins, sk_, sv_, sb_, S_list, ref.M_alloc, nl=TT.NL_CPU,
                        stage1=TT._stage1, pool=TT._pool, check=True, xpool=xp,
                        twin="fused" if "twin" in on else "torch",
                        rebuild="batched" if "rebuild" in on else "per_call")
    be = TT.Be()
    tp = GF.GilfreeTransport(EXT, host_k, host_v, sk_, sv_, BS, TT.D, threads=2, host_wait=be.host_wait, xpool=xp,
                             warm_up="warmup" in on, background=CTX["background"])
    cis = [t.total_cis[:B] for t in eng.twins]
    args = (TT.NL_LAYERS, B, TT.HKV, C, X, R_VER, tp, RES.TorchGather(host_k, host_v, sk_, sv_))
    skw = dict(store_b=sb_, cis=cis, block_bytes=BS * 2 * TT.D * 4, host_wait=be.host_wait, arrival_lag=LAG,
               checked=CTX["checked"])
    if xp is not None:
        skw["xpool"] = xp
    if "resolve" in on:
        skw["resolve"] = resolve_mode
        if plant:
            skw["fused_plant"] = plant
    svc = OWN2.OwnResidency(*args, device="cpu", nblk=n_host // BS, **skw)
    eng.attach_backend(TK.ResidencyBackend(eng, svc, host_k, host_v, lambda: TT.Ev(True), 3))
    return eng, SimpleNamespace(svc=svc, tp=tp, xp=xp, sk=sk)


def bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def handoffs(maps, NS):
    owner, n = {}, 0
    for m in maps:
        L, B, H, _ = m.shape
        for l in range(L):
            for b in range(B):
                for h in range(H):
                    for p in m[l, b, h, :NS].tolist():
                        if p < 0:
                            continue
                        prev = owner.get((l, h, p))
                        if prev is not None and prev != b:
                            n += 1
                        owner[(l, h, p)] = b
    return n


def run(on, toks, plant=0, resolve_mode="fused", hook=None):
    eng, w = build(on, plant, resolve_mode)
    if hook is not None:                                              # [integ3] a planted fault (its own record)
        hook(eng, w)
    svc = w.svc
    pages, ticks, maps, batched = [], [], [], []
    resolve0 = svc.resolve

    def resolve(l, rows, sel):
        p = resolve0(l, rows, sel)
        R, Hh = int(p.page_table.shape[0]), int(p.page_table.shape[1])
        pages.append((l, [[p.page_table[i, h, :int(p.num_pages[i, h])].tolist() for h in range(Hh)] for i in range(R)]))
        return p
    svc.resolve = resolve
    pb0 = eng._publish_batched

    def pb(pub, swap=False):
        batched.append(len(pub))
        return pb0(pub, swap=swap) if swap else pb0(pub)
    eng._publish_batched = pb

    def on_tick(rows, tokens, out):
        mem = dict(svc.memory())
        mem.pop("slots_physical")
        ctr = torch.stack([torch.stack((t.pos, t.n_comp, t.n_ccis, t.epoch)) for t in eng.twins])
        ticks.append(dict(rows=[(r.b, r.kind, r.pos, r.epoch, r.i, getattr(r, "m", None)) for r in rows],
                          tokens=[t.slot if isinstance(t, TK.Ref) else int(t) for t in tokens],
                          logits=bits(out.logits.clone()), pages=list(pages), memory=mem, ctr=ctr.clone()))
        pages.clear()
        if w.xp is not None:
            maps.append(w.xp.pos_of.clone())
    S_list = eng.S_list

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    reject_at, _ = TK.forced_schedule(3)
    rnd, err = None, None
    try:
        rnd = TK.run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [TT.STEPS] * eng.B, K=3, LAMBDA=7,
                           teacher=teacher, reject_at=reject_at, on_tick=on_tick)
    except Exception as e:                                            # noqa: BLE001 - recorded, compared
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    js = dict(svc.journal_counts)
    fr = svc.fusedres.report() if svc.fusedres is not None else None
    pre = dict(
        evictions=js.get("release:evict", 0) + js.get("release:verifier_evict", 0)
        + js.get("release:verifier_evict_inflight", 0),
        handoffs=handoffs(maps, w.xp.n_hist) if w.xp is not None else 0,
        xpool=w.xp is not None, xpool_err=int(w.xp.err) if w.xp is not None else 0,
        shared=w.sk.stats["shared"] if w.sk is not None else 0,
        union_builds=dict(w.sk.union_builds) if w.sk is not None else None,
        want_keys=bool(svc.want_keys),
        fused=None if fr is None else dict(launches=fr["launches"], layers=fr["layers"], err=fr["err"],
                                           compared=fr["compared"], diffs=fr["diffs"], first_diffs=fr["first_diffs"],
                                           d_grant=fr["stats"]["d_grant"]),
        twin=eng.twin_variant, rebuild=eng.rebuild, batched_max=max(batched) if batched else 0,
        warm_calls=w.tp.warm_calls,
        xpool_handoffs=int(w.xp.handoffs) if w.xp is not None else 0,                 # [integ3] the GPU witnesses
        xpool_report=w.xp.report() if w.xp is not None else None,
        pages_saved=int(w.sk.pages_saved) if w.sk is not None and w.sk.pages_saved is not None else 0,
        eng_batched=[eng.batched_calls, eng.batched_max])
    rec = dict(ticks=ticks, err=err, out=[list(r.out) for r in rnd.reqs] if rnd is not None else None, journal=js,
               mirror=eng.mirror_failures(), stats={k: eng.stats[k] for k in ("ticks", "rows", "rollbacks",
                                                                             "publications", "void_ticks")})
    w.tp.close()
    return rec, pre


def first_diff(a, b):
    n = min(len(a["ticks"]), len(b["ticks"]))
    for t in range(n):
        for f in ("rows", "tokens", "logits", "pages", "memory", "ctr"):
            x, y = a["ticks"][t][f], b["ticks"][t][f]
            if not (torch.equal(x, y) if torch.is_tensor(x) else x == y):
                return [t, f]
    if len(a["ticks"]) != len(b["ticks"]) or a["err"] != b["err"]:
        return [n, "ticks %d / %d, errors %r / %r" % (len(a["ticks"]), len(b["ticks"]), a["err"], b["err"])]
    for f in ("out", "journal", "stats"):
        if a[f] != b[f]:
            return ["end", f]
    return None


OUT = {}
toks, _ = TT.serial_ref(TT.make_engine(TT.make_ops(P3)))
if PART == "matrix":
    base, pre0 = run(set(), toks)
    OUT["none"] = dict(diff=None, pre=pre0, err=base["err"], mirror=base["mirror"], stats=base["stats"],
                       ticks=len(base["ticks"]))
    arms = [("full", set(COMPONENTS), 0)] + [("full-no-" + c, set(COMPONENTS) - {c}, 0) for c in COMPONENTS]
    arms.append(("plant2", set(COMPONENTS), 2))
    for name, on, plant in arms:
        rec, pre = run(on, toks, plant=plant)
        OUT[name] = dict(diff=first_diff(base, rec), pre=pre, err=rec["err"], mirror=rec["mirror"], stats=rec["stats"],
                         ticks=len(rec["ticks"]))
elif PART == "check":
    rec, pre = run(set(COMPONENTS), toks, resolve_mode="check")
    base, _ = run(set(), toks)
    OUT["check"] = dict(pre=pre, err=rec["err"], diff=first_diff(base, rec))
    # [integ3] review M5: a fused-path-only BIND fault (the fused pool binds the draft takes one slot off) must show
    # as a check-mode difference in the lane map; L34 precondition: the planted bind changed the map
    plant_log = []

    def bind_plant(eng, w):
        xp = w.xp
        orig = xp.bind

        def planted(l, bb, hh, slot, mask):
            if plant_log or not bool((mask.reshape(-1) != 0).any()):
                return orig(l, bb, hh, slot, mask)
            clean = xp.clone()                                        # (its __dict__ holds this plant: call the
            XPM.XPool.bind(clean, l, bb, hh, slot, mask)              # class's bind on the copy)
            out = orig(l, bb, hh, (slot + 1).clamp(max=xp.n_hist - 1), mask)
            plant_log.append(dict(l=int(l), changed=not torch.equal(clean.pos_of, xp.pos_of)))
            return out
        xp.bind = planted
    recp, prep = run(set(COMPONENTS), toks, resolve_mode="check", hook=bind_plant)
    OUT["check_plant"] = dict(pre=prep, err=recp["err"], plant=plant_log)
elif PART == "real":
    # [integ3] review M4: the production context on the CPU: the shared Triton KERNEL and the fused union KERNEL
    # (the interpreter) against piece 3's KERNEL, torch.inference_mode, the gilfree feeder thread, B2 unchecked
    CTX.update(impl="kernel", background=True, checked=False)
    with torch.inference_mode():
        base, pre0 = run(set(), toks)
        full, pre1 = run(set(COMPONENTS), toks)
        tlog = []

        def translate_plant(eng, w):                                  # a pool-path fault that changes LOGITS: one
            xp = w.xp                                                 # translated page of one row goes one position
            orig = xp.translate                                       # off, at the 9th keyed call (mid-round)
            calls = [0]

            def planted(l, table, seq_idx):
                p = orig(l, table, seq_idx)
                calls[0] += 1
                if calls[0] == 9 and not tlog:
                    ok = (p[:, 0, 0] >= 0).nonzero().view(-1)
                    if ok.numel():
                        r = int(ok[0])
                        before = int(p[r, 0, 0])
                        p = p.clone()
                        p[r, 0, 0] = (before + 1) % xp.A
                        tlog.append(dict(call=calls[0], l=int(l), row=r, before=before, after=int(p[r, 0, 0])))
                return p
            xp.translate = planted
        plant, prep = run(set(COMPONENTS), toks, hook=translate_plant)
    OUT["real"] = dict(none=dict(pre=pre0, err=base["err"], ticks=len(base["ticks"])),
                       full=dict(pre=pre1, err=full["err"], diff=first_diff(base, full), mirror=full["mirror"]),
                       plant=dict(log=tlog, err=plant["err"], diff=first_diff(full, plant)))
print("ST_JSON " + json.dumps(OUT))
'''


def _child(part, p3mod, pk):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", ST_CHILD, str(HERE), str(ROOT), p3mod.__file__, pk.gp.__file__,
                        os.path.dirname(pk.ext.__file__), part], env=env, cwd=str(ROOT), capture_output=True,
                       text=True, timeout=7200)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("ST_JSON ")]
    assert r.returncode == 0 and last, ("ST child %s exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                                        % (part, r.returncode, r.stdout[-4000:], r.stderr[-8000:]))
    return json.loads(last[-1][len("ST_JSON "):])


@pytest.fixture(scope="module")
def matrix(p3, pinned):
    if not _has_triton():
        pytest.skip("triton is not installed")
    return _child("matrix", p3, pinned)


def _on(pre):
    """Which components a run's record shows ON (the preconditions, L34): each from its own witness."""
    fu = pre["fused"]
    ub = pre["union_builds"] or {}
    return dict(twin=pre["twin"] == "fused",
                rebuild=pre["rebuild"] == "batched" and pre["batched_max"] >= 2,
                resolve=fu is not None and fu["launches"] > 0 and fu["layers"] > 0,
                layout=pre["xpool"] and pre["handoffs"] > 0,
                attn=pre["shared"] > 0 and ub.get("fused", 0) > 0 and ub.get("torch", 0) == 0 and pre["want_keys"],
                warmup=pre["warm_calls"] == 1)


# =====================================================================================================================
# ST1 / ST2 / ST3 the matrix
# =====================================================================================================================
def test_ST1_the_full_stack_equals_the_default_stack_bitwise(matrix):
    base, full = matrix["none"], matrix["full"]
    assert base["err"] is None and base["ticks"] > 10 and base["mirror"] == [], base
    assert base["pre"]["evictions"] > 0, ("precondition: no eviction in the pressure round", base["pre"])
    assert not any(_on(base["pre"]).values()), ("the default arm has a component ON", _on(base["pre"]))
    on = _on(full["pre"])
    assert all(on.values()), ("precondition: a component of the full stack did not act", on, full["pre"])
    assert full["pre"]["fused"]["err"] == 0 and full["pre"]["xpool_err"] == 0
    assert full["err"] is None and full["mirror"] == [], (full["err"], full["mirror"])
    assert full["diff"] is None, ("the full stack differs from the default stack", full["diff"])
    assert full["stats"] == base["stats"] and full["stats"]["publications"] > 0 and full["stats"]["rollbacks"] > 0
    print("ST1 OK: %d ticks; full stack == default bitwise; preconditions %s" % (full["ticks"], full["pre"]))


@pytest.mark.parametrize("comp", COMPONENTS)
def test_ST2_full_minus_each_component_equals_the_default(matrix, comp):
    arm = matrix["full-no-" + comp]
    on = _on(arm["pre"])
    assert not on[comp], ("precondition: the removed component is still ON", comp, arm["pre"])
    assert all(v for c, v in on.items() if c != comp), ("precondition: another component did not act", on)
    assert arm["err"] is None and arm["mirror"] == [], (arm["err"], arm["mirror"])
    assert arm["diff"] is None, ("full minus %s differs from the default" % comp, arm["diff"])


def test_ST3_the_matrix_comparison_sees_a_planted_fault(matrix):
    arm = matrix["plant2"]
    fu = arm["pre"]["fused"]
    assert fu is not None and fu["d_grant"] > 0, ("precondition (L34): plant 2 changed no draft take", fu)
    assert arm["diff"] is not None, "the planted off-by-one take slot was NOT detected by the matrix comparison"
    if arm["err"] is None:                                            # the plan's own pages name the planted slots
        assert arm["diff"][1] == "pages", ("the first difference is not in the logical page tables", arm["diff"])
    else:                                                             # or the plant voided a tick by its own flag
        assert "fused_res_err" in arm["err"], ("an error that is not the plant's own flag", arm["err"])
    print("ST3 control: plant 2 effective (d_grant %d), first difference %s" % (fu["d_grant"], arm["diff"]))


# =====================================================================================================================
# ST4 the fused resolve in check mode on the full stack
# =====================================================================================================================
def test_ST4_fused_resolve_check_mode_on_the_full_stack(p3, pinned):
    if not _has_triton():
        pytest.skip("triton is not installed")
    out = _child("check", p3, pinned)
    r = out["check"]
    fu = r["pre"]["fused"]
    assert r["err"] is None, r["err"]
    assert fu is not None and fu["compared"] > 0 and fu["compared"] == fu["layers"], fu
    assert fu["diffs"] == 0 and fu["err"] == 0, (fu["first_diffs"], fu["err"])
    assert r["pre"]["xpool"] and r["pre"]["handoffs"] > 0 and r["pre"]["want_keys"], r["pre"]
    assert r["diff"] is None, r["diff"]
    # ST4b (review M5): the shadow pool sees a fused-only bind fault
    p = out["check_plant"]
    assert p["plant"] and p["plant"][0]["changed"], ("precondition (L34): the planted bind left the map as it was", p)
    pf = p["pre"]["fused"]
    assert pf is not None and pf["diffs"] > 0, ("the planted bind fault was NOT seen by check mode", pf)
    assert any("xpool" in str(d[-1]) or "store_b" in str(d[-1]) for d in pf["first_diffs"]), pf["first_diffs"]


# =====================================================================================================================
# ST9 the production context: real kernels, inference mode, the feeder thread, B2 unchecked (review M4)
# =====================================================================================================================
def test_ST9_the_production_context_round_with_the_real_kernels(p3, pinned):
    if not _has_triton():
        pytest.skip("triton is not installed")
    r = _child("real", p3, pinned)["real"]
    assert r["none"]["err"] is None and r["none"]["ticks"] > 10, r["none"]
    full = r["full"]
    on = _on(full["pre"])
    assert all(on.values()), ("precondition: a component did not act in the production context", on, full["pre"])
    assert full["pre"]["pages_saved"] > 0, ("precondition: the union never saved a row page", full["pre"])
    assert full["err"] is None and full["mirror"] == [], (full["err"], full["mirror"])
    assert full["diff"] is None, ("kernels / feeder / inference mode: the full stack differs", full["diff"])
    pl = r["plant"]
    assert pl["log"] and pl["log"][0]["before"] != pl["log"][0]["after"], ("precondition (L34): no entry planted", pl)
    assert pl["diff"] is not None and pl["diff"][1] == "logits", ("the planted page was not seen in the logits", pl)


# =====================================================================================================================
# ST5 zero-stride views in the shared kernel and piece 3's kernel (the xpool flat address mode)
# =====================================================================================================================
ST5_CHILD = r'''
import importlib.util, os, sys
assert os.environ.get("TRITON_INTERPRET") == "1"
sys.dont_write_bytecode = True
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bi = load(sys.argv[1], "bi_interp_st5")
sk = load(sys.argv[2], "sk_interp_st5")
t = load(sys.argv[3], "sk_tests_st5")
fails = []
c = t.make_group_case(3, "half", rows_per_req=[4, 3, 1, 2])
q, k, v, bias, seq, table, npg, last = t.args_of(c)
N, SR, H, D = k.shape
S = SR // bi.PAGE
flat_k, flat_v, flat_b = k.reshape(N * SR, H, D), v.reshape(N * SR, H, D), bias.reshape(N * SR, H)
vk, vv, vb = (x.unsqueeze(0).expand((N,) + tuple(x.shape)) for x in (flat_k, flat_v, flat_b))
assert vk.stride(0) == 0 and vb.stride(0) == 0 and vk.stride(-1) == 1, "precondition: stride(0) == 0 views"
ftab = torch.where(table >= 0, seq.view(-1, 1, 1).to(torch.int32) * S + table, table).contiguous()
assert not torch.equal(ftab, table), "precondition: the absolute positions differ from the per-request slots"
ref = bi.bi_paged_decode(q, k, v, bias, seq, table, npg, last)
p3v = bi.bi_paged_decode(q, vk, vv, vb, seq, ftab, npg, last)
for r in range(ref.shape[0]):
    if not t.bits_equal(p3v[r], ref[r]):
        fails.append(("piece3 views", r))
for mode in ("split", "whole"):
    a = sk.bi_paged_decode_shared(bi, q, k, v, bias, seq, table, npg, last, c["key"], rows_max=4, mode=mode)
    pps = bi.PAGES_PER_SPLIT
    un = sk.build_union_fused(seq, ftab, npg, c["key"], 4, pps, mode == "whole")
    un_t = sk.build_union(seq, ftab, npg, c["key"], 4, pps, mode == "whole")
    for f in ("grow", "ut", "rowpos", "win"):
        if not torch.equal(un[f], un_t[f]):
            fails.append(("fused union on absolute positions", mode, f))
    b = sk.bi_paged_decode_shared(bi, q, vk, vv, vb, seq, ftab, npg, last, c["key"], rows_max=4, mode=mode, union=un)
    for r in range(ref.shape[0]):
        if not t.bits_equal(b[r], a[r]):
            fails.append(("shared views vs per-request", mode, r))
        if not t.bits_equal(b[r], ref[r]):
            fails.append(("shared views vs piece 3", mode, r))
    tw = sk.bi_paged_decode_shared_reference(bi, q, vk, vv, vb, seq, ftab, npg, last, c["key"], rows_max=4, mode=mode)
    if not torch.equal(tw, bi.bi_paged_decode_reference(q, k, v, bias, seq, table, npg, last)):
        fails.append(("shared twin on views", mode))
bad = torch.where(ftab >= 0, (ftab + 1) % (N * S), ftab).contiguous()          # NEGATIVE CONTROL: one position off
neg = sk.bi_paged_decode_shared(bi, q, vk, vv, vb, seq, bad, npg, last, c["key"], rows_max=4, mode="split")
if all(t.bits_equal(neg[r], ref[r]) for r in range(ref.shape[0])):
    fails.append(("the shifted table was NOT seen",))
if fails:
    print("FAILS", fails[:12])
    sys.exit(1)
print("ST5 OK")
'''


def test_ST5_zero_stride_views_in_the_shared_kernel_and_piece3():
    if not _has_triton():
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", ST5_CHILD, str(NOSI / "bi_paged_attention.py"),
                        str(NOSI / "bi_paged_attention_shared.py"), str(HERE / "test_bi_paged_attention_shared.py")],
                       env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=3000)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    assert r.returncode == 0 and last == "ST5 OK", (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))


# =====================================================================================================================
# ST6 static: the merge points
# =====================================================================================================================
def _fn(src, cls, name):
    tree = ast.parse(src)
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, f)


def test_ST6_the_merge_points_in_the_source():
    fr = (NOSI / "r2_fusedres.py").read_text()
    init, res, lay = (_fn(fr, "FusedResolver", n) for n in ("__init__", "resolve", "_layer"))
    assert "does not bind the shared X pool" not in fr                       # the refusal is gone
    assert "svc._check_tail_keys(rows)" in res and "svc._page_keys(" in res  # [sharedkv2] hooks kept
    assert res.index("svc._check_tail_keys(rows)") < res.index("svc.sync_mirror()")
    assert res.index("svc.sync_mirror()") < res.index("self.xp.release_free(svc.dmirror.code)")
    assert "DECISION_READ_BLOCKING" in res
    assert lay.count("self.xp.bind(") == 2 and "XP.draft_bias_fill_flat(" in lay
    assert lay.index("self.xp.bind(l, vdec.bh[0]") < lay.index("svc.dev.wait(l, vdec)")
    assert "flat_bias" in lay and "self.xp = getattr(svc, \"xpool\", None)" in init
    sc = (NOSI / "r2_residency_sc.py").read_text()
    sres = _fn(sc, "ResidencySC", "resolve")
    assert "self._check_tail_keys(rows)" in sres and "self.xpool.release_free(self.dmirror.code)" in sres
    gf = (NOSI / "r2_transport_gilfree.py").read_text()
    chk, sub, desc, rel = (_fn(gf, "GilfreeTransport", n) for n in ("_check", "_submit", "issue_desc", "_release"))
    assert "tr.phys" in chk and "history arena" in chk
    assert "tr.phys if flat else tr.b * self.n_slots + tr.slot" in sub and "unsqueeze(0)" in sub
    assert "self._check(layer, transfers)" in sub and "self._check(layer, transfers)" in rel
    assert "self.xpool is not None" in desc
    tk = (NOSI / "r2_tick.py").read_text()
    lay_t = _fn(tk, "TickEngine", "_layer")
    assert lay_t.count("**akw)") == 2                                          # both attention calls get the keys


# =====================================================================================================================
# ST7 the SG5 harness
# =====================================================================================================================
BASE = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]


def test_ST7_the_sg5_stack_presets_and_refusals():
    G = TT._sg5_gate()
    full = G.parse_args(BASE + ["--backends", "b2", "--X", "64", "--stack", "full", "--sel-check"])
    assert (full.twin, full.rebuild, full.b2_resolve, full.layout, full.attn, full.union, full.gilfree_warm_up) == (
        "fused", "batched", "fused", "pool", "shared", "fused", "on")
    assert full.variant_list == [("fused", "batched")]
    assert G.variant_label(full, "exact", "b2", full.variant_list[0]) == "SG5_exact_b2_stack_full"
    off = dict(twin="torch", rebuild="per_call", b2_resolve="torch", layout="stream", attn="per_row",
               gilfree_warm_up="off")
    keys = dict(twin="twin", rebuild="rebuild", resolve="b2_resolve", layout="layout", attn="attn",
                warmup="gilfree_warm_up")
    for c in COMPONENTS:
        a = G.parse_args(BASE + ["--backends", "b2", "--X", "64", "--stack", "full-no-" + c, "--sel-check"])
        for c2, k in keys.items():
            got = getattr(a, k)
            if c2 == c:
                assert got == off[k], (c, k, got)
            else:
                assert got == getattr(full, k), (c, c2, k, got)
    none = G.parse_args(BASE + ["--backends", "b2", "--stack", "none"])
    assert all(getattr(none, k) == v for k, v in off.items())
    d = G.parse_args(BASE)                                                    # the defaults are unchanged
    assert (d.stack, d.attn, d.union, d.sel_check, d.gilfree_warm_up, d.b2_resolve, d.layout, d.twin) == (
        "", "per_row", "torch", False, "on", "torch", "stream", "torch")
    for bad in (["--backends", "b2", "--stack", "full", "--twin", "fused", "--sel-check"],     # explicit + preset
                ["--backends", "sc", "--stack", "full", "--sel-check"],                       # resolve fused: b2 only
                ["--backends", "b2", "--stack", "full"],                                      # shared: no sel check
                ["--backends", "b2", "--stack", "full", "--sel-check", "--stack-compare"],    # compare: no lag
                ["--backends", "b2", "--stack-compare", "--arrival-lag", "2"],                # compare: no stack
                ["--backends", "v1", "--attn", "shared", "--sel-check"],                      # v1: no page keys
                ["--backends", "b2", "--attn", "shared"],                                     # no sel check
                ["--stack", "full-no-bogus"], ["--shipped-batch", "-1"]):
        with pytest.raises(SystemExit):
            G.parse_args(BASE + bad)
    ok = G.parse_args(BASE + ["--backends", "b2", "--stack", "full", "--shared-unchecked", "--repeat-docs",
                              "--batch", "216", "--shipped-batch", "64"])                   # [integ3] stack1b's B
    assert ok.shared_unchecked and ok.shipped_batch == 64 and ok.m_fix == 4 * 216
    src = (EFF / "r2_tick_sg5.py").read_text()
    rc = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                                          and n.name == "run_child"))
    assert 'cmd += ["--batch", str(a.shipped_batch)]' in rc
    me = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                                          and n.name == "make_engine"))
    assert 'warm_up=getattr(a, "gilfree_warm_up", "on") == "on"' in me and 'eng.check = bool(getattr(a, "sel_check"' in me


def test_ST10_the_gpu_witnesses_and_gates():
    """[integ3] review M1 / M2 / M3: the witnesses the stack cells read, and their gates."""
    XP = TT._load("r2_xpool")
    # XPool's handoff counter and per-(layer, head) peak: request 0 binds lanes, releases them, request 1 takes them
    xp = XP.XPool(1, 2, 1, 4, mode="pool", C=1, X_pool=3)                   # A = 2 x 1 + 3 = 5
    z = torch.zeros(4, dtype=torch.int64)
    xp.bind(0, z, z, torch.arange(4), torch.ones(4, dtype=torch.bool))      # request 0 takes slots 0..3
    assert int(xp.handoffs) == 0 and int(xp.peak.max()) == 4 and int(xp.err) == 0
    code = torch.full((1, 2, 1, 4), XP.CODE_FREE, dtype=torch.int64)
    xp.release_free(code)                                                   # every slot FREE: lanes back
    one = torch.ones(2, dtype=torch.int64)
    xp.bind(0, one, torch.zeros(2, dtype=torch.int64), torch.arange(2), torch.ones(2, dtype=torch.bool))
    assert int(xp.handoffs) == 2, int(xp.handoffs)                          # two lanes went from request 0 to 1
    rep = xp.report()
    assert rep["handoffs"] == 2 and rep["max_peak_lanes"] == 4 and rep["shared_region_used"] is True
    assert rep["pairs_over_base"] == 1                                      # 4 > B x C = 2 at (layer 0, head 0)
    c = xp.clone()
    assert int(c.handoffs) == 2 and c.owner is not xp.owner and torch.equal(c.owner, xp.owner)
    # the engine's batched counters exist from construction (constructor order, C12)
    src = (NOSI / "r2_tick.py").read_text()
    init = src[src.index("        self.rebuild = rebuild"):src.index("        self.model, self.ops = model, ops")]
    assert "self.batched_calls = 0" in init and "self.batched_max = 0" in init
    # SharedKV's pages_saved (device counter, opt-in): a group case with shared pages saves > 0 row pages
    T = _shared("test_bi_paged_attention_shared_st", "test_bi_paged_attention_shared.py")
    sk = T.sk                                                               # the module under test (by path)
    cse = T.make_group_case(4, "same", rows_per_req=[3, 2, 1, 2])
    obj = sk.SharedKV(T.bi, mode="split", impl="reference", union="torch")
    obj.bi_paged_decode(*T.args_of(cse), page_key=cse["key"], rows_max=3)
    assert obj.pages_saved is None                                          # off by default: no counter
    obj.count_saved = True
    obj.bi_paged_decode(*T.args_of(cse), page_key=cse["key"], rows_max=3)
    assert int(obj.pages_saved) > 0, int(obj.pages_saved)
    # stack_acted / stack_acted_gates and the REPORT dispatch gate (M1)
    G = TT._sg5_gate()
    a = SimpleNamespace(stack="full", stack_applied="full", twin="fused", rebuild="batched", b2_resolve="fused",
                        layout="pool", attn="shared", gilfree_warm_up="on", require_xregion=True, b2_fused_plant=0,
                        b2_res_profile=False)
    w = dict(twin_variant="fused", batched_calls=3, batched_max=2, fused=dict(launches=12, layers=4),
             xpool=dict(binds=8, handoffs=5, max_peak_lanes=6208, pairs_over_base=17),
             skv=dict(stats=dict(shared=7), union_builds=dict(fused=7, torch=0), pages_saved=40), warm_calls=1)
    g = {x["name"]: x for x in G.stack_acted_gates(a, "L", w, 64, 96)}
    assert all(g["L_ST_acted_%s" % c]["passed"] and g["L_ST_acted_%s" % c]["must"] for c in COMPONENTS), g
    assert g["L_ST_xregion"]["passed"] and g["L_ST_xregion"]["must"]
    for c, bad in (("twin", dict(twin_variant="torch")), ("rebuild", dict(batched_max=1)),
                   ("resolve", dict(fused=None)), ("layout", dict(xpool=dict(binds=8, handoffs=0))),
                   ("attn", dict(skv=dict(stats=dict(shared=7), union_builds=dict(fused=7), pages_saved=0))),
                   ("warmup", dict(warm_calls=0))):
        g2 = {x["name"]: x for x in G.stack_acted_gates(a, "L", dict(w, **bad), 64, 96)}
        assert not g2["L_ST_acted_%s" % c]["passed"], c                     # each witness discriminates
    w_low = dict(w, xpool=dict(binds=8, handoffs=5, max_peak_lanes=6007))
    assert not {x["name"]: x for x in G.stack_acted_gates(a, "L", w_low, 64, 96)}["L_ST_xregion"]["passed"]
    off = SimpleNamespace(stack="full", stack_applied="none", twin="torch", rebuild="per_call", b2_resolve="torch",
                          layout="stream", attn="per_row", gilfree_warm_up="off")
    g3 = {x["name"]: x for x in G.stack_acted_gates(off, "N", dict(w, fused=None, xpool=None, skv=None,
                                                                      twin_variant="torch", batched_calls=0,
                                                                      warm_calls=0), 64, 96)}
    assert all(g3["N_ST_off_%s" % c]["passed"] for c in COMPONENTS) and "N_ST_xregion" not in g3
    assert G.stack_acted_gates(SimpleNamespace(), "X", w, 64, 96) == []      # not a stack arm: no gate
    rep = dict(err=0, err_bits=[], launches_per_layer=6.0, sel_conversions=0, bias_casts=0, compiled={})
    a_fused = SimpleNamespace(b2_resolve="fused", b2_fused_plant=0, b2_res_profile=False, stack="full")
    gd = [x for x in G.fused_gates(a_fused, "L", "draft_only", rep) if x["name"].endswith("_fused_dispatch")]
    assert len(gd) == 1 and gd[0]["name"].startswith("R_") and not gd[0]["must"] and not gd[0]["passed"]
    a_fused.stack = ""
    gd = [x for x in G.fused_gates(a_fused, "L", "draft_only", rep) if x["name"].endswith("_fused_dispatch")]
    assert gd[0]["must"] and not gd[0]["name"].startswith("R_")              # FR-g's own cells: MUST, unchanged
    # stack_bands_verdict / write_stack_bands (M3)
    v = [33.0, 36.0, 220.0, 450.0]
    assert G.stack_bands_verdict(v, 34.2, [310.0], 0) == "IN"
    for args in ((37.0, [310.0], 0), (34.2, [500.0], 0), (34.2, [310.0], 1), (None, [310.0], 0), (34.2, [], 0),
                 (34.2, [310.0, 300.0], 0), (34.2, [None], 0)):
        assert G.stack_bands_verdict(v, *args) == "OUT", args
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        gates = []
        G.write_stack_bands(SimpleNamespace(stack_bands_v=v, out=d), dict(peak_gb=34.2),
                            dict(points=[dict(T_tick_ms=310.0)]), gates)
        lines = open(os.path.join(d, "stack_bands.txt")).read().splitlines()
        assert lines[-1] == "STACK_BANDS_VERDICT IN" and gates[-1]["name"] == "R_stack_bands"
        gates2 = [dict(name="x", must=True, passed=False)]
        G.write_stack_bands(SimpleNamespace(stack_bands_v=v, out=d), dict(peak_gb=34.2),
                            dict(points=[dict(T_tick_ms=310.0)]), gates2)
        assert open(os.path.join(d, "stack_bands.txt")).read().splitlines()[-1] == "STACK_BANDS_VERDICT OUT"
    for bad in (["--backends", "b2", "--stack", "full", "--sel-check", "--stack-bands", "1:2:3"],
                ["--backends", "b2", "--stack", "full", "--sel-check", "--stack-compare", "--arrival-lag", "2",
                 "--stack-bands", "33:36:220:450"],
                ["--backends", "b2", "--stack", "full-no-layout", "--sel-check", "--require-xregion"],
                ["--backends", "b2", "--require-xregion"]):
        with pytest.raises(SystemExit):
            G.parse_args(BASE + bad)
    ok = G.parse_args(BASE + ["--backends", "b2", "--X", "64", "--stack", "full", "--shared-unchecked",
                              "--stack-bands", "33:36:220:450", "--require-xregion"])
    assert ok.stack_bands_v == [33.0, 36.0, 220.0, 450.0] and ok.require_xregion


def test_ST7_one_point_stacks_gates_with_a_stand_in():
    G = TT._sg5_gate()

    def stand_in(perturb=None, tokens=None):
        calls = []

        def point(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates, label, bare=False,
                  layout=None, fp=None):
            arm = label.rsplit("_", 1)[1]
            calls.append((arm, a.stack_applied, a.twin, a.rebuild, a.b2_resolve, layout, a.attn, a.gilfree_warm_up,
                          ops))
            for t in range(4):
                v = torch.arange(3, dtype=torch.int64) * (t + 1)
                if perturb == arm and t == 2:
                    v = v + torch.tensor([0, 1, 0])
                fp.append(([(0, "v", 10 + t), (1, "d", 20 + t), (2, "v", 30 + t)], v))
            return dict(backend=kind, T_tick_ms={"none": 1300.0, "stack": 330.0, "none2": 1340.0}[arm],
                        peak_gb_round=40.0 if arm != "stack" else 31.5,
                        committed_tokens=[[1, 2, 4]] if tokens == arm else [[1, 2, 3]])
        point.calls = calls
        return point

    def run(**kw):
        gates = []
        pf = stand_in(**kw)
        a = SimpleNamespace(stack="full", twin="fused", rebuild="batched", variant_list=[("fused", "batched")])
        r = G.one_point_stacks(a, None, None, None, None, lambda aa: "ops-" + aa.attn, "b2", 96, 64, 16, None, 48,
                               None, gates, "SG5_exact_b2_stack_full", point_fn=pf)
        return r, {x["name"]: x for x in gates}, pf.calls
    r, g, calls = run()
    assert [c[0] for c in calls] == ["none", "stack", "none2"]
    assert calls[0][1:] == ("none", "torch", "per_call", "torch", "stream", "per_row", "off", "ops-per_row")
    assert calls[1][1:] == ("full", "fused", "batched", "fused", "pool", "shared", "on", "ops-shared")
    for name in ("_ST_replica", "_ST_committed", "_ST_bitwise"):
        assert g["SG5_exact_b2_stack_full" + name]["passed"] and g["SG5_exact_b2_stack_full" + name]["must"], name
    assert r["st"]["rows_compared"] == 12 and abs(r["st"]["tick_ratio"] - 330.0 / 1320.0) < 1e-9
    assert "inside" in g["R_SG5_exact_b2_stack_full_ST_tick"]["detail"]
    r, g, _ = run(perturb="stack")                                     # the stack differs: bitwise FAILS, named
    assert g["SG5_exact_b2_stack_full_ST_replica"]["passed"] and not g["SG5_exact_b2_stack_full_ST_bitwise"]["passed"]
    assert "tick 2 row 1" in g["SG5_exact_b2_stack_full_ST_bitwise"]["detail"]
    r, g, _ = run(perturb="none2")                                     # the replica differs: VOID
    assert not g["SG5_exact_b2_stack_full_ST_replica"]["passed"]
    assert g["SG5_exact_b2_stack_full_ST_bitwise"]["detail"].startswith("VOID")
    r, g, _ = run(tokens="stack")
    assert not g["SG5_exact_b2_stack_full_ST_committed"]["passed"] and not g["SG5_exact_b2_stack_full_ST_bitwise"]["passed"]


# =====================================================================================================================
# ST8 the step cells
# =====================================================================================================================
def test_ST8_the_step_cells_stack1(tmp_path):
    script = EFF / "r2_tick_sg5_step.sh"
    text = script.read_text()
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in text.splitlines():
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)
    lines = text.splitlines()
    root = next(ln for ln in lines if ln.startswith("NOSI_INTEG=")).split()[0][len("NOSI_INTEG="):]
    assert root == "/mnt/beegfs/ojerbi/nosi-r2-integ3"                      # [integ3] the stack cells' ROOT
    # [integ3] M3: the combined cell's order and its stop rules (static; a DRYRUN of it would recurse)
    blk = text[text.index('if [ "$CELL" = stack1 ]; then'):text.index("MODES=exact,fast")]
    order = [blk.index(s) for s in ("sub stack1a", 'sub "stack1n-$c"', "sub stack1p", "sub stack1t",
                                    "grep -qx 'STACK_BANDS_VERDICT IN' \"$bands\"", "sub stack1b")]
    assert order == sorted(order), order
    assert "bands=$OUTROOT/sg5_${RUN}_${JOB}_stack1t/stack_bands.txt" in blk and "exit 6" in blk
    assert "for c in twin rebuild resolve layout attn warmup; do" in blk
    for fam, path in (("NOSI_X1", "nosi-r2-xpool"), ("NOSI_BIGB", "nosi-r2-bigb"), ("FRNOSI", "nosi-r2-fusedres2")):
        got_root = next(ln for ln in lines if ln.startswith(fam + "=")).split()[0][len(fam) + 1:]
        assert got_root == "/mnt/beegfs/ojerbi/" + path, (fam, got_root)   # [integ2] no other family repointed
    assert next(ln for ln in lines if ln.startswith("NOSI=")).split()[0] == "NOSI=/mnt/beegfs/ojerbi/nosi-r2-sg5-p6"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True)
    untracked = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--", "nosi"],
                               capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip() or untracked.stdout.strip():
        pytest.skip("not a clean git checkout: the DRYRUN needs one")
    uid = "f31fc723-9dfe-407a-9947-28173556232a"
    (tmp_path / "418.gpu_uuid").write_text(uid + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}
    env.update(SG5_HOLDERS=str(tmp_path), DRYRUN="1")
    G = TT._sg5_gate()
    cmp_args = "--stack-compare --X-pool 64 --arrival-lag 2 --sel-check --require-xregion"
    want = {"stack1a": (["--stack full " + cmp_args, "--batch 64", "--C 96 ", "--mem=120G"],
                        ["--pool-baseline", "--shared-unchecked", "--stack-bands"]),
            "stack1p": (["--stack full " + cmp_args, "--batch 64", "--C 80 ", "--mem=120G"],
                        ["--pool-baseline", "--shared-unchecked", "--stack-bands"]),
            "stack1n-layout": (["--stack full-no-layout --stack-compare --arrival-lag 2 --sel-check", "--batch 64",
                                "--C 96 "], ["--X-pool", "--require-xregion", "--pool-baseline"]),
            "stack1t": (["--stack full --X-pool 64 --shared-unchecked --stack-bands 33:36:220:450", "--batch 64",
                         "--C 96 ", "--pool-baseline", "--mem=120G"],
                        ["--stack-compare", "--arrival-lag", "--sel-check"]),
            "stack1b": (["--stack full --X-pool 64 --shared-unchecked --repeat-docs --shipped-batch 64 --log-slots -1",
                         "--batch 216", "--mem=240G"], ["--pool-baseline", "--stack-compare", "--arrival-lag",
                                                       "--skip-shipped"])}                    # [integ2] B 216
    for c in ("twin", "rebuild", "resolve", "attn", "warmup"):                                  # [integ3] M1
        want["stack1n-" + c] = (["--stack full-no-%s %s" % (c, cmp_args), "--batch 64", "--C 96 "],
                                ["--pool-baseline", "--shared-unchecked"])
    for cell, (has, hasnt) in want.items():
        r = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "418", "t1", cell], env=env,
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            assert Path(root).resolve() != ROOT.resolve(), r.stderr.strip()[-300:]   # in the cells' ROOT a stop FAILS
            pytest.skip("DRYRUN stopped outside the stack cells' ROOT: %s" % r.stderr.strip()[-200:])
        line = r.stdout.strip().splitlines()[-1]
        got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
        for w in has + ["--modes exact --backends b2", "--X 64 --R-ver 16", "--steps 48",
                        "PYTHONPATH=/mnt/beegfs/ojerbi/nosi-r2-integ3/nosi"]:
            assert w in got or w in line, (cell, w)
        for w in hasnt + ["--sweep ", "--variants", "--b2-checked", "--layout both"]:
            assert w not in got, (cell, w)
        assert "--job-name=sg5-t1-418-%s " % cell.replace("_", "-") in r.stdout and "cell %s" % cell in r.stdout
        assert "[integ] root /mnt/beegfs/ojerbi/nosi-r2-integ3," in r.stdout
        pa = G.parse_args(got.split())
        assert pa.backends == ["b2"], cell
        if cell == "stack1n-layout":
            assert pa.stack == "full-no-layout" and pa.layout == "stream" and not pa.require_xregion
        elif cell.startswith("stack1n-"):
            assert pa.stack == "full-no-" + cell[len("stack1n-"):] and pa.layout == "pool" and pa.require_xregion
        else:
            assert pa.stack == "full" and pa.X_pool == 64 and pa.layout == "pool"
        if cell == "stack1b":                                             # [integ2] bigb's token log, sized
            assert pa.batch == 216 and pa.log_slots == -1 and pa.log_slots_n is not None
        if cell == "stack1t":                                             # [integ3] M3
            assert pa.stack_bands_v == [33.0, 36.0, 220.0, 450.0]
    for bad in (("stack1a", "stack1b"), ("stack1", "stack1a"), ("stack1c",), ("stack1n-bogus",)):
        rb = subprocess.run(["bash", str(script), "--nosi", head.stdout.strip(), "418", "t1", *bad], env=env,
                            capture_output=True, text=True, timeout=120)
        assert rb.returncode == 2 and "usage" in rb.stderr, bad
