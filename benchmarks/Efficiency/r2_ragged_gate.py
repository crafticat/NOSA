"""R2 milestone 2, step 6: the RAGGED ISOLATION GATE of the per-request decode path (nosi/nosi/r2_decode_path.py,
body r2, builder build_ragged), on the GPU, one model, arms run one after another in one process.

THE CLAIM: request 0's arithmetic does not depend on its neighbours. Request 0 (document 0, S0 = 16128 tokens, active
at every step) is decoded greedily for --steps steps (512: request 0's pooled width pos // 64 + 1 grows 8 times, and
its compressed-key count passes 1024 near step 272, the 128-column boundary of the SHIPPED score width) in a batch of
B requests, and every step its logits, its per-layer selection topk_idx[:, 0], _block_map[:, 0], _cache_lens[0] and
its twin counters (pos, n_comp, n_ccis) are recorded; after the last step also its tables (compressed keys, compressed
cis, total_cis, host K / V rows, slot-cache rows below cache_lens).

WIDTHS: the R2 path's own, PREALLOCATED for the run from S_max + --steps (r2_decode_path.ragged_widths), the same in
every arm: M_alloc pooled blocks, Lk_max compressed rows (a constant stage-1 score width, so no launch crosses a
128-column boundary), N_max = n_host rows. No captured graph, no re-warm. Every kernel launch has the same shape in
every arm of the same B.

ARMS (in this order):
  E    equal advance: documents 0 .. B-1, all S0 tokens, all active every step (the reference).
  E2   E again: the positive control (determinism). If E2 is not bitwise E, every later arm is NOT_RUN.
  R    RAGGED: request b has S0 - d_b tokens (--deltas; d_0 = 0) and is active at a step with probability
       --p-active (a seeded table; request 0 always): request 0 must be bitwise E at every step and in its tables.
  NS   negative control, a SHARED counter: R with every request using request B-1's key length and pooled width:
       request 0 must differ from E first at step 0.
  NF   negative control, a width FROZEN at the first step: E with the pooled width of step 0 kept: request 0 must differ
       from E first at its first block boundary, step 64 - S0 % 64 (64 at S0 = 16128), and not before.
  B1   REPORT only: request 0 alone (B = 1). The GEMMs then run at M = 1, not M = B; cuBLAS may pick another kernel,
       so a difference here is not an isolation failure.

F3: past decode step 1024 the shipped decode stops (total_cis has S + 1024 rows, findings F3). The R2 tables are
sized from the horizon; with --steps > 1024 (cell long) the arms E / E2 / R run past it and the summary says so.

ENVIRONMENT: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0. Exit: the number of failed MUST gates; 2 a
refusal.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_compare as C  # noqa: E402

ARMS = ("E", "E2", "R", "NS", "NF", "B1")


def log(msg):
    print("[r2_ragged %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--S0", type=int, default=16128)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--deltas", default="0,1,17,63,64,65,777,4097")
    ap.add_argument("--p-active", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--arms", default="E,E2,R,NS,NF,B1")
    ap.add_argument("--ns-steps", type=int, default=8)
    ap.add_argument("--nf-steps", type=int, default=80)
    a = ap.parse_args(argv)
    a.deltas = [int(x) for x in a.deltas.split(",") if x != ""]
    a.arms = [x for x in a.arms.split(",") if x]
    if [x for x in a.arms if x not in ARMS] or a.arms[:2] != ["E", "E2"]:
        ap.error("arms must start with E,E2 and name only %s (got %s)" % (ARMS, a.arms))
    if len(a.deltas) != a.batch or a.deltas[0] != 0 or min(a.deltas) < 0:
        ap.error("--deltas needs %d non-negative values with d_0 = 0" % a.batch)
    if a.S0 - max(a.deltas) < 32:
        ap.error("every prompt needs >= 32 tokens")
    if "NF" in a.arms and not first_boundary(a.S0) < a.nf_steps <= a.steps:
        ap.error("--nf-steps must pass request 0's first block boundary (step %d)" % first_boundary(a.S0))
    if not 0 < a.ns_steps <= a.steps or not 0 < a.p_active <= 1:
        ap.error("--ns-steps in 1 .. steps, --p-active in (0, 1]")
    return a


def first_boundary(S0: int) -> int:
    """The first decode step s > 0 whose written position S0 + s starts a new block (the pooled width grows)."""
    return 64 - S0 % 64


def activity(steps: int, B: int, p: float, seed: int, ragged: bool):
    """(steps, B) bool CPU: request 0 always active; the others Bernoulli(p) from a seeded generator (ragged) or always."""
    import torch
    if not ragged:
        return torch.ones((steps, B), dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)
    act = torch.rand((steps, B), generator=g) < p
    act[:, 0] = True
    return act


def arm_plan(name, a):
    """(prompt lengths, activity kind, neg, steps, batch) of an arm."""
    B = a.batch
    eq = [a.S0] * B
    rg = [a.S0 - d for d in a.deltas]
    return dict(E=(eq, False, None, a.steps, B), E2=(eq, False, None, a.steps, B), R=(rg, True, None, a.steps, B),
                NS=(rg, True, dict(kind="shared", request=B - 1, step=0), a.ns_steps, B),
                NF=(eq, False, dict(kind="frozen"), a.nf_steps, B), B1=([a.S0], False, None, a.steps, 1))[name]


def load_docs(path, B, L):
    import numpy as np
    import torch
    man = json.load(open(os.path.join(path, "c16k_manifest.json")))
    n, ctx = man["shape"]
    if ctx < L or not 1 <= B <= n:
        raise SystemExit("REFUSED: workload %s x %d cannot give %d documents of %d tokens" % (n, ctx, B, L))
    sums = {}
    for line in open(os.path.join(path, "SHA256SUMS")):
        h, fname = line.split()
        sums[fname] = h
    raw = open(os.path.join(path, "c16k_tokens.bin"), "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sums.get("c16k_tokens.bin") != sha:
        raise SystemExit("REFUSED: c16k_tokens.bin sha256 %s is not SHA256SUMS's" % sha)
    return torch.from_numpy(np.frombuffer(raw, dtype="<i4").reshape(n, ctx)[:B, :L].astype("int64")), sha


def record0(model, cache, st, lg):
    """Request 0's per-step record (CPU)."""
    import torch
    L = len(model.layers)
    eng = [cache.layers[l].cache_engine for l in range(L)]
    return dict(logits=(lg[0, -1].float().cpu(), None),
                topk=(torch.stack([st.topk_log[l][:, 0] for l in range(L)]).cpu(), None),
                block_map=(torch.stack([eng[l]._block_map[:, 0] for l in range(L)]).cpu(), None),
                cache_lens=(torch.stack([eng[l]._cache_lens[0] for l in range(L)]).cpu(), None),
                counters=(torch.stack([torch.stack([t.pos[0], t.n_comp[0], t.n_ccis[0]]) for t in st.twins]).cpu(), None))


def tables0(model, cache, st):
    """Request 0's tables after the last step (CPU), per layer concatenated."""
    out = {}
    for l, t in enumerate(st.twins):
        e = cache.layers[l].cache_engine
        n, m, p = int(t.n_comp[0]), int(t.n_ccis[0]), int(t.pos[0])
        cl = int(e._cache_lens[0])
        for k, v in (("comp_k", t.comp_k[0, :n]), ("ccis", t.ccis[:, 0, :m]), ("total_cis", t.total_cis[0, :p]),
                     ("host_k", e._k_cpu[0, :p]), ("host_v", e._v_cpu[0, :p]), ("win_k", e._k_gpu[0, :cl]),
                     ("win_v", e._v_gpu[0, :cl]), ("win_bias", e._kv_bias_gpu[0, :cl])):
            out["L%02d_%s" % (l, k)] = (v.to("cpu", copy=True), None)     # a copy: the pinned buffers are freed
    return out


def run_arm(name, a, model, docs, ref):
    """Run one arm; compare request 0 with ref (E's records) on the fly. Returns (result, records or None)."""
    import torch
    from nosi.r2_decode_path import build_ragged
    S_list, ragged, neg, steps, B = arm_plan(name, a)
    act = activity(steps, B, a.p_active, a.seed, ragged)
    prompts = [docs[b, :S_list[b]] for b in range(B)]
    keep = ref is None
    recs, tr = [], C.Tracker()
    t0 = time.time()
    with torch.inference_mode():
        cache, st, tok = build_ragged(model, prompts, a.steps, f4_fix=False, neg=neg)
        t_pre = time.time() - t0
        cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
        t1 = time.time()
        for s in range(steps):
            st.set_active(act[s])
            pos = st.pos_host.view(B, 1).to(device="cuda", dtype=torch.int32)
            lg = model.decode_inference(tok, cu, pos, cache)
            new = lg[:, -1].argmax(-1, keepdim=True)
            tok = torch.where(st.active.view(B, 1), new, tok)
            r = record0(model, cache, st, lg)
            if keep:
                recs.append(r)
            else:
                tr.update(s, None, C.compare_views(ref["steps"][s], r))
        torch.cuda.synchronize()
        t_dec = time.time() - t1
        tab = tables0(model, cache, st) if steps == a.steps else None
        if keep:
            recs = dict(steps=recs, tables=tab)
        elif tab is not None:
            tr.update(steps - 1, None, C.compare_views(ref["tables"], tab))
        meta = dict(st.meta, arm=name, S_list=S_list, ragged=ragged, neg=neg, steps=steps, B=B,
                    active_steps=[int(x) for x in act.sum(0)], prefill_s=round(t_pre, 1), decode_s=round(t_dec, 1),
                    ms_per_step=round(1e3 * t_dec / max(steps, 1), 2))
    del cache, st
    gc.collect()
    torch.cuda.empty_cache()
    return dict(arm=name, meta=meta, tracker=tr.summary() if not keep else None), (recs if keep else None), tr


def boundaries(ref, S0):
    """From E's records: request 0's pooled-width growth steps and the step its compressed keys pass 1024 (layer 0)."""
    grow, cross = [], None
    prev = None
    for s, r in enumerate(ref["steps"]):
        pos_before = int(r["counters"][0][0, 0]) - 1
        w = pos_before // 64 + 1
        if prev is not None and w != prev:
            grow.append(s)
        prev = w
        if cross is None and int(r["counters"][0][0, 1]) > 1024:
            cross = s
    return dict(width_growth_steps=grow, n_comp_passes_1024_at_step=cross,
                note="the SHIPPED score width round128(n_comp) changes there (B's sustained harness: about step 287 "
                     "at 16128); the R2 score width is fixed (round128(Lk_max))")


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    try:
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        from nosi import NOSALlama as Llama
        from nosi.r2_decode_path import ragged_widths
        if not NL.R2_DECODE:
            raise SystemExit("REFUSED: NOSI_R2_DECODE is not 1 (the gate drives the switch)")
        if NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_ATTN_SPLITS must be > 0 and NOSI_POOL_BLOCKS 0")
        docs, sha = load_docs(a.workload, a.batch, a.S0)
        os.makedirs(a.out)
        t0 = time.time()
        model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s" % (time.time() - t0))
        gates, results, ref, stop = [], [], None, None
        for name in a.arms:
            if stop:
                gates.append(dict(name=name, must=name != "B1", passed=False, detail="NOT_RUN: " + stop))
                continue
            log("arm %s" % name)
            try:
                res, recs, tr = run_arm(name, a, model, docs, ref)
            except Exception as e:
                traceback.print_exc()
                res, recs, tr = dict(arm=name, error="%s: %s" % (type(e).__name__, str(e)[:400])), None, None
            results.append(res)
            if name == "E":
                ok = recs is not None
                ref = recs
                gates.append(dict(name="E", must=True, passed=ok, detail="reference recorded" if ok else res.get("error", "")))
                if not ok:
                    stop = "the reference arm E failed"
                continue
            if tr is None:
                ok, why = False, "EXCEPTION " + res.get("error", "")
            elif name in ("E2", "R", "B1"):
                ok, why = C.verdict_equal(tr)
            elif name == "NS":
                ok, why = C.verdict_neg(tr, 0, None)
            else:
                ok, why = C.verdict_neg(tr, first_boundary(a.S0), None)
            gates.append(dict(name=name, must=name != "B1", passed=bool(ok), detail=why))
            log("arm %s: %s %s" % (name, "PASS" if ok else "FAIL", why[:300]))
            if name == "E2" and not ok:
                stop = "E2 is not bitwise E (nondeterminism voids the isolation comparison)"
        w = ragged_widths([a.S0], a.steps)
        meta = dict(S0=a.S0, steps=a.steps, batch=a.batch, deltas=a.deltas, p_active=a.p_active, seed=a.seed,
                    arms=a.arms, first_boundary=first_boundary(a.S0), widths=w,
                    widths_rule="the R2 path's own widths, preallocated from S_max + steps; no captured graph, no re-warm",
                    boundaries=boundaries(ref, a.S0) if ref else None, workload_sha256=sha,
                    f3=("past step 1024 the SHIPPED decode stops (total_cis S + 1024 rows, findings F3); the R2 twin's "
                        "total_cis has N_max = %d rows and the body bounds by min(n_host, N_max): lifted for this run"
                        % w["n_max"]) + ("; this run passes step 1024" if a.steps > 1024 else ""),
                    attn_splits=NL._ATTN_SPLITS, nosi_commit=os.environ.get("RAG_COMMIT", "unknown"),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        json.dump(dict(meta=meta, gates=gates, results=results), open(os.path.join(a.out, "summary.json"), "w"),
                  indent=1, sort_keys=True, default=str)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                    g["detail"][:400]))
        print("r2_ragged: %d MUST gate(s) failed" % nfail)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
