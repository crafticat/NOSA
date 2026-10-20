"""R2-M2 step 8 (X4): the CAPTURE GATE of the R2 decode step, on one A100 in the holder
(benchmarks/Efficiency/r2_capture_step.sh runs it). Design: docs/R2_CAPTURE_AUDIT.md.

THE CLAIM: one CUDA-graph capture of the capture-safe decode step (nosi/nosi/r2_capture.py decode_step_fixed),
replayed with ragged activity, computes exactly what the validated eager R2 path (r2_decode_path.decode_inference:
G0, ragged main / long PASS) computes, for EVERY request, bitwise; the state tensors keep their addresses.

BATCH (every arm the same): B requests, request b has S0 - d_b tokens (--deltas, d_0 = 0) of the pinned workload
(r2_ragged_gate.load_docs), built by r2_decode_path.build_ragged (f4_fix False, the widths preallocated from the
horizon); the ragged activity table r2_ragged_gate.activity(steps, B, p, seed, ragged=True) (request 0 always active).
Greedy tokens; an inactive request keeps its token.

ARMS (in this order):
  CE   the validated eager path (model.decode_inference through the NOSI_R2_DECODE switch): the reference. Records
       every step: the logits of every request (float32), every layer's selection (L, H, B, 64), _block_map,
       _cache_lens, the twins' counters; after the last step a sha256 digest per (layer, request, table) of the live
       tables (compressed keys, compressed cis, total_cis, host K / V, the attended window rows).
  CF   the EAGER capture-safe step for every step: MUST == CE at every step and in the final tables; refusal bits 0.
  CG   --warmup eager capture-safe steps, then ONE captured step replayed for the rest: MUST == CE likewise; refusal
       bits 0; MUST (PTR) every state tensor's data_ptr (r2_capture.state_pointers) is the same before the capture and
       after the last replay.
  NG1  CG with the pooled widths frozen at the capture (cb.freeze_widths right before capture_step): MUST differ
       from CE first at ng1_first_step (the first step whose widths differ from the capture step's).
  NG2  CG with the device activity mask left stale after the capture (prepare_step write_active=False): MUST differ
       from CE first at ng2_first_step (the first step whose activity differs from the capture step's).
  T    REPORT: median ms per decode step of CE, CF and CG (CUDA-synchronised wall time around the step only).

EXIT: the number of failed MUST gates; 20 refused; 30 an exception outside an arm (an arm's exception fails its gate).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_compare as C          # noqa: E402
import r2_ragged_gate as RG        # noqa: E402

ARMS = ("CE", "CF", "CG", "NG1", "NG2")


def log(msg):
    print("[r2_capture %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--S0", type=int, default=16128)
    ap.add_argument("--deltas", default="0,1,17,63,64,65,777,4097")
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--p-active", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args(argv)
    a.deltas = [int(x) for x in a.deltas.split(",") if x != ""]
    a.arms = [x for x in a.arms.split(",") if x]
    if [x for x in a.arms if x not in ARMS] or not a.arms or a.arms[0] != "CE":
        ap.error("arms must start with CE and name only %s (got %s)" % (ARMS, a.arms))
    if len(a.deltas) != a.batch or a.deltas[0] != 0 or min(a.deltas) < 0 or a.S0 - max(a.deltas) < 32:
        ap.error("--deltas needs %d non-negative values with d_0 = 0, every prompt >= 32 tokens" % a.batch)
    if not 1 <= a.warmup < a.steps or not 0 < a.p_active <= 1:
        ap.error("--warmup in 1 .. steps - 1, --p-active in (0, 1]")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested)
def prompt_lengths(a):
    return [a.S0 - d for d in a.deltas]


def positions_before(S_list, act):
    """(steps, B) int64: every request's position before every step (the host counter), from the activity table."""
    import torch
    S = torch.tensor(S_list, dtype=torch.int64)
    done = torch.cumsum(act.to(torch.int64), dim=0) - act.to(torch.int64)          # active steps strictly before s
    return S.view(1, -1) + done


def ng1_first_step(S_list, act, capture_step: int):
    """The first step s > capture_step whose pooled widths (pos_before // 64 + 1) differ from the capture step's
    for some request (None: never). NG1 freezes the capture step's widths."""
    pb = positions_before(S_list, act)
    w = pb // 64 + 1
    for s in range(capture_step + 1, act.shape[0]):
        if not bool((w[s] == w[capture_step]).all()):
            return s
    return None


def ng2_first_step(act, capture_step: int):
    """The first step s > capture_step whose activity differs from the capture step's (None: never). NG2 keeps the
    device mask of the capture step."""
    for s in range(capture_step + 1, act.shape[0]):
        if not bool((act[s] == act[capture_step]).all()):
            return s
    return None


def digest(t) -> int:
    """sha256 of a tensor's bytes (CPU copy), as a signed int64: equal bytes <=> equal digest (collisions ignored)."""
    import torch
    b = t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "little", signed=True)


def record_all(cache, twins, logits, topk_log):
    """One step's view of EVERY request: {name: (CPU tensor, request dim)} for r2_g0_compare.compare_views."""
    import torch
    L = len(twins)
    eng = [cache.layers[l].cache_engine for l in range(L)]
    return dict(logits=(logits[:, -1].float().cpu(), 0),
                topk=(topk_log.cpu(), 2),
                block_map=(torch.stack([e._block_map for e in eng]).cpu(), 2),
                cache_lens=(torch.stack([e._cache_lens for e in eng]).cpu(), 1),
                counters=(torch.stack([torch.stack([t.pos, t.n_comp, t.n_ccis]) for t in twins]).cpu(), 2))


def tables_digest(cache, twins):
    """After the last step: per table, a (B,) int64 tensor of digests of every request's LIVE rows, per layer."""
    import torch
    out = {}
    for l, t in enumerate(twins):
        e = cache.layers[l].cache_engine
        rows = {k: [] for k in ("comp_k", "ccis", "total_cis", "host_k", "host_v", "win_k", "win_v", "win_bias")}
        for b in range(t.B):
            n, m, p = int(t.n_comp[b]), int(t.n_ccis[b]), int(t.pos[b])
            cl = int(e._cache_lens[b])
            for k, v in (("comp_k", t.comp_k[b, :n]), ("ccis", t.ccis[:, b, :m]), ("total_cis", t.total_cis[b, :p]),
                         ("host_k", e._k_cpu[b, :p]), ("host_v", e._v_cpu[b, :p]), ("win_k", e._k_gpu[b, :cl]),
                         ("win_v", e._v_gpu[b, :cl]), ("win_bias", e._kv_bias_gpu[b, :cl])):
                rows[k].append(digest(v))
        for k, vals in rows.items():
            out["L%02d_%s" % (l, k)] = (torch.tensor(vals, dtype=torch.int64), 0)
    return out


def pointer_diff(before, after):
    """Names whose data_ptr changed, appeared or disappeared."""
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


# ------------------------------------------------------------------ the arms
def run_arm(name, a, model, docs, act, ref):
    import torch
    from nosi import r2_capture as RC
    from nosi.r2_decode_path import build_ragged
    S_list = prompt_lengths(a)
    B, steps, W = a.batch, a.steps, a.warmup
    prompts = [docs[b, :S_list[b]] for b in range(B)]
    keep = name == "CE"
    recs, tr = [], C.Tracker()
    res = dict(arm=name)
    times = []
    with torch.inference_mode():
        cache, st, tok = build_ragged(model, prompts, steps, f4_fix=False)
        cb, graph = None, None
        if name != "CE":
            kernels, wb = RC.default_kernels()
            cb = RC.CaptureBuffers(model, st, False, kernels, wb)
        cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
        err_seen, ptr_before = [], None
        for s in range(steps):
            if name == "CE":
                st.set_active(act[s])
                pos = st.pos_host.view(B, 1).to(device="cuda", dtype=torch.int32)
                torch.cuda.synchronize()
                t0 = time.time()
                lg = model.decode_inference(tok, cu, pos, cache)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
                topk = torch.stack(st.topk_log)
            else:
                captured = name in ("CG", "NG1", "NG2") and s >= W
                RC.prepare_step(st, cb, tok, act[s], write_active=not (name == "NG2" and s > W))
                if captured and graph is None:
                    if name == "NG1":
                        cb.freeze_widths(st)
                    torch.cuda.synchronize()
                    ptr_before = RC.state_pointers(st, cb, cache)
                    graph = RC.capture_step(model, st, cb, cache)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                t0 = time.time()
                if captured:
                    graph.replay()
                else:
                    RC.decode_step_fixed(model, st, cb, cache)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
                RC.finish_step(st)
                body_err, twin_err = RC.errors(st, cb)
                if body_err or any(twin_err):
                    err_seen.append((s, body_err, twin_err))
                lg = cb.logits
                topk = cb.topk_log
            new = lg[:, -1].argmax(-1, keepdim=True)
            tok = torch.where(st.active.view(B, 1), new, tok)
            r = record_all(cache, st.twins, lg, topk)
            if keep:
                recs.append(r)
            else:
                tr.update(s, None, C.compare_views(ref["steps"][s], r))
        torch.cuda.synchronize()
        tab = tables_digest(cache, st.twins)
        if keep:
            recs = dict(steps=recs, tables=tab)
        else:
            tr.update(steps - 1, None, C.compare_views(ref["tables"], tab))
        if ptr_before is not None:
            res["pointers_changed"] = pointer_diff(ptr_before, RC.state_pointers(st, cb, cache))
            res["pointers_checked"] = len(ptr_before)
        res.update(errors=err_seen[:10], n_error_steps=len(err_seen),
                   ms_per_step=round(1e3 * statistics.median(times), 2) if times else None,
                   warmup=W if name in ("CG", "NG1", "NG2") else None)
        del graph, cb
    del cache, st
    gc.collect()
    torch.cuda.empty_cache()
    return res, (recs if keep else None), (tr if not keep else None)


def verdicts(name, res, tr, a, act):
    """[(gate name, must, passed, detail)] for one arm."""
    W = a.warmup
    out = []
    if "error" in res:
        return [(name, True, False, "EXCEPTION " + res["error"])]
    if name in ("CF", "CG"):
        ok, why = C.verdict_equal(tr)
        out.append((name, True, ok, why))
        out.append((name + "-ERR", True, res["n_error_steps"] == 0, "refusal bits: %s" % (res["errors"] or "none")))
    if name == "CG":
        ch = res.get("pointers_changed")
        out.append(("CG-PTR", True, ch == [], "%s of %s state pointers changed across the replays: %s"
                    % (len(ch) if ch is not None else "?", res.get("pointers_checked"), (ch or [])[:10])))
    if name == "NG1":
        s0 = ng1_first_step(prompt_lengths(a), act, W)
        ok, why = C.verdict_neg(tr, s0, None) if s0 is not None else (False, "no width change after the capture")
        out.append((name, True, ok, "expected first step %s: %s" % (s0, why)))
    if name == "NG2":
        s0 = ng2_first_step(act, W)
        ok, why = C.verdict_neg(tr, s0, None) if s0 is not None else (False, "no activity change after the capture")
        out.append((name, True, ok, "expected first step %s: %s" % (s0, why)))
    return out


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 20
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 20
    import torch
    import nosi.cache_engine as CEN
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    if not NL.R2_DECODE or NL._ATTN_SPLITS <= 0 or CEN.POOL_BLOCKS != 0:
        print("REFUSED: needs NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS=0", file=sys.stderr)
        return 20
    try:
        docs, sha = RG.load_docs(a.workload, a.batch, a.S0)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 20
    os.makedirs(a.out)
    act = RG.activity(a.steps, a.batch, a.p_active, a.seed, True)
    t0 = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    log("model loaded in %.0f s" % (time.time() - t0))
    gates, results, ref = [], [], None
    for name in a.arms:
        if name != "CE" and ref is None:
            gates.append((name, True, False, "NOT_RUN: the reference arm CE failed"))
            continue
        log("arm %s" % name)
        try:
            res, recs, tr = run_arm(name, a, model, docs, act, ref)
        except Exception as e:
            traceback.print_exc()
            res, recs, tr = dict(arm=name, error="%s: %s" % (type(e).__name__, str(e)[:400])), None, None
        results.append(res)
        if name == "CE":
            ref = recs
            gates.append(("CE", True, recs is not None, "reference recorded" if recs is not None else res.get("error", "")))
            continue
        for g in verdicts(name, res, tr, a, act):
            gates.append(g)
            log("%s %s %s" % (g[0], "PASS" if g[2] else "FAIL", g[3][:300]))
    ms = {r["arm"]: r.get("ms_per_step") for r in results}
    gates.append(("T", False, True, "REPORT ms per step (median): CE %s, CF %s, CG %s" % (ms.get("CE"), ms.get("CF"), ms.get("CG"))))
    meta = dict(S0=a.S0, steps=a.steps, warmup=a.warmup, batch=a.batch, deltas=a.deltas, p_active=a.p_active,
                seed=a.seed, arms=a.arms, prompt_lengths=prompt_lengths(a),
                ng1_first_step=ng1_first_step(prompt_lengths(a), act, a.warmup),
                ng2_first_step=ng2_first_step(act, a.warmup), active_steps=[int(x) for x in act.sum(0)],
                workload_sha256=sha, attn_splits=NL._ATTN_SPLITS, nosi_commit=os.environ.get("CAP_COMMIT", "unknown"),
                allocation_rule="transient intermediates from the graph's private pool; state preallocated, in place",
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    out = dict(meta=meta, gates=[dict(name=g[0], must=g[1], passed=bool(g[2]), detail=g[3]) for g in gates],
               results=results)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    nfail = sum(1 for g in gates if g[1] and not g[2])
    for g in gates:
        print("%s %s %s: %s" % ("GATE" if g[1] else "REPORT", g[0], "PASS" if g[2] else "FAIL", g[3][:400]))
    print("r2_capture: %d MUST gate(s) failed" % nfail)
    return min(nfail, 19)


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 30
    sys.exit(rc)
