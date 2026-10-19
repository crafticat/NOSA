"""GATES T0 and T1 of the exact multi-position verifier (Path 1) at U = 1
(retroinfer-eval fork; spec docs/superpowers/specs/2026-09-19-multiposition-verify-path1.md,
section 5(d), first rung of the GPU ladder).

Teacher-forced NOSA-8B on PG-19, the pattern of trace_selections.py /
bias_convention_probe.py: prefill of an L-token prompt, then N decode steps fed
the document's own continuation. Three process modes (NOSI reads its knobs at
import, so each layout is its own process; NOSI_VERIFY_MODE):

  decode   prefill + N SHIPPED decode steps; saves the fp32 logits of the
           prefill row and of every step (and a sha256 per row, the
           transfer_trace.record_logits pattern). Run once with
           NOSI_VERIFY_ROUND_SLOTS=0 (the shipped layout, the reference) and
           once with > 0 (the T0 arm: the round region and the mirror are
           allocated, the verify is never called).
  verify   the same trace with NOSI_VERIFY_ROUND_SLOTS > 0 and, at every gate
           step (WARM .. N-1): snapshot -> Path 1 with U = 1 on the step's
           token -> restore -> Path 1 again (determinism) -> restore -> the
           shipped decode step on the same token. Saves, per (document,
           step): the two verify rows, the decode row, the union statistics.
  compare  reads the saved arms of one output directory and prints the gates:
             T0        every decode row of the R > 0 arm torch.equal to the
                       shipped arm's (the layout change is invisible);
             T1-argmax argmax(verify) == argmax(decode) at 100% of positions;
             T1-dlogit max |verify - decode| <= NOSI_VERIFY_TOL (0.25: the
                       measured bf16 envelope, CIS-bias probe job 2175358,
                       where the decode kernel and the prefill trick agreed
                       within 0.19 and the split-KV control moved 0.25);
             T1-det    the two verify calls torch.equal (B1 of spec section 3);
             T1-hyg    the verify arm's decode rows torch.equal to the shipped
                       arm's (the snapshot/restore left nothing behind, so the
                       decode rows T1 compares against are the shipped ones);
             T1-cover  every registered position ran (no RoundOverflow).
           Also reported, never gated: KL(decode || verify) per position, the
           union's new-block count per position (0 at L = 4032, where the
           64-block selection is the whole document; the round region is
           exercised only at a sparse length such as 16128).
           Exit code = number of failed gates.

Alignment: step it feeds forced[it] at position L+it; the decode and the two
verify rows of step it are all predictions of forced[it+1] from the same state.

Environment: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_VERIFY_OUT, NOSI_VERIFY_MODE,
NOSI_VERIFY_TAG, NOSI_VERIFY_L (4032: off the 128 boundary), NOSI_VERIFY_N (12),
NOSI_VERIFY_WARM (4: gate steps are WARM .. N-1, 8 positions), NOSI_VERIFY_DOCS (2),
NOSI_VERIFY_TOL (0.25), NOSI_VERIFY_ROUND_SLOTS (read by the engine at import).
"""
import hashlib
import json
import os
import sys
import time

import torch

MODE = os.environ.get("NOSI_VERIFY_MODE", "compare")
OUT = os.environ.get("NOSI_VERIFY_OUT", "nosi_verify_pilot")
TAG = os.environ.get("NOSI_VERIFY_TAG", MODE)
L = int(os.environ.get("NOSI_VERIFY_L", "4032"))
N = int(os.environ.get("NOSI_VERIFY_N", "12"))
WARM = int(os.environ.get("NOSI_VERIFY_WARM", "4"))
NDOCS = int(os.environ.get("NOSI_VERIFY_DOCS", "2"))
TOL = float(os.environ.get("NOSI_VERIFY_TOL", "0.25"))
os.makedirs(OUT, exist_ok=True)


def die(msg):
    print("[verify_pilot] REFUSED: " + msg, flush=True)
    sys.exit(2)


def check_budget():
    # the gate set is WARM .. N-1; the warm-up step (a shipped decode) must precede any verify
    if not (1 <= WARM < N):
        die("need 1 <= WARM=%d < N=%d (the warm-up decode step precedes every verify)" % (WARM, N))
    # keep the whole trace inside one tail block: no write-back, no rollover, no host-window question
    if (L % 64) + N >= 64:
        die("L%%64 + N = %d >= 64: the trace would fill the tail block; pick L, N with (L %% 64) + N < 64" % ((L % 64) + N))


def sha(row: torch.Tensor) -> str:
    return hashlib.sha256(row.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def load_docs(path):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset = load_dataset("parquet", data_files=os.environ["NOSI_PG19_PARQUET"])["train"]["text"]
    ids, rows = [], []
    for i in range(len(dataset)):
        t = tokenizer(dataset[i], return_tensors="pt").input_ids
        if t.shape[1] < L + N + 1:
            continue
        rows.append(i)
        ids.append(t[0, :L + N + 1])
        if len(rows) == NDOCS:
            break
    assert len(rows) == NDOCS, "only %d documents with >= %d tokens" % (len(rows), L + N + 1)
    return torch.stack(ids), rows


def union_stats(model):
    """n_new per (layer, head, request) of the LAST round of every layer: (Lyr, H, B) int64 on the host."""
    per = [lay._verify_last_round.union.n_new for lay in model.layers]
    return torch.stack(per, 0).cpu()


@torch.inference_mode()
def run_trace(path, ids, with_verify: bool):
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    from nosi import state_snapshot as ss
    from nosi.cache_engine import InfLLMv2Cache
    from nosi.verify.verify_step import RoundOverflow
    R = _ce.VERIFY_ROUND_SLOTS
    if with_verify and R <= 0:
        die("verify mode needs NOSI_VERIFY_ROUND_SLOTS > 0 (the engine read %d at import)" % R)
    if _ce.POOL_BLOCKS != 0:
        die("NOSI_POOL_BLOCKS must be 0 for this pilot")
    print("[pilot] mode=%s tag=%s L=%d N=%d warm=%d docs=%d ROUND_SLOTS=%d KV_BIAS_SCALE=%s ATTN_SPLITS=%s"
          % (MODE, TAG, L, N, WARM, NDOCS, R, _ce.KV_BIAS_SCALE, os.environ.get("NOSI_ATTN_SPLITS", "0")), flush=True)
    model = Llama(model_name=path, device="cuda", offload=True)
    B = ids.shape[0]
    x = ids.to("cuda")
    prompt, forced = x[:, :L], x[:, L:L + N]
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    t0 = time.time()
    logits, position_ids = model.batch_prefill(prompt, cache)
    torch.cuda.synchronize()
    print("[pilot] prefill %d x %d: %.1fs; allocation %s slots (W=%d)" % (
        B, L, time.time() - t0, tuple(cache.layers[0].cache_engine._k_gpu.shape),
        cache.layers[0].cache_engine._k_gpu.shape[1] // 64), flush=True)
    rows = [logits[:, -1, :].float().cpu()]                      # the prefill row predicts forced[0]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    gates = []
    snap = ss.CacheSnapshot(cache) if with_verify else None
    trans = None
    for it in range(N):
        tok = forced[:, it:it + 1]
        if with_verify and it >= WARM:
            if trans is None:
                trans = ss.transient_ids(model)
            snap.take()
            rec = dict(step=it, position=L + it, skipped=0, overflow=None)
            t1 = time.time()
            try:
                lv1 = model.verify_inference(tok, cu, position_ids, cache)       # (B, 1, V)
                torch.cuda.synchronize()
                st1 = union_stats(model)
                snap.restore(); ss.assert_transients_intact(model, trans)
                lv2 = model.verify_inference(tok, cu, position_ids, cache)
                torch.cuda.synchronize()
                st2 = union_stats(model)
                snap.restore(); ss.assert_transients_intact(model, trans)
                rec.update(lv1=lv1[:, 0, :].float().cpu(), lv2=lv2[:, 0, :].float().cpu(),
                           n_new_max=int(st1.max()), n_new_mean=float(st1.float().mean()),
                           n_new_per_layer_max=st1.amax(dim=(1, 2)).tolist(),
                           union_stats_identical=bool(torch.equal(st1, st2)), verify_ms=1e3 * (time.time() - t1) / 2)
            except RoundOverflow as e:
                torch.cuda.synchronize()
                snap.restore(); ss.assert_transients_intact(model, trans)
                rec.update(skipped=1, overflow=str(e), n_new_max=int(e.n_new.max()) if e.n_new is not None else None)
                print("[pilot] step %d: %s" % (it, e), flush=True)
            gates.append(rec)
        lg = model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
        torch.cuda.synchronize()
        rows.append(lg[:, -1, :].float().cpu())                   # predicts forced[it+1]
        if with_verify and it >= WARM:
            gates[-1]["ld"] = rows[-1].clone()
        position_ids = position_ids + 1
    out = torch.stack(rows, dim=1)                                # (B, N+1, V)
    hashes = [[sha(out[b, r]) for r in range(N + 1)] for b in range(B)]
    for b in range(B):
        print("[pilot] doc %d decode-row hashes: %s" % (b, " ".join(hashes[b])), flush=True)
    for rec in gates:
        if not rec["skipped"]:
            d = (rec["lv1"] - rec["ld"]).abs().max().item()
            ag = int((rec["lv1"].argmax(-1) == rec["ld"].argmax(-1)).sum())
            print("[pilot] step %d: argmax agree %d/%d  max|dlogit| %.4f  det %s  n_new max %d mean %.2f  %.0f ms/call"
                  % (rec["step"], ag, B, d, torch.equal(rec["lv1"], rec["lv2"]), rec["n_new_max"], rec["n_new_mean"], rec["verify_ms"]), flush=True)
    payload = dict(mode=MODE, tag=TAG, L=L, N=N, warm=WARM, docs=None, batch=B, round_slots=R,
                   kv_bias_scale=_ce.KV_BIAS_SCALE, attn_splits=int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0),
                   logits=out, hashes=hashes, gates=gates,
                   peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    return payload


def kl_rows(p_logits, q_logits):
    pa = torch.log_softmax(p_logits.double(), -1)
    pb = torch.log_softmax(q_logits.double(), -1)
    return (pa.exp() * (pa - pb)).sum(-1)


def compare() -> int:
    arms = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".pt") and (f.startswith("decode_") or f.startswith("verify_")):
            d = torch.load(os.path.join(OUT, f))
            arms[f[:-3]] = d
    assert arms, "no arms in %s" % OUT
    by_L = {}
    for name, d in arms.items():
        by_L.setdefault(int(d["L"]), []).append((name, d))
    lines, summary, nfail = [], {}, 0

    def gate(key, ok, detail):
        nonlocal nfail
        summary[key] = dict(passed=bool(ok), detail=detail)
        lines.append("| %s | %s | %s |" % (key, "PASS" if ok else "FAIL", detail))
        if not ok:
            nfail += 1

    lines.append("| gate | result | detail |\n|---|---|---|")
    for Lval, group in sorted(by_L.items()):
        refs = [(n, d) for n, d in group if d["mode"] == "decode" and int(d["round_slots"]) == 0]
        if len(refs) != 1:
            gate("ref[L=%d]" % Lval, False, "expected exactly one shipped decode arm (ROUND_SLOTS=0), found %d" % len(refs))
            continue
        ref_name, ref = refs[0]
        B = ref["logits"].shape[0]
        for name, d in group:
            if d["mode"] == "decode" and int(d["round_slots"]) > 0:
                same = torch.equal(d["logits"], ref["logits"])
                mx = float((d["logits"] - ref["logits"]).abs().max())
                gate("T0[L=%d,%s vs %s]" % (Lval, name, ref_name), same,
                     "decode rows torch.equal=%s max|d|=%.4g over %d docs x %d rows (ROUND_SLOTS=%d)" % (same, mx, B, d["N"] + 1, d["round_slots"]))
        for name, d in group:
            if d["mode"] != "verify":
                continue
            same = torch.equal(d["logits"], ref["logits"])
            gate("T1-hyg[L=%d,%s]" % (Lval, name), same,
                 "verify arm's decode rows torch.equal to shipped=%s max|d|=%.4g" % (same, float((d["logits"] - ref["logits"]).abs().max())))
            g = d["gates"]
            expected = d["N"] - d["warm"]
            ran = [r for r in g if not r["skipped"]]
            gate("T1-cover[L=%d,%s]" % (Lval, name), len(ran) == expected and len(g) == expected,
                 "%d of %d registered positions ran; skipped (RoundOverflow) %d" % (len(ran), expected, len(g) - len(ran)))
            agree = mx = det = 0
            tot = 0
            worst = 0.0
            kls = []
            lines.append("\n| L | step | doc | argmax verify/decode | max abs dlogit | KL(decode || verify) | det | n_new max | ms/call |\n|---|---|---|---|---|---|---|---|---|")
            for r in ran:
                lv1, lv2, ld = r["lv1"], r["lv2"], r["ld"]
                dd = torch.equal(lv1, lv2)
                det += int(dd)
                for b in range(B):
                    a1, a2 = int(lv1[b].argmax()), int(ld[b].argmax())
                    m = float((lv1[b] - ld[b]).abs().max())
                    k = float(kl_rows(ld[b], lv1[b]))
                    kls.append(k)
                    worst = max(worst, m)
                    agree += int(a1 == a2)
                    tot += 1
                    lines.append("| %d | %d | %d | %d/%d | %.4f | %.5f | %s | %d | %.0f |" % (
                        Lval, r["step"], b, a1, a2, m, k, dd, r["n_new_max"], r["verify_ms"]))
            lines.append("")
            gate("T1-argmax[L=%d,%s]" % (Lval, name), tot > 0 and agree == tot, "%d/%d positions agree" % (agree, tot))
            gate("T1-dlogit[L=%d,%s]" % (Lval, name), tot > 0 and worst <= TOL, "max |dlogit| %.4f <= tol %.3f; mean KL %.5f" % (worst, TOL, sum(kls) / max(1, len(kls))))
            gate("T1-det[L=%d,%s]" % (Lval, name), len(ran) > 0 and det == len(ran), "%d/%d rounds bit-identical on the second call" % (det, len(ran)))
            nn = [r["n_new_max"] for r in ran]
            lines.append("union: n_new max over positions %s (ROUND_SLOTS=%d); union stats identical across the two calls: %s"
                         % (max(nn) if nn else None, d["round_slots"], all(r.get("union_stats_identical", False) for r in ran)))
    lines.append("\nfailed gates: %d" % nfail)
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(OUT, "compare.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, "compare.json"), "w") as f:
        json.dump(dict(tol=TOL, gates=summary, failed=nfail), f, indent=1)
    return nfail


if __name__ == "__main__":
    if MODE == "compare":
        sys.exit(compare())
    check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    ids, rows = load_docs(path)
    print("[docs] %s  L=%d N=%d" % (rows, L, N), flush=True)
    if MODE == "decode":
        payload = run_trace(path, ids, with_verify=False)
    elif MODE == "verify":
        payload = run_trace(path, ids, with_verify=True)
    else:
        raise SystemExit("unknown NOSI_VERIFY_MODE %r" % MODE)
    payload["docs"] = rows
    fn = os.path.join(OUT, "%s_%s.pt" % (MODE, TAG))
    torch.save(payload, fn)
    print("[pilot] saved %s (peak %.2f GB)" % (fn, payload["peak_gb"]), flush=True)
