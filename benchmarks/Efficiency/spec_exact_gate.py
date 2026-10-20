"""R2 EXACT-VERIFY GATE (R2 verify branch, 2026-10-20). Core: nosi/nosi/spec_loop.py, section EXACT VERIFY
(exact_round, StoreServeHook, rollback_exact; the bitwise conditions E1-E7 are stated there). CPU tests:
benchmarks/Efficiency/tests_cpu/test_spec_exact.py. Holder step: benchmarks/Efficiency/spec_exact_r2_step.sh.

ONE PROCESS, ONE PREFILL (NOSI_VERIFY_ROUND_SLOTS = R > 0, pool off, NOSI_ATTN_SPLITS pinned by the caller):
  shipped   NOSA's shipped greedy decode (through the 64-slot views), TOTAL steps after the prompt, with a re-warm
            at every 64-token block boundary (needs_rewarm, as spec_loop_pilot.py): the reference tokens and the
            reference logits of every step (kept on the GPU in bf16: the logits are F.linear(bf16, bf16).float(),
            so the bf16 copy is lossless; checked once).
  for every K of NOSI_SPEC_K_LIST and every arm of NOSI_EXACT_ARMS (the engine state is restored to the prompt
  between arms):
  tf        TEACHER-FORCED, round mode: the draft's and the verify's inputs are the shipped tokens; the commit is a
            SCHEDULE (spec_loop.tf_commit: c = 1 + r mod U, and c = u_fill on every round whose fill position
            u_fill is >= 1, so every such fill is rolled back once; with the re-warm at every block boundary a
            fill is always the last position of its round) instead of the acceptance, so a round rolls back
            U - c positions. GATE tf-logits: EVERY computed position's fp32 logits torch.equal the shipped step's
            at the same (request, position) -- the rejected positions included, because their inputs are the
            shipped tokens too. GATE tf-control: position 0 of the first round compared with the NEXT shipped
            step must differ (the comparison is not vacuous). GATE tf-coverage: at least one rollback, and one
            rejected fill when a round reached a fill.
  round     free-running greedy, round mode (U positions, ONE decision sync, rollback of the rejected ones).
  early     free-running greedy, early mode (stop after the first position with a rejected draft: <= U - 1
            decision syncs, no rollback).
            GATE <arm>-seq: the committed sequence torch.equal the shipped sequence on every request.
            GATE <arm>-logits: the logits of every COMMITTED position torch.equal the shipped step's.
            GATE <arm>-syncs: round mode 1 decision sync per round; early mode <= U - 1.
  Plain steps (a re-warm, or a 1-position span before a block boundary) are the shipped step and are gated too.
TIMING (CUDA events; the clock from after the warm-up step): per round the draft steps, every verify position,
the rollback; committed tokens per second; the store's served / host window loads and the wire bytes.
OUTPUT: NOSI_SPEC_OUT/spec_exact_<tag>.pt (payload), spec_exact_<tag>.md / .json (gates and tables); the last
stdout line is the unique token SPEC_EXACT_GATE_DONE failed=<n>. Exit code = failed gates (0 = all PASS),
2 = refused before the measurement.

Environment: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_SPEC_OUT, NOSI_SPEC_TAG, NOSI_SPEC_L (16128), NOSI_SPEC_DOCS
(= batch, 64), NOSI_SPEC_DISTINCT (2), NOSI_SPEC_TOTAL_TOKENS (112), NOSI_SPEC_K_LIST ("1 2 3"), NOSI_EXACT_ARMS
("tf round early"); engine knobs read at import: NOSI_VERIFY_ROUND_SLOTS (> 0), NOSI_POOL_BLOCKS (0),
NOSI_ATTN_SPLITS (recorded; the gate script pins 4).
"""
import json
import os
import sys
import time

import torch

OUT = os.environ.get("NOSI_SPEC_OUT", "nosi_spec_exact")
TAG = os.environ.get("NOSI_SPEC_TAG", "gate")
L = int(os.environ.get("NOSI_SPEC_L", "16128"))
NDOCS = int(os.environ.get("NOSI_SPEC_DOCS", "64"))
DISTINCT = int(os.environ.get("NOSI_SPEC_DISTINCT", "2") or 0) or NDOCS
TOTAL = int(os.environ.get("NOSI_SPEC_TOTAL_TOKENS", "112"))
K_LIST = tuple(int(x) for x in os.environ.get("NOSI_SPEC_K_LIST", "1 2 3").split())
ARMS = tuple(os.environ.get("NOSI_EXACT_ARMS", "tf round early").split())


def die(msg):
    print("[spec_exact] REFUSED: " + msg, flush=True)
    sys.exit(2)


# ---------------------------------------------------------------------------
# pure helpers (CPU-tested in tests_cpu/test_spec_exact.py by AST extraction)
# ---------------------------------------------------------------------------
def identity_gate(spec_seq: torch.Tensor, ref_seq: torch.Tensor) -> dict:
    """spec_seq (B, n), ref_seq (B, m) int64 on the CPU: compares the first min(n, m) tokens of every request;
    per mismatching request the first position and the two tokens."""
    if spec_seq.dim() != 2 or ref_seq.dim() != 2 or spec_seq.shape[0] != ref_seq.shape[0]:
        raise ValueError("identity_gate: spec %s ref %s" % (tuple(spec_seq.shape), tuple(ref_seq.shape)))
    n = min(spec_seq.shape[1], ref_seq.shape[1])
    if n < 1:
        raise ValueError("identity_gate: nothing to compare")
    a, b = spec_seq[:, :n].to(torch.int64), ref_seq[:, :n].to(torch.int64)
    eq = a == b
    mism = []
    for r in range(a.shape[0]):
        if bool(eq[r].all()):
            continue
        j = int((~eq[r]).to(torch.int64).argmax())
        mism.append(dict(request=r, position=j, spec_token=int(a[r, j]), ref_token=int(b[r, j])))
    return dict(passed=len(mism) == 0, compared_tokens=n, requests=int(a.shape[0]), mismatches=mism)


def summarize_rounds(rounds: list) -> dict:
    """Per-round records -> the arm's means. A round record: kind ('round' | 'plain'), U, computed, c, draft_ms
    (list), verify_ms (list per position), rollback_ms (float or None), host_syncs. The first exact round is
    excluded from the timing means (JIT / allocator warm-up)."""
    ex = [r for r in rounds if r["kind"] == "round"]
    out = dict(records=len(rounds), exact_rounds=len(ex), plain_steps=sum(1 for r in rounds if r["kind"] == "plain"))
    if not ex:
        return out
    out["U_mean"] = sum(r["U"] for r in ex) / len(ex)
    out["computed_mean"] = sum(r["computed"] for r in ex) / len(ex)
    out["c_mean"] = sum(r["c"] for r in ex) / len(ex)
    out["rollbacks"] = sum(1 for r in ex if r.get("rollback_ms") is not None)
    out["host_syncs_per_round_max"] = max(r["host_syncs"] for r in ex)
    out["host_syncs_per_round_mean"] = sum(r["host_syncs"] for r in ex) / len(ex)
    timed = ex[1:] if len(ex) > 1 else ex

    def mean(v):
        v = [float(x) for x in v if x is not None]
        return sum(v) / len(v) if v else None
    out["draft_ms_per_step"] = mean([d for r in timed for d in r["draft_ms"]])
    out["verify_ms_per_position"] = mean([v for r in timed for v in r["verify_ms"]])
    out["verify_ms_per_round"] = mean([sum(r["verify_ms"]) for r in timed])
    out["rollback_ms_per_round"] = mean([r["rollback_ms"] or 0.0 for r in timed])
    out["round_ms"] = mean([sum(r["draft_ms"]) + sum(r["verify_ms"]) + (r["rollback_ms"] or 0.0) for r in timed])
    out["ms_per_committed_token"] = (sum(sum(r["draft_ms"]) + sum(r["verify_ms"]) + (r["rollback_ms"] or 0.0) for r in timed)
                                     / max(1, sum(r["c"] for r in timed)))
    return out


# ---------------------------------------------------------------------------
# data and setup (as spec_loop_pilot.py)
# ---------------------------------------------------------------------------
def load_docs(path):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset = load_dataset("parquet", data_files=os.environ["NOSI_PG19_PARQUET"])["train"]["text"]
    ids, rows = [], []
    want = min(DISTINCT, NDOCS)
    for i in range(len(dataset)):
        t = tokenizer(dataset[i], return_tensors="pt").input_ids
        if t.shape[1] < L:
            continue
        rows.append(i)
        ids.append(t[0, :L])
        if len(rows) == want:
            break
    distinct = len(rows)
    if distinct < 1:
        die("no document with >= %d tokens" % L)
    rows = [rows[i % distinct] for i in range(NDOCS)]
    ids = [ids[i % distinct] for i in range(NDOCS)]
    return torch.stack(ids), rows, distinct


def needs_rewarm(model, cache) -> bool:
    seq = cache.get_seq_length(0)
    return (not model.has_buffers) or (seq + 1 + model.block_size - 1) // model.block_size > model.pooling_buf_all.shape[-1]


class EventTimer:
    """timer(key, idx) records a CUDA event; ms(a, b, idx) after the final synchronize."""

    def __init__(self):
        self.ev = {}

    def __call__(self, key, idx=0):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ev[(key, int(idx))] = e

    def ms(self, a, b, idx=0):
        ea, eb = self.ev.get((a, idx)), self.ev.get((b, idx))
        return None if ea is None or eb is None else ea.elapsed_time(eb)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_shipped(model, cache, t0, pos0, total):
    """The shipped greedy decode: total steps, the token sequence (B, total + 1) and every step's logits in bf16
    (total, B, V), both on the device. No host sync inside the loop."""
    B = t0.shape[0]
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    tok, pos = t0, pos0.clone()
    seq = [t0.clone()]
    logits = None
    evs, rewarms = [], []
    model.has_buffers = False
    t_warm = None
    for it in range(total):
        if needs_rewarm(model, cache):
            model.has_buffers = False
            rewarms.append(it)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        lg = model.decode_inference(tok, cu, pos, cache)
        e1.record()
        row = lg[:, -1, :]
        if logits is None:
            logits = torch.empty((total, B, row.shape[-1]), dtype=torch.bfloat16, device=row.device)
            if not torch.equal(row.to(torch.bfloat16).float(), row):          # once, before the clock
                die("the step logits are not bf16-representable: the bf16 reference copy would be lossy")
        logits[it].copy_(row)
        tok = row.argmax(-1, keepdim=True)
        seq.append(tok)
        pos = pos + 1
        evs.append((e0, e1))
        if it == 0:
            torch.cuda.synchronize()
            t_warm = time.time()
    torch.cuda.synchronize()
    wall = time.time() - t_warm
    step_ms = [a.elapsed_time(b) for a, b in evs]
    steady = [m for i, m in enumerate(step_ms) if i > 0 and i not in rewarms]
    out = dict(seq=torch.cat(seq, 1), logits=logits, step_ms=step_ms, rewarm_steps=rewarms,
               step_ms_steady_mean=(sum(steady) / len(steady)) if steady else None,
               wall_s_after_warmup=wall, tok_s_after_warmup=(total - 1) * B / wall if wall > 0 else None)
    print("[spec_exact] shipped: %d steps, steady step %.1f ms, re-warm at %s, %.1f tok/s after warm-up"
          % (total, out["step_ms_steady_mean"] or float("nan"), rewarms, out["tok_s_after_warmup"] or float("nan")), flush=True)
    return out


@torch.inference_mode()
def run_exact_arm(model, cache, t0, pos0, K, total, arm, ship):
    from nosi import spec_loop as _sl
    from nosi import avail_policy as _avail
    from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
    from nosi.flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
    mode = "early" if arm == "early" else "round"
    B = t0.shape[0]
    eng0 = cache.layers[0].cache_engine
    H, bs = eng0.head_num, eng0.block_size
    layout = _sl.store_layout(eng0.topk, eng0.verify_round_slots)
    store = _sl.pin_window(_sl.Store(model.num_layers, H, B, layout, device="cuda"))
    ctx = _sl.SpecContext(store, side=torch.cuda.Stream(), make_event=torch.cuda.Event,
                          current_stream=torch.cuda.current_stream, stream_ctx=torch.cuda.stream,
                          num_layers=model.num_layers, block_size=bs,
                          kernel_size=model.layers[0].pooling_block_size, kernel_stride=model.layers[0].pooling_stride)
    ops = _sl.GatherOps(gather_k=flash_h2d_from_mask, gather_v=flash_h2d_from_mask_bias)
    hook = _sl.StoreServeHook([lay.cache_engine for lay in cache.layers], store, ops)
    marks = _sl.RoundMarks(model.num_layers, K) if mode == "round" else None
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    ship_seq, ship_lg = ship["seq"], ship["logits"]
    cmp = []          # (n, u, any-differs device bool, max |d| device scalar, kind)
    control = []

    def compare(row, n, u, kind):
        """On the device, no host sync: does the row differ from the shipped step whose input is token n, and by
        how much. tf arm: the first round's position 0 is also compared with the NEXT shipped step (control)."""
        ref = ship_lg[n].float()
        cmp.append((n, u, (row != ref).any(), (row - ref).abs().max(), kind))
        if arm == "tf" and kind == "round" and u == 0 and not control and n + 1 < total:
            control.append((row != ship_lg[n + 1].float()).any())

    clock = {}

    def after_warmup():
        win_ok = torch.stack([(lay.cache_engine._block_map[..., :eng0.topk - 1] >= 0).all() for lay in cache.layers]).all()
        if not bool(win_ok):                                                   # once, before the clock
            die("precondition (a): a window slot is empty after the warm-up step")
        torch.cuda.synchronize()
        clock["t_warm"] = time.time()

    out_loop = _sl.exact_loop(model, cache, ctx, hook, _avail, marks, ops, t0, pos0, cu, K, total, arm,
                              ref_seq=ship_seq if arm == "tf" else None, on_logits=compare,
                              timer_factory=EventTimer, after_warmup=after_warmup)
    torch.cuda.synchronize()
    wall = time.time() - clock["t_warm"]
    n = out_loop["n"]
    committed = out_loop["committed"]
    # harvest (after the clock)
    recs = []
    for rec in out_loop["rounds"]:
        tm = rec.pop("timer")
        if rec["kind"] == "plain":
            rec["step_ms"] = tm.ms("plain_begin", "plain_end")
        else:
            K_r = rec["U"] - 1
            rec["draft_ms"] = [tm.ms("draft_begin", "draft_end", j) for j in range(K_r)]
            rec["verify_ms"] = [tm.ms("verify_begin", "verify_end", u) for u in range(rec["computed"])]
            rec["rollback_ms"] = tm.ms("rollback_begin", "rollback_end", 0)
            rec["n_acc"] = rec["n_acc"].cpu().tolist()
        recs.append(rec)
    mism = [dict(n=a, u=b, kind=k, max_abs=float(m)) for a, b, d, m, k in cmp if bool(d)]
    counters = {k: int(v.sum()) for k, v in store.counters().items()}
    seq = torch.cat(committed, 1).cpu()
    out = dict(K=K, arm=arm, mode=mode, tokens=n, seq=seq, rounds=recs, compared_positions=len(cmp),
               logit_mismatches=mism, control_differs=[bool(x) for x in control],
               counters=counters, wire=_sl.wire_bytes_exact(counters), served_fraction=_sl.served_fraction(counters),
               host_syncs=ctx.host_syncs, wall_s_after_warmup=wall,
               tok_s_after_warmup=(n - 1) * B / wall if wall > 0 else None,
               peak_gb=torch.cuda.max_memory_allocated() / 1e9, W=layout.W, R=layout.R,
               summary=summarize_rounds(recs))
    s = out["summary"]
    print("[spec_exact] K=%d arm=%s: %d tokens, %d exact rounds + %d plain steps, c mean %s, verify %s ms/position, "
          "round %s ms, %s ms/committed token, %.1f tok/s, logits compared %d mismatches %d, served %s, syncs %d, peak %.1f GB"
          % (K, arm, n, s.get("exact_rounds", 0), s.get("plain_steps", 0), _f(s.get("c_mean"), "%.3f"),
             _f(s.get("verify_ms_per_position")), _f(s.get("round_ms")), _f(s.get("ms_per_committed_token")),
             out["tok_s_after_warmup"] or float("nan"), len(cmp), len(mism), _f(out["served_fraction"], "%.3f"),
             ctx.host_syncs, out["peak_gb"]), flush=True)
    return out


def _f(v, fmt="%.1f"):
    return "-" if v is None else (fmt % v)


# ---------------------------------------------------------------------------
# gates and report
# ---------------------------------------------------------------------------
def gates_for(cell: dict, ship_seq_cpu: torch.Tensor) -> list:
    """(name, passed, detail) for one arm."""
    out = []
    K, arm = cell["K"], cell["arm"]
    tag = "%s[K%d]" % (arm, K)
    n_cmp, mism = cell["compared_positions"], cell["logit_mismatches"]
    out.append(("%s-logits" % tag, n_cmp > 0 and not mism,
                "%d positions compared, %d differ%s" % (n_cmp, len(mism), ("; first %r" % (mism[0],)) if mism else "")))
    ex = [r for r in cell["rounds"] if r["kind"] == "round"]
    if arm == "tf":
        ctl = cell["control_differs"]
        out.append(("%s-control" % tag, bool(ctl) and all(ctl), "next-step comparison differs: %s" % ctl))
        nrb = sum(1 for r in ex if r.get("rollback") is not None)
        fills = [r for r in ex if r.get("u_fill") is not None and r["u_fill"] >= 1]
        rej = [r for r in ex if (r.get("rollback") or {}).get("undo_fill")]
        ok = nrb > 0 and (not fills or bool(rej))
        out.append(("%s-coverage" % tag, ok, "%d rollbacks; %d rounds reached a fill at u >= 1, %d rejected fills rolled back "
                    "(tail rows restored: %s)" % (nrb, len(fills), len(rej), [r["rollback"]["rows_restored"] for r in rej])))
    else:
        g = identity_gate(cell["seq"], ship_seq_cpu)
        out.append(("%s-seq" % tag, g["passed"], "%d requests x %d tokens, %d mismatching requests%s" % (
            g["requests"], g["compared_tokens"], len(g["mismatches"]), ("; first %r" % (g["mismatches"][0],)) if g["mismatches"] else "")))
        if arm == "round":
            ok = all(r["host_syncs"] == 1 for r in ex)
            out.append(("%s-syncs" % tag, ok, "decision syncs per round: %s" % sorted(set(r["host_syncs"] for r in ex))))
        else:
            ok = all(r["host_syncs"] <= r["U"] - 1 for r in ex)
            out.append(("%s-syncs" % tag, ok, "decision syncs per round (<= U - 1): %s" % sorted(set((r["host_syncs"], r["U"]) for r in ex))))
    return out


def report(payload: dict) -> int:
    ship = payload["shipped"]
    lines = ["# R2 exact-verify gate (%s)\n" % TAG,
             "L=%d batch=%d distinct=%d total=%d K=%s arms=%s W=%s engine knobs %s\n" % (
                 payload["L"], payload["batch"], payload["distinct_docs"], payload["total"], list(K_LIST), list(ARMS),
                 payload["W"], payload["engine_knobs"]),
             "| gate | result | detail |", "|---|---|---|"]
    nfail = 0
    gates = []
    for cell in payload["cells"]:
        for name, ok, detail in gates_for(cell, ship["seq_cpu"]):
            gates.append(dict(gate=name, passed=bool(ok), detail=detail))
            lines.append("| %s | %s | %s |" % (name, "PASS" if ok else "FAIL", detail))
            nfail += 0 if ok else 1
    lines.append("\n## round time (CUDA events, means over the exact rounds after the first; shipped steady step %s ms)\n"
                 % _f(ship["step_ms_steady_mean"]))
    lines.append("| arm | K | exact rounds | plain steps | U mean | computed mean | c mean | rollbacks | draft ms/step | "
                 "verify ms/position | verify ms/round | rollback ms/round | round ms | ms/committed token | tok/s | served "
                 "fraction | wire MB/token | syncs/round (mean, max) | peak GB |")
    lines.append("|" + "---|" * 20)
    for cell in payload["cells"]:
        s = cell["summary"]
        ntok = max(1, (cell["tokens"] - 1) * payload["batch"])
        lines.append("| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.3f | %s, %s | %.1f |" % (
            cell["arm"], cell["K"], s.get("exact_rounds", 0), s.get("plain_steps", 0), _f(s.get("U_mean"), "%.2f"),
            _f(s.get("computed_mean"), "%.2f"), _f(s.get("c_mean"), "%.3f"), s.get("rollbacks", "-"),
            _f(s.get("draft_ms_per_step")), _f(s.get("verify_ms_per_position")), _f(s.get("verify_ms_per_round")),
            _f(s.get("rollback_ms_per_round")), _f(s.get("round_ms")), _f(s.get("ms_per_committed_token")),
            _f(cell["tok_s_after_warmup"]), _f(cell["served_fraction"], "%.3f"),
            cell["wire"]["wire_bytes"] / ntok / 1e6, _f(s.get("host_syncs_per_round_mean"), "%.2f"),
            s.get("host_syncs_per_round_max", "-"), cell["peak_gb"]))
    lines.append("\nshipped: %d steps, steady step %s ms, %s tok/s after warm-up, re-warm at %s"
                 % (len(ship["step_ms"]), _f(ship["step_ms_steady_mean"]), _f(ship["tok_s_after_warmup"]), ship["rewarm_steps"]))
    lines.append("\nfailed gates: %d" % nfail)
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(OUT, "spec_exact_%s.md" % TAG), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, "spec_exact_%s.json" % TAG), "w") as f:
        json.dump(dict(tag=TAG, failed=nfail, gates=gates,
                       cells=[dict(K=c["K"], arm=c["arm"], summary=c["summary"], counters=c["counters"], wire=c["wire"],
                                   served_fraction=c["served_fraction"], tok_s_after_warmup=c["tok_s_after_warmup"],
                                   compared_positions=c["compared_positions"], logit_mismatches=c["logit_mismatches"],
                                   host_syncs=c["host_syncs"], peak_gb=c["peak_gb"]) for c in payload["cells"]],
                       shipped=dict(step_ms_steady_mean=ship["step_ms_steady_mean"], tok_s_after_warmup=ship["tok_s_after_warmup"],
                                    rewarm_steps=ship["rewarm_steps"]),
                       meta={k: payload[k] for k in ("L", "batch", "distinct_docs", "total", "W", "engine_knobs", "docs")}),
                  f, indent=1, default=str)
    return nfail


def main() -> int:
    from nosi import cache_engine as _ce
    if _ce.VERIFY_ROUND_SLOTS <= 0 or _ce.POOL_BLOCKS != 0:
        die("needs NOSI_VERIFY_ROUND_SLOTS > 0 and NOSI_POOL_BLOCKS = 0 (engine read R=%d P=%d)" % (_ce.VERIFY_ROUND_SLOTS, _ce.POOL_BLOCKS))
    if _ce.KV_BIAS_SCALE != 1.0 or os.environ.get("NOSI_AVAIL", "") not in ("", "0", "off"):
        die("run at the shipped settings: NOSI_KV_BIAS_SCALE 1 and NOSI_AVAIL unset")
    if not K_LIST or min(K_LIST) < 1 or max(K_LIST) >= 64:
        die("NOSI_SPEC_K_LIST %r: every K in [1, 63]" % (K_LIST,))
    if not ARMS or any(a not in ("tf", "round", "early") for a in ARMS):
        die("NOSI_EXACT_ARMS %r: tf / round / early" % (ARMS,))
    if TOTAL < 2:
        die("NOSI_SPEC_TOTAL_TOKENS >= 2")
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, "spec_exact_%s.pt" % TAG)
    if os.path.exists(fn):
        die("%s exists" % fn)
    path = os.environ["NOSI_MODEL_PATH"]
    ids, rows, distinct = load_docs(path)
    print("[spec_exact] docs %s (%d distinct) L=%d K=%s arms=%s total=%d R=%d splits=%s" % (
        rows, distinct, L, K_LIST, ARMS, TOTAL, _ce.VERIFY_ROUND_SLOTS, os.environ.get("NOSI_ATTN_SPLITS", "0")), flush=True)
    from nosi import NOSALlama as Llama
    from nosi import state_snapshot as ss
    from nosi.cache_engine import InfLLMv2Cache
    model = Llama(model_name=path, device="cuda", offload=True)
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    t_pf = time.time()
    logits, position_ids = model.batch_prefill(ids.to("cuda"), cache)
    torch.cuda.synchronize()
    print("[spec_exact] prefill %.1f s, peak %.1f GB" % (time.time() - t_pf, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    t0 = logits[:, -1, :].argmax(-1, keepdim=True)
    pos0 = position_ids[:, -1:] + 1
    snap = ss.PostPrefillSnapshot(cache).take()
    eng0 = cache.layers[0].cache_engine
    ship = run_shipped(model, cache, t0, pos0, TOTAL)
    ship["seq_cpu"] = ship["seq"].cpu()
    cells = []
    for K in K_LIST:
        for arm in ARMS:
            snap.restore()
            torch.cuda.reset_peak_memory_stats()
            cells.append(run_exact_arm(model, cache, t0, pos0, K, TOTAL, arm, ship))
    payload = dict(L=L, batch=NDOCS, distinct_docs=distinct, docs=rows, total=TOTAL, W=eng0._k_gpu.shape[1] // eng0.block_size,
                   engine_knobs=dict(NOSI_VERIFY_ROUND_SLOTS=_ce.VERIFY_ROUND_SLOTS, NOSI_POOL_BLOCKS=_ce.POOL_BLOCKS,
                                     NOSI_ATTN_SPLITS=int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0),
                                     NOSI_KV_BIAS_SCALE=_ce.KV_BIAS_SCALE),
                   shipped={k: v for k, v in ship.items() if k not in ("logits", "seq")}, cells=cells)
    torch.save(payload, fn)
    nfail = report(payload)
    print("[spec_exact] saved %s" % fn, flush=True)
    print("SPEC_EXACT_GATE_DONE failed=%d" % nfail, flush=True)
    return min(nfail, 99)


if __name__ == "__main__":
    sys.exit(main())
