"""PAIRED TICK PILOT, stage E2 (RESIDENT tick, no per-layer prefetch) of the
ladder in retroinfer-eval docs/superpowers/specs/2026-09-20-paired-tick.md.
Package: nosi/nosi/paired/ (core.py CPU-provable, tick.py the body,
twin.py the G3 twin); sbatch: retroinfer-eval env/slurm/nosi_paired_pilot.sbatch;
tests: tests/test_paired_core.py (CPU).

Teacher-forced NOSA-8B on PG-19 (the verify_pilot.py pattern): prefill of an
L-token prompt, then the document's own continuation forced in. One process
per arm (the engine reads its layout knobs at import). NOSI_PAIRED_MODE:

  shipped   prefill + N SHIPPED decode steps (decode_inference), CUDA-event
            timed; rows (B, N+1, V) fp32 + sha256 per row. NOSI_ATTN_SPLITS
            is the decode's split count (explicit; the sbatch passes 4).
  twin      steps 0 .. WARM-1 shipped (the warm-up step captures the pooling
            graph), then paired.tick.setup and, per forced step, the TWIN
            (twin.twin_step: V rows only, the shipped engine update, the rows
            kernel at U = 1 over the whole allocation) with the score / fetch /
            attn / rest brackets of verify_trace.py.
  twin2b    OPTIONAL diagnostic (DESIGN.md blocker 5): the same twin with every
            GEMM padded to M = 2B (twin.twin2b_step), so cuBLAS picks the tick's
            kernel; compare tolerates its absence.
  equiv     steps 0 .. WARM-1 shipped, setup, the SEED (S rows for every
            request at position L+WARM: the first draft), then per step t
            the PAIRED TICK (V at L+t on forced[t], S at L+t+1 on its draft)
            with the IN-SITU twin per layer (twin.InSituTwin: every term
            recomputed at the twin's shape from the same inputs; the record
            names the first differing term), accept / reject on forced[t+1]
            (teacher forcing: the token the next V consumes), the sequential
            RESTART for the rejected requests (S rows at L+t+1 on forced[t+1]),
            the per-(layer, head, request) accounts. NOSI_PAIRED_POISON=1:
            the provisional rows are NaN-poisoned after every S write's call
            (G1 / G7): a non-finite logit REFUSES the arm. With
            NOSI_PAIRED_S_OFF=1 (debug): the S rows stay in the GEMMs and the
            attention call but score nothing, write nothing and attend V's
            mask; no seed, no restart -- if V is then bit-exact to the twin,
            the contamination path is in what S touches.
  resident  the same tick loop WITHOUT the in-situ twin: the timing arm
            (tick / restart brackets; the twin and the shipped step are timed
            in their own arms in the same job).
  compare   reads the arms of NOSI_PAIRED_OUT (exit code = failed gates):
            G3[B,arm]      every tick's V logits torch.equal the twin arm's
                           row at the same forced step (splits equal, explicit);
                           on failure the first differing step and the equiv
                           arm's in-situ diagnosis (which term) are printed;
            G3-greedy      the author's gate, exact GREEDY equivalence to the
                           optimized ordinary target: argmax of every tick's V
                           row == argmax of the SHIPPED decode row, 100 %;
            G3-tol         the same rows: max |dlogit| <= 4 x tau_ctrl;
            G3-commit      the per-tick greedy tokens (B, ticks) torch.equal
                           the shipped greedy tokens of the same rows;
            G3-2b          only when G3 vs the U = 1 twin FAILED and a twin2b
                           arm exists: the tick's V logits must then be
                           torch.equal to twin2b (the kernel choice was the
                           only difference); reported otherwise;
            VERDICT lines  one per (B, arm) stating the gates separately;
            decomposition  per bracket of the resident tick vs the twin step:
                           measured delta vs the REGISTERED prediction
                           (predicted_deltas), the residual named per term;
            G2-tol[B]      the twin vs the shipped decode: max |dlogit| <=
                           4 x tau_ctrl over the compared rows (NOSI_VERIFY_TAU_CTRL,
                           0.4844 at 16K, job 2175374; REFUSES to run unset)
                           and argmax agreement, as its own row;
            finite[B,arm]  every paired logit finite (the poison control);
            G8[B,arm]      one committed token per request per tick;
            reported: acceptance a (teacher-forced and production-style),
            greedy agreement of V with the text, restart counts / rows / ms,
            divergence per layer (S's selection last tick vs V's now:
            precision / recall, |S\\V|, |V\\S|), V's residual-miss bytes per
            committed token, tick ms and brackets vs the twin step and the
            shipped step, peak GB per arm.

Alignment: row 0 = the prefill row (predicts forced[0]); row r >= 1 was fed
forced[r-1] at position L+r-1 and predicts forced[r]. The tick at step t feeds
forced[t] to V (row t+1) and commits forced[t+1] (teacher forcing), so the
document must provide N+1 continuation tokens (load_docs takes L+N+2).

BUDGET (guarded after the warm-up step, not assumed): the captured pooling
graph is fixed-shape, so no 16-token compress event may fire inside the run
(tick.compress_budget: the S position is one append ahead of V), and no tail
fill (L % 64 + N < 64); at L = 16128 that is N <= 15 with ticks WARM .. N-1.

Environment: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_PAIRED_OUT, NOSI_PAIRED_MODE,
NOSI_PAIRED_TAG, NOSI_PAIRED_L (16128), NOSI_PAIRED_N (15), NOSI_PAIRED_WARM (4),
NOSI_PAIRED_DOCS (= batch; cycled over the qualifying PG-19 documents),
NOSI_PAIRED_DISTINCT (distinct documents, default = batch), NOSI_VERIFY_TAU_CTRL
(compare), NOSI_ATTN_SPLITS / NOSI_PAIRED_POISON (resolved at dispatch by
paired.tick.config_from_env), NOSI_VERIFY_ROUND_SLOTS / NOSI_POOL_BLOCKS (the
engine, at import).
"""
import hashlib
import json
import math
import os
import sys
import time

import torch

MODE = os.environ.get("NOSI_PAIRED_MODE", "compare")
OUT = os.environ.get("NOSI_PAIRED_OUT", "nosi_paired_pilot")
TAG = os.environ.get("NOSI_PAIRED_TAG", MODE)
L = int(os.environ.get("NOSI_PAIRED_L", "16128"))
N = int(os.environ.get("NOSI_PAIRED_N", "15"))
WARM = int(os.environ.get("NOSI_PAIRED_WARM", "4"))
NDOCS = int(os.environ.get("NOSI_PAIRED_DOCS", "2"))
DISTINCT = int(os.environ.get("NOSI_PAIRED_DISTINCT", "0") or 0) or NDOCS
os.makedirs(OUT, exist_ok=True)

PAIRED_MODES = ("equiv", "resident")
TWIN_MODES = ("twin", "twin2b")


def die(msg):
    print("[paired_pilot] REFUSED: " + msg, flush=True)
    sys.exit(2)


def read_tau_ctrl() -> float:
    """verify_pilot.read_tau_ctrl: the tolerance's one source, no default."""
    raw = os.environ.get("NOSI_VERIFY_TAU_CTRL")
    if raw is None or raw.strip() == "":
        die("NOSI_VERIFY_TAU_CTRL is unset: the twin-vs-shipped row is max |dlogit| <= 4 x tau_ctrl and tau_ctrl is MEASURED "
            "per context by env/slurm/nosi_verify_ctrl.sbatch (0.4844 at L=16128, job 2175374); there is no default number")
    try:
        tau = float(raw)
    except ValueError:
        die("NOSI_VERIFY_TAU_CTRL=%r is not a number" % raw)
    if not (tau > 0) or math.isinf(tau):
        die("NOSI_VERIFY_TAU_CTRL=%r must be a finite positive number" % raw)
    return tau


def dlogit_tol(tau_ctrl: float) -> float:
    if not (tau_ctrl > 0):
        raise ValueError("tau_ctrl must be > 0")
    return 4.0 * tau_ctrl


def check_budget():
    if N < 2 or not (1 <= WARM <= N - 1):
        die("need N >= 2 and 1 <= WARM=%d <= N-1=%d (the warm-up decode step precedes the seed and the first tick)" % (WARM, N - 1))
    if (L % 64) + N >= 64:
        die("L%%64 + N = %d >= 64: a tail fill inside the run coincides with the S row's compress event (DESIGN.md); pick (L %% 64) + N < 64" % ((L % 64) + N))


def sha(row: torch.Tensor) -> str:
    return hashlib.sha256(row.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def load_docs(path):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset = load_dataset("parquet", data_files=os.environ["NOSI_PG19_PARQUET"])["train"]["text"]
    need = L + N + 2
    ids, rows = [], []
    for i in range(len(dataset)):
        t = tokenizer(dataset[i], return_tensors="pt").input_ids
        if t.shape[1] < need:
            continue
        rows.append(i)
        ids.append(t[0, :need])
        if len(rows) == min(NDOCS, DISTINCT):
            break
    distinct = len(rows)
    assert distinct >= 1, "no document with >= %d tokens" % need
    rows = [rows[i % distinct] for i in range(NDOCS)]
    ids = [ids[i % distinct] for i in range(NDOCS)]
    return torch.stack(ids), rows, distinct


# ---------------------------------------------------------------------------
# pure bookkeeping (CPU-testable)
# ---------------------------------------------------------------------------
def compared_rows(warm: int, n: int):
    """Rows every arm has from its own forward: the tick rows t+1 for t = warm .. n-1."""
    return list(range(warm + 1, n + 1))


def row_gate(a: torch.Tensor, b: torch.Tensor, rows) -> dict:
    """a, b (B, N+1, V): per compared row torch.equal, max |d|, argmax agreement."""
    eq, mx, agree = [], [], []
    for r in rows:
        x, y = a[:, r], b[:, r]
        eq.append(bool(torch.equal(x, y)))
        mx.append(float((x - y).abs().max()))
        agree.append(int((x.argmax(-1) == y.argmax(-1)).sum()))
    return dict(rows=list(rows), equal=eq, maxabs=mx, argmax_agree=agree, all_equal=all(eq) if eq else False,
                worst=max(mx) if mx else None, first_bad=next((r for r, e in zip(rows, eq) if not e), None))


def render_timing(rows) -> str:
    out = ["| B | arm | call | n | ms mean | ms min | score | fetch | attn | rest | peak GB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["n"] > 0:
            out.append("| %d | %s | %s | %d | %.1f | %.1f | %s | %s | %s | %s | %.1f |" % (
                r["B"], r["arm"], r["call"], r["n"], r["ms_mean"], r["ms_min"],
                *["%.1f" % r[k] if r.get(k) is not None else "-" for k in ("score", "fetch", "attn", "rest")], r["peak_gb"]))
        else:
            out.append("| %d | %s | %s | 0 | - | - | - | - | - | - | %.1f |" % (r["B"], r["arm"], r["call"], r["peak_gb"]))
    return "\n".join(out)


def predicted_deltas(B: int) -> dict:
    """REGISTERED predictions of the resident tick's brackets against the U = 1
    twin step at the same B (None = no registered number at this B: never
    invented). rest_delta: the GEMM set at M = 2B vs M = B, the M-scan of job
    2175525 (14.59 -> 20.03 ms at B = 128); attn_abs: the rows attention with
    U = 2 on the 64-slot window, 13.7 ms at B = 128 (reuse ledger (ii-a));
    score_delta: the present serial scoring chain, one extra position,
    3.56 + 0.0281 B ms (job 2175382, U = 1 -> 2); fetch_delta: the bias build,
    +1..2 ms per call (reuse ledger), point 1.5."""
    at128 = int(B) == 128
    return dict(rest_delta=(20.03 - 14.59) if at128 else None, attn_abs=13.7 if at128 else None,
                score_delta=3.56 + 0.0281 * int(B), fetch_delta=1.5, fetch_band=(1.0, 2.0),
                source=dict(rest="M-scan job 2175525 (B=128 only)", attn="reuse ledger (ii-a) point 13.7 ms at B=128",
                            score="delta_scoring 3.56 + 0.0281 B (job 2175382)", fetch="bias build +1..2 ms (reuse ledger)"))


def render_decomposition(B: int, twin_row: dict, tick_row: dict) -> str:
    """Per bracket: the twin step's and the tick's measured means, the measured
    delta, the registered predicted delta and the RESIDUAL (measured -
    predicted), so the unexplained part is named per term."""
    pred = predicted_deltas(B)
    out = ["| B=%d bracket | twin ms | tick ms | delta measured | delta predicted | residual | prediction source |" % B,
           "|---|---|---|---|---|---|---|"]
    if not (twin_row.get("n", 0) and tick_row.get("n", 0)):
        out.append("| (no timed twin or tick calls) | - | - | - | - | - | - |")
        return "\n".join(out)
    for name in ("score", "fetch", "attn", "rest"):
        tw, tk = twin_row.get(name), tick_row.get(name)
        if tw is None or tk is None:
            out.append("| %s | - | - | - | - | - | %s |" % (name, pred["source"][name]))
            continue
        delta = tk - tw
        if name == "attn":
            p = (pred["attn_abs"] - tw) if pred["attn_abs"] is not None else None
        else:
            p = pred[name + "_delta"]
        out.append("| %s | %.2f | %.2f | %+.2f | %s | %s | %s |" % (
            name, tw, tk, delta, ("%+.2f" % p) if p is not None else "-", ("%+.2f" % (delta - p)) if p is not None else "-", pred["source"][name]))
    tot_tw, tot_tk = twin_row["ms_mean"], tick_row["ms_mean"]
    out.append("| total | %.2f | %.2f | %+.2f | - | - | sum of the brackets above |" % (tot_tw, tot_tk, tot_tk - tot_tw))
    return "\n".join(out)


def timing_row(B, arm, call, per_call, peak_gb) -> dict:
    """per_call: list of dicts with total_ms and optional score_ms/fetch_ms/attn_ms/rest_ms; the FIRST call is
    excluded (JIT, cuBLAS heuristics for a new M), as verify_pilot.aggregate_cost does."""
    timed = [c for c in per_call if c.get("total_ms") is not None][1:]
    row = dict(B=B, arm=arm, call=call, n=len(timed), peak_gb=peak_gb)
    if timed:
        tot = [c["total_ms"] for c in timed]
        row.update(ms_mean=sum(tot) / len(tot), ms_min=min(tot))
        for k in ("score", "fetch", "attn", "rest"):
            vals = [c.get(k + "_ms") for c in timed]
            row[k] = (sum(vals) / len(vals)) if all(v is not None for v in vals) else None
    return row


# ---------------------------------------------------------------------------
# GPU drivers
# ---------------------------------------------------------------------------
def _setup(path, ids, need_round_slots: bool):
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    from nosi.cache_engine import InfLLMv2Cache
    R = _ce.VERIFY_ROUND_SLOTS
    if need_round_slots and R <= 0:
        die("%s mode needs NOSI_VERIFY_ROUND_SLOTS > 0 (the provisional slot is round slot 0; the engine read %d at import)" % (MODE, R))
    if _ce.POOL_BLOCKS != 0:
        die("NOSI_POOL_BLOCKS must be 0 for this pilot")
    print("[paired_pilot] mode=%s tag=%s L=%d N=%d warm=%d docs=%d distinct<=%d ROUND_SLOTS=%d ATTN_SPLITS=%s POISON=%s"
          % (MODE, TAG, L, N, WARM, NDOCS, DISTINCT, R, os.environ.get("NOSI_ATTN_SPLITS", "unset"), os.environ.get("NOSI_PAIRED_POISON", "0")), flush=True)
    model = Llama(model_name=path, device="cuda", offload=True)
    B = ids.shape[0]
    x = ids.to("cuda")
    prompt, forced = x[:, :L], x[:, L:L + N + 1]                  # N+1 continuation tokens: the last tick commits forced[N]
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    t0 = time.time()
    logits, position_ids = model.batch_prefill(prompt, cache)
    torch.cuda.synchronize()
    eng = cache.layers[0].cache_engine
    W = eng._k_gpu.shape[1] // 64
    alloc_gb = sum(t.numel() * t.element_size() for t in (eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu)) * model.num_layers / 1e9
    print("[paired_pilot] prefill %d x %d: %.1fs; allocation %s (W=%d), K+V+bias over %d layers %.2f GB" % (
        B, L, time.time() - t0, tuple(eng._k_gpu.shape), W, model.num_layers, alloc_gb), flush=True)
    meta = dict(mode=MODE, tag=TAG, L=L, N=N, warm=WARM, batch=B, round_slots=R, W=W, alloc_gb=alloc_gb,
                attn_splits_env=os.environ.get("NOSI_ATTN_SPLITS"), poison=os.environ.get("NOSI_PAIRED_POISON", "0") == "1",
                kv_bias_scale=_ce.KV_BIAS_SCALE)
    return model, cache, logits, position_ids, forced, meta


def _shipped_steps(model, cache, forced, position_ids, cu, upto: int, rows, ev):
    """Steps 0 .. upto-1 through decode_inference (step 0 = the warm-up), each CUDA-event timed."""
    for it in range(upto):
        e0, e1 = ev[it]
        e0.record()
        lg = model.decode_inference(forced[:, it:it + 1], cu, position_ids, cache, warmup=(it == 0))
        e1.record()
        torch.cuda.synchronize()
        rows.append(lg[:, -1, :].float().cpu())
        position_ids = position_ids + 1
    return position_ids


def _budget_or_die(cache):
    from nosi.paired import tick as T
    layer0 = cache.layers[0]
    left = int(layer0.no_compress_k_len)
    # V appends at steps WARM .. N-1 (N - WARM more), the S / restart position is one append ahead: the last check
    # before an append sees len_now + (N - WARM) (the S of the last tick); it must stay below the compress size
    if left + (N - WARM) >= layer0.no_compress_k_cache.shape[1]:
        die("compress budget: no_compress_k_len=%d after the warm-up steps + (N - WARM)=%d reaches the 32-token compress "
            "(the captured pooling graph is fixed-shape; tick.compress_budget); lower N or move L" % (left, N - WARM))
    return T


@torch.inference_mode()
def run_shipped(path, ids):
    model, cache, logits, position_ids, forced, meta = _setup(path, ids, need_round_slots=False)
    B = ids.shape[0]
    rows = [logits[:, -1, :].float().cpu()]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]
    _shipped_steps(model, cache, forced, position_ids, cu, N, rows, ev)
    out = torch.stack(rows, dim=1)
    ms = [ev[it][0].elapsed_time(ev[it][1]) for it in range(N)]
    calls = [dict(label=("shipped", it), total_ms=ms[it]) for it in range(N)]
    print("[paired_pilot] shipped step ms: %s" % " ".join("%.1f" % m for m in ms), flush=True)
    return dict(meta, logits=out, hashes=[[sha(out[b, r]) for r in range(N + 1)] for b in range(B)], calls=calls,
                calls_timed=[c for c in calls if c["label"][1] >= WARM], peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                reserved_gb=torch.cuda.max_memory_reserved() / 1e9)


@torch.inference_mode()
def run_twin(path, ids, pad2b: bool = False):
    from nosi import verify_trace as _vtr
    from nosi.paired import twin as TW
    model, cache, logits, position_ids, forced, meta = _setup(path, ids, need_round_slots=True)
    B = ids.shape[0]
    rows = [logits[:, -1, :].float().cpu()]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(WARM)]
    position_ids = _shipped_steps(model, cache, forced, position_ids, cu, WARM, rows, ev)
    Tk = _budget_or_die(cache)
    cfg = Tk.config_from_env(gemm_pad_rows=(B if pad2b else 0))
    sc = Tk.setup(model, cache, B)
    vt = _vtr.VerifyTrace(model.num_layers)
    step = TW.twin2b_step if pad2b else TW.twin_step
    for it in range(WARM, N):
        vt.label(("twin", it))
        res = step(model, cache, sc, cfg, forced[:, it], position_ids[:, 0], trace=vt)
        torch.cuda.synchronize()
        rows.append(res.logits_v.cpu())
        position_ids = position_ids + 1
    timing = vt.harvest()
    out = torch.stack(rows, dim=1)
    for r in timing:
        print("[paired_pilot] %s step %d: %.1f ms (score %.1f fetch %.1f attn %.1f rest %.1f)" % (MODE, r["label"][1], r["total_ms"], r["score_ms"], r["fetch_ms"], r["attn_ms"], r["rest_ms"]), flush=True)
    return dict(meta, cfg=cfg._asdict(), pad2b=bool(pad2b), logits=out, hashes=[[sha(out[b, r]) for r in range(N + 1)] for b in range(B)],
                calls=timing, peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)


def _cpu_tick_record(rec: dict) -> dict:
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in rec.items()}


@torch.inference_mode()
def run_paired(path, ids, insitu_on: bool):
    from nosi import verify_trace as _vtr
    from nosi.paired import core as C
    from nosi.paired import twin as TW
    model, cache, logits, position_ids, forced, meta = _setup(path, ids, need_round_slots=True)
    B = ids.shape[0]
    rows = [logits[:, -1, :].float().cpu()]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(WARM)]
    position_ids = _shipped_steps(model, cache, forced, position_ids, cu, WARM, rows, ev)
    Tk = _budget_or_die(cache)
    cfg = Tk.config_from_env()
    sc = Tk.setup(model, cache, B)
    vt = _vtr.VerifyTrace(model.num_layers)
    state = C.DraftState(B, device="cuda")
    ledger = C.Ledger()
    pos = position_ids[:, 0].contiguous()          # (B,) the committed position tau = L + WARM
    s_off = bool(cfg.s_off)
    if s_off:
        print("[paired_pilot] NOSI_PAIRED_S_OFF=1: S rows present in the GEMMs / attention only; no seed, no restart", flush=True)
    else:
        # THE SEED: S rows at L+WARM on forced[WARM] -> the draft for L+WARM+1 (spec section 3: the first tick needs an S input)
        vt.label(("seed", WARM))
        seed = Tk.s_rows_forward(model, cache, sc, cfg, forced[:, WARM], pos, sc.req_all, trace=vt)
        torch.cuda.synchronize()
        state.apply_restart(sc.req_all, seed.logits_v.argmax(-1))
        ledger.add_restart(WARM - 1, rows=B, ms=None, layers=seed.accounts)
        if any(p is None for p in sc.prev_sel_s):
            die("the seed left a layer without an S prediction")
    ticks, diags, committed_rows, greedy_rows, non_finite = [], [], [], [], 0
    for t in range(WARM, N):
        v_tok = forced[:, t]
        s_tok = state.s_inputs(v_tok)
        had_draft = (state.draft >= 0).clone()
        insitu = TW.InSituTwin() if insitu_on else None
        vt.label(("tick", t))
        res = Tk.paired_tick(model, cache, sc, cfg, v_tok, s_tok, pos, trace=vt, insitu=insitu)
        torch.cuda.synchronize()
        lv, ls = res.logits_v, res.logits_s
        if not (torch.isfinite(lv).all() and torch.isfinite(ls).all()):
            non_finite += 1
            print("[paired_pilot] tick %d: NON-FINITE logits (V %s, S %s): the poison control fired" % (t, bool(torch.isfinite(lv).all()), bool(torch.isfinite(ls).all())), flush=True)
        rows.append(lv.cpu())
        committed = forced[:, t + 1]                                  # teacher forcing: what the next V consumes
        v_arg = lv.argmax(-1)
        outcome = state.tick_end(committed, ls.argmax(-1))
        committed_rows.append(outcome.committed)
        greedy_rows.append(v_arg)
        greedy_agree = v_arg == committed
        prod_accept = had_draft & (s_tok == v_arg)                    # production-style accept: the draft equals argmax(V)
        n_restart = 0
        if t < N - 1 and not s_off:
            vt.label(("restart", t))
            n_restart, r_accts = Tk.restart_after(model, cache, sc, cfg, state, outcome, committed, pos + 1, trace=vt)
            torch.cuda.synchronize()
            if n_restart:
                ledger.add_restart(t, rows=n_restart, ms=None, layers=r_accts)
        ledger.add_tick(t, res.accounts, outcome.accepted, greedy_agree, n_restart)
        rec = dict(tick=t, accepted=int(outcome.accepted.sum()), had_draft=int(had_draft.sum()), prod_accept=int(prod_accept.sum()),
                   greedy_agree=int(greedy_agree.sum()), n_restart=n_restart, restart_idx=outcome.restart_idx.cpu(),
                   s_hash=sha(ls), v_hash=sha(lv))
        if insitu is not None:
            d = TW.diagnose(res.insitu)
            rec["insitu"] = res.insitu
            rec["diagnosis"] = d
            rec["diagnosis_layers"] = TW.diagnose_layers(res.insitu)
            diags.append(d)
        ticks.append(rec)
        print("[paired_pilot] tick %d: accept %d/%d (prod-style %d/%d)  greedy agree %d/%d  restart rows %d%s" % (
            t, rec["accepted"], B, rec["prod_accept"], B, rec["greedy_agree"], B, n_restart,
            ("  in-situ: %s (%d layers depart)" % (d or "all terms torch.equal", len(rec["diagnosis_layers"]))) if insitu is not None else ""), flush=True)
        if insitu is not None and rec["diagnosis_layers"]:
            for line in rec["diagnosis_layers"][:6]:
                print("[paired_pilot]     " + line, flush=True)
        pos = pos + 1
    if non_finite:
        die("%d ticks produced non-finite logits under POISON=%s: a row read the provisional slot without writing it (G1/G7)" % (non_finite, cfg.poison))
    C.lockstep_check(committed_rows, B)
    timing = vt.harvest()
    for r in timing:
        for rec in ledger.ticks:
            if r["label"] == ("tick", rec["tick"]):
                rec["ms"] = r["total_ms"]
        for rec in ledger.restarts:
            if r["label"] in (("restart", rec["tick"]), ("seed", rec["tick"] + 1)):
                rec["ms"] = r["total_ms"]
    out = torch.stack(rows, dim=1)
    summary = ledger.summary()
    print("[paired_pilot] ticks %d  acceptance %.3f  greedy agree %s/%s  restarts %d (rows %d)  v_miss bytes/token %.0f  prefetch precision %s recall %s"
          % (summary["ticks"], summary["acceptance_rate"] or float("nan"), summary["greedy_agree"], summary["rows"], summary["restarts"], summary["restart_rows"],
             summary.get("v_miss_bytes_per_committed_token", float("nan")), summary.get("prefetch_precision"), summary.get("prefetch_recall")), flush=True)
    for r in timing:
        print("[paired_pilot] %s %d: %.1f ms (score %.1f fetch %.1f attn %.1f rest %.1f)" % (r["label"][0], r["label"][1], r["total_ms"], r["score_ms"], r["fetch_ms"], r["attn_ms"], r["rest_ms"]), flush=True)
    return dict(meta, cfg=cfg._asdict(), insitu=insitu_on, s_off=s_off, logits=out, hashes=[[sha(out[b, r]) for r in range(N + 1)] for b in range(B)],
                greedy_tokens=torch.stack(greedy_rows, 1).cpu(), committed_tokens=torch.stack(committed_rows, 1).cpu(),
                calls=timing, ticks=ticks, ledger_ticks=[_cpu_tick_record(r) for r in ledger.ticks], ledger_restarts=[_cpu_tick_record(r) for r in ledger.restarts],
                summary=summary, diagnoses=diags, peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def compare() -> int:
    tau = read_tau_ctrl()
    tol = dlogit_tol(tau)
    arms = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".pt"):
            arms[f[:-3]] = torch.load(os.path.join(OUT, f), weights_only=False)
    if not arms:
        die("no arms in %s" % OUT)
    by_B = {}
    for name, d in arms.items():
        by_B.setdefault(int(d["batch"]), []).append((name, d))
    lines, summary, nfail, timing_rows, decompositions = [], {}, 0, [], []

    def gate(key, ok, detail):
        nonlocal nfail
        summary[key] = dict(passed=bool(ok), detail=detail)
        lines.append("| %s | %s | %s |" % (key, "PASS" if ok else "FAIL", detail))
        nfail += int(not ok)

    lines.append("tolerance row: max |dlogit| <= 4 x tau_ctrl = 4 x %.4f = %.4f (NOSI_VERIFY_TAU_CTRL)\n" % (tau, tol))
    lines.append("| gate | result | detail |\n|---|---|---|")
    for B, group in sorted(by_B.items()):
        shipped = [d for n, d in group if d["mode"] == "shipped"]
        twins = [d for n, d in group if d["mode"] == "twin"]
        twin2b = [d for n, d in group if d["mode"] == "twin2b"]          # optional diagnostic arm
        paired = [(n, d) for n, d in group if d["mode"] in PAIRED_MODES]
        if len(shipped) != 1 or len(twins) != 1:
            gate("arms[B=%d]" % B, False, "expected one shipped and one twin arm, found %d and %d" % (len(shipped), len(twins)))
            continue
        sh, tw = shipped[0], twins[0]
        rows = compared_rows(int(tw["warm"]), int(tw["N"]))
        g = row_gate(tw["logits"], sh["logits"], rows)
        gate("G2-tol[B=%d,twin vs shipped]" % B, g["worst"] is not None and g["worst"] <= tol,
             "max |dlogit| %.4f <= %.4f over rows %d..%d; argmax agree %d/%d; torch.equal rows %d/%d (different split seams expected)"
             % (g["worst"] or float("nan"), tol, rows[0], rows[-1], sum(g["argmax_agree"]), B * len(rows), sum(g["equal"]), len(rows)))
        timing_rows.append(timing_row(B, "shipped", "decode step", sh["calls_timed"], sh["peak_gb"]))
        twin_timing = timing_row(B, "twin", "twin step", tw["calls"], tw["peak_gb"])
        timing_rows.append(twin_timing)
        t2b = None
        if len(twin2b) == 1 and twin2b[0]["logits"].shape == tw["logits"].shape and twin2b[0]["cfg"]["num_splits"] == tw["cfg"]["num_splits"]:
            t2b = twin2b[0]
            g2 = row_gate(t2b["logits"], tw["logits"], rows)
            lines.append("| report[B=%d,twin2b vs twin] | - | GEMMs at M = 2B vs M = B alone: torch.equal rows %d/%d, max |d| %.4g (the kernel-choice term) |" % (B, sum(g2["equal"]), len(rows), g2["worst"]))
            timing_rows.append(timing_row(B, "twin2b", "twin2b step", t2b["calls"], t2b["peak_gb"]))
        elif twin2b:
            lines.append("| report[B=%d,twin2b] | - | present but not comparable (shape or splits differ): ignored |" % B)
        else:
            lines.append("| report[B=%d,twin2b] | - | absent (optional diagnostic arm) |" % B)
        for name, d in paired:
            if d["warm"] != tw["warm"] or d["N"] != tw["N"] or d["logits"].shape != tw["logits"].shape:
                gate("shape[B=%d,%s]" % (B, name), False, "arm N=%s warm=%s %s vs twin N=%s warm=%s %s" % (d["N"], d["warm"], tuple(d["logits"].shape), tw["N"], tw["warm"], tuple(tw["logits"].shape)))
                continue
            if d["cfg"]["num_splits"] != tw["cfg"]["num_splits"]:
                gate("splits[B=%d,%s]" % (B, name), False, "paired splits %s != twin %s" % (d["cfg"]["num_splits"], tw["cfg"]["num_splits"]))
                continue
            g3 = row_gate(d["logits"], tw["logits"], rows)
            detail = "V logits torch.equal the twin on %d/%d rows (max |d| %.4g)" % (sum(g3["equal"]), len(rows), g3["worst"])
            if not g3["all_equal"]:
                t_bad = g3["first_bad"] - 1
                diag = None
                if d.get("insitu"):
                    trec = next((r for r in d["ticks"] if r["tick"] == t_bad), None)
                    diag = (trec or {}).get("diagnosis")
                    diag = diag or "in-situ: every term torch.equal at tick %d -- the difference is CARRIED from an earlier row (state), not produced here" % t_bad
                    for line in (trec or {}).get("diagnosis_layers", []):
                        lines.append("| diag[B=%d,%s tick %d] | - | %s |" % (B, name, t_bad, line))
                else:
                    diag = "no in-situ record (resident arm); see the equiv arm's diagnosis"
                detail += "; first differing row %d (tick %d): %s" % (g3["first_bad"], t_bad, diag)
            if d.get("s_off"):
                name = name + "[S_OFF]"
            gate("G3[B=%d,%s vs twin]" % (B, name), g3["all_equal"], detail)
            verdict = ["G3-exact(twin)=%s" % ("PASS" if g3["all_equal"] else "FAIL")]
            # the author's gate: exact GREEDY equivalence to the optimized ordinary target (the shipped decode rows)
            gp = row_gate(d["logits"], sh["logits"], rows)
            greedy_ok = sum(gp["argmax_agree"]) == B * len(rows)
            gate("G3-greedy[B=%d,%s vs shipped]" % (B, name), greedy_ok, "argmax agree %d/%d over rows %d..%d" % (sum(gp["argmax_agree"]), B * len(rows), rows[0], rows[-1]))
            tol_ok = gp["worst"] is not None and gp["worst"] <= tol
            gate("G3-tol[B=%d,%s vs shipped]" % (B, name), tol_ok, "max |dlogit| %.4f <= 4 x tau_ctrl %.4f" % (gp["worst"] if gp["worst"] is not None else float("nan"), tol))
            greedy_tokens = d.get("greedy_tokens")
            if greedy_tokens is None:
                greedy_tokens = d["logits"][:, rows].argmax(-1)
            shipped_greedy = sh["logits"][:, rows].argmax(-1)
            commit_ok = tuple(greedy_tokens.shape) == tuple(shipped_greedy.shape) and bool(torch.equal(greedy_tokens.to(shipped_greedy.dtype), shipped_greedy))
            gate("G3-commit[B=%d,%s vs shipped]" % (B, name), commit_ok, "per-tick greedy tokens %s torch.equal the shipped greedy tokens: %s" % (tuple(greedy_tokens.shape), commit_ok))
            verdict += ["G3-greedy(shipped)=%s" % ("PASS" if greedy_ok else "FAIL"), "G3-tol(shipped)=%s" % ("PASS" if tol_ok else "FAIL"),
                        "G3-commit(shipped)=%s" % ("PASS" if commit_ok else "FAIL")]
            if t2b is not None:
                g2b = row_gate(d["logits"], t2b["logits"], rows)
                det2 = "V logits torch.equal twin2b on %d/%d rows (max |d| %.4g)" % (sum(g2b["equal"]), len(rows), g2b["worst"])
                if not g3["all_equal"]:
                    gate("G3-2b[B=%d,%s vs twin2b]" % (B, name), g2b["all_equal"], det2 + " -- G3 vs the U = 1 twin failed: equality here means the kernel choice at M = 2B was the only difference")
                else:
                    lines.append("| report[B=%d,%s vs twin2b] | - | %s |" % (B, name, det2))
                verdict.append("G3-2b(twin2b)=%s" % ("PASS" if g2b["all_equal"] else "FAIL"))
            lines.append("| VERDICT[B=%d,%s] | - | %s |" % (B, name, " ; ".join(verdict)))
            fin = bool(torch.isfinite(d["logits"][:, rows]).all())
            gate("finite[B=%d,%s]" % (B, name), fin, "all logits finite (poison=%s)" % d["poison"])
            s = d["summary"]
            gate("G8[B=%d,%s]" % (B, name), s["ticks"] == len(rows) and s["rows"] == B * len(rows), "%d ticks x %d requests = %d committed tokens" % (s["ticks"], B, s["rows"]))
            lines.append("| report[B=%d,%s acceptance] | - | a = %.3f (teacher-forced), prod-style %d/%d, greedy agree %s/%d, restarts %d rows %d, restart ms %s |" % (
                B, name, s["acceptance_rate"] or float("nan"), sum(r["prod_accept"] for r in d["ticks"]), s["rows"], s["greedy_agree"], s["rows"], s["restarts"], s["restart_rows"],
                ["%.1f" % m for m in s["restart_ms"] if m is not None]))
            lines.append("| report[B=%d,%s bytes] | - | V residual misses %s blocks = %.3g B/token; S selections mean %s; S resident hits mean %s; divergence precision %s recall %s |" % (
                B, name, s.get("v_miss_total"), s.get("v_miss_bytes_per_committed_token", float("nan")), s.get("s_sel_mean_per_layer_head_req"), s.get("s_hit_mean_per_layer_head_req"),
                s.get("prefetch_precision"), s.get("prefetch_recall")))
            if s.get("div_v_not_s_per_layer_mean"):
                lines.append("| report[B=%d,%s divergence per layer] | - | needed-not-predicted %s ; predicted-not-needed %s |" % (
                    B, name, ["%.2f" % x for x in s["div_v_not_s_per_layer_mean"]], ["%.2f" % x for x in s["div_s_not_v_per_layer_mean"]]))
            if d["mode"] == "resident":
                tick_timing = timing_row(B, name, "paired tick", [c for c in d["calls"] if c["label"][0] == "tick"], d["peak_gb"])
                timing_rows.append(tick_timing)
                timing_rows.append(timing_row(B, name, "restart", [c for c in d["calls"] if c["label"][0] == "restart"], d["peak_gb"]))
                decompositions.append("\nresident tick decomposition at B=%d (%s vs the twin step; predictions REGISTERED in predicted_deltas):\n%s" % (B, name, render_decomposition(B, twin_timing, tick_timing)))
            if d.get("diagnoses") is not None:
                bad = [x for x in d["diagnoses"] if x]
                lines.append("| report[B=%d,%s in-situ] | - | %d/%d ticks with every term torch.equal%s |" % (
                    B, name, len(d["diagnoses"]) - len(bad), len(d["diagnoses"]), ("; first: " + bad[0]) if bad else ""))
    lines.append("\ntiming (first call of every kind excluded; the equiv arm is not timed: its in-situ twin adds work):\n" + render_timing(timing_rows))
    lines.extend(decompositions)
    lines.append("\nfailed gates: %d" % nfail)
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(OUT, "compare.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, "compare.json"), "w") as f:
        json.dump(dict(tau_ctrl=tau, tol=tol, gates=summary, timing=timing_rows, failed=nfail), f, indent=1)
    return nfail


if __name__ == "__main__":
    if MODE == "compare":
        sys.exit(compare())
    check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    ids, rows, distinct = load_docs(path)
    print("[docs] %s (%d distinct)  L=%d N=%d" % (rows, distinct, L, N), flush=True)
    if MODE == "shipped":
        payload = run_shipped(path, ids)
    elif MODE in TWIN_MODES:
        payload = run_twin(path, ids, pad2b=(MODE == "twin2b"))
    elif MODE in PAIRED_MODES:
        payload = run_paired(path, ids, insitu_on=(MODE == "equiv"))
    else:
        raise SystemExit("unknown NOSI_PAIRED_MODE %r" % MODE)
    payload["docs"] = rows
    payload["distinct_docs"] = distinct
    fn = os.path.join(OUT, "%s_%s.pt" % (MODE, TAG))
    torch.save(payload, fn)
    print("[paired_pilot] saved %s (peak %.2f GB)" % (fn, payload["peak_gb"]), flush=True)
