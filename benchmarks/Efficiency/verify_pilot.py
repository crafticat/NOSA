"""GATES T0 / T1 (U = 1), G-MPV-2 (U = 2, 3, 5) and the VERIFY COST of the exact
multi-position verifier (Path 1) (retroinfer-eval fork; spec
docs/superpowers/specs/2026-09-19-multiposition-verify-path1.md, sections 3, 4, 5(d)).

Teacher-forced NOSA-8B on PG-19, the pattern of trace_selections.py /
bias_convention_probe.py: prefill of an L-token prompt, then N decode steps fed
the document's own continuation. Process modes (NOSI reads its knobs at
import, so each layout is its own process; NOSI_VERIFY_MODE):

  decode    prefill + N SHIPPED decode steps; saves the fp32 logits of the
            prefill row and of every step (and a sha256 per row). Run once with
            NOSI_VERIFY_ROUND_SLOTS=0 (the shipped layout, the reference) and
            once with > 0 (the T0 arm: the round region and the mirror are
            allocated, the verify is never called).
  verify    the same trace with NOSI_VERIFY_ROUND_SLOTS > 0 and, at every gate
            step (WARM .. N-1): snapshot -> Path 1 with U = 1 on the step's
            token -> restore -> Path 1 again (determinism) -> restore -> the
            shipped decode step on the same token.
  verify_u  G-MPV-2: the same with U = NOSI_VERIFY_U positions per round (K + 1
            for K = 1, 2, 4 -> U = 2, 3, 5). At every gate step t (WARM .. N-U):
            snapshot -> Path 1 on the U forced tokens forced[t : t+U] (the
            teacher-forced draft = the true continuation), the U per-position
            logits recorded -> restore -> Path 1 again -> restore -> the shipped
            decode advanced by ONE step on forced[t], as today.
  compare   reads the saved arms of one output directory and prints the gates
            (exit code = failed gates):
              T0            every decode row of an R > 0 decode arm torch.equal
                            to the shipped arm's (the layout change is invisible);
              *-hyg         the verify arm's own decode rows torch.equal to the
                            shipped arm's (snapshot/restore left nothing behind,
                            so the decode rows the verify is compared against
                            are the shipped ones);
              *-cover       every registered round ran (no RoundOverflow);
              *-argmax[u]   argmax(verify position u) == argmax(decode row) at
                            100% of (round, document) for EVERY u;
              *-dlogit[u]   max |verify - decode| <= 4 x tau_ctrl per u, tau_ctrl
                            = NOSI_VERIFY_TAU_CTRL, MEASURED per context by
                            env/slurm/nosi_verify_ctrl.sbatch (0.4844 at L = 16128,
                            job 2175374: the shipped decode against itself under
                            another legitimate split-KV partition). compare
                            REFUSES to run when the variable is unset: there is no
                            default number (spec section 3, ledger 2026-09-19);
              *-det         the two verify calls torch.equal (spec B1).
            '*' is T1 for the U = 1 arm and MPV2-U<U> for a verify_u arm.
            Reported, never gated: KL(decode || verify) per (t, u), n_new (blocks
            fetched into the round region) and the union size per round, and B2
            (position-0 logits across the arms of one step: the attention CTA
            of position 0 reads the same slots whatever U is, but the qkv/wo/FFN
            GEMMs run at B*U rows and cuBLAS may pick another kernel for another
            M, so bit-identity across U is not promised at the logits).
  cost      VERIFY COST at NOSI_VERIFY_DOCS = batch (64 and 128 are the targets),
            one process per batch, every U of NOSI_VERIFY_U_LIST (space separated)
            in it: prefill, N shipped decode steps timed with CUDA events, and at
            every gate step t, for every U with t <= N-U: a LIGHT snapshot
            (state_snapshot.CounterSnapshot: the counters and the small tables,
            not the allocation -- the full snapshot is ~34 GB at batch 128) ->
            one Path 1 call on forced[t : t+U] timed with the brackets of
            verify_trace.py (score / fetch / attention / rest) -> restore. Per
            (batch, U): ms per call, mean over the gated steps with the FIRST
            call excluded, the shipped decode step ms at the same batch, the
            ratio, union slots, n_new, overflow count, peak memory. With
            NOSI_VERIFY_COST_HYG=1 (batch 64 in the sbatch) one full
            CacheSnapshot at the first gate step proves the light restore:
            the decode logits with and without the rounds in between are
            torch.equal, and the positive control (a round WITHOUT a restore)
            differs.
  cost_table  reads the cost_*.pt of the directory, prints the table
            batch | U | verify ms | decode step ms | ratio | union slots | n_new mean | peak GB
            plus the split, writes cost.md and cost.json (the number
            scripts/nosi_equal_memory_frontier.py::spec_round prices with a
            formula today; the JSON is written so it can replace that formula).
            Exit code = requested cells (NOSI_VERIFY_COST_BATCHES x U_LIST) with
            no timed call + a failed hygiene probe.

Alignment (spec 5d, ledger). The decode arm's tensor is (B, N+1, V): row 0 is
the prefill row (predicts forced[0]); row r >= 1 is decode step r-1, which
was fed forced[r-1] at position L+r-1 and predicts forced[r]. A verify round at
step t feeds forced[t+u] at position L+t+u for u = 0..U-1, so position u of the
round at step t predicts forced[t+u+1] from the same state as decode step t+u,
i.e. it must match decode row t+u+1 (decode_row / expected_rows below).

Environment: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_VERIFY_OUT, NOSI_VERIFY_MODE,
NOSI_VERIFY_TAG, NOSI_VERIFY_L (4032: off the 128 boundary; 16128: the sparse
length where the round region is fetched), NOSI_VERIFY_N (12), NOSI_VERIFY_WARM
(4), NOSI_VERIFY_DOCS (2; = the batch in cost mode, cycled over the qualifying
PG-19 documents when the batch exceeds them), NOSI_VERIFY_U (verify_u),
NOSI_VERIFY_U_LIST ("1 2 3 5", cost), NOSI_VERIFY_TAU_CTRL (compare; no default),
NOSI_VERIFY_COST_HYG (cost), NOSI_VERIFY_COST_BATCHES ("64 128", cost_table),
NOSI_VERIFY_ROUND_SLOTS (read by the engine at import).
"""
import hashlib
import json
import math
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
U = int(os.environ.get("NOSI_VERIFY_U", "1"))                                        # verify_u: positions per round
U_LIST = tuple(int(x) for x in os.environ.get("NOSI_VERIFY_U_LIST", "1 2 3 5").split())   # cost (no commas: apptainer --env splits on them)
COST_HYG = os.environ.get("NOSI_VERIFY_COST_HYG", "0") == "1"
COST_BATCHES = tuple(int(x) for x in os.environ.get("NOSI_VERIFY_COST_BATCHES", "64 128").split())
os.makedirs(OUT, exist_ok=True)


def die(msg):
    print("[verify_pilot] REFUSED: " + msg, flush=True)
    sys.exit(2)


def read_tau_ctrl() -> float:
    """The tolerance's one source. Read at compare time, never at import; no default."""
    raw = os.environ.get("NOSI_VERIFY_TAU_CTRL")
    if raw is None or raw.strip() == "":
        die("NOSI_VERIFY_TAU_CTRL is unset: the dlogit gate is max |dlogit| <= 4 x tau_ctrl and tau_ctrl is "
            "MEASURED per context by env/slurm/nosi_verify_ctrl.sbatch (0.4844 at L=16128, job 2175374); "
            "there is no default number (spec 2026-09-19 section 3)")
    try:
        tau = float(raw)
    except ValueError:
        die("NOSI_VERIFY_TAU_CTRL=%r is not a number" % raw)
    if not (tau > 0) or math.isinf(tau):
        die("NOSI_VERIFY_TAU_CTRL=%r must be a finite positive number" % raw)
    return tau


def check_budget():
    # the gate set is WARM .. N-U; the warm-up step (a shipped decode) must precede any verify
    u_max = max(U_LIST) if MODE == "cost" else U
    if u_max < 1:
        die("U must be >= 1")
    if not (1 <= WARM <= N - u_max):
        die("need 1 <= WARM=%d <= N-U=%d (the warm-up decode step precedes every verify; a round of U=%d at step t needs decode rows up to t+U <= N)" % (WARM, N - u_max, u_max))
    # keep the whole trace inside one tail block: no write-back, no rollover, no host-window question
    # (a round at step t <= N-U writes tail rows up to (L % 64) + t + U - 1 <= (L % 64) + N - 1)
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
    distinct = len(rows)
    assert distinct >= 1, "no document with >= %d tokens" % (L + N + 1)
    if distinct < NDOCS:
        # only the cost mode may cycle: a batch of 64/128 requests from PG-19 test's qualifying
        # documents (the sweeps repeated ONE document B times, test_nosa_pg19.py:142; a cycle of
        # distinct documents keeps the per-request selections different). A gate arm must not.
        if MODE != "cost":
            die("only %d documents with >= %d tokens, %d requested (cycling is allowed in cost mode only)" % (distinct, L + N + 1, NDOCS))
        rows = [rows[i % distinct] for i in range(NDOCS)]
        ids = [ids[i % distinct] for i in range(NDOCS)]
    return torch.stack(ids), rows, distinct


# ---------------------------------------------------------------------------
# pure bookkeeping (CPU-tested in retroinfer-eval tests/test_nosi_verify_u.py)
# ---------------------------------------------------------------------------
def decode_row(t: int, u: int) -> int:
    """The decode arm's row that position u of the round at step t must match."""
    return t + u + 1


def expected_rows(decode_logits: torch.Tensor, t: int, U: int) -> torch.Tensor:
    """(B, N+1, V) decode rows -> (B, U, V): rows decode_row(t, u), u = 0..U-1."""
    if decode_logits.dim() != 3:
        raise ValueError("decode logits must be (B, N+1, V), got %s" % (tuple(decode_logits.shape),))
    n = decode_logits.shape[1] - 1
    if U < 1 or t < 0 or t + U > n:
        raise ValueError("round at step t=%d with U=%d needs decode rows up to %d but the arm has N=%d steps" % (t, U, decode_row(t, U - 1), n))
    return decode_logits[:, decode_row(t, 0):decode_row(t, U - 1) + 1, :]


def as_rounds(lv: torch.Tensor) -> torch.Tensor:
    """A saved verify record: (B, V) from a U = 1 arm or (B, U, V). Always (B, U, V) here."""
    return lv.unsqueeze(1) if lv.dim() == 2 else lv


def kl_rows(p_logits, q_logits):
    pa = torch.log_softmax(p_logits.double(), -1)
    pb = torch.log_softmax(q_logits.double(), -1)
    return (pa.exp() * (pa - pb)).sum(-1)


def score_round(lv1: torch.Tensor, lv2: torch.Tensor, ld: torch.Tensor) -> dict:
    """lv1, lv2: the two verify calls (B, U, V); ld: the decode rows (B, U, V) of
    expected_rows. Per position u: argmax agreement (B,), max |dlogit| (B,),
    KL(decode || verify) (B,); det = the two calls torch.equal."""
    if not (lv1.shape == lv2.shape == ld.shape) or lv1.dim() != 3:
        raise ValueError("score_round: shapes lv1 %s lv2 %s ld %s must be one (B, U, V)" % (tuple(lv1.shape), tuple(lv2.shape), tuple(ld.shape)))
    per_u = []
    for u in range(lv1.shape[1]):
        a, b = lv1[:, u].argmax(-1), ld[:, u].argmax(-1)
        per_u.append(dict(u=u, agree=(a == b), argmax_verify=a, argmax_decode=b,
                          maxabs=(lv1[:, u] - ld[:, u]).abs().amax(-1), kl=kl_rows(ld[:, u], lv1[:, u])))
    return dict(det=torch.equal(lv1, lv2), per_u=per_u)


def dlogit_tol(tau_ctrl: float) -> float:
    """Spec section 3, rule T2: 4 x tau_ctrl. The one place the factor lives."""
    if not (tau_ctrl > 0):
        raise ValueError("tau_ctrl must be > 0, got %r" % tau_ctrl)
    return 4.0 * tau_ctrl


def gate_verify_arm(name, Lval, d, ref, tau_ctrl, gate, lines, b2_rows):
    """All the gates of one verify / verify_u arm against the shipped decode arm."""
    Uarm = int(d.get("U", 1))
    prefix = "T1" if d["mode"] == "verify" else "MPV2-U%d" % Uarm
    key = "[L=%d,%s]" % (Lval, name)
    B = ref["logits"].shape[0]
    same = torch.equal(d["logits"], ref["logits"])
    gate(prefix + "-hyg" + key, same,
         "verify arm's decode rows torch.equal to shipped=%s max|d|=%.4g" % (same, float((d["logits"] - ref["logits"]).abs().max())))
    g = d["gates"]
    expected = d["N"] - d["warm"] - Uarm + 1
    ran = [r for r in g if not r["skipped"]]
    gate(prefix + "-cover" + key, len(ran) == expected and len(g) == expected,
         "%d of %d registered rounds ran; skipped (RoundOverflow) %d" % (len(ran), expected, len(g) - len(ran)))
    agree = [0] * Uarm
    tot = [0] * Uarm
    worst = [0.0] * Uarm
    kls = [[] for _ in range(Uarm)]
    det = 0
    lines.append("\n| L | U | step | u | decode row | doc | argmax verify/decode | max abs dlogit | KL(decode || verify) | det | n_new max | union slots max | ms/call |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ran:
        lv1, lv2 = as_rounds(r["lv1"]), as_rounds(r["lv2"])
        ld = expected_rows(ref["logits"], int(r["step"]), Uarm)
        sc = score_round(lv1, lv2, ld)
        det += int(sc["det"])
        b2_rows.setdefault((Lval, int(r["step"])), {})[name] = lv1[:, 0, :]
        for pu in sc["per_u"]:
            u = pu["u"]
            for b in range(B):
                m = float(pu["maxabs"][b])
                worst[u] = max(worst[u], m)
                kls[u].append(float(pu["kl"][b]))
                agree[u] += int(pu["agree"][b])
                tot[u] += 1
                lines.append("| %d | %d | %d | %d | %d | %d | %d/%d | %.4f | %.5f | %s | %d | %s | %.0f |" % (
                    Lval, Uarm, r["step"], u, decode_row(int(r["step"]), u), b, int(pu["argmax_verify"][b]), int(pu["argmax_decode"][b]),
                    m, float(pu["kl"][b]), sc["det"], r["n_new_max"], r.get("union_max", "-"), r["verify_ms"]))
    lines.append("")
    tol = dlogit_tol(tau_ctrl)
    for u in range(Uarm):
        suffix = "" if Uarm == 1 else "[u=%d]" % u
        gate(prefix + "-argmax" + key + suffix, tot[u] > 0 and agree[u] == tot[u], "%d/%d positions agree" % (agree[u], tot[u]))
        gate(prefix + "-dlogit" + key + suffix, tot[u] > 0 and worst[u] <= tol,
             "max |dlogit| %.4f <= 4 x tau_ctrl %.4f = %.4f; mean KL %.5f" % (worst[u], tau_ctrl, tol, sum(kls[u]) / max(1, len(kls[u]))))
    gate(prefix + "-det" + key, len(ran) > 0 and det == len(ran), "%d/%d rounds bit-identical on the second call" % (det, len(ran)))
    nn = [r["n_new_max"] for r in ran]
    nm = [r["n_new_mean"] for r in ran if r.get("n_new_mean") is not None]
    us = [r["union_max"] for r in ran if r.get("union_max") is not None]
    lines.append("union[%s]: n_new max over rounds %s, n_new mean over rounds %s, union slots max %s (ROUND_SLOTS=%d, U=%d); union stats identical across the two calls: %s"
                 % (name, max(nn) if nn else None, ("%.2f" % (sum(nm) / len(nm))) if nm else None, max(us) if us else None,
                    d["round_slots"], Uarm, all(r.get("union_stats_identical", False) for r in ran)))
    return dict(worst=worst, agree=agree, tot=tot)


def render_cost_table(rows) -> str:
    """rows: dicts with batch, U, verify_ms, decode_step_ms, union_slots_mean, n_new_mean,
    peak_gb (+ score_ms, fetch_ms, attn_ms, rest_ms, n_calls, overflow, union_slots_max,
    n_new_max, W). A row with n_calls == 0 renders '-' in every timed column: an empty
    measurement is visible, never a number."""
    out = ["| batch | U | verify ms | decode step ms | ratio | union slots | n_new mean | peak GB |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (int(r["batch"]), int(r["U"]))):
        if int(r.get("n_calls", 0)) > 0:
            out.append("| %d | %d | %.1f | %.1f | %.2f | %.1f | %.2f | %.1f |" % (
                r["batch"], r["U"], r["verify_ms"], r["decode_step_ms"], r["verify_ms"] / r["decode_step_ms"],
                r["union_slots_mean"], r["n_new_mean"], r["peak_gb"]))
        else:
            out.append("| %d | %d | - | %.1f | - | - | - | %.1f |" % (r["batch"], r["U"], r["decode_step_ms"], r["peak_gb"]))
    out += ["", "| batch | U | calls timed | score ms | fetch ms | attn ms | rest ms | union slots max | n_new max | overflow rounds | W slots |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (int(r["batch"]), int(r["U"]))):
        if int(r.get("n_calls", 0)) > 0:
            out.append("| %d | %d | %d | %.1f | %.1f | %.1f | %.1f | %d | %d | %d | %d |" % (
                r["batch"], r["U"], r["n_calls"], r["score_ms"], r["fetch_ms"], r["attn_ms"], r["rest_ms"],
                r["union_slots_max"], r["n_new_max"], r["overflow"], r["W"]))
        else:
            out.append("| %d | %d | 0 | - | - | - | - | - | - | %d | %d |" % (r["batch"], r["U"], r.get("overflow", 0), r.get("W", 0)))
    return "\n".join(out)


def aggregate_cost(payload) -> list:
    """One cost payload (one batch, every U) -> one row per U; the first call of
    every U is excluded (JIT, cuBLAS heuristics for the new M = B*U)."""
    rows = []
    calls = payload["calls"]
    for u in payload["U_list"]:
        mine = [c for c in calls if c["U"] == u]
        timed = [c for c in mine if not c["skipped"] and c.get("total_ms") is not None][1:]
        ov = sum(1 for c in mine if c["skipped"])
        row = dict(batch=payload["batch"], U=u, L=payload["L"], W=payload["W"], round_slots=payload["round_slots"],
                   n_calls=len(timed), overflow=ov, decode_step_ms=payload["decode_step_ms"], peak_gb=payload["peak_gb"],
                   distinct_docs=payload.get("distinct_docs"))
        if timed:
            def mean(k):
                return sum(c[k] for c in timed) / len(timed)
            tot = [c["total_ms"] for c in timed]
            row.update(verify_ms=mean("total_ms"), verify_ms_std=(sum((x - mean("total_ms")) ** 2 for x in tot) / len(tot)) ** 0.5,
                       verify_ms_min=min(tot), score_ms=mean("score_ms"), fetch_ms=mean("fetch_ms"), attn_ms=mean("attn_ms"), rest_ms=mean("rest_ms"),
                       union_slots_mean=mean("union_mean"), union_slots_max=max(c["union_max"] for c in timed),
                       n_new_mean=mean("n_new_mean"), n_new_max=max(c["n_new_max"] for c in timed))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# GPU drivers
# ---------------------------------------------------------------------------
def union_stats(model):
    """Of the LAST round of every layer: n_new and the union size per (layer, head, request), int64 on the host."""
    n_new = torch.stack([lay._verify_last_round.union.n_new for lay in model.layers], 0).cpu()
    size = torch.stack([(lay._verify_last_round.union.union_map >= 0).sum(-1) for lay in model.layers], 0).cpu()
    return n_new, size


def _setup(path, ids, need_round_slots: bool):
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    from nosi.cache_engine import InfLLMv2Cache
    R = _ce.VERIFY_ROUND_SLOTS
    if need_round_slots and R <= 0:
        die("%s mode needs NOSI_VERIFY_ROUND_SLOTS > 0 (the engine read %d at import)" % (MODE, R))
    if _ce.POOL_BLOCKS != 0:
        die("NOSI_POOL_BLOCKS must be 0 for this pilot")
    print("[pilot] mode=%s tag=%s L=%d N=%d warm=%d docs=%d U=%s ROUND_SLOTS=%d KV_BIAS_SCALE=%s ATTN_SPLITS=%s"
          % (MODE, TAG, L, N, WARM, NDOCS, U_LIST if MODE == "cost" else U, R, _ce.KV_BIAS_SCALE, os.environ.get("NOSI_ATTN_SPLITS", "0")), flush=True)
    model = Llama(model_name=path, device="cuda", offload=True)
    B = ids.shape[0]
    x = ids.to("cuda")
    prompt, forced = x[:, :L], x[:, L:L + N]
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    t0 = time.time()
    logits, position_ids = model.batch_prefill(prompt, cache)
    torch.cuda.synchronize()
    eng = cache.layers[0].cache_engine
    W = eng._k_gpu.shape[1] // 64
    alloc_gb = sum(t.numel() * t.element_size() for t in (eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu)) * model.num_layers / 1e9
    print("[pilot] prefill %d x %d: %.1fs; allocation %s slots (W=%d), K+V+bias over %d layers %.2f GB" % (
        B, L, time.time() - t0, tuple(eng._k_gpu.shape), W, model.num_layers, alloc_gb), flush=True)
    meta = dict(mode=MODE, tag=TAG, L=L, N=N, warm=WARM, docs=None, batch=B, round_slots=R, W=W, alloc_gb=alloc_gb,
                kv_bias_scale=_ce.KV_BIAS_SCALE, attn_splits=int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0))
    return model, cache, logits, position_ids, forced, meta


@torch.inference_mode()
def run_trace(path, ids, with_verify: bool, U: int = 1):
    """decode (with_verify=False), verify (U = 1) and verify_u (U > 1) arms."""
    from nosi import state_snapshot as ss
    from nosi.verify.verify_step import RoundOverflow
    model, cache, logits, position_ids, forced, meta = _setup(path, ids, need_round_slots=with_verify)
    B = ids.shape[0]
    rows = [logits[:, -1, :].float().cpu()]                      # the prefill row predicts forced[0]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    ar = torch.arange(U, device="cuda", dtype=position_ids.dtype).unsqueeze(0)   # (1, U): positions L+t .. L+t+U-1
    gates = []
    snap = ss.CacheSnapshot(cache) if with_verify else None
    trans = None
    for it in range(N):
        tok = forced[:, it:it + 1]
        if with_verify and WARM <= it <= N - U:
            if trans is None:
                trans = ss.transient_ids(model)
            toks = forced[:, it:it + U]                            # the round: forced[t : t+U], the true continuation
            pos = position_ids + ar
            snap.take()
            rec = dict(step=it, position=L + it, U=U, skipped=0, overflow=None)
            t1 = time.time()
            try:
                lv1 = model.verify_inference(toks, cu, pos, cache)       # (B, U, V)
                torch.cuda.synchronize()
                n1, s1 = union_stats(model)
                snap.restore(); ss.assert_transients_intact(model, trans)
                lv2 = model.verify_inference(toks, cu, pos, cache)
                torch.cuda.synchronize()
                n2, s2 = union_stats(model)
                snap.restore(); ss.assert_transients_intact(model, trans)
                rec.update(lv1=lv1.float().cpu(), lv2=lv2.float().cpu(),
                           n_new_max=int(n1.max()), n_new_mean=float(n1.float().mean()),
                           n_new_per_layer_max=n1.amax(dim=(1, 2)).tolist(),
                           union_max=int(s1.max()), union_mean=float(s1.float().mean()),
                           union_stats_identical=bool(torch.equal(n1, n2) and torch.equal(s1, s2)),
                           verify_ms=1e3 * (time.time() - t1) / 2)
            except RoundOverflow as e:
                torch.cuda.synchronize()
                snap.restore(); ss.assert_transients_intact(model, trans)
                rec.update(skipped=1, overflow=str(e), n_new_max=int(e.n_new.max()) if e.n_new is not None else None)
                print("[pilot] step %d: %s" % (it, e), flush=True)
            gates.append(rec)
        lg = model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
        torch.cuda.synchronize()
        rows.append(lg[:, -1, :].float().cpu())                   # predicts forced[it+1]
        position_ids = position_ids + 1
    out = torch.stack(rows, dim=1)                                # (B, N+1, V)
    hashes = [[sha(out[b, r]) for r in range(N + 1)] for b in range(B)]
    for b in range(B):
        print("[pilot] doc %d decode-row hashes: %s" % (b, " ".join(hashes[b])), flush=True)
    for rec in gates:
        if not rec["skipped"]:
            sc = score_round(rec["lv1"], rec["lv2"], expected_rows(out, rec["step"], U))
            for pu in sc["per_u"]:
                print("[pilot] step %d u %d (decode row %d): argmax agree %d/%d  max|dlogit| %.4f  det %s  n_new max %d mean %.2f  union max %d  %.0f ms/call"
                      % (rec["step"], pu["u"], decode_row(rec["step"], pu["u"]), int(pu["agree"].sum()), B, float(pu["maxabs"].max()),
                         sc["det"], rec["n_new_max"], rec["n_new_mean"], rec["union_max"], rec["verify_ms"]), flush=True)
    payload = dict(meta, U=U, logits=out, hashes=hashes, gates=gates, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    return payload


@torch.inference_mode()
def run_cost(path, ids, distinct: int):
    """One process per batch, every U of U_LIST at every gate step; light snapshot between rounds."""
    from nosi import state_snapshot as ss
    from nosi import verify_trace as _vtr
    from nosi.verify.verify_step import RoundOverflow
    model, cache, logits, position_ids, forced, meta = _setup(path, ids, need_round_slots=True)
    B = ids.shape[0]
    Umax = max(U_LIST)
    rows = [logits[:, -1, :].float().cpu()]
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    ar = torch.arange(Umax, device="cuda", dtype=position_ids.dtype).unsqueeze(0)
    vt = _vtr.VerifyTrace(model.num_layers)
    _vtr.TRACE = vt
    light = ss.CounterSnapshot(cache)
    full = ss.CacheSnapshot(cache) if COST_HYG else None
    trans = None
    calls, hygiene = [], None
    dec_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]

    def round_once(Uc, it):
        toks, pos = forced[:, it:it + Uc], position_ids + ar[:, :Uc]
        lv = model.verify_inference(toks, cu, pos, cache)
        torch.cuda.synchronize()
        return lv

    for it in range(N):
        tok = forced[:, it:it + 1]
        if it >= WARM:
            if trans is None:
                trans = ss.transient_ids(model)
            if full is not None and it == WARM:
                # THE PROOF of the light restore (CounterSnapshot docstring): the decode step
                # with and without the rounds in between must be torch.equal; a round WITHOUT
                # a restore (positive control) must move it. No event is recorded here: the
                # trace is reset afterwards.
                full.take()
                A = model.decode_inference(tok, cu, position_ids, cache)[:, -1, :].float().cpu()
                torch.cuda.synchronize()
                full.restore(); ss.assert_transients_intact(model, trans)
                round_once(Umax, it)                                     # no restore on purpose
                P = model.decode_inference(tok, cu, position_ids, cache)[:, -1, :].float().cpu()
                torch.cuda.synchronize()
                full.restore(); ss.assert_transients_intact(model, trans)
                for Uc in U_LIST:
                    if it <= N - Uc:
                        light.take()
                        round_once(Uc, it)
                        light.restore(); ss.assert_transients_intact(model, trans)
                Bv = model.decode_inference(tok, cu, position_ids, cache)[:, -1, :].float().cpu()
                torch.cuda.synchronize()
                full.restore(); ss.assert_transients_intact(model, trans)
                vt.reset()
                hygiene = dict(step=it, same=bool(torch.equal(A, Bv)), max_abs=float((A - Bv).abs().max()),
                               control_differs=not torch.equal(A, P), control_max_abs=float((A - P).abs().max()), U_control=Umax)
                print("[pilot] cost-hyg at step %d: decode rows after light-restored rounds torch.equal=%s (max|d| %.4g); control (a round without restore) differs=%s (max|d| %.4g)"
                      % (it, hygiene["same"], hygiene["max_abs"], hygiene["control_differs"], hygiene["control_max_abs"]), flush=True)
            for Uc in U_LIST:
                if it > N - Uc:
                    continue
                light.take()
                rec = dict(U=Uc, step=it, skipped=0, overflow=None, total_ms=None)
                vt.label((Uc, it))
                try:
                    round_once(Uc, it)
                    n1, s1 = union_stats(model)
                    rec.update(n_new_max=int(n1.max()), n_new_mean=float(n1.float().mean()),
                               union_max=int(s1.max()), union_mean=float(s1.float().mean()))
                except RoundOverflow as e:
                    torch.cuda.synchronize()
                    vt.abort_call()
                    rec.update(skipped=1, overflow=str(e), n_new_max=int(e.n_new.max()) if e.n_new is not None else None)
                    print("[pilot] U=%d step %d: %s" % (Uc, it, e), flush=True)
                light.restore(); ss.assert_transients_intact(model, trans)
                calls.append(rec)
        e0, e1 = dec_ev[it]
        e0.record()
        lg = model.decode_inference(tok, cu, position_ids, cache, warmup=(it == 0))
        e1.record()
        torch.cuda.synchronize()
        rows.append(lg[:, -1, :].float().cpu())
        position_ids = position_ids + 1
    _vtr.TRACE = None
    timing = {r["label"]: r for r in vt.harvest()}
    for rec in calls:
        tr = timing.get((rec["U"], rec["step"]))
        if tr is not None and not rec["skipped"]:
            rec.update(total_ms=tr["total_ms"], score_ms=tr["score_ms"], fetch_ms=tr["fetch_ms"], attn_ms=tr["attn_ms"], rest_ms=tr["rest_ms"], layers_timed=tr["layers"])
    decode_ms = [dec_ev[it][0].elapsed_time(dec_ev[it][1]) for it in range(N)]
    gated = decode_ms[WARM:]
    out = torch.stack(rows, dim=1)
    hashes = [[sha(out[b, r]) for r in range(N + 1)] for b in range(B)]
    payload = dict(meta, U_list=list(U_LIST), distinct_docs=distinct, calls=calls, decode_ms=decode_ms,
                   decode_step_ms=sum(gated) / len(gated), hashes=hashes, hygiene=hygiene,
                   peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
    for rec in calls:
        if not rec["skipped"] and rec.get("total_ms") is not None:
            print("[pilot] U=%d step %d: %.1f ms (score %.1f fetch %.1f attn %.1f rest %.1f)  n_new max %d mean %.2f  union max %d mean %.1f"
                  % (rec["U"], rec["step"], rec["total_ms"], rec["score_ms"], rec["fetch_ms"], rec["attn_ms"], rec["rest_ms"],
                     rec["n_new_max"], rec["n_new_mean"], rec["union_max"], rec["union_mean"]), flush=True)
    print("[pilot] decode step ms per step: %s -> mean over steps %d..%d = %.1f" % (" ".join("%.1f" % m for m in decode_ms), WARM, N - 1, payload["decode_step_ms"]), flush=True)
    print(render_cost_table(aggregate_cost(payload)), flush=True)
    return payload


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def compare() -> int:
    tau_ctrl = read_tau_ctrl()                                        # refuses BEFORE any arm is read
    arms = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".pt") and (f.startswith("decode_") or f.startswith("verify_")):
            arms[f[:-3]] = torch.load(os.path.join(OUT, f))
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

    lines.append("tolerance: max |dlogit| <= 4 x tau_ctrl = 4 x %.4f = %.4f (NOSI_VERIFY_TAU_CTRL, measured by nosi_verify_ctrl.sbatch)\n" % (tau_ctrl, dlogit_tol(tau_ctrl)))
    lines.append("| gate | result | detail |\n|---|---|---|")
    b2_rows = {}
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
            if d["mode"] in ("verify", "verify_u"):
                if d["N"] != ref["N"] or d["warm"] != ref["warm"] or d["logits"].shape != ref["logits"].shape:
                    gate("shape[L=%d,%s]" % (Lval, name), False, "arm N=%s warm=%s logits %s vs shipped N=%s warm=%s logits %s" % (
                        d["N"], d["warm"], tuple(d["logits"].shape), ref["N"], ref["warm"], tuple(ref["logits"].shape)))
                    continue
                gate_verify_arm(name, Lval, d, ref, tau_ctrl, gate, lines, b2_rows)
    # B2, reported only (see the module docstring)
    b2 = []
    for (Lval, step), per_arm in sorted(b2_rows.items()):
        names = sorted(per_arm)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = per_arm[names[i]], per_arm[names[j]]
                b2.append("| %d | %d | %s | %s | %s | %.4g |" % (Lval, step, names[i], names[j], torch.equal(a, b), float((a - b).abs().max())))
    if b2:
        lines.append("\nB2 (position-0 logits across arms of one step; reported, not gated: the GEMMs run at B*U rows):\n| L | step | arm | arm | torch.equal | max abs d |\n|---|---|---|---|---|---|")
        lines.extend(b2)
    lines.append("\nfailed gates: %d" % nfail)
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(OUT, "compare.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, "compare.json"), "w") as f:
        json.dump(dict(tau_ctrl=tau_ctrl, tol=dlogit_tol(tau_ctrl), gates=summary, failed=nfail), f, indent=1)
    return nfail


def cost_table() -> int:
    payloads = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".pt") and f.startswith("cost_"):
            payloads.append((f[:-3], torch.load(os.path.join(OUT, f))))
    rows, cells, nfail, notes = [], {}, 0, []
    for name, p in payloads:
        for r in aggregate_cost(p):
            r["arm"] = name
            rows.append(r)
            cells["(%d, %d, %d)" % (r["L"], r["batch"], r["U"])] = r
        h = p.get("hygiene")
        if h is not None:
            ok = h["same"] and h["control_differs"]
            notes.append("cost-hyg[%s,B=%d]: %s -- decode rows after light-restored rounds torch.equal=%s (max|d| %.4g); control (a round of U=%d without restore) differs=%s (max|d| %.4g)"
                         % (name, p["batch"], "PASS" if ok else "FAIL", h["same"], h["max_abs"], h["U_control"], h["control_differs"], h["control_max_abs"]))
            nfail += int(not ok)
        else:
            notes.append("cost-hyg[%s,B=%d]: not run (NOSI_VERIFY_COST_HYG=0; the proof is the batch-64 probe: a full CacheSnapshot at batch 128 is ~34 GB)" % (name, p["batch"]))
        notes.append("provenance[%s]: L=%d N=%d warm=%d batch=%d distinct docs=%s ROUND_SLOTS=%d W=%d K+V+bias %.1f GB peak %.1f GB reserved %.1f GB U_list=%s"
                     % (name, p["L"], p["N"], p["warm"], p["batch"], p.get("distinct_docs"), p["round_slots"], p["W"], p["alloc_gb"], p["peak_gb"], p.get("reserved_gb", float("nan")), p["U_list"]))
    for b in COST_BATCHES:
        for u in U_LIST:
            have = [r for r in rows if r["batch"] == b and r["U"] == u]
            if not have or have[0]["n_calls"] == 0:
                nfail += 1
                notes.append("MISSING cell batch=%d U=%d: %s" % (b, u, "no cost arm at this batch" if not have else "no timed call (overflow %d)" % have[0]["overflow"]))
    text = render_cost_table(rows) + "\n\n" + "\n".join(notes) + "\n\nfailed: %d (missing/empty cells + failed hygiene probes)" % nfail
    print(text, flush=True)
    with open(os.path.join(OUT, "cost.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, "cost.json"), "w") as f:
        json.dump(dict(cells=cells, requested_batches=list(COST_BATCHES), U_list=list(U_LIST), failed=nfail,
                       columns=dict(verify_ms="mean ms per verify_inference call over the gated steps, first call per U excluded (CUDA events around the call)",
                                    score_ms="per-position scoring (decode kernels), summed over layers", fetch_ms="verify_round_update_kv: tail writes, union build, round-region gathers",
                                    attn_ms="verify_attention: the two union-wide value copies and the two varlen calls", rest_ms="the call minus the three: GEMMs, norms, rope, lm_head",
                                    decode_step_ms="mean shipped decode step ms over steps warm..N-1 in the same process", union_slots_mean="mean union size per (layer, head, request)",
                                    n_new_mean="mean blocks fetched into the round region per (layer, head, request)", peak_gb="torch.cuda.max_memory_allocated of the process")),
                  f, indent=1)
    return nfail


if __name__ == "__main__":
    if MODE == "compare":
        sys.exit(compare())
    if MODE == "cost_table":
        sys.exit(cost_table())
    check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    if os.environ.get("NOSI_ROWS_BENCH", "0") == "1":
        # retroinfer-eval fork: rows-attention microbench beside Path 1 (verify/rows_bench_hook.py); the forward keeps Path 1's output
        from nosi.verify import rows_bench_hook
        rows_bench_hook.install(OUT, TAG, splits_list=tuple(int(x) for x in os.environ.get("NOSI_ROWS_SPLITS", "4 8 0").split()))
    if os.environ.get("NOSI_ROWS_DECODE_BENCH", "0") == "1":
        # retroinfer-eval fork: the rows mechanism on the ordinary decode allocation (verify/rows_decode_hook.py); mode decode
        from nosi.verify import rows_decode_hook
        rows_decode_hook.install(OUT, TAG, u_list=tuple(int(x) for x in os.environ.get("NOSI_ROWS_DECODE_U", "2 3").split()),
                                 splits=int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0))
    ids, rows, distinct = load_docs(path)
    print("[docs] %s (%d distinct)  L=%d N=%d" % (rows, distinct, L, N), flush=True)
    if MODE == "decode":
        payload = run_trace(path, ids, with_verify=False)
    elif MODE == "verify":
        payload = run_trace(path, ids, with_verify=True, U=1)
    elif MODE == "verify_u":
        if U < 2:
            die("verify_u needs NOSI_VERIFY_U >= 2 (U = K + 1); use mode verify for U = 1")
        payload = run_trace(path, ids, with_verify=True, U=U)
    elif MODE == "cost":
        payload = run_cost(path, ids, distinct)
    else:
        raise SystemExit("unknown NOSI_VERIFY_MODE %r" % MODE)
    payload["docs"] = rows
    fn = os.path.join(OUT, "%s_%s.pt" % (MODE, TAG))
    torch.save(payload, fn)
    print("[pilot] saved %s (peak %.2f GB)" % (fn, payload["peak_gb"]), flush=True)
