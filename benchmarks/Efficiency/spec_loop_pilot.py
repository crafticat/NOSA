"""END-TO-END SPECULATIVE DECODE LOOP PILOT on NOSA-8B / NOSI (retroinfer-eval
fork, 2026-09-20). Schedule = 'round:K-draft-then-verify'. Core:
nosi/nosi/spec_loop.py (CPU-provable); engine: cache_engine.spec_*; model:
nosa_llama.draft_forward / spec_verify_forward / spec_rollback; sbatch:
retroinfer-eval env/slurm/nosi_spec_loop.sbatch; tests:
retroinfer-eval tests/test_nosi_spec_loop.py.

WHAT IS MEASURED (greedy, free-running; every arm on the same prompts, the
same batch composition, the same engine knobs):
  spec       one process per batch composition, NOSI_VERIFY_ROUND_SLOTS = R,
             pool off. prefill ONCE -> PostPrefillSnapshot -> the SHIPPED
             reference arm (cap 63: the shipped decode through the 64-slot
             views, gate T0 = byte-identical to R = 0) -> restore -> for every
             K of NOSI_SPEC_K_LIST: restore, warm-up decode step, then rounds
             until every request has committed NOSI_SPEC_TOTAL_TOKENS tokens.
             Per round (CUDA events + store counters): draft ms per step (with
             the score / tail / plan / launch+bias / attention brackets),
             prefetch blocks issued, timely-arrived blocks (event queried at
             verify start), late-wait blocks (in flight, waited for), late-fetch
             blocks (fetched at the round, synchronously: the late-miss stall),
             verify ms with its split (score / tail / plan / gather / copies /
             call 1 / call 2 / quotient / rest, host syncs counted), rollback
             ms, per-request n_acc, the lockstep commit c. Per run: committed
             tokens per second by a wall clock (from after the warm-up step;
             and over the throughput window after NOSI_SPEC_GATE_TOKENS), wire
             bytes per committed token (prefetch + late; wasted = evicted
             before use), peak HBM (allocated and reserved) and host pinned.
  decode_ref the CACHED reference (usable 113 = pool 50): NOSI_POOL_BLOCKS = P,
             ROUND_SLOTS = 0 (the engine refuses both together), the same
             prompts and batch composition, the shipped decode timed the same
             way; its greedy sequence must equal the shipped arm's (the pool
             is served-output-identical by construction).
  report     reads every .pt of the output directory: the SEQUENCE IDENTITY
             gate (every spec cell's committed sequence torch.equal to the
             shipped arm's on every request over the tokens both have; the
             cached arm vs the shipped arm; a mismatch is reported with the
             shipped logits' top-2 margin at that position: below the verifier's
             4 x tau_ctrl it is a near tie of the Path-1 numerics, above it a
             bug), the ROLLBACK PROBE (spec mode with NOSI_SPEC_PROBE=1: the
             post-round engine state torch.equal to a fresh replay of the
             committed tokens from a full snapshot, with a positive control),
             the overflow count, the tables and spec_loop.json.
             Exit code = failed gates.

THE COMMIT IS LOCKSTEP (spec_loop.py, module docstring): the batch commits
c = 1 + min_b n_acc(b) per round, exact for every request. The per-request
n_acc is recorded so the ragged-engine yield (1 + mean n_acc) is reported
beside it as a PROJECTION. NOSI_SPEC_DISTINCT limits the distinct documents
in the batch (cycled, as the verify cost mode does): with 2 distinct
documents the lockstep yield is 1 + a^2 and the loop is not degenerate; with
B distinct documents it is 1 + a^B.

TOKEN BUDGET. The pooling buffers are sized at warm-up (nosa_llama.py
decode_inference: out_len = ceil((L+1)/64)), so a run past the next 64-token
block boundary needs a RE-WARM (has_buffers = False -> the next decode step
recaptures the pooling graph at the current length). Every arm re-warms at
the same committed length (needs_rewarm); in the spec arm a re-warm is one
shipped exact step committing one token (the store is re-synced from
_block_map afterwards). Both arms count it inside the clock.

Environment: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_SPEC_OUT, NOSI_SPEC_MODE,
NOSI_SPEC_TAG, NOSI_SPEC_L (16128), NOSI_SPEC_DOCS (= batch), NOSI_SPEC_DISTINCT
(distinct documents, default = batch), NOSI_SPEC_GATE_TOKENS (64),
NOSI_SPEC_TOTAL_TOKENS (112), NOSI_SPEC_K_LIST ("1 2"), NOSI_SPEC_PROBE (0),
NOSI_SPEC_PROBE_ROUND (3), NOSI_SPEC_TAU_CTRL (report: the margin classifier,
0.4844 at 16K; no default), NOSI_VERIFY_ROUND_SLOTS / NOSI_POOL_BLOCKS /
NOSI_ATTN_SPLITS (read by the engine at import; recorded in the provenance).
"""
import hashlib
import json
import math
import os
import sys
import time

import torch

MODE = os.environ.get("NOSI_SPEC_MODE", "report")
OUT = os.environ.get("NOSI_SPEC_OUT", "nosi_spec_loop")
TAG = os.environ.get("NOSI_SPEC_TAG", MODE)
L = int(os.environ.get("NOSI_SPEC_L", "16128"))
NDOCS = int(os.environ.get("NOSI_SPEC_DOCS", "2"))
DISTINCT = int(os.environ.get("NOSI_SPEC_DISTINCT", "0") or 0) or NDOCS
GATE_TOKENS = int(os.environ.get("NOSI_SPEC_GATE_TOKENS", "64"))
TOTAL_TOKENS = int(os.environ.get("NOSI_SPEC_TOTAL_TOKENS", "112"))
K_LIST = tuple(int(x) for x in os.environ.get("NOSI_SPEC_K_LIST", "1 2").split())
PROBE = os.environ.get("NOSI_SPEC_PROBE", "0") == "1"
PROBE_ROUND = int(os.environ.get("NOSI_SPEC_PROBE_ROUND", "3"))
SCHEDULE = "round:K-draft-then-verify"
os.makedirs(OUT, exist_ok=True)

# The memory formula of the equal-memory frontier (scripts/nosi_equal_memory_frontier.py
# fit_peak over the measured [proof] rows of jobs 2174649 / 2174662; the task statement's
# constants), plus the pilot's own scratch, stated: the masked bias scratch per layer
# (B x W x 64 x 2 x 2 B x 32 layers), the verify's two union-wide value copies (one layer
# at a time, 2 x B x W x 64 x 256 x 2 B), the journals and the post-prefill snapshot.
PEAK_A_GB, PEAK_B_GB_PER_REQ, PEAK_C_GB_PER_REQ_SLOT = 19.0, 0.0244, 0.002111
SCRATCH_GB_PER_REQ = 0.012     # ~ (2 x 128 x 64 x 256 x 2 + 32 x 128 x 64 x 2 x 2) B = 8.4 MB + 2.1 MB per request at W = 128
SCRATCH_GB_FIXED = 1.5         # journals, snapshots, events, the store bookkeeping, allocator slack


def die(msg):
    print("[spec_loop] REFUSED: " + msg, flush=True)
    sys.exit(2)


# ---------------------------------------------------------------------------
# pure bookkeeping (CPU-tested in retroinfer-eval tests/test_nosi_spec_loop.py)
# ---------------------------------------------------------------------------
def peak_formula_gb(B: int, slots: int) -> float:
    """The frontier's fit: 19.0 + 0.0244 x B + 0.002111 x B x slots (GB)."""
    return PEAK_A_GB + PEAK_B_GB_PER_REQ * B + PEAK_C_GB_PER_REQ_SLOT * B * slots


def max_batch(budget_gb: float, slots: int, scratch_fixed_gb: float = SCRATCH_GB_FIXED,
              scratch_per_req_gb: float = SCRATCH_GB_PER_REQ) -> int:
    """Largest B with peak_formula(B, slots) + scratch <= budget. slots = topk + R + 2
    for the loop's store (W), or the cached arm's physical slots."""
    per_req = PEAK_B_GB_PER_REQ + PEAK_C_GB_PER_REQ_SLOT * slots + scratch_per_req_gb
    b = int(math.floor((budget_gb - PEAK_A_GB - scratch_fixed_gb) / per_req))
    if b < 1:
        raise ValueError("no batch fits: budget %.1f GB, slots %d" % (budget_gb, slots))
    return b


def needs_rewarm_len(seq_length: int, out_len: int, block_size: int = 64) -> bool:
    """The next decode step needs ceil((seq_length + 1) / block_size) pooled
    blocks (nosa_llama.decode_inference: out_len = (total_len + bs - 1) // bs
    with total_len = seq + 1); True when the captured buffers are too short."""
    return (seq_length + 1 + block_size - 1) // block_size > out_len


def identity_gate(spec_seq: torch.Tensor, ref_seq: torch.Tensor, ref_margin=None) -> dict:
    """spec_seq (B, n), ref_seq (B, m) int64 token sequences; ref_margin (B, m-1):
    the reference's top-2 logit margin at the step that produced ref_seq[:, j+1].
    Compares the first min(n, m) tokens of every request. Returns the verdict,
    the number of compared tokens, and per mismatching request the position,
    the two tokens and the reference margin there."""
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
        m = None
        if ref_margin is not None and j >= 1 and j - 1 < ref_margin.shape[1]:
            m = float(ref_margin[r, j - 1])
        mism.append(dict(request=r, position=j, spec_token=int(a[r, j]), ref_token=int(b[r, j]), ref_margin=m))
    return dict(passed=len(mism) == 0, compared_tokens=n, requests=int(a.shape[0]), mismatches=mism)


def classify_mismatches(gate: dict, tol: float) -> dict:
    """Split a failed identity gate's mismatches by the reference margin at
    the divergence: <= tol (4 x tau_ctrl, the verifier's own tolerance) is a
    near tie of the Path-1 numerics, > tol a bookkeeping bug; unknown when no
    margin was recorded."""
    near = bug = unknown = 0
    for m in gate["mismatches"]:
        if m["ref_margin"] is None:
            unknown += 1
        elif m["ref_margin"] <= tol:
            near += 1
        else:
            bug += 1
    return dict(near_tie=near, above_tol=bug, unknown=unknown, tol=tol)


def kernel_host_table(split: dict) -> dict:
    """One verify (or draft) split row from verify_trace.split_marks -> the
    kernel-vs-non-kernel partition the author asked for. score, gather, the
    two varlen calls are kernel windows (score contains the small table ops
    too); tail, plan (with the host-sync gap), copies, quotient are torch ops
    and host gaps; rest holds the GEMMs plus every host gap between kernels
    and is listed separately, never as 'PCIe wait' or 'arithmetic'."""
    def g(k):
        v = split.get(k)
        return 0.0 if v is None else float(v)
    kernel = dict(score_ms=g("score_ms"), gather_ms=g("gather_ms"), attn1_ms=g("attn1_ms"), attn2_ms=g("attn2_ms"))
    nonk = dict(tail_ms=g("tail_ms"), plan_ms=g("plan_ms"), copies_ms=g("copies_ms"), quot_ms=g("quot_ms"))
    return dict(kernel=kernel, kernel_total_ms=sum(kernel.values()), non_kernel=nonk,
                non_kernel_total_ms=sum(nonk.values()), rest_ms=g("rest_ms"), total_ms=g("total_ms"),
                host_syncs=int(split.get("host_sync", 0)))


def aggregate_rounds(rounds: list, gate_tokens: int) -> dict:
    """The per-round records of one spec cell -> the cell's numbers. A round
    record: c, n_acc (list of B ints), draft_ms (list per step), verify_ms,
    verify_split (dict), rollback_ms, committed_before (tokens per request
    before the round), rewarm (bool), overflow (bool), counters (dict of
    ints: store counters summed over layers/heads/requests at the END of the
    round). Only spec rounds (not re-warm steps) enter the timing means; the
    first spec round is excluded from the timing means (JIT)."""
    spec = [r for r in rounds if not r.get("rewarm")]
    timed = spec[1:] if len(spec) > 1 else spec
    out = dict(rounds=len(rounds), spec_rounds=len(spec), rewarm_steps=sum(1 for r in rounds if r.get("rewarm")),
               overflow_rounds=sum(1 for r in spec if r.get("overflow")))
    if spec:
        cs = [int(r["c"]) for r in spec]
        out["c_mean"] = sum(cs) / len(cs)
        out["c_hist"] = {str(c): cs.count(c) for c in sorted(set(cs))}
        nacc = [float(sum(r["n_acc"]) / len(r["n_acc"])) for r in spec]
        out["n_acc_mean_per_request"] = sum(nacc) / len(nacc)
        out["yield_lockstep_per_round"] = out["c_mean"]
        out["yield_ragged_projection_per_round"] = 1.0 + out["n_acc_mean_per_request"]
        acc_flat = [int(x) for r in spec for x in r["n_acc"]]
        out["n_acc_hist_per_request"] = {str(k): acc_flat.count(k) for k in sorted(set(acc_flat))}
    if timed:
        def mean(vals):
            vals = [float(v) for v in vals if v is not None]
            return (sum(vals) / len(vals)) if vals else None
        out["draft_step_ms"] = mean([d for r in timed for d in r["draft_ms"]])
        out["verify_ms"] = mean([r["verify_ms"] for r in timed])
        out["rollback_ms"] = mean([r["rollback_ms"] for r in timed])
        keys = set()
        for r in timed:
            keys |= set((r.get("verify_split") or {}).keys())
        out["verify_split"] = {k: mean([(r.get("verify_split") or {}).get(k) for r in timed]) for k in sorted(keys)
                               if k not in ("label", "layers")}
        dkeys = set()
        for r in timed:
            for d in r.get("draft_split") or []:
                dkeys |= set(d.keys())
        out["draft_split"] = {k: mean([d.get(k) for r in timed for d in (r.get("draft_split") or [])]) for k in sorted(dkeys)
                              if k not in ("label", "layers")}
        out["round_ms"] = mean([sum(r["draft_ms"]) + float(r["verify_ms"] or 0) + float(r["rollback_ms"] or 0) for r in timed])
    return out


def tokens_per_second(tokens_per_request: int, batch: int, wall_s: float) -> float:
    if wall_s <= 0:
        raise ValueError("wall must be > 0")
    return tokens_per_request * batch / wall_s


def sha(row: torch.Tensor) -> str:
    return hashlib.sha256(row.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# data
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
    assert distinct >= 1, "no document with >= %d tokens" % L
    rows = [rows[i % distinct] for i in range(NDOCS)]
    ids = [ids[i % distinct] for i in range(NDOCS)]
    return torch.stack(ids), rows, distinct


# ---------------------------------------------------------------------------
# GPU drivers
# ---------------------------------------------------------------------------
def provenance():
    from nosi import cache_engine as _ce
    here = os.path.dirname(os.path.abspath(__file__))
    nosa_src = open(os.path.join(here, "..", "..", "nosi", "nosi", "nosa_llama.py")).read()
    return dict(
        schedule=SCHEDULE, mode=MODE, tag=TAG, L=L, batch=NDOCS, distinct_docs=DISTINCT,
        gate_tokens=GATE_TOKENS, total_tokens=TOTAL_TOKENS, K_list=list(K_LIST),
        engine_knobs=dict(NOSI_VERIFY_ROUND_SLOTS=_ce.VERIFY_ROUND_SLOTS, NOSI_POOL_BLOCKS=_ce.POOL_BLOCKS,
                          NOSI_ATTN_SPLITS=int(os.environ.get("NOSI_ATTN_SPLITS", "0") or 0),
                          NOSI_KV_BIAS_SCALE=_ce.KV_BIAS_SCALE, NOSI_AVAIL=os.environ.get("NOSI_AVAIL", ""),
                          fix_A_present="ORDINARY FIX A" in nosa_src),
        overlap_evidence=dict(
            draft_prefetch_stream="side stream (torch.cuda.Stream) in cache_engine.spec_draft_update step C: side.wait_event(main event recorded after the tail write) -> flash_h2d_from_mask + flash_h2d_from_mask_bias -> side event recorded and pushed to spec_loop.PrefetchQueue",
            residency_gate="spec_loop.Store.mark_arrived(queue.done_gen()) at the start of every draft step and at verify start: a slot is RESIDENT only after the host polled its issue event complete (event.query(), no sync); the draft's bias denies every non-RESIDENT slot (plan_draft), so the draft's attention NEVER waits on a prefetch event",
            verify_union_fetch="synchronous on the main stream (cache_engine.spec_round_update: the two gathers of plan.fetch after the union plan); before the round the main stream waits on the side stream's last event (wait_event) so in-flight prefetches are waited for, not re-fetched: late_wait; blocks never requested are fetched at the round: late_fetch = the late-miss stall",
            draft_verify_overlap="NONE: the verify round starts after the last draft step's kernels are enqueued on the same main stream; no verify compute overlaps any draft compute (schedule round:K-draft-then-verify, not the pipelined V/S two-row schedule)",
        ),
        custom_kernels="none: the draft uses flash_attn_nosa_with_kvcache with a per-row bias (the shipped decode kernel), the prefetch/late fetch the shipped Triton gathers, the verify the IV2 prefill varlen op; every new path is pure torch (adaptation candidates: plan_draft / plan_round integer ops, the bias scratch copy + masked_fill, the union-wide value copies, the journal restore/replay)",
    )


def _setup(path, ids):
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    from nosi.cache_engine import InfLLMv2Cache
    print("[spec_loop] mode=%s tag=%s L=%d batch=%d distinct=%d K=%s ROUND_SLOTS=%d POOL=%d SPLITS=%s gate=%d total=%d probe=%s"
          % (MODE, TAG, L, NDOCS, DISTINCT, K_LIST, _ce.VERIFY_ROUND_SLOTS, _ce.POOL_BLOCKS, os.environ.get("NOSI_ATTN_SPLITS", "0"),
             GATE_TOKENS, TOTAL_TOKENS, PROBE), flush=True)
    model = Llama(model_name=path, device="cuda", offload=True)
    x = ids.to("cuda")
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    t0 = time.time()
    logits, position_ids = model.batch_prefill(x, cache)
    torch.cuda.synchronize()
    eng = cache.layers[0].cache_engine
    W = eng._k_gpu.shape[1] // eng.block_size
    alloc_gb = sum(t.numel() * t.element_size() for t in (eng._k_gpu, eng._v_gpu, eng._kv_bias_gpu)) * model.num_layers / 1e9
    host_gb = sum(t.numel() * t.element_size() for t in (eng._k_cpu, eng._v_cpu)) * model.num_layers / 1e9
    print("[spec_loop] prefill %d x %d: %.1fs; allocation W=%d slots, K+V+bias %.2f GB, host pinned %.1f GB, peak %.1f GB"
          % (x.shape[0], L, time.time() - t0, W, alloc_gb, host_gb, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    meta = dict(provenance(), W=W, alloc_gb=alloc_gb, host_pinned_gb=host_gb, prefill_s=time.time() - t0,
                prefill_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    return model, cache, logits, position_ids, meta


def needs_rewarm(model, cache) -> bool:
    return (not model.has_buffers) or needs_rewarm_len(cache.get_seq_length(0), model.pooling_buf_all.shape[-1], model.block_size)


@torch.inference_mode()
def run_decode_arm(model, cache, t0, position_ids, total: int, label: str) -> dict:
    """The shipped (or pooled) greedy decode: total tokens after t0, the
    sequence, the top-2 margins, per-step CUDA events, the wall clock from
    after the warm-up step and over the throughput window. No host sync
    inside the loop (the argmax stays on the device)."""
    B = t0.shape[0]
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    tok = t0
    pos = position_ids.clone()
    seq = [t0.clone()]
    margins = []
    evs = []
    model.has_buffers = False
    t_warm = t_gate = None
    rewarms = []
    for it in range(total):
        if needs_rewarm(model, cache):
            model.has_buffers = False
            rewarms.append(it)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        lg = model.decode_inference(tok, cu, pos, cache)
        e1.record()
        row = lg[:, -1, :]
        top2 = row.topk(2, dim=-1).values
        margins.append((top2[:, 0] - top2[:, 1]).clone())
        tok = row.argmax(-1, keepdim=True)
        seq.append(tok.clone())
        pos = pos + 1
        evs.append((e0, e1))
        if it == 0:
            torch.cuda.synchronize()
            t_warm = time.time()
        if it + 1 == GATE_TOKENS:
            torch.cuda.synchronize()
            t_gate = time.time()
    torch.cuda.synchronize()
    t_end = time.time()
    step_ms = [a.elapsed_time(b) for a, b in evs]
    out = dict(label=label, tokens=total, seq=torch.cat(seq, 1).cpu(), margins=torch.stack(margins, 1).cpu(),
               step_ms=step_ms, rewarm_steps=rewarms, wall_s_after_warmup=t_end - t_warm,
               tok_s_after_warmup=tokens_per_second(total - 1, B, t_end - t_warm))
    if t_gate is not None and total > GATE_TOKENS:
        out["wall_s_window"] = t_end - t_gate
        out["tok_s_window"] = tokens_per_second(total - GATE_TOKENS, B, t_end - t_gate)
    out["step_ms_mean_window"] = sum(step_ms[GATE_TOKENS:]) / max(1, len(step_ms[GATE_TOKENS:])) if total > GATE_TOKENS else None
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out["reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    print("[spec_loop] %s: %d tokens, %.1f tok/s after warm-up, window %s tok/s, step ms mean %.1f, rewarm at %s, peak %.1f GB"
          % (label, total, out["tok_s_after_warmup"], ("%.1f" % out["tok_s_window"]) if "tok_s_window" in out else "-",
             sum(step_ms[1:]) / max(1, len(step_ms) - 1), rewarms, out["peak_gb"]), flush=True)
    return out


def _capture_state(cache, tail_len: int, seq_length: int, host_lo: int):
    """The engine state the rollback probe compares: counters, block map,
    tail-slot rows below tail_len, layer tables up to their lengths, host rows
    [host_lo, seq_length). The mirrors and the rows above tail_len are scratch."""
    st = []
    for lay in cache.layers:
        eng = lay.cache_engine
        bs = eng.block_size
        ts = eng._tail_block_idx_on_gpu * bs
        d = dict(seq_length=int(eng.seq_length), tail_len=int(eng._tail_block_len_on_gpu),
                 cache_lens=eng._cache_lens.clone(), block_map=eng._block_map.clone(),
                 k_tail=eng._k_gpu[:, ts:ts + tail_len].clone(), v_tail=eng._v_gpu[:, ts:ts + tail_len].clone(),
                 b_tail=eng._kv_bias_gpu[:, ts:ts + tail_len].clone(),
                 k_host=eng._k_cpu[:, host_lo:seq_length].clone(), v_host=eng._v_cpu[:, host_lo:seq_length].clone(),
                 no_compress_k=lay.no_compress_k_cache.clone(), tail_cis=lay.tail_cis.clone(),
                 compressed_cis=lay.compressed_cis[..., :lay.comp_cis_len].clone(), total_cis=lay.total_cis[:, :lay.cis_len].clone(),
                 compress_k=lay.compress_k_cache_varlen.clone(), cu=lay.cached_compressed_cu_seqlens.clone(),
                 scalars=tuple(int(getattr(lay, n)) for n in ("cached_compressed_max_seqlen", "no_compress_k_len", "comp_cis_len", "tail_cis_len", "cis_len", "seq_length")))
        st.append(d)
    return dict(layers=st, seen=int(cache._seen_tokens))


def _states_equal(a, b) -> tuple:
    if a["seen"] != b["seen"]:
        return False, "seen_tokens %d vs %d" % (a["seen"], b["seen"])
    for l, (x, y) in enumerate(zip(a["layers"], b["layers"])):
        for k in x:
            if torch.is_tensor(x[k]):
                if x[k].shape != y[k].shape or not torch.equal(x[k], y[k]):
                    return False, "layer %d %s differs" % (l, k)
            elif x[k] != y[k]:
                return False, "layer %d %s %r vs %r" % (l, k, x[k], y[k])
    return True, "equal"


@torch.inference_mode()
def run_spec_arm(model, cache, t0, position_ids, K: int, total: int, probe: bool) -> dict:
    from nosi import spec_loop as _sl
    from nosi import state_snapshot as ss
    from nosi import verify_trace as _vtr
    from nosi.verify.tail_write import write_tail
    from nosi.verify.verify_step import RoundOverflow
    from nosi.cache_engine import _bias_rows
    B = t0.shape[0]
    eng0 = cache.layers[0].cache_engine
    H, bs = eng0.head_num, eng0.block_size
    layout = _sl.store_layout(eng0.topk, eng0.verify_round_slots)
    store = _sl.Store(model.num_layers, H, B, layout, device="cuda")
    ctx = _sl.SpecContext(store, side=torch.cuda.Stream(), make_event=torch.cuda.Event,
                          current_stream=torch.cuda.current_stream, stream_ctx=torch.cuda.stream,
                          num_layers=model.num_layers, block_size=bs,
                          kernel_size=model.layers[0].pooling_block_size, kernel_stride=model.layers[0].pooling_stride)
    vt = _vtr.VerifyTrace(model.num_layers)
    _vtr.TRACE = vt
    ctx.mark = vt.rec
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    pos = position_ids.clone()
    U = K + 1

    def shipped_step(tok, pos_, why):
        """One exact shipped step (warm-up or re-warm): drains the side stream
        first (its gathers write the round region; diff writes the window
        only, but a synced store is the invariant), then re-syncs the store."""
        ctx.side.synchronize()
        ctx.queue.drain()
        store.mark_arrived(ctx.gen)
        model.has_buffers = False
        lg = model.decode_inference(tok, cu, pos_, cache)
        nxt = lg[:, -1, :].argmax(-1, keepdim=True)
        tick = ctx.next_tick()
        for l, lay in enumerate(cache.layers):
            store.sync_window(l, lay.cache_engine._block_map, tick)
        return nxt

    # warm-up: the shipped step (feeds t0 -> t1), excluded from the clock in every arm
    next_tok = shipped_step(t0, pos, "warmup")
    pos = pos + 1
    committed = [t0.clone(), next_tok.clone()]
    n_committed = 1
    torch.cuda.synchronize()
    t_warm = time.time()
    t_gate = None
    rounds = []
    probe_result = None
    r = 0
    while n_committed < total:
        if t_gate is None and n_committed >= GATE_TOKENS:
            torch.cuda.synchronize()
            t_gate = time.time()
        if needs_rewarm(model, cache):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            next_tok = shipped_step(next_tok, pos, "rewarm")
            e1.record()
            pos = pos + 1
            committed.append(next_tok.clone())
            n_committed += 1
            rounds.append(dict(r=r, rewarm=True, c=1, ev=(e0, e1), committed_before=n_committed - 1))
            r += 1
            continue
        r += 1
        tail_id = eng0.seq_length // bs
        do_probe = probe and r == PROBE_ROUND
        if do_probe:
            torch.cuda.synchronize()
            ctx.side.synchronize()
            ctx.queue.drain()
            store.mark_arrived(ctx.gen)
            full = ss.CacheSnapshot(cache).take()
            store_snap = store.snapshot()
            seq0_probe, tl0_probe = int(eng0.seq_length), int(eng0._tail_block_len_on_gpu)
        ctx.begin_round(cache, tail_id)
        rec = dict(r=r, rewarm=False, K=K, committed_before=n_committed, overflow=False, ev_draft=[], ev_roll=None)
        # DRAFT
        drafted = []
        tok, p = next_tok, pos
        for j in range(K):
            ctx.next_tick()
            store.mark_arrived(ctx.queue.done_gen())
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            vt.label(("draft", r, j))
            e0.record()
            lg = model.draft_inference(tok, cu, p, cache, ctx)
            e1.record()
            tok = lg[:, -1, :].argmax(-1, keepdim=True)
            drafted.append(tok)
            p = p + 1
            rec["ev_draft"].append((e0, e1))
        ctx.restore_before_verify(cache)
        # arrivals at verify start: what the host can see, then the GPU-side wait for the rest
        store.mark_arrived(ctx.queue.done_gen())
        ctx.was_inflight = (store.state == _sl.INFLIGHT).clone()
        if ctx.queue.last_event is not None:
            torch.cuda.current_stream().wait_event(ctx.queue.last_event)
        ctx.queue.drain()
        store.mark_arrived(ctx.gen)
        ctx.next_tick()
        # VERIFY
        toks = torch.cat([next_tok] + drafted, dim=1)                       # (B, U)
        pu = pos + torch.arange(U, device="cuda", dtype=pos.dtype).unsqueeze(0)
        vt.label(("verify", r))
        drafted_t = torch.cat(drafted, dim=1) if drafted else torch.zeros((B, 0), dtype=torch.int64, device="cuda")
        try:
            lv = model.spec_verify_inference(toks, cu, pu, cache, ctx)
        except RoundOverflow as e:
            # the union did not fit: undo the layers that wrote, fall back to U = 1 (never overflows)
            vt.abort_call()
            print("[spec_loop] round %d: %s -> fallback U=1" % (r, e), flush=True)
            for l, lay in enumerate(cache.layers):
                if getattr(lay.cache_engine, "_spec_round_tail", None) is not None:
                    lay.cache_engine.spec_rollback_tail(0, layout.mirror_lo)
                model.layers[l]._spec_round_inputs = None
            ctx.restore_before_verify(cache)
            ctx.was_inflight = torch.zeros_like(store.state, dtype=torch.bool)
            rec["overflow"] = True
            vt.label(("verify_fallback", r))
            lv = model.spec_verify_inference(next_tok, cu, pos, cache, ctx)
            drafted_t = torch.zeros((B, 0), dtype=torch.int64, device="cuda")
        # ACCEPT
        acc = _sl.accept_commit(lv[:, :, :].argmax(-1), drafted_t)          # one host sync (the min)
        ctx.host_syncs += 1
        # ROLLBACK
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        tr = model.spec_rollback(cache, ctx, acc.c)
        e1.record()
        rec["ev_roll"] = (e0, e1)
        rec.update(c=acc.c, n_acc=acc.n_acc.clone(), truncate=dict(tail_len=tr.tail_len, undo_rollover=tr.undo_rollover, rows_restored=tr.rows_restored),
                   counters={k: v.sum().clone() for k, v in store.counters().items()},
                   union_size_mean=store.resident_count().float().mean().clone())
        committed.append(acc.committed.clone())
        next_tok = acc.next_token
        pos = pos + acc.c
        n_committed += acc.c
        if do_probe:
            # THE ROLLBACK PROBE: the post-round state must equal a fresh replay of the c committed tokens from the round-start snapshot
            torch.cuda.synchronize()
            c = acc.c
            host_lo = max(0, seq0_probe - bs)
            e1s = _capture_state(cache, int(eng0._tail_block_len_on_gpu), int(eng0.seq_length), host_lo)
            inputs = [(model.layers[l]._spec_probe_inputs) for l in range(model.num_layers)]

            def replay(keep):
                full.restore()
                store.restore(store_snap)
                for l, lay in enumerate(cache.layers):
                    key, val, cis = inputs[l]
                    ctx.journals[l].replay(lay, key, cis, keep, model.layers[l].pooling_block_size, model.layers[l].pooling_stride)
                    write_tail(lay.cache_engine, key[:, :keep], val[:, :keep], _bias_rows(lay.total_cis), layout.mirror_lo)
                cache._seen_tokens = ctx.seen0 + keep
                torch.cuda.synchronize()
                return _capture_state(cache, int(eng0._tail_block_len_on_gpu), int(eng0.seq_length), host_lo)

            same, why = _states_equal(e1s, replay(c))
            alt = c + 1 if c < U else c - 1
            diff, why_alt = _states_equal(e1s, replay(alt))
            e2 = replay(c)          # leave the engine at the committed state
            same2, _ = _states_equal(e1s, e2)
            probe_result = dict(round=r, c=c, U=U, same=bool(same), detail=why, control_keep=alt,
                                control_differs=not bool(diff), control_detail=why_alt, replay_deterministic=bool(same2),
                                passed=bool(same and not diff and same2))
            print("[spec_loop] rollback probe at round %d (c=%d, U=%d): state == replay(c): %s (%s); control replay(%d) differs: %s (%s)"
                  % (r, c, U, same, why, alt, not diff, why_alt), flush=True)
            for l in range(model.num_layers):
                model.layers[l]._spec_probe_inputs = None
        rounds.append(rec)
    torch.cuda.synchronize()
    t_end = time.time()
    ctx.side.synchronize()
    _vtr.TRACE = None
    timing = {row["label"]: row for row in vt.harvest()}
    # harvest the per-round records
    out_rounds = []
    prev_counters = None
    for rec in rounds:
        if rec.get("rewarm"):
            e0, e1 = rec["ev"]
            out_rounds.append(dict(r=rec["r"], rewarm=True, c=1, step_ms=e0.elapsed_time(e1), committed_before=rec["committed_before"]))
            continue
        row = dict(r=rec["r"], rewarm=False, K=rec["K"], c=rec["c"], overflow=rec["overflow"], committed_before=rec["committed_before"],
                   n_acc=rec["n_acc"].cpu().tolist(), truncate=rec["truncate"],
                   draft_ms=[a.elapsed_time(b) for a, b in rec["ev_draft"]],
                   draft_split=[timing.get(("draft", rec["r"], j)) for j in range(rec["K"])],
                   rollback_ms=rec["ev_roll"][0].elapsed_time(rec["ev_roll"][1]),
                   union_size_mean=float(rec["union_size_mean"]))
        vrow = timing.get(("verify_fallback", rec["r"]) if rec["overflow"] else ("verify", rec["r"]))
        row["verify_ms"] = None if vrow is None else vrow["total_ms"]
        row["verify_split"] = vrow
        cnt = {k: int(v) for k, v in rec["counters"].items()}
        row["counters_cum"] = cnt
        row["counters"] = {k: cnt[k] - (prev_counters or {}).get(k, 0) for k in cnt}
        prev_counters = cnt
        out_rounds.append(row)
    final_counters = {k: int(v) for k, v in store.counters().items()}
    wire = _sl.wire_bytes(final_counters)
    seq = torch.cat(committed, dim=1).cpu()
    result = dict(K=K, U=U, schedule=SCHEDULE, tokens=n_committed, seq=seq, rounds=out_rounds,
                  counters=final_counters, wire=wire, timely_recall=_sl.timely_recall(final_counters),
                  host_syncs=ctx.host_syncs, wall_s_after_warmup=t_end - t_warm,
                  tok_s_after_warmup=tokens_per_second(n_committed - 1, B, t_end - t_warm),
                  peak_gb=torch.cuda.max_memory_allocated() / 1e9, reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                  probe=probe_result, W=layout.W, R=layout.R)
    if t_gate is not None:
        n_gate = next((rr["committed_before"] for rr in out_rounds if rr["committed_before"] >= GATE_TOKENS), None)
        if n_gate is not None and n_committed > n_gate:
            result["wall_s_window"] = t_end - t_gate
            result["tok_s_window"] = tokens_per_second(n_committed - n_gate, B, t_end - t_gate)
            result["window_tokens"] = n_committed - n_gate
    agg = aggregate_rounds(out_rounds, GATE_TOKENS)
    result["agg"] = agg
    lock, ragged = _sl.lockstep_yield([rr["n_acc"] for rr in out_rounds if not rr.get("rewarm")])
    result["yield_lockstep_tokens"] = lock
    result["yield_ragged_projection_tokens"] = ragged
    print("[spec_loop] K=%d: %d committed tokens in %d rounds (%d re-warm steps, %d overflow); c mean %.3f, n_acc mean %.3f; draft %.1f ms/step, verify %.1f ms, rollback %.1f ms; %.1f tok/s after warm-up, window %s; recall %s; wire %.2f MB/committed token (prefetch %.1f, late %.1f, wasted %.1f MB); peak %.1f GB; host syncs %d"
          % (K, n_committed, len(out_rounds), agg["rewarm_steps"], agg["overflow_rounds"], agg.get("c_mean", float("nan")),
             agg.get("n_acc_mean_per_request", float("nan")), agg.get("draft_step_ms") or float("nan"), agg.get("verify_ms") or float("nan"),
             agg.get("rollback_ms") or float("nan"), result["tok_s_after_warmup"],
             ("%.1f tok/s" % result["tok_s_window"]) if "tok_s_window" in result else "-",
             ("%.3f" % result["timely_recall"]) if result["timely_recall"] is not None else "-",
             wire["wire_bytes"] / max(1, (n_committed - 1) * B) / 1e6, wire["prefetch_bytes"] / 1e6, wire["late_bytes"] / 1e6, wire["wasted_bytes"] / 1e6,
             result["peak_gb"], ctx.host_syncs), flush=True)
    return result


def run_spec_process(path, ids, rows, distinct):
    from nosi import state_snapshot as ss
    model, cache, logits, position_ids, meta = _setup(path, ids)
    B = ids.shape[0]
    t0 = logits[:, -1, :].argmax(-1, keepdim=True)
    pos0 = position_ids[:, -1:] + 1
    snap = ss.PostPrefillSnapshot(cache).take()
    arms = {}
    arms["shipped"] = run_decode_arm(model, cache, t0, pos0, TOTAL_TOKENS, "shipped")
    cells = {}
    for K in K_LIST:
        snap.restore()
        torch.cuda.reset_peak_memory_stats()
        cells["K%d" % K] = run_spec_arm(model, cache, t0, pos0, K, TOTAL_TOKENS, PROBE)
    payload = dict(meta, docs=rows, distinct_docs=distinct, arms=arms, cells=cells, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    return payload


def run_decode_ref_process(path, ids, rows, distinct):
    model, cache, logits, position_ids, meta = _setup(path, ids)
    t0 = logits[:, -1, :].argmax(-1, keepdim=True)
    pos0 = position_ids[:, -1:] + 1
    arm = run_decode_arm(model, cache, t0, pos0, TOTAL_TOKENS, "cached" if meta["engine_knobs"]["NOSI_POOL_BLOCKS"] > 0 else "shipped_ref")
    return dict(meta, docs=rows, distinct_docs=distinct, arms={arm["label"]: arm}, cells={}, peak_gb=torch.cuda.max_memory_allocated() / 1e9)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def read_tau_ctrl():
    raw = os.environ.get("NOSI_SPEC_TAU_CTRL")
    if raw is None or raw.strip() == "":
        die("NOSI_SPEC_TAU_CTRL is unset: the identity gate classifies a mismatch by the shipped top-2 margin against 4 x tau_ctrl "
            "(0.4844 at L=16128, job 2175374); no default")
    tau = float(raw)
    if not (tau > 0) or math.isinf(tau):
        die("NOSI_SPEC_TAU_CTRL=%r must be a finite positive number" % raw)
    return tau


def report() -> int:
    tau = read_tau_ctrl()
    tol = 4.0 * tau
    payloads = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".pt") and (f.startswith("spec_") or f.startswith("decode_ref_")):
            payloads[f[:-3]] = torch.load(os.path.join(OUT, f))
    if not payloads:
        die("no spec_*.pt / decode_ref_*.pt in %s" % OUT)
    lines, gates, nfail = [], {}, 0

    def gate(key, ok, detail):
        nonlocal nfail
        gates[key] = dict(passed=bool(ok), detail=detail)
        lines.append("| %s | %s | %s |" % (key, "PASS" if ok else "FAIL", detail))
        if not ok:
            nfail += 1

    lines.append("# spec loop pilot (schedule %s)\n" % SCHEDULE)
    lines.append("identity tolerance for the near-tie classification: 4 x tau_ctrl = 4 x %.4f = %.4f\n" % (tau, tol))
    lines.append("| gate | result | detail |\n|---|---|---|")
    cells_out = {}
    shipped_by_key = {}
    for name, p in payloads.items():
        key = (int(p["L"]), int(p["batch"]), int(p.get("distinct_docs", p["batch"])), tuple(p["docs"]))
        if "shipped" in p["arms"]:
            shipped_by_key[key] = (name, p["arms"]["shipped"])
    for name, p in payloads.items():
        key = (int(p["L"]), int(p["batch"]), int(p.get("distinct_docs", p["batch"])), tuple(p["docs"]))
        ref = shipped_by_key.get(key)
        for cname, c in p["cells"].items():
            tag = "%s/%s" % (name, cname)
            if ref is None:
                gate("identity[%s]" % tag, False, "no shipped arm with the same (L, batch, distinct, docs)")
            else:
                g = identity_gate(c["seq"], ref[1]["seq"], ref[1]["margins"])
                cl = classify_mismatches(g, tol)
                gate("identity[%s vs %s]" % (tag, ref[0]), g["passed"],
                     "%d requests x %d tokens; mismatches %d (near-tie %d, above tol %d, unknown %d)%s"
                     % (g["requests"], g["compared_tokens"], len(g["mismatches"]), cl["near_tie"], cl["above_tol"], cl["unknown"],
                        ("; first: %r" % (g["mismatches"][0],)) if g["mismatches"] else ""))
            if c.get("probe") is not None:
                pr = c["probe"]
                gate("rollback-probe[%s]" % tag, pr["passed"],
                     "round %d c=%d U=%d: state == replay(c) %s (%s); control replay(%d) differs %s (%s); deterministic %s"
                     % (pr["round"], pr["c"], pr["U"], pr["same"], pr["detail"], pr["control_keep"], pr["control_differs"], pr["control_detail"], pr["replay_deterministic"]))
            gate("overflow[%s]" % tag, c["agg"]["overflow_rounds"] == 0, "%d overflow rounds (fell back to U=1)" % c["agg"]["overflow_rounds"])
            cells_out[tag] = dict(payload=name, K=c["K"], U=c["U"], batch=p["batch"], distinct_docs=p.get("distinct_docs"), L=p["L"],
                                  W=c["W"], R=c["R"], tokens=c["tokens"], agg=c["agg"], counters=c["counters"], wire=c["wire"],
                                  timely_recall=c["timely_recall"], host_syncs=c["host_syncs"],
                                  tok_s_after_warmup=c["tok_s_after_warmup"], tok_s_window=c.get("tok_s_window"),
                                  wall_s_after_warmup=c["wall_s_after_warmup"], wall_s_window=c.get("wall_s_window"),
                                  peak_gb=c["peak_gb"], reserved_gb=c["reserved_gb"], host_pinned_gb=p["host_pinned_gb"],
                                  yield_lockstep_tokens=c["yield_lockstep_tokens"], yield_ragged_projection_tokens=c["yield_ragged_projection_tokens"],
                                  kernel_host=kernel_host_table(c["agg"].get("verify_split") or {}),
                                  draft_kernel_host=kernel_host_table(c["agg"].get("draft_split") or {}),
                                  provenance=dict(schedule=p.get("schedule"), engine_knobs=p["engine_knobs"], overlap_evidence=p["overlap_evidence"], custom_kernels=p["custom_kernels"]))
        for aname, a in p["arms"].items():
            if aname == "shipped":
                continue
            if ref is None:
                gate("identity[%s/%s]" % (name, aname), False, "no shipped arm with the same (L, batch, distinct, docs)")
            else:
                g = identity_gate(a["seq"], ref[1]["seq"], ref[1]["margins"])
                gate("identity[%s/%s vs %s]" % (name, aname, ref[0]), g["passed"],
                     "%d requests x %d tokens; mismatches %d%s" % (g["requests"], g["compared_tokens"], len(g["mismatches"]),
                                                                    ("; first: %r" % (g["mismatches"][0],)) if g["mismatches"] else ""))
    # tables
    lines.append("\n## reference arms (tok/s by the same wall clock; window = tokens after the gate window)\n")
    lines.append("| payload | arm | batch | distinct | pool | splits | tokens | tok/s after warm-up | tok/s window | step ms (window) | peak GB | reserved GB | host pinned GB |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    refs_out = {}
    for name, p in payloads.items():
        for aname, a in p["arms"].items():
            lines.append("| %s | %s | %d | %s | %d | %s | %d | %.1f | %s | %s | %.1f | %.1f | %.1f |" % (
                name, aname, p["batch"], p.get("distinct_docs"), p["engine_knobs"]["NOSI_POOL_BLOCKS"], p["engine_knobs"]["NOSI_ATTN_SPLITS"],
                a["tokens"], a["tok_s_after_warmup"], ("%.1f" % a["tok_s_window"]) if a.get("tok_s_window") else "-",
                ("%.1f" % a["step_ms_mean_window"]) if a.get("step_ms_mean_window") else "-", a["peak_gb"], a["reserved_gb"], p["host_pinned_gb"]))
            refs_out["%s/%s" % (name, aname)] = dict(payload=name, arm=aname, batch=p["batch"], distinct_docs=p.get("distinct_docs"),
                                                     engine_knobs=p["engine_knobs"], tokens=a["tokens"], tok_s_after_warmup=a["tok_s_after_warmup"],
                                                     tok_s_window=a.get("tok_s_window"), step_ms_mean_window=a.get("step_ms_mean_window"),
                                                     peak_gb=a["peak_gb"], reserved_gb=a["reserved_gb"], host_pinned_gb=p["host_pinned_gb"],
                                                     rewarm_steps=a.get("rewarm_steps"))
    lines.append("\n## speculative cells (lockstep commit; ragged yield is a projection from the measured per-request n_acc)\n")
    lines.append("| cell | batch | distinct | K | rounds | c mean | n_acc mean/req | yield lockstep | yield ragged (proj) | draft ms/step | verify ms | rollback ms | round ms | tok/s after warm-up | tok/s window | recall | wire MB/tok | late MB/tok | wasted MB | peak GB | host syncs |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, c in cells_out.items():
        ag = c["agg"]
        ntok = max(1, (c["tokens"] - 1) * c["batch"])
        lines.append("| %s | %d | %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %s | %s | %s | %s | %.1f | %s | %s | %.3f | %.3f | %.1f | %.1f | %d |" % (
            tag, c["batch"], c["distinct_docs"], c["K"], ag["spec_rounds"], ag.get("c_mean", float("nan")), ag.get("n_acc_mean_per_request", float("nan")),
            ag.get("yield_lockstep_per_round", float("nan")), ag.get("yield_ragged_projection_per_round", float("nan")),
            _f(ag.get("draft_step_ms")), _f(ag.get("verify_ms")), _f(ag.get("rollback_ms")), _f(ag.get("round_ms")),
            c["tok_s_after_warmup"], _f(c.get("tok_s_window")), _f(c["timely_recall"], "%.3f"),
            c["wire"]["wire_bytes"] / ntok / 1e6, c["wire"]["late_bytes"] / ntok / 1e6, c["wire"]["wasted_bytes"] / 1e6, c["peak_gb"], c["host_syncs"]))
    lines.append("\n## verify round: kernel windows vs torch-op / host-gap windows (ms, mean over timed rounds; the brackets are GPU-timeline windows: a torch-op window is launch-bound small kernels, a host sync is idle time inside plan_ms)\n")
    lines.append("| cell | score (stage-1 + graph + table ops) | gather (late-miss stall, Triton) | attn call 1 | attn call 2 | KERNEL total | tail writes | plan (+1 host sync/layer) | value copies (+2 host syncs/layer) | quotient | NON-KERNEL total | rest (GEMMs + gaps) | total | host syncs/call |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, c in cells_out.items():
        k = c["kernel_host"]
        lines.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %d |" % (
            tag, k["kernel"]["score_ms"], k["kernel"]["gather_ms"], k["kernel"]["attn1_ms"], k["kernel"]["attn2_ms"], k["kernel_total_ms"],
            k["non_kernel"]["tail_ms"], k["non_kernel"]["plan_ms"], k["non_kernel"]["copies_ms"], k["non_kernel"]["quot_ms"], k["non_kernel_total_ms"],
            k["rest_ms"], k["total_ms"], k["host_syncs"]))
    lines.append("\n## draft step: windows (ms, mean over timed steps; gather = side-stream launch + bias scratch build on the main stream, the gathers run on the side stream)\n")
    lines.append("| cell | score | tail write | plan | launch + bias | attention (whole allocation, biased) | rest | total |\n|---|---|---|---|---|---|---|---|")
    for tag, c in cells_out.items():
        k = c["draft_kernel_host"]
        lines.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
            tag, k["kernel"]["score_ms"], k["non_kernel"]["tail_ms"], k["non_kernel"]["plan_ms"], k["kernel"]["gather_ms"],
            float((c["agg"].get("draft_split") or {}).get("attn_ms") or 0.0), k["rest_ms"], k["total_ms"]))
    lines.append("\nfailed gates: %d" % nfail)
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(OUT, "spec_loop.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, "spec_loop.json"), "w") as f:
        json.dump(dict(schedule=SCHEDULE, tau_ctrl=tau, tol=tol, gates=gates, failed=nfail, cells=cells_out, references=refs_out,
                       columns=dict(
                           tok_s_after_warmup="committed tokens x batch / wall seconds from the end of the warm-up decode step to the end of the run (every arm), re-warm steps included",
                           tok_s_window="the same over the rounds after the gate window (NOSI_SPEC_GATE_TOKENS)",
                           yield_lockstep="tokens committed per round under c = 1 + min_b n_acc (the exact lockstep policy this engine allows)",
                           yield_ragged_projection="1 + mean_b n_acc per round: what a per-request (ragged-tail) engine would commit; a PROJECTION, not a measurement",
                           wire_MB_per_tok="(prefetch + late-fetch blocks) x 32 KiB per (layer, head, request) / committed tokens; late_wait blocks are inside prefetch",
                           recall="timely / (timely + late_wait + late_fetch) over the blocks the verify needed that were not resident at round start",
                           verify_split="verify_trace.split_marks windows summed over layers, mean over timed rounds (first spec round excluded)",
                           kernel_host="kernel windows = score (stage-1 + captured graph + small table ops), gather (the two Triton gathers of the late misses), attn call 1, attn call 2; non-kernel = tail writes (strided copies), plan (integer torch ops + one host sync per layer), value copies (elementwise, after two host-sync guards per layer), quotient; rest = GEMMs/norms/rope/lm_head and the host gaps between them")),
                  f, indent=1, default=str)
    return nfail


def _f(v, fmt="%.1f"):
    return "-" if v is None else (fmt % v)


if __name__ == "__main__":
    if MODE == "report":
        sys.exit(report())
    if TOTAL_TOKENS < 2 or GATE_TOKENS < 1:
        die("need NOSI_SPEC_TOTAL_TOKENS >= 2 and NOSI_SPEC_GATE_TOKENS >= 1")
    path = os.environ["NOSI_MODEL_PATH"]
    ids, rows, distinct = load_docs(path)
    print("[docs] %s (%d distinct)  L=%d" % (rows, distinct, L), flush=True)
    if MODE == "spec":
        from nosi import cache_engine as _ce
        if _ce.VERIFY_ROUND_SLOTS <= 0 or _ce.POOL_BLOCKS != 0:
            die("spec mode needs NOSI_VERIFY_ROUND_SLOTS > 0 and NOSI_POOL_BLOCKS = 0 (engine read R=%d P=%d)" % (_ce.VERIFY_ROUND_SLOTS, _ce.POOL_BLOCKS))
        payload = run_spec_process(path, ids, rows, distinct)
    elif MODE == "decode_ref":
        payload = run_decode_ref_process(path, ids, rows, distinct)
    else:
        raise SystemExit("unknown NOSI_SPEC_MODE %r" % MODE)
    fn = os.path.join(OUT, "%s_%s.pt" % (MODE, TAG))
    torch.save(payload, fn)
    print("[spec_loop] saved %s (peak %.2f GB)" % (fn, payload["peak_gb"]), flush=True)
