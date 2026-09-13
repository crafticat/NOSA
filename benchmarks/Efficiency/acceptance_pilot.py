"""NOSA-8B on NOSI: how often does a draft decoded with only PARTIALLY
AVAILABLE sparse KV produce a token the EXACT target accepts?

NOSA and NOSI stay fixed. This is an ACCURACY pilot. Nothing measured here is a
performance claim, and the sequential verification below is explicitly NOT
evidence about batched-verifier speed.

READ THIS FIRST, because it bounds every number in the output.

  1. THE SHIPPED ENGINE NEVER RUNS BELOW 100% AVAILABILITY. NOSI's decode is
     fully synchronous: the two host gathers are launched on the current stream
     and attention is called immediately afterwards on the same stream
     (cache_engine.py:344-345 then nosa_llama.py:610), there is no copy stream
     and no event anywhere in the gather (flash_h2d_mask.py:79-86,
     flash_h2d_mask_bias.py:103-112), and the archived 64-step run has
     attn_after_fetch = 1 on every step. ASYNCHRONOUS FETCHING IS NOT
     IMPLEMENTED. Every point below 100% here is a MODELLED DRAFT POLICY, not a
     measurement of the deployed engine.

  2. TWO SIMULATION MODES, named on EVERY ROW as `sim_mode`, and the manifest's
     caption is built from the modes the grid actually contains.
       FIXED RESIDENCY (d=inf): nothing completes for the whole rollout; the
         availability set is frozen at what was resident when the rollout
         started, AND THE IN-FLIGHT QUEUE IS CLEARED at the start of the rollout
         (avail_policy.begin_rollout) so the fetches the last VERIFIED step
         issued do not land at draft step 1. Without that clearing a d=inf point
         would report a d=1 availability at j=1 under a fixed-residency label.
         The pessimistic bound, and the right model of a draft shorter than one
         PCIe round trip.
       DELAYED ARRIVAL (d stated): a block requested at draft step j is usable
         at step j+d. Completion assumptions, all four: the request is issued at
         the step that first needs the block; it is never cancelled; it is never
         re-issued while in flight; and it lands at the start of step j+d
         whatever else is in flight -- THE LINK IS NOT MODELLED, so d >= 4 is a
         policy point and not an achievable configuration. d=1 is the value at
         which the modelled availability equals the classic LRU hit fraction,
         which is why the whole existing capacity sweep is a d=1 measurement.
         d=0 is the shipped engine and is THE TARGET, never a draft result.

  2b. (b) AND (c) COINCIDE ON THE DELAYED-ARRIVAL FAMILY, and that is a property
     of the model, not an oversight. In VirtualCache.step the residency `before`
     is taken AFTER the step's arrivals, so a block that has not arrived is
     exactly a block that is not resident: with f = 0 and d > 0,
     ready-at-attention EQUALS the modelled LRU hit fraction at that capacity,
     row by row. The two separate only at d = 0 (ready is 1 by definition) and at
     f > 0, which is a partial instant-completion assumption, is labelled
     `policy_instep_completion_f=...`, and is NOT on the default grid. The
     analysis prints both columns and says which is which rather than implying a
     separation the grid does not exercise.

  3. NATURAL LRU OMISSION IS PRIMARY. Capacity moves the availability, and the
     omitted blocks are the ones a real cache omits. Measured on the archived
     real selections at L=16128, d=1: C=54 -> 0.805, C=60 -> 0.897, C=65 ->
     0.952, C=72 -> 0.972, C=160 -> 0.989, d=0 -> 1.000; and C=64..70 give
     0.948/0.952/0.956/0.960/0.963/0.966/0.968, which is the DENSE COVERAGE
     NEAR 95 with no ablation at all. Random omission is available only as a
     MATCHED-AVAILABILITY ABLATION and is labelled as one: real misses are the
     blocks the selection newly reached, and the forced local window plus the
     init block (max_pooling_fused.py:61-63) guarantee they are the FAR blocks.

  4. THE ACCEPTANCE IS SPECULATIVE-SAMPLING ACCEPTANCE. x ~ q, accept with
     min(1, p(x)/q(x)) on the SAMPLED token. The overlap sum_v min(p_v, q_v) is
     reported as the zero-variance form of the same expectation. ARGMAX
     AGREEMENT IS A SEPARATE DIAGNOSTIC and is never called acceptance.

  5. STATE HYGIENE. The draft writes into the live buffers and a SNAPSHOT taken
     at the verified prefix is the scratch boundary (state_snapshot.py). The
     verifier always recomputes its own K and V from the draft's proposed
     tokens; it never reads a byte the draft wrote. Inside a K-token rollout the
     draft is NOT reset between tokens -- that is the point.

  6. RETIRED MoBA / SpecKV ACCEPTANCE NUMBERS ARE NOT NOSI EVIDENCE and appear
     nowhere in this driver or its analysis.

MODES (NOSI_ACC_MODE):
  probe       the kv_bias mask-mechanism device probe (G0). No model weights.
  upstream    G1: a plain teacher-forced trace that prints the sha256 of each
              step's fp32 logits. Run TWICE, once with NOSI_AVAIL unset and once
              with it set to a policy that denies nothing; the two hash lists
              must be identical. It has to be two PROCESSES because the knob is
              read once at import and cannot be unset inside a run, and it has to
              be THIS driver because it is the only thing that calls
              avail_policy.install() -- a trace driver that never installs the
              policy would compare an un-hooked run with an un-hooked run and
              pass while proving nothing.
  gate        G2..G8 on one document. Exits non-zero on the first failure.
  diagnostic  the ONE-STEP diagnostic: one draft step against one target step
              from the same verified prefix, at every availability point.
  rollout     K-token rollouts (NOSI_ACC_K, default "2").

ENVIRONMENT: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_ACC_OUT, NOSI_BENCH_L,
NOSI_ACC_DOCS, NOSI_ACC_PROBES, NOSI_ACC_WARM, NOSI_ACC_K, NOSI_ACC_POINTS,
NOSI_ACC_WORKLOAD, NOSI_ACC_TEMP/TOPP/TOPK, NOSI_ACC_FULL_DIST, NOSI_AVAIL.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.environ.get("NOSI_REPO", "/mnt/central/users/ojerbi/retroinfer-eval")

# The NOSA virtualenv ships its own top-level `scripts` package, which shadows
# this repo's namespace package inside the container
# (env/slurm/nosi_pool.sbatch:72). Load the acceptance mathematics by absolute
# path so there is exactly ONE implementation and it is the one the CPU tests
# exercise.
_spec = importlib.util.spec_from_file_location(
    "nosi_accept_math", os.path.join(REPO, "retro_eval", "nosi_accept_math.py"))
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)

MODE = os.environ.get("NOSI_ACC_MODE", "gate")
OUT = os.environ.get("NOSI_ACC_OUT", "nosi_acceptance")
L = int(os.environ.get("NOSI_BENCH_L", 16128))
NDOCS = int(os.environ.get("NOSI_ACC_DOCS", 4))
NPROBES = int(os.environ.get("NOSI_ACC_PROBES", 8))
WARM = int(os.environ.get("NOSI_ACC_WARM", 8))
KS = tuple(int(x) for x in os.environ.get("NOSI_ACC_K", "2").split(",") if x)
WORKLOAD = os.environ.get("NOSI_ACC_WORKLOAD", "pg19")
TEMP = float(os.environ.get("NOSI_ACC_TEMP", "1.0"))
TOPP = float(os.environ.get("NOSI_ACC_TOPP", "1.0"))
TOPK = int(os.environ.get("NOSI_ACC_TOPK", "0"))
FULL_DIST = os.environ.get("NOSI_ACC_FULL_DIST", "0") == "1"
TOPN = int(os.environ.get("NOSI_ACC_TOPN", "512"))
DOC_OFFSET = int(os.environ.get("NOSI_ACC_DOC_OFFSET", "0"))

# Availability points: '|'-separated NOSI_AVAIL-style specs. The DEFAULT grid is
# the one the offline replay says lands on 80/90/95/97/99/100 at L=16128, with
# SINGLE-SLOT RESOLUTION THROUGH THE 95% BAND (C = 64,65,66,68 measured
# 0.9476/0.9521/0.9564/0.9633 on the archived real selections), which is the
# dense coverage near 95 the brief asks for and needs no ablation at all. Every
# point is natural LRU omission except:
#   d=0        the SHIPPED ENGINE. It is the TARGET and the identity anchor, and
#              it is labelled `identity_anchor_shipped_engine`, never natural.
#   C=63,d=inf FIXED RESIDENCY at the shipped capacity. It is natural omission
#              (the omitted blocks are the ones a frozen real cache omits) and it
#              is the only point on this grid that exercises the second named
#              simulation mode, so the manifest's mode caption is earned.
POINTS = os.environ.get(
    "NOSI_ACC_POINTS",
    "d=0|C=160,d=1|C=72,d=1|C=68,d=1|C=66,d=1|C=65,d=1|C=64,d=1|"
    "C=60,d=1|C=54,d=1|C=63,d=inf")

BLOCK = 64
TAIL_SLOTS = 1
NON_TAIL = 63          # THE DENOMINATOR of every availability rate

# THE GATE POINTS, and they are declared here because they must be REGISTERED
# BEFORE THE DOCUMENT'S FIRST DECODE STEP, exactly like the measurement points.
# A point registered later has an EMPTY virtual LRU while the physical engine is
# already warm, so G5 compares an empty modelled resident set against ~59
# resident slots and dies, and the hygiene and NaN gates silently run at
# ready = 0.0 while printing the capacity they were meant to test.
P_FULL, P_LOW, P_63, P_65 = 900, 901, 902, 903
GATE_POINT_IDS = (P_FULL, P_LOW, P_63, P_65)

# Exit codes, distinct per gate, the way env/slurm/nosi_pool.sbatch does it.
RC_MASK_MECH = 250
RC_UPSTREAM = 251
RC_HUNDRED = 252
RC_K1 = 253
RC_LRU = 254
RC_HOST = 255
RC_HYGIENE = 249
RC_KGT1 = 248
RC_NAN = 247
RC_CROSSK = 246
RC_BUDGET = 245


def die(code: int, msg: str):
    print("[GATE FAIL rc=%d] %s" % (code, msg), flush=True)
    raise SystemExit(code)


def parse_points(spec: str):
    pts = []
    for i, term in enumerate(spec.split("|")):
        term = term.strip()
        if not term:
            continue
        cfg = dict(C=63, d=1.0, f=0.0, ablate="none", ready=1.0)
        for kv in term.split(","):
            k, v = kv.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "C":
                cfg["C"] = int(v)
            elif k == "d":
                cfg["d"] = float("inf") if v.startswith("inf") else float(v)
            elif k == "f":
                cfg["f"] = float(v)
            elif k == "ablate":
                cfg["ablate"] = v
            elif k == "ready":
                cfg["ready"] = float(v)
            else:
                raise ValueError("availability point %r: unknown key %r" % (term, k))
        cfg["id"] = i
        cfg["spec"] = term
        # NATURAL MEANS "a real cache would have omitted exactly these blocks".
        # Three things break that, and the flag must see all three (avail_policy
        # .is_natural carries the same rule and a test pins them together):
        #   ablate != none  a random omission is not a real miss;
        #   f > 0           a fraction of THIS step's own misses is declared
        #                   usable inside the step that requested it -- a partial
        #                   INSTANT-COMPLETION ORACLE with no link model, which
        #                   the brief forbids in the main curve;
        #   d == 0          the full instant-completion oracle, i.e. the shipped
        #                   engine, which is the TARGET and the identity anchor.
        cfg["natural"] = (cfg["ablate"] == "none" and cfg["f"] == 0.0
                          and cfg["d"] != 0)
        cfg["omission"] = omission_label(cfg)
        cfg["sim_mode"] = sim_mode_of(cfg["d"])
        pts.append(cfg)
    return pts


def omission_label(cfg) -> str:
    """The series a point belongs to, as ONE string that the archive, the tables
    and the legend all read. It mirrors avail_policy.AvailPolicy.omission_label
    and a test asserts the two agree, so a point cannot be one series in the
    engine and another in the figure."""
    if cfg["d"] == 0:
        return "identity_anchor_shipped_engine"
    if cfg["ablate"] != "none":
        return "ablation_" + cfg["ablate"]
    if cfg["f"] > 0:
        return "policy_instep_completion_f=%g" % cfg["f"]
    return "natural_lru"


def sim_mode_of(delay: float) -> str:
    """The simulation mode of ONE point, on the row. The manifest used to carry
    a CONSTANT string naming both modes, on a grid that contained neither d=inf
    nor anything but d=1: a caption asserting an arm that was never run."""
    if delay == 0:
        return "shipped_engine"
    if delay == float("inf"):
        return "fixed_residency"
    return "delayed_arrival_d=%g" % delay


# ---------------------------------------------------------------------------
# G0: the kv_bias mask mechanism, on the device, BEFORE any model weight loads.
# ---------------------------------------------------------------------------
def gate_mask_mechanism() -> int:
    """Two checks, both cheap, both necessary.

    (1) EXCLUSION. A large negative FINITE bias on the second 64-row block must
        make a cache_seqlens=128 call equal a cache_seqlens=64 call. That is the
        whole mask mechanism: the bias is added to the pre-softmax score
        (flash_fwd_kernel.h:342 and :971) and a masked block must carry zero
        softmax weight and produce no NaN.

    (2) HEAD MAPPING. One of this pilot's two design notes claimed kv_bias is
        GQA-misindexed -- that query head j reads the bias of KV head j%2, up to
        15 columns away -- from flash_fwd_kernel.h:339-341 (`bidh *
        kv_bias_head_stride`) against :655-656 (`bidh / h_h_k_ratio` for K/V).
        THAT IS NOT THIS CALL PATH: flash_api.cpp:351-357 sets
        seqlenq_ngroups_swapped for seqlen_q == 1 with num_heads > num_heads_k,
        both window sizes negative and no alibi -- all true here -- and then
        num_heads := num_heads_k, so h_h_k_ratio is 1 at :100 and `bidh` IS the
        KV head. This probe puts a one-hot bias on ONE (column, kv head) and
        requires the output to move for exactly the 16 query heads of that KV
        head, at that column and no other. Reading a kernel is not running one.
    """
    from flash_attn_nosa import flash_attn_with_kvcache as fa
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    B, S, HQ, HK, D = 1, 128, 32, 2, 128
    q = torch.randn(B, 1, HQ, D, device=dev, dtype=dt)
    k = torch.randn(B, S, HK, D, device=dev, dtype=dt)
    v = torch.randn(B, S, HK, D, device=dev, dtype=dt)
    bias = torch.zeros(B, S, HK, device=dev, dtype=dt)

    o_short = fa(q, k[:, :64].contiguous(), v[:, :64].contiguous(),
                 bias[:, :64].contiguous(),
                 cache_seqlens=torch.full((B,), 64, dtype=torch.int32, device=dev))
    from nosi.avail_policy import MASK_BIAS
    bias_m = bias.clone()
    bias_m[:, 64:, :] = MASK_BIAS
    o_masked = fa(q, k, v, bias_m,
                  cache_seqlens=torch.full((B,), 128, dtype=torch.int32, device=dev))
    ok_nan = bool(torch.isfinite(o_masked).all())
    d1 = (o_masked.float() - o_short.float()).abs().max().item()
    print("[G0.1] masked-block exclusion: max|masked(128) - plain(64)| = %.3e  finite=%s"
          % (d1, ok_nan), flush=True)

    o_ref = fa(q, k, v, bias,
               cache_seqlens=torch.full((B,), 128, dtype=torch.int32, device=dev))
    col, kvh = 37, 1
    bias_h = bias.clone()
    bias_h[0, col, kvh] = -20.0
    o_h = fa(q, k, v, bias_h,
             cache_seqlens=torch.full((B,), 128, dtype=torch.int32, device=dev))
    moved = (o_h.float() - o_ref.float()).abs().reshape(HQ, D).amax(dim=-1)
    grp = HQ // HK
    want = torch.zeros(HQ, dtype=torch.bool)
    want[kvh * grp:(kvh + 1) * grp] = True
    got = moved > 1e-3
    ok_head = bool((got.cpu() == want).all())
    print("[G0.2] one-hot bias at (col=%d, kv_head=%d) moved query heads %s; expected %s -> %s"
          % (col, kvh, got.nonzero().flatten().tolist(), want.nonzero().flatten().tolist(),
             "OK" if ok_head else "GQA MISINDEXING PRESENT"), flush=True)

    if not (ok_nan and d1 < 5e-2 and ok_head):
        print("[G0] the mask mechanism is NOT usable on this kernel. Run the pilot "
              "with mech=stale and label it the stale-occupant arm.", flush=True)
        return RC_MASK_MECH
    print("[G0] PASS: kv_bias excludes a block cleanly and indexes the KV head correctly.",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# workloads
# ---------------------------------------------------------------------------
def load_documents(tokenizer, need: int):
    """Return (ids (N, L+need), source rows). CONTROL is PG-19 book
    continuation, the workload every existing NOSI trace used, with the same
    selection rule (trace_selections.py:49-62). The other two workloads change
    the SELECTION pattern -- which blocks the top-64 reaches and how fast the
    set drifts -- and that is the only reason to vary them: acceptance is a
    property of two distributions from the SAME model, so the model does not
    have to be good at summarisation."""
    want = L + need
    rows, ids = [], []
    if WORKLOAD == "pg19":
        from datasets import load_dataset
        pq = os.environ["NOSI_PG19_PARQUET"]
        data = load_dataset("parquet", data_files=pq)["train"]["text"]
        skipped = 0
        for i in range(len(data)):
            t = tokenizer(data[i], return_tensors="pt").input_ids
            if t.shape[1] < want:
                continue
            if skipped < DOC_OFFSET:
                skipped += 1
                continue
            rows.append(i)
            ids.append(t[0, :want])
            if len(rows) == NDOCS:
                break
    elif WORKLOAD in ("gov_report", "qmsum", "sections"):
        task = "gov_report" if WORKLOAD == "sections" else WORKLOAD
        path = os.path.join(REPO, "data", "longbench", task + ".jsonl")
        texts = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ctx = (rec.get("context") or "").strip()
                if ctx:
                    texts.append(ctx)
        instr = (("\n\nList each section of the document above and give one line of "
                  "extracted fact per section.\n\nSection 1:")
                 if WORKLOAD == "sections" else
                 "\n\nWrite a summary of the report above.\n\nSummary:")
        # TRUNCATE THE CONTEXT, NOT THE PROMPT. `ids.append(t[0, :want])` on
        # tokenize(ctx + instr) cuts the instruction off the END of every
        # document long enough to survive the length filter, so "sections" and
        # "gov_report" fed BYTE-IDENTICAL token streams under two workload names
        # and neither was a structured-extraction task -- both were plain
        # continuation of a government report.
        instr_ids = tokenizer(instr, return_tensors="pt",
                              add_special_tokens=False).input_ids[0]
        n_instr = int(instr_ids.shape[0])
        if n_instr >= want:
            raise SystemExit("the %s instruction is %d tokens, >= the %d-token prompt"
                             % (WORKLOAD, n_instr, want))
        skipped = 0
        for i, ctx in enumerate(texts):
            ctx_ids = tokenizer(ctx, return_tensors="pt").input_ids[0]
            if int(ctx_ids.shape[0]) + n_instr < want:
                continue
            if skipped < DOC_OFFSET:
                skipped += 1
                continue
            prompt_ids = torch.cat((ctx_ids[:want - n_instr], instr_ids))
            # the instruction MUST be the last tokens the model reads, or the
            # workload label names a task the prompt does not ask for
            if not torch.equal(prompt_ids[-n_instr:], instr_ids):
                raise SystemExit("workload %s: the instruction is not a suffix of the "
                                 "prompt after truncation" % WORKLOAD)
            rows.append(i)
            ids.append(prompt_ids[:want])
            if len(rows) == NDOCS:
                break
    else:
        raise ValueError("unknown NOSI_ACC_WORKLOAD %r" % WORKLOAD)
    if len(rows) != NDOCS:
        raise SystemExit("only %d %s documents with >= %d tokens after offset %d"
                         % (len(rows), WORKLOAD, want, DOC_OFFSET))
    return torch.stack(ids), rows


# ---------------------------------------------------------------------------
# the pilot
# ---------------------------------------------------------------------------
class Pilot:
    def __init__(self, require_policy: bool = True):
        from nosi import NOSALlama as Llama
        from nosi import avail_policy as ap
        from nosi import state_snapshot as ss
        from nosi.cache_engine import InfLLMv2Cache, POOL_BLOCKS
        if POOL_BLOCKS != 0:
            raise SystemExit("NOSI_POOL_BLOCKS=%d: the availability pilot runs with the "
                             "victim pool OFF (the pooled decode hands attention a "
                             "narrowed view, and pool_update can park the victim of a "
                             "fetch the policy denies)." % POOL_BLOCKS)
        self.ap, self.ss = ap, ss
        self.InfLLMv2Cache = InfLLMv2Cache
        self.path = os.environ["NOSI_MODEL_PATH"]
        self.model = Llama(model_name=self.path, device="cuda", offload=True)
        # ONE knob. avail_policy reads NOSI_AVAIL at import, so the same value
        # that arms the constructor refusals in cache_engine is the value that
        # builds the policy here; there is no second place to disagree.
        self.policy = ap.install()
        if self.policy is None and require_policy:
            raise SystemExit(
                "NOSI_AVAIL is unset, so the fork is upstream-exact and no availability "
                "can be restricted. Set it, e.g. NOSI_AVAIL='mech=mask,C=63,d=1'.")
        self.points = parse_points(POINTS)
        _np = os.environ.get("NOSI_ACC_NPOINTS")
        if _np is not None and int(_np) != len(self.points):
            raise SystemExit(f"NOSI_ACC_POINTS carries {len(self.points)} point(s) but the host "
                             f"passed {_np}: the grid was TRUNCATED in transit (apptainer --env splits "
                             f"on commas; use APPTAINERENV_). Refusing to measure a partial grid.")
        # THE LOWEST-AVAILABILITY MEASUREMENT POINT is what the hygiene and NaN
        # gates must exercise, so it is chosen HERE, from the parsed grid, before
        # anything is registered. A grid with no delayed point falls back to the
        # calibrated 80% cell.
        low = min((q for q in self.points if q["d"] != 0),
                  key=lambda q: q["C"], default=None)
        self.gate_points = {
            P_FULL: _pt(P_FULL, 63, 0.0),                 # d=0 IS the shipped engine
            P_63: _pt(P_63, 63, 1.0),
            P_65: _pt(P_65, 65, 1.0),
            P_LOW: (_pt(P_LOW, low["C"], low["d"], low["f"], low["ablate"], low["ready"])
                    if low is not None else _pt(P_LOW, 54, 1.0)),
        }
        if self.policy is not None:
            self.policy.clear_points()
            for pt in self.points:
                self.policy.register_point(pt["id"], pt["C"], pt["d"], pt["f"],
                                           pt["ablate"], pt["ready"])
            # REGISTERED WITH THE MEASUREMENT POINTS, NOT INSIDE run_gates.
            # new_document() clears every point's streams and the ROLE_WARM
            # branch advances every point that EXISTS at that warm step, so a
            # point registered after the warm-up never sees the verified
            # trajectory. In gate mode only, to keep the measurement cells from
            # paying for four virtual caches per (layer, KV head, request) that
            # nothing reads.
            if MODE == "gate":
                for pt in self.gate_points.values():
                    self.policy.register_point(pt["id"], pt["C"], pt["d"], pt["f"],
                                               pt["ablate"], pt["ready"])
        self.cu = torch.arange(0, 1 + 1, dtype=torch.int, device="cuda")
        self.rows = []
        self.vectors = []
        self.avail_rows = []

    # -- per document ------------------------------------------------------
    def open_document(self, prompt):
        # RELEASE THE PREVIOUS DOCUMENT FIRST. Each document pins ~800 MB of
        # host KV at L=16128 B=1 (_k_cpu/_v_cpu are (B, S+8192, H, D) bf16 per
        # layer, cache_engine.py:171-172) and the old InfLLMv2Cache is cyclic
        # garbage the refcount does not free -- the same hazard that SIGKILLed
        # job 2170892 on the pool cells.
        self.cache = None
        self.snap = None
        gc.collect()
        torch.cuda.empty_cache()
        cache = self.InfLLMv2Cache(config=self.model.config,
                                   num_hidden_layers=self.model.config.num_hidden_layers,
                                   has_kv_bias=True)
        t0 = time.time()
        logits, position_ids = self.model.batch_prefill(prompt, cache)
        torch.cuda.synchronize()
        self.prefill_s = time.time() - t0
        self.cache = cache
        self.pos = position_ids[:, -1:] + 1
        self.snap = self.ss.CacheSnapshot(cache)
        self.host_digest = self.host_digest_now()
        return logits

    def host_digest_now(self) -> str:
        """THE WINDOW G8 ACTUALLY WATCHES. A digest over [0, L) cannot change:
        decode's only host writer is the tail write-back, which writes rows
        [seq-64, seq) with seq a multiple of 64 and seq > L, and the gathers are
        host->device only. This window is [L-64, L+128) -- the first rows a
        write-back could ever reach, and the rows the snapshot's own host window
        saves and restores -- so a mis-sized restore or an escaped write-back
        moves it. At reach <= 63 the write-back cannot fire at all (see
        budget_or_die), and what the digest then proves is that the host window
        save/restore corrupted nothing, which is said plainly rather than
        dressed up as write-back coverage."""
        return self.ss.host_kv_digest(self.cache, L + 2 * BLOCK, frm=max(0, L - BLOCK))

    def step(self, token, role: str):
        """ONE decode step. The role is set BEFORE the call, so the hook inside
        decode_inference never sees anything about the token."""
        if self.policy is not None:
            self.policy.set_role(role)
        logits = self.model.decode_inference(token, self.cu, self.pos, self.cache)
        self.pos = self.pos + 1
        return logits[:, -1, :].float()

    # -- distributions ------------------------------------------------------
    @staticmethod
    def dist(logits_row):
        return am.normalise(logits_row.cpu().numpy().astype(np.float64),
                            temperature=TEMP, top_k=TOPK, top_p=TOPP)

    # -- one measurement site ----------------------------------------------
    def rollout(self, doc, probe, first_token, K, point):
        """(1) snapshot; (2) K DRAFT steps at restricted availability, feeding
        the draft its OWN samples and NEVER resetting between them; (3) restore;
        (4) K TARGET steps, teacher-forced on the draft's tokens, at full
        availability, so the target recomputes its own selections and its own K
        and V -- that they may differ from the draft's is part of the
        phenomenon; (5) restore."""
        self.snap.take()
        pos0 = self.pos.clone()
        self.policy.set_site(probe)
        self.policy.begin_rollout(point["id"])
        rec0 = len(self.policy.records)

        def draft_step(j, prev):
            tok = first_token if prev is None else torch.tensor(
                [[prev]], dtype=torch.long, device="cuda")
            lg = self.step(tok, self.ap.ROLE_DRAFT)
            if not bool(torch.isfinite(lg).all()):
                die(RC_NAN, "draft logits contain NaN/Inf at doc=%d probe=%d j=%d point=%s"
                    % (doc, probe, j, point["spec"]))
            return self.dist(lg[0])

        # NO reset_fn: the draft is NOT put back to the target state between the
        # K tokens. Accumulating its own approximate KV, its own compressed
        # keys, its own cis and its own selections is the point.
        qs, xs = am.draft_rollout(draft_step, K, doc, probe)
        draft_records = self.policy.records[rec0:]

        self.snap.restore()
        self.pos = pos0.clone()
        self.policy.end_rollout()

        def target_step(j, prev):
            tok = first_token if prev is None else torch.tensor(
                [[prev]], dtype=torch.long, device="cuda")
            return self.dist(self.step(tok, self.ap.ROLE_TARGET)[0])

        ps = am.verify_sequential(target_step, xs)

        self.snap.restore()
        self.pos = pos0.clone()
        return qs, ps, xs, draft_records

    @staticmethod
    def avail_of(records):
        """READY AT THE INSTANT THE DRAFT ATTENDS, and the two other quantities
        that must never be conflated with it, per draft step and per (layer,
        head). Denominator: 63 non-tail slots per (layer, KV head, request)."""
        per_j = {}
        for r in records:
            call = r[0]
            per_j.setdefault(call, []).append(r)
        out = []
        for j, (call, rs) in enumerate(sorted(per_j.items()), start=1):
            sel = sum(x[7] for x in rs)
            out.append(dict(
                j=j,
                sel_slots=sel,
                ready_slot_frac=sum(x[11] for x in rs) / sel if sel else float("nan"),
                lru_hit_frac_physical=sum(x[8] for x in rs) / sel if sel else float("nan"),
                lru_hit_frac_modelled=(sum(max(x[9], 0) for x in rs) / sel) if sel else float("nan"),
                denied=sum(x[10] for x in rs),
                unexpressible=sum(x[12] for x in rs)))
        return out

    def emit(self, doc, doc_row, probe, K, point, qs, ps, xs, avail):
        for j in range(1, K + 1):
            p, q, x = ps[j - 1], qs[j - 1], xs[j - 1]
            a, ratio = am.acceptance(p, q, x)
            ov, tv = am.overlap_and_tv(p, q)
            urng = np.random.default_rng(am.seed_accept(doc, probe, j))
            u = float(urng.random())
            ap_ = int(np.argmax(p))
            aq = int(np.argmax(q))
            av = avail[j - 1] if j - 1 < len(avail) else {}
            row = dict(
                workload=WORKLOAD, doc=doc, doc_row=doc_row, context=L, probe=probe,
                K=K, j=j, point_id=point["id"], point=point["spec"],
                capacity=point["C"], delay=("inf" if point["d"] == float("inf") else point["d"]),
                instep=point["f"], mechanism=self.policy.mech,
                omission=point["omission"], natural=int(point["natural"]),
                sim_mode=point["sim_mode"],
                temperature=TEMP, top_p=TOPP, top_k=TOPK,
                truncation_order="temperature,top_k,top_p", kept_crossing_token=1,
                dtype="float32_logits_float64_math",
                rng_seed_token=am.seed_token(doc, probe, j),
                rng_seed_accept=am.seed_accept(doc, probe, j),
                x=x, q_of_x=float(q[x]), p_of_x=float(p[x]),
                ratio=ratio, accept_prob=a, u=u, accept_draw=int(u < a),
                overlap=ov, tv=tv, overlap_identity_err=abs(ov - (1.0 - tv)),
                argmax_p=ap_, argmax_q=aq, argmax_agree=int(ap_ == aq),
                entropy_p=am.entropy(p), entropy_q=am.entropy(q),
                p_truncated_zero=int(p[x] == 0.0),
                nucleus_size_p=int((p > 0).sum()), nucleus_size_q=int((q > 0).sum()),
                rank_of_x_under_p=int((p > p[x]).sum()),
                rank_of_x_under_q=int((q > q[x]).sum()),
                sel_slots=av.get("sel_slots", -1),
                ready_slot_frac=av.get("ready_slot_frac", float("nan")),
                lru_hit_frac_physical=av.get("lru_hit_frac_physical", float("nan")),
                lru_hit_frac_modelled=av.get("lru_hit_frac_modelled", float("nan")),
                denied=av.get("denied", -1),
                unexpressible=av.get("unexpressible", -1),
                verification="sequential (accuracy pilot only; NOT evidence about "
                             "batched-verifier speed)",
            )
            if row["overlap_identity_err"] > 1e-9:
                die(RC_HYGIENE, "overlap != 1 - TV by %.3e at doc=%d probe=%d j=%d"
                    % (row["overlap_identity_err"], doc, probe, j))
            # THE IDENTITY ANCHOR IS RE-CHECKED ON EVERY DOCUMENT AND EVERY
            # PROBE, not once inside the gates. d=0 denies nothing, so the draft
            # forward IS the target forward from the same restored state: q must
            # equal p bitwise and the acceptance must be exactly 1. A state leak
            # that only appears on document 2 would otherwise be written to
            # rows.jsonl and plotted as the 100% anchor.
            if point["d"] == 0 and not (np.array_equal(p, q) and a == 1.0):
                die(RC_K1, "the d=0 identity anchor is not the identity at doc=%d "
                           "probe=%d K=%d j=%d: accept=%r max|p-q|=%.3e -- a state "
                           "leak or a numeric fault, NOT an availability effect"
                    % (doc, probe, K, j, a, float(np.abs(p - q).max())))
            # THE ROUND IS A CORRECT SPECULATIVE ROUND. On a rejection the token
            # is resampled from the residual p' = normalise(max(0, p - q)); only
            # the scalar summary is stored and the bonus token is NEVER counted
            # in the accepted-prefix length or quoted as throughput.
            if row["accept_draw"] == 0:
                pr, mass = am.residual(p, q)
                rrng = np.random.default_rng(am.seed_accept(doc, probe, j) ^ 0x9E3779B9)
                xr = am.sample(pr, rrng)
                row["residual_mass"] = float(mass)
                row["residual_token"] = int(xr)
                row["residual_identity_err"] = abs(float(mass) - tv)
            else:
                row["residual_mass"] = float("nan")
                row["residual_token"] = -1
                row["residual_identity_err"] = float("nan")
            # THE SAMPLED TOKEN IS FORCED INTO THE DUMP. x is drawn from the FULL
            # q, so it can sit outside both top-N sets; without it vectors.pt
            # cannot reproduce the ratio p(x)/q(x) this pilot is about. The flag
            # is written BEFORE the row is appended so the archive never depends
            # on a later mutation of an object already handed away.
            idx, pv, qv, ptail, qtail = am.topn_union(p, q, TOPN, must_include=(x,))
            row["x_in_dump"] = int(x in set(idx.tolist()))
            row["dump_index_size"] = int(idx.size)
            self.rows.append(row)
            vec = dict(doc=doc, probe=probe, K=K, j=j, point_id=point["id"],
                       x=x, idx=idx, p=pv, q=qv, p_tail=ptail, q_tail=qtail)
            if FULL_DIST:
                vec["p_full"] = p.astype(np.float32)
                vec["q_full"] = q.astype(np.float32)
            self.vectors.append(vec)

    def write(self, meta):
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "rows.jsonl"), "a") as fh:
            for r in self.rows:
                fh.write(json.dumps(r) + "\n")
        torch.save(dict(vectors=self.vectors, meta=meta), os.path.join(OUT, "vectors.pt"))
        if self.avail_rows:
            np.savez_compressed(os.path.join(OUT, "avail.npz"),
                                rows=np.asarray(self.avail_rows, dtype=np.int64),
                                header=np.asarray(list(self.policy.rows_header)))
        with open(os.path.join(OUT, "manifest.json"), "w") as fh:
            json.dump(meta, fh, indent=1, default=str)


def budget_reason(n_verified_steps: int, kmax: int) -> str:
    """THE BUDGET IS ON THE MAXIMUM SEQUENCE LENGTH EVER REACHED, not on the
    number of forward calls, because draft steps are rolled back. NOSI sizes the
    pooling and mask buffers ONCE at the first decode step and the flag is
    sticky (nosa_llama.py:777 `warmup = not self.has_buffers`, :779-791, :793),
    so from a block-aligned prompt exactly 64 steps of SEQUENCE ADVANCE fit.

    Returns the reason a configuration is out of budget, or "" when it fits. The
    hygiene gate needs to ASK before it moves its site, rather than die."""
    reach = n_verified_steps + kmax
    if L % BLOCK != 0:
        return "L=%d is not a multiple of 64 (limit b)" % L
    if reach > 63:
        return ("verified steps %d + K_max %d = %d > 63: the decode would cross the next "
                "64-token block boundary and the warm-up-sized pooling buffer"
                % (n_verified_steps, kmax, reach))
    compressed = L // 16 - 1
    room = ((compressed + 127) // 128) * 128 - compressed
    if room * 16 < reach:
        return ("limit (c): L=%d leaves room for %d compressed keys (%d tokens) before "
                "score_buf's 128 boundary, < %d" % (L, room, room * 16, reach))
    if L // BLOCK < 128:
        return ("L=%d is under 128 blocks; topk_idx can degenerate and diff_offload's "
                "distinctness argument no longer holds" % L)
    return ""


def budget_or_die(n_verified_steps: int, kmax: int):
    why = budget_reason(n_verified_steps, kmax)
    if why:
        die(RC_BUDGET, why)


def run_upstream() -> int:
    """G1. A plain teacher-forced trace whose per-step fp32 logits are hashed.
    With NOSI_AVAIL unset the policy is not installed and NOT ONE LINE of the
    guard bodies in cache_engine.py or nosa_llama.py executes; with it set to a
    policy that denies nothing, every hook body runs and must change nothing.
    The two runs are two processes, and the sbatch diffs their hash lists."""
    steps = int(os.environ.get("NOSI_ACC_UPSTREAM_STEPS", "12"))
    pilot = Pilot(require_policy=False)
    ids, doc_rows = load_documents(pilot.model.tokenizer, steps + 2)
    ids = ids.to("cuda")
    prompt, forced = ids[0:1, :L], ids[0:1, L:]
    pilot.open_document(prompt)
    os.makedirs(OUT, exist_ok=True)
    lines = []
    for t in range(steps):
        lg = pilot.step(forced[:, t:t + 1], pilot.ap.ROLE_WARM)
        h = hashlib.sha256(lg.cpu().contiguous().numpy().tobytes()).hexdigest()
        lines.append("%d,%s" % (t, h))
        print("[upstream] step %d logits_sha256 %s" % (t, h), flush=True)
    tag = "set" if pilot.policy is not None else "unset"
    with open(os.path.join(OUT, "logits_hashes.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[upstream] NOSI_AVAIL %s (policy %s), %d steps, doc row %d -> %s"
          % (os.environ.get("NOSI_AVAIL", ""), tag, steps, doc_rows[0],
             os.path.join(OUT, "logits_hashes.txt")), flush=True)
    return 0


def main() -> int:
    if MODE == "probe":
        return gate_mask_mechanism()
    if MODE == "upstream":
        return run_upstream()

    kmax = max(KS)
    probes = list(range(WARM + 1, WARM + 1 + NPROBES))
    budget_or_die(1 + WARM + NPROBES, kmax)

    pilot = Pilot()
    need = 1 + WARM + NPROBES + kmax + 2
    ids, doc_rows = load_documents(pilot.model.tokenizer, need)
    ids = ids.to("cuda")
    meta = dict(workload=WORKLOAD, context=L, docs=doc_rows, n_docs=NDOCS,
                probes=probes, warm=WARM, K=list(KS), mode=MODE,
                points=[p["spec"] for p in pilot.points],
                mechanism=pilot.policy.mech,
                sampling=dict(temperature=TEMP, top_p=TOPP, top_k=TOPK,
                              order="temperature,top_k,top_p",
                              kept_crossing_token=True, dtype="fp32 logits, fp64 math"),
                vector_dump=("full fp32 vocabulary" if FULL_DIST else
                             "union of top-%d of p and top-%d of q on ONE shared index "
                             "set, plus each vector's residual tail mass" % (TOPN, TOPN)),
                availability_denominator="63 non-tail slots per (layer, KV head, request, step)",
                async_fetch_implemented=False,
                # BUILT FROM THE GRID THAT WILL ACTUALLY RUN, never a constant.
                # A caption naming both simulation modes on a run that contains
                # one is a labelling claim the artefact does not support.
                simulation_modes=sorted({p["sim_mode"] for p in pilot.points}),
                omission_series=sorted({p["omission"] for p in pilot.points}),
                completion_assumptions=(
                    "delayed arrival: the request is issued at the step that first needs "
                    "the block; never cancelled; never re-issued while in flight; lands at "
                    "the start of step j+d whatever else is in flight -- THE LINK IS NOT "
                    "MODELLED. fixed residency: the in-flight queue is CLEARED at the start "
                    "of the rollout, so nothing the last verified step requested lands "
                    "during it. shipped_engine (d=0): every request completes before the "
                    "step's own attention; it is the TARGET and the identity anchor."),
                b_and_c_coincide=(
                    "(b) the modelled LRU hit fraction at capacity C and (c) ready at "
                    "attention are EQUAL BY CONSTRUCTION at every point with f=0 and d>0, "
                    "because a block that has not arrived is exactly a block that is not "
                    "resident (avail_policy.VirtualCache.step). They separate only at d=0 "
                    "(c=1 by definition) and at f>0, which is a labelled policy assumption "
                    "and is not on the default grid."),
                verification="sequential; NOT evidence about batched-verifier speed",
                model=os.path.basename(pilot.path.rstrip("/")))
    print("[setup] " + json.dumps({k: meta[k] for k in
                                   ("workload", "context", "n_docs", "probes", "K",
                                    "points", "mechanism")}), flush=True)

    n_fail = 0
    for d in range(NDOCS):
        prompt = ids[d:d + 1, :L]
        forced = ids[d:d + 1, L:]
        pilot.policy.new_document(d)
        pilot.open_document(prompt)
        print("[doc %d] row=%d prefill %.1fs" % (d, doc_rows[d], pilot.prefill_s), flush=True)
        # step 0 is the warm-up: it sizes the per-step buffers and CAPTURES the
        # pooling graph (nosa_llama.py:777-793, :448-476). Run it once per
        # document, as a verified step with the policy denying nothing, and
        # never re-enter that path afterwards -- a recapture against different
        # buffers makes nothing after it comparable.
        pilot.step(forced[:, 0:1], pilot.ap.ROLE_WARM)
        trans = pilot.ss.transient_ids(pilot.model)
        for t in range(1, 1 + WARM):
            pilot.step(forced[:, t:t + 1], pilot.ap.ROLE_WARM)

        if MODE == "gate":
            # one document is enough for every gate, and a gate job must not
            # spend GPU time on documents it will not measure
            rc = run_gates(pilot, forced, trans)
            pilot.write(meta)
            return rc

        for probe in probes:
            first = forced[:, probe:probe + 1]
            for K in (KS if MODE == "rollout" else (1,)):
                for point in pilot.points:
                    qs, ps, xs, recs = pilot.rollout(d, probe, first, K, point)
                    av = pilot.avail_of(recs)
                    for r in recs:
                        # point_id (r[14]) and the draft position (r[15]) are the
                        # two columns without which per_layer_head() pools an 80%
                        # arm with a 100% arm and calls the mixture "the
                        # per-layer variation of availability".
                        pilot.avail_rows.append([r[0], r[1], r[2],
                                                 {"target": 0, "draft": 1, "warm": 2}[r[3]],
                                                 r[4], r[5], r[6], r[7], r[8], r[9],
                                                 r[10], r[11], r[12], r[13],
                                                 r[14], r[15]])
                    pilot.emit(d, doc_rows[d], probe, K, point, qs, ps, xs, av)
            pilot.step(first, pilot.ap.ROLE_WARM)     # advance the verified trajectory
        pilot.ss.assert_transients_intact(pilot.model, trans)
        got = pilot.host_digest_now()
        if got != pilot.host_digest:
            die(RC_HOST, "G8: the host K/V over [%d,%d) changed during document %d -- a "
                         "draft write-back escaped the snapshot, or the snapshot's host "
                         "window was mis-sized" % (max(0, L - BLOCK), L + 2 * BLOCK, d))
        print("[doc %d] rows=%d" % (d, len(pilot.rows)), flush=True)

    pilot.write(meta)
    print("[done] %d distribution rows -> %s" % (len(pilot.rows), OUT), flush=True)
    return n_fail


def _pt(pid, C, d, f=0.0, ablate="none", ready=1.0):
    cfg = dict(id=pid, spec="C=%s,d=%s" % (C, d), C=C, d=float(d), f=f,
               ablate=ablate, ready=ready)
    cfg["natural"] = (ablate == "none" and f == 0.0 and float(d) != 0)
    cfg["omission"] = omission_label(cfg)
    cfg["sim_mode"] = sim_mode_of(float(d))
    return cfg


def run_gates(pilot, forced, trans) -> int:
    """G2..G8, in order, on one document. Nothing downstream runs on a failure.

    G0 (the kv_bias mask mechanism) runs in mode `probe`, before any model
    weight is loaded, and G1 (the fork is a no-op with NOSI_AVAIL unset) is a
    two-process comparison the sbatch makes from the logits archive: neither can
    be done from inside this process.
    """
    ap = pilot.ap
    tok = forced[:, WARM + 1:WARM + 2]
    probe = WARM + 1
    kmax = max(KS)
    # P_LOW MIRRORS THE LOWEST-AVAILABILITY MEASUREMENT POINT and is registered
    # in Pilot.__init__ from the parsed grid, so it warms on the same verified
    # trajectory at the same capacity. It is a separate id on purpose: run_gates
    # then works on a grid that contains no delayed point at all, and the gates
    # never depend on which measurement ids happen to exist.
    low = pilot.gate_points[P_LOW]

    # THE GATE POINTS ARE ALREADY REGISTERED, in Pilot.__init__, BEFORE the
    # document's warm-up and WARM verified steps. This self-check makes the old
    # mistake -- registering them here, after the warm loop, so that every one
    # entered its rollout with an EMPTY virtual LRU -- impossible to repeat
    # silently: G5 then compared an empty modelled resident set against the
    # engine's ~59 resident slots and killed the job before any cell ran, while
    # the hygiene gate and G6 ran at ready = 0.0 under a label saying 0.80.
    for pid in GATE_POINT_IDS:
        if pid not in pilot.policy.points:
            die(RC_LRU, "gate point %d was never registered: register every point in "
                        "Pilot.__init__, before the first decode step" % pid)
        if not pilot.policy.points[pid]["streams"]:
            die(RC_LRU, "gate point %d has an EMPTY virtual LRU after %d verified decode "
                        "steps: it was registered after the warm-up and never saw the "
                        "verified trajectory, so it would measure ready = 0 at capacity %d"
                % (pid, pilot.policy.call + 1, pilot.policy.points[pid]["C"]))
    print("[gates] all %d gate points warmed on the verified trajectory (%d streams each, "
          "lowest availability point %s)"
          % (len(GATE_POINT_IDS), len(pilot.policy.points[P_63]["streams"]), low["spec"]),
          flush=True)

    # ---- G2: the hook must be a NO-OP at 100% availability, bitwise. --------
    # "By construction" arguments have failed in this tree before (job 2173299),
    # so it is executed rather than argued.
    pilot.snap.take()
    pos0 = pilot.pos.clone()
    prev = ap.uninstall()
    lg_up = pilot.model.decode_inference(tok, pilot.cu, pilot.pos,
                                         pilot.cache).float().clone()
    ap.reinstall(prev)
    pilot.snap.restore(); pilot.pos = pos0.clone()
    pilot.policy.set_site(probe); pilot.policy.begin_rollout(P_FULL)
    lg_100 = pilot.step(tok, ap.ROLE_DRAFT).clone()
    pilot.policy.end_rollout()
    pilot.snap.restore(); pilot.pos = pos0.clone()
    if not torch.equal(lg_up.reshape(-1), lg_100.reshape(-1)):
        die(RC_HUNDRED, "G2: the hook is not a no-op at 100%% availability "
                        "(max |delta| = %.3e)"
            % (lg_up.reshape(-1) - lg_100.reshape(-1)).abs().max().item())
    print("[G2] PASS: at 100%% availability the logits are bit-identical to the "
          "un-hooked path (fp32, %d entries)" % lg_up.numel(), flush=True)

    # ---- G3: at 100% availability and K = 1 the draft IS the target. --------
    qs, ps, xs, _ = pilot.rollout(0, probe, tok, 1, pilot.gate_points[P_FULL])
    q, p, x = qs[0], ps[0], xs[0]
    a, _ = am.acceptance(p, q, x)
    ov, _ = am.overlap_and_tv(p, q)
    # THE SAMPLED-TOKEN LEG IS RUN, NOT ASSUMED. The verifier is teacher-forced
    # on the draft's x, so no target-side sample exists unless one is drawn here.
    # p == q bitwise implies it under the shared seed, but the gate used to PRINT
    # "same sampled token" for a check it never ran -- a claim that would survive
    # a later refactor (a point-dependent seed, say) after the implication had
    # stopped holding.
    x_p = am.sample(p, np.random.default_rng(am.seed_token(0, probe, 1)))
    if not np.array_equal(p, q) or a != 1.0 or abs(1.0 - ov) > 1e-9 or x_p != x:
        die(RC_K1, "G3: K=1 at 100%% availability is not the identity "
                   "(accept=%r overlap=%.12f max|p-q|=%.3e draft token %d vs target "
                   "token %d)"
            % (a, ov, float(np.abs(p - q).max()), x, x_p))
    print("[G3] PASS: K=1 at 100%% availability -- q == p over all %d entries, "
          "acceptance exactly 1.0, overlap 1.0, and the token sampled from p under the "
          "same seed is the same token %d the draft sampled from q" % (p.size, x),
          flush=True)

    # ---- STATE HYGIENE: L1 == L2 around a rollout. -------------------------
    # WHAT THIS GATE DOES AND DOES NOT COVER, stated instead of asserted.
    #   COVERED at any site: every tensor and scalar in state_snapshot's list
    #     that a plain decode step mutates.
    #   NOT REACHABLE AT ALL in a legal cell: the host tail write-back
    #     (cache_engine.py:352-353). prefill_update sets _tail_block_len_on_gpu
    #     = S % 64 = 0 at a block-aligned L, so `tail_full` first fires on the
    #     64th step of sequence advance, and budget_or_die refuses reach > 63.
    #     The snapshot's _k_cpu/_v_cpu window is therefore reserved for a
    #     longer-horizon configuration, and G8 says so rather than printing a
    #     PASS for coverage it does not have.
    #   COVERED ONLY IF THE SITE CROSSES IT: the compress_k_cache_varlen torch.cat
    #     rebinding (cache_engine.py:658-661), which fires when
    #     no_compress_k_len reaches kernel_size=32. That is a REAL path inside a
    #     draft rollout at the later probes, so the second run below moves the
    #     site onto it instead of asserting a probe placement the sbatch never
    #     made.
    def hygiene(site, token, tag):
        lay0 = pilot.cache.layers[0]
        pilot.snap.take(); pos_h = pilot.pos.clone()
        n0 = int(getattr(lay0, "no_compress_k_len", -1))
        v0 = tuple(getattr(lay0, "compress_k_cache_varlen").shape)
        l1 = pilot.step(token, ap.ROLE_TARGET).clone()
        # read INSIDE the step's effect, before the restore undoes it
        n_after = int(getattr(lay0, "no_compress_k_len", -1))
        v_after = tuple(getattr(lay0, "compress_k_cache_varlen").shape)
        fired = (v_after != v0)
        pilot.snap.restore(); pilot.pos = pos_h.clone()
        v_back = tuple(getattr(lay0, "compress_k_cache_varlen").shape)
        n_back = int(getattr(lay0, "no_compress_k_len", -1))
        if (v_back, n_back) != (v0, n0):
            die(RC_HYGIENE, "state hygiene (%s): the restore did not put "
                            "compress_k_cache_varlen/no_compress_k_len back "
                            "(%s,%d -> %s,%d)" % (tag, v0, n0, v_back, n_back))
        pilot.rollout(0, site, token, kmax, low)
        l2 = pilot.step(token, ap.ROLE_TARGET).clone()
        pilot.snap.restore(); pilot.pos = pos_h.clone()
        if not torch.equal(l1, l2):
            die(RC_HYGIENE, "state hygiene (%s): the same target step differs before and "
                            "after a K=%d draft rollout at %s (max |delta| = %.3e) -- an "
                            "approximate byte survived the restore"
                % (tag, kmax, low["spec"], (l1 - l2).abs().max().item()))
        print("[hygiene:%s] PASS: L1 == L2 around a K=%d rollout at %s, site %d. "
              "no_compress_k_len %d (the torch.cat rebinding fires when it reaches 32); "
              "the next decode step %s it (compress_k_cache_varlen %s -> %s) and the "
              "restore put it back."
              % (tag, kmax, low["spec"], site, n0,
                 "FIRED" if fired else "did NOT fire", v0, v_after), flush=True)
        return fired

    hygiene(probe, tok, "early")

    # SECOND RUN, ON THE COMPRESSION BOUNDARY. The rebinding fires on the decode
    # call at which no_compress_k_len would reach kernel_size=32
    # (cache_engine.update_no_compress_k_decode:671-680), i.e. the (33 - n)-th
    # call from here. Advance verified steps so that call lands on DRAFT STEP 1
    # of the rollout -- the path no gate has ever validated, because with
    # WARM=8/PROBES=8 it first fires inside the LAST probe's rollout, long after
    # the gates have passed.
    n_now = int(getattr(pilot.cache.layers[0], "no_compress_k_len", -1))
    verified = 1 + WARM
    if n_now < 0:
        print("[hygiene:boundary] SKIPPED: no_compress_k_len is not exposed on this cache",
              flush=True)
    else:
        v = 33 - n_now - 1
        why = budget_reason(verified + v, kmax) if v >= 0 else "the boundary is behind us"
        avail_tokens = int(forced.shape[1]) - (WARM + 1)
        if v < 0 or why or v + 1 > avail_tokens:
            print("[hygiene:boundary] SKIPPED: the compression boundary is %d verified "
                  "steps away (no_compress_k_len=%d) and that is outside this cell (%s)"
                  % (v, n_now, why or "not enough forced tokens"), flush=True)
        else:
            for t in range(v):
                pilot.step(forced[:, WARM + 1 + t:WARM + 2 + t], ap.ROLE_WARM)
            site2 = WARM + 1 + v
            if not hygiene(site2, forced[:, site2:site2 + 1], "boundary"):
                die(RC_HYGIENE, "the boundary hygiene site was placed at %d verified "
                                "steps to make the compress_k_cache_varlen rebinding "
                                "fire and it did not: the gate would be vacuous"
                    % (verified + v))

    # ---- G4: K > 1 at 100% availability must ALSO be exact. ----------------
    # At full availability the draft is bit-identical to the target, so its
    # accumulated state IS the target's state. A failure here is numeric or a
    # state leak, never an availability effect, and the curve must not be
    # believed. This is the distinction the pilot was asked to keep: full
    # availability is not in general the same as exact suffix state, but in THIS
    # design it is, and the gate exists to find any deviation.
    for K in KS:
        qs, ps, xs, _ = pilot.rollout(0, probe, tok, K, pilot.gate_points[P_FULL])
        for j in range(K):
            a, _ = am.acceptance(ps[j], qs[j], xs[j])
            if a != 1.0:
                die(RC_KGT1, "G4: K=%d at 100%% availability accepted with %.12f at "
                             "position %d" % (K, a, j + 1))
    print("[G4] PASS: K in %s at 100%% availability accepts with probability exactly 1 "
          "at every position" % (list(KS),), flush=True)

    # ---- G5: LRU fidelity, per slot, on the device. ------------------------
    pilot.snap.take(); pos0 = pilot.pos.clone()
    pilot.policy.set_site(probe); pilot.policy.begin_rollout(P_63)
    rec0 = len(pilot.policy.records)
    pilot.step(tok, ap.ROLE_DRAFT)
    pilot.policy.end_rollout()
    pilot.snap.restore(); pilot.pos = pos0.clone()
    rs = [r for r in pilot.policy.records[rec0:] if r[13] >= 0]
    if not rs or not all(r[13] == 1 for r in rs):
        bad = sum(1 for r in rs if r[13] != 1)
        die(RC_LRU, "G5: the virtual LRU hit set differs from the engine's own load mask "
                    "on %d of %d (layer, KV head, request) rows at C=63, d=1"
            % (bad, len(rs)))
    g5 = pilot.policy.summary(pilot.policy.records[rec0:])
    print("[G5] PASS: virtual LRU == engine load mask on all %d (layer, KV head, request) "
          "rows at C=63, d=1; modelled hit %.6f, engine hit %.6f, ready %.6f "
          "(the three coincide here BECAUSE d=1 and f=0)"
          % (len(rs), g5["lru_hit_frac_modelled"], g5["lru_hit_frac_physical"],
             g5["ready_slot_frac"]), flush=True)

    # ---- G6: no NaN at the LOWEST availability point, every step. ----------
    pilot.snap.take(); pos0 = pilot.pos.clone()
    pilot.policy.set_site(probe); pilot.policy.begin_rollout(P_LOW)
    rec0 = len(pilot.policy.records)
    for _ in range(kmax):
        lg = pilot.step(tok, ap.ROLE_DRAFT)
        if not bool(torch.isfinite(lg).all()):
            die(RC_NAN, "G6: draft logits contain NaN/Inf at the lowest availability "
                        "point %s" % low["spec"])
    g6 = pilot.policy.summary(pilot.policy.records[rec0:])
    pilot.policy.end_rollout()
    pilot.snap.restore(); pilot.pos = pos0.clone()
    # THE MEASURED AVAILABILITY IS PRINTED, not assumed from the label. A cold
    # (never warmed) point reports ready = 0.0 here while wearing its capacity as
    # a name, which is exactly how the gate used to run at 0% under an 80% label.
    if not (g6["ready_slot_frac"] > 0.0):
        die(RC_NAN, "G6: the lowest availability point %s measured ready = %.4f -- every "
                    "non-tail block masked. Its virtual LRU was not warmed."
            % (low["spec"], g6["ready_slot_frac"]))
    print("[G6] PASS: no NaN and no Inf in the draft logits at %s over %d draft steps, "
          "MEASURED ready-at-attention %.4f over %d rows (a fully masked split would "
          "show here; the bias is finite for exactly that reason)"
          % (low["spec"], kmax, g6["ready_slot_frac"], g6["rows"]), flush=True)

    # ---- G7: q at draft position 1 is identical across K. ------------------
    ref = None
    for K in sorted(set(KS) | {1}):
        qs, _, _, _ = pilot.rollout(0, probe, tok, K, pilot.gate_points[P_65])
        if ref is None:
            ref = qs[0]
        elif not np.array_equal(ref, qs[0]):
            die(RC_CROSSK, "G7: q at draft position 1 differs between K=1 and K=%d at the "
                           "same site and availability (max |delta| = %.3e)"
                % (K, float(np.abs(ref - qs[0]).max())))
    print("[G7] PASS: q at draft position 1 is identical across K in %s"
          % sorted(set(KS) | {1}), flush=True)

    # ---- G8: the verified host history is untouched. -----------------------
    pilot.ss.assert_transients_intact(pilot.model, trans)
    got = pilot.host_digest_now()
    if got != pilot.host_digest:
        die(RC_HOST, "G8: the host K/V over [%d,%d) changed during the gates -- a draft "
                     "write-back escaped the snapshot, or the snapshot's host window was "
                     "mis-sized" % (max(0, L - BLOCK), L + 2 * BLOCK))
    print("[G8] PASS: host K/V digest over [%d,%d) unchanged -- the window the snapshot "
          "saves and restores, and the first rows a tail write-back could reach. AT "
          "reach <= 63 THE WRITE-BACK CANNOT FIRE (budget_or_die), so what this proves "
          "here is that the host window save/restore corrupted nothing, not that a "
          "write-back was caught. Captured-graph buffers intact."
          % (max(0, L - BLOCK), L + 2 * BLOCK), flush=True)
    print("[gates] ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
