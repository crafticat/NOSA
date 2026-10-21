"""R2 step 7, PIECE 4: SG5, the R-PERF BUDGET of the flattened tick on piece 5's Residency (docs/R2_STEP7_TICK_SG5.md;
docs/R2_STEP7_SPEC_ROUND.md 1.2 and 7.3; C's answers Q-S1 .. Q-S5, 2026-10-21). REPORT only: the 8 % design target is a
target, not a gate; the MUST gates below guard the run's validity. Holder step: benchmarks/Efficiency/r2_tick_sg5_step.sh.

THE RUN (B documents of pg19-test-c16k, all S tokens, the same documents for every arm; --repeat-docs tiles the 94
documents when B > 94, labelled):
  SHIPPED  in a CHILD process each (its memory freed at exit; NOSI_POOL_BLOCKS is read at import): the shipped decode
           (batch prefill, decode_inference without an R2 state, NOSI_ATTN_SPLITS 4), --shipped-steps steps after 2
           warm-ups, CUDA events per step: T_shipped. With --pool-baseline also at the EQUAL MEMORY of the primary
           point: NOSI_POOL_BLOCKS = C + X + 4 - 64 (C, Q-S5).
  BUILD    ONE build_ragged (body r2), the R2 path's widths from the horizon; its GPU windows are freed (the tick uses
           its own store); its pinned host rows are the Residency's host rows; its twins are copied into every engine
           (TickEngine.from_r2(hist_slots = C + X): the store of C + X historical slots + 4 tail pages per stream).
  per GEMM mode (--modes: exact = G-b PaddedLinear(4B), fast = natural cuBLAS) and per residency backend (--backends:
  sc = ResidencySC, sb = ResidencySB, v1 = Residency; C's 2ec08da; [b2] b2 = OwnResidency, B's array-backed ownership
  table on the store's device over S-C's device path, unchecked unless --b2-checked: r2_own2.py), at the PRIMARY
  point (--C, --X, --R-ver):
    T_d    the DRAFT-ONLY tick: --draft-ticks ticks of B draft rows (each draft key decided when its cohort's positions
           are drafted), per-op CUDA events.
    TRANSPORT  --transport gilfree (D5, the default: C's adapter over agent E's packer v2) or gather (GatherTransport
           below: NOSI's Triton gather on a side stream), so one transport's failure does not hide the other terms.
    PREFLIGHT  before the long arms, a short round (--preflight-steps tokens) per backend on the run's transport
           (MUST; a failing backend is left out of the long arms) and on gather too when the run's is gilfree (REPORT).
    ROUND  the free-running greedy round (r2_tick.run_round) to --steps committed tokens per request, per-op CUDA
           events (TickTimer): T_tick per tick, r = committed tokens / request-ticks, r x T_shipped / T_tick, the
           budget dGEMM / dATT / dSEL / dTWIN / dSTORE / dRES (tick minus T_d, per category), the host ms per
           category (rows, backend, read: dHOST per backend), the residual (T_tick minus the categorized device time:
           uncategorized ops + device idle), the service's 4.3 / 4.6 / 4.8 report (4.3 layer_wait_ns: the superset
           in-flight wait of S-C, C's Q6), memory (peak retained, slots), refusals, masked fraction, decision reads,
           void ticks, the twins' step cost (the 'twin' category: both shape-stable means every step).
    BARE   with --bare, the same round with only the tick events (the instrumentation's own cost).
  SWEEP    --sweep C:X,... (R_ver = X / 2) with the first mode and the FIRST BACKEND OF --backends THAT PASSED ITS
           PREFLIGHT (recorded as sweep_backend; none passed: a REPORT "sweep skipped: no backend passed"; p3 took the
           first backend of the list and so skipped the whole sweep when S-C's preflight failed), --sweep-steps tokens:
           the same report.
MUST gates: every run completes (no exception; a CapacityRefusal is reported with its configuration), no void tick,
the twins' counters equal the host mirrors after each round, the gilfree team ran on the requested threads.
[xpool] THE SHARED X POOL LAYOUT (step S6, opt-in; nosi/nosi/r2_xpool.py, docs/R2_XPOOL_PLAN.md): --layout stream (the
default: every arm as before) | pool (the store of B (C + 4) + X_pool positions per layer; --X-pool, default X; sc and
b2 only) | both (each point runs the per-stream arm, then the pool arm, labelled <label>_stream / <label>_pool; the
preflight runs the pool layout; MUST gate <label>_XP_bitwise: the committed tokens and every tick's rows and per-row
logit fingerprints equal; REPORTS <label>_XP_peak (the round's peak GB, per arm, after a reset of the peak counter)
and <label>_XP_tick (T_tick pool / stream)). --pool-rule stream (default: NOSI_POOL_BLOCKS = C + X + 4 - 64) | k5
(r2_xpool.equal_memory_pool: C + 4 - 64 + floor(H X_pool / (B H)), H = --kv-heads) | both.
OUTPUT: --out/summary.json (after every run). The last stdout line is R2_TICK_SG5_DONE failed=<n>; exit = n (2 =
refused). ENVIRONMENT: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0 (the parent; a child sets its own).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_gate as G0  # noqa: E402  (load_workload)
import r2_g0bi_gate as GB  # noqa: E402  (load_by_path, free, gate)

BACKENDS = ("sc", "sb", "v1", "b2")                                                     # [b2] b2: OwnResidency
MODES = ("exact", "fast")
TRANSPORTS = ("gilfree", "gather")
LAYOUTS = ("stream", "pool", "both")                                                    # [xpool]
XPOOL_BACKENDS = ("sc", "b2")                                                           # [xpool] S-C's device path
_PEAK = [0]                                                                             # [xpool] the max before resets


def log(msg):
    print("[r2_tick_sg5 %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy; v2 7c826c1)")
    ap.add_argument("--gilfree", required=True, help="the directory of the pinned gilfree_pack.py / .cpp (E's f2b981a)")
    ap.add_argument("--gilfree-build", required=True, help="the extension's build directory")
    ap.add_argument("--threads", type=int, default=8, help="the gilfree packer's OpenMP team")
    ap.add_argument("--transport", default="gilfree", choices=TRANSPORTS,
                    help="gilfree (D5, the default) or gather (GatherTransport: NOSI's Triton gather on a side stream)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--repeat-docs", action="store_true")
    ap.add_argument("--steps", type=int, default=48, help="committed tokens per request in a round")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--LAMBDA", type=int, default=7)
    ap.add_argument("--m-fix", type=int, default=0, help="G-b's row count; 0 = (1 + K) B")
    ap.add_argument("--modes", default="exact,fast")
    ap.add_argument("--backends", default="sc,sb,v1")
    ap.add_argument("--C", type=int, default=96, help="C's recommendation after holder 2190876 (132 slots, 17.79 GB)")
    ap.add_argument("--X", type=int, default=32)
    ap.add_argument("--R-ver", type=int, default=16)
    ap.add_argument("--sweep", default="", help="C:X,... (R_ver = X / 2); empty: none")
    ap.add_argument("--sweep-steps", type=int, default=24)
    ap.add_argument("--draft-ticks", type=int, default=12)
    ap.add_argument("--shipped-steps", type=int, default=24)
    ap.add_argument("--pool-baseline", action="store_true", help="also the shipped decode at the primary point's memory")
    ap.add_argument("--bare", action="store_true", help="also each round with the tick events only")
    ap.add_argument("--preflight-steps", type=int, default=4,
                    help="PREFLIGHT: a short round per backend and transport before the long arms (0: none)")
    ap.add_argument("--twin", default="torch", choices=("torch", "fused"),                       # [twin-variant]
                    help="the twins' unchecked step: torch ops (default) or r2_twin_fused (one kernel per layer)")
    ap.add_argument("--variants", default="",                                                    # [batched-rebuild]
                    help="twin/rebuild,... (twin torch|fused, rebuild per_call|batched): one arm per variant at every "
                         "(mode, backend), labelled SG5_<mode>_<backend>_<twin>_<rebuild>; empty (the default): "
                         "the one variant --twin/per_call with p3's labels")
    ap.add_argument("--b2-checked", action="store_true",                                         # [b2]
                    help="b2: every B2 predicate (checked mode); default: unchecked (the timed path)")
    ap.add_argument("--layout", default="stream", choices=LAYOUTS,                               # [xpool]
                    help="the store: stream (per-stream, the default), pool (the shared X pool), both (compare)")
    ap.add_argument("--X-pool", type=int, default=-1,                                            # [xpool]
                    help="the pool's positions per layer (H lanes each); -1: X (K5's own bound)")
    ap.add_argument("--pool-rule", default="stream", choices=("stream", "k5", "both"),          # [xpool]
                    help="the equal-memory shipped baseline's NOSI_POOL_BLOCKS rule (with --pool-baseline)")
    ap.add_argument("--kv-heads", type=int, default=2, help="H for --pool-rule k5 (NOSA-8B: 2)")  # [xpool]
    ap.add_argument("--baseline-only", type=int, default=-1, help=argparse.SUPPRESS)   # the child: NOSI_POOL_BLOCKS
    ap.add_argument("--baseline-out", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.modes = [m for m in a.modes.split(",") if m]
    a.backends = [b for b in a.backends.split(",") if b]
    if not a.modes or any(m not in MODES for m in a.modes) or not a.backends or any(b not in BACKENDS for b in a.backends):
        ap.error("--modes in %s, --backends in %s" % (MODES, BACKENDS))
    a.sweep_pts = []
    for p in [p for p in a.sweep.split(",") if p]:
        c, x = p.split(":")
        a.sweep_pts.append((int(c), int(x), int(x) // 2))
    if a.m_fix == 0:
        a.m_fix = (1 + a.K) * a.batch
    if a.m_fix < (1 + a.K) * a.batch or a.C < 1 or a.X < 0 or not 0 <= a.R_ver <= a.X:
        ap.error("m_fix >= (1 + K) B, C >= 1, 0 <= R_ver <= X")
    if a.S < 64 * 64 or a.steps < 4 or a.draft_ticks < 2 or a.shipped_steps < 2:
        ap.error("S >= 4096, steps >= 4, draft-ticks >= 2, shipped-steps >= 2")
    a.variant_list = parse_variants(a.variants, a.twin)                                          # [batched-rebuild]
    if a.variant_list is None:                                                                   # [batched-rebuild]
        ap.error("--variants: twin/rebuild,... with twin in torch|fused and rebuild in per_call|batched, distinct")
    a.rebuild = a.variant_list[0][1]                                                             # [batched-rebuild]
    if a.layout != "stream" and any(b not in XPOOL_BACKENDS for b in a.backends):                # [xpool]
        ap.error("--layout %s: the shared pool runs on %s only (S-C's device path binds the lanes)"
                 % (a.layout, XPOOL_BACKENDS))
    if a.X_pool != -1 and (a.layout == "stream" or a.X_pool < a.X):                              # [xpool]
        ap.error("--X-pool needs --layout pool / both and X_pool >= X (K5's bound per head)")
    if a.kv_heads < 1:
        ap.error("--kv-heads >= 1")
    return a


def pool_positions(a, X):                                                                       # [xpool]
    """[xpool] The pool's positions per layer at a point with K5 bound X: --X-pool, or X when it is -1."""
    return X if a.X_pool < 0 else a.X_pool


def pool_rule_value(a, rule):                                                                   # [xpool]
    """[xpool] NOSI_POOL_BLOCKS of the equal-memory shipped baseline at the primary point under `rule` (before max(0,
    .)): "stream" C + X + 4 - 64; "k5" C + 4 - 64 + floor(H X_pool / (B H)) = r2_xpool.equal_memory_pool's k5 rule
    with pool_lanes = H X_pool (the CPU test XP4 pins the two together; the parent does not import nosi here)."""
    if rule == "stream":
        return a.C + a.X + 4 - 64
    return a.C + 4 - 64 + (a.kv_heads * pool_positions(a, a.X)) // (a.batch * a.kv_heads)


def parse_variants(text, twin):                                                                 # [batched-rebuild]
    """[batched-rebuild] --variants -> [(twin, rebuild)]; "" -> [(twin, "per_call")] (p3's arm, its labels unchanged);
    None for a malformed list (an unknown twin or rebuild, a duplicate, an empty item)."""
    if not text:
        return [(twin, "per_call")]
    out = []
    for item in text.split(","):
        parts = item.split("/")
        if len(parts) != 2 or parts[0] not in ("torch", "fused") or parts[1] not in ("per_call", "batched"):
            return None
        if tuple(parts) in out:
            return None
        out.append(tuple(parts))
    return out


def variant_label(a, mode, kind, variant):                                                       # [batched-rebuild]
    """[batched-rebuild] the point's label: p3's SG5_<mode>_<backend> without --variants, else with _<twin>_<rebuild>."""
    if not a.variants:
        return "SG5_%s_%s" % (mode, kind)
    return "SG5_%s_%s_%s_%s" % (mode, kind, variant[0], variant[1])


def stats(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    n = len(xs)
    return dict(n=n, mean=sum(xs) / n, median=xs[n // 2], p90=xs[min(n - 1, int(0.9 * n))], min=xs[0], max=xs[-1])


def load_docs(a):
    """B documents of S tokens; with --repeat-docs the workload's documents are tiled (labelled)."""
    import torch
    import json as _j
    man = _j.load(open(os.path.join(a.workload, "c16k_manifest.json")))
    n = int(man["shape"][0])
    if a.batch <= n:
        docs, sha = G0.load_workload(a.workload, a.batch, a.S)
        return docs, sha, False
    if not a.repeat_docs:
        raise SystemExit("REFUSED: B %d > %d documents (pass --repeat-docs to tile them, labelled)" % (a.batch, n))
    docs, sha = G0.load_workload(a.workload, n, a.S)
    reps = -(-a.batch // n)
    return torch.cat([docs] * reps, 0)[:a.batch].contiguous(), sha, True


# ------------------------------------------------------------------ the shipped decode (a child process)
def shipped_child(a) -> int:
    import torch
    import nosi.cache_engine as CE
    from nosi import NOSALlama as Llama
    from nosi.cache_engine import InfLLMv2Cache
    if CE.POOL_BLOCKS != a.baseline_only:
        print("REFUSED: NOSI_POOL_BLOCKS %d != %d" % (CE.POOL_BLOCKS, a.baseline_only), file=sys.stderr)
        return 2
    docs, _, rep = load_docs(a)
    model = Llama(model_name=a.model, device="cuda", offload=True)
    B = a.batch
    with torch.inference_mode():
        cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
        lg, pos = model.batch_prefill(docs[:B, :a.S].to("cuda"), cache)
        model.has_buffers = False                                    # the shipped warm-up captures for this batch
        tok = lg[:, -1].argmax(-1, keepdim=True)
        p = (pos[:, -1:] + 1).to(torch.int32)
        cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
        ms = []
        for s in range(2 + a.shipped_steps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            out = model.decode_inference(tok, cu, p, cache)[:, -1]
            e1.record()
            torch.cuda.synchronize()
            if s >= 2:
                ms.append(e0.elapsed_time(e1))
            tok = out.argmax(-1, keepdim=True)
            p = p + 1
    json.dump(dict(pool_blocks=a.baseline_only, ms=ms, stats=stats(ms), batch=B, S=a.S, repeated_docs=rep,
                   peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)), open(a.baseline_out, "w"), indent=1)
    return 0


def run_child(a, argv, pool, path):
    env = dict(os.environ, NOSI_POOL_BLOCKS=str(pool))
    cmd = [sys.executable, os.path.abspath(__file__)] + list(argv) + ["--baseline-only", str(pool), "--baseline-out", path]
    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0 or not os.path.exists(path):
        return None, "child exit %d" % r.returncode
    d = json.load(open(path))
    d["wall_s"] = round(time.time() - t0, 1)
    return d, None


# ------------------------------------------------------------------ the engine on the Residency
class GatherTransport:
    """SG5's "gather" transport (C's note, section 4: the GPU-gather baseline of paper 4.4), so the tick's arms can run
    while another transport is fixed: K3X's interface issue / poll / done / force, plus stream_wait (the verifier waits
    on the device). One issue = ONE launch per tensor of NOSI's zero-copy Triton gather flash_h2d_from_mask (it reads
    the pinned host rows) on a SIDE stream, with a load map (H, B, C + X) over the store's historical slots; the side
    stream first waits (device waits) on every transfer's hist_ev (T4: the tick's write-back of the host rows) and
    reader_ev (T3: the destination slot's last reader). One completion event per issue. The bias rows are not moved: the
    service fills them from the GPU cis table at take."""

    name = "gather"

    def __init__(self, host_k, host_v, store_k, store_v, n_slots: int, bs: int = 64, xpool=None):
        import torch
        from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
        self.h2d = flash_h2d_from_mask
        self.host_k, self.host_v, self.store_k, self.store_v = host_k, host_v, store_k, store_v
        self.n_slots, self.bs = int(n_slots), int(bs)
        self.xpool = xpool                          # [xpool] the flat store: one entry per transfer, its lane (phys)
        self.side = torch.cuda.Stream(device=store_k[0].device)
        self.stats = dict(releases=0, transfers=0, threads_min=None, threads_max=None)

    def _issue_flat(self, layer, transfers):                                                     # [xpool]
        """[xpool] ONE launch of r2_xpool.h2d_entries (entries b, h, block, lane) on the side stream, after the same
        waits; the bias rows are not moved (the service fills them at take)."""
        import torch
        from nosi.r2_xpool import h2d_entries
        dst = set()
        for tr in transfers:
            if tr.phys is None or (tr.h, tr.phys) in dst:
                raise ValueError("gather (xpool): transfer seq %d has no lane or shares (h %d, lane %r)"
                                 % (tr.seq, tr.h, tr.phys))
            dst.add((tr.h, tr.phys))
        ent = torch.tensor([[tr.b, tr.h, tr.block, tr.phys, 1] for tr in transfers], dtype=torch.int64).t().contiguous()
        with torch.cuda.stream(self.side):
            for tr in transfers:
                for ev in (tr.hist_ev, tr.reader_ev):
                    if ev is not None:
                        self.side.wait_event(ev)
            d = ent.pin_memory().to(self.store_k[layer].device, non_blocking=True)
            h2d_entries(self.store_k[layer], self.store_v[layer], None, self.host_k[layer], self.host_v[layer], None, d,
                        self.bs)
            ev = torch.cuda.Event()
            ev.record(self.side)
        for tr in transfers:
            tr.ev = ev
        self.stats["releases"] += 1
        self.stats["transfers"] += len(transfers)

    def issue(self, layer, transfers):
        import torch
        if not transfers:
            return
        if self.xpool is not None:                                                               # [xpool]
            return self._issue_flat(layer, transfers)
        B, H = int(self.store_k[layer].shape[0]), int(self.store_k[layer].shape[2])
        ids = torch.full((H, B, self.n_slots), -1, dtype=torch.int64)
        for tr in transfers:
            if ids[tr.h, tr.b, tr.slot] != -1:
                raise ValueError("gather: two transfers into one slot (b %d h %d slot %d)" % (tr.b, tr.h, tr.slot))
            ids[tr.h, tr.b, tr.slot] = tr.block
        with torch.cuda.stream(self.side):
            for tr in transfers:
                for ev in (tr.hist_ev, tr.reader_ev):
                    if ev is not None:
                        self.side.wait_event(ev)
            d = ids.pin_memory().to(self.store_k[layer].device, non_blocking=True)
            self.h2d(self.store_k[layer], self.host_k[layer], d, self.bs)
            self.h2d(self.store_v[layer], self.host_v[layer], d, self.bs)
            ev = torch.cuda.Event()
            ev.record(self.side)
        for tr in transfers:
            tr.ev = ev
        self.stats["releases"] += 1
        self.stats["transfers"] += len(transfers)

    def poll(self):
        pass

    def done(self, tr):
        return tr.ev is not None and tr.ev.query()

    def force(self, tr):
        if tr.ev is None:
            raise RuntimeError("gather: transfer seq %d was never issued" % tr.seq)
        tr.ev.synchronize()

    def stream_wait(self, tr):
        import torch
        torch.cuda.current_stream().wait_event(tr.ev)

    def threads_ok(self):
        return True

    def close(self):
        pass


class _LazyOps:
    """S-C's device operations bound after the service exists (CudaDeviceOps takes the service)."""

    def __init__(self):
        self.impl = None

    def wait(self, l, dec):
        return self.impl.wait(l, dec)

    def gather(self, l, dec):
        return self.impl.gather(l, dec)


def cuda_event():
    import torch
    e = torch.cuda.Event()
    e.record()
    return e


def arm_layout(a, layout=None):                                                                 # [xpool]
    """[xpool] The store layout of an arm: the caller's, else --layout ("both": the pool, e.g. in the preflight)."""
    lay = layout or getattr(a, "layout", "stream")
    return "pool" if lay == "both" else lay


def make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, timer_cats=None, transport=None, layout=None):
    """A fresh engine on the Residency layout with its service and gilfree transport (module docstring). [xpool]
    layout (default --layout): "pool" builds the shared-pool store (sc / b2 only) and hands its XPool to the
    transports and the service."""
    import torch
    from nosi.r2_residency import Residency, TorchGather
    from nosi.r2_residency_sb import ResidencySB
    from nosi.r2_residency_sc import CudaDeviceOps, ResidencySC
    from nosi.r2_tick import ResidencyBackend, TickEngine, TickTimer
    from nosi.r2_transport_gilfree import GilfreeTransport
    xspec = None                                                                                 # [xpool]
    if arm_layout(a, layout) == "pool":                                                          # [xpool]
        if kind not in XPOOL_BACKENDS:                                                           # [xpool]
            raise RuntimeError("the shared pool runs on %s only (got %s)" % (XPOOL_BACKENDS, kind))
        xspec = (C, pool_positions(a, X))                                                        # [xpool]
    if xspec is None:                                                                            # the default layout
        eng = TickEngine.from_r2(model, cache, st, ops, hist_slots=C + X, twin=a.twin)           # [twin-variant]
    else:                                                                                        # [xpool]
        eng = TickEngine.from_r2(model, cache, st, ops, hist_slots=C + X, twin=a.twin, xpool=xspec)
    eng.rebuild = getattr(a, "rebuild", "per_call")         # [batched-rebuild] the arm's rebuild (parse_variants checks)
    B, L, H = eng.B, eng.L, eng.H
    xp = eng.xpool                                                                               # [xpool] or None
    host_k = [lay.cache_engine._k_cpu for lay in cache.layers]
    host_v = [lay.cache_engine._v_cpu for lay in cache.layers]
    n_host = int(host_k[0].shape[1])
    cis = [t.total_cis[:B] for t in eng.twins]
    hw = lambda ev: ev.synchronize()                                 # noqa: E731
    if (transport or a.transport) == "gilfree":
        if ext is None:
            raise RuntimeError("the gilfree transport needs the packer extension (not built in this run)")
        tp = GilfreeTransport(ext, host_k, host_v, eng.store_k, eng.store_v, 64, eng.D, threads=a.threads, host_wait=hw,
                              background=True, xpool=xp)                                         # [xpool] xp or None
    else:
        tp = GatherTransport(host_k, host_v, eng.store_k, eng.store_v, C + X, xpool=xp)          # [xpool] xp or None
    gather = TorchGather(host_k, host_v, eng.store_k, eng.store_v)
    args = (L, B, H, C, X, R_ver, tp, gather)
    kw = dict(store_b=eng.store_bias, cis=cis, block_bytes=64 * 2 * eng.D * 2, host_wait=hw)
    if xp is not None:                                                                           # [xpool]
        kw["xpool"] = xp
    if kind == "v1":
        svc = Residency(*args, **kw)
    elif kind == "sb":
        svc = ResidencySB(*args, device="cuda", nblk=n_host // 64, **kw)
    elif kind == "b2":                                               # [b2] B2's table on the store's device; S-C's
        from nosi.r2_own2 import OwnResidency                        # [b2] device path: CudaDeviceOps (the superset
        lazy = _LazyOps()                                            # [b2] wait) behind _LazyOps, as sc
        svc = OwnResidency(*args, device=eng.store_k[0].device, nblk=n_host // 64, devops=lazy,             # [b2]
                           checked=bool(getattr(a, "b2_checked", False)), **kw)                               # [b2]
        lazy.impl = CudaDeviceOps(svc, host_k, host_v, eng.store_k, eng.store_v, eng.store_bias, cis)       # [b2]
    else:
        lazy = _LazyOps()
        svc = ResidencySC(*args, device="cuda", nblk=n_host // 64, devops=lazy, **kw)
        lazy.impl = CudaDeviceOps(svc, host_k, host_v, eng.store_k, eng.store_v, eng.store_bias, cis)
    eng.attach_backend(ResidencyBackend(eng, svc, host_k, host_v, cuda_event, a.K))
    eng.timer = TickTimer(cuda=True)
    if timer_cats is not None:                                       # BARE: the tick events only
        base_span = eng.timer.span
        import contextlib

        def span(cat):
            return base_span(cat) if cat in timer_cats else contextlib.nullcontext()
        eng.timer.span = span
    return eng, svc, tp


def run_draft_only(eng, first, n, K):
    """n ticks of B draft rows (the draft-only tick T_d); a draft key is decided once its cohort's positions are all
    drafted (as a verification would)."""
    from types import SimpleNamespace
    tick = sys.modules[type(eng).__module__]                         # the engine's own module: nosi.r2_tick on the
    Ref, cohort_of = tick.Ref, tick.cohort_of                        # GPU, the CPU tests' path-loaded copy (job 2190926)
    B = eng.B
    toks = list(first)
    for t in range(n):
        i = t + 1
        rows = [SimpleNamespace(b=b, kind="d", pos=eng.bank_pos[B + b], epoch=0, i=i, m=0) for b in range(B)]
        eng.timer.next_tick()
        with eng._span("tick"):
            out = eng.run_rows(rows, toks)
        toks = [Ref(s) for s in out.slots]
        if cohort_of(i + 1, K) != cohort_of(i, K):
            for b in range(B):
                eng.backend.svc.decided((b, 0, cohort_of(i, K)))
    eng.check_overflow()


def summarize(rep, skip=1):
    """Per category: the mean per tick (device ms and host ms), skipping the first `skip` ticks (warm-up)."""
    out = dict(ticks=rep["ticks"])
    for kind in ("device_ms", "host_ms"):
        out[kind] = {c: stats(v[skip:]) for c, v in rep[kind].items()}
    tick = rep["device_ms"].get("tick", [])
    cats = [c for c in rep["device_ms"] if c != "tick"]
    resid = [tick[t] - sum(rep["device_ms"][c][t] for c in cats) for t in range(len(tick))]
    out["residual_ms"] = stats(resid[skip:])
    return out


def budget(rnd_sum, d_sum):
    """dX = the round's mean per tick minus the draft-only tick's, per device category (docs/R2_STEP7_TICK_SG5.md 3)."""
    out = {}
    for c, v in rnd_sum["device_ms"].items():
        dv = d_sum["device_ms"].get(c)
        if v is not None and dv is not None:
            out["d_" + c] = v["mean"] - dv["mean"]
    if rnd_sum["residual_ms"] and d_sum["residual_ms"]:
        out["d_residual"] = rnd_sum["residual_ms"]["mean"] - d_sum["residual_ms"]["mean"]
    return out


SC_COUNTERS = ("mirror_uploads", "mirror_slots_uploaded", "row_uploads")   # ResidencySC's own (r2_residency_sc.py)
DEV_COUNTERS = ("events_waited", "submission_waits", "waited", "gathered")  # CudaDeviceOps; CpuDeviceSim (CPU tests)


def sc_counters(svc):
    """REPORT only (no gate; analysis/sg5_p4/RES_SC.md section 6): a SNAPSHOT of the counters the service itself
    keeps, read once after its run: mirror_uploads (the host-to-device mirror deltas), mirror_slots_uploaded (the slot
    records in them), row_uploads, journal_counts (the stream transitions per kind), and its device operations' wait
    counters (devops: events_waited / submission_waits of CudaDeviceOps behind _LazyOps; waited / gathered of the CPU
    stand-in), each with the object's class name. Cumulative over the service's life (the draft-only engine: its
    --draft-ticks ticks, tick 0 included; the round: every tick). None for a service that keeps none of them (sb,
    v1); a subclass of S-C (e.g. b2) reports whichever it has."""
    out = {k: int(getattr(svc, k)) for k in SC_COUNTERS if hasattr(svc, k)}
    if hasattr(svc, "journal_counts"):
        out["journal_counts"] = {str(k): int(v) for k, v in dict(svc.journal_counts).items()}
    dev = getattr(svc, "dev", None)
    dev = getattr(dev, "impl", dev)                                 # _LazyOps -> its CudaDeviceOps
    d = {k: int(getattr(dev, k)) for k in DEV_COUNTERS if dev is not None and hasattr(dev, k)}
    if d:
        out["devops"] = dict(d, kind=type(dev).__name__)
    if out:
        out["service"] = type(svc).__name__
    return out or None


def logit_fingerprint(logits):                                                                  # [xpool]
    """[xpool] Per row, sum_i bits(logit_i) * (2 i + 1) over the raw float32 bits as int64 (wrapping): device ops, no
    read. One differing element always changes it (odd weights: w 2^k != 0 mod 2^64 for k < 32); several could cancel
    in principle, so equal fingerprints are strong evidence of bitwise rows, not a proof."""
    import torch
    V = logits.shape[-1]
    w = 2 * torch.arange(V, device=logits.device, dtype=torch.int64) + 1
    return (logits.float().contiguous().view(torch.int32).to(torch.int64) * w).sum(-1)


def _reset_peak():                                                                              # [xpool]
    """[xpool] Remember the run's peak so far (meta.peak_gb keeps its meaning), then reset the counter for one arm."""
    import torch
    _PEAK[0] = max(_PEAK[0], torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()


def one_point(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates, label, bare=False,
              layout=None, fp=None):
    """One point (module docstring). [xpool] layout None: --layout (the default arm, unchanged); "stream" / "pool":
    that arm, with the round's peak GB after a reset (peak_gb_round) and fp (a list): per tick (rows, the rows' logit
    fingerprints on the device) and info["committed_tokens"]."""
    import torch
    from nosi.r2_tick import run_round
    info = dict(backend=kind, C=C, X=X, R_ver=R_ver, gemm=ops.mode, steps=steps)
    if kind == "b2":                                                 # [b2] the mode (report()["b2"] repeats it)
        info["b2_checked"] = bool(getattr(a, "b2_checked", False))
    if layout is not None or getattr(a, "layout", "stream") != "stream":                         # [xpool]
        info["layout"] = arm_layout(a, layout)
        if info["layout"] == "pool":
            info["X_pool"] = pool_positions(a, X)
    # T_d: the draft-only tick
    eng, svc, tp = make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, layout=layout)
    try:
        run_draft_only(eng, first, a.draft_ticks, a.K)
        info["draft_only"] = summarize(eng.timer.report())
        info["draft_only"]["d2h_reads"] = getattr(svc, "d2h_reads", None)
        info["draft_only"]["sc"] = sc_counters(svc)                    # REPORT (RES_SC.md 6)
    finally:
        tp.close()
        del eng, svc, tp
        GB.free()
    # the round
    if layout is not None:                                                                       # [xpool] this arm's peak
        _reset_peak()
    eng, svc, tp = make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, layout=layout)
    req_ticks = [0]

    def on_tick(rows, tokens, out):
        req_ticks[0] += len({r.b for r in rows})
        if fp is not None:                                                                       # [xpool] no read here
            fp.append(([(r.b, r.kind, r.pos) for r in rows], logit_fingerprint(out.logits)))
    try:
        t0 = time.time()
        rnd = run_round(eng, first, [steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA, on_tick=on_tick)
        torch.cuda.synchronize()
        info["wall_s"] = round(time.time() - t0, 1)
        if layout is not None:                                                                   # [xpool]
            info["peak_gb_round"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            info["store_gb"] = round(sum(t.numel() * t.element_size() for t in eng.store_k + eng.store_v
                                         + eng.store_bias) / 1e9, 3)
            if eng.xpool is not None:
                info["xpool"] = eng.xpool.report()
        if fp is not None:                                                                       # [xpool]
            info["committed_tokens"] = [list(r.out) for r in rnd.reqs]
        rs = summarize(eng.timer.report())
        info["round"] = rs
        committed = sum(len(r.out) - 1 for r in rnd.reqs)
        r = committed / max(1, req_ticks[0])
        T_tick = rs["device_ms"]["tick"]["mean"] if rs["device_ms"].get("tick") else None
        info.update(ticks=rnd.ticks, committed=committed, request_ticks=req_ticks[0], r=r, T_tick_ms=T_tick,
                    ratio=(r * T_shipped / T_tick) if (T_shipped and T_tick) else None,
                    budget=budget(rs, info["draft_only"]), stats=dict(eng.stats), round_stats=dict(rnd.stats),
                    decision_reads_per_verify_tick=eng.stats["decision_reads"] / max(1, rnd.syncs),
                    residency=dict(k43=dict(svc.metrics.k43), k46=dict(svc.metrics.k46),
                                   peak_retained_slots=svc.metrics.peak_retained_slots, memory=svc.memory(),
                                   refused=svc.metrics.refused, masked=svc.metrics.masked,
                                   draft_pairs=svc.metrics.draft_pairs,
                                   masked_fraction=svc.metrics.masked / max(1, svc.metrics.draft_pairs),
                                   counts=dict(svc.metrics.counts), d2h_reads=getattr(svc, "d2h_reads", None),
                                   sc=sc_counters(svc)),                 # REPORT (RES_SC.md 6)
                    transport=dict(tp.stats), threads_ok=tp.threads_ok())
        try:
            info["residency"]["report"] = svc.report()
        except Exception as e:                                        # the report is a REPORT
            info["residency"]["report_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        mirror = eng.mirror_failures()
        gates.append(GB.gate("%s_completed" % label, True, True, "ticks %d, T_tick %s ms, r %.3f, ratio %s"
                             % (rnd.ticks, None if T_tick is None else round(T_tick, 2), r,
                                None if info["ratio"] is None else round(info["ratio"], 3))))
        gates.append(GB.gate("%s_bookkeeping" % label, not mirror and not eng.void and eng.stats["void_ticks"] == 0, True,
                             "twin counters vs mirrors: %d differ; void ticks %d" % (len(mirror), eng.stats["void_ticks"])))
        gates.append(GB.gate("%s_gilfree_team" % label, tp.threads_ok() or tp.stats["releases"] == 0, True,
                             "requested %d threads; teams %s .. %s over %d releases" % (
                                 a.threads, tp.stats["threads_min"], tp.stats["threads_max"], tp.stats["releases"])))
    finally:
        tp.close()
        del eng, svc, tp
        GB.free()
    if bare:
        eng, svc, tp = make_engine(a, model, cache, st, ops, kind, C, X, R_ver, ext, timer_cats=("tick",),
                                   layout=layout)
        try:
            run_round(eng, first, [steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA)
            bs = summarize(eng.timer.report())
            info["bare_T_tick"] = bs["device_ms"].get("tick")
        finally:
            tp.close()
            del eng, svc, tp
            GB.free()
    return info


XP_TICK_BAND = 0.15                                                                             # [xpool] registered


def compare_layouts(recs):                                                                     # [xpool]
    """[xpool] recs[layout] = (per-tick rows, per-tick fingerprint lists, committed tokens). Returns (equal, compared
    rows, the first difference or None)."""
    s, p = recs["stream"], recs["pool"]
    if s[2] != p[2]:
        return False, 0, "committed tokens differ"
    if len(s[0]) != len(p[0]):
        return False, 0, "tick counts differ (%d / %d)" % (len(s[0]), len(p[0]))
    n = 0
    for t, (rs, rp, fs, fq) in enumerate(zip(s[0], p[0], s[1], p[1])):
        if rs != rp:
            return False, n, "tick %d: rows differ" % t
        for k, (x, y) in enumerate(zip(fs, fq)):
            if x != y:
                return False, n, "tick %d row %d %s: fingerprint %d / %d" % (t, k, rs[k], x, y)
            n += 1
    return True, n, None


def one_point_both(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates, label,
                   bare=False, point_fn=None):
    """[xpool] --layout both: the per-stream arm, then the pool arm (one_point with layout and fingerprints), then
    the MUST gate <label>_XP_bitwise and the REPORTS <label>_XP_peak / <label>_XP_tick (docs/R2_XPOOL_PLAN.md
    section 6). point_fn: the CPU test's stand-in of one_point."""
    import torch
    point_fn = point_fn or one_point
    out, recs = {}, {}
    for lay in ("stream", "pool"):
        fp = []
        info = point_fn(a, model, cache, st, first, ops, kind, C, X, R_ver, ext, steps, T_shipped, gates,
                        "%s_%s" % (label, lay), bare=bare, layout=lay, fp=fp)
        vals = torch.cat([v for (_, v) in fp]).tolist() if fp else []             # ONE read per arm, after it
        per_tick, at = [], 0
        for (rows, v) in fp:
            per_tick.append(vals[at:at + len(rows)])
            at += len(rows)
        recs[lay] = ([r for (r, _) in fp], per_tick, info.pop("committed_tokens", None))
        out[lay] = info
    same, n, diff = compare_layouts(recs)
    gates.append(GB.gate(label + "_XP_bitwise", same and n > 0, True,
                         "%s: %d rows of %d ticks compared, committed tokens %s" % (
                             "EQUAL" if same and n > 0 else "DIFFER (%s)" % (diff or "no row compared"), n,
                             len(recs["stream"][0]), "equal" if recs["stream"][2] == recs["pool"][2] else "differ")))
    ps, pp = out["stream"].get("peak_gb_round"), out["pool"].get("peak_gb_round")
    ss, sp = out["stream"].get("store_gb"), out["pool"].get("store_gb")
    drop = None if ps is None or pp is None else round(ps - pp, 3)
    gates.append(GB.gate("R_" + label + "_XP_peak", True, False, "round peak GB stream %s pool %s (drop %s); store GB "
                         "stream %s pool %s (drop %s)" % (ps, pp, drop, ss, sp,
                                                          None if ss is None or sp is None else round(ss - sp, 3))))
    ts, tq = out["stream"].get("T_tick_ms"), out["pool"].get("T_tick_ms")
    ratio = None if not ts or tq is None else tq / ts
    gates.append(GB.gate("R_" + label + "_XP_tick", True, False, "T_tick stream %s ms pool %s ms: ratio %s (registered "
                         "band 1 +- %.2f: %s)" % (ts, tq, None if ratio is None else round(ratio, 4), XP_TICK_BAND,
                                                  "inside" if ratio is not None and abs(ratio - 1) <= XP_TICK_BAND
                                                  else "OUTSIDE")))
    return dict(backend=kind, layouts=out, xp=dict(bitwise=same, rows_compared=n, first_difference=diff,
                                                    peak_drop_gb=drop, tick_ratio=ratio))


def preflight(a, model, cache, st, first, ops, ext, gates, res, engine_fn=None, round_fn=None):
    """PREFLIGHT (lead, 2026-10-22: an integration fault must cost minutes, not the whole step): before the long arms,
    a SHORT free-running round (--preflight-steps committed tokens per request: draft rows, the first verify ticks,
    their decisions) per backend on the run's transport (MUST; a backend that fails is not run in the long arms), and
    on the gather transport too when the run's is gilfree (REPORT: does the fault follow the transport?). Returns the
    backends that passed on the run's transport (all of them with --preflight-steps 0). engine_fn / round_fn: the
    CPU test's stand-ins of make_engine / r2_tick.run_round (default: the real ones)."""
    import torch
    if round_fn is None:
        from nosi.r2_tick import run_round as round_fn
    engine_fn = engine_fn or make_engine
    if a.preflight_steps <= 0:
        return list(a.backends)
    tps = [a.transport] + (["gather"] if a.transport == "gilfree" else [])
    ok, out = [], []
    for tpname in tps:
        for kind in a.backends:
            label = "SG5_preflight_%s_%s" % (tpname, kind)
            log(label)
            must = tpname == a.transport
            t0 = time.time()
            eng = svc = tp = None
            try:
                eng, svc, tp = engine_fn(a, model, cache, st, ops, kind, a.C, a.X, a.R_ver, ext, timer_cats=("tick",),
                                         transport=tpname)
                rnd = round_fn(eng, first, [a.preflight_steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                mirror = eng.mirror_failures()
                good = not mirror and rnd.syncs > 0
                detail = "%d ticks, %d decision ticks, twin counters %s, %.0f s" % (
                    rnd.ticks, rnd.syncs, "OK" if not mirror else "DIFFER", time.time() - t0)
            except Exception as e:
                traceback.print_exc()
                good, detail = False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])
            finally:
                if tp is not None:
                    tp.close()
                del eng, svc, tp
                GB.free()
            gates.append(GB.gate(label if must else "R_" + label, good or not must, must,
                                 detail if good else "FAILED: " + detail))
            out.append(dict(transport=tpname, backend=kind, passed=good, detail=detail))
            if must and good:
                ok.append(kind)
    res["preflight"] = out
    return ok


def sweep_backend(order, passed):
    """The sweep's backend: the FIRST backend of the step's --backends order that PASSED its preflight; None when none
    did. (p3 used order[0] whatever its preflight: its S-C preflight failed, so the whole sweep was skipped.)"""
    for kind in order:
        if kind in passed:
            return kind
    return None


def run_sweep(a, make_ops, passed, gates, res, save, point):
    """The sweep (--sweep C:X,...; R_ver = X / 2) on sweep_backend(a.backends, passed), recorded as
    res["sweep_backend"]; with no backend passed, a REPORT "sweep skipped: no backend passed". point(ops, kind, C, X,
    R_ver, label) runs one sweep point (one_point); its failures are REPORTS (a configuration result, e.g. a
    CapacityRefusal), as before."""
    res["sweep"] = []
    if not a.sweep_pts:
        res["sweep_backend"] = None
        return
    kind = sweep_backend(a.backends, passed)
    res["sweep_backend"] = kind
    if kind is None:
        gates.append(GB.gate("R_SG5sweep_skipped", True, False, "sweep skipped: no backend passed (preflight order %s)"
                             % (list(a.backends),)))
        save()
        return
    log("SWEEP on %s: the first backend of %s that passed its preflight" % (kind, list(a.backends)))
    ops = make_ops()
    for (C, X, R_ver) in a.sweep_pts:
        label = "SG5sweep_%s_C%d_X%d" % (kind, C, X)
        log(label)
        try:
            res["sweep"].append(point(ops, kind, C, X, R_ver, label))
        except Exception as e:
            traceback.print_exc()
            gates.append(GB.gate("R_" + label, True, False, "EXCEPTION %s: %s (a configuration result, "
                                 "e.g. a CapacityRefusal)" % (type(e).__name__, str(e)[:400])))
        GB.free()
        save()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    if a.baseline_only >= 0:
        return shipped_child(a)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    try:
        os.makedirs(a.out)
        gates, res = [], {}
        meta = dict(args={k: v for k, v in vars(a).items() if k not in ("sweep_pts",)}, sweep=a.sweep_pts,
                    nosi_commit=os.environ.get("TICK_COMMIT", "unknown"),
                    draft_note="the drafts read the Residency's READY blocks (paper 3.1): this is the representative "
                               "draft, unlike SG1-SG4's hashed stand-in")

        def save():
            json.dump(dict(meta=meta, gates=gates, results=res), open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1, sort_keys=True, default=str)
        log("SHIPPED baselines in child processes")
        base, err = run_child(a, argv, 0, os.path.join(a.out, "shipped_pool0.json"))
        res["shipped"] = base
        gates.append(GB.gate("shipped_baseline", base is not None, True, err or "T_shipped median %.2f ms"
                             % base["stats"]["median"]))
        if a.pool_baseline:
            pool = a.C + a.X + 4 - 64
            if a.pool_rule in ("stream", "both"):                                              # [xpool] the default
                eq, err = run_child(a, argv, max(0, pool), os.path.join(a.out, "shipped_pool_eq.json"))
                res["shipped_equal_memory"] = eq
                gates.append(GB.gate("R_shipped_equal_memory", True, False, err or "NOSI_POOL_BLOCKS %d: median %.2f ms"
                                     % (pool, eq["stats"]["median"])))
            if a.pool_rule in ("k5", "both"):                                                  # [xpool] the K5 rule
                pk = pool_rule_value(a, "k5")
                eq, err = run_child(a, argv, max(0, pk), os.path.join(a.out, "shipped_pool_eq_k5.json"))
                res["shipped_equal_memory_k5"] = eq
                gates.append(GB.gate("R_shipped_equal_memory_k5", True, False, err or "NOSI_POOL_BLOCKS %d (k5 rule): "
                                     "median %.2f ms" % (pk, eq["stats"]["median"])))
        save()
        T_shipped = base["stats"]["median"] if base else None
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        from nosi import NOSALlama as Llama
        from nosi.r2_bi_decode import BIOps
        from nosi.r2_decode_path import build_ragged
        if not NL.R2_DECODE or NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0 and NOSI_POOL_BLOCKS 0 are needed")
        docs, sha, rep = load_docs(a)
        meta.update(workload_sha256=sha, repeated_docs=rep, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        attn = GB.load_by_path(a.biattn, "piece3_bi_paged_attention")
        ext = None
        if a.transport == "gilfree":
            gp = GB.load_by_path(os.path.join(a.gilfree, "gilfree_pack.py"), "gilfree_pack_pinned")
            ext = gp.load(a.gilfree_build)
            meta["gilfree_openmp"] = int(getattr(ext, "OPENMP", 0))
        meta["transport"] = a.transport
        meta["layout"] = a.layout                                                              # [xpool]
        meta["twin"] = a.twin                                                                  # [twin-variant]
        if a.variants:                                                                         # [batched-rebuild]
            meta["variants"] = ["%s/%s" % v for v in a.variant_list]                            # [batched-rebuild]
        a.twin, a.rebuild = a.variant_list[0]           # [batched-rebuild] the preflight / sweep arms: the first one
        t0 = time.time()
        model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s" % (time.time() - t0))
        with torch.inference_mode():
            prompts = [docs[b, :a.S] for b in range(a.batch)]
            horizon = max(a.steps, a.sweep_steps) + a.LAMBDA + 64
            cache, st, first_t = build_ragged(model, prompts, horizon, f4_fix=True)
            for lay in cache.layers:                                   # the tick's own store replaces the windows
                e = lay.cache_engine
                e._k_gpu = e._v_gpu = e._kv_bias_gpu = None
            GB.free()
            first = [int(x) for x in first_t.view(-1).tolist()]
            meta["build_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            backends = preflight(a, model, cache, st, first, BIOps(attn, a.m_fix, gemm=a.modes[0]), ext, gates, res)
            save()
            if not backends:
                log("PREFLIGHT: no backend ran a short round on the %s transport: the long arms are skipped"
                    % a.transport)
            res["points"] = []
            for mode in a.modes:
                ops = BIOps(attn, a.m_fix, gemm=mode)
                for kind in backends:
                    for variant in a.variant_list:                                             # [batched-rebuild]
                        a.twin, a.rebuild = variant                                            # [batched-rebuild]
                        label = variant_label(a, mode, kind, variant)                          # [batched-rebuild]
                        log("%s at C %d X %d R_ver %d" % (label, a.C, a.X, a.R_ver))
                        try:
                            point = one_point_both if a.layout == "both" else one_point                # [xpool]
                            pt = point(a, model, cache, st, first, ops, kind, a.C, a.X, a.R_ver, ext, a.steps,
                                       T_shipped, gates, label, bare=a.bare)
                            pt["twin"], pt["rebuild"] = variant                                # [batched-rebuild]
                            res["points"].append(pt)
                        except Exception as e:
                            traceback.print_exc()
                            gates.append(GB.gate(label + "_completed", False, True, "EXCEPTION %s: %s"
                                                 % (type(e).__name__, str(e)[:400])))
                        GB.free()
                        save()
            a.twin, a.rebuild = a.variant_list[0]                                              # [batched-rebuild]
            run_sweep(a, lambda: BIOps(attn, a.m_fix, gemm=a.modes[0]), backends, gates, res, save,
                      lambda ops, kind, C, X, R_ver, label: one_point(a, model, cache, st, first, ops, kind, C, X, R_ver,
                                                                      ext, a.sweep_steps, T_shipped, gates, label))
        meta["peak_gb"] = round(max(_PEAK[0], torch.cuda.max_memory_allocated()) / 1e9, 2)     # [xpool] resets
        save()
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                    g["detail"][:400]), flush=True)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        print("R2_TICK_SG5_DONE failed=%d" % nfail, flush=True)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
