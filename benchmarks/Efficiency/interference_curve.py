"""INTERFERENCE CURVE beside NOSI's resident decode (authorized 2026-10-01, the user via Codex; retroinfer-eval REPRODUCE.md
'INTERFERENCE-CURVE EXTENSION AUTHORIZED'). Measurement harness only: no production-baseline change; the CPU-pack / layout
experiment is not repeated. Built on feature/nosi-cpupack-c @ d9348e5 (the harness of job 2179735: Runner, gated(), the
golden gate cpupack_golden.py, the CONTROLS process, DecodeGuard, the AloneTrace capture) on fork branch
feature/nosi-interference-curve. Pure accounting and the pump: curve_core.py.

QUESTION. NOSA-8B / PG-19 / L = 16128 / A100 / C63: how does the interference between back-to-back resident decode ticks
and a SUSTAINED transfer of genuine natural LRU miss plans change with the batch, for NOSI's GPU gather (W8, primary) and
the pinned HiSparse small-grid kernel with the NOSI i256 per-head adapter, and what delivery rate does each keep?

LABELS (curve_core): 'transport/placement-ready, NOT live-LRU-ready'; 'GPU-resident trace replay' (SUSTAINED_LABEL);
saturated replay = 'resource-contention control, NOT live verifier throughput'; every copy row is a 'prebuilt-plan copy
microbenchmark, NOT an integrated LRU getter baseline'; the HiSparse arms are 'HiSparse kernel, NOSI-layout i256 adapter --
NOT native HiSparse layout' (native HiSparse results stay those of job 2175759); resident tokens/s are RESIDENT.

ARMS (the same train and the same scratch destinations for every arm; matched useful bytes):
  w<W>    NOSI's GPU gather: flash_h2d_persistent(n_ctas=W, num_warps=4, bypass_cache=False), K and V, reading the natural
          load ids straight from the GPU-resident plan store (W8 = the primary arm of jobs 2179683 / 2179735).
  hs<W>   SGLang HiSparse copy_cache_planned_kernel (sglang 87db743, vendored: nosi/flash_cache_engine/hisparse_copy), W blocks
          x 1024 threads, one launch per request (K then V), fed the i256 plan of the SAME ids (hisparse_copy/plan.py
          build_plan, item 'i256'): built ON THE DEVICE from the plan store OUTSIDE every window (the plan cost is recorded,
          isolated, NOT charged); its (source row, destination row) pairs and bytes must equal the NOSI descriptor's
          (cpupack_core.build_desc, orig layout) for every request, else the batch fails.
  cpu8    (secondary; B64 and B336 only) the 2179735 CPU-pack transport: every request's list D2H after the gate (charged),
          descriptors built LIVE on the coordinator (never the prebuilt cache), pack on 8 physical cores, one H2D per 8 MiB
          chunk, GPU scatter; its in-flight bound is the 3-slot staging ring. The finite-burst CPU8 results are REUSED from
          job 2179735 (not rerun).
  At matched W the SM footprints differ: w<W> = W CTAs x 128 threads; hs<W> = W blocks x 1024 threads.

THE SUSTAINED WINDOW (Runner.window; one row). At a gated step (gated(): the golden gate BEFORE any timing and after the
advance, cpupack_golden), every window: poison the scratch, synchronize, record the GATE on the decode stream, hand the
train to the coordinator thread (curve_core.pump: bounded in-flight queue CV_QUEUE, scratch-dependency waits, every request
charged), then K back-to-back resident decode TICKS on the decode thread (_ticks): each tick = the sanctioned restore (the
CounterSnapshot restore with its per-restore host synchronize elided: restore_nosync) + the resident decode of the gated
step + device-side receipts (loads of the tick, logits bit-exact against the step's flushed reference), recorded into
device buffers and read only after the window. The decode thread never sleeps, never synchronizes and never waits for
the transfer inside a window; the pump never touches the decode stream. Controls on the SAME work: decode-alone windows
(ticks, no train; pre and post every arm = the drift) and transfer-alone trains (train, no ticks). After every transfer the
whole scratch is fingerprinted (cpupack_golden.fp_t, K and V) against the train's reference (the shipped
flash_h2d_from_mask gather of the same train, in order, anchored by the logical reference of its last request and the
union canary). K (curve_core.choose_K) is chosen ONCE per batch at the first gated step from untimed calibration windows
and frozen; the same K and the same train serve every arm of a step; arm order is reversed on odd steps.
restore_nosync: CounterSnapshot (host_window = False) restores no pinned host rows; the only asynchronous device->host write
of a decode step is the tail write-back at a block rollover (cache_engine.py:707-708), which gated()'s rollover tripwire
excludes (cpupack_transport.py gated(), d9348e5:831-833) and which the K ticks of one step cannot reach (every tick replays the same step).
The elided synchronize is therefore a no-op for correctness here; every tick is checked (0 loads, logits bit-exact) and the
advance after the window is gated against the golden.

STAGES of one batch (run_curve; a process may run several batches with ONE loaded model, CV_BATCHES; each batch re-creates
the cache engines with VA._setup, its own Runner, golden pass and teardown): prefill -> restart snapshot (hosted) ->
CAPTURE (CP_NCAP natural steps: only what the gated steps and the trains need; a saved 2179683 export, when given, must
equal the fresh capture on that prefix) -> restart -> GOLDEN pass -> restart -> MEASURED (gated steps CV_STEPS: the curve
windows; the finite-burst W8 cross-check of cpupack_transport.main_step at CV_BURST_BATCHES / CV_BURST_STEPS).
Per batch: <tag>.json (payload), <tag>.result.json (rc, stop class, fails, partial, memory), the plan export.
CONTROLS (CV_MODE=controls, a SEPARATE process, first in the job): cpupack_transport's destructive controls plus the tick
controls (tick_clean PASS, tick_window_row DETECTED by the receipts and the gate, tick_load DETECTED by the zero-load
receipt, tick_without_restore RAISED_AND_CLEAN by the guard).
TABLE (CV_MODE=table, CPU only): curve_table.md / curve_summary.csv / curve_windows.csv / ticks_curve.csv.gz /
requests_curve.csv.gz; only rows that pass their own checks AND whose gated step passed the golden gate (row_keep).
EXIT: 0 ok; 1..19 failure count; 20 restart proof; 21 correctness / plan mismatch; 22 memory trigger (capacity: the stop
class is in the result marker); 23 crash or a leak after a batch's teardown (the sbatch resumes the remaining batches in a
new process); 24 CPU placement refused.
"""
import csv
import gc
import glob
import gzip
import json
import math
import os
import re
import sys
import threading
import time
import traceback

import cpupack_cpu as CC

MODE = os.environ.get("CV_MODE", "run")                              # run | controls | table
os.environ["CP_MODE"] = {"run": "run", "controls": "controls", "table": "table"}.get(MODE, "run")
EARLY = None
if __name__ == "__main__" and MODE == "run":
    _gn = os.environ.get("CP_GPU_NUMA_NODE", "").strip()
    EARLY = CC.early_placement(int(os.environ.get("CP_TEAM_MAX", "8")), int(_gn) if _gn.isdigit() else None)

STEPS = tuple(int(x) for x in os.environ.get("CV_STEPS", "4 5 6 7 8 9").split())
TRAIN_STEPS = int(os.environ.get("CV_TRAIN_STEPS", "4"))
os.environ.setdefault("CP_NCAP", str(max(STEPS) + TRAIN_STEPS))   # capture only what the gated steps and the trains need
os.environ.setdefault("CP_CONTIG_MB", "1")                         # the contiguous ceiling is not an arm here
os.environ.setdefault("CP_SWEEP", "0")
os.environ.setdefault("CP_CONFIRM", "1")
os.environ.setdefault("CP_CONTROL_STEPS", "4 5")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cpupack_core as K  # noqa: E402
import cpupack_golden as G  # noqa: E402
import cpupack_plans as PL  # noqa: E402
import cpupack_transport as CT  # noqa: E402
import curve_core as CV  # noqa: E402
import verify_alone as VA  # noqa: E402

CT.EARLY = EARLY
BATCHES = tuple(int(x) for x in os.environ.get("CV_BATCHES", os.environ.get("CP_B", "64")).split())
ARMS_BASE = tuple(os.environ.get("CV_ARMS", "w8 hs8").split())
TRADE_ARMS = tuple(os.environ.get("CV_TRADE_ARMS", "w4 hs4 w2 hs2").split())
TRADE_BATCHES = tuple(int(x) for x in os.environ.get("CV_TRADE_BATCHES", "").split())
CPU_BATCHES = tuple(int(x) for x in os.environ.get("CV_CPU_BATCHES", "").split())
QUEUE = int(os.environ.get("CV_QUEUE", "16"))
REGIONS = int(os.environ.get("CV_REGIONS", "1"))
REPS = int(os.environ.get("CV_REPS", "2"))
K_MARGIN = float(os.environ.get("CV_K_MARGIN", "1.5"))
K_MIN = int(os.environ.get("CV_K_MIN", "4"))
K_MAX = int(os.environ.get("CV_K_MAX", "96"))
CAL_TICKS = int(os.environ.get("CV_CAL_TICKS", "4"))
COVER_MIN = float(os.environ.get("CV_COVER_MIN", "0.95"))
SKIP_FIRST = int(os.environ.get("CV_SKIP_FIRST", "1"))
HS_THREADS = int(os.environ.get("CV_HS_THREADS", "1024"))
BURST_BATCHES = tuple(int(x) for x in os.environ.get("CV_BURST_BATCHES", "").split())
BURST_STEPS = tuple(int(x) for x in os.environ.get("CV_BURST_STEPS", "4 5").split())
CTL_TICKS = int(os.environ.get("CV_CTL_TICKS", "4"))
LEAK_LIMIT_GB = float(os.environ.get("CV_LEAK_LIMIT_GB", "2.0"))
STEP_EST_S = float(os.environ.get("CV_STEP_EST_S", "60"))
BATCH_EST_S_PER_REQ = float(os.environ.get("CV_BATCH_EST_S_PER_REQ", "3.2"))   # prefill ~2.9 s per request + margin
BATCH_EST_FIXED_S = float(os.environ.get("CV_BATCH_EST_FIXED_S", "240"))
REUSE_TABLE = os.environ.get("CV_REUSE_TABLE", "")                    # the 2179735 table (CPU8 finite burst, reused)
OUT = VA.OUT


def saved_for(B):
    return os.environ.get("CV_SAVED_%d" % B, ""), os.environ.get("CV_SAVED_SHA_%d" % B, "")


# ---------------------------------------------------------------------------------------------------------------- arms
def curve_arm(name):
    m = re.match(r"^(w|hs)(\d+)$", name)
    if m:
        W = int(m.group(2))
        if m.group(1) == "w":
            return dict(name=name, kind="w8", W=W, label="NOSI GPU gather flash_h2d_persistent n_ctas=%d num_warps=4 bypass_cache=False" % W,
                        threads=128 * W)
        return dict(name=name, kind="hs", W=W, label="%s; %d blocks x %d threads" % (CV.HS_LABEL, W, HS_THREADS), threads=W * HS_THREADS)
    if name == "cpu8":
        return dict(name=name, kind="cpu", cores=8, mode="full", lite=False, packer="row", placer="row", chunk_kb=CT.CHUNK_KB,
                    label="CPU pack on 8 physical cores + one H2D per %d KiB chunk + GPU scatter (live descriptors)" % CT.CHUNK_KB)
    raise ValueError("unknown curve arm %r" % name)


def arms_for(B, base=None, trade=None, trade_batches=None, cpu_batches=None):
    base = ARMS_BASE if base is None else base
    names = list(base)
    if B in (TRADE_BATCHES if trade_batches is None else trade_batches):
        names += [a for a in (TRADE_ARMS if trade is None else trade) if a not in names]
    if B in (CPU_BATCHES if cpu_batches is None else cpu_batches) and "cpu8" not in names:
        names.append("cpu8")
    return [curve_arm(n) for n in names]


def _no_sync():
    return None


def _bits(t):
    return t.reshape(-1).view({8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}[t.element_size()])


# ------------------------------------------------------------------------------------------------------------- runner
class CurveRunner(CT.Runner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.windows = []
        self.curve = dict(K=None, cal=None, trains={}, hs_build=[], refs={}, notes=[], partial=False, steps_done=[], deadline_notes=[])
        self.hs_plans = {}
        self.refs = {}
        self.tick_hook = None                                            # CONTROLS only (guard.destructive inside)
        self.arms = []
        self.mem_samples = []
        self.stop_class = None

    # ------------------------------------------------------------------------------------------ rows and gates
    def row_marks(self):
        return super().row_marks() + (len(self.windows),)

    def stamp_rows(self, marks, it, ok):
        super().stamp_rows(marks, it, ok)
        for r in self.windows[marks[4]:]:
            r["gate_ok"] = bool(ok)
            r["gate_step"] = it

    # ------------------------------------------------------------------------------------------ setup
    def setup_curve(self, minimal=False):
        """Backends of the two threads, receipt buffers, the HiSparse extension and the CPU arm's pinned list rows."""
        dev = self.dev
        if dev == "cuda":
            self.s_main = torch.cuda.current_stream()
            self.aux_main, self.aux_pump = torch.cuda.Stream(), torch.cuda.Stream()
            streams = dict(main=self.s_main)
            if hasattr(self, "s_plan"):
                streams["plan"] = self.s_plan
            self.be_main = K.CudaBackend(streams, self.aux_main, pool_size=4 * K_MAX + 64)
            if not minimal:
                self.be_pump = K.CudaBackend(dict(side=self.s_side), self.aux_pump, pool_size=4 * 32 * TRAIN_STEPS + 64)
        self.rc_loads = torch.zeros(K_MAX + 1, dtype=torch.int64, device=dev)
        self.rc_eq = torch.zeros(K_MAX + 1, dtype=torch.bool, device=dev)
        self.arms = [] if minimal else arms_for(self.B)
        self.hs_mod = None
        if any(a["kind"] == "hs" for a in self.arms):
            from nosi.flash_cache_engine.hisparse_copy import plan as HSP
            self.HSP = HSP
            if dev == "cuda":
                from nosi.flash_cache_engine import hisparse_copy as hs
                hs.load()
                self.hs_mod = hs
        if any(a["kind"] == "cpu" for a in self.arms):
            n_req, W = TRAIN_STEPS * self.NL, self.W

            def alloc(c):
                try:
                    return torch.full((n_req, W), -7, dtype=torch.int32).pin_memory() if dev == "cuda" else torch.full((n_req, W), -7, dtype=torch.int32)
                except RuntimeError as e:
                    raise CT.MemGate("pinned list rows allocation failed: %s" % e)
            self.plan_t = self.coord.call(alloc) if hasattr(self, "coord") else alloc(None)

    # ------------------------------------------------------------------------------------------ trains
    def groups_of(self, step, layer):
        return int((self.masks[step, layer][..., :K.TAIL_SLOT] >= 0).sum())

    def train_for(self, it):
        steps = list(range(it, it + TRAIN_STEPS))
        if max(steps) >= self.masks.shape[0]:
            raise RuntimeError("train steps %s beyond the captured %d steps" % (steps, self.masks.shape[0]))
        tr = CV.build_train(self.groups_of, steps, self.NL, REGIONS)
        info = dict(steps=steps, digest=CV.train_digest(tr), **CV.train_bytes(tr))
        self.curve["trains"][it] = info
        return tr

    def ids_of(self, r):
        return self.store[r.step, r.layer, 1 + self.HB:].view(self.H, self.B, self.M)

    def cpu_rows(self, r):
        return torch.cat([torch.zeros(1, dtype=torch.int32), (self.masks[r.step, r.layer][..., :K.TAIL_SLOT] >= 0).sum(-1).to(torch.int32).reshape(-1),
                          self.masks[r.step, r.layer].to(torch.int32).reshape(-1)])

    def hs_estimate_bytes(self, steps_list):
        """GPU bytes of the i256 plans of the largest train: B x P x 12 per request, P = 64 x max groups of one request."""
        best = 0
        for it in steps_list:
            tot = 0
            for s in range(it, min(it + TRAIN_STEPS, self.masks.shape[0])):
                per_req = (self.masks[s][..., :K.TAIL_SLOT] >= 0).sum(dim=(1, 3))      # (L, B)
                tot += int((self.B * 64 * per_req.max(dim=1).values.clamp(min=1) * 12).sum())
            best = max(best, tot)
        return best

    def prepare_hs(self, train):
        """The i256 plans of the train (module docstring, hs<W>): built on the device outside every window, validated, and
        checked against the NOSI descriptor (same source rows, same destination rows, same bytes). Plans of steps outside
        the train are freed."""
        want = {(r.step, r.layer) for r in train}
        for k in [k for k in self.hs_plans if k not in want]:
            del self.hs_plans[k]
        bad = 0
        for r in train:
            key = (r.step, r.layer)
            if key in self.hs_plans:
                continue
            ids = self.ids_of(r)
            self.sync()
            t = time.perf_counter()
            pl = self.HSP.build_plan(ids, s_cpu=self.S_cpu, s_dst=self.S_dst, n_heads=self.H, head_dim=self.D, elem_size=2,
                                     block_rows=self.R, item="i256")
            self.sync()
            build_ms = 1e3 * (time.perf_counter() - t)
            self.HSP.validate_plan(pl)
            d = K.build_desc(ids, src_layout="orig", dst_layout="orig", s_src=self.S_cpu, s_dst=self.S_dst)
            R_, P = pl.src.shape
            used = torch.arange(P, device=pl.src.device)[None, :] < pl.counts.to(torch.int64)[:R_, None]
            s_, d_ = pl.src[used], pl.dst[used].to(torch.int64)
            k_hs = torch.sort(s_ * (1 << 32) + d_).values
            k_w8 = torch.sort(d.src_idx.to(s_.device) * (1 << 32) + d.dst_idx.to(s_.device)).values
            equal = bool(pl.n_items == d.n * self.R and k_hs.numel() == k_w8.numel() and torch.equal(k_hs, k_w8))
            bytes_ok = int(pl.bytes_per_launch) == int(r.useful)
            rec = dict(step=r.step, layer=r.layer, items=int(pl.n_items), groups=int(d.n), plan_stride=int(pl.plan_stride),
                       plan_bytes=int(sum(x.numel() * x.element_size() for x in (pl.src, pl.dst, pl.counts, pl.num_real))),
                       build_device_ms=build_ms, equal_to_nosi_desc=equal, bytes_ok=bytes_ok)
            self.curve["hs_build"].append(rec)
            bad += int(not (equal and bytes_ok))
            self.hs_plans[key] = pl
        if bad:
            raise _PlanMismatch("%d HiSparse i256 plans differ from the NOSI descriptor (rows or bytes)" % bad)

    # ------------------------------------------------------------------------------------------ reference
    def train_reference(self, train):
        """The shipped gather (flash_h2d_from_mask) of the whole train, in order, into the poisoned scratch: fp of K and V,
        the union canary and the last loading request's logical rows. Cached per train digest."""
        dg = CV.train_digest(train)
        if dg in self.refs:
            return self.refs[dg]
        self.sync()
        K.poison_(self.scr_k)
        K.poison_(self.scr_v)
        if self.dev == "cuda":
            from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
            for r in train:
                ids = self.ids_of(r)
                flash_h2d_from_mask(self.scr_k, self.host_phys[r.layer][0], ids, self.R)
                flash_h2d_from_mask(self.scr_v, self.host_phys[r.layer][1], ids, self.R)
        else:
            for r in train:
                ids = self.ids_of(r)
                K.direct_gather_reference(self.host_phys[r.layer][0], ids, self.scr_k, self.R)
                K.direct_gather_reference(self.host_phys[r.layer][1], ids, self.scr_v, self.R)
        self.sync()
        fp = (G.fp_t(self.scr_k), G.fp_t(self.scr_v))
        union = torch.zeros((self.B, self.S_dst, self.H), dtype=torch.bool)
        owner = torch.full((self.B, self.S_dst, self.H), -1, dtype=torch.int32)    # the LAST request that writes each row
        last, loading = None, []
        for r in train:
            d = K.build_desc(self.cpu_rows(r)[1 + self.HB:].view(self.H, self.B, self.M), src_layout="orig", dst_layout="orig",
                             s_src=self.S_cpu, s_dst=self.S_dst)
            if d.n:
                b, t, h = K.dest_rows_logical(d)
                union[b, t, h] = True
                owner[b, t, h] = r.i
                last = (r, d)
                loading.append(r.i)
        out = dict(digest=dg, fp_k=fp[0], fp_v=fp[1], union_rows=int(union.sum()))
        # WHAT THE END-OF-WINDOW FINGERPRINT CAN SEE: a request whose every destination row is rewritten later in the train
        # leaves no trace in the final scratch, so its delivery is evidenced only by the pump (one launch + g0 / g1 per
        # request, curve_core.pump), not by the content check. Reported, never silently assumed.
        surv = set(int(x) for x in owner.unique().tolist()) - {-1}
        out["requests_loading"] = len(loading)
        out["requests_content_verified"] = len(surv)
        out["requests_fully_overwritten"] = [i for i in loading if i not in surv][:128]
        ck = K.canary(self.scr_k, union.reshape(-1).to(self.dev))
        cv = K.canary(self.scr_v, union.reshape(-1).to(self.dev))
        out["canary_ok"] = bool(ck["ok"] and cv["ok"])
        if last is not None:
            r, d = last
            b, t, h = K.dest_rows_logical(d)
            rk = K.reference_rows(self.host_phys[r.layer][0], d).to(self.dev)
            rv = K.reference_rows(self.host_phys[r.layer][1], d).to(self.dev)
            bd, td, hd = b.to(self.dev), t.to(self.dev), h.to(self.dev)
            out["last_request"] = r.i
            out["last_content_ok"] = bool(K.bits_equal(self.scr_k[bd, td, hd], rk) and K.bits_equal(self.scr_v[bd, td, hd], rv))
        else:
            out["last_request"], out["last_content_ok"] = None, True
        out["ok"] = bool(out["canary_ok"] and out["last_content_ok"])
        self.refs = {dg: out}                                            # one train's reference at a time
        self.curve["refs"][dg] = {k: v for k, v in out.items()}
        if not out["ok"]:
            raise _PlanMismatch("the train reference failed its anchors (canary %s, last request %s)" % (out["canary_ok"], out["last_content_ok"]))
        return out

    # ------------------------------------------------------------------------------------------ launches
    def launcher(self, arm):
        if arm["kind"] == "w8":
            kw = dict(n_ctas=arm["W"], num_warps=4, bypass_cache=False)
            if self.dev == "cuda":
                from nosi.flash_cache_engine.flash_h2d_persistent import flash_h2d_persistent

                def f(r, s):
                    ids = self.ids_of(r)
                    flash_h2d_persistent(self.scr_k, self.host_phys[r.layer][0], ids, self.R, **kw)
                    flash_h2d_persistent(self.scr_v, self.host_phys[r.layer][1], ids, self.R, **kw)
            else:                                                          # CPU tests: the logical per-head copy (cpupack_core)
                def f(r, s):
                    ids = self.ids_of(r)
                    K.direct_gather_reference(self.host_phys[r.layer][0], ids, self.scr_k, self.R)
                    K.direct_gather_reference(self.host_phys[r.layer][1], ids, self.scr_v, self.R)
            return f
        if arm["kind"] == "hs":
            W = arm["W"]
            if self.dev == "cuda":
                hs = self.hs_mod

                def f(r, s):
                    hs.copy_plan(self.hs_plans[(r.step, r.layer)], self.host_phys[r.layer][0], self.host_phys[r.layer][1], self.scr_k,
                                 self.scr_v, W, HS_THREADS)
            else:                                                          # CPU tests: the kernel's CPU twin (plan.py)
                def f(r, s):
                    self.HSP.simulate_planned_copy(self.hs_plans[(r.step, r.layer)], self.host_phys[r.layer][0], self.host_phys[r.layer][1],
                                                   self.scr_k, self.scr_v)
            return f
        raise ValueError(arm)

    def run_pump(self, arm, train, gate):
        """On the COORDINATOR thread: curve_core.pump over the side stream."""
        be = self.be_pump
        be.reset()
        return CV.pump(be, train, self.launcher(arm), gate, QUEUE)

    def enqueue_lists(self, train, gate):
        """cpu8, MAIN thread, after the gate: every request's list D2H on the plan stream (charged; the GPU-produced list
        must reach the host)."""
        be = self.be_main
        evs = []
        with be.stream("plan"):
            be.stream_wait("plan", gate)
            for r in train:
                be.memcpy(self.plan_t[r.i], self.store[r.step, r.layer], "plan")
                evs.append(be.event_rec("plan"))
        return evs

    def run_cpu_train(self, arm, train, lists):
        """On the COORDINATOR thread: per request, wait for its list, build the descriptor LIVE, run the pipe."""
        be = self.cbe
        be.reset()
        pipe = self.pipe(arm["chunk_kb"] * 1024 // K.useful_bytes(1))
        pipe.reset()
        out = []
        for r in train:
            t = time.perf_counter_ns()
            be.host_wait(lists[r.i])
            rec = dict(i=r.i, wait_ns=time.perf_counter_ns() - t, hk=be.marker())
            t = time.perf_counter_ns()
            d = K.build_desc(self.plan_t[r.i][1 + self.HB:].view(self.H, self.B, self.M), src_layout="orig", dst_layout="orig",
                             s_src=self.S_cpu, s_dst=self.S_dst, packer=arm["packer"], placer=arm["placer"])
            rec["desc_ns"] = time.perf_counter_ns() - t
            rec["desc_ev"] = be.marker()
            rec["n"] = d.n
            sk, dk = K.views_for(d, self.host_phys[r.layer][0], self.scr_k)
            sv, dv = K.views_for(d, self.host_phys[r.layer][1], self.scr_v)
            rec["chunks"] = pipe.layer(d, sk, sv, dk, dv, mode="full")
            out.append(rec)
        return out

    # ------------------------------------------------------------------------------------------ the decode thread
    def restore_nosync(self):
        """The sanctioned per-tick restore (module docstring): the CounterSnapshot's own restore with state_snapshot's
        per-restore host synchronize elided, then the transient-identity check and the guard's arm."""
        snap = self.snap
        if getattr(snap, "host_window", True):
            raise RuntimeError("restore_nosync needs a snapshot without the host window (CounterSnapshot)")
        ss = self.ss
        real = ss._sync_host_window
        ss._sync_host_window = _no_sync
        try:
            snap.restore()
        finally:
            ss._sync_host_window = real
        ss.assert_transients_intact(self.model, self.trans)
        self.guard.on_restore()

    def receipt(self, k, lg, ref):
        """Device-side, stream-ordered after the tick's decode and before the next restore: loads of the tick and logits
        bit-exact against the flushed reference. Nothing is read on the host here."""
        self.rc_loads[k] = torch.cat([m.reshape(-1) for m in self._masks]).ge(0).sum()
        self.rc_eq[k] = (_bits(lg) == _bits(ref)).all() if lg.shape == ref.shape and lg.dtype == ref.dtype else False

    def _ticks(self, ctx, K_):
        """THE DECODE THREAD inside a window: K_ back-to-back ticks. No sleep, no synchronize, no host read of a device
        value, no wait for the transfer (tests/test_interference_curve.py checks this body and spies on it)."""
        be, ref, step = self.be_main, ctx["lg_ref"], ctx["step_fn"]
        hook = self.tick_hook
        t0s, t1s, host, idle = [], [], [], []
        prev = None
        for k in range(K_):
            h0 = time.perf_counter()
            if hook is None or not hook(k, "skip_restore"):
                self.restore_nosync()
            if hook is not None:
                hook(k, "pre_decode")
            idle.append(None if prev is None else CV.ev_done(prev))
            a = be.event_rec("main")
            lg = step()
            b = be.event_rec("main")
            host.append(1e3 * (time.perf_counter() - h0))
            self.receipt(k, lg, ref)
            t0s.append(a)
            t1s.append(b)
            prev = b
        return t0s, t1s, host, idle

    # ------------------------------------------------------------------------------------------ one window
    def window(self, ctx, arm, train, K_, phase, rep, ticks=True, transfer=True):
        """One sustained window (module docstring). Returns its row (appended to self.windows)."""
        it = ctx["it"]
        kind = arm["kind"] if (transfer and arm) else "alone"
        rec = dict(stage="SUSTAIN", batch=self.B, step=it, plan_step=it, arm=(arm["name"] if (transfer and arm) else "alone"), kind=kind,
                   phase=phase, rep=rep, K=(K_ if ticks else 0), with_decode=bool(ticks), with_train=bool(transfer), queue=QUEUE,
                   regions=REGIONS, label=K.LABEL, replay_label=CV.SUSTAINED_LABEL, saturated_label=CV.SATURATED_LABEL,
                   copy_label=CV.PREBUILT_LABEL, arm_label=(arm or {}).get("label"), W=(arm or {}).get("W"),
                   train=(self.curve["trains"].get(it) if transfer else None))
        self.sync()
        if transfer:
            K.poison_(self.scr_k)
            K.poison_(self.scr_v)
            want = arm.get("cores") or 1
            if hasattr(self, "coord") and getattr(self.coord, "n", None) != want:
                self.coord.configure(want)
            if kind == "cpu":
                self.plan_t.fill_(-7)                                    # a list the D2H did not deliver is never a stale valid one
        if ticks:
            self.rc_loads.zero_()
            self.rc_eq.zero_()
            self._masks = [e._load_mask for e in self.engines]
        retries0 = _alloc_retries()
        self.be_main.reset()
        self.sync()
        gate = self.be_main.event_rec("main")
        job, lists, tk, err = None, None, None, None
        try:
            if transfer:
                if kind in ("w8", "hs"):
                    job = self.coord.submit(lambda co, a=arm, t=train, g=gate: self.run_pump(a, t, g))
                else:
                    lists = self.enqueue_lists(train, gate)
                    job = self.coord.submit(lambda co, a=arm, t=train, ls=lists: self.run_cpu_train(a, t, ls))
            if ticks:
                tk = self._ticks(ctx, K_)
        except (G.UnsanctionedDecode, torch.cuda.OutOfMemoryError, CT.MemGate):
            if job is not None:                                          # the guard (CONTROLS) or a capacity stop: never swallowed
                job.done.wait()
            self.sync()
            raise
        except Exception:
            err = traceback.format_exc()[-3000:]
        end = self.be_main.event_rec("main")
        if job is not None:
            job.done.wait()
        self.sync()
        if job is not None and isinstance(job.exc, (torch.cuda.OutOfMemoryError, CT.MemGate, MemoryError)):
            raise job.exc                                                # a capacity stop on the coordinator thread is never swallowed
        rec["alloc_retries"] = _alloc_retries() - retries0
        if err:
            rec["error"] = err
        if job is not None and job.error:
            rec["coord_error"] = job.error[-3000:]
        ms = lambda e: CV.ev_ms(gate, e)
        rec["end_ms"] = ms(end)
        reqs = []
        if job is not None and not job.error:
            if kind in ("w8", "hs"):
                for q in job.result:
                    reqs.append(dict(i=q["i"], step=q["step"], layer=q["layer"], groups=q["groups"], useful=q["useful"], wire=q["wire"],
                                     region=q["region"], issue=ms(q["issue"]), g0=ms(q["g0"]), g1=ms(q["g1"]), wait_ms=q["wait_ns"] / 1e6,
                                     api_ms=q["api_ns"] / 1e6, waited_for=q["waited_for"], inflight_at_issue=q["inflight_at_issue"], dep=q["dep"]))
            else:
                by = {r.i: r for r in train}
                for q in job.result:
                    r = by[q["i"]]
                    ch = [[ms(c.get("pk")), ms(c.get("h2d0")), ms(c.get("h2d1")), ms(c.get("sc0")), ms(c.get("sc1")), c["bytes"],
                           c["pack_ns"] / 1e6, c["bp_ns"] / 1e6, c["g1"] - c["g0"]] for c in q["chunks"]]
                    g1 = max((c[4] for c in ch if c[4] is not None), default=ms(q["desc_ev"]))
                    reqs.append(dict(i=r.i, step=r.step, layer=r.layer, groups=r.groups, useful=r.useful, wire=sum(c[5] for c in ch),
                                     region=r.region, issue=ms(lists[r.i]), g0=ms(q["hk"]), g1=g1, wait_ms=q["wait_ns"] / 1e6,
                                     desc_ms=q["desc_ns"] / 1e6, waited_for=None, inflight_at_issue=None, chunks=ch))
        rec["requests"] = reqs
        if tk is not None:
            t0s, t1s, host, idle = tk
            rec["ticks"] = [[ms(a), ms(b)] for a, b in zip(t0s, t1s)]
            rec["tick_host_ms"] = host
            rec["main_idle_at_submit"] = idle
            rec["tick_loads"] = [int(x) for x in self.rc_loads[:K_].tolist()]
            rec["tick_bit_exact"] = [bool(x) for x in self.rc_eq[:K_].tolist()]
        ticks_ms = [tuple(x) for x in rec.get("ticks", [])]
        rec["m"] = CV.window_metrics(ticks_ms, reqs, COVER_MIN, SKIP_FIRST) if ticks_ms else CV.transfer_metrics(reqs)
        # ---- checks
        ok = "error" not in rec and "coord_error" not in rec
        if transfer:
            ref = self.refs.get(CV.train_digest(train))
            fk, fv = G.fp_t(self.scr_k), G.fp_t(self.scr_v)
            rec["content_ok"] = bool(ref is not None and fk == ref["fp_k"] and fv == ref["fp_v"])
            rec["all_requests_done"] = len(reqs) == len(train)
            rec["bytes_charged"] = sum(q["useful"] for q in reqs) == sum(r.useful for r in train)
            ok &= rec["content_ok"] and rec["all_requests_done"] and rec["bytes_charged"]
            if kind == "cpu":
                rows = torch.stack([self.store[r.step, r.layer].cpu() for r in train]) if self.dev == "cuda" else \
                    torch.stack([self.store[r.step, r.layer] for r in train])
                rec["list_ok"] = bool(torch.equal(self.plan_t[:len(train)], rows))
                ok &= rec["list_ok"]
            if kind in ("w8", "hs"):
                mx = max((q["inflight_at_issue"] for q in reqs), default=0)
                rec["queue_ok"] = mx < QUEUE
                ok &= rec["queue_ok"]
        if ticks:
            rec["ticks_complete"] = len(rec.get("ticks", [])) == K_
            rec["receipts_ok"] = bool(rec["ticks_complete"] and all(x == 0 for x in rec["tick_loads"]) and all(rec["tick_bit_exact"]))
            ok &= rec["receipts_ok"]
        rec["ok"] = bool(ok)
        if self.dev == "cuda":
            free, total = torch.cuda.mem_get_info()
            rec["device_used_gb"] = (total - free) / 1e9
            rec["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        self.windows.append(rec)
        self.fails += int(not ok)
        return rec

    # ------------------------------------------------------------------------------------------ one gated step
    def calibrate(self, ctx, train):
        """Untimed: warm-up of every arm (JIT, the extension, first touch) + the inputs of the registered K rule."""
        dec = self.window(ctx, None, train, CAL_TICKS, "cal_decode_alone", 0, ticks=True, transfer=False)
        alone = {}
        for a in self.arms:
            w = self.window(ctx, a, train, 0, "cal_transfer_alone", 0, ticks=False, transfer=True)
            alone[a["name"]] = w["m"].get("full_ready_ms")
        tick = CV.pctl(CV.steady_ticks([tuple(x) for x in dec.get("ticks", [])], SKIP_FIRST), 50)
        self.curve["K"] = CV.choose_K(tick, {k: v for k, v in alone.items() if v == v and v is not None}, K_MARGIN, K_MIN, K_MAX, SKIP_FIRST)
        self.curve["cal"] = dict(step=ctx["it"], tick_p50_ms=tick, transfer_alone_full_ms=alone)
        self.log("K = %d (slowest %s, %.1f ms alone; decode-alone tick %.2f ms)" % (
            self.curve["K"]["K"], self.curve["K"]["slowest"], self.curve["K"].get("need_ms", 0) / K_MARGIN, tick))

    def curve_step(self, ctx):
        it = ctx["it"]
        train = self.train_for(it)
        if any(a["kind"] == "hs" for a in self.arms):
            self.prepare_hs(train)
        self.train_reference(train)
        if self.curve["K"] is None:
            self.calibrate(ctx, train)
        K_ = self.curve["K"]["K"]
        self.window(ctx, None, train, K_, "decode_alone_pre", 0, ticks=True, transfer=False)
        order = list(self.arms) if it % 2 == 0 else list(self.arms)[::-1]
        for rep in range(REPS):
            for a in order:
                self.window(ctx, a, train, K_, "transfer_alone", rep, ticks=False, transfer=True)
                self.window(ctx, a, train, K_, "overlap", rep, ticks=True, transfer=True)
        self.window(ctx, None, train, K_, "decode_alone_post", 0, ticks=True, transfer=False)
        if self.B in BURST_BATCHES and it in BURST_STEPS:                 # the finite-burst W8 cross-check (2179735's MAIN)
            self.fails += self.main_step(ctx, [CT.arm("w8", "w8")])
        self.curve["steps_done"].append(it)

    # ------------------------------------------------------------------------------------------ the batch
    def verify_saved_prefix(self, path, sha):
        """The SAVED 2179683 export against this process's fresh capture on the captured PREFIX: file sha256, per-step plan
        digests, logits, load counts and the ids (any mismatch fails the batch)."""
        r = dict(path=path, file_sha256=self.file_sha256(path), expected_sha256=sha or None)
        r["file_sha_ok"] = (not sha) or r["file_sha256"] == sha
        z = PL.load_npz(path)
        n = int(self.masks.shape[0])
        mine = PL.plan_hashes(self.masks, self.maps)["step_digest"]
        r["steps_compared"] = n
        r["step_digest_mismatch"] = [i for i, (a, b) in enumerate(zip(mine, z["hashes"]["step_digest"][:n])) if a != b][:16]
        r["logits_sha_equal"] = [str(x) for x in self.logits_sha] == [str(x) for x in z["logits_sha"][:n]]
        r["step_loaded_equal"] = [int(x) for x in self.step_loaded] == [int(x) for x in z["step_loaded"][:n]]
        r["ids_equal"] = VA.sha(self.ids.to(torch.float32)) == z["meta"].get("ids_sha")
        r["ok"] = bool(r["file_sha_ok"] and not r["step_digest_mismatch"] and r["logits_sha_equal"] and r["step_loaded_equal"] and r["ids_equal"]
                       and len(z["hashes"]["step_digest"]) >= n)
        return r

    def alloc_measure(self):
        """The burst's work rows, the scratch; the memory gate (margin 0: refuse only what cannot fit) with the i256 plans."""
        self.work = torch.zeros((self.NL, self.W), dtype=torch.int32, device="cuda")
        need = CT.mem_need_gb(self.B, self.S_dst, self.H, self.D, CT.RING, self.slot, CT.CONTIG_BYTES)
        hs = self.hs_estimate_bytes(STEPS) / 1e9 if any(a["kind"] == "hs" for a in self.arms) else 0.0
        free = torch.cuda.mem_get_info()[0] / 1e9
        self.payload_extra["memory_gate"] = dict(free_gb=free, need_gb=need, hs_plans_gb=hs, margin_gb=0.0)
        if free < need + hs:
            raise CT.MemGate("free HBM %.2f GB < need %.2f GB (scratch / landing / receipts) + %.2f GB (i256 plans)" % (free, need, hs))
        self.set_scratch("orig")
        torch.cuda.reset_peak_memory_stats()
        self.snap_inventory("after_setup")

    @torch.inference_mode()
    def run_curve(self):
        if getattr(self, "mc", None) is None:
            from nosi.verify import miss_control as mc
            self.mc = mc
        self.setup_early()
        self.setup_model()
        self.setup_curve()
        self.schedule = (list(STEPS), [])
        try:
            pps = self.ss.PostPrefillSnapshot(self.cache).take()
        except torch.cuda.OutOfMemoryError as e:
            self.stop_class = "GPU_OOM"
            raise CT.MemGate("restart snapshot: CUDA out of memory: %s" % str(e)[:200])
        self.d0 = self.start_digest()
        self.payload_extra["restart_snapshot"] = dict(hosted_bytes=G.snapshot_offload(pps), start_digest_families=G.FAMILIES_START)
        gc.collect()
        torch.cuda.empty_cache()
        cap = self.capture(snapshot=False, digest_steps=range(VA.WARM))
        ref_warm = cap["warm"]
        self.capture_rec = dict(cap, warm=[{k: v for k, v in x.items() if k != "digest"} for x in cap["warm"]])
        if not cap["accepted"]:
            self.fails += 1
            self.flush_payload(False)
            return CT.RC_CORRECT
        sv, sh = saved_for(self.B)
        if sv:
            self.plans_vs_saved = self.verify_saved_prefix(sv, sh)
            self.log("fresh capture vs SAVED plans (prefix %d steps) %s: %s" % (self.masks.shape[0], sv,
                                                                            "IDENTICAL" if self.plans_vs_saved["ok"] else json.dumps(self.plans_vs_saved)[:600]))
            if not self.plans_vs_saved["ok"]:
                self.fails += 1
                self.flush_payload(False)
                return CT.RC_CORRECT
        self.restart(pps, "after capture")
        self.flush_payload(True)
        self.golden = dict(source="in-process golden pass (option (i))", meta=self.golden_meta(), warm=None, steps={})
        self.golden_mode = "record"
        pos, warm, ok = self.warm_pass("golden", ref=ref_warm)
        if not ok:
            raise CT._ProofFail("the golden pass's warm steps differ from the capture")
        self.golden["warm"] = warm
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        self.gated_sequence(pos)
        gpath = os.path.join(OUT, "%s_golden.json" % CT.TAG)
        G.golden_export(gpath, self.golden)
        self.payload_extra["golden_export"] = gpath
        if self.golden_fail:
            self.fails += len(self.golden_fail)
            self.flush_payload(False)
            return CT.RC_CORRECT
        self.restart(pps, "after golden")
        del pps
        gc.collect()
        torch.cuda.empty_cache()
        self.golden_mode = "check"
        self.alloc_measure()
        pos, _, ok = self.warm_pass("measured", ref=ref_warm)
        if not ok:
            raise CT._ProofFail("the measured pass's warm steps differ from the reference pass")
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        return self.measured_curve(pos)

    def measured_curve(self, pos):
        if EARLY and EARLY.get("launch") is not None:
            CC.set_mask([EARLY["launch"]])
        order, _ = self.schedule
        cur = VA.WARM
        took = []
        for it in order:
            est = max(took) if took else STEP_EST_S
            if CT.STAGE_DEADLINE and time.time() + est > CT.STAGE_DEADLINE - CT.FINAL_RESERVE_S:
                self.curve["deadline_notes"].append("deadline: stopped before gated step %d (%d of %d done)" % (it, len(took), len(order)))
                self.curve["partial"] = True
                break
            while cur < it:
                self.decode(self.forced[:, cur:cur + 1], pos)
                self.sync()
                pos = pos + 1
                cur += 1
            t = time.time()
            try:
                self.gated(it, pos, self.curve_step)
            except _PlanMismatch as e:
                self.fails += 1
                self.curve["notes"].append("PLAN / REFERENCE MISMATCH at step %d: %s" % (it, e))
                self.flush_payload(False)
                return CT.RC_CORRECT
            pos = pos + 1
            cur = it + 1
            took.append(time.time() - t)
            self.mem_samples.append(dict(step=it, peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                                         peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                                         device_used_gb=(lambda f: (f[1] - f[0]) / 1e9)(torch.cuda.mem_get_info())))
            self.log("step %d done (%.0fs, fails %d)" % (it, time.time() - self.t_start, self.fails))
            self.flush_payload(True)
        self.snap_inventory("after_curve")
        self.flush_payload(False)
        return min(self.fails, 19)

    # ------------------------------------------------------------------------------------------ CONTROLS (tick controls)
    def control_specs(self):
        s0 = CT.CONTROL_STEPS[0]
        return super().control_specs() + [
            dict(name="tick_clean", kind="tick", what=None, tick=None, expect="PASS", step=s0),
            dict(name="tick_window_row", kind="tick", what="window_row", layer=7 % self.NL, tick=2, expect="DETECTED", step=s0),
            dict(name="tick_load", kind="tick", what="block_map_free", layer=3 % self.NL, tick=1, expect="DETECTED", step=s0),
            dict(name="tick_without_restore", kind="tick", what="skip_restore", tick=1, expect="RAISED_AND_CLEAN", step=s0)]

    def control_pass(self, spec, warm):
        if spec["kind"] != "tick":
            return super().control_pass(spec, warm)
        if not hasattr(self, "rc_loads"):
            self.setup_curve(minimal=True)
        name, s0 = spec["name"], spec["step"]
        self.golden_mode = "check"
        pos, _, ok = self.warm_pass(name, ref=warm)
        if not ok:
            raise CT._ProofFail("CONTROLS pass %s: the warm steps differ from the golden pass" % name)
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        extra = {}

        def hook(k, where):
            if spec["what"] is None or k != spec["tick"]:
                return False
            if spec["what"] == "skip_restore":
                return where == "skip_restore"
            if where != "pre_decode":
                return False
            with self.guard.destructive(name):
                e = self.engines[spec["layer"]]
                if spec["what"] == "window_row":                          # the whole non-tail slot 0 of one layer, every request
                    e._k_gpu.view(torch.int16)[:, 0:self.R].bitwise_xor_(0x0101)   # and head: certain to move the logits
                    extra["perturbed"] = "layer %d _k_gpu[:, 0:%d] ^= 0x0101 (non-tail slot 0, all requests / heads)" % (spec["layer"], self.R)
                elif spec["what"] == "block_map_free":
                    e._block_map[1 % self.H, 1 % self.B, 0] = -1
                    extra["perturbed"] = "layer %d _block_map[%d, %d, 0] = -1 after the tick's restore" % (spec["layer"], 1 % self.H, 1 % self.B)
            return True

        def work(ctx):
            if ctx["it"] != s0:
                return
            self.tick_hook = hook
            n0 = self.guard.decodes
            try:
                w = self.window(ctx, None, None, CTL_TICKS, "control_" + name, 0, ticks=True, transfer=False)
                extra.update(loads=w.get("tick_loads"), bit_exact=w.get("tick_bit_exact"), receipts_ok=w.get("receipts_ok"))
                self.fails -= int(not w["ok"])                           # a control window's expected detection is not a failure
                self.curve.setdefault("control_windows", []).append(self.windows.pop())
            except G.UnsanctionedDecode as e:
                extra.update(raised=True, message=str(e)[:300], decodes_before_raise=self.guard.decodes - n0)
            finally:
                self.tick_hook = None
        start = len(self.gate_log)
        self.gated_sequence(pos, work)
        log = self.gate_log[start:]
        compact = [dict(step=g["step"], ok=g["ok"], work_ran=g["work_ran"], pre_ok=g["pre"]["ok"], pre_why=g["pre"].get("why"),
                        post_ok=g["post"]["ok"], post_why=g["post"].get("why")) for g in log]
        return dict(name=name, step=s0, expect=spec["expect"], family=None, layer=spec.get("layer"),
                    verdict=self.control_verdict(spec, compact, extra), extra=extra, gate=compact)

    def control_verdict(self, spec, log, extra):
        if spec["kind"] != "tick":
            return super().control_verdict(spec, log, extra)
        by = {g["step"]: g for g in log}
        steps = list(self.schedule[0]) + list(self.schedule[1])
        clean_gates = len(log) == len(steps) and all(g["ok"] for g in log)
        t = spec.get("tick")
        loads, eq = extra.get("loads") or [], extra.get("bit_exact") or []
        if spec["what"] is None:
            return "PASS" if (clean_gates and extra.get("receipts_ok") and len(loads) == CTL_TICKS) else "FAIL"
        if spec["what"] == "skip_restore":
            good = extra.get("raised") and extra.get("decodes_before_raise") == t and clean_gates
            return "RAISED_AND_CLEAN" if good else "FAIL"
        if len(loads) != CTL_TICKS or len(eq) != CTL_TICKS:
            return "UNTESTABLE"
        before_clean = all(x == 0 for x in loads[:t]) and all(eq[:t])
        if spec["what"] == "window_row":
            g = by.get(spec["step"])
            nxt = [s for s in steps if s > spec["step"]]
            g1 = by.get(nxt[0]) if nxt else None
            gate_caught = (g is not None and not g["post_ok"]) or (g1 is not None and not g1["pre_ok"])
            return "DETECTED" if (before_clean and not eq[t] and gate_caught) else "NOT_DETECTED"
        if spec["what"] == "block_map_free":
            return "DETECTED" if (before_clean and loads[t] > 0) else "NOT_DETECTED"
        return "UNKNOWN"

    # ------------------------------------------------------------------------------------------ payload
    def payload(self, partial):
        p = super().payload(partial)
        p.update(curve=dict(self.curve, arms=self.arms, batch=self.B, queue=QUEUE, regions=REGIONS, reps=REPS, train_steps=TRAIN_STEPS,
                            steps=STEPS, k_margin=K_MARGIN, k_min=K_MIN, k_max=K_MAX, cal_ticks=CAL_TICKS, cover_min=COVER_MIN,
                            skip_first=SKIP_FIRST, hs_threads=HS_THREADS, burst_batches=BURST_BATCHES, burst_steps=BURST_STEPS,
                            labels=dict(transport=K.LABEL, replay=CV.SUSTAINED_LABEL, saturated=CV.SATURATED_LABEL, copy=CV.PREBUILT_LABEL,
                                        hisparse=CV.HS_LABEL, resident=CV.RESIDENT_LABEL),
                            mem_samples=self.mem_samples, stop_class=self.stop_class,
                            pinned_plan=(CV.pinned_plan(self.B, self.S_cpu) if hasattr(self, "S_cpu") else None)),
                 windows=self.windows)
        return p


class _PlanMismatch(Exception):
    pass


def _alloc_retries():
    try:
        return int(torch.cuda.memory_stats().get("num_alloc_retries", 0)) if torch.cuda.is_available() else 0
    except Exception:
        return 0


# ------------------------------------------------------------------------------------------------------------ table
def _wkeep(r):
    return CT.row_keep(r)


def _p(xs, q):
    return CV.pctl(xs, q)


def summarize_batch(p):
    """Per arm: the curve numbers of one batch payload, from kept rows only (row_keep); every exclusion counted."""
    B = p.get("batch")
    rows = p.get("windows") or []
    excl, kept = {}, []
    for r in rows:
        key = (r.get("arm"), r.get("phase"))
        c = excl.setdefault(key, dict(total=0, kept=0, not_ok=0, gate_fail=0))
        c["total"] += 1
        k, why = _wkeep(r)
        if k:
            c["kept"] += 1
            kept.append(r)
        else:
            c[why if why in c else "not_ok"] += 1
    alone = {}
    for r in kept:
        if r.get("phase") in ("decode_alone_pre", "decode_alone_post"):
            alone.setdefault(r["step"], []).extend(CV.steady_ticks([tuple(x) for x in r.get("ticks", [])], SKIP_FIRST))
    drift, drift_step = {}, {}
    for ph in ("decode_alone_pre", "decode_alone_post"):
        drift[ph] = _p([x for r in kept if r.get("phase") == ph for x in CV.steady_ticks([tuple(t) for t in r.get("ticks", [])], SKIP_FIRST)], 50)
        for r in kept:
            if r.get("phase") == ph:
                drift_step.setdefault(r["step"], {}).setdefault(ph, []).extend(CV.steady_ticks([tuple(t) for t in r.get("ticks", [])], SKIP_FIRST))
    arms = []
    for a in (p.get("curve") or {}).get("arms") or []:
        n = a["name"]
        ov = [r for r in kept if r.get("arm") == n and r.get("phase") == "overlap"]
        ta = [r for r in kept if r.get("arm") == n and r.get("phase") == "transfer_alone"]
        conc_cov, conc_all = {}, {}
        for r in ov:
            tk = [tuple(x) for x in r.get("ticks", [])]
            cov = CV.tick_coverage(tk, [(q["g0"], q["g1"]) for q in r.get("requests", []) if q.get("g0") is not None])
            conc_cov.setdefault(r["step"], []).extend(CV.steady_ticks(tk, SKIP_FIRST, cov, COVER_MIN))
            conc_all.setdefault(r["step"], []).extend(CV.steady_ticks(tk, SKIP_FIRST))
        pc = CV.paired_by_step(alone, conc_cov)
        pa_ = CV.paired_by_step(alone, conc_all)
        m_ov = [r["m"] for r in ov]
        m_ta = [r["m"] for r in ta]
        tick = pc["conc_p50"]
        s = dict(batch=B, arm=n, kind=a["kind"], W=a.get("W"), label=a.get("label"), n_overlap=len(ov), n_transfer_alone=len(ta),
                 steps=sorted({r["step"] for r in ov}), decode_alone_p50=pc["alone_p50"], decode_alone_p95=pc["alone_p95"],
                 tick_p50=tick, tick_p95=pc["conc_p95"], extra_ms=pc["extra_ms_mean"], extra_ms_ci=pc["extra_ms_ci"],
                 extra_pct=pc["extra_pct_mean"], extra_pct_ci=pc["extra_pct_ci"], n_covered_ticks=pc["n_conc"], n_alone_ticks=pc["n_alone"],
                 per_step=pc["per_step"], extra_pct_all_ticks=pa_["extra_pct_mean"], tick_p50_all=pa_["conc_p50"],
                 resident_tps_alone=CV.resident_tps(B, pc["alone_p50"]), resident_tps_conc=CV.resident_tps(B, tick),
                 useful_bytes=_p([m.get("useful_bytes") for m in m_ta], 50), wire_bytes=_p([m.get("wire_bytes") for m in m_ta], 50),
                 alone_gbps=_p([m.get("alone_gbps") for m in m_ta], 50), alone_full_ms=_p([m.get("full_ready_ms") for m in m_ta], 50),
                 window_gbps=_p([m.get("window_gbps") for m in m_ov], 50), window_gbps_prorata=_p([m.get("window_gbps_prorata") for m in m_ov], 50),
                 during_gbps=_p([m.get("during_gbps") for m in m_ov], 50), conc_full_ms=_p([m.get("full_ready_ms") for m in m_ov], 50),
                 overlap_frac=_p([m.get("overlap_frac") for m in m_ov], 50), overlap_frac_min=min((m.get("overlap_frac", float("nan")) for m in m_ov), default=float("nan")),
                 decode_duty=_p([m.get("decode_duty") for m in m_ov], 50), inter_tick_gap_ms=_p([m.get("inter_tick_gap_p50_ms") for m in m_ov], 50),
                 train_done_in_window=sum(1 for m in m_ov if m.get("train_done_in_window")),
                 pump_latency_p95_ms=_p([m.get("pump_latency_p95_ms") for m in m_ov], 50),
                 side_idle_ms=_p([(m.get("side_idle") or {}).get("total_ms") for m in m_ov], 50),
                 ms_per_gbps=((pc["extra_ms_mean"] / _p([m.get("during_gbps") for m in m_ov], 50)) if m_ov else float("nan")),
                 receipts_ticks=sum(len(r.get("tick_loads") or []) for r in ov),
                 receipts_zero_load=all(x == 0 for r in ov for x in r.get("tick_loads") or []),
                 receipts_bit_exact=all(x for r in ov for x in r.get("tick_bit_exact") or []),
                 host_bound_ticks=sum(1 for r in ov for x in (r.get("main_idle_at_submit") or []) if x))
        arms.append(s)
    burst = []
    for r in p.get("rows") or []:
        k, _ = CT.row_keep(r)
        if k and r.get("stage") == "MAIN":
            burst.append(r)
    bsum = None
    if burst:
        da = [r["main_ms"] for r in burst if str(r.get("phase", "")).startswith("decode_alone")]
        dc = [r["main_ms"] for r in burst if r.get("phase") == "conc" and r.get("arm") == "w8"]
        a, c = _p(da, 50), _p(dc, 50)
        bsum = dict(decode_alone_p50=a, w8_conc_p50=c, w8_slowdown_pct=(100 * (c / a - 1) if a == a and a > 0 else float("nan")), n_alone=len(da), n_conc=len(dc))
    drift_rows = [dict(step=st, pre_p50=_p(v.get("decode_alone_pre", []), 50), post_p50=_p(v.get("decode_alone_post", []), 50),
                       n_pre=len(v.get("decode_alone_pre", [])), n_post=len(v.get("decode_alone_post", []))) for st, v in sorted(drift_step.items())]
    return dict(batch=B, arms=arms, excl=excl, drift=drift, drift_steps=drift_rows, kept=len(kept), total=len(rows), burst=bsum)


def controls_lines(root):
    L, ok_all = [], True
    for fn in sorted(glob.glob(os.path.join(root, "controls", "ctl_b*.json"))):
        if fn.endswith(("_golden.json", ".result.json")):     # the golden pass export and the stop marker are not CONTROLS
            continue                                            # payloads (miniature 2179945: a golden file was judged FAIL)
        with open(fn) as f:
            p = json.load(f)
        names = [c.get("name") for c in p.get("controls") or []]
        ok, why = CT.controls_verdict(p, expected=[s for s in names] or None)
        ok_all &= ok
        L.append("## CONTROLS %s (B=%s): %s (judged by each control's own verdict; raw fails %s = the expected GATE_FAILs of "
                 "detected controls%s)" % (os.path.basename(fn), p.get("batch"), "PASS" if ok else "FAIL", p.get("fails"),
                                            "" if ok else "; " + "; ".join(why)))
        for c in p.get("controls") or []:
            L.append("- control %s: expect %s, got %s -> %s" % (c.get("name"), c.get("expect"), c.get("verdict"), "PASS" if c.get("pass") else "FAIL"))
        L.append("")
    return L, ok_all


def table_curve(out_dir, csv_requests=None):
    """Markdown + CSV of every cv_b<B>.json under out_dir (module docstring, TABLE)."""
    csv_requests = (os.environ.get("CV_TABLE_CSV", "1") == "1") if csv_requests is None else csv_requests
    t0 = time.time()
    root = os.path.dirname(os.path.abspath(out_dir))
    L = ["# Interference curve (%s; %s; saturated replay = %s)" % (K.LABEL, CV.SUSTAINED_LABEL, CV.SATURATED_LABEL), "",
         "Every copy row is a %s. HiSparse arms: %s. Resident tokens/s: %s." % (CV.PREBUILT_LABEL, CV.HS_LABEL, CV.RESIDENT_LABEL),
         "Rows enter a statistic only when ok AND their gated step passed the golden gate (row_keep). Decode ticks: the first %d tick(s) "
         "of every window excluded; beside-transfer ticks = COVERED ticks (>= %.0f%% inside the train's busy time). Extra = per trace step "
         "p50(covered) - p50(decode-alone pre + post), mean over steps, 95%% bootstrap CI over steps." % (SKIP_FIRST, 100 * COVER_MIN), ""]
    ok_all = True
    cl, cok = controls_lines(root)
    L += cl
    ok_all &= cok
    files = sorted(glob.glob(os.path.join(out_dir, "cv_b*.json")), key=lambda f: int(re.search(r"cv_b(\d+)", f).group(1)))
    files = [f for f in files if re.match(r"^cv_b\d+\.json$", os.path.basename(f))]
    summ_rows, sums = [], []
    tick_sink = gzip.open(os.path.join(out_dir, "ticks_curve.csv.gz"), "wt", compresslevel=1, newline="") if csv_requests else None
    req_sink = gzip.open(os.path.join(out_dir, "requests_curve.csv.gz"), "wt", compresslevel=1, newline="") if csv_requests else None
    tw = csv.writer(tick_sink) if tick_sink else None
    rw = csv.writer(req_sink) if req_sink else None
    if tw:
        tw.writerow(("batch", "step", "arm", "phase", "rep", "tick", "t0_ms", "t1_ms", "tick_ms", "coverage", "loads", "bit_exact", "host_ms",
                     "main_idle_at_submit", "kept"))
        rw.writerow(("batch", "step", "arm", "phase", "rep", "i", "plan_step", "layer", "groups", "useful", "wire", "issue_ms", "g0_ms", "g1_ms",
                     "queue_ms", "busy_ms", "wait_ms", "inflight_at_issue", "dep", "kept"))
    n_ticks = n_req = 0
    try:
        for fn in files:
            with open(fn) as f:
                p = json.load(f)
            B = p.get("batch")
            res = {}
            rfn = fn[:-5] + ".result.json"
            if os.path.exists(rfn):
                with open(rfn) as f:
                    res = json.load(f)
            gs = CT.gate_summary(p)
            s = summarize_batch(p)
            sums.append(s)
            cur = p.get("curve") or {}
            bad = (p.get("fails", 1) != 0 or p.get("crash") or gs["bad"] or cur.get("partial") or res.get("rc", 0) not in (0,))
            ok_all &= not bad
            L.append("## B=%s (%s): fails %s%s%s; rc %s, stop class %s; kept %d of %d window rows" % (
                B, os.path.basename(fn), p.get("fails"), " PARTIAL" if cur.get("partial") else "", " CRASH(%s)" % p["crash"].get("kind") if p.get("crash") else "",
                res.get("rc"), res.get("class"), s["kept"], s["total"]))
            L.append("- golden gate: %d gated steps; GATE_FAIL before timing at %s, after the advance at %s; restarts %d/%d bit-identical; warm %d/%d; "
                     "saved-plan prefix equal: %s" % (gs["steps"], gs["pre_fail"] or "none", gs["post_fail"] or "none", gs["restarts_ok"], gs["restarts"],
                                                      gs["warm_ok"], gs["warm"], gs["plans_vs_saved"]))
            k = cur.get("K") or {}
            L.append("- K = %s (rule: %s; slowest arm %s; decode-alone tick %.2f ms; capped %s); queue %s requests; train %s steps; reps %s" % (
                k.get("K"), k.get("rule"), k.get("slowest"), k.get("tick_p50_ms", float("nan")), k.get("capped"), cur.get("queue"),
                cur.get("train_steps"), cur.get("reps")))
            hsb = cur.get("hs_build") or []
            if hsb:
                L.append("- i256 plans: %d built, %d equal to the NOSI descriptor (rows + bytes); device build p50 %.3f ms per request (isolated, NOT "
                         "charged); plan bytes p50 %.1f MB" % (len(hsb), sum(1 for x in hsb if x["equal_to_nosi_desc"] and x["bytes_ok"]),
                                                              _p([x["build_device_ms"] for x in hsb], 50), _p([x["plan_bytes"] for x in hsb], 50) / 1e6))
            refs = [x for x in (cur.get("refs") or {}).values() if "requests_loading" in x]
            if refs:
                L.append("- content-check coverage per train (requests leaving rows in the final scratch / loading requests): %s; a fully "
                         "rewritten request's delivery is evidenced by the pump's per-request launch + events only" % ", ".join(
                             "%d/%d" % (x["requests_content_verified"], x["requests_loading"]) for x in refs))
            L.append("- decode-alone drift: pre %.3f ms -> post %.3f ms" % (s["drift"].get("decode_alone_pre", float("nan")),
                                                                             s["drift"].get("decode_alone_post", float("nan"))))
            inv = (p.get("inventory") or {}).get("after_curve") or (p.get("inventory") or {}).get("after_setup") or {}
            hbm, pin = inv.get("hbm") or {}, inv.get("pinned") or {}
            mem = cur.get("mem_samples") or []
            L.append("- memory: peak allocated %.2f GB, peak reserved %.2f GB, device used max %.2f GB (of %.2f); pinned host cache %.1f GB logical / "
                     "%.1f GB reserved (power-of-two blocks)" % (
                         max([m["peak_allocated_gb"] for m in mem] or [float("nan")]), max([m["peak_reserved_gb"] for m in mem] or [float("nan")]),
                         max([m["device_used_gb"] for m in mem] or [float("nan")]), hbm.get("total_gb", float("nan")),
                         pin.get("host_cache_bytes", 0) / 1e9, pin.get("host_cache_reserved_pow2", 0) / 1e9))
            if s["burst"]:
                b = s["burst"]
                L.append("- FINITE-BURST W8 cross-check (cpupack_transport main_step, all 32 plans ready at the gate, ONE decode): decode alone %.3f -> "
                         "W8 %.3f ms = %+.2f%% (n %d / %d); job 2179735 for comparison: B64 +22.57%%, B336 +12.78%%" % (
                             b["decode_alone_p50"], b["w8_conc_p50"], b["w8_slowdown_pct"], b["n_alone"], b["n_conc"]))
            L += ["", "| arm | steps | decode-alone p50 / p95 ms | tick p50 / p95 ms (covered n) | extra ms [CI] | extra %% [CI] | extra %% all ticks | "
                  "RESIDENT tok/s alone -> conc | useful / wire MB | GB/s alone | window GB/s (pro-rata) | GB/s during overlap | full ready ms alone / conc | "
                  "overlap frac p50 (min) | trains done in window | extra ms per GB/s | pump p95 ms | side idle ms | receipts ticks (0 loads, bit-exact) | "
                  "host-bound ticks | decode duty / inter-tick gap ms |", "|" + "---|" * 21]
            for a in s["arms"]:
                L.append("| %s | %s | %.3f / %.3f | %.3f / %.3f (%d) | %+.3f [%.3f, %.3f] | %+.2f [%.2f, %.2f] | %+.2f | %.0f -> %.0f | %.1f / %.1f | %.2f | "
                         "%.2f (%.2f) | %.2f | %.1f / %.1f | %.3f (%.3f) | %d of %d | %.3f | %.2f | %.2f | %d (%s, %s) | %d | %.3f / %.2f |" % (
                             a["arm"], a["steps"], a["decode_alone_p50"], a["decode_alone_p95"], a["tick_p50"], a["tick_p95"], a["n_covered_ticks"],
                             a["extra_ms"], a["extra_ms_ci"][0], a["extra_ms_ci"][1], a["extra_pct"], a["extra_pct_ci"][0], a["extra_pct_ci"][1],
                             a["extra_pct_all_ticks"], a["resident_tps_alone"], a["resident_tps_conc"], a["useful_bytes"] / 1e6, a["wire_bytes"] / 1e6,
                             a["alone_gbps"], a["window_gbps"], a["window_gbps_prorata"], a["during_gbps"], a["alone_full_ms"], a["conc_full_ms"],
                             a["overlap_frac"], a["overlap_frac_min"], a["train_done_in_window"], a["n_overlap"], a["ms_per_gbps"], a["pump_latency_p95_ms"],
                             a["side_idle_ms"], a["receipts_ticks"], a["receipts_zero_load"], a["receipts_bit_exact"], a["host_bound_ticks"],
                             a["decode_duty"], a["inter_tick_gap_ms"]))
                summ_rows.append({k2: v for k2, v in a.items() if k2 != "per_step"})
            L += ["", "| exclusions: arm / phase | total | kept | not ok | GATE_FAIL |", "|---|---|---|---|---|"]
            for (arm_, ph), c in sorted(s["excl"].items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
                L.append("| %s / %s | %d | %d | %d | %d |" % (arm_, ph, c["total"], c["kept"], c["not_ok"], c["gate_fail"]))
            L.append("")
            if tw:
                for r in p.get("windows") or []:
                    kp = int(_wkeep(r)[0])
                    tk = [tuple(x) for x in r.get("ticks", [])]
                    rq = r.get("requests") or []
                    cov = CV.tick_coverage(tk, [(q["g0"], q["g1"]) for q in rq if q.get("g0") is not None]) if tk else []
                    for i, (a0, a1) in enumerate(tk):
                        tw.writerow((B, r.get("step"), r.get("arm"), r.get("phase"), r.get("rep"), i, a0, a1, a1 - a0, cov[i] if cov else "",
                                     (r.get("tick_loads") or [""] * len(tk))[i], (r.get("tick_bit_exact") or [""] * len(tk))[i],
                                     (r.get("tick_host_ms") or [""] * len(tk))[i], (r.get("main_idle_at_submit") or [""] * len(tk))[i], kp))
                        n_ticks += 1
                    for q in rq:
                        rw.writerow((B, r.get("step"), r.get("arm"), r.get("phase"), r.get("rep"), q.get("i"), q.get("step"), q.get("layer"), q.get("groups"),
                                     q.get("useful"), q.get("wire"), q.get("issue"), q.get("g0"), q.get("g1"),
                                     (q["g0"] - q["issue"]) if q.get("issue") is not None and q.get("g0") is not None else "",
                                     (q["g1"] - q["g0"]) if q.get("g0") is not None and q.get("g1") is not None else "", q.get("wait_ms"),
                                     q.get("inflight_at_issue"), q.get("dep"), kp))
                        n_req += 1
    finally:
        if tick_sink:
            tick_sink.close()
            req_sink.close()
    # result markers without a payload (a batch that never produced one: capacity stop, skip, crash)
    for rfn in sorted(glob.glob(os.path.join(out_dir, "cv_b*.result.json"))):
        if not os.path.exists(rfn[:-len(".result.json")] + ".json"):
            with open(rfn) as f:
                res = json.load(f)
            ok_all &= res.get("rc") == 0
            L.append("## B=%s: NO PAYLOAD; rc %s, stop class %s (%s side): %s" % (res.get("batch"), res.get("rc"), res.get("class"),
                                                                              CV.stop_side(res.get("class") or ""), (res.get("error") or "")[:300]))
    L += _capacity_lines(out_dir)
    L += _reuse_lines()
    if summ_rows:
        keys = sorted({k for r in summ_rows for k in r})
        with open(os.path.join(out_dir, "curve_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summ_rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()})
    with open(os.path.join(out_dir, "curve_per_step.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("batch", "arm", "step", "decode_alone_p50", "tick_p50_covered", "extra_ms", "extra_pct", "n_alone_ticks", "n_covered_ticks",
                    "decode_alone_pre_p50", "decode_alone_post_p50"))
        for s in sums:
            dd = {d["step"]: d for d in s.get("drift_steps", [])}
            for a in s["arms"]:
                for x in a["per_step"]:
                    d = dd.get(x["step"], {})
                    w.writerow((s["batch"], a["arm"], x["step"], x["alone_p50"], x["conc_p50"], x["extra_ms"], x["extra_pct"], x["n_alone"], x["n_conc"],
                                d.get("pre_p50"), d.get("post_p50")))
    with open(os.path.join(out_dir, "curve_windows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("batch", "arm", "phase", "total", "kept", "not_ok", "gate_fail"))
        for s in sums:
            for (arm_, ph), c in sorted(s["excl"].items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
                w.writerow((s["batch"], arm_, ph, c["total"], c["kept"], c["not_ok"], c["gate_fail"]))
    if tw:
        L.append("- per-tick receipts: %d rows -> ticks_curve.csv.gz; per-request queue / delivery timeline: %d rows -> requests_curve.csv.gz" % (n_ticks, n_req))
    L.append("table built in %.1f s" % (time.time() - t0))
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "curve_table.md"), "w") as f:
        f.write(text)
    print(text)
    return 0 if ok_all else 1


def _capacity_lines(out_dir):
    L = ["", "### Capacity (C63, the full curve harness + staging; bounded checks only)", ""]
    res = []
    for rfn in glob.glob(os.path.join(out_dir, "cv_b*.result.json")):
        with open(rfn) as f:
            res.append(json.load(f))
    ok_b = sorted(r["batch"] for r in res if r.get("rc") == 0 and r.get("class") == "OK")
    stops = sorted((r["batch"], r.get("class")) for r in res if r.get("class") not in (None, "OK"))
    L.append("- batches that completed: %s" % (ok_b or "none"))
    for b, c in stops:
        L.append("- B=%d stopped: %s (%s side)" % (b, c, CV.stop_side(c)))
    if ok_b:
        lo = max(ok_b)
        up = min([b for b, c in stops if b > lo and CV.stop_side(c) in ("GPU", "HOST")] or [None]) if stops else None
        L.append("- largest feasible MEASURED here: B=%d. %s" % (lo, ("first measured stop above it: B=%d" % up) if up else
                                                                "no measured stop above it in this harness: the GPU maximum is NOT established"))
    pp = CV.pinned_plan(345, 24320, job_mem_bytes=480 * (1 << 30))
    L.append("- host / pinned (ANALYTIC, not run): B=345 -> %.1f GB of pinned power-of-two blocks > the 480 GiB job memory (fits %s); B=344 -> %.1f GB "
             "(largest batch below the 4 GiB block boundary at S_cpu = 24320: %d)" % (
                 pp["host_cache_reserved"] / 1e9, pp["fits_job_mem"], CV.pinned_plan(344, 24320)["host_cache_reserved"] / 1e9, CV.largest_pow2_batch(24320)))
    L.append("- reference receipts (other harnesses, not re-run): C63 B344 passed in job 2175791 (hbm sweep; B345 = analytic pinned refusal, not a "
             "measured GPU OOM); C128 B216 passed, B224 measured prefill CUDA OOM, 217-223 untested; ORDINARY OFFLOADED C128 B216 = 3160.82 tok/s at "
             "68.34 ms (eager, job 2175791) is the offloaded reference, NOT a resident control. C128 is not expanded here.")
    return L


def _reuse_lines():
    L = ["", "### CPU8 finite burst: REUSED from job 2179735 (not rerun)", ""]
    if REUSE_TABLE and os.path.exists(REUSE_TABLE):
        with open(REUSE_TABLE) as f:
            for line in f:
                if line.startswith("| MAIN | orig>orig | cpu8 |") or line.startswith("| MAIN | orig>orig | w8 |") or line.startswith("## B="):
                    L.append(line.rstrip())
        L.append("(source: %s)" % REUSE_TABLE)
    else:
        L.append("- the 2179735 table was not given (CV_REUSE_TABLE); scored values: docs/evidence/cpupack_confirm_2179735/README.md")
    return L


# ------------------------------------------------------------------------------------------------------------ main
def write_result(tag, rec):
    path = os.path.join(OUT, "%s.result.json" % tag)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, default=str)
    os.replace(path + ".tmp", path)
    return path


def teardown(runner, model, base_alloc):
    """Drop every reference of one batch (cache engines, snapshots, plans, scratch, pinned staging) and prove the release."""
    co = getattr(runner, "coord", None)
    if co is not None:
        co.q.put(None)
        co.join(timeout=60)
    for name in list(vars(runner)):
        if name not in ("model",):
            try:
                delattr(runner, name)
            except AttributeError:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch._C, "_host_emptyCache"):                         # the caching host allocator keeps freed pinned blocks
            torch._C._host_emptyCache()
    model.has_buffers = False
    if EARLY and EARLY.get("main_mask_at_init"):
        CC.set_mask(EARLY["main_mask_at_init"])
    alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return dict(allocated_gb=alloc / 1e9, base_gb=base_alloc / 1e9, leaked_gb=(alloc - base_alloc) / 1e9, meminfo=CC.meminfo())


def main():
    if MODE == "table":
        return table_curve(OUT)
    os.makedirs(OUT, exist_ok=True)
    if MODE == "run":
        why = CT.placement_problem(EARLY)
        if why:
            print("[curve] PLACEMENT REFUSED (exit %d): %s; placement %s" % (CT.RC_PLACEMENT, why, json.dumps(EARLY)), flush=True)
            return CT.RC_PLACEMENT
    VA.check_budget()
    batches = BATCHES
    todo = [B for B in batches if not os.path.exists(os.path.join(OUT, "%s.result.json" % (("ctl_b%d" if MODE == "controls" else "cv_b%d") % B)))]
    if not todo:
        print("[curve] every batch of %s already has a result marker" % (batches,), flush=True)
        return 0
    path = os.environ["NOSI_MODEL_PATH"]
    corpus = VA.load_corpus(path, max(todo))
    model = VA.load_model(path)
    base_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    total = 0
    for B in todo:
        tag = ("ctl_b%d" if MODE == "controls" else "cv_b%d") % B
        CT.TAG = tag
        if CT.STAGE_DEADLINE and MODE == "run":
            est = BATCH_EST_FIXED_S + BATCH_EST_S_PER_REQ * B
            if time.time() + est > CT.STAGE_DEADLINE:
                write_result(tag, dict(batch=B, tag=tag, rc=1, est_s=est, left_s=CT.STAGE_DEADLINE - time.time(), **{"class": "SKIPPED_DEADLINE"}))
                print("[curve] B=%d SKIPPED: estimated %.0f s > %.0f s left before the stage deadline" % (B, est, CT.STAGE_DEADLINE - time.time()), flush=True)
                total += 1
                continue
        ids, docs, distinct = VA.pick_batch(corpus, B, 0)
        print("[curve] B=%d L=%d Ncap=%d mode=%s arms=%s docs %s%s (%d distinct) placement %s" % (
            B, VA.L, VA.N, MODE, [a["name"] for a in arms_for(B)] if MODE == "run" else "controls", docs[:6], "..." if len(docs) > 6 else "",
            distinct, json.dumps(EARLY)), flush=True)
        fn = os.path.join(OUT, "%s.json" % tag)

        def flush(p, fn=fn):
            tmp = fn + ".tmp"
            with open(tmp, "w") as f:
                json.dump(p, f, default=str)
            os.replace(tmp, fn)
        runner = CurveRunner(model, ids, docs, distinct, flush)
        t_b = time.time()
        cls, err, rc = "OK", None, 0
        try:
            rc = runner.run_controls() if MODE == "controls" else runner.run_curve()
            if rc:
                cls = "FAILED_CHECKS" if rc < 20 else {CT.RC_CORRECT: "CORRECTNESS"}.get(rc, "RC%d" % rc)
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            rc = CT.exit_code_for(runner, e)
            cx = CV.classify_exception(e)                                 # a host stop that is not a registered memory trigger
            cls = runner.stop_class or (cx if (rc == CT.RC_MEMGATE or (rc == CT.RC_CRASH and cx in CV.HOST_CLASSES)) else   # DefaultCPUAllocator
                                        {CT.RC_HYGIENE: "RESTART_PROOF", CT.RC_PLACEMENT: "PLACEMENT", CT.RC_CRASH: "CRASH"}.get(rc, "CRASH"))
            err = "%s: %s" % (type(e).__name__, str(e)[:600])
            if rc == CT.RC_MEMGATE:
                runner.memgate.append(dict(trigger=err, cls=cls))
            else:
                runner.crash = dict(kind=cls, exit=rc, error=traceback.format_exc()[-4000:])
            print("[curve] B=%d stopped (exit %d, %s): %s" % (B, rc, cls, err), flush=True)
            traceback.print_exc()
            try:
                runner.flush_payload(False)
            except Exception:
                pass
        res = dict(batch=B, tag=tag, rc=rc, error=err, fails=getattr(runner, "fails", None), seconds=time.time() - t_b,
                   partial=(getattr(runner, "curve", {}) or {}).get("partial"), **{"class": cls})
        try:
            if torch.cuda.is_available():
                res.update(peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
            res["pinned"] = CV.pinned_plan(B, runner.S_cpu) if hasattr(runner, "S_cpu") else None
        except Exception:
            pass
        poisoned = rc == CT.RC_CRASH and "CUDA error" in (err or "") and "out of memory" not in (err or "").lower()
        td = teardown(runner, model, base_alloc)
        del runner
        res["teardown"] = td
        write_result(tag, res)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("[curve] B=%d done rc=%d class=%s (%.0f s); after teardown %.2f GB allocated (leak %.2f GB)" % (
            B, rc, cls, res["seconds"], td["allocated_gb"], td["leaked_gb"]), flush=True)
        total += rc if rc < 20 else 1
        if poisoned or td["leaked_gb"] > LEAK_LIMIT_GB:
            print("[curve] STOP after B=%d (%s): the sbatch resumes the remaining batches in a new process" % (
                B, "CUDA context poisoned" if poisoned else "leak %.2f GB > %.1f" % (td["leaked_gb"], LEAK_LIMIT_GB)), flush=True)
            return CT.RC_CRASH
    return min(total, 19)


if __name__ == "__main__":
    sys.exit(main())
