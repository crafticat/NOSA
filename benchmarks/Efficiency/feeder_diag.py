"""FEEDER DIAGNOSTIC driver (benchmark-only; authorized 2026-10-04 by the user via Codex: one focused feeder diagnostic, a
narrow follow-up to the d9348e5 scheduling attribution; registration retroinfer-eval docs/evidence/feeder_diag/REGISTRATION.md).
No production feeder, no full verifier, no live-cache installation, no native / GIL rewrite. The data plane is cpupack_core's
(descriptors, chunk_requests, stage_views, native index_select(out=...), one H2D per chunk, index_copy_ scatter); the split
and its ownership rules are feeder_core's; the metrics, targets and budget feeder_metrics'.

LABEL: finite-burst transport / placement-ready in SCRATCH only; NOT sustained live drafting or committed throughput.

CELL (fixed): head-major CPU source (the host cache reordered IN PLACE, bit-exact per request, cpupack_core.convert_inplace),
original-layout GPU scratch (the window's geometry), group packer / row placer, 8 MiB useful chunk capacity (256 groups:
8,519,680 wire bytes per full chunk), three pinned staging slots and three device landing slots of the historical 16 MiB
alternate size (512 groups: 17,039,360 bytes each), the frozen saved 2179683 plans of plan steps FD_PLAN_STEPS.
ARMS: S0 (serial: feeder_core.run_s0 -> Pipe.layer), S1 (producer / sole CUDA submitter split, descriptor waits for the
previous layer's final-submission acknowledgement), S2 (S1 + at most one next-layer descriptor ahead; packing still waits).
CORES: the 8 physical transport cores (cpupack_cpu team) hold the producer (the coordinator thread, 8 OpenMP pack threads)
AND the submitter thread; the decode launch core and every other core are excluded (audited before the prefill: exit 24).

PER BATCH (one process, the model loaded once; batches FD_BATCHES in order, B336 first):
  prefill -> in-place head-major conversion -> the saved plans (file sha256, sidecar digest, ids, the REGISTERED per-step
  digests) -> PostPrefillSnapshot (hosted) -> GOLDEN pass over the frozen schedule (no transfers) -> restart (bit-identical
  start digest) -> MEASURED pass: every gated step is checked against the golden BEFORE any timing and after the advance.
  Gated step 1: CORRECT (positive only: every layer exact for S0 / S1 / S2, the whole-scratch fingerprint of the full rep
  against the shipped direct gather, synthetic plans) + 3 warm-ups per arm. Then FD_N_FULL full-instrumentation paired
  blocks and FD_N_LIGHT light-instrumentation blocks; a block = for each arm in its registered order: transfer-alone,
  restore -> resident-alone (one resident tick), restore -> the same one tick beside the transfer. Restores are OUTSIDE the
  brackets (time, bytes and the gap to the next gate recorded); every bracket drains fully at its start and its end.
  Every repetition: fresh poisoned scratch, fresh GPU plan replay with a new nonce, the 32 list D2Hs after the gate, live
  descriptors and packing. Each bracket row is appended to <tag>.raw.jsonl as soon as its checks ran (streamed); the gate
  verdict of each gated step follows as its own line. DEADLINE (FD_STAGE_DEADLINE, epoch s): a block starts only if its
  estimate (the slowest block so far) plus FD_FINAL_RESERVE_S fits; the rest is recorded INCOMPLETE (never silently cut).
CONTROLS (FD_MODE=controls, a SEPARATE model-free GPU process run FIRST by the job): every destructive / ownership control
  of control_specs() on two real saved-plan layers with a synthetic host source; each must be DETECTED by its registered
  detector, the clean ones must PASS; any failure stops the job before measurement.
SCORE (FD_MODE=score, CPU only): feeder_metrics.score_dir over the raw streams.
EXIT: 0 ok; 1..19 failure count; 21 correctness / plan mismatch; 22 capacity (memory) stop; 23 crash; 24 placement refused.
"""
import gc
import json
import os
import sys
import threading
import time
import traceback

import cpupack_cpu as CC

MODE = os.environ.get("FD_MODE", "run")                              # run | controls | score
EARLY = None
if __name__ == "__main__" and MODE in ("run", "controls"):
    _gn = os.environ.get("CP_GPU_NUMA_NODE", "").strip()
    EARLY = CC.early_placement(int(os.environ.get("CP_TEAM_MAX", "8")), int(_gn) if _gn.isdigit() else None)

os.environ["CP_MODE"] = "table" if MODE == "score" else "run"
os.environ.setdefault("CP_CONTIG_MB", "1")                           # no contiguous ceiling arm here
os.environ.setdefault("CP_SWEEP", "0")
os.environ.setdefault("CP_CONFIRM", "1")
os.environ.setdefault("CP_CORES", "8")
os.environ.setdefault("CP_CHUNK_KB", "8192")                         # active cap 256 groups
os.environ.setdefault("CP_CHUNK_KB_ALT", "16384")                    # physical slots of 512 groups (the historical alternate)
os.environ.setdefault("CP_RING", "3")
os.environ.setdefault("CP_NCAP", "63")

import torch  # noqa: E402

import cpupack_core as K  # noqa: E402
import cpupack_golden as G  # noqa: E402
import cpupack_plans as PL  # noqa: E402
import cpupack_transport as CT  # noqa: E402
import feeder_core as FC  # noqa: E402
import feeder_metrics as FM  # noqa: E402
import verify_alone as VA  # noqa: E402

CT.EARLY = EARLY
BATCHES = tuple(int(x) for x in os.environ.get("FD_BATCHES", "336 64").split())
FALLBACK = {int(a): int(b) for a, b in (x.split(":") for x in os.environ.get("FD_FALLBACK", "336:320").split() if ":" in x)}
PLAN_STEPS = tuple(int(x) for x in os.environ.get("FD_PLAN_STEPS", "4 5 6 7").split())
N_FULL = int(os.environ.get("FD_N_FULL", "12"))
N_LIGHT = int(os.environ.get("FD_N_LIGHT", "3"))
WARM_PER_ARM = int(os.environ.get("FD_WARM_PER_ARM", "3"))
FULL_ORDERS = tuple(os.environ.get("FD_FULL_ORDERS", " ".join(FC.FULL_ORDERS)).split())
LIGHT_ORDERS = tuple(os.environ.get("FD_LIGHT_ORDERS", " ".join(FC.LIGHT_ORDERS)).split())
SRC_LAYOUT, DST_LAYOUT = "hm", "orig"
PACKER, PLACER = os.environ.get("FD_PACKER", "group"), os.environ.get("FD_PLACER", "row")
TEAM = int(os.environ.get("CP_TEAM_MAX", "8"))
DEADLINE = float(os.environ.get("FD_STAGE_DEADLINE", "0") or 0) or None
FINAL_RESERVE_S = float(os.environ.get("FD_FINAL_RESERVE_S", "60"))
BLOCK_EST_S = float(os.environ.get("FD_BLOCK_EST_S", "15"))
CORRECT_EST_S = float(os.environ.get("FD_CORRECT_EST_S", "60"))
BATCH_EST_S_PER_REQ = float(os.environ.get("FD_BATCH_EST_S_PER_REQ", "3.5"))
BATCH_EST_FIXED_S = float(os.environ.get("FD_BATCH_EST_FIXED_S", "60"))
LEAK_LIMIT_GB = float(os.environ.get("FD_LEAK_LIMIT_GB", "2.0"))
CONTROLS_B = int(os.environ.get("FD_CONTROLS_B", "64"))
CONTROL_DELAY_MS = float(os.environ.get("FD_CONTROL_DELAY_MS", "20"))
LIST_DELAY_MS = float(os.environ.get("FD_LIST_DELAY_MS", "50"))
ACK_DELAY_S = float(os.environ.get("FD_ACK_DELAY_S", "0.05"))
OUT = VA.OUT
RC_CORRECT, RC_MEMGATE, RC_CRASH, RC_PLACEMENT = CT.RC_CORRECT, CT.RC_MEMGATE, CT.RC_CRASH, CT.RC_PLACEMENT


def saved_for(B):
    return os.environ.get("FD_SAVED_%d" % B, ""), os.environ.get("FD_SAVED_SHA_%d" % B, "")


def registered_digests(B):
    """FD_PLAN_DIGESTS_<B> = 'step:sha256 ...' (the registration's frozen per-step plan digests)."""
    out = {}
    for x in os.environ.get("FD_PLAN_DIGESTS_%d" % B, "").split():
        s, d = x.split(":", 1)
        out[int(s)] = d
    return out


class _CorrectFail(Exception):
    pass


def snapshot_bytes(snap) -> int:
    """Bytes the CounterSnapshot restore copies (every saved tensor of every layer): the restoration traffic."""
    tot = 0
    for slot in getattr(snap, "layers", []) or []:
        for part in ("engine", "layer"):
            for v in (slot.get(part) or {}).values():
                if torch.is_tensor(v):
                    tot += v.numel() * v.element_size()
    return tot


# ------------------------------------------------------------------------------------------------------------- runner
class FeederRunner(CT.Runner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fd = dict(label=FC.LABEL, occupancy_label=FC.OCCUPANCY_LABEL, notes=[], partial=False, incomplete=[], correct=None, schedule=None,
                       schedule_digest=None, plans=None, affinity=None, memory=None, conversion=None, timing={}, restore=None, timing_started=False,
                       controls=None, rows_total=0, rows_not_ok=0)
        self.allow_faults = False
        self.raw = None
        self.raw_path = None
        self.sched = []
        self.restore_bytes = None

    # ------------------------------------------------------------------------------------------- device-agnostic bits
    def sync(self):
        if self.dev == "cuda":
            torch.cuda.synchronize()
        else:
            self.be_main.synchronize()

    def wait_ev(self, ev):
        """Wait for a PUBLISHED handle (a list event or an H2D end event); never touches an event pool."""
        if self.dev == "cuda":
            ev.synchronize()
        else:
            self.be_main.host_wait(ev)

    def ev_ms(self, gate, e):
        if e is None or gate is None:
            return None
        if hasattr(e, "elapsed_time"):
            return round(gate.elapsed_time(e), 5)
        if getattr(e, "t_ns", None) is None or getattr(gate, "t_ns", None) is None:
            return None
        return (e.t_ns - gate.t_ns) / 1e6

    def gate_open(self):
        """GPU: drain, the aux-marker lag, then the 50 ms sleep and the gate on the decode stream (cpupack_transport bracket,
        unchanged); CPU: a completed marker."""
        if self.dev != "cuda":
            return self.be_main.marker(), 0.0
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.bev["lag"].record(self.aux)
        self.bev["lag"].synchronize()
        lag = (time.perf_counter() - t) * 1e6
        self.be_main.event_rec("main")                                  # pre
        torch.cuda._sleep(self.sleep_cycles)
        return self.be_main.event_rec("main"), lag

    # ------------------------------------------------------------------------------------------- setup
    def setup_feeder(self):
        """After setup_early (the coordinator on the 8 team cores with its 8-thread OpenMP pack team, pinned staging and
        device landing of 3 x 17,039,360 bytes, the copy / scatter / plan / aux streams): the sole submitter thread on the
        same 8 cores, its own event pool, the main thread's bracket backend, the warm-ups (outside timing) and the placement
        audit (refused before the prefill)."""
        team = list(self.coord.team[:TEAM])
        self.sub = FC.SubmitterThread(team)
        self.sub.start()
        self.sub.ready.wait()
        self.be_main = K.CudaBackend(dict(main=self.main, plan=self.s_plan), self.aux, pool_size=256)
        self.cap_max_chunks = 4096
        self.be_sub = FC.OwnedBackend(K.CudaBackend(dict(copy=self.s_copy, scatter=self.s_scatter), self.aux, pool_size=6 * self.cap_max_chunks + 64))
        self.split = FC.SplitState(CT.RING)
        self.nonce_t = torch.zeros(1, dtype=torch.int32, device="cuda")
        land, stage = self.land, self.stage

        def warm(w):
            torch.set_num_threads(1)                                     # the submitter never runs a CPU-parallel op
            self.be_sub.bind()
            be = self.be_sub
            be.reset()
            with be.stream("copy"):
                be.event_rec("copy")
                be.memcpy(land[0][:4096], stage[0][:4096], "copy")
                e1 = be.event_rec("copy")
            x = torch.zeros((4, K.D_DEF), dtype=torch.bfloat16, device="cuda")
            with be.stream("scatter"):
                be.stream_wait("scatter", e1)
                be.index_copy(x, torch.arange(4, device="cuda"), x.clone(), "scatter")
                be.event_rec("scatter")
            be.marker()
            torch.cuda.synchronize()
            return threading.get_native_id()
        self.sub.call(warm)
        self.affinity_audit(refuse=True)

    def feeder_tids(self):
        t = dict(main=threading.get_native_id())
        co = getattr(self, "coord", None)
        if co is not None and getattr(co, "tid", None):
            t["producer"] = co.tid
            for h in sorted(getattr(co, "helpers", ())):
                t["helper_%d" % h] = h
        sub = getattr(self, "sub", None)
        if sub is not None and sub.tid:
            t["submitter"] = sub.tid
        return t

    def affinity_audit(self, refuse=False, extra=None):
        team = list(self.coord.team[:TEAM])
        tids = self.feeder_tids()
        threads = {k: FC.tid_mask(v) for k, v in tids.items() if k != "main"}
        launch = (EARLY or {}).get("launch")
        bad = FC.audit_affinity(threads, team, forbidden=[launch] if launch is not None else [])
        on_team = []
        for r in CC.census():
            cpus = set(CC.parse_cpulist(r.get("cpus_allowed")))
            if cpus & set(team):
                on_team.append(dict(tid=r["tid"], comm=r["comm"], cpus=r.get("cpus_allowed")))
        rec = dict(team=team, launch=launch, threads=threads, violations=bad, threads_allowed_on_team=on_team,
                   feeder_threads=len(threads), note="producer = the coordinator thread (OpenMP master) + its helpers; the submitter "
                   "shares the same 8 cores (9 runnable transport threads on 8 cores when packing and submitting overlap)")
        if extra:
            rec.update(extra)
        self.fd["affinity"] = rec
        if bad and refuse:
            raise CT.PlacementRefused("transport threads outside the 8 transport cores: %s" % bad)
        return rec

    def memory_record(self):
        cap, cmax = self.cap_main, self.cap_max
        rec = dict(staging=dict(slots=CT.RING, slot_bytes=int(self.stage.shape[1]), allocated_bytes=int(self.stage.numel()),
                                reserved_pow2_bytes=CT.pow2(int(self.stage.numel())), pinned=bool(self.stage.is_pinned()) if self.dev == "cuda" else None,
                                active_cap_groups=cap, active_full_chunk_wire_bytes=K.slot_bytes(cap), physical_slot_groups=cmax,
                                physical_slot_bytes=K.slot_bytes(cmax)),
                   landing=dict(slots=CT.RING, allocated_bytes=int(self.land.numel()),
                                caching_allocator_rounded_bytes=((int(self.land.numel()) + (2 << 20) - 1) // (2 << 20)) * (2 << 20)),
                   plan_list_pinned_bytes=int(self.plan_h.numel() * 4),
                   event_pools=dict(main=len(getattr(self.be_main, "pool", []) or []), submitter=len(getattr(self.be_sub.be, "pool", []) or []),
                                    coordinator=len(getattr(self.cbe, "pool", []) or [])),
                   note="active = what one full chunk uses (cap 256 groups); allocated = the 3-slot rings of 512-group slots kept from "
                        "the historical 16 MiB alternate; reserved = torch 2.6's power-of-two pinned rounding / the 2 MiB caching-allocator rounding")
        if self.dev == "cuda":
            rec["hbm"] = dict(peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                              free_gb=torch.cuda.mem_get_info()[0] / 1e9)
        rec["meminfo"] = CC.meminfo()
        self.fd["memory"] = rec
        return rec

    # ------------------------------------------------------------------------------------------- raw stream
    def raw_open(self):
        self.raw_path = os.path.join(OUT, "%s.raw.jsonl" % CT.TAG)
        if os.path.exists(self.raw_path):
            raise RuntimeError("raw stream %s exists (nothing is overwritten)" % self.raw_path)
        self.raw = open(self.raw_path, "a")
        self.fd["raw_path"] = self.raw_path

    def emit(self, rec):
        rec.setdefault("tag", CT.TAG)
        rec.setdefault("batch", self.B)
        if self.raw is not None:
            self.raw.write(json.dumps(rec, default=str) + "\n")
            self.raw.flush()

    # ------------------------------------------------------------------------------------------- plans and references
    def prepare_rows(self, gpu, cpu, key, layers=None, cap=None, fp=True):
        L = list(range(self.NL)) if layers is None else list(layers)
        cap = int(cap or self.cap_main)
        req = {l: cpu[l][1 + self.HB:].view(self.H, self.B, self.M)[..., :K.TAIL_SLOT].ge(0).sum(dim=(0, 2)).tolist() for l in L}
        chunks = FC.chunk_list(req, L, cap)
        c = dict(key=key, gpu=gpu, cpu=cpu, layers=L, req=req, chunks=chunks, cap=cap, digest=FC.chunk_digest(chunks, cap, PLACER),
                 useful={l: K.useful_bytes(sum(req[l])) for l in L}, max_chunks=len(chunks) + 8)
        c["useful_total"] = sum(c["useful"].values())
        c["union"] = self.dest_mask(cpu, L)
        c["last"] = self.ref_layer(cpu, L[-1])
        c["fp_ref"] = self.reference_fp(gpu, cpu, L) if fp else None
        return c

    def prepare_fd(self, ps, layers=None, cap=None, fp=True):
        """Per plan step, untimed and cached (the registered plan steps only): rows, chunk list + digest, useful bytes, the
        expected rows, the last layer's logical reference, the whole-scratch reference fingerprint."""
        key = (ps, self.host_layout, self.scr_layout, tuple(layers or ()), cap)
        cache = self.__dict__.setdefault("fd_cache", {})
        if key not in cache:
            gpu, cpu = self.rows_of(ps)
            cache[key] = self.prepare_rows(gpu, cpu, key, layers, cap, fp)
        return cache[key]

    def reference_fp(self, gpu, cpu, L):
        """The whole-scratch fingerprint of the layers applied IN ORDER by the shipped direct gather (flash_h2d_from_mask; the
        CPU twin cpupack_core.direct_gather_reference): every arm's final scratch must equal it bit for bit (fp64 of
        cpupack_golden, K and V)."""
        self.sync()
        K.poison_(self.scr_k)
        K.poison_(self.scr_v)
        for l in L:
            if self.dev == "cuda":
                from nosi.flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
                ids = gpu[l][1 + self.HB:].view(self.H, self.B, self.M)
                flash_h2d_from_mask(self.scr_log(0), self.src_log(l, 0), ids, self.R)
                flash_h2d_from_mask(self.scr_log(1), self.src_log(l, 1), ids, self.R)
            else:
                p = cpu[l][1 + self.HB:].view(self.H, self.B, self.M)
                K.direct_gather_reference(self.src_log(l, 0), p, self.scr_log(0))
                K.direct_gather_reference(self.src_log(l, 1), p, self.scr_log(1))
        self.sync()
        return (G.fp_t(self.scr_k), G.fp_t(self.scr_v))

    def synthetic_fd(self, kind, seed=0):
        """Layer-0 synthetic plans (labelled SYNTHETIC; host blocks below the source length)."""
        g = torch.Generator().manual_seed(seed)
        plan = torch.full((self.H, self.B, self.M), -1, dtype=torch.int32)
        nb = min(VA.L, self.S_cpu) // self.R
        tail = min(K.TAIL_SLOT, self.M - 1)                              # 63 at the real geometry (the tail slot never loads)
        if kind == "one_full":
            plan[0, 0, :tail] = torch.randperm(nb, generator=g)[:tail].to(torch.int32)
        elif kind == "dup_src":
            plan[0, 0, 3 % tail] = 5
            plan[0, 0, (40 if tail > 40 else tail - 1)] = 5
            plan[1 % self.H, self.B - 1, 5 % tail] = 5
        elif kind == "perm":
            for h in range(self.H):
                for b in range(self.B):
                    k = min(int(torch.randint(0, 8, (1,), generator=g)), tail)
                    if k:
                        plan[h, b, torch.randperm(tail, generator=g)[:k]] = torch.randperm(nb, generator=g)[:k].to(torch.int32)
        elif kind != "empty":
            raise ValueError(kind)
        cpu = torch.zeros((self.NL, self.W), dtype=torch.int32)
        cpu[:, 1 + self.HB:] = -1
        cpu[0, 1:1 + self.HB] = (plan[..., :tail] >= 0).sum(-1).reshape(-1)
        cpu[0, 1 + self.HB:] = plan.reshape(-1)
        return cpu.to(self.dev), cpu

    # ------------------------------------------------------------------------------------------- the bracket
    def fbracket(self, arm, c, with_decode=False, step_fn=None, layers=None, faults=None, full=True, cap=None, delay_list=None):
        """ONE repetition (cpupack_transport.Runner.bracket's structure): drain, poison the scratch, the gate (50 ms GPU sleep
        then the gate event on the decode stream), the plan stream's replay of every layer (work row + nonce -> pr[l] = the
        common plan-ready origin -> the list D2H into pinned memory -> list[l]), the handoff to the feeder threads, t0, the
        decode (with_decode), tm, wait for the threads, drain. arm 'alone' = the resident-alone bracket (no transfer)."""
        L = list(range(self.NL)) if layers is None else list(layers)
        be = self.be_main
        cap = int(cap or self.cap_main)
        f = faults or FC.FeederFaults()
        if (f.any() or delay_list) and not self.allow_faults:
            raise RuntimeError("a fault or a delay in a measurement process (the controls run in the separate CONTROLS process)")
        self.sync()
        K.poison_(self.scr_k)
        K.poison_(self.scr_v)
        self.plan_h.fill_(-7)
        self.nonce += 1
        nonce = self.nonce
        self.nonce_t.fill_(nonce)
        be.reset()
        spec = run = pipe = None
        if arm in FC.ARMS:
            spec = FC.RepSpec(arm=arm, layers=[FC.LayerIn(l, None, self.plan_h[l], self.host_phys[l][0], self.host_phys[l][1]) for l in L],
                              dst_k=self.scr_k, dst_v=self.scr_v, H=self.H, B=self.B, M=self.M, s_src=self.S_cpu, s_dst=self.S_dst,
                              src_layout=self.host_layout, dst_layout=self.scr_layout, packer=PACKER, placer=PLACER, cap=cap, full=bool(full),
                              faults=f, max_chunks=c["max_chunks"], R=self.R, D=self.D)
            if arm == "S0":
                pipe = self.pipe(cap)
                pipe.reset()
                self.cbe.reset()
            else:
                p0 = K.idx_region(cap, self.R)                           # Pipe.reset's index-region zeroing, for the split arms
                self.stage[:, :p0].zero_()
                self.land[:, :p0].zero_()
                run = FC.SplitRun(spec, self.split, self.be_sub, self.stage, self.land, wait_list=self.wait_ev, wait_handle=self.wait_ev)
        tids = self.feeder_tids()
        s_before = FC.sched_snapshot(tids)
        self.sync()
        gate, lag_us = self.gate_open()
        t_gate_host = time.perf_counter_ns()
        evp, evl = [None] * len(L), [None] * len(L)
        if spec is not None:
            with be.stream("plan"):
                be.stream_wait("plan", gate)
                for j, l in enumerate(L):
                    if delay_list is not None and delay_list[0] == l:
                        be.sleep(delay_list[1], "plan")                  # CONTROLS ONLY: a late list D2H of layer l
                    be.memcpy(self.work[l], c["gpu"][l], "plan")
                    be.memcpy(self.work[l, 0:1], self.nonce_t, "plan")
                    evp[j] = be.event_rec("plan")
                    be.memcpy(self.plan_h[l], self.work[l], "plan")
                    evl[j] = be.event_rec("plan")
                    spec.layers[j].list_ev = evl[j]
        jobs = []
        if arm == "S0":
            jobs.append(self.coord.submit(lambda co: FC.run_s0(spec, self.cbe, pipe, self.wait_ev)))
        elif run is not None:
            jobs.append(self.coord.submit(lambda co: run.produce(co)))
            jobs.append(self.sub.submit(lambda w: run.submit(w)))
        t_go = time.perf_counter_ns()
        t0 = be.event_rec("main")
        out, dec_ns = None, None
        if with_decode:
            h1 = time.perf_counter_ns()
            out = step_fn()
            dec_ns = time.perf_counter_ns() - h1
        tm = be.event_rec("main")
        err = None
        if run is not None:
            try:
                FC.wait_split(run)
            except FC.FeederError as e:
                err = "%s\n%s" % (e, e.origin_tb)
        for j in jobs:
            j.done.wait()
            if j.error and err is None:
                err = j.error
        self.sync()
        s_after = FC.sched_snapshot(tids)
        ms = lambda e: self.ev_ms(gate, e)
        r = dict(type="bracket", bracket=arm, arm=arm, full=bool(full), with_decode=bool(with_decode), nonce=nonce, layers=L, cap=cap,
                 lag_us=round(lag_us, 2), pr=[ms(e) for e in evp] if spec is not None else [], list=[ms(e) for e in evl] if spec is not None else [],
                 t0=ms(t0), tm=ms(tm), t_gate_host=t_gate_host, t_go=t_go, decode_host_ns=dec_ns, sched=FC.sched_delta(s_before, s_after),
                 error=err, list_bytes=(4 * self.W * len(L) if spec is not None else 0),
                 list_padding_bytes=(4 * sum(self.HB * self.M - int((c["cpu"][l][1 + self.HB:] >= 0).sum()) for l in L) if spec is not None else 0))
        if spec is not None and err is None:
            rec = FC.collect_s0(spec, jobs[0].result, cap) if arm == "S0" else run.collect()
            rec["evms_cols"] = FC.EV_COLS
            rec["evms"] = [[ms(e) for e in row] for row in rec.pop("ev")]
            rec.pop("ev_cols", None)
            r["rec"] = rec
            r["useful"] = sum(int(x[FC.CHUNK_COLS.index("useful")]) for x in rec["ch"])
            r["wire"] = sum(int(x[FC.CHUNK_COLS.index("wire")]) for x in rec["ch"])
        if self.dev == "cuda":
            r["pool"] = dict(submitter=len(self.be_sub.be.pool), coordinator=len(self.cbe.pool), main=len(be.pool))
        return out, r

    def fd_checks(self, r, c, expect_mask, ref, layers, fp_ref=None, lg=None, lg_ref=None):
        """Per-repetition correctness OUTSIDE the bracket. Transfer rows: list delivery (nonce + rows), no feeder error, the
        canary of the expected rows (no stray / missing row), the content of the reference layer, the whole-scratch
        fingerprint (when given), the feeder audit (barriers, ordering, ownership, byte conservation, chunk list == the plan's,
        zero final backlog), the chunk-list digest. Rows beside the decode: logits torch.equal the reference, 0 loads."""
        ok, why = True, []
        if r["bracket"] in FC.ARMS:
            ph = self.plan_h
            good = True
            for l in layers:
                good &= int(ph[l, 0]) == r["nonce"] and bool(torch.equal(ph[l].cpu(), torch.cat([torch.tensor([r["nonce"]], dtype=torch.int32),
                                                                                                c["cpu"][l][1:].cpu()])))
            r["list_ok"] = bool(good)
            if not good:
                why.append("list delivery")
            if r.get("error"):
                why.append("feeder error")
            ch = self.check_scratch(expect_mask, ref)
            r.update(ch)
            if not ch["canary_ok"]:
                why.append("canary (extra %d, missing %d)" % (ch["extra_rows"], ch["missing_rows"]))
            if ref is not None and not ch.get("content_ok", True):
                why.append("content")
            if fp_ref is not None:
                got = (G.fp_t(self.scr_k), G.fp_t(self.scr_v))
                r["fp_ok"] = got == tuple(fp_ref)
                if not r["fp_ok"]:
                    why.append("whole-scratch fingerprint")
            rec = r.get("rec")
            exp_chunks = [x for x in c["chunks"] if x[0] in set(layers)]
            if rec is None:
                why.append("no record")
                r["audit"] = ["no record"]
            else:
                r["audit"] = FC.audit_rep(rec, expect_useful=sum(c["useful"][l] for l in layers), expect_chunks=exp_chunks)
                if r["audit"]:
                    why.append("audit")
                got = [(x[FC.CHUNK_COLS.index("l")], x[FC.CHUNK_COLS.index("g0")], x[FC.CHUNK_COLS.index("g1")]) for x in rec["ch"]]
                r["chunk_digest"] = FC.chunk_digest(got, r["cap"], PLACER)
                r["chunk_digest_ok"] = r["chunk_digest"] == FC.chunk_digest(exp_chunks, r["cap"], PLACER)
                if not r["chunk_digest_ok"]:
                    why.append("chunk digest")
        if lg is not None:
            r["logits_equal"] = bool(torch.equal(lg, lg_ref))
            r["loads"] = self.loaded_now()
            if not (r["logits_equal"] and r["loads"] == 0):
                why.append("resident decode (logits_equal %s, loads %d)" % (r["logits_equal"], r["loads"]))
        r["why"] = why
        r["ok"] = not why
        return r["ok"]

    def timed_restore(self, ctx):
        """The sanctioned restore OUTSIDE the bracket: host time to completion (drained), bytes copied."""
        t0 = time.perf_counter_ns()
        ctx["restore"]()
        self.sync()
        t1 = time.perf_counter_ns()
        if not self.restore_bytes:                                       # the snapshot holds its tensors after gated()'s take()
            self.restore_bytes = snapshot_bytes(self.snap)
            self.fd["restore"] = dict(bytes=self.restore_bytes, kind="CounterSnapshot (sanctioned; outside every bracket; drained before the next gate)")
        return dict(restore_ms=(t1 - t0) / 1e6, restore_bytes=self.restore_bytes, t_restore_end=t1)

    def _row(self, r, x, arm, phase, pos, extra=None):
        r.update(kind=x["kind"], block=x["block"], order=x["order"], pos=pos, arm=arm, phase=phase, plan_step=x["plan_step"],
                 gated_step=x["gated_step"])
        if extra:
            r.update(extra)
            if extra.get("t_restore_end") is not None and r.get("t_gate_host") is not None:
                r["restore_gap_ms"] = (r["t_gate_host"] - extra["t_restore_end"]) / 1e6      # host clock - host clock
        self.fd["rows_total"] += 1
        if not r.get("ok"):
            self.fd["rows_not_ok"] += 1
            self.fails += 1
        self.emit(r)
        return r

    # ------------------------------------------------------------------------------------------- CORRECT + warm-ups
    def exact_fd(self, arm, c, l, ref=None, mask=None):
        _, r = self.fbracket(arm, c, False, layers=[l])
        ref = self.ref_layer(c["cpu"], l) if ref is None else ref
        mask = self.dest_mask(c["cpu"], [l]) if mask is None else mask
        ok = self.fd_checks(r, c, mask, ref, [l])
        return dict(arm=arm, layer=l, ok=ok, why=r["why"], audit=r.get("audit"), chunks=(r.get("rec") or {}).get("n_chunks"))

    def correct_fd(self, ctx, ps):
        """CORRECT (positive only; the destructive controls run in the separate CONTROLS process)."""
        res = dict(plan_step=ps, label=FC.LABEL)
        nf = 0
        gpu, cpu = self.rows_of(ps)
        res["store_equals_saved"] = bool(torch.equal(gpu.cpu(), cpu))
        nf += int(not res["store_equals_saved"])
        c = self.prepare_fd(ps)
        per = []
        for l in range(self.NL):
            ref, mask = self.ref_layer(cpu, l), self.dest_mask(cpu, [l])     # one logical reference per layer, every arm
            for arm in FC.ARMS:
                x = self.exact_fd(arm, c, l, ref, mask)
                per.append(x)
                nf += int(not x["ok"])
        res["per_layer"] = per
        res["per_layer_all_ok"] = all(x["ok"] for x in per)
        full = []
        for arm in FC.ARMS:
            _, r = self.fbracket(arm, c, False)
            ok = self.fd_checks(r, c, c["union"], c["last"], c["layers"], fp_ref=c["fp_ref"])
            full.append(dict(arm=arm, ok=ok, why=r["why"], chunks=(r.get("rec") or {}).get("n_chunks"), fp_ok=r.get("fp_ok")))
            nf += int(not ok)
        res["full_rep"] = full
        syn = []
        for kind in ("empty", "one_full", "dup_src", "perm"):
            sg, sc = self.synthetic_fd(kind)
            c2 = self.prepare_rows(sg, sc, ("synthetic", kind), layers=[0], fp=False)
            for arm in FC.ARMS:
                _, r = self.fbracket(arm, c2, False, layers=[0])
                ok = self.fd_checks(r, c2, self.dest_mask(sc, [0]), self.ref_layer(sc, 0), [0])
                syn.append(dict(arm=arm, synthetic=kind, ok=ok, why=r["why"]))
                nf += int(not ok)
        res["synthetic"] = syn
        aff = self.affinity_audit()
        res["affinity_violations"] = aff["violations"]
        nf += int(bool(aff["violations"]))
        ctx["restore"]()
        lg = ctx["step_fn"]()
        self.sync()
        res["resident_zero_load"] = self.loaded_now() == 0 and bool(torch.equal(lg, ctx["lg_ref"]))
        nf += int(not res["resident_zero_load"])
        ctx["restore"]()
        res["fails"] = nf
        return nf, res

    # ------------------------------------------------------------------------------------------- one gated step
    def step_work(self, ctx, x):
        ps = x["plan_step"]
        if x["kind"] == "correct_warmup":
            t = time.time()
            nf, res = self.correct_fd(ctx, ps)
            self.fd["correct"] = res
            self.fd["timing"]["correct_s"] = time.time() - t
            if nf:
                self.fails += nf
                raise _CorrectFail()
            self.memory_record()
            c = self.prepare_fd(ps)
            for i, arm in enumerate(FC.arms_of(x["order"])):
                _, r = self.fbracket(arm, c, False, full=True)
                self.fd_checks(r, c, c["union"], c["last"], c["layers"], fp_ref=c["fp_ref"])
                self._row(r, x, arm, "warmup", i)
            self.fd["timing_started"] = True
            return
        self.fd["timing_started"] = True
        c = self.prepare_fd(ps)
        full = x["kind"] == "full"
        for i, arm in enumerate(FC.arms_of(x["order"])):
            _, r = self.fbracket(arm, c, False, full=full)
            self.fd_checks(r, c, c["union"], c["last"], c["layers"], fp_ref=c["fp_ref"])
            self._row(r, x, arm, "alone", i)
            rs = self.timed_restore(ctx)
            lg, r = self.fbracket("alone", c, True, ctx["step_fn"], full=full)
            self.fd_checks(r, c, None, None, [], lg=lg, lg_ref=ctx["lg_ref"])
            self._row(r, x, arm, "resident", i, rs)
            rs = self.timed_restore(ctx)
            lg, r = self.fbracket(arm, c, True, ctx["step_fn"], full=full)
            self.fd_checks(r, c, c["union"], c["last"], c["layers"], fp_ref=c["fp_ref"], lg=lg, lg_ref=ctx["lg_ref"])
            self._row(r, x, arm, "conc", i, rs)

    def measured_fd(self, pos):
        if EARLY and EARLY.get("launch") is not None:
            CC.set_mask([EARLY["launch"]])                               # the decode launch thread alone on its core
        cur = VA.WARM
        took = []
        for idx, x in enumerate(self.sched):
            it = x["gated_step"]
            est = CORRECT_EST_S if x["kind"] == "correct_warmup" else FM.block_estimate(took, BLOCK_EST_S)
            if not FM.fits(time.time(), est, DEADLINE, FINAL_RESERVE_S):
                left = (DEADLINE - time.time()) if DEADLINE else None
                for y in self.sched[idx:]:
                    self.fd["incomplete"].append(dict(kind=y["kind"], block=y["block"], gated_step=y["gated_step"], order=y["order"],
                                                      why="NOT STARTED: deadline (%.0f s left < estimate %.0f s + reserve %.0f s)"
                                                      % (left or -1, est, FINAL_RESERVE_S)))
                self.fd["partial"] = True
                self.fd["notes"].append("deadline: stopped before gated step %d (%d of %d gated steps done)" % (it, idx, len(self.sched)))
                self.log("DEADLINE: stopped before gated step %d" % it)
                break
            while cur < it:                                              # natural steps between gated steps (none in the
                self.decode(self.forced[:, cur:cur + 1], pos)            # registered schedule: consecutive steps)
                self.sync()
                pos = pos + 1
                cur += 1
            t = time.time()
            try:
                self.gated(it, pos, lambda ctx, x=x: self.step_work(ctx, x))
            except _CorrectFail:
                self.log("CORRECTNESS FAILED (%d): no timing" % self.fails)
                self.emit(dict(type="gate", gated_step=it, ok=False, why="CORRECT failed"))
                self.flush_payload(False)
                return RC_CORRECT
            g = self.gate_log[-1]
            self.emit(dict(type="gate", gated_step=it, kind=x["kind"], block=x["block"], ok=bool(g["ok"]), pre_ok=bool(g["pre"]["ok"]),
                           post_ok=bool(g["post"]["ok"]), work_ran=bool(g["work_ran"]), pre_why=g["pre"].get("why"), post_why=g["post"].get("why")))
            pos = pos + 1
            cur = it + 1
            dt = time.time() - t
            if x["kind"] != "correct_warmup":
                took.append(dt)
            self.fd.setdefault("step_s", []).append(dict(gated_step=it, kind=x["kind"], block=x["block"], seconds=dt))
            self.log("gated step %d (%s %s) done in %.1f s (fails %d)" % (it, x["kind"], x["block"], dt, self.fails))
            self.flush_payload(True)
        self.memory_record()
        self.snap_inventory("after_measure")
        self.flush_payload(False)
        return min(self.fails, 19)

    # ------------------------------------------------------------------------------------------- the batch
    def load_plans(self, path):
        """cpupack_transport.Runner.load_plans on this runner's device (the CPU tests run it too): the saved export -> the
        plan store [step, layer, 1 + HB + HB*M] int32 (0, per-stream counts, the load mask)."""
        z = PL.load_npz(path)
        if int(z["meta"]["batch"]) != self.B or int(z["meta"]["L"]) != VA.L:
            raise RuntimeError("plan export %s is for batch %s L %s" % (path, z["meta"]["batch"], z["meta"]["L"]))
        self.masks, self.maps = z["masks"], z["maps"]
        self.logits_sha, self.step_loaded = z["logits_sha"], z["step_loaded"]
        ncap = self.masks.shape[0]
        m32 = self.masks.to(torch.int32)
        rows = torch.cat([torch.zeros((ncap, self.NL, 1), dtype=torch.int32),
                          (m32[..., :K.TAIL_SLOT] >= 0).sum(-1).to(torch.int32).reshape(ncap, self.NL, -1), m32.reshape(ncap, self.NL, -1)], dim=2)
        self.store = rows.contiguous().to(self.dev)
        return dict(source="export %s (cross-process)" % path, ncap=ncap, run_digest=z["hashes"]["run_digest"], meta=z["meta"],
                    invariants="not recomputed: the export is sha256-verified (I1-I9 = 0 at its 2179683 capture; fresh capture == saved in 2179735)")

    def verify_plans_fd(self, path, sha):
        """The saved export: file sha256, sidecar run digest (PL.load_npz), batch / L / N, the ids, and the REGISTERED
        per-step digests of the plan steps (frozen before the run)."""
        r = dict(path=path, file_sha256=self.file_sha256(path), expected_sha256=sha or None)
        r["file_sha_ok"] = bool(sha) and r["file_sha256"] == sha
        z = PL.load_npz(path)
        meta = z["meta"]
        r["meta"] = {k: meta.get(k) for k in ("batch", "L", "ncap", "ids_sha", "nosi_commit")}
        r["meta_ok"] = int(meta.get("batch", -1)) == self.B and int(meta.get("L", -1)) == VA.L and int(meta.get("ncap", -1)) == VA.N
        r["ids_sha_fresh"] = VA.sha(self.ids.to(torch.float32))
        r["ids_equal"] = r["ids_sha_fresh"] == meta.get("ids_sha")
        sd = z["hashes"]["step_digest"]
        r["step_digests"] = {int(s): sd[s] for s in PLAN_STEPS}
        reg = registered_digests(self.B)
        r["registered_digests"] = reg
        r["digests_ok"] = bool(reg) and all(reg.get(s) == sd[s] for s in PLAN_STEPS)
        r["run_digest"] = z["hashes"]["run_digest"]
        r["ok"] = bool(r["file_sha_ok"] and r["meta_ok"] and r["ids_equal"] and r["digests_ok"])
        return r

    @torch.inference_mode()
    def run_fd(self):
        if getattr(self, "mc", None) is None:
            from nosi.verify import miss_control as mc
            self.mc = mc
        T = self.fd["timing"]
        t = time.time()
        self.setup_early()
        self.setup_feeder()
        T["setup_s"] = time.time() - t
        t = time.time()
        self.setup_model()
        T["prefill_and_model_setup_s"] = time.time() - t
        t = time.time()
        conv = self.convert_host(SRC_LAYOUT)
        T["conversion_s"] = time.time() - t
        self.fd["conversion"] = {k: v for k, v in conv.items() if k != "per_layer"}
        if conv["bad_requests"] or conv["final_layout"] != SRC_LAYOUT:
            self.fails += 1
            self.log("head-major conversion NOT exact (%s bad requests, final %s)" % (conv["bad_requests"], conv["final_layout"]))
            self.flush_payload(False)
            return RC_CORRECT
        path, sha = saved_for(self.B)
        if not path:
            raise SystemExit("no saved plans for B=%d (FD_SAVED_%d)" % (self.B, self.B))
        t = time.time()
        pv = self.verify_plans_fd(path, sha)
        self.fd["plans"] = pv
        if not pv["ok"]:
            self.fails += 1
            self.log("SAVED plans refused: %s" % json.dumps({k: pv[k] for k in ("file_sha_ok", "meta_ok", "ids_equal", "digests_ok")}))
            self.flush_payload(False)
            return RC_CORRECT
        self.capture_rec = self.load_plans(path)
        T["plans_s"] = time.time() - t
        self.sched = FC.block_schedule(PLAN_STEPS, VA.WARM, N_FULL, N_LIGHT, WARM_PER_ARM, FULL_ORDERS, LIGHT_ORDERS)
        self.schedule = ([x["gated_step"] for x in self.sched], [])
        self.fd["schedule"] = self.sched
        self.fd["schedule_digest"] = FC.schedule_digest(self.sched, pv["step_digests"])
        if max(self.schedule[0]) >= VA.N:
            raise RuntimeError("the schedule needs decode step %d >= the no-rollover budget %d" % (max(self.schedule[0]), VA.N))
        t = time.time()
        try:
            pps = self.ss.PostPrefillSnapshot(self.cache).take()
        except torch.cuda.OutOfMemoryError as e:
            raise CT.MemGate("restart snapshot: CUDA out of memory: %s" % str(e)[:200])
        self.d0 = self.start_digest()
        self.payload_extra["restart_snapshot"] = dict(hosted_bytes=G.snapshot_offload(pps), start_digest_families=G.FAMILIES_START)
        gc.collect()
        if self.dev == "cuda":
            torch.cuda.empty_cache()
        self.golden = dict(source="in-process golden pass (option (i))", meta=self.golden_meta(), warm=None, steps={})
        self.golden_mode = "record"
        pos, warm, ok = self.warm_pass("golden")                         # the warm steps must reproduce the saved plans
        if not ok:
            raise CT._ProofFail("the golden pass's warm steps differ from the saved plans (masks / logits)")
        self.golden["warm"] = warm
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        self.gated_sequence(pos)
        gpath = os.path.join(OUT, "%s_golden.json" % CT.TAG)
        G.golden_export(gpath, self.golden)
        self.payload_extra["golden_export"] = gpath
        T["golden_s"] = time.time() - t
        if self.golden_fail:
            self.fails += len(self.golden_fail)
            self.log("the golden pass's own resident check FAILED at steps %s" % self.golden_fail)
            self.flush_payload(False)
            return RC_CORRECT
        self.restart(pps, "after golden")
        del pps
        gc.collect()
        if self.dev == "cuda":
            torch.cuda.empty_cache()
        self.golden_mode = "check"
        t = time.time()
        self.alloc_measure()                                             # the memory gate (a registered capacity stop: exit 22)
        pos, _, ok = self.warm_pass("measured", ref=warm)
        if not ok:
            raise CT._ProofFail("the measured pass's warm steps differ from the golden pass")
        self.snap = self.ss.CounterSnapshot(self.cache)
        self.trans = self.ss.transient_ids(self.model)
        T["measure_setup_s"] = time.time() - t
        self.raw_open()
        self.flush_payload(True)
        return self.measured_fd(pos)

    # ------------------------------------------------------------------------------------------- CONTROLS
    def control_specs_fd(self):
        """(name, arm, faults, layers 'one'|'pair', delay_list on the second layer (ms) or None, expect, detector).
        detector: 'data' = canary / content / whole-scratch fingerprint; 'audit:<text>' = feeder_core.audit_rep names it;
        'error:<Exception>' = the repetition raised it (propagated, both threads returned); 'clean' = PASS."""
        D, LD, AK = CONTROL_DELAY_MS, LIST_DELAY_MS, ACK_DELAY_S
        F = FC.FeederFaults
        return [
            ("clean_S0", "S0", F(), "pair", None, "PASS", "clean"),
            ("clean_S1", "S1", F(), "pair", None, "PASS", "clean"),
            ("clean_S2", "S2", F(), "pair", None, "PASS", "clean"),
            ("delays_with_waits_S1", "S1", F(delay_copy_ms=D, delay_scatter_ms=D), "pair", None, "PASS", "clean"),
            ("delays_with_waits_S2", "S2", F(delay_copy_ms=D, delay_scatter_ms=D, delay_ack_s=AK), "pair", ("second", LD), "PASS", "clean"),
            ("free_on_pop", "S1", F(free_on_pop=True, delay_copy_ms=D), "one", None, "DETECTED", "data"),
            ("free_on_publish", "S1", F(free_on_publish=True, delay_copy_ms=D), "one", None, "DETECTED", "data"),
            ("landing_reuse_before_scatter", "S1", F(no_landing_wait=True, delay_scatter_ms=D), "one", None, "DETECTED", "data"),
            ("scatter_before_own_h2d", "S2", F(no_h2d_wait=True, delay_copy_ms=D), "one", None, "DETECTED", "data"),
            ("stale_generation", "S1", F(stale_generation=(0, 1)), "one", None, "DETECTED", "error:StaleGeneration"),
            ("s2_packing_crosses_barrier", "S2", F(s2_pack_cross=True, delay_ack_s=AK), "pair", None, "DETECTED", "audit:S2 barrier"),
            ("descriptor_before_d2h", "S2", F(desc_before_d2h=True), "pair", ("second", LD), "DETECTED", "data"),
            ("producer_exception", "S1", F(producer_raise_at=(0, 2)), "pair", None, "DETECTED", "error:FeederInjected"),
            ("submitter_exception", "S2", F(submitter_raise_at=(1, 0)), "pair", None, "DETECTED", "error:FeederInjected"),
            ("skip_scatter", "S1", F(skip_scatter_chunk=0), "one", None, "DETECTED", "data"),
            ("poison_stage", "S2", F(poison_stage=True), "one", None, "DETECTED", "data"),
            ("s0_no_slot_wait", "S0", F(free_on_publish=True, delay_copy_ms=D), "one", None, "DETECTED", "data"),
            ("s0_no_landing_wait", "S0", F(no_landing_wait=True, delay_scatter_ms=D), "one", None, "DETECTED", "data"),
            ("s0_no_h2d_wait", "S0", F(no_h2d_wait=True, delay_copy_ms=D), "one", None, "DETECTED", "data"),
            ("after_failures_clean_S2", "S2", F(), "pair", None, "PASS", "clean"),
        ]

    def run_fd_controls(self, ps, extra_cpu=None):
        """Every control of control_specs_fd on plan step ps: the layer with the most groups (and the next layer for the
        pair controls) with the chunk cap lowered until there are more chunks than ring slots (cpupack_transport
        lifetime_control_layer); then the ninth-core and the event-ownership controls. Returns (failures, results)."""
        gpu, cpu = self.rows_of(ps)
        lc = self.lifetime_control_layer(cpu)
        l1 = lc["layer"]
        l2 = (l1 + 1) % self.NL
        cap = lc["cap"]
        res = dict(plan_step=ps, lifetime=lc, layers=[l1, l2], cap=cap, controls=[])
        nf = 0
        if not lc["testable"]:
            res["controls"].append(dict(name="lifetime_layer", verdict="UNTESTABLE", pass_=False))
            return 1, res
        cache = {}
        for name, arm, f, which, dl, expect, det in self.control_specs_fd():
            L = [l1] if which == "one" else [l1, l2]
            key = tuple(L)
            if key not in cache:
                cache[key] = self.prepare_rows(gpu, cpu, ("controls",) + key, layers=L, cap=cap, fp=True)
            c = cache[key]
            delay = (l2, dl[1]) if dl else None
            _, r = self.fbracket(arm, c, False, layers=L, faults=f, cap=cap, delay_list=delay)
            self.fd_checks(r, c, c["union"], c["last"], L, fp_ref=c["fp_ref"])
            data_bad = (not r.get("canary_ok", True)) or (not r.get("content_ok", True)) or (r.get("fp_ok") is False)
            audit = r.get("audit") or []
            err = r.get("error") or ""
            if det == "clean":
                fired = r["ok"]
                verdict = "PASS" if fired else "FAIL"
            elif det == "data":
                fired = data_bad
                verdict = "DETECTED" if fired else "NOT_DETECTED"
            elif det.startswith("audit:"):
                fired = any(det[6:] in a for a in audit)
                verdict = "DETECTED" if fired else "NOT_DETECTED"
            else:
                fired = det[6:] in err
                verdict = "DETECTED" if fired else "NOT_DETECTED"
            alive = self.coord.is_alive() and self.sub.is_alive()
            ok = verdict == expect and alive
            nf += int(not ok)
            res["controls"].append(dict(name=name, arm=arm, layers=L, expect=expect, detector=det, verdict=verdict, pass_=ok, threads_alive=alive,
                                        why=r.get("why"), audit=audit[:6], error=(err.splitlines()[0] if err else None), chunks=(r.get("rec") or {}).get("n_chunks")))
            self.log("CONTROL %-30s expect %-9s got %s" % (name, expect, verdict))
        # the submitter on a ninth core: the placement audit must refuse it
        team = list(self.coord.team[:TEAM])
        nine = extra_cpu if extra_cpu is not None else next((cc for cc in sorted(os.sched_getaffinity(0) | set((EARLY or {}).get("others") or []))
                                                             if cc not in team), None)
        if nine is None:
            v = dict(name="submitter_on_ninth_core", expect="DETECTED", verdict="UNTESTABLE", pass_=False)
        else:
            orig = list(self.sub.cpus)
            self.sub.set_cpus(team + [nine])
            bad = self.affinity_audit()["violations"]
            self.sub.set_cpus(orig)
            back = self.affinity_audit()["violations"]
            v = dict(name="submitter_on_ninth_core", expect="DETECTED", ninth=nine, violations=bad, restored_clean=not back,
                     verdict="DETECTED" if (bad and not back) else "NOT_DETECTED")
            v["pass_"] = v["verdict"] == "DETECTED"
        nf += int(not v["pass_"])
        res["controls"].append(v)
        try:                                                             # a non-owner thread may not touch the submitter's pool
            self.be_sub.event_rec("copy")
            v = dict(name="non_owner_event_allocation", expect="DETECTED", verdict="NOT_DETECTED", pass_=False)
        except FC.OwnershipViolation as e:
            v = dict(name="non_owner_event_allocation", expect="DETECTED", verdict="DETECTED", pass_=True, error=str(e)[:200])
        nf += int(not v["pass_"])
        res["controls"].append(v)
        return nf, res

    def payload(self, partial):
        p = super().payload(partial)
        p["fd"] = dict(self.fd, config=dict(batches=BATCHES, fallback=FALLBACK, plan_steps=PLAN_STEPS, n_full=N_FULL, n_light=N_LIGHT, warm_per_arm=WARM_PER_ARM,
                                            full_orders=FULL_ORDERS, light_orders=LIGHT_ORDERS, src_layout=SRC_LAYOUT, dst_layout=DST_LAYOUT, packer=PACKER,
                                            placer=PLACER, team=TEAM, deadline=DEADLINE, final_reserve_s=FINAL_RESERVE_S, block_est_s=BLOCK_EST_S,
                                            correct_est_s=CORRECT_EST_S, batch_est=(BATCH_EST_S_PER_REQ, BATCH_EST_FIXED_S),
                                            chunk_cols=FC.CHUNK_COLS, ev_cols=FC.EV_COLS, layer_cols=FC.LAYER_COLS))
        return p


# ------------------------------------------------------------------------------------------------------------- main
def write_result(tag, rec):
    path = os.path.join(OUT, "%s.result.json" % tag)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, default=str)
    os.replace(path + ".tmp", path)
    return path


def teardown(runner, model, base_alloc):
    """Drop every reference of one batch (engines, snapshots, plans, scratch, pinned staging, threads) and prove the release
    (interference_curve.teardown)."""
    for th in (getattr(runner, "coord", None), getattr(runner, "sub", None)):
        if th is not None:
            th.q.put(None)
            th.join(timeout=60)
    raw = getattr(runner, "raw", None)
    if raw is not None:
        raw.close()
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
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    if model is not None:
        model.has_buffers = False
    if EARLY and EARLY.get("main_mask_at_init"):
        CC.set_mask(EARLY["main_mask_at_init"])
    alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return dict(allocated_gb=alloc / 1e9, base_gb=base_alloc / 1e9, leaked_gb=(alloc - base_alloc) / 1e9, meminfo=CC.meminfo())


def _flush_to(fn):
    def flush(p):
        tmp = fn + ".tmp"
        with open(tmp, "w") as f:
            json.dump(p, f, default=str)
        os.replace(tmp, fn)
    return flush


@torch.inference_mode()
def controls_main():
    """The CONTROLS process: no model. The saved plans of FD_CONTROLS_B, a synthetic head-major host source for the two
    control layers (S_cpu = L + 8192, the engine's host length), the orig scratch, the same coordinator / submitter /
    staging / landing / streams as a measurement process."""
    B = CONTROLS_B
    path, sha = saved_for(B)
    tag = "ctl_fd_b%d" % B
    CT.TAG = tag
    runner = FeederRunner(None, torch.zeros((B, VA.L + VA.N + 1), dtype=torch.long), [], 0, _flush_to(os.path.join(OUT, "%s.json" % tag)))
    runner.allow_faults = True
    runner.setup_early()
    runner.setup_feeder()
    runner.NL = 32
    runner.S_cpu, runner.S_dst = VA.L + 8192, K.M_DEF * K.R_DEF
    if not path or runner.file_sha256(path) != sha:
        runner.fd["controls"] = dict(error="saved plans %s: sha256 mismatch or missing" % path)
        runner.fails += 1
        runner.flush_payload(False)
        return RC_CORRECT
    runner.capture_rec = runner.load_plans(path)
    runner.work = torch.zeros((runner.NL, runner.W), dtype=torch.int32, device="cuda")
    ps = PLAN_STEPS[0]
    _, cpu = runner.rows_of(ps)
    lc = runner.lifetime_control_layer(cpu)
    l1, l2 = lc["layer"], (lc["layer"] + 1) % runner.NL
    g = torch.Generator().manual_seed(1234)
    runner.host_phys = [None] * runner.NL
    for l in (l1, l2):
        pair = []
        for _ in range(2):                                               # pinned: the shipped direct gather (the reference
            t = K.alloc_phys(SRC_LAYOUT, B, runner.S_cpu, runner.H, runner.D, torch.bfloat16, "cpu")   # fingerprint) reads mapped
            t.view(torch.int16).copy_(torch.randint(-20000, 20000, t.shape, generator=g, dtype=torch.int16))   # host memory
            pair.append(t.pin_memory())
            del t
        runner.host_phys[l] = tuple(pair)
    runner.host_layout = SRC_LAYOUT
    runner.scr_k = K.alloc_phys(DST_LAYOUT, B, runner.S_dst, runner.H, runner.D, torch.bfloat16, "cuda")
    runner.scr_v = K.alloc_phys(DST_LAYOUT, B, runner.S_dst, runner.H, runner.D, torch.bfloat16, "cuda")
    runner.scr_layout = DST_LAYOUT
    runner.chk = {}
    nf, res = runner.run_fd_controls(ps)
    res["label"] = "CONTROLS (model-free; synthetic host source of the real geometry; the saved B%d plans); never timed" % B
    runner.fd["controls"] = res
    runner.fails += nf
    runner.flush_payload(False)
    print("[feeder] CONTROLS: %d of %d failed" % (nf, len(res["controls"])), flush=True)
    return min(nf, 19)


def main():
    if MODE == "score":
        res = FM.score_dir(OUT)
        paths = FM.write_outputs(OUT, res)
        print(FM.render(res), flush=True)
        print("[feeder] score written: %s" % json.dumps(paths), flush=True)
        return 0
    os.makedirs(OUT, exist_ok=True)
    why = CT.placement_problem(EARLY)
    if why:
        print("[feeder] PLACEMENT REFUSED (exit %d): %s; placement %s" % (RC_PLACEMENT, why, json.dumps(EARLY)), flush=True)
        return RC_PLACEMENT
    if MODE == "controls":
        try:
            return controls_main()
        except Exception as e:
            print("[feeder] CONTROLS crashed: %s: %s" % (type(e).__name__, str(e)[:400]), flush=True)
            traceback.print_exc()
            return RC_CRASH
    VA.check_budget()
    path = os.environ["NOSI_MODEL_PATH"]
    todo = list(BATCHES)
    corpus = VA.load_corpus(path, max(todo + [FALLBACK.get(b, 0) for b in todo]))
    model = VA.load_model(path)
    base_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    total, i = 0, 0
    while i < len(todo):
        B = todo[i]
        i += 1
        tag = "fd_b%d" % B
        CT.TAG = tag
        est = FM.predict_batch_s(B, BATCH_EST_S_PER_REQ, BATCH_EST_FIXED_S)
        if not FM.fits(time.time(), est, DEADLINE, 0.0):
            left = DEADLINE - time.time()
            write_result(tag, dict(batch=B, tag=tag, rc=1, est_s=est, left_s=left, **{"class": "NOT_RUN_DEADLINE"}))
            print("[feeder] B=%d NOT RUN: predicted %.0f s > %.0f s left before the deadline (cells INCOMPLETE)" % (B, est, left), flush=True)
            total += 1
            continue
        ids, docs, distinct = VA.pick_batch(corpus, B, 0)
        print("[feeder] B=%d L=%d N=%d docs %s%s (%d distinct) placement %s" % (B, VA.L, VA.N, docs[:6], "..." if len(docs) > 6 else "", distinct,
                                                                           json.dumps(EARLY)), flush=True)
        runner = FeederRunner(model, ids, docs, distinct, _flush_to(os.path.join(OUT, "%s.json" % tag)))
        t_b = time.time()
        cls, err, rc = "OK", None, 0
        try:
            rc = runner.run_fd()
            if rc:
                cls = "FAILED_CHECKS" if rc < 20 else {RC_CORRECT: "CORRECTNESS"}.get(rc, "RC%d" % rc)
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            rc = CT.exit_code_for(runner, e)
            cls = {RC_MEMGATE: "CAPACITY", CT.RC_HYGIENE: "RESTART_PROOF", RC_PLACEMENT: "PLACEMENT", RC_CRASH: "CRASH"}.get(rc, "CRASH")
            err = "%s: %s" % (type(e).__name__, str(e)[:600])
            if rc == RC_MEMGATE:
                runner.memgate.append(dict(trigger=err))
            else:
                runner.crash = dict(kind=cls, exit=rc, error=traceback.format_exc()[-4000:])
            print("[feeder] B=%d stopped (exit %d, %s): %s" % (B, rc, cls, err), flush=True)
            traceback.print_exc()
            try:
                runner.flush_payload(False)
            except Exception:
                pass
        timing_started = bool(getattr(runner, "fd", {}).get("timing_started"))
        res = dict(batch=B, tag=tag, rc=rc, error=err, fails=getattr(runner, "fails", None), seconds=time.time() - t_b, timing_started=timing_started,
                   partial=getattr(runner, "fd", {}).get("partial"), incomplete=len(getattr(runner, "fd", {}).get("incomplete") or []), **{"class": cls})
        if torch.cuda.is_available():
            res.update(peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
        poisoned = rc == RC_CRASH and "CUDA error" in (err or "") and "out of memory" not in (err or "").lower()
        td = teardown(runner, model, base_alloc)
        del runner
        res["teardown"] = td
        write_result(tag, res)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("[feeder] B=%d done rc=%d class=%s (%.0f s); after teardown %.2f GB allocated (leak %.2f GB)" % (
            B, rc, cls, res["seconds"], td["allocated_gb"], td["leaked_gb"]), flush=True)
        total += rc if rc < 20 else 1
        if rc == RC_MEMGATE and not timing_started and B in FALLBACK:
            fb = FALLBACK[B]
            print("[feeder] B=%d CAPACITY STOP before any timing -> the registered common fallback B=%d (ALL arms), if it fits the deadline"
                  % (B, fb), flush=True)
            todo.insert(i, fb)
        if poisoned or td["leaked_gb"] > LEAK_LIMIT_GB:
            print("[feeder] STOP after B=%d (%s)" % (B, "CUDA context poisoned" if poisoned else "leak %.2f GB > %.1f" % (td["leaked_gb"], LEAK_LIMIT_GB)),
                  flush=True)
            return RC_CRASH
    return min(total, 19)


if __name__ == "__main__":
    sys.exit(main())
