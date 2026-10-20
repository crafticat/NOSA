"""R2 step 7, W4: the TC GATE of the CAPTURED fixed-capacity tick (nosi/nosi/r2_tick_capture.py; docs/
R2_TICK_CAPTURE_PLAN.md section 9). One model, the parts one after another in one process; the setup is the tick
gate's (benchmarks/Efficiency/r2_tick_gate.py): B ragged documents of pg19-test-c16k, ONE build_ragged (f4_fix True),
BASE = TickEngine.from_r2 before its first step, REF = the BI serial decode, EXACT GEMMs at m_fix = 4B, piece 3 v2.
Every arm runs SG1's teacher-forced round (forced_schedule; model-free, so the gate knows every tick's rows before it
runs) on its own copy of BASE.

  TE   the EAGER tick (r2_tick.TickEngine, the reference): per tick the rows, the live logits and argmax, the twins'
       counters (pos, n_comp, n_ccis, epoch of every layer and bank) and the host mirrors; the token log and every
       table at the end. MUST: its committed tokens equal REF's; the schedule is complete and equals its simulation.
  TF   FixedTickEngine (the fixed-capacity region, eager)                MUST: == TE every tick and at the end; flags 0
  TG   CapturedTickEngine (one graph per capacity, one pool, warm 1)   MUST: == TE; flags 0; PTR (no state data_ptr
       moved since the last capture); captures == the graphs, replays + eager ticks == ticks
  TX   TG with cap_mode max (every tick on G_v)                         MUST: == TE
  TS   the serial mode on the captured engine (G_d), --ts-steps steps  MUST: every row == REF's row; the control
  TR   the refusals (a bank above T_max, C's twin capacity condition, the token log's trash slot): MUST each refused
       before any write; a tick at the capacity is accepted
  NG1  PadLeak: the padded rows counted into bank 0                     MUST: first difference at EXACTLY the tick
       predicted from TE's plan (the first tick with padding whose bank 0 has room left)
  NG2  StaleTable: the table copy skipped at tick --stale-at             MUST: first difference at exactly --stale-at
  NG3  FrozenCursorCapture: the token log written at the HOST offset of the region's capture (a host value frozen by
       the capture)                                                        MUST: first difference at exactly the first
       tick that replays a graph captured at an EARLIER tick
  T    REPORT: ms per tick of TE, TF and TG (uncompared runs), the speed-up.
OUTPUT: --out/summary.json (written after every part); the last stdout line is R2_TICK_CAPTURE_GATE_DONE failed=<n>;
exit = n (2 = refused). ENVIRONMENT: NOSI_R2_DECODE=1, NOSI_ATTN_SPLITS > 0, NOSI_POOL_BLOCKS = 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_g0_gate as G0           # noqa: E402  (load_workload)
import r2_g0bi_gate as GB         # noqa: E402  (decode_loop, load_by_path, free, gate)
import r2_tick_gate as TGATE      # noqa: E402  (cmp_rows)

PARTS = ("TF", "TG", "TX", "TS", "TR", "NG1", "NG2", "NG3", "T")


def log(msg):
    print("[r2_tick_capture %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--biattn", required=True, help="piece 3's bi_paged_attention.py (a pinned copy; v2 7c826c1)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--S", type=int, default=16128)
    ap.add_argument("--deltas", default="0,1,17,63,64,65,777,4097")
    ap.add_argument("--steps", type=int, default=96, help="committed tokens per request (the first token included)")
    ap.add_argument("--ts-steps", type=int, default=64)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--LAMBDA", type=int, default=7)
    ap.add_argument("--draft-drop", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--m-fix", type=int, default=0, help="G-b's row count; 0 = (1 + K) B")
    ap.add_argument("--warm", type=int, default=1, help="eager ticks per capacity before its capture")
    ap.add_argument("--stale-at", type=int, default=10, help="NG2: the tick whose table copy is skipped")
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args(argv)
    a.deltas = [int(x) for x in a.deltas.split(",") if x != ""]
    a.parts = [p for p in a.parts.split(",") if p]
    if not a.parts or any(p not in PARTS for p in a.parts):
        ap.error("parts: a list of %s" % (PARTS,))
    if len(a.deltas) != a.batch or min(a.deltas) < 0:
        ap.error("--deltas needs %d non-negative values" % a.batch)
    if a.S - max(a.deltas) < 64 * 64:
        ap.error("every prompt needs >= 4096 tokens")
    if a.m_fix == 0:
        a.m_fix = (1 + a.K) * a.batch
    if a.m_fix < (1 + a.K) * a.batch:
        ap.error("m_fix %d < (1 + K) B = %d: the captured tick has R_cap = (1 + K) B rows" % (a.m_fix, (1 + a.K) * a.batch))
    if a.steps < 2 or not 1 <= a.ts_steps <= a.steps or not 0.0 <= a.draft_drop < 1.0:
        ap.error("steps >= 2, 1 <= ts-steps <= steps, draft-drop in [0, 1)")
    if a.warm < 1 or a.stale_at < 1:
        ap.error("warm >= 1, stale-at >= 1")
    return a


# ------------------------------------------------------------------ pure parts (CPU-tested: test_r2_tick_graph.py)
def counts_of(rows, B):
    """Per bank (exact 0 .. B-1, draft B .. 2B-1) the rows of a tick given as [(b, kind, pos)]."""
    c = [0] * (2 * B)
    for (b, kind, _) in rows:
        c[b if kind == "v" else B + b] += 1
    return c


def plan_caps(plan, probe):
    """The capacity (R_cap, T_cap) of every tick of a plan ([[(b, kind, pos)]] per tick), by probe.choose_cap."""
    return [probe.choose_cap(len(rows), max(counts_of(rows, probe.B))) for rows in plan]


def predicted_pad_leak(plan, caps, B):
    """NG1: the first tick with padding (R < R_cap) whose bank 0 has room left (its rows < T_cap)."""
    for t, (rows, (R_cap, T_cap)) in enumerate(zip(plan, caps)):
        if len(rows) < R_cap and counts_of(rows, B)[0] < T_cap:
            return t
    return None


def predicted_frozen_capture(caps, warm):
    """NG3: per capacity, use `warm` runs eagerly and use `warm` + 1 captures and replays with its own values; the
    first difference is the earliest use `warm` + 2 of any capacity (the first replay of an earlier capture)."""
    seen, first = {}, None
    for t, cap in enumerate(caps):
        seen[cap] = seen.get(cap, 0) + 1
        if seen[cap] == warm + 2 and (first is None or t < first):
            first = t
    return first


# ------------------------------------------------------------------ the planted faults (importable by the CPU tests)
def fault_classes(CapturedTickEngine=None, XFIELDS=None):
    """The NG1-NG3 engines over CapturedTickEngine (default: nosi's; the CPU tests pass the CPU-loaded class)."""
    if CapturedTickEngine is None:
        from nosi.r2_tick_capture import CapturedTickEngine
    if XFIELDS is None:
        from nosi.r2_tick_fixed import XFIELDS

    class PadLeak(CapturedTickEngine):
        """NG1: the padded rows counted into bank 0 (request 0's exact bank advances by them)."""

        def padded_table(self, tab, count, R, cb, lo):
            host = super().padded_table(tab, count, R, cb, lo)
            host[len(XFIELDS) * cb.R_cap] += cb.R_cap - R
            return host

    class StaleTable(CapturedTickEngine):
        """NG2: tick stale_at runs on the table its capacity's buffer already holds (the copy is skipped)."""
        stale_at = 10

        def upload(self, cb, host):
            if self.stats["ticks"] == self.stale_at:
                return
            super().upload(cb, host)

    class FrozenCursorCapture(CapturedTickEngine):
        """NG3: the region writes the token log at a HOST offset (lo, R of the tick it runs in): a captured graph
        keeps the values of its capture tick. The logslot column sends every row to the trash slot instead."""

        def padded_table(self, tab, count, R, cb, lo):
            host = super().padded_table(tab, count, R, cb, lo)
            NF = len(XFIELDS)
            t = host[:NF * cb.R_cap].view(NF, cb.R_cap)
            t[XFIELDS.index("logslot")] = self.LOG - 1
            self._lo, self._R = lo, R
            return host

        def _forward_fixed(self, cb):
            logits, arg = super()._forward_fixed(cb)
            self.tok_log[self._lo:self._lo + self._R].copy_(arg[:self._R])
            return logits, arg
    return dict(NG1=PadLeak, NG2=StaleTable, NG3=FrozenCursorCapture)


# ------------------------------------------------------------------ records and comparisons (GPU)
def counters_of(eng):
    import torch
    return torch.stack([torch.stack((t.pos, t.n_comp, t.n_ccis, t.epoch)) for t in eng.twins])


def host_of(eng):
    return (list(eng.bank_pos), list(eng.committed), list(eng.wb), list(eng.synced), eng.n_tok)


def run_forced(eng, a, toks, on_tick=None):
    from nosi.r2_tick import forced_schedule, run_round
    S_list = eng.S_list
    reject_at, sched = forced_schedule(a.K)

    def teacher(b, p):
        return int(toks[p - S_list[b], b])
    rnd = run_round(eng, [int(toks[0, b]) for b in range(eng.B)], [a.steps] * eng.B, K=a.K, LAMBDA=a.LAMBDA,
                    teacher=teacher, reject_at=reject_at, on_tick=on_tick)
    return rnd, sched


def record_te(eng, a, toks):
    ticks = []

    def on_tick(rows, tokens, out):
        ticks.append(dict(rows=[(r.b, r.kind, r.pos) for r in rows], logits=out.logits.clone(),
                          arg=out.argmax.clone(), ctr=counters_of(eng), host=host_of(eng)))
    rnd, sched = run_forced(eng, a, toks, on_tick)
    return dict(ticks=ticks, rnd=rnd, sched=sched, eng=eng, out=[r.out for r in rnd.reqs],
                log=eng.tok_log[:eng.n_tok].clone())


def final_diff(eng, ref_eng):
    """The tables that differ between two engines after their rounds (the trash slot excluded)."""
    import torch
    bad = []
    n = (ref_eng.NB + 4) * 64
    for l, (t, r) in enumerate(zip(eng.twins, ref_eng.twins)):
        for k in ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis"):
            if not torch.equal(getattr(t, k), getattr(r, k)):
                bad.append("L%d.%s" % (l, k))
        for name in ("store_k", "store_v", "store_bias"):
            if not torch.equal(getattr(eng, name)[l][:, :n], getattr(ref_eng, name)[l][:, :n]):
                bad.append("L%d.%s" % (l, name))
    if eng.n_tok != ref_eng.n_tok or not torch.equal(eng.tok_log[:eng.n_tok], ref_eng.tok_log[:ref_eng.n_tok]):
        bad.append("tok_log")
    return bad


def compare_run(eng, a, toks, te):
    """The forced round on eng compared with TE tick by tick, then at the end. Returns dict(first=(tick, what) or None,
    ticks, error, rnd)."""
    import torch
    first, t_idx = [None], [0]

    def on_tick(rows, tokens, out):
        t = t_idx[0]
        t_idx[0] += 1
        if first[0] is not None:
            return
        if t >= len(te["ticks"]):
            first[0] = (t, "more ticks than TE")
            return
        x = te["ticks"][t]
        what = None
        if [(r.b, r.kind, r.pos) for r in rows] != x["rows"]:
            what = "rows"
        elif not torch.equal(out.logits, x["logits"]):
            what = "logits"
        elif not torch.equal(out.argmax, x["arg"]):
            what = "argmax"
        elif not torch.equal(counters_of(eng), x["ctr"]):
            what = "twin counters"
        elif host_of(eng) != x["host"]:
            what = "host mirrors"
        elif not torch.equal(eng.tok_log[:eng.n_tok], te["log"][:eng.n_tok]):
            what = "token log"
        if what:
            first[0] = (t, what)
    err, rnd = None, None
    try:
        rnd, _ = run_forced(eng, a, toks, on_tick)
    except Exception as e:                                              # noqa: BLE001 (the faults may void a run)
        err = "%s: %s" % (type(e).__name__, str(e)[:300])
    if first[0] is None and err is None:
        if t_idx[0] != len(te["ticks"]):
            first[0] = (t_idx[0], "tick count %d / %d" % (t_idx[0], len(te["ticks"])))
        elif [r.out for r in rnd.reqs] != te["out"]:
            first[0] = (t_idx[0], "committed tokens")
        else:
            bad = final_diff(eng, te["eng"])
            if bad:
                first[0] = (t_idx[0], "final %s" % bad[:6])
    return dict(first=first[0], ticks=t_idx[0], error=err, rnd=rnd)


# ------------------------------------------------------------------ the parts
def part_equal(name, eng, a, toks, te, gates, res, ptr_check=False):
    from nosi.r2_tick_capture import pointer_diff, state_pointers
    ptr = {}
    if ptr_check:
        def post(e, cb, rows):
            if e.stats["captures"] != ptr.get("n"):
                ptr["n"] = e.stats["captures"]
                ptr["snap"] = state_pointers(e)
        eng.post_region = post
    t0 = time.time()
    r = compare_run(eng, a, toks, te)
    sec = time.time() - t0
    ok = r["first"] is None and r["error"] is None
    gates.append(GB.gate("%s_equals_the_eager_tick" % name, ok, True,
                         "first difference %s; error %s; %d ticks" % (r["first"], r["error"], r["ticks"])))
    flags_ok, why = True, "none"
    try:
        eng.check_flags()
    except Exception as e:                                              # noqa: BLE001
        flags_ok, why = False, str(e)[:200]
    gates.append(GB.gate("%s_ERR_flags_clear" % name, flags_ok, True, "selection check / twin overflow: %s" % why))
    st = eng.stats
    info = dict(sec=round(sec, 1), ticks=r["ticks"], caps={str(k): v for k, v in st.get("caps", {}).items()},
                padded_rows=st.get("padded_rows"), decision_reads=st["decision_reads"],
                rollbacks=st["rollbacks"], publications=st["publications"], writebacks=st["writebacks"])
    pre = len(st.get("caps", {})) >= (1 if eng.cap_mode == "max" else 2) and st.get("padded_rows", 0) > 0
    gates.append(GB.gate("%s_precondition_capacities_and_padding" % name, pre, True,
                         "capacities used %s, padded rows %s" % (info["caps"], info["padded_rows"])))
    if hasattr(eng, "graphs"):
        info.update(captures=st["captures"], replays=st["replays"], eager_region=st["eager_region"],
                    capture_ms=round(st["capture_ms"], 1))
        cnt = st["captures"] == len(eng.graphs) and st["replays"] + st["eager_region"] == st["ticks"] and st["replays"] > 0
        gates.append(GB.gate("%s_graphs_captured_and_replayed" % name, cnt, True,
                             "captures %d (graphs %d), replays %d, eager region ticks %d, ticks %d"
                             % (st["captures"], len(eng.graphs), st["replays"], st["eager_region"], st["ticks"])))
    if ptr_check:
        now = state_pointers(eng)
        moved = [k for k, v in ptr.get("snap", {}).items() if now.get(k) != v]
        gates.append(GB.gate("%s_PTR_no_state_pointer_moved" % name, bool(ptr.get("snap")) and not moved, True,
                             "%d state pointers checked after the last capture, moved %s; all changes %s"
                             % (len(ptr.get("snap", {})), moved[:10], pointer_diff(ptr.get("snap", {}), now)[:10])))
    res[name] = info


def part_ts(a, base, ref, gates, res):
    import torch
    from nosi.r2_tick_capture import CapturedTickEngine
    eng = CapturedTickEngine.from_engine(base, v_cap=a.K, warm=a.warm)
    S_list = eng.S_list
    toks = ref["tok_host"]
    eqs, nxts = [], []
    for s in range(a.ts_steps):
        out = eng.run_serial([int(toks[s, b]) for b in range(eng.B)])
        e, n = TGATE.cmp_rows(out.logits, ref["logits"], S_list, [(b, b, S_list[b] + s) for b in range(eng.B)])
        eqs.append(e)
        nxts.append(n)
    eq, nx = torch.stack(eqs).cpu(), torch.stack(nxts).cpu()
    bad = (~eq).nonzero().tolist()
    gates.append(GB.gate("TS_captured_serial_equals_BI_serial", not bad, True,
                         "%d of %d (step, request) rows differ%s" % (len(bad), eq.numel(), "; first %s" % bad[:4] if bad else "")))
    gates.append(GB.gate("TS_control_next_position_differs", not bool(nx.any()), True,
                         "%d rows equal REF's row at the next position" % int(nx.sum())))
    st = eng.stats
    gates.append(GB.gate("TS_one_graph_replayed", st["captures"] == 1 and st["replays"] == a.ts_steps - a.warm, True,
                         "captures %d, replays %d (expected 1, %d)" % (st["captures"], st["replays"], a.ts_steps - a.warm)))
    res["TS"] = dict(steps=a.ts_steps, captures=st["captures"], replays=st["replays"])


def part_tr(a, base, gates, res):
    import torch
    from types import SimpleNamespace
    from nosi.r2_tick import TickError
    from nosi.r2_tick_fixed import FixedTickEngine
    eng = FixedTickEngine.from_engine(base, v_cap=a.K)
    S0 = eng.S_list[0]
    c0 = counters_of(eng).clone()
    out = []
    over = [SimpleNamespace(b=0, kind="v", pos=S0 + i) for i in range(a.K + 1)]
    for label, fn in (("a bank above T_max", lambda: eng.run_rows(over, list(range(1, a.K + 2)))),
                      ("C's twin capacity condition", lambda: eng.capacity_check(
                          [eng.twins[0].n_max - eng.bank_pos[0] + 1] + [0] * (2 * eng.B - 1))),
                      ("the token log's trash slot", None)):
        if fn is None:
            eng.n_tok = eng.LOG - 2
            fn = (lambda: eng.run_rows([SimpleNamespace(b=0, kind="v", pos=S0)], [1]))
        try:
            fn()
            out.append((label, False, "accepted"))
        except TickError as e:
            out.append((label, True, str(e)[:120]))
        eng.n_tok = 0
    unchanged = torch.equal(counters_of(eng), c0) and not eng.bufs and eng.stats["ticks"] == 0
    ok = all(x[1] for x in out) and unchanged
    acc = eng.run_rows(over[:a.K], list(range(1, a.K + 1)))
    gates.append(GB.gate("TR_refused_before_any_write", ok and acc.cap == (eng.R_max, eng.T_max), True,
                         "refusals %s; state unchanged %s; a tick at the capacity accepted at %s"
                         % ([(x[0], x[1]) for x in out], unchanged, acc.cap)))
    res["TR"] = dict(refusals=out)


def part_ng(name, cls, a, base, toks, te, probe, gates, res):
    from nosi.r2_tick_capture import CapturedTickEngine   # noqa: F401
    plan = [x["rows"] for x in te["ticks"]]
    caps = plan_caps(plan, probe)
    if name == "NG1":
        pred = predicted_pad_leak(plan, caps, probe.B)
    elif name == "NG2":
        pred = a.stale_at if a.stale_at < len(plan) else None
    else:
        pred = predicted_frozen_capture(caps, a.warm)
    eng = cls.from_engine(base, v_cap=a.K, warm=a.warm)
    if name == "NG2":
        eng.stale_at = a.stale_at
    r = compare_run(eng, a, toks, te)
    got = r["first"][0] if r["first"] else None
    gates.append(GB.gate("%s_first_difference_at_the_predicted_tick" % name, pred is not None and got == pred, True,
                         "predicted %s, first difference %s; error after it: %s" % (pred, r["first"], r["error"])))
    res[name] = dict(predicted=pred, first=r["first"], error=r["error"], ticks=r["ticks"])


def part_t(a, base, toks, res):
    import torch
    from nosi.r2_tick_capture import CapturedTickEngine
    from nosi.r2_tick_fixed import FixedTickEngine
    out = {}
    # check=False in all three: the eager tick's selection check reads the device twice per layer (a host sync), the
    # fixed and captured ticks' is a device flag; timing compares the ticks without it
    for name, mk in (("TE", lambda: base.clone(check=False)),
                     ("TF", lambda: FixedTickEngine.from_engine(base, v_cap=a.K, check=False)),
                     ("TG", lambda: CapturedTickEngine.from_engine(base, v_cap=a.K, warm=a.warm, check=False))):
        eng = mk()
        torch.cuda.synchronize()
        t0 = time.time()
        rnd, _ = run_forced(eng, a, toks)
        torch.cuda.synchronize()
        sec = time.time() - t0
        n = max(1, eng.stats["ticks"])
        out[name] = dict(sec=round(sec, 2), ticks=eng.stats["ticks"], ms_per_tick=round(1e3 * sec / n, 2),
                         host_ms_rows_per_tick=round(eng.stats["ms_rows"] / n, 2))
        del eng
        GB.free()
    out["speedup_TG_vs_TE"] = round(out["TE"]["ms_per_tick"] / max(1e-9, out["TG"]["ms_per_tick"]), 2)
    res["T"] = out


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    if not a.model:
        print("REFUSED: no --model / NOSI_MODEL_PATH", file=sys.stderr)
        return 2
    try:
        from nosi.r2_tick import schedule_coverage, simulate_schedule
        S_list = [a.S - d for d in a.deltas]
        sim_log, _ = simulate_schedule(S_list, [a.steps] * a.batch, K=a.K, LAMBDA=a.LAMBDA)
        cov = schedule_coverage(sim_log, a.K)
        if not cov["complete"]:
            raise SystemExit("REFUSED: the forced schedule would be blind for these prompts and steps: %s" % cov)
        import torch
        import nosi.cache_engine as CE
        import nosi.nosa_llama as NL
        from nosi import NOSALlama as Llama
        from nosi.r2_bi_decode import BIOps, attach_bi
        from nosi.r2_decode_path import build_ragged
        from nosi.r2_tick import TickEngine
        from nosi.r2_tick_capture import CapturedTickEngine
        from nosi.r2_tick_fixed import FixedTickEngine
        if not NL.R2_DECODE:
            raise SystemExit("REFUSED: NOSI_R2_DECODE is not 1 (the BI serial decode is the R2 path's mode)")
        if NL._ATTN_SPLITS <= 0 or CE.POOL_BLOCKS != 0:
            raise SystemExit("REFUSED: NOSI_ATTN_SPLITS must be > 0 and NOSI_POOL_BLOCKS 0")
        docs, sha = G0.load_workload(a.workload, a.batch, a.S)
        os.makedirs(a.out)
        attn = GB.load_by_path(a.biattn, "piece3_bi_paged_attention")
        ops = BIOps(attn, a.m_fix, gemm="exact")
        t0 = time.time()
        model = Llama(model_name=a.model, device="cuda", offload=True)
        log("model loaded in %.0f s" % (time.time() - t0))
        gates, res = [], {}
        n_ref = a.steps + 16
        meta = dict(batch=a.batch, S=a.S, deltas=a.deltas, steps=a.steps, n_ref=n_ref, ts_steps=a.ts_steps, K=a.K,
                    LAMBDA=a.LAMBDA, draft_drop=a.draft_drop, seed=a.seed, m_fix=a.m_fix, warm=a.warm,
                    stale_at=a.stale_at, bi=ops.describe(), biattn=a.biattn, parts=a.parts,
                    attn_splits=NL._ATTN_SPLITS, workload_sha256=sha,
                    nosi_commit=os.environ.get("TICK_COMMIT", "unknown"), torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0),
                    storage="ResidentStore + the trash slot NB + 4 (C's Q-T2); S-C in the graph (W5) is CPU-proven "
                            "only: the tick has no residency seam yet (SG5 step 1-2)")

        def save():
            json.dump(dict(meta=meta, gates=gates, results=res), open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1, sort_keys=True, default=str)

        prompts = [docs[b, :a.S - a.deltas[b]] for b in range(a.batch)]
        with torch.inference_mode():
            t0 = time.time()
            cache, st, first = build_ragged(model, prompts, n_ref + 80, f4_fix=True)
            base = TickEngine.from_r2(model, cache, st, ops, draft_drop=a.draft_drop, drop_seed=a.seed, check=True)
            attach_bi(st, ops)
            pos0 = st.pos_host.view(-1, 1).to(device="cuda", dtype=torch.int32)
            r = GB.decode_loop(model, cache, first, pos0, n_ref)
            ref = dict(logits=r["logits"], tokens=r["tokens"], tok_host=r["tokens"].cpu())
            res["REF"] = dict(steps=n_ref, ms_per_step=round(1e3 * r["sec"] / n_ref, 1), S_list=base.S_list)
            del cache, st, r
            GB.free()
            log("build + REF in %.0f s; TE" % (time.time() - t0))
            toks = ref["tok_host"]
            te = record_te(base.clone(), a, toks)
            committed_ok = [te["out"][b] == [int(toks[s, b]) for s in range(a.steps)] for b in range(a.batch)]
            gates.append(GB.gate("TE_committed_equal_REF", all(committed_ok), True,
                                 "requests whose %d committed tokens equal REF's: %d of %d"
                                 % (a.steps, sum(committed_ok), a.batch)))
            gates.append(GB.gate("TE_schedule_complete_and_simulated", te["sched"] == sim_log and cov["complete"], True,
                                 "the run's %d decisions %s the simulation's %d; coverage %s"
                                 % (len(te["sched"]), "equal" if te["sched"] == sim_log else "DIFFER from",
                                    len(sim_log), cov)))
            res["TE"] = dict(ticks=len(te["ticks"]), rows=te["eng"].stats["rows"])
            save()
            probe = FixedTickEngine.from_engine(base, v_cap=a.K)
            faults = fault_classes()
            for part in a.parts:
                log(part)
                try:
                    if part == "TF":
                        part_equal("TF", FixedTickEngine.from_engine(base, v_cap=a.K), a, toks, te, gates, res)
                    elif part == "TG":
                        part_equal("TG", CapturedTickEngine.from_engine(base, v_cap=a.K, warm=a.warm), a, toks, te,
                                   gates, res, ptr_check=True)
                    elif part == "TX":
                        part_equal("TX", CapturedTickEngine.from_engine(base, v_cap=a.K, warm=a.warm, cap_mode="max"),
                                   a, toks, te, gates, res, ptr_check=True)
                    elif part == "TS":
                        part_ts(a, base, ref, gates, res)
                    elif part == "TR":
                        part_tr(a, base, gates, res)
                    elif part in ("NG1", "NG2", "NG3"):
                        part_ng(part, faults[part], a, base, toks, te, probe, gates, res)
                    else:
                        part_t(a, base, toks, res)
                except Exception as e:
                    traceback.print_exc()
                    gates.append(GB.gate("PART_%s" % part, False, True, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])))
                GB.free()
                save()
        meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        save()
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                    g["detail"][:400]), flush=True)
        if "T" in res:
            print("REPORT T: %s" % json.dumps(res["T"], sort_keys=True), flush=True)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        print("R2_TICK_CAPTURE_GATE_DONE failed=%d" % nfail, flush=True)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
