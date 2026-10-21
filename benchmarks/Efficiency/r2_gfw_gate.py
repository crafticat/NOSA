"""R2 step 7, piece 5: the GPU gate GFW of the gilfree C++ WORKER (Q-C2, 2026-10-21): packer v3's worker
(feature/nosi-r2-gilfree-worker: benchmarks/Efficiency/gilfree_pack.cpp, nosi/nosi/r2_transport_gilfree.py
worker="cpp"), proven on the A100 before the captured tick depends on it. Holder step: r2_gfw_step.sh (run id gfw1).
The packers are loaded BY PATH from the step record (v3: the blobs of this commit; v2: the pins of f2b981a), the nosi
modules by path from --nosi-pkg (a bare package: nosi/__init__.py is not imported).

MUST
  GFW_ready  packer v3 (VERSION 3, the OpenMP build, NativePipe.submit) and the pinned v2 (the OpenMP build) built.
  GFW1       S-C's exact wait WITHOUT its submission wait (CudaDeviceOps._submitted, r2_residency_sc.py:675-683) on the
             C++ worker (tests_cpu/test_r2_transport_gilfree.py GTG1, as a gate): a 4-transfer release whose T4 event
             completes only after a --spin-cycles GPU sleep on a side stream is issued (queued: the host does not wait),
             the main stream spins on the generation board (r2_residency_sc._wait_kernel, compiled before), then
             --launches launches go behind the spin (more than the ~1024-entry launch queue). PASS: the T4 event was
             still pending at issue, err 0, the spin waited > 10 ms (a real spin), every launch ran, the rows are the
             host truth, torch's Stream.wait_event accepts the job's event (_JobEv.wait, S-C's superset path), the
             worker ran on the requested team.
  GFW1n      the negative control: the same with the worker's publication planted off (set_skip_publish): the spin
             ends by its bound (max_spin 2^16) with err 1; every launch still ran; the rows are the host truth.
REPORT
  GFW_T<MiB> per release of <MiB> of K + V (bf16, D 128, R 64: 32 KiB per group; B 64 x H 8 streams, the groups over
             32 layers' sources, the destinations distinct, cap 256 groups per chunk), on --threads OpenMP threads, the
             device idle at each release's start (as one release per decode step): the extension-measured service time
             (total_ns: pack + ring waits + submission) of the C++ WORKER (submit; the worker's own total) against
             packer v2's release() in the same step (A-B-A: v2, worker, v3 release(), v2), and the worker's handoff
             (submit -> the submitted counter, the main thread polling it). Beside: v2's record in b64_s512_gilfree2
             (8 threads: service p50 4.59 ms at 64 MiB, 12.36 ms at 192 MiB; native total 4.20 / 11.43; 4 threads
             15.86 ms at 192 MiB), measured beside a decode.
REGISTERED (frozen 2026-10-21 before gfw1; the step script's header holds the same numbers): per size,
  ratio = median(worker total) / median(v2 total, both v2 blocks):
  PREDICTION |ratio - 1| <= 0.15; FALSIFIER ratio >= 1.30 (the worker slower by 30 % or more); otherwise GREY.
OUTPUT: --out/summary.json. The last stdout line is R2_GFW_DONE failed=<n>; exit = n (2 = refused).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import types

REGISTERED = dict(pred_band=0.15, falsifier_ratio=1.30)
V2_RECORD = {64: dict(svc_p50_ms=4.59, native_total_ms=4.20), 192: dict(svc_p50_ms=12.36, native_total_ms=11.43,
                                                                       svc_p50_ms_t4=15.86)}
B_T, H_T, D_T, R_T, LAYERS_T, NSRC_T, NBLK_T, M_T, CAP_T = 64, 8, 128, 64, 32, 4, 64, 16, 256
GROUP_BYTES = R_T * D_T * 2 * 2                                # K + V of one (b, h, block) in bf16: 32 KiB


def log(msg):
    print("[r2_gfw %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gate(name, passed, must, detail):
    return dict(name=name, passed=bool(passed), must=bool(must), detail=detail)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nosi-pkg", required=True, help="the nosi package directory (nosi/nosi) of the step's commit")
    ap.add_argument("--v3", required=True, help="the directory of packer v3's gilfree_pack.py / .cpp (this commit)")
    ap.add_argument("--v3-build", required=True)
    ap.add_argument("--v2", required=True, help="the directory of the pinned packer v2 (f2b981a)")
    ap.add_argument("--v2-build", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--releases", type=int, default=20, help="timed releases per arm and size (after the warm-ups)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sizes", default="64,192", help="MiB of K + V per release")
    ap.add_argument("--launches", type=int, default=1500, help="launches behind the spin (> the ~1024-entry queue)")
    ap.add_argument("--spin-cycles", type=int, default=200_000_000, help="the GPU sleep before the T4 event")
    a = ap.parse_args(argv)
    a.sizes = [int(s) for s in a.sizes.split(",") if s]
    if not a.sizes or any(s <= 0 or s % 16 or s > 256 for s in a.sizes):
        ap.error("--sizes: MiB multiples of 16 up to 256 (32 KiB groups spread evenly over B 64 x H 8 streams, <= 16 "
                 "destinations each)")
    if not 1 <= a.threads <= 64 or a.releases < 3 or a.warmup < 0 or a.launches < 1100 or a.spin_cycles < 1_000_000:
        ap.error("threads 1 .. 64, releases >= 3, warmup >= 0, launches >= 1100, spin-cycles >= 1e6")
    return a


def by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_nosi(pkg_dir, pkg="r2gfw"):
    """r2_transport_gilfree and r2_residency_sc from pkg_dir as the bare package `pkg` (their relative imports resolve
    inside it; nosi/__init__.py, the model's imports, is not run)."""
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [str(pkg_dir)]
        sys.modules[pkg] = p

    def ld(mod):
        full = pkg + "." + mod
        if full not in sys.modules:
            by_path(os.path.join(str(pkg_dir), mod + ".py"), full)
        return sys.modules[full]
    return ld("r2_transport_gilfree"), ld("r2_residency_sc")


def stats(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    n = len(xs)
    return dict(n=n, mean=sum(xs) / n, median=xs[n // 2], p90=xs[min(n - 1, int(0.9 * n))], min=xs[0], max=xs[-1])


def verdict(ratio):
    """REGISTERED: HELD if |ratio - 1| <= pred_band; FALSIFIED if ratio >= falsifier_ratio; else GREY."""
    if ratio is None:
        return "MISSING"
    if abs(ratio - 1.0) <= REGISTERED["pred_band"]:
        return "HELD"
    return "FALSIFIED" if ratio >= REGISTERED["falsifier_ratio"] else "GREY"


def plan(mib, seed, B=B_T, H=H_T, layers=LAYERS_T, M=M_T, nblk=NBLK_T):
    """One release of mib MiB: groups = mib MiB / 32 KiB, every (b, h) stream gets groups / (B H) destinations
    m = 0 .. k - 1 (distinct in the release), each group a random layer (0 .. layers - 1) and block. Returns the five
    int64 CPU vectors (b, h, l, m, blk)."""
    import torch
    groups = mib * (1 << 20) // GROUP_BYTES
    per = groups // (B * H)
    if per * B * H != groups or not 1 <= per <= M:
        raise ValueError("%d MiB = %d groups do not spread evenly over %d streams with <= %d destinations" % (
            mib, groups, B * H, M))
    g = torch.Generator().manual_seed(seed)
    b = torch.arange(B).repeat_interleave(H * per)
    h = torch.arange(H).repeat_interleave(per).repeat(B)
    m = torch.arange(per).repeat(B * H)
    l = torch.randint(0, layers, (groups,), generator=g)
    blk = torch.randint(0, nblk, (groups,), generator=g)
    return [t.to(torch.int64).contiguous() for t in (b, h, l, m, blk)]


def plan_ok(vecs, B=B_T, H=H_T, M=M_T, nblk=NBLK_T):
    """The plan's destinations (b, h, m) are distinct and every field is in range."""
    b, h, l, m, blk = (v.tolist() for v in vecs)
    dst = list(zip(b, h, m))
    return (len(set(dst)) == len(dst) and all(0 <= x < B for x in b) and all(0 <= x < H for x in h)
            and all(0 <= x < M for x in m) and all(0 <= x < nblk for x in blk) and all(x >= 0 for x in l))


# ------------------------------------------------------------------------------------------------ GFW1 / GFW1n
def gfw1(GF, SCG, ext3, planted, max_spin, spin_cycles, launches, threads):
    """GTG1 as a gate (module docstring). Returns its observations."""
    import torch
    dev = torch.device("cuda")
    L, Bp, Hp, Dp, nblk, npos, BS = 2, 2, 2, 64, 16, 12, 64
    g = torch.Generator().manual_seed(5 + int(planted))
    host_k = [torch.randn(Bp, nblk * BS, Hp, Dp, generator=g) for _ in range(L)]
    host_v = [torch.randn(Bp, nblk * BS, Hp, Dp, generator=g) for _ in range(L)]
    sk = [torch.zeros(Bp, npos * BS, Hp, Dp, device=dev) for _ in range(L)]
    sv = [torch.zeros(Bp, npos * BS, Hp, Dp, device=dev) for _ in range(L)]
    gf = GF.GilfreeTransport(ext3, host_k, host_v, sk, sv, BS, Dp, threads=threads,
                             host_wait=lambda ev: ev.synchronize(), cap_groups=4, worker="cpp")
    board = SCG.GenBoard(L, Bp, Hp, npos, dev)
    gf.attach_board(board)
    if planted:
        gf.npipe.set_skip_publish(True)                        # NEGATIVE CONTROL
    groups = [(b, h, (3 * b + h) % nblk, 2 * h + b) for b in range(Bp) for h in range(Hp)]
    gens = [i + 5 for i in range(len(groups))]
    idx = torch.tensor([board.flat(1, b, h, s) for (b, h, _, s) in groups], dtype=torch.int64, device=dev)
    gen = torch.tensor(gens, dtype=torch.int64, device=dev)
    err = torch.zeros((), dtype=torch.int32, device=dev)
    x = torch.zeros((), device=dev)
    kern = SCG._wait_kernel()
    kern[(1,)](board.t, torch.full_like(idx, -1), gen, idx.shape[0], err, max_spin, BLOCK=1024)   # compile, no spin
    torch.cuda.synchronize()                                   # BEFORE the sleep: the T4 event must be pending
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(int(spin_cycles))
        hist = torch.cuda.Event()
        hist.record(side)
    trs = [GF.Transfer(i, "u%d" % b, 1, b, h, blk, s, gens[i], "draft", ("u%d" % b, 1, 1), hist_ev=hist)
           for i, (b, h, blk, s) in enumerate(groups)]
    pending_at_issue = not hist.query()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    gf.issue(1, trs)                                           # queued: the host does not wait for the submission
    issue_ms = 1e3 * (time.perf_counter() - t0)
    e0.record()
    kern[(1,)](board.t, idx, gen, idx.shape[0], err, max_spin, BLOCK=1024)
    e1.record()
    for _ in range(int(launches)):
        x.add_(1.0)
    enqueue_ms = 1e3 * (time.perf_counter() - t0)
    superset = "ok"
    try:
        torch.cuda.current_stream().wait_event(trs[0].ev)      # _JobEv.wait: the superset protocol
    except Exception as e:                                     # an observation, judged by the gate
        superset = "%s: %s" % (type(e).__name__, str(e)[:200])
    torch.cuda.synchronize()
    gf.close()
    rows_ok = all(torch.equal(sk[1][b, s * BS:(s + 1) * BS, h].cpu(), host_k[1][b, k * BS:(k + 1) * BS, h])
                  and torch.equal(sv[1][b, s * BS:(s + 1) * BS, h].cpu(), host_v[1][b, k * BS:(k + 1) * BS, h])
                  for (b, h, k, s) in groups)
    return dict(planted=bool(planted), max_spin=int(max_spin), err=int(err.item()), launches_done=float(x.item()),
                spin_ms=float(e0.elapsed_time(e1)), rows_ok=bool(rows_ok), superset_wait=superset,
                threads_ok=bool(gf.threads_ok()), stats=dict(gf.stats), t4_pending_at_issue=bool(pending_at_issue),
                issue_ms=round(issue_ms, 3), enqueue_ms=round(enqueue_ms, 3),
                board=[int(v) for v in board.t.view(-1)[idx].tolist()], want=[int(v) for v in gen.tolist()])


# ------------------------------------------------------------------------------------------------ GFW_T
def timing(ext2, ext3, sizes, threads, releases, warmup):
    """GFW_T (module docstring): per size the arms v2 release() | v3 worker | v3 release() | v2 release() (A-B-A),
    each `warmup` + `releases` releases of one plan per release (seeded), the device idle at each release's start."""
    import torch
    dev = torch.device("cuda")
    dt = torch.bfloat16
    src_k = [torch.full((B_T, NBLK_T * R_T, H_T, D_T), float(i + 1), dtype=dt) for i in range(NSRC_T)]
    src_v = [torch.full((B_T, NBLK_T * R_T, H_T, D_T), float(-i - 1), dtype=dt) for i in range(NSRC_T)]
    scr_k = torch.zeros((B_T, M_T * R_T, H_T, D_T), dtype=dt, device=dev)
    scr_v = torch.zeros_like(scr_k)
    elem = 2
    keep = []

    def pipe(ext):
        extra = int(getattr(ext, "SLOT_EXTRA_PER_GROUP", 0))
        nb = CAP_T * R_T * 8 + CAP_T * extra + 2 * CAP_T * R_T * D_T * elem
        stage = torch.zeros((2, nb), dtype=torch.uint8, pin_memory=True)
        land = torch.zeros((2, nb), dtype=torch.uint8, device=dev)
        cs, ss = torch.cuda.Stream(), torch.cuda.Stream()
        trip = lambda s: [int(s.stream_id), int(s.device_index), int(s.device_type)]   # noqa: E731
        p = ext.NativePipe(stage, land, CAP_T, R_T, trip(cs), trip(ss))
        p.set_sources(src_k, src_v)
        p.set_threads(threads)
        keep.append((stage, land, cs, ss))
        return p
    p2, p3r, p3w = pipe(ext2), pipe(ext3), pipe(ext3)
    p3w.start_worker(False, False, int(dev.index if dev.index is not None else torch.cuda.current_device()), [])
    dk, dv = scr_k.view(-1, D_T), scr_v.view(-1, D_T)
    out = {}
    for mib in sizes:
        plans = [plan(mib, 1000 * mib + i) for i in range(warmup + releases)]
        lists = [[v.tolist() for v in pl] for pl in plans]

        def run_release(p, tag):
            rec = dict(total_ms=[], wall_ms=[], pack_ms=[], wait_ms=[], submit_ms=[], team=[])
            for i, pl in enumerate(plans):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                st = [int(v) for v in p.release(pl[0], pl[1], pl[2], pl[3], pl[4], dk, dv)]
                wall = 1e3 * (time.perf_counter() - t0)
                if i >= warmup:
                    rec["total_ms"].append(st[6] / 1e6)
                    rec["wall_ms"].append(wall)
                    rec["pack_ms"].append(st[3] / 1e6)
                    rec["wait_ms"].append(st[4] / 1e6)
                    rec["submit_ms"].append(st[5] / 1e6)
                    rec["team"].append((st[7], st[8]))
            torch.cuda.synchronize()
            return rec

        def run_worker():
            rec = dict(total_ms=[], wall_ms=[], handoff_ms=[], pack_ms=[], wait_ms=[], submit_ms=[], team=[])
            for i, ls in enumerate(lists):
                torch.cuda.synchronize()
                w0 = [int(v) for v in p3w.worker_stats()]
                t0 = time.perf_counter()
                seq = int(p3w.submit(ls[0], ls[1], ls[2], ls[3], ls[4], scr_k, scr_v, [], [], [], []))
                while int(p3w.submitted) < seq:                # the main thread polls (the worker needs no GIL)
                    pass
                wall = 1e3 * (time.perf_counter() - t0)
                w1 = [int(v) for v in p3w.worker_stats()]
                err = p3w.take_error()
                if err:
                    raise RuntimeError("the worker's job %d failed: %s" % (seq, err))
                if i >= warmup:
                    tot = (w1[6] - w0[6]) / 1e6
                    rec["total_ms"].append(tot)
                    rec["wall_ms"].append(wall)
                    rec["handoff_ms"].append(wall - tot)
                    rec["pack_ms"].append((w1[3] - w0[3]) / 1e6)
                    rec["wait_ms"].append((w1[4] - w0[4]) / 1e6)
                    rec["submit_ms"].append((w1[5] - w0[5]) / 1e6)
                    rec["team"].append((w1[7], w1[8]))
            torch.cuda.synchronize()
            return rec
        log("GFW_T %d MiB: v2, worker, v3 release, v2" % mib)
        arms = dict(v2_a=run_release(p2, "v2_a"), v3_worker=run_worker(), v3_release=run_release(p3r, "v3_release"),
                    v2_b=run_release(p2, "v2_b"))
        summ = {k: {f: stats(v) for f, v in r.items() if f != "team"} for k, r in arms.items()}
        teams = {k: sorted(set(map(tuple, r["team"]))) for k, r in arms.items()}
        v2_all = arms["v2_a"]["total_ms"] + arms["v2_b"]["total_ms"]
        v2_med = stats(v2_all)["median"]
        w_med = summ["v3_worker"]["total_ms"]["median"]
        ratio = w_med / v2_med if v2_med else None
        out[mib] = dict(groups=len(plans[0][0]), bytes=len(plans[0][0]) * GROUP_BYTES, arms=summ, teams=teams,
                        v2_total_median_ms=v2_med, worker_total_median_ms=w_med, ratio=ratio, verdict=verdict(ratio),
                        teams_ok=all(t == [(threads, threads)] for t in teams.values()), record_v2=V2_RECORD.get(mib))
    p3w.stop_worker()
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    try:
        os.makedirs(a.out)
        gates, res = [], {}
        meta = dict(args=vars(a), commit=os.environ.get("GFW_COMMIT", "unknown"), registered=REGISTERED,
                    v2_record=V2_RECORD)

        def save():
            json.dump(dict(meta=meta, gates=gates, results=res), open(os.path.join(a.out, "summary.json"), "w"),
                      indent=1, sort_keys=True, default=str)
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("REFUSED: no CUDA device")
        meta.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        GF, SCG = load_nosi(a.nosi_pkg)
        ext3 = ext2 = None
        try:
            t0 = time.time()
            gp3 = by_path(os.path.join(a.v3, "gilfree_pack.py"), "gilfree_pack_v3_gate")
            ext3 = gp3.load(a.v3_build)
            gp2 = by_path(os.path.join(a.v2, "gilfree_pack.py"), "gilfree_pack_v2_gate")
            ext2 = gp2.load(a.v2_build)
            ready = (int(getattr(ext3, "VERSION", 0)) == 3 and int(ext3.OPENMP) == 1 and int(ext2.OPENMP) == 1
                     and callable(getattr(ext3.NativePipe, "submit", None)))
            detail = "v3 VERSION %s OPENMP %s; v2 OPENMP %s; built in %.0f s" % (
                getattr(ext3, "VERSION", None), ext3.OPENMP, ext2.OPENMP, time.time() - t0)
        except Exception as e:
            traceback.print_exc()
            ready, detail = False, "EXCEPTION %s: %s" % (type(e).__name__, str(e)[:400])
        gates.append(gate("GFW_ready", ready, True, detail))
        save()
        if ready:
            for name, planted, max_spin in (("GFW1", False, 1 << 24), ("GFW1n", True, 1 << 16)):
                log(name)
                try:
                    r = gfw1(GF, SCG, ext3, planted, max_spin, a.spin_cycles, a.launches, a.threads)
                    res[name] = r
                    if planted:
                        ok = r["err"] == 1 and r["launches_done"] == float(a.launches) and r["rows_ok"]
                    else:
                        ok = (r["err"] == 0 and r["launches_done"] == float(a.launches) and r["spin_ms"] > 10.0
                              and r["rows_ok"] and r["superset_wait"] == "ok" and r["threads_ok"]
                              and r["t4_pending_at_issue"])
                    detail = ("err %d, launches %d/%d, spin %.1f ms, rows %s, superset wait %s, team ok %s, T4 pending "
                              "at issue %s, issue %.2f ms, enqueue %.1f ms, board %s want %s" % (
                                  r["err"], int(r["launches_done"]), a.launches, r["spin_ms"], r["rows_ok"],
                                  r["superset_wait"], r["threads_ok"], r["t4_pending_at_issue"], r["issue_ms"],
                                  r["enqueue_ms"], r["board"], r["want"]))
                except Exception as e:
                    traceback.print_exc()
                    res[name] = dict(error="%s: %s" % (type(e).__name__, str(e)[:400]))
                    ok, detail = False, "EXCEPTION " + res[name]["error"]
                gates.append(gate(name, ok, True, detail))
                save()
            log("GFW_T")
            try:
                t = timing(ext2, ext3, a.sizes, a.threads, a.releases, a.warmup)
                res["GFW_T"] = t
                for mib, r in t.items():
                    rec = r["record_v2"] or {}
                    hs = r["arms"]["v3_worker"]["handoff_ms"]
                    gates.append(gate("R_GFW_T%d" % mib, r["verdict"] == "HELD" and r["teams_ok"], False,
                                      "%s: worker %.2f ms vs v2 %.2f ms (ratio %.3f; band +-%.0f %%, falsifier >= %.2f);"
                                      " v3 release() %.2f ms; worker handoff median %.3f ms; teams ok %s; v2 record "
                                      "(beside a decode) svc p50 %s ms, native %s ms" % (
                                          r["verdict"], r["worker_total_median_ms"], r["v2_total_median_ms"],
                                          r["ratio"], 100 * REGISTERED["pred_band"], REGISTERED["falsifier_ratio"],
                                          r["arms"]["v3_release"]["total_ms"]["median"],
                                          hs["median"] if hs else float("nan"), r["teams_ok"],
                                          rec.get("svc_p50_ms"), rec.get("native_total_ms"))))
            except Exception as e:
                traceback.print_exc()
                res["GFW_T"] = dict(error="%s: %s" % (type(e).__name__, str(e)[:400]))
                gates.append(gate("R_GFW_T", False, False, "EXCEPTION " + res["GFW_T"]["error"]))
            save()
        meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        save()
        for g in gates:
            print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL",
                                    g["detail"][:600]), flush=True)
        nfail = sum(1 for g in gates if g["must"] and not g["passed"])
        print("R2_GFW_DONE failed=%d" % nfail, flush=True)
        return min(nfail, 125)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
