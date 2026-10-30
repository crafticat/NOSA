"""RECON7 cell (a), the PROBE (KS/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs):
I4C's profiler window (benchmarks/Efficiency/r2_i4c_cell.py, a BYTE COPY of nosi-r2-i4c 3aef060, imported here, not
edited) ported onto the best arms L4PTFDGUDCD and L4PTFDGUDKWCD (benchmarks/Efficiency/r2_i4dc_compile.py: CUDA graphs
for the joint section and for the one-row passes, compiled helpers). MEASUREMENT ONLY: no arm's computation changes.
NO NEW KERNEL: torch.profiler, CUDA events, record_function ranges and torch ops on the probe's own counters. The cell:
benchmarks/Efficiency/r2_recon7_cell.py; the protocol: docs/NOTES_recon7.md.

WHY A NEW PROBE (I4C's Probe would change the path of a graph arm). The graph arms run their joint section EAGERLY when
vs.marks is set or the meter times (r2_i4g_graph.py TentStateFDG._why), and a joint section that I4C labels through
vs.marks (_Sink) is an eager section. The one-row passes REPLAY per-layer graphs (r2_i4h_graph.py _dreplay): I4C's lean
row wrappers (_rows, _row_lean, _write_lean, attn) never run in a replay. So this probe:
  * PROFILED ROUNDS (--profile-rounds): vs.marks None, T.checking False, the meter's timing False (the STEADY key of
    both graph families), and ONLY when that key's graphs exist for every layer (no warm, no capture inside a window):
    ONE torch.profiler window per pass "i4c.joint", "i4c.commit", "i4c.draft0", "i4c.draft1" (I4C's range names). The
    COMMIT window is split: it opens when the joint pass returns and closes when the next pass starts (the decision,
    the commit and the D-slot plan, spec_tentative.py:972-1019). After each window: the path check (the joint replayed
    L graphs, no eager / warm / capture; a one-row pass was one replay pass). LABELLED rounds (--label-rounds) add the
    graph-path labels (JOINT_G / JOINT_GK, DRAFT_G, COMMIT_LABELS) as record_function ranges at host boundaries that a
    replay still crosses: a graph's kernels take the range of their cudaGraphLaunch IF CUPTI correlates them (the cell
    reports the unlabelled GPU ops: the GPU run decides).
  * COUNTED ROUNDS (--count-rounds): the joint pass runs as every probe round's does (eager: marks and timing on), with
    (1) GatherTap on rs.ops.kernels["gather_k"] (the meter's wrapper is wrapped, restored after the pass): every host
    gather that the meter counts, classified window load / store fill / other, its NEW blocks and the two R3 D2 flags;
    (2) a _rows_union wrapper that reads vs.fork_len (H, NS) after each layer's fork call: the union pages per (layer,
    KV head, request) (R2 D3) and vs.npg (the pages per row); (3) the meter's block delta of the pass (R2 D7).
  * RECORDING ROUNDS (= every counted round - 1, STEADY rounds that the cell then EXCLUDES from every arm's steady set):
    each D-slot pass that is a graph replay pass runs with a wrapper on T._resid_lean (the eager residency between A(l)
    and B(l)): SkipRecord marks the blocks that a draft row skipped (skip & lm >= 0) and the window slots whose map
    entry the skip cleared (spec_tentative.py:475, r2_i4a_lean.py:219) since the round's joint pass.
  * MARKED ROUNDS (the other probe rounds): I4C's joint marks (the eager probe-round pass: L4PTF's marks + embed, pre,
    twin, select, head, end) and, for a one-row REPLAY pass, the graph-path marks (static, A, resid, B per layer, head,
    end) as CUDA events.
THE PROBE STATISTIC FIX (R2 section 3.5): a probe round is a DRAIN round when a request had finished (count >= N) when
the round began (the runner's host counts h_count, read by the dispatcher: no device read). Every probe median of the
record (the marked joint passes, the marked draft passes, the reference arm's probe mark sum, L7) is given WITHOUT the
drain rounds; the all-rounds value and the difference (the bias of the old statistic) are reported beside it.
D2 FLAGS (R3), per NEW host block of a counted joint pass: flagA "a draft row skipped this block id of this (layer, KV
head, request) since the previous joint pass"; flagB (window loads only) "a draft skip cleared this window slot since
the previous joint pass and no load refilled it". HOST SYNCS: none in a timed round; the profiled / counted passes read
host values only after their pass (probe rounds, outside the steady set).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_i4c_cell as I4C         # noqa: E402  nosi-r2-i4c 3aef060, byte copy (tests check the blob)

I3C, SC, WA = I4C.I3C, I4C.SC, I4C.WA
P3 = I4C.P3
_med, _patch, _unpatch, Marker, _NullClock = I4C._med, I4C._patch, I4C._unpatch, I4C.Marker, I4C._NullClock
_wrap_twin, profile_pass, pass_profile, pass_detail, window_ops = (I4C._wrap_twin, I4C.profile_pass, I4C.pass_profile,
                                                                  I4C.pass_detail, I4C.window_ops)
RANGE_PREFIX, LABEL_PREFIX = I4C.RANGE_PREFIX, I4C.LABEL_PREFIX
JOINT, DRAFT = I4C.JOINT, I4C.DRAFT
JOINT_LAYER = I4C.JOINT_LAYER                       # marked joint passes (eager probe-round path): I4C's labels
JOINT_G = ("pre", "twin", "select", "static", "resid", "section", "post")            # labelled joint pass (graph path)
JOINT_GK = ("pre", "twin", "select", "kpre", "static", "resid", "section", "post")   # the same, the keep arm
DRAFT_G = ("A", "resid", "B")                      # one-row replay pass: graph A(l), the eager residency, graph B(l)
COMMIT_LABELS = ("decide", "accept", "filter", "plant", "fp", "ctr", "commit", "twins", "tokens", "dprep")
COMMIT_PHASE = COMMIT_LABELS[:8]                   # the round's commit phase ev2 .. ev3 (spec_tentative.py:974-999)
CHECK_ONLY_TENT = ("forced", "injected", "edge_rows", "fill_rows")   # TentState's check-only counters (checking False)
WORK_NAMES = (("fork", "unified_attention"), ("flash", "flash_fwd"), ("stage1", "stage1"), ("pool", "pool"))
COUNTERS = ("new_win", "new_store", "new_other", "a_win", "a_store", "b_win", "ab_win", "oor", "skips", "skip_oor")


# ------------------------------------------------------------------ pure parts (CPU-tested: tests_cpu/test_recon7.py)
def expected_labels_r7(kind: str, L: int, keep: bool = False):
    """(labels, layers) of one fully labelled / marked pass. kind "joint": I4C's (the eager marked pass); "jointg": the
    graph path (embed @ 0, JOINT_G or JOINT_GK per layer, head and end @ L); "draftg": a one-row replay pass (static @ 0,
    DRAFT_G per layer, head and end @ L); "commit": COMMIT_LABELS @ 0 (the window closes without an end mark)."""
    if kind == "joint":
        return I4C.expected_labels("joint", L)
    if kind == "commit":
        return list(COMMIT_LABELS), [0] * len(COMMIT_LABELS)
    if kind == "jointg":
        first, per = "embed", (JOINT_GK if keep else JOINT_G)
    elif kind == "draftg":
        first, per = "static", DRAFT_G
    else:
        raise ValueError("kind %r" % kind)
    labs, lays = [first], [0]
    for l in range(L):
        labs += list(per)
        lays += [l] * len(per)
    return labs + ["head", "end"], lays + [L, L]


def drain_rounds(finished) -> list:
    """The DRAIN rounds: rounds whose start had at least one finished request ({round: finished count}; None = unknown,
    not a drain round)."""
    return sorted(int(g) for g, n in (finished or {}).items() if n is not None and int(n) > 0)


def probe_stat(pairs, drain) -> dict:
    """The fixed probe statistic: pairs [(round, value)] -> the median WITHOUT the drain rounds (the value to use), the
    median of ALL rounds (the old statistic) and their difference (all - nodrain: the old statistic's bias)."""
    d = set(int(x) for x in drain)
    al = [v for _, v in pairs if v is not None]
    nd = [v for g, v in pairs if v is not None and int(g) not in d]
    a, n = _med(al), _med(nd)
    return dict(nodrain=n, all=a, bias=(round(a - n, 4) if (a is not None and n is not None) else None),
                n_all=len(al), n_nodrain=len(nd), excluded=sorted(int(g) for g, v in pairs if int(g) in d))


def joint_probe_summary_r7(jst, rd, drain) -> dict:
    """I4C.joint_probe_summary over the marked joint passes WITHOUT the drain rounds ("nodrain", the record's value) and
    over all of them ("all", the old statistic); probe_joint: the probe joint phase per round with probe_stat."""
    d = set(int(x) for x in drain)
    keep = [x for x in jst if int(x["round"]) not in d]
    pj = [(x["round"], rd["phases"][x["round"]][JOINT] if x["round"] < len(rd["phases"]) else None) for x in jst]
    return dict(nodrain=I4C.joint_probe_summary(keep, rd), all=I4C.joint_probe_summary(jst, rd),
                probe_joint=probe_stat(pj, drain), total=probe_stat([(x["round"], x["total"]) for x in jst], drain),
                drain_excluded=sorted(int(x["round"]) for x in jst if int(x["round"]) in d))


def draft_probe_summary_g(dst, rd, drain) -> dict:
    """The marked one-row REPLAY passes (pass_stats with round, slot), without the drain rounds and with all: medians
    per label (static, A, resid, B, head), total, per round the draft phase - the sum of its passes (outside)."""
    d = set(int(x) for x in drain)

    def summ(xs):
        if not xs:
            return dict(n=0)
        labels = sorted({k for x in xs for k in x["labels"]})
        rounds = sorted({x["round"] for x in xs})
        per = []
        for g in rounds:
            ys = [x for x in xs if x["round"] == g]
            ph = rd["phases"][g][DRAFT] if g < len(rd["phases"]) else None
            tot = sum(x["total"] for x in ys if x["total"] is not None)
            per.append(dict(round=g, passes=len(ys), draft_phase=ph, passes_total=round(tot, 4),
                            outside=(round(ph - tot, 4) if (ph is not None and len(ys) == 2) else None)))
        return dict(n=len(xs), rounds=rounds, labels_median={k: _med([x["labels"].get(k, 0.0) for x in xs]) for k in labels},
                    total_median=_med([x["total"] for x in xs]), per_round=per,
                    outside_median=_med([x["outside"] for x in per]), two_pass_rounds=sum(1 for x in per
                                                                                          if x["passes"] == 2))
    return dict(nodrain=summ([x for x in dst if int(x["round"]) not in d]), all=summ(dst),
                total=probe_stat([(x["round"], x["total"]) for x in dst], drain))


def ref_probe_total(marks_rounds, drain) -> dict:
    """A reference arm's probe-round fork phases ([(round, phase dict)], the runner's own probe marks): the median of
    the fork groups' "total" (I3C.fork_groups: L4PTF's marks) with probe_stat (the L7 input without the drain rounds)."""
    pairs = [(g, I3C.fork_groups(p)["total"]) for g, p in (marks_rounds or []) if p]
    return probe_stat(pairs, drain)


def l7_r7(steady_ref, ref_tot, js) -> dict:
    """L7 (I4C.l7_record) with the fixed statistic ("nodrain") and the old one ("all"): gap = REF steady joint - REF probe
    mark sum; its split from the marked passes of the probed arm."""
    return dict(nodrain=I4C.l7_record(steady_ref, ref_tot.get("nodrain"), js.get("nodrain") or {}),
                all=I4C.l7_record(steady_ref, ref_tot.get("all"), js.get("all") or {}))


def work_counts(trace, name: str) -> dict:
    """The work counters of one profiled pass from its Kineto trace (the GPU ops of pass_profile's window): GPU ops,
    kernels, copies, per class (r2_p3_profile classes: count, device ms), GEMM calls, fork calls, H2D gather kernels,
    named counts (WORK_NAMES), the runtime calls that launched graphs."""
    win = window_ops(trace, name)
    cls = {}
    for o in win:
        x = cls.setdefault(o["cls"], [0, 0.0])
        x[0] += 1
        x[1] += o["dur_us"]
    named = {k: sum(1 for o in win if pat in o["name"]) for k, pat in WORK_NAMES}
    ev = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    glaunch = sum(1 for e in ev if e.get("ph") == "X" and e.get("cat") in P3.LAUNCH_CATS
                  and "GraphLaunch" in str(e.get("name", "")))
    return dict(gpu_ops=len(win), kernels=sum(1 for o in win if o["cat"] == "kernel"),
                copies=sum(1 for o in win if o["cat"] != "kernel"),
                classes={k: dict(ops=v[0], ms=round(v[1] / 1e3, 4)) for k, v in sorted(cls.items())},
                gemm_calls=cls.get("GEMM", [0])[0], h2d_kernels=cls.get("H2D", [0])[0],
                h2d_ms=round(cls.get("H2D", [0, 0.0])[1] / 1e3, 4), fork_calls=named["fork"], named=named,
                graph_launches_traced=glaunch, unranged_ops=sum(1 for o in win if o["range"] is None))


def h2d_rate(blocks, block_bytes: int, h2d_ms):
    """(bytes, GB/s) of a pass's NEW host blocks over its H2D kernel ms (None when either is missing or 0)."""
    if blocks is None:
        return None, None
    by = int(blocks) * int(block_bytes)
    return by, (round(by / 1e9 / (h2d_ms / 1e3), 3) if h2d_ms else None)


def commit_record(passes) -> dict:
    """The commit windows: per window device busy, GPU ops, the label device ms; medians; the commit phase (COMMIT_PHASE
    labels) and the decision part (decide + accept) of the labelled windows."""
    cw = [p for p in passes if p.get("slot") == "commit"]
    lab = [p for p in cw if (p.get("detail") or {}).get("labels")]

    def lsum(p, names):
        ls = p["detail"]["labels"]
        return round(sum((ls.get(k) or {}).get("ms", 0.0) for k in names), 4)
    return dict(n=len(cw), rounds=[p.get("round") for p in cw], busy_ms_median=_med([p.get("busy_ms") for p in cw]),
                wall_ms_median=_med([p.get("wall_ms") for p in cw]), gpu_ops_median=_med([p.get("n_gpu_ops") for p in cw]),
                same_round=[p.get("same_round") for p in cw],
                labels=({k: (lab[-1]["detail"]["labels"].get(k) or {}).get("ms") for k in COMMIT_LABELS} if lab else None),
                phase_device_ms=(lsum(lab[-1], COMMIT_PHASE) if lab else None),
                decision_device_ms=(lsum(lab[-1], ("decide", "accept")) if lab else None),
                commit_device_ms=(lsum(lab[-1], ("commit", "twins")) if lab else None))


def pass_work_record(passes, block_bytes: int) -> list:
    """Per profiled joint / draft pass: rows, active rows, the work counters, the NEW host blocks of the pass (R2 D7),
    its H2D bytes and rate, the path check."""
    out = []
    for p in passes:
        if p.get("slot") == "commit":
            continue
        w = p.get("work") or {}
        by, rate = h2d_rate(p.get("new_blocks"), block_bytes, w.get("h2d_ms"))
        out.append(dict(round=p.get("round"), slot=p.get("slot"), labelled=p.get("labelled"), rows=p.get("rows"),
                        rows_e=p.get("rows_e"), active=p.get("active"), gpu_ops=w.get("gpu_ops"),
                        kernels=w.get("kernels"), copies=w.get("copies"), gemm_calls=w.get("gemm_calls"),
                        fork_calls=w.get("fork_calls"), h2d_kernels=w.get("h2d_kernels"), h2d_ms=w.get("h2d_ms"),
                        graph_launches=p.get("graph_launches"), unranged_ops=w.get("unranged_ops"),
                        new_blocks=p.get("new_blocks"), h2d_bytes=by, h2d_GBps=rate, busy_ms=p.get("busy_ms"),
                        path_ok=p.get("path_ok"), path=p.get("path")))
    return out


def counted_record(counted, L: int, H: int, B: int, block_bytes: int) -> dict:
    """The counted joint passes (GatherTap / UnionReader / meter delta, read after the run): per pass and in total the
    NEW host blocks by kind, the D2 flag shares, the union pages per (layer, KV head, request) (R2 D3), the pages per row,
    the store-fill listed blocks per (KV head, request), the meter's NEW blocks (D7)."""
    rows = []
    for c in counted:
        t, u = c.get("tap") or {}, c.get("union") or {}
        new = (t.get("new_win") or 0) + (t.get("new_store") or 0) + (t.get("new_other") or 0)
        a = (t.get("a_win") or 0) + (t.get("a_store") or 0)
        us, up = sum(u.get("u_sum") or []), sum(u.get("u_pos") or [])
        rows.append(dict(round=c["round"], recorded=c.get("recorded"), new=new, meter_new=c.get("meter_new"),
                         new_win=t.get("new_win"), new_store=t.get("new_store"), new_other=t.get("new_other"),
                         flagA=a, flagA_share=(round(a / new, 4) if new else None),
                         flagA_win=t.get("a_win"), flagA_store=t.get("a_store"), flagB_win=t.get("b_win"),
                         flagB_share_win=(round(t["b_win"] / t["new_win"], 4) if t.get("new_win") else None),
                         flagAB_win=t.get("ab_win"), oor=t.get("oor"), skips=c.get("skips"),
                         skip_oor=c.get("skip_oor"), host_calls=t.get("calls"),
                         union_pages_per_stream=(round(us / up, 4) if up else None),
                         union_pages_per_lhb=round(us / float(L * H * B), 4) if (L and H and B) else None,
                         union_max=u.get("u_max"), union_streams=up,
                         pages_per_row=(round(sum(u.get("npg_sum") or []) / sum(u.get("npg_pos") or []), 4)
                                        if sum(u.get("npg_pos") or []) else None),
                         union_layers=u.get("layers"), union_by_layer=[round(a_ / b_, 3) if b_ else None for a_, b_ in
                                                                      zip(u.get("u_sum") or [], u.get("u_pos") or [])],
                         store_listed_sum=t.get("st_sum"), store_listed_max=t.get("st_max"),
                         store_listed_streams=t.get("st_n"), new_bytes=(c.get("meter_new") or 0) * int(block_bytes),
                         eager_layers=c.get("eager_layers")))
    tot = {k: sum((r.get(k) or 0) for r in rows) for k in ("new", "meter_new", "new_win", "new_store", "flagA",
                                                            "flagA_win", "flagA_store", "flagB_win", "flagAB_win")}
    return dict(n=len(rows), passes=rows, total=tot,
                flagA_share=(round(tot["flagA"] / tot["new"], 4) if tot["new"] else None),
                flagB_share_win=(round(tot["flagB_win"] / tot["new_win"], 4) if tot["new_win"] else None),
                union_pages_per_stream_median=_med([r["union_pages_per_stream"] for r in rows]),
                pages_per_row_median=_med([r["pages_per_row"] for r in rows]),
                store_listed_max=max([r["store_listed_max"] or 0 for r in rows], default=None))


# ------------------------------------------------------------------ gates (pure)
def peq_gate_r7(arm, cp, cref, ref: str, steps: int):
    """<arm>-PEQ (MUST): r2_i4a_cell.eq_gate (committed tokens, check-round fingerprints, acceptance record, state
    checksums) of the probed arm against the PLAIN arm `ref` of the same job, bit for bit."""
    n, m, ok, d = I4C.I4A.eq_gate(arm, cp, cref, steps, True)
    return ("%s-PEQ" % arm, True, ok, "against %s: %s" % (ref, d.replace("no L4PTF record", "no %s record" % ref)))


def counts_gate_r7(arm, c, c_ref, ref: str, keys):
    """<arm>-COUNTS (MUST): every counter of `keys` (r2_i4g_cell.COUNT_KEYS) = the plain arm's, except TentState's
    check-only counters in "tent" (CHECK_ONLY_TENT: a profiled pass runs with checking False; declared in the detail)."""
    if c_ref is None:
        return ("%s-COUNTS" % arm, True, False, "no %s record in this job" % ref)
    bad, declared = [], []
    for k in keys:
        a, b = c.get(k), c_ref.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            for s in sorted(set(a) | set(b), key=str):
                if a.get(s) != b.get(s):
                    (declared if (k == "tent" and s in CHECK_ONLY_TENT) else bad).append(
                        "%s.%s: %s vs %s" % (k, s, str(a.get(s))[:60], str(b.get(s))[:60]))
        elif a != b:
            bad.append("%s: differs" % k)
    return ("%s-COUNTS" % arm, True, not bad, "%d keys compared with %s's (not compared: tent.%s, check-only); "
            "differences %d %s; check-only differences %d %s" % (len(keys), ref, "/".join(CHECK_ONLY_TENT), len(bad),
                                                                 bad[:4], len(declared), declared[:4]))


def prof_path_gate(arm, prof_rounds, done, passes, skipped, n_dslots: int):
    """<arm>-PROF-PATH (MUST): every profiled round was profiled; in each, the joint pass, the commit window and
    n_dslots one-row passes have a window; every joint / one-row window ran the steady REPLAY path (path_ok); no profiled
    pass was skipped (a pass whose steady graphs did not exist)."""
    bad = []
    for g in prof_rounds:
        slots = sorted(p.get("slot") for p in passes if p.get("round") == g)
        want = sorted(["joint", "commit"] + ["draft%d" % j for j in range(n_dslots)])
        if slots != want:
            bad.append((g, "windows %s != %s" % (slots, want)))
    bad += [(p.get("round"), "%s path %s" % (p.get("slot"), p.get("path"))) for p in passes
            if p.get("slot") != "commit" and not p.get("path_ok")]
    bad += [(p.get("round"), "%s error %s" % (p.get("slot"), p.get("error"))) for p in passes if p.get("error")]
    sk = [s for s in (skipped or []) if s[0] in set(prof_rounds)]
    ok = bool(prof_rounds) and sorted(done) == sorted(prof_rounds) and not bad and not sk
    return ("%s-PROF-PATH" % arm, True, ok, "profiled rounds %s (done %s); windows %d; skipped %s; bad %s" % (
        list(prof_rounds), list(done), len(passes), sk[:3], bad[:4]))


def marks_neutral_gate_r7(arm, rd, pe: int, mark_rounds, prof_rounds, count_rounds, rec_rounds, xr, rec):
    """<arm>-MARKS-NEUTRAL (MUST): the marked, profiled and counted rounds are probe rounds and pairwise disjoint; the
    recording rounds (counted - 1) are not probe rounds and ARE in the steady exclusion xr (with the capture rounds,
    for every arm); the steady joint median recomputed on the steady set without xr = the record's."""
    G = rd["rounds"]
    probe = {g for g in range(G) if pe > 0 and g > 0 and g % pe == 0}
    xs = set(int(x) for x in xr)
    steady = [g for g in SC.steady_rounds(G, pe) if g not in xs]
    mr, pr, cr, rr = (set(mark_rounds), set(prof_rounds), set(count_rounds), set(rec_rounds))
    jm = WA._med([rd["phases"][g][JOINT] for g in steady if g < len(rd["phases"]) and len(rd["phases"][g]) > JOINT])
    want = (rec.get("phases_median") or {}).get("joint")
    ok = (pe > 0 and (mr | pr | cr) <= probe and not (mr & pr) and not (mr & cr) and not (pr & cr)
          and not (rr & probe) and rr <= xs and not ((mr | pr | cr | rr) & set(steady)) and jm == want)
    return ("%s-MARKS-NEUTRAL" % arm, True, ok,
            "%d rounds, %d probe; marked %s; profiled %s; counted %s; recording %s (in the exclusion %s: %s); steady "
            "rounds used %d; steady joint median recomputed %s = record %s" % (
                G, len(probe), sorted(mr), sorted(pr), sorted(cr), sorted(rr), sorted(xs), rr <= xs, len(steady), jm,
                want))


def marks_shape_r7(arm, jm, dm, L: int, draft_marks: bool):
    """<arm>-MARKS-SHAPE (MUST): every marked joint pass has I4C's labels (expected_labels_r7 "joint"); every marked
    one-row pass the graph-path labels ("draftg"); at least one marked joint pass."""
    okj, dj = I4C.marks_shape([x["seq"] for x in jm], [x["layers"] for x in jm], "joint", L)
    if not draft_marks or not dm:
        okd, dd = True, "draft marks: %s" % ("off" if not draft_marks else "none ran")
    else:
        want, wl = expected_labels_r7("draftg", L)
        bad = [x["round"] for x in dm if list(x["seq"]) != want or list(x["layers"]) != wl]
        okd, dd = not bad, "%d marked one-row replay passes of %d marks; bad rounds %s" % (len(dm), len(want), bad[:3])
    return ("%s-MARKS-SHAPE" % arm, True, okj and okd, "%s; %s" % (dj, dd))


def count_gate(arm, count_rounds, counted, recorded, L: int, n_dslots: int):
    """<arm>-COUNT (MUST): every counted round counted ONE eager joint pass (L eager layers, the union reader at every
    layer, at least L host gathers seen, no capture seen, no block id out of range) and its recording round recorded
    n_dslots one-row replay passes (L residencies each) before it."""
    bad = []
    by = {c["round"]: c for c in counted}
    for g in count_rounds:
        c = by.get(g)
        if c is None:
            bad.append((g, "not counted"))
            continue
        t, u = c.get("tap") or {}, c.get("union") or {}
        if c.get("eager_layers") != L or u.get("layers") != L or (t.get("calls") or 0) < L or t.get("captured"):
            bad.append((g, "eager %s union layers %s host gathers %s captured %s" % (
                c.get("eager_layers"), u.get("layers"), t.get("calls"), t.get("captured"))))
        if (t.get("oor") or 0) or (c.get("skip_oor") or 0):
            bad.append((g, "ids out of range %s / %s" % (t.get("oor"), c.get("skip_oor"))))
        rp = [x for x in recorded if x[0] == g - 1]
        if len(rp) != n_dslots or c.get("recorded_layers") != n_dslots * L:
            bad.append((g, "recorded passes %s (layers %s; want %d x %d)" % (len(rp), c.get("recorded_layers"),
                                                                          n_dslots, L)))
    return ("%s-COUNT" % arm, True, bool(count_rounds) and not bad,
            "counted rounds %s; counted %s; recorded %s; bad %s" % (list(count_rounds), sorted(by),
                                                                    sorted({x[0] for x in recorded}), bad[:4]))


# ------------------------------------------------------------------ the counters (GPU; CPU-tested on stand-ins)
class SkipRecord:
    """R3 D2's draft-skip record (module docstring): blk[l] (H, B, NB + 1) bool, a block id that a draft row's residency
    skipped (column NB: the sentinel of masked entries); clr[l] (H, B, T) bool, the window slots whose map entry a skip
    cleared and no later load refilled. Device ops only; allocated at the first record."""

    def __init__(self, L: int, NB: int):
        self.L, self.NB = int(L), int(NB)
        self.blk = self.clr = None
        self.round = None
        self.passes = 0
        self.layers = 0
        self.n_skip = self.oor = None

    def reset(self, g: int, like) -> None:
        import torch
        H, B, T = like.shape
        if self.blk is None:
            self.blk = torch.zeros((self.L, H, B, self.NB + 1), dtype=torch.bool, device=like.device)
            self.clr = torch.zeros((self.L, H, B, T), dtype=torch.bool, device=like.device)
            self.n_skip = torch.zeros((), dtype=torch.int64, device=like.device)
            self.oor = torch.zeros((), dtype=torch.int64, device=like.device)
        else:
            self.blk.zero_()
            self.clr.zero_()
            self.n_skip.zero_()
            self.oor.zero_()
        self.round, self.passes, self.layers = int(g), 0, 0

    def record(self, l: int, lm, skip) -> None:
        """After a one-row residency of layer l: lm (H, B, T) the load ids (eng._load_mask), skip (H, B, T)."""
        import torch
        m = skip & (lm >= 0)
        inr = lm < self.NB
        idx = torch.where(m & inr, lm, torch.full_like(lm, self.NB))
        self.blk[l].scatter_(-1, idx, m)
        self.clr[l].logical_and_(lm < 0).logical_or_(skip)
        self.n_skip += m.sum()
        self.oor += (m & ~inr).sum()
        self.layers += 1

    def free(self) -> None:
        self.blk = self.clr = None


class GatherTap:
    """rs.ops.kernels["gather_k"] for ONE pass (attach / detach): the wrapped call runs first (the meter's wrapper:
    its block count and spans); then, for a host gather that the meter counts (meter.on and a host source), the NEW
    blocks by kind (window: the destination is the layer's eng._k_gpu; store: vs.ext_k; other), the D2 flags against a
    SkipRecord (rec None: no flags), the store fill's listed ids per (KV head, request). Device counters only; nothing
    while a stream capture is open (it would enter the graph: counted in `captured`)."""

    def __init__(self, runner, rec=None):
        rs = runner.rs
        self.kern = rs.ops.kernels
        self.meter, self.vs, self.rec = runner.meter, runner.vs, rec
        self.layer_of = {lay.cache_engine._k_cpu.data_ptr(): l for l, lay in enumerate(rs.cache.layers)}
        self.win_ptr = {l: lay.cache_engine._k_gpu.data_ptr() for l, lay in enumerate(rs.cache.layers)}
        self.c = None
        self.calls = 0
        self.kinds = dict(win=0, store=0, other=0)
        self.captured = 0
        self.orig = None
        self.st_hist = None

    def attach(self) -> None:
        self.orig = self.kern["gather_k"]
        self._fn = self._gk                                            # ONE bound method: the identity check below
        self.kern["gather_k"] = self._fn

    def detach(self) -> None:
        if self.kern["gather_k"] is not self._fn:
            raise RuntimeError("GatherTap: gather_k is not the tap (another wrapper was installed over it)")
        self.kern["gather_k"] = self.orig

    def _zero(self, dev):
        import torch
        self.c = {k: torch.zeros((), dtype=torch.int64, device=dev) for k in COUNTERS + ("st_sum", "st_max", "st_n")}

    def _gk(self, gpu, cpu, ids, bs):
        out = self.orig(gpu, cpu, ids, bs)
        m = self.meter
        if not (m.on and cpu.data_ptr() in m.host):
            return out
        import torch
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self.captured += 1
            return out
        if self.c is None:
            self._zero(ids.device)
        self.calls += 1
        l = self.layer_of.get(cpu.data_ptr())
        c = self.c
        new = ids >= 0
        n = new.sum()
        if l is not None and gpu.data_ptr() == self.win_ptr[l]:
            kind = "win"
        elif self.vs is not None and getattr(self.vs, "ext_k", None) is not None and gpu.data_ptr() == \
                self.vs.ext_k.data_ptr():
            kind = "store"
        else:
            kind = "other"
        self.kinds[kind] += 1
        c["new_" + kind] += n
        if kind == "store":
            per = new.sum(-1)                                          # (H, B): listed host blocks of the fill
            c["st_sum"] += per.sum()
            c["st_n"] += (per > 0).sum()
            c["st_max"].copy_(torch.maximum(c["st_max"], per.max()))
        rec = self.rec
        if rec is not None and rec.blk is not None and l is not None:
            NB = rec.NB
            inr = ids < NB
            c["oor"] += (new & ~inr).sum()
            ix = torch.where(new & inr, ids, torch.full_like(ids, NB))
            a = new & inr & rec.blk[l].gather(-1, ix)
            if kind == "win":
                c["a_win"] += a.sum()
                if ids.shape[-1] == rec.clr.shape[-1]:
                    b = new & rec.clr[l]
                    c["b_win"] += b.sum()
                    c["ab_win"] += (a & b).sum()
            elif kind == "store":
                c["a_store"] += a.sum()
        return out

    def state(self) -> dict:
        """Host values (after a sync): the counters, the host calls, the kinds, captures seen."""
        out = {k: (int(v) if v is not None else None) for k, v in (self.c or {}).items()}
        out.update(calls=self.calls, kinds=dict(self.kinds), captured=self.captured)
        return out


class UnionReader:
    """T._rows_union (an instance attribute for ONE joint pass): after each layer's section, vs.fork_len (H, NS: the
    union pages per KV head and request, the fork kernel's build_union output) and vs.npg (E, H: the pages per row) into
    device sums per layer. Device ops only."""

    def __init__(self, T, vs, L: int):
        self.T, self.vs, self.L = T, vs, int(L)
        self.acc = None
        self.layers = 0

    def install(self, stack) -> None:
        r0 = self.T._rows_union

        def rows_union(l, *a, **k):
            out = r0(l, *a, **k)
            self._read(l)
            return out
        _patch(stack, self.T, "_rows_union", rows_union)

    def _read(self, l: int) -> None:
        import torch
        vs = self.vs
        fl = vs.fork_len
        if self.acc is None:
            z = lambda: torch.zeros(self.L, dtype=torch.int64, device=fl.device)   # noqa: E731
            self.acc = dict(u_sum=z(), u_pos=z(), npg_sum=z(), npg_pos=z(),
                            u_max=torch.zeros((), dtype=torch.int64, device=fl.device))
        a = self.acc
        f64 = fl.to(torch.int64)
        a["u_sum"][l] += f64.sum()
        a["u_pos"][l] += (f64 > 0).sum()
        a["u_max"].copy_(torch.maximum(a["u_max"], f64.max()))
        n64 = vs.npg.to(torch.int64)
        a["npg_sum"][l] += n64.sum()
        a["npg_pos"][l] += (n64 > 0).sum()
        self.layers += 1

    def state(self) -> dict:
        a = self.acc or {}
        return dict(layers=self.layers, u_sum=[int(x) for x in a["u_sum"].cpu()] if a else None,
                    u_pos=[int(x) for x in a["u_pos"].cpu()] if a else None,
                    npg_sum=[int(x) for x in a["npg_sum"].cpu()] if a else None,
                    npg_pos=[int(x) for x in a["npg_pos"].cpu()] if a else None,
                    u_max=int(a["u_max"]) if a else None)


# ------------------------------------------------------------------ the probe (GPU; CPU-tested on the stand-ins)
class ProbeR7:
    """MEASUREMENT ONLY (module docstring) on a D-arm runner (r2_i4dc_compile: TentRunnerFDGUDCD / TentRunnerFDGUDKCD,
    phase_marks True): wraps runner.T.run_pass (an instance attribute; removed by detach). Rounds g >= 1 only.
    box: the requested rounds, passes (profile windows with round, slot, labelled, rows, active, path, work), done,
    joint / draft (marks), skipped [(round, slot, reason)], finished {round: finished requests at the round start},
    counted, recorded [(round, slot)]."""

    def __init__(self, runner, clock, out_dir, arm: str, profile_rounds=(), label_rounds=(), count_rounds=(),
                 draft_marks: bool = True, block_bytes: int = 0):
        T = runner.T
        for a in ("_g", "_dg", "_dreplay", "_dapply", "_resid_eager", "_static", "_resid_lean"):
            if not hasattr(T, a):
                raise ValueError("ProbeR7 takes a graph D arm (r2_i4dc_compile); the state has no %s" % a)
        pr, lr, cr = (sorted(int(g) for g in x) for x in (profile_rounds, label_rounds, count_rounds))
        if not set(lr) <= set(pr):
            raise ValueError("label rounds %s: a subset of the profiled rounds %s" % (lr, pr))
        if set(cr) & set(pr) or any(g < 2 for g in cr):
            raise ValueError("count rounds %s: >= 2 and not profiled (%s)" % (cr, pr))
        if pr and out_dir is None:
            raise ValueError("profiled rounds need an output directory for the traces")
        self.r, self.T, self.clock = runner, T, clock
        self.out_dir, self.arm = out_dir, arm
        self.L = len(T.rs.st.twins)
        self.keep = hasattr(T, "_kpre")
        self.prof, self.lab, self.cnt = pr, lr, cr
        self.recr = sorted(g - 1 for g in cr)
        self.draft_marks = bool(draft_marks)
        self.block_bytes = int(block_bytes)
        self.box = dict(profile_rounds=list(pr), label_rounds=list(lr), count_rounds=list(cr),
                        recording_rounds=list(self.recr), passes=[], done=[], joint=[], draft=[], skipped=[],
                        finished={}, counted=[], recorded=[])
        self.skips = SkipRecord(self.L, int(T.rs.st.M_alloc))
        self._slots = {}
        self._win = None
        self._dev = []                                                 # (record, key, b0, b1): meter deltas to read
        self._stack = []
        self._run = T.run_pass
        _patch(self._stack, T, "run_pass", self._run_pass)

    def detach(self) -> None:
        if self._win is not None:
            self._close_commit(None, None, why="detach")
        _unpatch(self._stack)
        self.skips.free()

    # -------------------------------------------------------------- the dispatch
    def _fin(self):
        r = self.r
        N, hc = getattr(r, "N", None), getattr(r, "h_count", None)
        if N is None or hc is None:
            return None
        return sum(1 for c in hc if int(c) >= int(N))

    def _run_pass(self, model, plan):
        r = self.r
        g = r.round
        if self._win is not None:
            self._close_commit(g, plan)
        if g is None or g < 1:
            return self._run(model, plan)
        if g not in self.box["finished"]:
            self.box["finished"][g] = self._fin()
        if plan.R > 1:
            slot = "joint"
        elif plan.kind == "draft":
            n = self._slots.get(g, 0)
            self._slots[g] = n + 1
            slot = "draft%d" % n
        else:
            if g in self.prof or g in self.cnt:
                self.box["skipped"].append((g, plan.kind, "a one-row J slot"))
            return self._run(model, plan)
        if g in self.prof:
            return self._profiled(model, plan, g, slot)
        if g in self.cnt:
            return self._counted(model, plan, g) if slot == "joint" else self._run(model, plan)
        if g in self.recr and slot != "joint":
            return self._recording(model, plan, g, slot)
        if r.vs.marks is not None and (slot == "joint" or self.draft_marks):
            return self._marked(model, plan, g, slot)
        return self._run(model, plan)

    # -------------------------------------------------------------- the path checks
    def _joint_replay(self, plan) -> str:
        """None: every layer of this joint pass replays its graph (the steady key exists for every layer); else why."""
        T, vs = self.T, self.r.vs
        why = [T._why(l, vs, plan) for l in range(self.L)]
        if any(w is not None for w in why):
            return "eager %s" % sorted({w for w in why if w})
        key = T._key(plan)
        miss = [l for l in range(self.L) if l not in T._g or T._g[l]["key"] != key]
        return "no graph of key %s for layers %s" % (key, miss[:4]) if miss else None

    def _draft_replay(self, plan) -> str:
        T = self.T
        why = T._dwhy(plan)
        if why is not None:
            return "eager %s" % why
        k = T._dkey(plan)
        return None if k in T._dg else "no draft graphs of key %s" % (k,)

    def _gsnap(self):
        g, d = self.T.graph, self.T.dgraph
        return dict(replays=g["replays"], eager=g["eager"], warm=g["warm"], captures=g["captures"],
                    dreplay_passes=d["replay_passes"], dcapture_passes=d["capture_passes"], dwarm=d["warm"],
                    deager=d["eager"])

    def _path(self, joint: bool, s0, s1):
        d = {k: s1[k] - s0[k] for k in s0}
        if joint:
            ok = (d["replays"] == self.L and d["eager"] == 0 and d["warm"] == 0 and d["captures"] == 0
                  and d["dcapture_passes"] == 0 and d["dwarm"] == 0)
        else:
            ok = (d["dreplay_passes"] == 1 and d["dcapture_passes"] == 0 and d["dwarm"] == 0 and d["deager"] == 0
                  and d["captures"] == 0)
        return ok, d

    # -------------------------------------------------------------- the wrappers of one pass
    def _install_joint_marks(self, stack, mk) -> None:
        """I4C's joint wrappers (Probe._install with one_row False): "pre" / "head" at T._layer, "twin" / "select" at
        each twin's first step / comp_k_flat of the layer."""
        T, L = self.T, self.L
        layer0 = T._layer

        def layer(l, *a, **k):
            mk.mark(l, "pre")
            out = layer0(l, *a, **k)
            if l == L - 1:
                mk.mark(L, "head")
            return out
        _patch(stack, T, "_layer", layer)
        for l, tw in enumerate(T.rs.st.twins):
            _wrap_twin(stack, mk, l, tw)

    def _install_jointg(self, stack, mk) -> None:
        """The graph-path joint labels (JOINT_G / JOINT_GK): I4C's pre / twin / select / head and the host calls around
        the replay: _kpre ("kpre", keep), _static ("static"), _resid_eager ("resid"; "section" when it returns: the
        replay), _kpost ("post", keep) or _rows_union's return ("post")."""
        T = self.T
        self._install_joint_marks(stack, mk)
        st = dict(l=0)

        def wrap(name, label, after=None):
            f0 = getattr(T, name)

            def w(l, *a, **k):
                st["l"] = l
                mk.mark(l, label)
                out = f0(l, *a, **k)
                if after is not None:
                    mk.mark(l, after)
                return out
            _patch(stack, T, name, w)
        if self.keep:
            wrap("_kpre", "kpre")
            wrap("_kpost", "post")
        s0 = T._static                                                 # no layer argument: the layer of _rows_union

        def static(*a, **k):
            mk.mark(st["l"], "static")
            return s0(*a, **k)
        _patch(stack, T, "_static", static)
        wrap("_resid_eager", "resid", after="section")
        if not self.keep:
            r0 = T._rows_union

            def rows_union(l, *a, **k):
                st["l"] = l
                out = r0(l, *a, **k)
                mk.mark(l, "post")
                return out
            _patch(stack, T, "_rows_union", rows_union)
        else:
            r1 = T._rows_union

            def rows_union_k(l, *a, **k):
                st["l"] = l
                return r1(l, *a, **k)
            _patch(stack, T, "_rows_union", rows_union_k)

    def _install_draftg(self, stack, mk) -> None:
        """The graph-path one-row labels (DRAFT_G): "A" @ 0 at _dreplay, "resid" @ l after A(l)'s count application,
        "B" @ l when the eager residency returns, "A" @ l + 1 (or "head" @ L) after B(l)'s count application."""
        T, L = self.T, self.L
        st = dict(n=0)
        d0, a0, r0 = T._dreplay, T._dapply, T._resid_lean

        def dreplay(gs):
            st["n"] = 0
            mk.mark(0, "A")
            return d0(gs)

        def dapply(d):
            out = a0(d)
            n = st["n"]
            st["n"] = n + 1
            l = n // 2
            if n % 2 == 0:
                mk.mark(l, "resid")
            elif l + 1 < L:
                mk.mark(l + 1, "A")
            else:
                mk.mark(L, "head")
            return out

        def resid(l, *a, **k):
            out = r0(l, *a, **k)
            mk.mark(l, "B")
            return out
        _patch(stack, T, "_dreplay", dreplay)
        _patch(stack, T, "_dapply", dapply)
        _patch(stack, T, "_resid_lean", resid)

    # -------------------------------------------------------------- a marked pass (CUDA events)
    def _marked(self, model, plan, g: int, slot: str):
        vs, L = self.r.vs, self.L
        joint = plan.R > 1
        if not joint:
            why = self._draft_replay(plan)
            if why is not None:
                self.box["skipped"].append((g, slot, "marks: " + why))
                return self._run(model, plan)
        sink = vs.marks if joint else []
        mk = Marker(sink, vs.clock if joint else self.clock)
        i0 = len(sink)
        stack = []
        if joint:
            self._install_joint_marks(stack, mk)
        else:
            self._install_draftg(stack, mk)
        try:
            mk.mark(0, "embed" if joint else "static")
            out = self._run(model, plan)
            mk.mark(L, "end")
        finally:
            _unpatch(stack)
        (self.box["joint"] if joint else self.box["draft"]).append((g, slot, list(sink[i0:])))
        return out

    # -------------------------------------------------------------- a profiled pass (torch.profiler)
    def _profiled(self, model, plan, g: int, slot: str):
        r, T = self.r, self.T
        vs, L = r.vs, self.L
        joint = plan.R > 1
        labelled = g in self.lab
        saved = (vs.marks, vs.clock, T.checking, r.meter.timing)
        vs.marks, T.checking, r.meter.timing = None, False, False
        why = self._joint_replay(plan) if joint else self._draft_replay(plan)
        if why is not None:                                            # never a warm / capture inside a window
            vs.marks, vs.clock, T.checking, r.meter.timing = saved
            self.box["skipped"].append((g, slot, "profile: " + why))
            return self._run(model, plan)
        stack = []
        mk = Marker() if labelled else None
        if mk is not None:
            (self._install_jointg if joint else self._install_draftg)(stack, mk)
        b0 = r.meter.blocks.clone()
        s0 = self._gsnap()

        def fn():
            try:
                if mk is not None:
                    mk.mark(0, "embed" if joint else "static")
                out_ = self._run(model, plan)
                if mk is not None:
                    mk.mark(L, "end")
                return out_
            finally:
                if mk is not None:
                    mk.close()
        path = os.path.join(self.out_dir, "trace_%s_g%d_%s.json" % (self.arm, g, slot))
        try:
            out, rec = profile_pass(fn, self.clock, path, RANGE_PREFIX + slot)
        finally:
            _unpatch(stack)
            vs.marks, vs.clock, T.checking, r.meter.timing = saved
        b1 = r.meter.blocks.clone()
        ok, d = self._path(joint, s0, self._gsnap())
        rec.update(round=g, slot=slot, labelled=labelled, rows=int(plan.R), rows_e=int(plan.R) * int(T.B),
                   active=int(plan.act.sum()), path_ok=ok, path=d)
        self._dev.append((rec, "new_blocks", b0, b1))
        self._detail(rec, slot, labelled)
        self.box["passes"].append(rec)
        if g not in self.box["done"]:
            self.box["done"].append(g)
        if joint:
            self._open_commit(g, labelled)
        return out

    def _detail(self, rec, slot: str, labelled: bool) -> None:
        if not rec.get("trace"):
            return
        try:
            with gzip.open(os.path.join(self.out_dir, rec["trace"]), "rt") as f:
                tr = json.load(f)
            rec["detail"] = pass_detail(tr, RANGE_PREFIX + slot, labelled, self.L)
            rec["work"] = work_counts(tr, RANGE_PREFIX + slot)
        except Exception as ex:                                        # evidence, not a gate input
            rec["detail"] = dict(error="%s: %s" % (type(ex).__name__, str(ex)[:200]))

    # -------------------------------------------------------------- the commit window (split: open .. close)
    def _commit_patches(self, stack, mk) -> None:
        r, T = self.r, self.T
        SRm = None
        for c in type(r).__mro__:
            if c.__name__ == "TentRunner":
                SRm = getattr(sys.modules[c.__module__], "SR", None)
        pairs = [(r, "_filter", "filter"), (r, "_plant_commit", "plant"), (r, "_fingerprints", "fp"),
                 (T, "counters0", "ctr"), (T, "commit", "commit"), (T, "set_twins", "twins")]
        if SRm is not None:
            pairs.insert(0, (SRm, "accept_prefix", "accept"))
        for obj, name, label in pairs:
            f0 = getattr(obj, name)

            def w(*a, _f0=f0, _label=label, **k):
                mk.mark(0, _label)
                return _f0(*a, **k)
            _patch(stack, obj, name, w)
        t0 = r._commit_tokens_t

        def tokens(*a, **k):
            mk.mark(0, "tokens")
            out = t0(*a, **k)
            mk.mark(0, "dprep")
            return out
        _patch(stack, r, "_commit_tokens_t", tokens)

    def _open_commit(self, g: int, labelled: bool) -> None:
        import torch
        try:
            from torch.profiler import ProfilerActivity, profile, record_function
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
            prof = profile(activities=acts)
            prof.__enter__()
        except Exception as ex:                                        # the profiler is evidence, not a gate
            self.box["passes"].append(dict(round=g, slot="commit", error="start: %s: %s" % (type(ex).__name__,
                                                                                          str(ex)[:200]),
                                           n_gpu_ops=0, busy_ms=None))
            return
        rf = record_function(RANGE_PREFIX + "commit")
        rf.__enter__()
        e0 = self.clock.event()
        self.clock.record(e0)
        stack, mk = [], (Marker() if labelled else None)
        if mk is not None:
            self._commit_patches(stack, mk)
            mk.mark(0, "decide")
        self._win = dict(g=g, prof=prof, rf=rf, e0=e0, mk=mk, stack=stack, labelled=labelled)

    def _close_commit(self, g_now, plan, why: str = "next pass") -> None:
        w, self._win = self._win, None
        e1 = self.clock.event()
        self.clock.record(e1)
        try:
            if w["mk"] is not None:
                w["mk"].close()
        finally:
            _unpatch(w["stack"])
            w["rf"].__exit__(None, None, None)
            self.clock.sync()
            w["prof"].__exit__(None, None, None)
        name = RANGE_PREFIX + "commit"
        rec = dict(round=w["g"], slot="commit", labelled=w["labelled"], wall_ms=round(float(self.clock.elapsed(
            w["e0"], e1)), 4), trace=None, error=None, closed_by=why, same_round=(g_now == w["g"]),
                   next_kind=(getattr(plan, "kind", None) if plan is not None else None))
        path = os.path.join(self.out_dir, "trace_%s_g%d_commit.json" % (self.arm, w["g"]))
        try:
            w["prof"].export_chrome_trace(path)
            with open(path, "rb") as f:
                raw = f.read()
            with gzip.open(path + ".gz", "wb") as gz:
                gz.write(raw)
            os.remove(path)
            rec["trace"] = os.path.basename(path) + ".gz"
            tr = json.loads(raw.decode("utf-8"))
            rec.update(pass_profile(tr, name))
            rec["detail"] = pass_detail(tr, name, w["labelled"], self.L)
            rec["work"] = work_counts(tr, name)
        except Exception as ex:
            rec.update(error="read: %s: %s" % (type(ex).__name__, str(ex)[:200]), n_gpu_ops=0, busy_ms=None)
        self.box["passes"].append(rec)

    # -------------------------------------------------------------- a counted joint pass (eager probe-round path)
    def _counted(self, model, plan, g: int):
        r, T = self.r, self.T
        rec = self.skips if (self.skips.round == g - 1 and self.skips.passes > 0) else None
        tap = GatherTap(r, rec)
        ur = UnionReader(T, r.vs, self.L)
        stack = []
        ur.install(stack)
        b0 = r.meter.blocks.clone()
        e0 = T.graph["eager"]
        tap.attach()
        try:
            out = self._run(model, plan)
        finally:
            tap.detach()
            _unpatch(stack)
        b1 = r.meter.blocks.clone()
        c = dict(round=g, tap=tap, union=ur, eager_layers=T.graph["eager"] - e0, recorded=rec is not None,
                 recorded_layers=(rec.layers if rec is not None else 0),
                 skips=(rec.n_skip.clone() if rec is not None else None),
                 skip_oor=(rec.oor.clone() if rec is not None else None))
        self._dev.append((c, "meter_new", b0, b1))
        self.box["counted"].append(c)
        return out

    # -------------------------------------------------------------- a recording one-row pass (steady round g)
    def _recording(self, model, plan, g: int, slot: str):
        why = self._draft_replay(plan)
        if why is not None:
            self.box["skipped"].append((g, slot, "record: " + why))
            return self._run(model, plan)
        T, rec = self.T, self.skips
        r0 = T._resid_lean
        stack = []
        st = dict(reset=(rec.round != g))

        def resid(l, eng, *a, **k):
            skip = r0(l, eng, *a, **k)
            if st["reset"]:
                rec.reset(g, skip)
                st["reset"] = False
            rec.record(l, eng._load_mask, skip)
            return skip
        _patch(stack, T, "_resid_lean", resid)
        try:
            out = self._run(model, plan)
        finally:
            _unpatch(stack)
        rec.passes += 1
        self.box["recorded"].append((g, slot))
        return out

    # -------------------------------------------------------------- after the run (events complete: after a sync)
    def evidence(self, elapsed) -> dict:
        for obj, key, b0, b1 in self._dev:
            obj[key] = int(b1 - b0)
        counted = []
        for c in self.box["counted"]:
            counted.append(dict(round=c["round"], tap=c["tap"].state(), union=c["union"].state(),
                                eager_layers=c["eager_layers"], recorded=c["recorded"],
                                recorded_layers=c["recorded_layers"], meter_new=c.get("meter_new"),
                                skips=(int(c["skips"]) if c["skips"] is not None else None),
                                skip_oor=(int(c["skip_oor"]) if c["skip_oor"] is not None else None)))
        j = [dict(I4C.pass_stats(m, elapsed, self.L), round=g) for g, _, m in self.box["joint"]]
        d = [dict(I4C.pass_stats(m, elapsed, self.L), round=g, slot=s) for g, s, m in self.box["draft"]]
        fin = {int(k): v for k, v in self.box["finished"].items()}
        p = dict(rounds=list(self.prof), label_rounds=list(self.lab), passes=list(self.box["passes"]),
                 done=list(self.box["done"]), skipped=list(self.box["skipped"]))
        return dict(joint=j, draft=d, profile=p, counted=counted, recorded=list(self.box["recorded"]),
                    count_rounds=list(self.cnt), recording_rounds=list(self.recr), finished=fin,
                    drain=drain_rounds(fin), keep=self.keep)
