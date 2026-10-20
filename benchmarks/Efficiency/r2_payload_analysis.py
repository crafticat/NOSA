"""R2 pilot payload analysis (CPU only; run inside a CPU job, never on the login node).

    python benchmarks/Efficiency/r2_payload_analysis.py --run <run dir> [--run <run dir> ...] --tau 0.4844 --out <dir>

Reads every spec_*.pt / decode_ref_*.pt of the run directories (spec_loop_pilot.py payloads, map_location cpu) and
writes <out>/analysis.json and <out>/analysis.md. For every spec cell (lockstep or per-request) against the shipped arm
with the same (L, batch, distinct, documents):

  FIRST MISMATCHES, per request: the index j of seq (seq[0] = t0; seq[j] is produced by feeding seq[j-1] at position
    p = L + j - 1), the shipped top-2 margin there and its class against 4 x tau (near tie / above / unknown), whether
    p % 64 == 63 (findings F4: the SHIPPED step at that position attends without its tail block), the round that
    committed seq[j] in the spec cell (its K, the request's n_acc and keep, the slot u in the verify, whether the
    verify crossed a block boundary for that request and whether u is after the fill), the smallest shipped margin
    before j.
  SAME-DOCUMENT CONSISTENCY: requests that share a document (payload 'docs') see the same prompt, so a correct engine
    gives them the same committed sequence and, round by round, the same n_acc. The shipped arm is the control. A
    group whose spec members differ, or whose n_acc differ in some round, carries request-dependent state: the
    signature of the slot-map defect fixed on 2026-10-20 (spec_loop._assign returned a non-contiguous map to the
    Triton gathers, so request b's prefetches landed in other slots / other requests' slots / slot 63).
  ABOVE-TOLERANCE first mismatches: for each, the facts that decide 'cascade or real': whether the request's state had
    already left its same-document partners before j (its n_acc or committed tokens differ from theirs in an earlier
    round: an earlier divergence of state, so the mismatch is a consequence, not a fresh numerical event), whether j
    is an F4 step, whether a partner with the same tokens up to j agreed with the shipped token at j.
  THE PROBES: the recorded probe dicts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                   # an older torch without weights_only
        return torch.load(path, map_location="cpu")


def load_payloads(dirs):
    out = {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith(".pt") and (f.startswith("spec_") or f.startswith("decode_ref_")):
                out["%s/%s" % (os.path.basename(os.path.normpath(d)), f[:-3])] = load(os.path.join(d, f))
    return out


def key_of(p):
    return (int(p["L"]), int(p["batch"]), int(p.get("distinct_docs", p["batch"])), tuple(p["docs"]))


def classify(margin, tol):
    if margin is None:
        return "unknown"
    return "near_tie" if margin <= tol else "above_tol"


def first_mismatches(spec_seq, ref_seq, margins, tol):
    """Per request: None, or dict(j, spec_token, ref_token, margin, cls, min_margin_before)."""
    n = min(spec_seq.shape[1], ref_seq.shape[1])
    out = []
    for r in range(spec_seq.shape[0]):
        a, b = spec_seq[r, :n].to(torch.int64), ref_seq[r, :n].to(torch.int64)
        ne = (a != b).nonzero().flatten()
        if ne.numel() == 0:
            out.append(None)
            continue
        j = int(ne[0])
        m = float(margins[r, j - 1]) if (margins is not None and j >= 1 and j - 1 < margins.shape[1]) else None
        mb = float(margins[r, :j - 1].min()) if (margins is not None and j >= 2) else None
        out.append(dict(j=j, spec_token=int(a[j]), ref_token=int(b[j]), margin=m, cls=classify(m, tol), min_margin_before=mb))
    return out


def token_rounds(cell, B):
    """seq index j -> (round record index, u, keep, K, n_acc, committed_before) per request, for every committed token.
    Per-request rounds: committed_before / keep per request; lockstep rounds: committed_before int, c; re-warm rounds
    commit one token with u = 0, K = 0."""
    perreq = cell.get("commit") == "perreq"
    m = [dict() for _ in range(B)]
    for i, r in enumerate(cell["rounds"]):
        for b in range(B):
            if perreq:
                n, k = int(r["committed_before"][b]), int(r["keep"][b])
                K, na = int(r["K"]), int(r["n_acc"][b])
            elif r.get("rewarm"):
                n, k, K, na = int(r["committed_before"]), 1, 0, 0
            else:
                n, k, K, na = int(r["committed_before"]), int(r["c"]), int(r["K"]), int(r["n_acc"][b])
            for u in range(k):
                m[b][n + u + 1] = dict(round=i, u=u, keep=k, K=K, n_acc=na, committed_before=n)
    return m


def doc_groups(docs):
    g = defaultdict(list)
    for b, d in enumerate(docs):
        g[d].append(b)
    return [v for v in g.values() if len(v) > 1]


def group_consistency(seq, groups):
    """Per group: the members whose sequence differs from the group's first member, and the first such index."""
    out = []
    for g in groups:
        b0 = g[0]
        diff = []
        for b in g[1:]:
            ne = (seq[b] != seq[b0]).nonzero().flatten()
            if ne.numel():
                diff.append((b, int(ne[0])))
        out.append(dict(members=len(g), differing=len(diff), first_index=min([x[1] for x in diff]) if diff else None))
    return out


def nacc_disagreement(cell, groups):
    """(group, round) pairs in which the active members of one document group report different n_acc. Returns the
    count and, per request, the index of the first round in which its group disagreed."""
    perreq = cell.get("commit") == "perreq"
    count = 0
    first = {}
    for i, r in enumerate(cell["rounds"]):
        if r.get("rewarm") or "n_acc" not in r:
            continue
        for g in groups:
            act = [b for b in g if not (perreq and r["done_before"][b])]
            vals = set(int(r["n_acc"][b]) for b in act)
            if len(vals) > 1:
                count += 1
                for b in act:
                    first.setdefault(b, i)
    return count, first


def crossing(L, rinfo, U_of_round):
    """Whether the verify of that round crossed a block boundary for the request and whether slot u is after the fill."""
    p0 = L + rinfo["committed_before"]                      # the position of x_n, fed first in the verify
    tl0 = p0 % 64
    U = U_of_round
    rolled = tl0 + U >= 64
    f = 63 - tl0
    return dict(round_start_pos=p0, tail_len0=tl0, rolled=bool(rolled), u_after_fill=bool(rolled and rinfo["u"] > f))


def analyse_cell(name, cname, p, cell, ref, tol):
    B = int(p["batch"])
    L = int(p["L"])
    fm = first_mismatches(cell["seq"], ref["seq"], ref.get("margins"), tol)
    tr = token_rounds(cell, B)
    groups = doc_groups(list(p["docs"]))
    cons_spec = group_consistency(cell["seq"], groups)
    cons_ref = group_consistency(ref["seq"], groups)
    ndis, nfirst = nacc_disagreement(cell, groups)
    rows = []
    for b, m in enumerate(fm):
        if m is None:
            continue
        j = m["j"]
        pos = L + j - 1
        ri = tr[b].get(j)
        row = dict(request=b, doc=p["docs"][b], **m, position=pos, f4_step=(pos % 64 == 63))
        if ri is not None:
            rr = cell["rounds"][ri["round"]]
            U = ri["K"] + 1
            row.update(round=ri["round"], u=ri["u"], K=ri["K"], keep=ri["keep"], n_acc=ri["n_acc"], bonus=ri["u"] == ri["keep"] - 1)
            row.update(crossing(L, ri, U))
            row["round_is_rewarm"] = bool(rr.get("rewarm"))
        row["state_left_partners_before_j"] = bool(b in nfirst and ri is not None and nfirst[b] < ri["round"])
        row["first_partner_disagreement_round"] = nfirst.get(b)
        partners = [x for g in groups if b in g for x in g if x != b]
        same_prefix_agree = [x for x in partners if bool((cell["seq"][x, :j] == cell["seq"][b, :j]).all())
                             and int(cell["seq"][x, j]) == int(ref["seq"][b, j])]
        row["partners_same_prefix_and_shipped_token_at_j"] = same_prefix_agree[:8]
        rows.append(row)
    cls = defaultdict(int)
    for r in rows:
        cls[r["cls"]] += 1
    return dict(payload=name, cell=cname, commit=cell.get("commit", "lockstep"), K=cell.get("K"), batch=B, L=L,
                mismatching_requests=len(rows), classes=dict(cls),
                f4_first_mismatches=sum(1 for r in rows if r["f4_step"]),
                same_doc_groups=len(groups), spec_groups_with_differing_members=sum(1 for c in cons_spec if c["differing"]),
                ref_groups_with_differing_members=sum(1 for c in cons_ref if c["differing"]),
                nacc_disagreement_rounds=ndis, above_tol=[r for r in rows if r["cls"] == "above_tol"],
                first_mismatch_index_hist=dict(sorted(((r["j"], sum(1 for x in rows if x["j"] == r["j"])) for r in rows))),
                rows=rows, probe=cell.get("probe"))


def analyse(payloads, tau):
    tol = 4.0 * tau
    shipped = {}
    for name, p in payloads.items():
        if "shipped" in p.get("arms", {}):
            shipped[key_of(p)] = (name, p["arms"]["shipped"])
    cells = []
    for name, p in payloads.items():
        # the payload's own shipped arm first (the report keys by (L, batch, distinct, docs) only)
        ref = (name, p["arms"]["shipped"]) if "shipped" in p.get("arms", {}) else shipped.get(key_of(p))
        for cname, c in p.get("cells", {}).items():
            if ref is None:
                cells.append(dict(payload=name, cell=cname, error="no shipped arm with the same key"))
                continue
            cells.append(dict(analyse_cell(name, cname, p, c, ref[1], tol), reference=ref[0]))
    return dict(tau=tau, tol=tol, cells=cells)


def to_md(res):
    lines = ["# R2 payload analysis (tol = 4 x %.4f = %.4f)\n" % (res["tau"], res["tol"])]
    lines.append("| payload / cell | commit | K | mismatching requests | classes | F4-step first mismatches | spec same-doc groups differing (shipped) | n_acc disagreement rounds | probe |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in res["cells"]:
        if "error" in c:
            lines.append("| %s/%s | - | - | %s | | | | | |" % (c["payload"], c["cell"], c["error"]))
            continue
        pr = c.get("probe")
        lines.append("| %s/%s | %s | %s | %d | %s | %d | %d of %d (%d) | %d | %s |" % (
            c["payload"], c["cell"], c["commit"], c["K"], c["mismatching_requests"], c["classes"], c["f4_first_mismatches"],
            c["spec_groups_with_differing_members"], c["same_doc_groups"], c["ref_groups_with_differing_members"],
            c["nacc_disagreement_rounds"], "-" if pr is None else "passed=%s same=%s det=%s (%s)" % (
                pr.get("passed"), pr.get("same"), pr.get("replay_deterministic"), pr.get("detail"))))
    for c in res["cells"]:
        for r in c.get("above_tol", []):
            lines.append("\nABOVE TOL %s/%s request %d: j=%d position %d (F4 step %s), margin %.4f (min before %s), round %s u=%s K=%s keep=%s "
                         "n_acc=%s bonus=%s rolled=%s u_after_fill=%s; state left its partners before j: %s (first group "
                         "n_acc disagreement in round %s); partners with the same prefix and the shipped token at j: %s"
                         % (c["payload"], c["cell"], r["request"], r["j"], r["position"], r["f4_step"], r["margin"], r["min_margin_before"],
                            r.get("round"), r.get("u"), r.get("K"), r.get("keep"), r.get("n_acc"), r.get("bonus"), r.get("rolled"),
                            r.get("u_after_fill"), r["state_left_partners_before_j"], r["first_partner_disagreement_round"],
                            r["partners_same_prefix_and_shipped_token_at_j"]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = analyse(load_payloads(a.run), a.tau)
    os.makedirs(a.out, exist_ok=False)
    with open(os.path.join(a.out, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    md = to_md(res)
    with open(os.path.join(a.out, "analysis.md"), "w") as f:
        f.write(md)
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
