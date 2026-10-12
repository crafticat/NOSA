"""Stage A of the per-layer cache allocation study (accuracy only, NOSA-8B on NOSI).

WHAT IT MEASURES. At the same restored exact prefix, each ARM drafts K tokens with
its own causally warmed virtual cache (its own per-layer capacities, its own
LRU / in-flight / protection state), recomputing every downstream hidden state
and NOSA selection. The exact target p is computed at full availability. Rows
carry the full-vocabulary overlap sum_v min(p_v, q_v) = 1 - TV, the sampled
acceptance, the accepted-prefix length and the arm's memory metadata.

WHAT IT REUSES, UNCHANGED. benchmarks/Efficiency/acceptance_pilot.py (Pilot:
model, prefill, snapshot, step, dist, emit and its identity / NaN / overlap
checks), nosi/avail_policy.py (VirtualCache, mask mechanism; the per-layer
extension is opt-in and inert for archived points), nosi/state_snapshot.py and
retro_eval/nosi_accept_math.py + retro_eval/accept_alloc.py (loaded by path from
NOSI_REPO). The gates G0..G8 run in the archived driver in the same job.

LABELS. Retention is an OLDER-POLICY DIAGNOSTIC (pilot VirtualCache plus
rollout-local protection emulation), not the M1-tested reference. Arrival
models are accuracy assumptions, not live throughput. The committed prefix is
the document's own teacher-forced continuation. Verification is sequential.
The one-group arms differ in base slots on purpose (unequal memory).

Spec: retroinfer-eval docs/superpowers/specs/2026-10-12-accept-alloc-stage-a-prereg.md

ENVIRONMENT: NOSI_REPO, NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_AVAIL (must be
set, mech=mask), NOSI_ALLOC_OUT, NOSI_ALLOC_ROWS ("0,1"), NOSI_ALLOC_TEXT_SHA
("0:<sha>;1:<sha>"), NOSI_ALLOC_SITES ("26,34,42,50"), NOSI_ALLOC_K ("3"),
NOSI_ALLOC_DIGEST (the registered stage_a_digest), NOSI_ALLOC_DEADLINE_S,
NOSI_BENCH_L (16128).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.environ.get("NOSI_REPO", "/mnt/central/users/ojerbi/retroinfer-eval")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


aa = _load(os.path.join(REPO, "retro_eval", "accept_alloc.py"), "accept_alloc")

# The archived pilot reads its knobs at import. Give it a valid one-point grid
# and rollout mode; every point it registers is cleared below before any decode.
os.environ.setdefault("NOSI_ACC_POINTS", "C=81,d=1")
os.environ["NOSI_ACC_MODE"] = "rollout"
os.environ.pop("NOSI_ACC_NPOINTS", None)
ap = _load(os.path.join(HERE, "acceptance_pilot.py"), "acceptance_pilot_archived")
am = ap.am

OUT = os.environ.get("NOSI_ALLOC_OUT", "accept_alloc_out")
ROWS = [int(x) for x in os.environ.get("NOSI_ALLOC_ROWS", "0,1").split(",") if x]
SITES = [int(x) for x in os.environ.get("NOSI_ALLOC_SITES", "26,34,42,50").split(",") if x]
K = int(os.environ.get("NOSI_ALLOC_K", "3"))
DEADLINE_S = float(os.environ.get("NOSI_ALLOC_DEADLINE_S", "1200"))
PROTECT = int(os.environ.get("NOSI_ALLOC_PROTECT", "1"))

RC_ROWS = 240
RC_TWIN = 241
RC_LOCALITY = 242
RC_PREUSE = 243
RC_DEADLINE = 244
RC_DIGEST = 239


def text_sha_registry() -> dict:
    out = {}
    for part in os.environ.get("NOSI_ALLOC_TEXT_SHA", "").split(";"):
        part = part.strip()
        if part:
            r, s = part.split(":", 1)
            out[int(r)] = s.strip()
    return out


def load_rows(tokenizer, rows, need):
    """Explicit parquet rows, never ordinals. A7: the full-text sha256 of every
    row must equal the registered value, and the tokenized prefix is hashed."""
    import pandas as pd
    pq = os.environ["NOSI_PG19_PARQUET"]
    with open(pq, "rb") as fh:
        pq_sha = hashlib.sha256(fh.read()).hexdigest()
    texts = pd.read_parquet(pq, columns=["text"])["text"]
    reg = text_sha_registry()
    want = ap.L + need
    ids, ident = [], []
    for r in rows:
        t = texts.iloc[r]
        tsha = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if r not in reg or reg[r] != tsha:
            ap.die(RC_ROWS, "A7: row %d full-text sha256 %s != registered %s"
                   % (r, tsha, reg.get(r)))
        tok = tokenizer(t, return_tensors="pt").input_ids[0]
        if tok.shape[0] < want:
            ap.die(RC_ROWS, "A7: row %d has %d tokens < %d" % (r, tok.shape[0], want))
        pref = tok[:want].to(torch.int64).contiguous()
        ids.append(pref)
        ident.append(dict(row=r, text_sha256=tsha, n_tokens=int(tok.shape[0]),
                          prefix_tokens=want,
                          prefix_ids_sha256=hashlib.sha256(pref.numpy().tobytes()).hexdigest()))
    return torch.stack(ids), ident, pq_sha


def to_pilot_point(pid, cfg):
    """The point dict the archived Pilot.emit reads, plus the arm metadata."""
    d = float(cfg["d"])
    pt = dict(id=pid, spec=cfg["arm"], C=aa.scalar_C(cfg), d=d, f=0.0,
              ablate="none", ready=1.0)
    pt["natural"] = d != 0 and cfg["deny"] is None
    pt["omission"] = ("identity_anchor_shipped_engine" if d == 0 else
                      "positive_control_deny_group" if cfg["deny"] is not None else
                      "natural_lru_per_layer")
    pt["sim_mode"] = ap.sim_mode_of(d)
    pt["cfg"] = cfg
    pt["meta"] = aa.arm_meta(cfg)
    return pt


class Run:
    def __init__(self):
        self.pilot = ap.Pilot()
        self.pol = self.pilot.policy
        if self.pol is None or self.pol.mech != "mask":
            ap.die(RC_DIGEST, "NOSI_AVAIL must be set with mech=mask")
        digest = aa.stage_a_digest(PROTECT)
        if os.environ.get("NOSI_ALLOC_DIGEST") != digest:
            ap.die(RC_DIGEST, "arm list digest %s != registered NOSI_ALLOC_DIGEST %s"
                   % (digest, os.environ.get("NOSI_ALLOC_DIGEST")))
        self.digest = digest
        self.points = [to_pilot_point(i, c) for i, c in enumerate(aa.stage_a_points(PROTECT))]
        # REGISTER EVERY ARM BEFORE THE FIRST DECODE STEP, each with its own
        # capacities: avail_policy then warms each arm's own virtual caches on
        # the shared verified trajectory. No arm is a copy of another.
        self.pol.clear_points()
        for pt in self.points:
            c = pt["cfg"]
            self.pol.register_point(pt["id"], aa.scalar_C(c), c["d"], 0.0, "none", 1.0,
                                    layer_caps=c["Cl"], deny_layers=aa.deny_layers(c),
                                    protect=bool(c["protect"]))
        self.by_arm = {pt["spec"]: pt for pt in self.points}
        self.pcache = {}
        self.n_target_runs = 0
        self.n_p_reused = 0
        self.n_p_audited = 0
        os.makedirs(OUT, exist_ok=True)
        self.rows_fh = open(os.path.join(OUT, "rows.jsonl"), "a")
        self.sel_fh = open(os.path.join(OUT, "selections.jsonl"), "a")
        self.t0 = time.time()

    # -- one arm at one site ---------------------------------------------------
    def rollout(self, doc, site, first, pt, audit):
        pil, pol = self.pilot, self.pol
        pil.snap.take()
        pos0 = pil.pos.clone()
        pol.set_site(site)
        pol.begin_rollout(pt["id"])
        pol.sel_log = {}
        pol.prot_log = []
        rec0 = len(pol.records)
        calls = []

        def draft_step(j, prev):
            tok = first if prev is None else torch.tensor([[prev]], dtype=torch.long, device="cuda")
            lg = pil.step(tok, pil.ap.ROLE_DRAFT)
            calls.append(pol.call)
            if not bool(torch.isfinite(lg).all()):
                ap.die(ap.RC_NAN, "draft NaN/Inf doc=%d site=%d j=%d arm=%s"
                       % (doc, site, j, pt["spec"]))
            return pil.dist(lg[0])

        qs, xs = am.draft_rollout(draft_step, K, doc, site)
        recs = pol.records[rec0:]
        sel = {(calls.index(c) + 1, l, h): v for (c, l, h, b), v in pol.sel_log.items()}
        prot = list(pol.prot_log)
        pol.sel_log = None
        pil.snap.restore()
        pil.pos = pos0.clone()
        pol.end_rollout()

        # p REUSE ONLY ON AN IDENTICAL (doc, site, path) KEY
        keys = [aa.p_key(doc, site, xs[:j]) for j in range(K)]
        need = audit or any(k not in self.pcache for k in keys)
        if need:
            def target_step(j, prev):
                tok = first if prev is None else torch.tensor([[prev]], dtype=torch.long, device="cuda")
                return pil.dist(pil.step(tok, pil.ap.ROLE_TARGET)[0])
            ps = am.verify_sequential(target_step, xs)
            pil.snap.restore()
            pil.pos = pos0.clone()
            self.n_target_runs += 1
            for k, p in zip(keys, ps):
                if k in self.pcache:
                    self.n_p_audited += 1
                    if not np.array_equal(self.pcache[k], p):
                        ap.die(RC_PREUSE, "A5: recomputed p differs from the cached p at key %r"
                               % (k,))
                else:
                    self.pcache[k] = p
        else:
            ps = [self.pcache[k] for k in keys]
            self.n_p_reused += 1
        return qs, ps, xs, recs, sel, prot

    # -- gates A1 / A2 at one site ----------------------------------------------
    def check_site(self, doc, site, res):
        base, twin = res["U81_d1"], res["U81v_d1"]
        for j in range(K):
            if not np.array_equal(base["qs"][j], twin["qs"][j]):
                ap.die(RC_TWIN, "A1: U81v q != U81 q at doc=%d site=%d j=%d" % (doc, site, j + 1))
        rb = [r[7:14] for r in base["recs"]]
        rt = [r[7:14] for r in twin["recs"]]
        if rb != rt:
            ap.die(RC_TWIN, "A1: U81v availability records != U81 at doc=%d site=%d" % (doc, site))
        if base["sel"] != twin["sel"]:
            ap.die(RC_TWIN, "A1: U81v selection log != U81 at doc=%d site=%d" % (doc, site))
        for arm, r in res.items():
            m = aa.arm_meta(self.by_arm[arm]["cfg"])
            if m["group"] is None or not arm.endswith("_d1"):
                continue
            lo = 4 * m["group"]
            hi = lo + aa.GROUP_SIZE
            for (j, l, h), (s, dn) in r["sel"].items():
                if j != 1:
                    continue
                bs, bd = base["sel"][(j, l, h)]
                if l < lo and (s, dn) != (bs, bd):
                    ap.die(RC_LOCALITY, "A2: arm %s changed layer %d (< group start %d) at j=1 "
                           "doc=%d site=%d" % (arm, l, lo, doc, site))
                if l >= hi and s == bs and dn != bd:
                    ap.die(RC_LOCALITY, "A2: arm %s denied a different set at layer %d with "
                           "the SAME selection as U81 at j=1 doc=%d site=%d" % (arm, l, doc, site))

    def emit(self, doc, row, site, pt, r, base_sel):
        pil = self.pilot
        n0 = len(pil.rows)
        av = pil.avail_of(r["recs"])
        pil.emit(doc, row, site, K, pt, r["qs"], r["ps"], r["xs"], av)
        acc = [row_["accept_draw"] for row_ in pil.rows[n0:]]
        prefix = am.prefix_length(acc)
        per_layer = {}
        for rec in r["recs"]:
            per_layer.setdefault(rec[4], [0, 0])
            per_layer[rec[4]][0] += rec[11]
            per_layer[rec[4]][1] += rec[7]
        changed = {}
        for (j, l, h), (s, _) in r["sel"].items():
            if base_sel is not None and (j, l, h) in base_sel and base_sel[(j, l, h)][0] != s:
                changed.setdefault(j, 0)
                changed[j] += 1
        prot_only = [x[2] for x in r["prot"]]
        for row_ in pil.rows[n0:]:
            row_.update(pt["meta"])
            row_.update(site=site, arm_digest=self.digest, accepted_prefix=prefix,
                        caps_digest=self.pol.caps_digest(pt["cfg"]["Cl"]),
                        sel_changed_vs_U81=changed.get(row_["j"], 0),
                        protected_only_peak=max(prot_only) if prot_only else 0,
                        protected_only_mean=float(np.mean(prot_only)) if prot_only else 0.0,
                        ready_per_layer_all_j={int(l): v[0] / v[1] for l, v in per_layer.items() if v[1]},
                        committed_prefix="teacher_forced_document",
                        arrival_label="accuracy assumption, not live throughput")
            self.rows_fh.write(json.dumps(row_) + "\n")
        self.rows_fh.flush()
        for (j, l, h), (s, dn) in sorted(r["sel"].items()):
            self.sel_fh.write(json.dumps([doc, site, pt["spec"], j, l, h, s, dn]) + "\n")
        self.sel_fh.flush()
        del pil.rows[n0:]
        pil.vectors.clear()


def main() -> int:
    run = Run()
    pil = run.pilot
    smax = max(SITES)
    ap.budget_or_die(smax + 1, K)
    need = smax + K + 2
    ids, ident, pq_sha = load_rows(pil.model.tokenizer, ROWS, need)
    ids = ids.to("cuda")
    meta = dict(study="accept-alloc Stage A", rows=ident, parquet_sha256=pq_sha, sites=SITES,
                K=K, L=ap.L, arm_digest=run.digest, arms=[p["spec"] for p in run.points],
                terms=aa.stage_a_terms(PROTECT), sampling=dict(temperature=ap.TEMP, top_p=ap.TOPP,
                                                                top_k=ap.TOPK),
                retention_label=aa.arm_meta(run.points[1]["cfg"])["retention_label"],
                complete=False)
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    status = "COMPLETE"
    rc = 0
    for d, row in enumerate(ROWS):
        prompt, forced = ids[d:d + 1, :ap.L], ids[d:d + 1, ap.L:]
        run.pol.new_document(d)
        pil.open_document(prompt)
        pil.step(forced[:, 0:1], pil.ap.ROLE_WARM)
        trans = pil.ss.transient_ids(pil.model)
        t = 1
        for si, site in enumerate(SITES):
            while t < site:
                pil.step(forced[:, t:t + 1], pil.ap.ROLE_WARM)
                t += 1
            if time.time() - run.t0 > DEADLINE_S:
                status, rc = "INCOMPLETE_DEADLINE", RC_DEADLINE
                break
            first = forced[:, site:site + 1]
            res = {}
            for pt in run.points:
                qs, ps, xs, recs, sel, prot = run.rollout(d, site, first, pt, audit=(si == 0))
                res[pt["spec"]] = dict(qs=qs, ps=ps, xs=xs, recs=recs, sel=sel, prot=prot)
            run.check_site(d, site, res)
            base_sel = res["U81_d1"]["sel"]
            for pt in run.points:
                run.emit(d, row, site, pt, res[pt["spec"]],
                         res["U81_dinf"]["sel"] if pt["spec"].endswith("_dinf") else base_sel)
        if rc:
            break
        pil.ss.assert_transients_intact(pil.model, trans)
        if pil.host_digest_now() != pil.host_digest:
            ap.die(ap.RC_HOST, "G8: host K/V changed during document %d" % d)
    meta.update(complete=(rc == 0), status=status, n_target_runs=run.n_target_runs,
                n_p_reused=run.n_p_reused, n_p_audited=run.n_p_audited,
                wall_s=time.time() - run.t0)
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print("[accept-alloc] %s rc=%d target_runs=%d p_reused=%d p_audited=%d -> %s"
          % (status, rc, run.n_target_runs, run.n_p_reused, run.n_p_audited, OUT), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
