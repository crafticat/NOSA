"""Layer-sensitivity diagnostic (accuracy only, NOSA-8B on NOSI, B = 1, K = 1).

At each restored exact prefix (PG-19 rows 0-3, 16 sites), the exact target p is
computed once at full availability, then each of the 86 registered arms drafts
ONE token from the same restored state with its own causally warmed virtual
cache, recomputing every hidden state and selection. Per arm and round we keep
the full fp32 q (and p), TV / overlap / sampled acceptance, the per-layer denial
counts, and WHAT mask_bias HANDED TO ATTENTION at every layer (scratch or not,
MASK_BIAS entry count, implied count, byte digest).

Instrument checks, fail-fast (exit codes in the sbatch header):
  C0  bias delivery: per draft call and layer, MASK_BIAS entries == denied
      slots x 64, and a denying layer hands attention the scratch tensor;
  C1  path: every full-denial arm A_l changes the logits bitwise versus U81;
  C2  consumption (first site of each document, every layer): A_l plus garbage
      written into the denied K/V rows of layer l after the mask equals A_l
      bitwise (denied rows are not read), and U81 plus garbage in ONE visible
      slot of layer l differs from U81 (the poison is detectable). Method and
      poison values from benchmarks/Efficiency/mask_check.py (fork branch
      feature/nosi-maskcheck @35eb117, job 2175726), reused, not re-derived;
  C3  far-only rule: every far-denial stream keeps exactly 16 near blocks;
  plus the U81v twin (bitwise), the D0 identity (q == p bitwise), frozen rows by
  full-text sha256 and the arm digest. The archived gates G0..G8 run first in
  the same job (sbatch).

Labels: older-policy availability model (pilot VirtualCache, d=1, no
protection; K = 1 so protection would be inert); arrival is an accuracy
assumption; committed prefix is the document's teacher-forced continuation;
denial and capacity arms are unequal-memory diagnostics.

Spec: retroinfer-eval docs/superpowers/specs/2026-10-19-layer-sens-protocol.md
ENV: NOSI_REPO, NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_AVAIL (mech=mask...),
NOSI_LS_OUT, NOSI_LS_DIGEST, NOSI_LS_DEADLINE_S, NOSI_BENCH_L (16128).
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


ls = _load(os.path.join(REPO, "retro_eval", "layer_sens.py"), "layer_sens")
os.environ["NOSI_ACC_POINTS"] = "C=81,d=1"      # cleared below before any decode
os.environ["NOSI_ACC_MODE"] = "rollout"
os.environ["NOSI_ACC_K"] = "1"
os.environ.pop("NOSI_ACC_NPOINTS", None)
ap = _load(os.path.join(HERE, "acceptance_pilot.py"), "acceptance_pilot_archived")
am = ap.am

OUT = os.environ.get("NOSI_LS_OUT", "layer_sens_out")
DEADLINE_S = float(os.environ.get("NOSI_LS_DEADLINE_S", "1200"))
K_POISON, V_POISON = 64.0, 1.0e4                 # mask_check.py values

RC_DIGEST, RC_ROWS, RC_C0, RC_C1, RC_C2, RC_C3, RC_TWIN, RC_DEADLINE, RC_D0 = (
    230, 231, 232, 233, 234, 235, 236, 237, 238)


def load_rows(tokenizer, need):
    import pandas as pd
    pq = os.environ["NOSI_PG19_PARQUET"]
    with open(pq, "rb") as fh:
        pq_sha = hashlib.sha256(fh.read()).hexdigest()
    texts = pd.read_parquet(pq, columns=["text"])["text"]
    want = ap.L + need
    ids, ident = [], []
    for r in ls.ROWS:
        t = texts.iloc[r]
        tsha = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if tsha != ls.TEXT_SHA[r]:
            ap.die(RC_ROWS, "row %d full-text sha256 %s != registered %s" % (r, tsha, ls.TEXT_SHA[r]))
        tok = tokenizer(t, return_tensors="pt").input_ids[0]
        if tok.shape[0] < want:
            ap.die(RC_ROWS, "row %d has %d tokens < %d" % (r, tok.shape[0], want))
        pref = tok[:want].to(torch.int64).contiguous()
        ids.append(pref)
        ident.append(dict(row=r, text_sha256=tsha, n_tokens=int(tok.shape[0]), prefix_tokens=want,
                          prefix_ids_sha256=hashlib.sha256(pref.numpy().tobytes()).hexdigest()))
    return torch.stack(ids), ident, pq_sha


class Run:
    def __init__(self):
        self.pilot = ap.Pilot()
        self.pol = self.pilot.policy
        if self.pol is None or self.pol.mech != "mask":
            ap.die(RC_DIGEST, "NOSI_AVAIL must be set with mech=mask")
        if os.environ.get("NOSI_LS_DIGEST") != ls.digest():
            ap.die(RC_DIGEST, "arm digest %s != registered NOSI_LS_DIGEST %r"
                   % (ls.digest(), os.environ.get("NOSI_LS_DIGEST")))
        self.cfgs = ls.points()
        self.pol.clear_points()
        for pid, c in enumerate(self.cfgs):
            kw = ls.register_kwargs(c)
            self.pol.register_point(pid, kw["capacity"], kw["delay"], 0.0, "none", 1.0,
                                    layer_caps=kw["layer_caps"], deny_layers=kw["deny_layers"],
                                    deny_mode=kw["deny_mode"])
        self.pid = {c["arm"]: i for i, c in enumerate(self.cfgs)}
        # THE ATTENTION-SIDE HOOK: the logged mask_bias, then (only in a C2
        # consumption arm) garbage written into the layer's live window AFTER the
        # mask and BEFORE attention, exactly where mask_check.py put it.
        self.poison = None                       # (layer, "denied" | "visible", seed)
        orig = self.pol.mask_bias

        def hooked(kv_bias):
            out = orig(kv_bias)
            if self.poison is not None and self.pol.role == self.pilot.ap.ROLE_DRAFT \
                    and self.pol.layer == self.poison[0]:
                self._poison(self.poison)
            return out
        self.pol.mask_bias = hooked
        if self.pilot.ap.POLICY is not self.pol:
            ap.die(RC_C2, "the model does not call this policy instance")
        os.makedirs(os.path.join(OUT, "vectors"), exist_ok=True)
        self.rows_fh = open(os.path.join(OUT, "rows.jsonl"), "a")
        self.cons_fh = open(os.path.join(OUT, "consumption.jsonl"), "a")
        self.bias_fh = open(os.path.join(OUT, "bias.jsonl"), "a")
        self.t0 = time.time()

    def _engine(self, layer):
        return self.pilot.cache.layers[layer].cache_engine

    def _poison_rows(self, eng, b, h, r0, r1, gen):
        n = r1 - r0
        if n <= 0:
            return
        ks = torch.randint(0, 2, (n, eng._k_gpu.shape[-1]), generator=gen).to(eng._k_gpu.device) * 2 - 1
        vs = torch.randint(0, 2, (n, eng._v_gpu.shape[-1]), generator=gen).to(eng._v_gpu.device) * 2 - 1
        eng._k_gpu[b, r0:r1, h, :] = (ks * K_POISON).to(eng._k_gpu.dtype)
        eng._v_gpu[b, r0:r1, h, :] = (vs * V_POISON).to(eng._v_gpu.dtype)

    def _poison(self, spec):
        layer, kind, seed = spec
        eng = self._engine(layer)
        H, B, M = eng._block_map.shape
        bs = eng.block_size
        deny = self.pol._deny_dev.cpu() if self.pol._deny_dev is not None else torch.zeros((H, B, M), dtype=torch.bool)
        gen = torch.Generator().manual_seed(seed * 7919 + layer)
        for b in range(B):
            if kind == "denied":
                for h in range(H):
                    for m in range(M - 1):
                        if bool(deny[h, b, m]):
                            self._poison_rows(eng, b, h, m * bs, (m + 1) * bs, gen)
                    self._poison_rows(eng, b, h, int(eng._cache_lens[b]), M * bs, gen)
            else:
                ok = [m for m in range(M - 1) if not bool(deny[0, b, m]) and int(eng._block_map[0, b, m]) >= 0]
                if not ok:
                    ap.die(RC_C2, "no visible slot to poison at layer %d" % layer)
                m = ok[len(ok) // 2]
                self._poison_rows(eng, b, 0, m * bs, (m + 1) * bs, gen)

    def draft(self, arm, first, poison=None):
        """One draft step of `arm` from the snapshot; returns logits, records,
        bias log, near log. The caller restores."""
        pol = self.pol
        pol.begin_rollout(self.pid[arm])
        pol.bias_log, pol.near_log = [], []
        rec0 = len(pol.records)
        self.poison = poison
        try:
            lg = self.pilot.step(first, self.pilot.ap.ROLE_DRAFT).clone()
        finally:
            self.poison = None
        recs, blog, nlog = pol.records[rec0:], pol.bias_log, pol.near_log
        pol.bias_log, pol.near_log = None, []
        pol.end_rollout()
        return lg, recs, blog, nlog


def check_c0(arm, blog):
    """C0 on the tensor mask_bias handed to attention, every layer of the call."""
    for (_call, layer, role, is_scratch, n_region, denied_ok, rest_ok, _sha) in blog:
        if role != "draft":
            continue
        if not (denied_ok and rest_ok) or (n_region > 0 and not is_scratch):
            ap.die(RC_C0, "C0: arm %s layer %d: %d denied entries, all masked=%d, rest unchanged=%d, "
                   "scratch=%d" % (arm, layer, n_region, denied_ok, rest_ok, is_scratch))


def main() -> int:
    run = Run()
    pil, pol = run.pilot, run.pol
    smax = max(ls.SITES)
    ap.budget_or_die(smax + 1, ls.K)
    ids, ident, pq_sha = load_rows(pil.model.tokenizer, smax + ls.K + 2)
    ids = ids.to("cuda")
    meta = dict(study="layer-sensitivity diagnostic", rows=ident, parquet_sha256=pq_sha,
                sites=list(ls.SITES), K=ls.K, L=ap.L, arm_digest=ls.digest(), arms=ls.arms(),
                terms=ls.terms(), sampling=dict(temperature=ap.TEMP, top_p=ap.TOPP, top_k=ap.TOPK),
                vectors="vectors/p_d{doc}_s{site}.npy (V,) and q_d{doc}_s{site}.npy (86, V), fp32, arm order = arms",
                complete=False, status="RUNNING")
    mpath = os.path.join(OUT, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    status, rc = "COMPLETE", 0
    arms = ls.arms()
    for d, row in enumerate(ls.ROWS):
        prompt, forced = ids[d:d + 1, :ap.L], ids[d:d + 1, ap.L:]
        pol.new_document(d)
        pil.open_document(prompt)
        pil.step(forced[:, 0:1], pil.ap.ROLE_WARM)
        trans = pil.ss.transient_ids(pil.model)
        t = 1
        for si, site in enumerate(ls.SITES):
            while t < site:
                pil.step(forced[:, t:t + 1], pil.ap.ROLE_WARM)
                t += 1
            if time.time() - run.t0 > DEADLINE_S:
                status, rc = "INCOMPLETE_DEADLINE", RC_DEADLINE
                break
            first = forced[:, site:site + 1]
            pil.snap.take()
            pos0 = pil.pos.clone()

            def restore():
                pil.snap.restore()
                pil.pos = pos0.clone()

            pol.set_site(site)
            lg_T = pil.step(first, pil.ap.ROLE_TARGET).clone()
            restore()
            p = pil.dist(lg_T[0])
            Q = np.empty((len(arms), p.size), dtype=np.float32)
            logits = {}
            for i, arm in enumerate(arms):
                lg, recs, blog, nlog = run.draft(arm, first)
                restore()
                check_c0(arm, blog)
                cfg = run.cfgs[i]
                if cfg["mode"] == "far" and cfg["deny"] is not None:
                    bad = [x for x in nlog if x[4] != ls.NEAR_PER_STREAM]
                    if not nlog or bad:
                        ap.die(RC_C3, "C3: arm %s near-block counts %s (want %d)"
                               % (arm, sorted({x[4] for x in nlog}), ls.NEAR_PER_STREAM))
                if arm in ("U81", "U81v") or arm.startswith("A"):
                    logits[arm] = lg
                q = pil.dist(lg[0])
                Q[i] = q.astype(np.float32)
                if arm == "D0" and not (torch.equal(lg, lg_T) and np.array_equal(p, q)):
                    ap.die(RC_D0, "D0 is not the target at doc %d site %d" % (d, site))
                if arm == "U81v" and not torch.equal(lg, logits["U81"]):
                    ap.die(RC_TWIN, "U81v != U81 bitwise at doc %d site %d" % (d, site))
                if arm.startswith("A") and torch.equal(lg, logits["U81"]):
                    ap.die(RC_C1, "C1: full denial at layer %s leaves the logits bitwise equal to U81 "
                           "(doc %d site %d): the mask has no path to the output" % (arm[1:], d, site))
                ov, tv = am.overlap_and_tv(p, q)
                x = am.sample(q, np.random.default_rng(am.seed_token(d, site, 1)))
                a, _ = am.acceptance(p, q, x)
                per_layer = [[0, 0] for _ in range(ls.N_LAYERS)]
                for r_ in recs:
                    per_layer[r_[4]][0] += r_[10]
                    per_layer[r_[4]][1] += r_[7]
                if cfg["deny"] is not None and cfg["mode"] == "all":
                    # C0, full denial: attention gets MASK_BIAS on EVERY selected
                    # non-tail block of each denied layer (64 rows per block)
                    nreg = {e[1]: e[4] for e in blog if e[2] == "draft"}
                    for l_ in cfg["deny"]:
                        if nreg.get(l_) != per_layer[l_][1] * 64:
                            ap.die(RC_C0, "C0: arm %s layer %d masked %r entries, %d selected blocks x 64 = %d"
                                   % (arm, l_, nreg.get(l_), per_layer[l_][1], per_layer[l_][1] * 64))
                row_ = dict(arm=arm, doc=d, row=row, site=site, j=1, tv=tv, overlap=ov,
                            accept_prob=a, x=int(x), argmax_agree=int(np.argmax(p) == np.argmax(q)),
                            entropy_p=am.entropy(p), entropy_q=am.entropy(q),
                            denied=int(sum(v[0] for v in per_layer)), selected=int(sum(v[1] for v in per_layer)),
                            denied_per_layer=[v[0] for v in per_layer],
                            masked_entries=int(sum(e[4] for e in blog if e[2] == "draft")),
                            masked_per_layer=[e[4] for e in blog if e[2] == "draft"],
                            arm_digest=ls.digest(),
                            labels="older-policy availability model d=1; teacher-forced prefix; K=1; "
                                   "unequal-memory diagnostic")
                run.rows_fh.write(json.dumps(row_) + "\n")
                for e in blog:
                    run.bias_fh.write(json.dumps([arm, d, site] + list(e)) + "\n")
            run.rows_fh.flush()
            run.bias_fh.flush()
            np.save(os.path.join(OUT, "vectors", "p_d%d_s%d.npy" % (d, site)), p.astype(np.float32))
            np.save(os.path.join(OUT, "vectors", "q_d%d_s%d.npy" % (d, site)), Q)
            if si == 0:
                # C2: consumption, every layer, first site of each document
                for l in range(ls.N_LAYERS):
                    lgP, _, blogP, _ = run.draft("A%d" % l, first, poison=(l, "denied", d * 1009 + site))
                    restore()
                    check_c0("P%d" % l, blogP)
                    okP = bool(torch.equal(lgP, logits["A%d" % l]))
                    lgV, _, _, _ = run.draft("U81", first, poison=(l, "visible", d * 1009 + site))
                    restore()
                    okV = not bool(torch.equal(lgV, logits["U81"]))
                    for arm, ok in (("P%d" % l, okP), ("V%d" % l, okV)):
                        run.cons_fh.write(json.dumps(dict(arm=arm, doc=d, row=row, site=site, layer=l, ok=ok)) + "\n")
                    run.cons_fh.flush()
                    if not okP:
                        ap.die(RC_C2, "C2: garbage in the DENIED rows of layer %d changed the logits: "
                               "denied K/V is read (doc %d)" % (l, d))
                    if not okV:
                        ap.die(RC_C2, "C2: garbage in a VISIBLE slot of layer %d did not change the logits: "
                               "the poison is not detectable there (doc %d)" % (l, d))
            pil.ss.assert_transients_intact(pil.model, trans)
        if rc:
            break
        if pil.host_digest_now() != pil.host_digest:
            ap.die(ap.RC_HOST, "G8: host K/V changed during document %d" % d)
    meta.update(complete=(rc == 0), status=status, wall_s=time.time() - run.t0)
    with open(mpath, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print("[layer-sens] %s rc=%d -> %s" % (status, rc, OUT), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
