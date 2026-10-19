"""Layer-sensitivity diagnostic (accuracy only, NOSA-8B on NOSI, B = 1, K = 1).

At each restored exact prefix (PG-19 rows 0-3, 16 sites), the exact target p is
computed once at full availability, then each of the 86 registered arms drafts
ONE token from the same restored state with its own causally warmed virtual
cache, recomputing every hidden state and selection. Per arm and round we keep
the q vector (fp32 rounding, see below), TV / overlap / sampled acceptance, the
per-layer denial counts, and WHAT mask_bias HANDED TO ATTENTION at every layer
(scratch or not, MASK_BIAS entry count, implied count, byte digest).

Instrument checks (exit codes in the sbatch header). A located defect stops the
run; an inconclusive control is a MEASURED outcome and is recorded:
  C0  bias delivery: per draft call and layer, MASK_BIAS entries == denied
      slots x 64, and a denying layer hands attention the scratch tensor (stop);
  C1  path, every round and layer: A_l versus U81. Logits changed -> PASS.
      Unchanged logits are NOT a defect by themselves: a delivered mask can have
      a negligible or rounded-away final effect. The LOCAL WITNESS decides: the
      attention output of layer l (what flash_attn_nosa_with_kvcache returned),
      bitwise against U81's at layer l, with the untouched inputs (q, K, V)
      required bitwise equal, plus an fp32 reference attention over the same
      inputs (layer_sens.classify / unexplained). Local output moved ->
      INCONCLUSIVE_ABSORBED; local output unchanged within the kernel's own
      resolution -> INCONCLUSIVE_NEGLIGIBLE; unchanged although the reference
      says it must move, unequal untouched inputs, or logits moved without a
      local change -> DEFECT (stop);
  C2  consumption (first site of each document, every layer). P_l = A_l plus
      garbage written into the denied K/V rows of layer l after the mask must
      equal A_l (locally and in the logits): this proves an OUTPUT invariance
      for that finite poison, not that the hardware never loads those bytes.
      V_l = U81 plus garbage in ONE visible slot of layer l is classified like
      C1 (an unchanged final output is inconclusive, not a defect, unless the
      local witness locates one). Method and poison values from
      benchmarks/Efficiency/mask_check.py (feature/nosi-maskcheck @35eb117,
      job 2175726), reused, not re-derived;
  C3  far-only rule: every far-denial stream keeps exactly 16 near blocks;
  plus the U81v twin (bitwise), the D0 identity (q == p bitwise), frozen rows by
  full-text sha256 and the arm digest. The archived gates G0..G8 run first in
  the same job (sbatch).

Before ANY stop a bounded failing-comparison packet is written
(packets/NNN_<tag>.json always; .npz with the logits, selection / denial
identity, the actual attention inputs of the affected layer and both local
outputs while the PACKET_BYTES_MAX budget lasts). Every exit, including a
deadline, a SIGTERM from the sbatch hard cap and an unexpected exception, ends
with a manifest carrying an explicit terminal status and the partial round.

Labels: older-policy availability model (pilot VirtualCache, d=1, no
protection; K = 1 so protection would be inert); arrival is an accuracy
assumption; committed prefix is the document's teacher-forced continuation;
denial and capacity arms are unequal-memory diagnostics.

Spec: retroinfer-eval docs/superpowers/specs/2026-10-19-layer-sens-protocol.md
ENV: NOSI_REPO, NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_AVAIL (mech=mask...),
NOSI_LS_OUT, NOSI_LS_DIGEST, NOSI_LS_DEADLINE_EPOCH (absolute unix time),
NOSI_BENCH_L (16128).
"""
from __future__ import annotations

import time

T_ENTRY = time.time()            # process entry: the deadline budget includes the model load

import hashlib
import importlib
import importlib.util
import json
import math
import os
import signal
import sys
import traceback

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
K_POISON, V_POISON = 64.0, 1.0e4                 # mask_check.py values

RC_DIGEST, RC_ROWS, RC_C0, RC_C1, RC_C2, RC_C3, RC_TWIN, RC_DEADLINE, RC_D0 = (
    230, 231, 232, 233, 234, 235, 236, 237, 238)
RC_EXC = 239                     # unexpected exception (traceback in the log, message in the manifest)
RC_KILLED = 229                  # SIGTERM (the sbatch hard cap): partial results kept
RC_TERMINAL = 228                # the terminal manifest itself could not be written
STATUS_OF = {0: "COMPLETE", RC_DEADLINE: "INCOMPLETE_DEADLINE", RC_KILLED: "KILLED_SIGTERM",
             RC_DIGEST: "FAILED_DIGEST", RC_ROWS: "FAILED_ROWS", RC_C0: "FAILED_C0", RC_C1: "FAILED_C1",
             RC_C2: "FAILED_C2", RC_C3: "FAILED_C3", RC_TWIN: "FAILED_TWIN", RC_D0: "FAILED_D0",
             RC_EXC: "FAILED_EXCEPTION"}

VECTORS_LABEL = (
    "vectors/p_d{doc}_s{site}.npy (V,) and q_d{doc}_s{site}.npy (86, V), arm order = arms: the fp32 ROUNDING of "
    "the float64 distributions that were scored (rows.jsonl tv / overlap / accept_prob are float64, from "
    "am.normalise of the fp32 logits); recomputing scores from these files agrees only to fp32 precision. "
    "vectors/logits_d{doc}_s{site}.npy (2, V) fp32 = [target, U81] logits, from which the scored float64 p and "
    "U81 q are exactly reconstructible (Pilot.dist); failure / inconclusive packets keep the fp32 logits of "
    "every arm they compare. A partial round is q_d{doc}_s{site}.partial.npy (rows 0..n_done-1 valid).")


# ---------------------------------------------------------------------------
# terminal record: every exit writes a manifest with an explicit status
# ---------------------------------------------------------------------------
class Terminal:
    def __init__(self):
        self.out = None
        self.meta = {}
        self.message = ""
        self.round = None                # dict(doc, site, p, Q, n_done, p_saved)
        self.files = []
        self.packets = None

    def start(self, out, meta):
        self.out = out
        self.meta = dict(meta)
        self.write()

    def write(self):
        path = os.path.join(self.out, "manifest.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(self.meta, fh, indent=1, default=str)
        os.replace(path + ".tmp", path)

    def finish(self, rc):
        for fh in self.files:
            fh.flush()
        part = None
        r = self.round
        if rc != 0 and r is not None:
            vd = os.path.join(self.out, "vectors")
            if not r.get("p_saved") and r.get("p") is not None:
                np.save(os.path.join(vd, "p_d%d_s%d.npy" % (r["doc"], r["site"])), r["p"].astype(np.float32))
            if r.get("Q") is not None:
                np.save(os.path.join(vd, "q_d%d_s%d.partial.npy" % (r["doc"], r["site"])), r["Q"])
            part = dict(doc=r["doc"], site=r["site"], n_done=r["n_done"], stage=r.get("stage"))
        self.meta.update(complete=(rc == 0), status=STATUS_OF.get(rc, "FAILED_RC%d" % rc), rc=rc,
                         message=self.message, partial_round=part, wall_s=time.time() - T_ENTRY,
                         packets=(self.packets.summary() if self.packets is not None else None))
        self.write()


TERM = Terminal()


def die(code, msg):
    TERM.message = msg
    ap.die(code, msg)


def deadline_epoch(environ=None):
    """The ABSOLUTE deadline (unix seconds) the sbatch computed from its own
    start: a relative budget measured after the model load overran by the load."""
    env = os.environ if environ is None else environ
    if env.get("NOSI_LS_DEADLINE_S") is not None:
        die(RC_DEADLINE, "NOSI_LS_DEADLINE_S is the old relative budget; pass NOSI_LS_DEADLINE_EPOCH")
    try:
        t = float(env["NOSI_LS_DEADLINE_EPOCH"])
    except (KeyError, ValueError):
        die(RC_DEADLINE, "NOSI_LS_DEADLINE_EPOCH (absolute unix time) missing or not a number")
    if not t > T_ENTRY:
        die(RC_DEADLINE, "deadline %.0f is not after process entry %.0f" % (t, T_ENTRY))
    return t


# ---------------------------------------------------------------------------
# bounded packets, written BEFORE any stop
# ---------------------------------------------------------------------------
def _np(t):
    if isinstance(t, np.ndarray):
        return t, str(t.dtype)
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy(), "bfloat16 (stored as int16 bits)"
    return t.contiguous().numpy(), str(t.dtype)


class Packets:
    BULK = ("k_x", "v_x", "k_r", "v_r")          # dropped first when the budget runs out

    def __init__(self, out_dir, budget):
        self.dir = out_dir
        self.budget = int(budget)
        self.used = 0
        self.n = 0
        self.tags = []
        self.truncated = 0
        os.makedirs(out_dir, exist_ok=True)

    def write(self, tag, meta, arrays=None):
        self.n += 1
        base = "%03d_%s" % (self.n, tag)
        arrs, dtypes = {}, {}
        for k, v in (arrays or {}).items():
            if v is not None:
                arrs[k], dtypes[k] = _np(v)
        size = lambda: sum(a.nbytes for a in arrs.values())
        dropped = []
        for k in self.BULK:
            if self.used + size() <= self.budget:
                break
            if k in arrs:
                dropped.append(k)
                del arrs[k]
        if self.used + size() > self.budget:
            dropped += sorted(arrs)
            arrs = {}
        if dropped:
            self.truncated += 1
        if arrs:
            np.savez(os.path.join(self.dir, base + ".npz"), **arrs)
            self.used += os.path.getsize(os.path.join(self.dir, base + ".npz"))
        rec = dict(meta, packet=base, arrays=sorted(arrs), dtypes={k: dtypes[k] for k in arrs},
                   dropped=dropped, budget_used=self.used, budget=self.budget)
        with open(os.path.join(self.dir, base + ".json"), "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        self.tags.append(base)
        return base

    def summary(self):
        return dict(n=self.n, bytes=self.used, budget=self.budget, truncated=self.truncated,
                    first=self.tags[:20])


# ---------------------------------------------------------------------------
# the local witness at the affected attention call
# ---------------------------------------------------------------------------
def ref_attention(q, k, v, bias, n):
    """fp32 reference of flash_attn_nosa_with_kvcache for one decode token:
    scores = (q.k + bias) / sqrt(D) (the kernel adds the bf16 bias to the raw
    q.k accumulator, flash_fwd_kernel.h:321-342, before the softmax scale), keys
    0..n-1 only, GQA query head i -> KV head i // (Hq / Hk) (flash_api.cpp:351-355).
    q (B, 1, Hq, D); k, v (B, S, Hk, D); bias (B, S, Hk); n (B,). Returns (B, 1, Hq, D) fp32."""
    B, _, Hq, D = q.shape
    Hk = k.shape[2]
    g = Hq // Hk
    out = torch.empty((B, 1, Hq, D), dtype=torch.float32, device=q.device)
    for b in range(B):
        L = int(n[b])
        kk, vv = k[b, :L].float(), v[b, :L].float()
        bb = bias[b, :L].float()
        qq = q[b, 0].float().reshape(Hk, g, D)
        s = (torch.einsum("hgd,lhd->hgl", qq, kk) + bb.t().unsqueeze(1)) / math.sqrt(D)
        out[b, 0] = torch.einsum("hgl,lhd->hgd", torch.softmax(s, dim=-1), vv).reshape(Hq, D)
    return out


class Witness:
    """Captures, for the armed layers of a DRAFT call only, what the attention
    kernel was handed and what it returned. Output-preserving: the wrapper
    returns the kernel's own tensor, so every bitwise check (twin, D0, C1)
    sees the unmodified computation."""

    def __init__(self, pol, role_draft):
        self.pol = pol
        self.role_draft = role_draft
        self.layers = frozenset()
        self.cap = {}

    def arm(self, layers):
        self.layers = frozenset(layers)
        self.cap = {}

    def disarm(self):
        self.layers = frozenset()

    def wanted(self):
        return bool(self.layers) and self.pol.role == self.role_draft and self.pol.layer in self.layers

    def on_bias(self, caller_bias, handed, block_map):
        if self.wanted():
            deny = self.pol._deny_dev
            self.cap[self.pol.layer] = dict(
                caller_bias=caller_bias.detach().clone(), block_map=block_map.detach().clone(),
                deny=(deny.detach().clone() if deny is not None else None))

    def wrap(self, fn):
        if getattr(fn, "_ls_witness", None) is not None:
            raise RuntimeError("the attention kernel is already wrapped")

        def flash(*args, **kw):
            out = fn(*args, **kw)
            if self.wanted():
                # flash_attn_with_kvcache(q, k_cache, v_cache, kv_bias, ...); nosa_llama passes them positionally
                names = ("q", "k_cache", "v_cache", "kv_bias")
                got = dict(zip(names, args))
                got.update({nm: kw[nm] for nm in names[len(args):]})
                n = kw.get("cache_seqlens")
                o = out[0] if isinstance(out, tuple) else out
                c = self.cap.setdefault(self.pol.layer, {})
                c.update(q=got["q"].detach().clone(), k=got["k_cache"].detach().clone(),
                         v=got["v_cache"].detach().clone(),
                         bias=got["kv_bias"].detach().clone(), out=o.detach().clone(),
                         n=(n.detach().clone() if torch.is_tensor(n) else
                            torch.full((got["q"].shape[0],), int(n), dtype=torch.int32)))
            return out
        flash._ls_witness = self
        flash.__wrapped__ = fn
        return flash


CAP_KEYS = ("q", "k", "v", "bias", "n", "out", "caller_bias", "block_map")


def need_cap(caps, layer, arm, rc):
    """The witness capture of `arm` at `layer`, or a located stop: a missing or
    partial capture means the wrapped kernel / hooked mask_bias was not on the
    path this call took, and nothing it would have decided can be trusted."""
    c = caps.get(layer)
    if c is None or any(k not in c for k in CAP_KEYS):
        die(rc, "witness capture missing for arm %s layer %d (have %s): the attention call was not observed"
            % (arm, layer, sorted(c) if c else None))
    return c


def judge(kind, touched, cx, cr, lg_x, lg_r):
    """One control comparison at the affected layer. `touched` is what the
    intervention changes ('bias' for a mask arm, 'kv' for a poison arm); every
    other attention input must be bitwise equal. Returns (outcome, info)."""
    eq = {k: bool(torch.equal(cx[k], cr[k])) for k in ("q", "k", "v", "bias", "n")}
    untouched = ("q", "n") + (("k", "v") if touched == "bias" else ("bias",))
    same = all(eq[k] for k in untouched)
    local_equal = bool(torch.equal(cx["out"], cr["out"]))
    logits_equal = bool(torch.equal(lg_x, lg_r))
    info = dict(eq=eq, untouched=list(untouched), local_equal=local_equal, logits_equal=logits_equal,
                local_dmax=float((cx["out"].float() - cr["out"].float()).abs().max()),
                logits_dmax=float((lg_x - lg_r).abs().max()))
    unexp = None
    if kind == ls.EXPECT_CHANGE and same and local_equal and logits_equal:
        rx = ref_attention(cx["q"], cx["k"], cx["v"], cx["bias"], cx["n"])
        rr = ref_attention(cr["q"], cr["k"], cr["v"], cr["bias"], cr["n"])
        dev_x = float((cx["out"].float() - rx).abs().max())
        dev_r = float((cr["out"].float() - rr).abs().max())
        refmax = float(torch.maximum(rx.abs().max(), rr.abs().max()))
        unexp = ls.unexplained(dev_x, dev_r, refmax)
        info.update(dev_x=dev_x, dev_r=dev_r, ref_max_abs=refmax, ref_dmax=float((rx - rr).abs().max()),
                    unexplained=unexp, ref_x=rx, ref_r=rr)
    return ls.classify(kind, same, local_equal, logits_equal, unexp), info


def packet_arrays(cx, cr, lg_T, lg_r, lg_x, info):
    a = dict(logits_target=lg_T, logits_ref=lg_r, logits_x=lg_x)
    for side, c in (("x", cx), ("r", cr)):
        if c is None:
            continue
        for k in ("q", "k", "v", "bias", "caller_bias", "out", "n", "block_map", "deny"):
            if c.get(k) is not None:
                a["%s_%s" % (k, side)] = c[k]
    for k in ("ref_x", "ref_r"):
        if info.get(k) is not None:
            a[k] = info[k]
    return a


def jsonable(info):
    return {k: v for k, v in info.items() if not torch.is_tensor(v)}


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
            die(RC_ROWS, "row %d full-text sha256 %s != registered %s" % (r, tsha, ls.TEXT_SHA[r]))
        tok = tokenizer(t, return_tensors="pt").input_ids[0]
        if tok.shape[0] < want:
            die(RC_ROWS, "row %d has %d tokens < %d" % (r, tok.shape[0], want))
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
            die(RC_DIGEST, "NOSI_AVAIL must be set with mech=mask")
        if os.environ.get("NOSI_LS_DIGEST") != ls.digest():
            die(RC_DIGEST, "arm digest %s != registered NOSI_LS_DIGEST %r"
                % (ls.digest(), os.environ.get("NOSI_LS_DIGEST")))
        self.cfgs = ls.points()
        self.pol.clear_points()
        for pid, c in enumerate(self.cfgs):
            kw = ls.register_kwargs(c)
            self.pol.register_point(pid, kw["capacity"], kw["delay"], 0.0, "none", 1.0,
                                    layer_caps=kw["layer_caps"], deny_layers=kw["deny_layers"],
                                    deny_mode=kw["deny_mode"])
        self.pid = {c["arm"]: i for i, c in enumerate(self.cfgs)}
        # THE ATTENTION-SIDE HOOK: the logged mask_bias, the witness capture of
        # the caller's and the handed bias with the slot -> block map and the
        # denial, then (only in a C2 consumption arm) garbage written into the
        # layer's live window AFTER the mask and BEFORE attention, exactly where
        # mask_check.py put it.
        self.poison = None                       # (layer, "denied" | "visible", seed)
        self.witness = Witness(self.pol, self.pilot.ap.ROLE_DRAFT)
        orig = self.pol.mask_bias

        def hooked(kv_bias):
            out = orig(kv_bias)
            if self.witness.wanted():
                self.witness.on_bias(kv_bias, out, self._engine(self.pol.layer)._block_map)
            if self.poison is not None and self.pol.role == self.pilot.ap.ROLE_DRAFT \
                    and self.pol.layer == self.poison[0]:
                self._poison(self.poison)
            return out
        self.pol.mask_bias = hooked
        if self.pilot.ap.POLICY is not self.pol:
            die(RC_C2, "the model does not call this policy instance")
        nl = importlib.import_module("nosi.nosa_llama")
        nl.flash_attn_nosa_with_kvcache = self.witness.wrap(nl.flash_attn_nosa_with_kvcache)
        self.rows_fh = open(os.path.join(OUT, "rows.jsonl"), "a")
        self.cons_fh = open(os.path.join(OUT, "consumption.jsonl"), "a")
        self.bias_fh = open(os.path.join(OUT, "bias.jsonl"), "a")
        TERM.files += [self.rows_fh, self.cons_fh, self.bias_fh]

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
                    die(RC_C2, "no visible slot to poison at layer %d" % layer)
                m = ok[len(ok) // 2]
                self._poison_rows(eng, b, 0, m * bs, (m + 1) * bs, gen)

    def draft(self, arm, first, poison=None, witness=()):
        """One draft step of `arm` from the snapshot; returns logits, records,
        bias log, near log, witness captures {layer: capture}. The caller restores."""
        pol = self.pol
        pol.begin_rollout(self.pid[arm])
        pol.bias_log, pol.near_log = [], []
        rec0 = len(pol.records)
        self.poison = poison
        self.witness.arm(witness)
        try:
            lg = self.pilot.step(first, self.pilot.ap.ROLE_DRAFT).clone()
        finally:
            self.poison = None
            self.witness.disarm()
        recs, blog, nlog = pol.records[rec0:], pol.bias_log, pol.near_log
        pol.bias_log, pol.near_log = None, []
        pol.end_rollout()
        cap = self.witness.cap
        self.witness.cap = {}
        return lg, recs, blog, nlog, cap


def check_c0(arm, blog, packet=None):
    """C0 on the tensor mask_bias handed to attention, every layer of the call.
    `packet(tag, meta)` is written BEFORE the stop."""
    for (_call, layer, role, is_scratch, n_region, denied_ok, rest_ok, _sha) in blog:
        if role != "draft":
            continue
        if not (denied_ok and rest_ok) or (n_region > 0 and not is_scratch):
            msg = ("C0: arm %s layer %d: %d denied entries, all masked=%d, rest unchanged=%d, "
                   "scratch=%d" % (arm, layer, n_region, denied_ok, rest_ok, is_scratch))
            if packet is not None:
                packet("C0_%s_L%d" % (arm, layer), dict(check="C0", arm=arm, layer=layer, message=msg,
                                                        bias_log=[list(e) for e in blog]))
            die(RC_C0, msg)


def body(run, deadline):
    pil, pol = run.pilot, run.pol
    pk = TERM.packets
    smax = max(ls.SITES)
    ap.budget_or_die(smax + 1, ls.K)
    ids, ident, pq_sha = load_rows(pil.model.tokenizer, smax + ls.K + 2)
    ids = ids.to("cuda")
    TERM.meta.update(rows=ident, parquet_sha256=pq_sha, status="RUNNING", model_ready_s=time.time() - T_ENTRY)
    TERM.write()
    arms = ls.arms()

    def past(where):
        if time.time() > deadline:
            die(RC_DEADLINE, "absolute deadline %.0f reached %s (%.0f s after process entry)"
                % (deadline, where, time.time() - T_ENTRY))

    def rec_layer(recs, l):
        return [list(r_) for r_ in recs if r_[4] == l]

    past("after the model load")
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
            past("before doc %d site %d" % (d, site))
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
            vd = os.path.join(OUT, "vectors")
            np.save(os.path.join(vd, "p_d%d_s%d.npy" % (d, site)), p.astype(np.float32))
            Q = np.zeros((len(arms), p.size), dtype=np.float32)
            TERM.round = dict(doc=d, site=site, p=p, Q=Q, n_done=0, p_saved=True, stage="arms")
            logits, capU, capA = {}, {}, {}
            where = lambda arm: dict(arm=arm, doc=d, row=row, site=site)
            for i, arm in enumerate(arms):
                wl = range(ls.N_LAYERS) if arm == "U81" else ((int(arm[1:]),) if arm.startswith("A") else ())
                lg, recs, blog, nlog, cap = run.draft(arm, first, witness=wl)
                restore()
                check_c0(arm, blog, packet=lambda tag, m: pk.write(tag, dict(m, **where(arm)),
                                                                   dict(logits_target=lg_T, logits_x=lg)))
                cfg = run.cfgs[i]
                if cfg["mode"] == "far" and cfg["deny"] is not None:
                    bad = [x for x in nlog if x[4] != ls.NEAR_PER_STREAM]
                    if not nlog or bad:
                        msg = ("C3: arm %s near-block counts %s (want %d)"
                               % (arm, sorted({x[4] for x in nlog}), ls.NEAR_PER_STREAM))
                        pk.write("C3_%s" % arm, dict(where(arm), check="C3", message=msg,
                                                     near_log=[list(x) for x in nlog]),
                                 dict(logits_target=lg_T, logits_x=lg))
                        die(RC_C3, msg)
                if arm == "U81":
                    capU = cap
                if arm in ("U81", "U81v") or arm.startswith("A"):
                    logits[arm] = lg
                q = pil.dist(lg[0])
                Q[i] = q.astype(np.float32)
                TERM.round["n_done"] = i + 1
                if arm == "D0" and not (torch.equal(lg, lg_T) and np.array_equal(p, q)):
                    msg = "D0 is not the target at doc %d site %d" % (d, site)
                    pk.write("D0", dict(where(arm), check="D0", message=msg,
                                        logits_dmax=float((lg - lg_T).abs().max())),
                             dict(logits_target=lg_T, logits_x=lg))
                    die(RC_D0, msg)
                if arm == "U81v" and not torch.equal(lg, logits["U81"]):
                    msg = "U81v != U81 bitwise at doc %d site %d" % (d, site)
                    pk.write("TWIN", dict(where(arm), check="twin", message=msg,
                                          logits_dmax=float((lg - logits["U81"]).abs().max())),
                             dict(logits_target=lg_T, logits_ref=logits["U81"], logits_x=lg))
                    die(RC_TWIN, msg)
                c1 = None
                if arm.startswith("A"):
                    l = int(arm[1:])
                    cx, cr = need_cap(cap, l, arm, RC_C1), need_cap(capU, l, "U81", RC_C1)
                    c1, info = judge(ls.EXPECT_CHANGE, "bias", cx, cr, lg, logits["U81"])
                    if c1 != ls.PASS:
                        meta = dict(where(arm), check="C1", layer=l, ref="U81", outcome=c1,
                                    records_x=rec_layer(recs, l), **jsonable(info))
                        pk.write("C1_%s_%s" % (c1, arm), meta,
                                 packet_arrays(cx, cr, lg_T, logits["U81"], lg, info))
                    if c1 in ls.DEFECTS:
                        die(RC_C1, "C1 %s at layer %d (doc %d site %d): untouched inputs equal=%s, local "
                            "output equal=%s, logits equal=%s; packet written"
                            % (c1, l, d, site, info["eq"], info["local_equal"], info["logits_equal"]))
                    if si == 0:
                        capA[l] = cx
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
                            msg = ("C0: arm %s layer %d masked %r entries, %d selected blocks x 64 = %d"
                                   % (arm, l_, nreg.get(l_), per_layer[l_][1], per_layer[l_][1] * 64))
                            pk.write("C0_%s_L%d" % (arm, l_), dict(where(arm), check="C0", layer=l_, message=msg,
                                                                  records_x=rec_layer(recs, l_)),
                                     dict(logits_target=lg_T, logits_x=lg))
                            die(RC_C0, msg)
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
                if c1 is not None:
                    row_.update(c1=c1, c1_local_dmax=info["local_dmax"], c1_logits_dmax=info["logits_dmax"])
                run.rows_fh.write(json.dumps(row_) + "\n")
                for e in blog:
                    run.bias_fh.write(json.dumps([arm, d, site] + list(e)) + "\n")
            run.rows_fh.flush()
            run.bias_fh.flush()
            np.save(os.path.join(vd, "q_d%d_s%d.npy" % (d, site)), Q)
            np.save(os.path.join(vd, "logits_d%d_s%d.npy" % (d, site)),
                    torch.stack((lg_T[0], logits["U81"][0])).cpu().numpy().astype(np.float32))
            TERM.round = None
            if si == 0:
                # C2: consumption, every layer, first site of each document
                TERM.round = dict(doc=d, site=site, p=p, Q=None, n_done=len(arms), p_saved=True, stage="C2")
                for l in range(ls.N_LAYERS):
                    past("before C2 doc %d layer %d" % (d, l))
                    lgP, recsP, blogP, _, capP = run.draft("A%d" % l, first, poison=(l, "denied", d * 1009 + site),
                                                           witness=(l,))
                    restore()
                    check_c0("P%d" % l, blogP, packet=lambda tag, m: pk.write(
                        tag, dict(m, **where("P%d" % l)), dict(logits_target=lg_T, logits_x=lgP)))
                    lgV, recsV, _, _, capV = run.draft("U81", first, poison=(l, "visible", d * 1009 + site),
                                                       witness=(l,))
                    restore()
                    res = []
                    for name, kind, cx, cr, lx, lr_, rn, rx in (
                            ("P%d" % l, ls.EXPECT_SAME, need_cap(capP, l, "P%d" % l, RC_C2),
                             need_cap(capA, l, "A%d" % l, RC_C2), lgP, logits["A%d" % l], "A%d" % l, recsP),
                            ("V%d" % l, ls.EXPECT_CHANGE, need_cap(capV, l, "V%d" % l, RC_C2),
                             need_cap(capU, l, "U81", RC_C2), lgV, logits["U81"], "U81", recsV)):
                        out_, info = judge(kind, "kv", cx, cr, lx, lr_)
                        if out_ != ls.PASS:
                            meta = dict(where(name), check="C2", layer=l, ref=rn, outcome=out_,
                                        poison=("denied" if name[0] == "P" else "visible"),
                                        records_x=rec_layer(rx, l), **jsonable(info))
                            pk.write("C2_%s_%s" % (out_, name), meta, packet_arrays(cx, cr, lg_T, lr_, lx, info))
                        res.append((name, out_, info))
                        run.cons_fh.write(json.dumps(dict(
                            arm=name, doc=d, row=row, site=site, layer=l, ok=out_ not in ls.DEFECTS, outcome=out_,
                            local_equal=info["local_equal"], logits_equal=info["logits_equal"],
                            local_dmax=info["local_dmax"], logits_dmax=info["logits_dmax"])) + "\n")
                    run.cons_fh.flush()
                    for name, out_, info in res:
                        if out_ in ls.DEFECTS:
                            die(RC_C2, "C2 %s: %s at layer %d (doc %d): untouched inputs equal=%s, local output "
                                "equal=%s, logits equal=%s; packet written"
                                % (name, out_, l, d, info["eq"], info["local_equal"], info["logits_equal"]))
                capA = {}
                TERM.round = None
            pil.ss.assert_transients_intact(pil.model, trans)
        if pil.host_digest_now() != pil.host_digest:
            die(ap.RC_HOST, "G8: host K/V changed during document %d" % d)
    return 0


def _on_sigterm(signum, frame):
    TERM.message = TERM.message or "SIGTERM (sbatch hard cap): partial results kept"
    raise SystemExit(RC_KILLED)


def main() -> int:
    os.makedirs(os.path.join(OUT, "vectors"), exist_ok=True)
    TERM.packets = Packets(os.path.join(OUT, "packets"), ls.PACKET_BYTES_MAX)
    TERM.start(OUT, dict(study="layer-sensitivity diagnostic", sites=list(ls.SITES), K=ls.K, L=ap.L,
                         arm_digest=ls.digest(), arms=ls.arms(), terms=ls.terms(),
                         sampling=dict(temperature=ap.TEMP, top_p=ap.TOPP, top_k=ap.TOPK),
                         vectors=VECTORS_LABEL, process_entry=T_ENTRY, complete=False, status="STARTING"))
    signal.signal(signal.SIGTERM, _on_sigterm)
    rc = 0
    try:
        deadline = deadline_epoch()
        TERM.meta.update(deadline_epoch=deadline)
        run = Run()
        rc = body(run, deadline)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else RC_EXC
        if not isinstance(e.code, int):
            TERM.message = TERM.message or str(e.code)
    except BaseException as e:           # noqa: BLE001 -- the terminal record must still be written
        traceback.print_exc()
        rc = RC_EXC
        TERM.message = "%s: %s" % (type(e).__name__, e)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        TERM.finish(rc)
    except BaseException as e:           # noqa: BLE001
        print("[layer-sens] terminal manifest NOT written: %r" % (e,), flush=True)
        rc = rc or RC_TERMINAL
    print("[layer-sens] %s rc=%d -> %s" % (STATUS_OF.get(rc, "FAILED_RC%d" % rc), rc, OUT), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
