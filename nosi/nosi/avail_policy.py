"""RESTRICTED AVAILABILITY for a NOSA decode step (retroinfer-eval fork).

WHAT THIS IS FOR. The acceptance pilot asks a single question: when a draft
attends with only PART of the sparse historical KV its own selection asked for,
how often does the exact target accept the token it drafts? This module is the
only thing in the fork that can make a decode step run with less than all of
that KV. It is INERT unless the environment variable NOSI_AVAIL is set, in the
same way `transfer_trace.TRACE` and `cache_engine.POOL_BLOCKS` are inert: the
knob is read ONCE, at import, so a run cannot change the policy halfway.

THE MODEL'S OWN SELECTION IS NEVER RESTRICTED. NOSA scores blocks from the
COMPRESSED keys, which are GPU-resident and never offloaded (nosa_llama.py
:565-571 builds them, :577-587 scores them, and the captured pooling+top-k
graph replays at :597 -- all BEFORE cache_engine.decode_update_kv at :603). The
policy reads `topk_idx` and never writes it, so at every availability level the
draft selects exactly the 64 blocks it would have selected at 100%. Only what
attention is allowed to USE changes.

TWO MECHANISMS, and they measure different things.

  mech=mask  (DEFAULT, and the literal reading of "attention sees only the
             blocks declared available"). NOSA's decode attention takes no
             block table -- it reads rows 0.._cache_lens-1 as one flat sequence
             (cache_engine.py:37-38) -- so a block cannot be dropped by index.
             It is dropped by BIAS. `kv_bias`, the fourth positional argument of
             flash_attn_nosa_with_kvcache (nosa_llama.py:610-616), is added to
             the pre-softmax score inside the kernel
             (dependencies/flash-attention-nosa/csrc/flash_attn_nosa/src/
             flash_fwd_kernel.h:342 and :971, `acc_s_4d(...) +=
             __bfloat162float(bias_ptr[index])`, before
             softmax_rescale_o at :1021). Writing a large negative FINITE value
             into the 64 rows of an unavailable slot removes that block from the
             softmax and renormalises attention over what is left. Nothing else
             is touched: the engine still fetches the block, its bookkeeping is
             upstream-exact, and the availability set is a property of THIS
             module's virtual cache, not of the engine's.

  mech=stale (the second, separately labelled arm). The fetch itself is
             suppressed: `_load_mask` gets -1 for the denied slots, so both
             Triton gathers return immediately for them (flash_h2d_mask.py:32-33
             and flash_h2d_mask_bias.py:36-37), and `_new_block_map_buf` is
             rewritten to the OLD occupant so the engine's belief keeps matching
             the physical bytes. Attention then reads the PREVIOUS occupant of
             that slot -- a real block of the same document, selected a few
             steps earlier. This is the physically honest model of "issue the
             fetch and do not wait", but it conflates MISSING with WRONG, so it
             is never the headline.

WHY MASK IS THE DEFAULT, against one of the two design notes this fork was
given. That note argued mask mode is impossible because `kv_bias` is
GQA-misindexed: it read flash_fwd_kernel.h:339-341 (`bidh *
kv_bias_head_stride`) next to :655-656 (`bidh / params.h_h_k_ratio` for K and
V) and concluded query head j reads the bias of KV head j%2, 8*j columns away.
THAT IS NOT WHAT RUNS ON THIS CALL PATH. flash_api.cpp:351 sets
`seqlenq_ngroups_swapped` when seqlen_q == 1, num_heads > num_heads_k, both
window sizes are negative and there is no alibi -- every one of which holds for
the NOSA decode call (seqlen_q 1 at nosa_llama.py:611, window_size default
(-1,-1) and alibi None at flash_attn_interface.py:1159-1162). Lines :352-357
then set num_heads := num_heads_k, so `params.h_h_k_ratio = h / h_k` at :100 is
1 and `bidh` inside the kernel IS the KV head. With bcache pinned to
(batch, seqlen_k, num_heads_k) at :366 and contiguous, head_stride = stride(-1)
= 1 and row_stride = stride(-2) = num_heads_k, so the index at :969-970 lands
on element [b, col, kv_head] -- correct. The other note's claim that
flash_attn_nosa's source is not in this tree is also wrong: it is at
external/NOSI/dependencies/flash-attention-nosa. The G0 device probe in the
pilot still runs, because reading a kernel is not the same as running one.

NOT -inf, -3e4. flash-attention splits the KV range and recombines partial
softmax accumulators (REPRODUCE.md's split-KV note; flash_fwd_kernel.h:594).
At 80% availability ~12.6 of the 63 non-tail slots are hidden, so a whole split
can be hidden; with -inf that split's row max is -inf and the max subtraction
gives NaN. A finite -3e4 gives every masked row a well-defined partial output
whose log-sum-exp is ~2.6e3 below any real one, so the recombination weights it
to exactly 0 in fp32. Gate G6 checks for NaN at the lowest availability point.

NEVER MASK _kv_bias_gpu IN PLACE. That buffer is persistent and the gather only
rewrites the rows of slots actually fetched, so a -3e4 left in a slot that is
not re-fetched next step would poison every later step silently. `mask_bias`
copies into a scratch tensor allocated once per layer with the same shape,
stride and contiguity, masks the scratch, and returns the scratch.

THE TAIL SLOT IS NEVER DENIED. nosa_pooling forces +inf on the first
`init_blocks` and the last `local_blocks+1` block indices
(max_pooling_fused.py:61-63), so the tail block is always inside the selection,
`diff` takes its old_hit branch at slot topk-1 and `_load_mask[..., 63]` is
always -1. The policy forces deny[..., tail] = False anyway, so no attention
row can ever be left without the current token.

CAUSALITY. `on_diff` receives ONLY `_block_map`, `_new_block_map_buf`,
`_load_mask`, `topk_idx` and the tail slot index -- all of which exist at
cache_engine.py:339 -- plus the layer index. It holds no reference to any token
stream, and tests/test_nosi_acceptance.py asserts that signature and that no
attribute of the policy is a token tensor. There is no instant-completion
oracle in any arm except d=0, which IS the unmodified engine and is labelled
the target, never a draft result.

COST. The deny decision is taken on the host: one device->host copy of a
(2, H, B, 64) int64 tensor per (layer, step) and one host->device copy of an
(H, B, 64) bool. That is 32 synchronisations per decode step. This is an
ACCURACY pilot; nothing here may be quoted as a speed number.
"""
from __future__ import annotations

import os
import random
from collections import OrderedDict

import torch

# The knob, read once at import (cache_engine.py:70 does the same for the pool).
SPEC = os.environ.get("NOSI_AVAIL", "").strip()

MASK_BIAS = -3.0e4          # finite on purpose; see the module docstring
INF = float("inf")

# The arrival delay of the VERIFIED trajectory. It is the shipped engine, whose
# decode is synchronous, so every fetch it issues has landed by the next step.
WARM_DELAY = 1.0

ROLE_TARGET = "target"      # deny nothing, freeze the virtual cache
ROLE_DRAFT = "draft"        # deny, and advance the virtual cache
ROLE_WARM = "warm"          # deny nothing, advance the virtual cache

MECHANISMS = ("mask", "stale")
ABLATIONS = ("none", "random")


def _parse(spec: str) -> dict:
    """"mech=mask,C=65,d=1" -> dict. Unknown keys are a hard error: a typo in a
    sbatch cell must not silently run the default policy and be recorded as the
    cell it was not."""
    out = dict(mech="mask", C=63, d=1.0, f=0.0, ablate="none", ready=1.0, seed=0)
    if spec in ("", "0", "off"):
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("NOSI_AVAIL term %r is not key=value" % part)
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in out:
            raise ValueError("NOSI_AVAIL: unknown key %r (known: %s)" % (k, sorted(out)))
        if k == "mech":
            if v not in MECHANISMS:
                raise ValueError("NOSI_AVAIL mech=%r not in %s" % (v, MECHANISMS))
            out[k] = v
        elif k == "ablate":
            if v not in ABLATIONS:
                raise ValueError("NOSI_AVAIL ablate=%r not in %s" % (v, ABLATIONS))
            out[k] = v
        elif k in ("C", "seed"):
            out[k] = int(v)
        else:
            out[k] = INF if v in ("inf", "Inf", "INF") else float(v)
    if out["C"] < 1:
        raise ValueError("NOSI_AVAIL C must be >= 1")
    if out["d"] < 0:
        raise ValueError("NOSI_AVAIL d must be >= 0 (d=0 is the shipped engine, d=inf is fixed residency)")
    if not (0.0 <= out["f"] <= 1.0):
        raise ValueError("NOSI_AVAIL f must be in [0,1]")
    if not (0.0 <= out["ready"] <= 1.0):
        raise ValueError("NOSI_AVAIL ready must be in [0,1]")
    return out


class VirtualCache:
    """One (layer, kv head, request) stream: an LRU of `capacity` blocks plus an
    in-flight queue with arrival delay `delay`.

    THE SEMANTICS ARE THE ONES THAT WERE VALIDATED. At capacity 63 and delay 1
    this reproduces the shipped engine's own load mask exactly -- that identity
    is what gate G5 checks per slot on the device, and it is the anchor for the
    whole availability axis (the offline replay at C=63, d=1 returns 0.943170 at
    L=16128, the same number as scripts/nosi_cache_sweep.py's C=63 row, which
    was itself validated against the engine's load mask at ratio 1.000).

      delay = 0    every request completes before this step's own attention:
                   availability is 1.0 by construction. THIS IS THE SHIPPED
                   ENGINE and is the TARGET, never a draft result.
      delay = 1    a block requested at step t is usable at step t+1: the
                   classic LRU hit fraction.
      delay = inf  FIXED RESIDENCY: nothing completes for the whole rollout,
                   and AvailPolicy.begin_rollout clears the in-flight queue so
                   the set really is frozen at what was RESIDENT when the
                   rollout started. Without that clearing, the fetches the last
                   verified step issued would land at draft step 1 and the point
                   would report a d=1 availability at j=1 under a d=inf label.
      f            the fraction of THIS step's own misses that land before this
                   step's attention, taken in issue order, never at random.
                   Only used to reach above the compulsory-miss floor.
    """

    __slots__ = ("capacity", "delay", "instep", "res", "inflight", "t")

    def __init__(self, capacity: int, delay: float, instep: float = 0.0):
        self.capacity = capacity
        self.delay = delay
        self.instep = instep
        self.res: "OrderedDict[int, None]" = OrderedDict()
        self.inflight: dict = {}
        self.t = 0

    def clone(self) -> "VirtualCache":
        c = VirtualCache(self.capacity, self.delay, self.instep)
        c.res = OrderedDict(self.res)
        c.inflight = dict(self.inflight)
        c.t = self.t
        return c

    def resident(self) -> set:
        return set(self.res)

    def step(self, sel) -> tuple:
        """Advance one decode step. `sel` is this step's selected block ids with
        the tail EXCLUDED. Returns (available_set, resident_before_set)."""
        t = self.t
        self.t += 1
        for b, a in list(self.inflight.items()):
            if a <= t:
                self.res[b] = None
                self.res.move_to_end(b)
                del self.inflight[b]
        while len(self.res) > self.capacity:
            self.res.popitem(last=False)
        before = set(self.res)
        hits = [b for b in sel if b in self.res]
        miss = sorted(b for b in sel if b not in self.res)
        if self.delay == 0:
            avail = set(sel)
            n_instep = len(miss)
        else:
            n_instep = int(self.instep * len(miss) + 1e-9)
            avail = set(hits) | set(miss[:n_instep])
        for b in hits:
            self.res.move_to_end(b)
        for i, b in enumerate(miss):
            if self.delay == 0 or i < n_instep:
                self.res[b] = None
                self.res.move_to_end(b)
            elif self.delay != INF and b not in self.inflight:
                self.inflight[b] = t + self.delay
        while len(self.res) > self.capacity:
            self.res.popitem(last=False)
        return avail, before


class AvailPolicy:
    """The process-wide policy. `install()` builds it; the driver drives it.

    THE STATE MODEL, because getting it wrong is how a pilot like this quietly
    measures the wrong thing.

      * Each AVAILABILITY POINT (a capacity/delay/ablation cell) owns its own
        set of virtual caches, one per (layer, KV head, request). They are
        advanced ONLY on the verified trajectory -- the teacher-forced target
        steps the driver runs between measurement sites -- so every point sees
        the SAME verified selection history and warms independently at its own
        capacity. Nothing else could be right: an LRU of 54 slots and one of 160
        do not reach the same resident set from the same history.
      * A ROLLOUT clones the point's caches into a working set, advances that
        working set for the K draft steps, and throws it away. The point's own
        caches never see a draft step, so rolling the engine back rolls the
        availability model back with it, for free and exactly.
      * A TARGET step neither denies nor advances anything: the verifier runs at
        full availability and its selections belong to the verifier.

    CAUSALITY. Every cache is driven by selections the model has already made.
    There is no lookahead, no future token, and no instant-completion oracle
    except d=0, which is the unmodified engine and is labelled the target.
    """

    def __init__(self, spec: str = None):
        cfg = _parse(SPEC if spec is None else spec)
        self.spec = SPEC if spec is None else spec
        self.mech = cfg["mech"]
        self.seed = cfg["seed"]
        # the point registered from the env spec itself, always id 0
        self.points: dict = {}
        self.register_point(0, cfg["C"], cfg["d"], cfg["f"], cfg["ablate"], cfg["ready"])
        self.active = 0
        self.capacity = cfg["C"]
        self.delay = cfg["d"]
        self.instep = cfg["f"]
        self.ablate = cfg["ablate"]
        self.ready_target = cfg["ready"]

        self.role = ROLE_WARM
        self.doc = 0
        self.site = 0
        self.call = -1
        self.layer = None
        self.tail_slot = None
        self.streams: dict = {}          # the ROLLOUT working set
        self._deny_dev = None
        self._deny_cpu = None
        self._scratch: dict = {}         # layer -> masked-bias scratch
        self._deny_buf: dict = {}        # layer -> broadcast bool buffer
        self.records: list = []
        self._stream_checked = False
        self.draft_j = 0                 # draft position inside the current rollout
        self._pre_deny_mask = None       # the load mask BEFORE any stale-mode denial
        # POINT_ID AND POS ARE APPENDED, NEVER INSERTED. Every consumer indexes
        # this tuple positionally, and without `point_id` the per-(layer, head)
        # archive pools an 80% arm with a 100% arm and calls the mixture "the
        # per-layer variation of availability". `pos` is the draft position j, so
        # a per-layer band can be read at one availability AND one position.
        self.rows_header = ("call", "doc", "site", "role", "layer", "head", "req",
                            "sel", "phys_hit", "model_hit", "denied", "ready",
                            "unexpressible", "lru_match", "point_id", "pos")

    # -- availability points ---------------------------------------------------
    def register_point(self, pid: int, capacity: int, delay: float, instep: float = 0.0,
                       ablate: str = "none", ready: float = 1.0):
        if ablate not in ABLATIONS:
            raise ValueError("ablate=%r not in %s" % (ablate, ABLATIONS))
        self.points[int(pid)] = dict(C=int(capacity), d=float(delay), f=float(instep),
                                     ablate=ablate, ready=float(ready), streams={})
        return self.points[int(pid)]

    def clear_points(self):
        self.points = {}

    def point_label(self, pid: int = None) -> str:
        c = self.points[self.active if pid is None else pid]
        return ("mech=%s,C=%d,d=%s,f=%g,ablate=%s,ready=%g"
                % (self.mech, c["C"], c["d"], c["f"], c["ablate"], c["ready"]))

    def is_natural(self, pid: int = None) -> bool:
        """True when the omitted blocks are the ones a real cache omits AND
        nothing is declared complete that a cache would not have.

        THREE THINGS DISQUALIFY A POINT, not one.
          ablate != "none"  a random omission is NOT a real miss: real misses are
                            the blocks the selection newly reached, and the
                            forced local window plus the init block make those
                            the FAR blocks. LABELLED ABLATION.
          f > 0             a fraction of THIS step's own misses is declared
                            usable in the step that requested it. That is a
                            PARTIAL INSTANT-COMPLETION ORACLE with no link model,
                            which the brief forbids in the main curve. It is a
                            POLICY ASSUMPTION and gets its own series.
          d == 0            every request completes before this step's attention:
                            the FULL instant-completion oracle. It is the SHIPPED
                            ENGINE and is the TARGET, an identity anchor, never a
                            draft result and never a member of the natural family.
        The analysis draws these as separate series and never fits one line
        through them."""
        c = self.points[self.active if pid is None else pid]
        return c["ablate"] == "none" and c["f"] == 0.0 and c["d"] != 0

    def omission_label(self, pid: int = None) -> str:
        """The series a point belongs to, as ONE string that the archive, the
        table and the legend all read. Never derived twice."""
        c = self.points[self.active if pid is None else pid]
        if c["d"] == 0:
            return "identity_anchor_shipped_engine"
        if c["ablate"] != "none":
            return "ablation_" + c["ablate"]
        if c["f"] > 0:
            return "policy_instep_completion_f=%g" % c["f"]
        return "natural_lru"

    @staticmethod
    def sim_mode_of(delay: float) -> str:
        """The simulation mode of ONE point, named on the row rather than
        asserted once in the manifest for a grid that may not contain it."""
        if delay == 0:
            return "shipped_engine"
        if delay == INF:
            return "fixed_residency"
        return "delayed_arrival_d=%g" % delay

    # -- role / step boundaries -------------------------------------------------
    def set_role(self, role: str):
        if role not in (ROLE_TARGET, ROLE_DRAFT, ROLE_WARM):
            raise ValueError("role=%r" % (role,))
        self.role = role

    def new_document(self, doc: int):
        self.doc = doc
        self.site = 0
        self.call = -1
        self.records = []
        self.streams = {}
        for c in self.points.values():
            c["streams"] = {}

    def set_site(self, site: int):
        self.site = site

    def begin_rollout(self, pid: int):
        """Clone this point's warmed caches into the working set and switch them
        from the shipped arrival delay to THIS POINT'S policy. The draft branches
        off the state the engine really had; from here it is on its own.

        A COLD POINT IS A HARD ERROR, not a 0%-availability measurement. If a
        point is registered AFTER the document's verified steps have run, its
        `streams` dict is empty, every block reads as a miss, and the rollout
        silently measures ready = 0 while being labelled with its capacity. That
        exact mistake made gate G5 compare an empty modelled resident set against
        the engine's ~59 resident slots and killed the job before any cell ran,
        so it is refused here rather than reported.

        FIXED RESIDENCY IS LITERALLY FROZEN. d = inf clears the in-flight queue,
        so nothing that the last VERIFIED step requested lands during the
        rollout. Keeping those arrivals would make the j = 1 availability of a
        d=inf point equal a d=1 point's, which is not what "the availability set
        is frozen at what was resident when the rollout started" says.
        """
        c = self.points[int(pid)]
        if not c["streams"] and self.call >= 0:
            raise RuntimeError(
                "availability point %d has an EMPTY virtual LRU but %d verified decode "
                "step(s) have already run on document %d. It was registered after the "
                "warm-up, so it never saw the verified trajectory and would report "
                "ready = 0 at capacity %d. Register every point (measurement AND gate) "
                "before the first decode step of the document."
                % (int(pid), self.call + 1, self.doc, c["C"]))
        self.active = int(pid)
        self.capacity, self.delay, self.instep = c["C"], c["d"], c["f"]
        self.ablate, self.ready_target = c["ablate"], c["ready"]
        self.draft_j = 0
        self.streams = {}
        for k, v in c["streams"].items():
            vc = v.clone()
            vc.delay = c["d"]
            vc.instep = c["f"]
            if c["d"] == INF:
                vc.inflight.clear()
            self.streams[k] = vc

    def end_rollout(self):
        self.streams = {}

    def begin_step(self):
        """Called from nosa_llama.decode_inference, next to the transfer_trace
        guard. No arguments: the driver sets the role BEFORE the call, so this
        hook cannot see anything about the token."""
        self.call += 1
        self.layer = None
        self._deny_dev = None
        self._deny_cpu = None
        self._pre_deny_mask = None
        # the draft position j inside the rollout, so every archived row says
        # WHICH step of the rollout it belongs to. A target or warm step is 0.
        self.draft_j = (self.draft_j + 1) if self.role == ROLE_DRAFT else 0

    def begin_layer(self, layer_idx: int):
        self.layer = layer_idx
        self._deny_dev = None
        self._deny_cpu = None
        self._pre_deny_mask = None

    # -- the hook ---------------------------------------------------------------
    def on_diff(self, block_map, new_block_map_buf, load_mask, topk_idx, tail_slot):
        """Between diff.diff_offload and self._block_map.copy_ (cache_engine.py
        :339-340). At this instant `block_map` still holds the OLD occupant of
        every slot, `new_block_map_buf` holds the REQUESTED occupant, and
        `load_mask[h,b,m] >= 0` names exactly the blocks the selection newly
        reached. Nothing else is in scope, which is what makes the policy
        provably blind to the future. `topk_idx` is accepted so the signature
        states what the hook may see; the requested set is read off
        `new_block_map_buf`, which is that same selection placed in slots."""
        self.tail_slot = int(tail_slot)
        H, B, M = new_block_map_buf.shape
        # THE ORDERING THIS HOOK DEPENDS ON. diff_offload launches with a bare
        # <<<>>> onto the LEGACY DEFAULT stream (diff_offload_kernel.cu), and
        # PyTorch's other streams do not synchronise with it implicitly, so
        # reading its output from a non-default stream would read stale
        # bookkeeping and deny the wrong blocks -- finite, plausible and wrong.
        # cache_engine's pooled decode TORCH_CHECKs exactly this; check it once
        # per process here for the same reason.
        if not self._stream_checked and new_block_map_buf.is_cuda:
            self._stream_checked = True
            cur = torch.cuda.current_stream(new_block_map_buf.device)
            if cur != torch.cuda.default_stream(new_block_map_buf.device):
                raise RuntimeError(
                    "the availability hook runs on stream %r, not the default stream: "
                    "diff_offload writes its output on the legacy default stream and "
                    "nothing orders the two. Do not wrap this decode in "
                    "torch.cuda.stream() or capture it in a CUDA graph." % (cur,))
        # ONE device->host copy per (layer, step): the requested slot->block map
        # and the physical load mask together.
        both = torch.stack((new_block_map_buf, load_mask)).cpu()
        new_map = both[0].tolist()
        lmask = both[1].tolist()
        tail = self.tail_slot
        deny = [[[False] * M for _ in range(B)] for _ in range(H)]
        any_deny = False
        for h in range(H):
            for b in range(B):
                row_map = new_map[h][b]
                row_mask = lmask[h][b]
                sel = [row_map[m] for m in range(M) if m != tail and row_map[m] >= 0]
                phys_hit = sum(1 for m in range(M) if m != tail and row_mask[m] < 0)
                key = (self.layer, h, b)
                if self.role == ROLE_TARGET:
                    # the verifier runs at full availability and its selections
                    # belong to the verifier: deny nothing, advance nothing
                    self.records.append((self.call, self.doc, self.site, self.role,
                                         self.layer, h, b, len(sel), phys_hit,
                                         -1, 0, len(sel), 0, -1, self.active, 0))
                    continue
                if self.role == ROLE_WARM:
                    # THE VERIFIED TRAJECTORY IS THE SHIPPED ENGINE. It fetches
                    # normally and every fetch completes, so each point's cache
                    # is warmed at ITS OWN CAPACITY but at the SHIPPED arrival
                    # delay (WARM_DELAY = 1), and the point's own delay applies
                    # only inside the rollout. Warming a d=inf point at d=inf
                    # would leave it with an EMPTY cache and report availability
                    # 0, which is not what fixed residency means: fixed residency
                    # freezes whatever the engine HAD when the rollout started.
                    # This is also what makes the fixed-residency prediction
                    # sensible -- ~94.4% at draft position 1, decaying by about
                    # 3.58 blocks per (layer, KV head, request) per step.
                    for c in self.points.values():
                        vc = c["streams"].get(key)
                        if vc is None:
                            vc = VirtualCache(c["C"], WARM_DELAY, 0.0)
                            c["streams"][key] = vc
                        vc.step(sel)
                    # a warm step advances EVERY point, so it belongs to none:
                    # point_id -1 keeps it out of every per-point aggregate.
                    self.records.append((self.call, self.doc, self.site, self.role,
                                         self.layer, h, b, len(sel), phys_hit,
                                         -1, 0, len(sel), 0, -1, -1, 0))
                    continue
                vc = self.streams.get(key)
                if vc is None:
                    vc = VirtualCache(self.capacity, self.delay, self.instep)
                    self.streams[key] = vc
                avail, before = vc.step(sel)
                model_hit = len(before & set(sel))
                if self.ablate == "random":
                    # LABELLED ABLATION, and a matched-availability control only:
                    # it exists to MEASURE whether a random omission gives the
                    # same acceptance as a real miss, never to stand in for one.
                    rng = random.Random((self.seed * 1000003 + self.doc) * 1009
                                        + ((self.site * 131 + vc.t) * 97 + self.layer) * 13
                                        + h * 3 + b)
                    keep = int(round(self.ready_target * len(sel)))
                    avail = set(rng.sample(sorted(sel), min(max(keep, 0), len(sel))))
                lru_match = -1
                if self.capacity == 63 and self.delay == 1 and self.instep == 0.0:
                    phys_res = {row_map[m] for m in range(M) if m != tail and row_mask[m] < 0}
                    lru_match = int(phys_res == (before & set(sel)))
                unexpressible = denied = 0
                for m in range(M):
                    if m == tail:
                        continue
                    blk = row_map[m]
                    if blk < 0 or blk in avail:
                        continue
                    if self.mech == "stale" and row_mask[m] < 0:
                        # the block is already physically resident, so a fetch
                        # cannot be un-issued: stale mode cannot express this cell
                        unexpressible += 1
                        continue
                    deny[h][b][m] = True
                    denied += 1
                    any_deny = True
                self.records.append((self.call, self.doc, self.site, self.role,
                                     self.layer, h, b, len(sel), phys_hit,
                                     model_hit, denied, len(sel) - denied,
                                     unexpressible, lru_match, self.active,
                                     self.draft_j))
        if self.role != ROLE_DRAFT or not any_deny:
            self._deny_cpu = None
            self._deny_dev = None
            return
        dev = torch.tensor(deny, dtype=torch.bool, device=new_block_map_buf.device)
        assert not bool(dev[..., tail].any()), "the tail slot must never be denied"
        self._deny_cpu = deny
        self._deny_dev = dev
        if self.mech == "stale":
            # exactly two edits, both on bookkeeping tensors that never reach
            # attention: the gathers return early on a negative id, and the map
            # keeps naming the block that is physically in the slot, so block ids
            # stay pairwise distinct and diff_offload's free-slot counting holds.
            #
            # THE ARCHIVE MUST SEE THE MASK AS IT WAS BEFORE THE DENIAL.
            # transfer_trace derives (b), the LRU miss count at this capacity,
            # from the mask it is handed; taking it after masked_fill_ would
            # count every SUPPRESSED fetch as a hit and understate (b) by exactly
            # the number of denials. cache_engine reads this clone and hands it
            # to record_mask instead of the live tensor.
            self._pre_deny_mask = load_mask.clone()
            load_mask.masked_fill_(dev, -1)
            torch.where(dev, block_map, new_block_map_buf, out=new_block_map_buf)

    def denied(self):
        """The int8 (H, B, M) denial mask of the current layer, for the archive."""
        if self._deny_dev is None:
            return None
        return self._deny_dev.to(torch.int8)

    def pre_denial_mask(self):
        """The load mask of the current layer AS IT WAS BEFORE stale mode's
        denial, or None when nothing was rewritten (which is every mask-mode step
        and every non-denying step). cache_engine hands this to
        transfer_trace.record_mask so quantity (b), the LRU miss count at this
        capacity, is read before any denial in EVERY arm."""
        return self._pre_deny_mask

    # -- the attention-side hook -------------------------------------------------
    def mask_bias(self, kv_bias):
        """Called between cache_engine.decode_update_kv and
        flash_attn_nosa_with_kvcache (nosa_llama.py:603 -> :610). Returns the
        caller's own tensor unchanged in every case but a DENYING DRAFT step in
        mask mode."""
        if self.mech != "mask" or self._deny_dev is None or self.role != ROLE_DRAFT:
            return kv_bias
        B, S, H = kv_bias.shape
        M = self._deny_dev.shape[-1]
        block = S // M
        scratch = self._scratch.get(self.layer)
        if scratch is None or scratch.shape != kv_bias.shape or scratch.dtype != kv_bias.dtype:
            # same shape, stride and contiguity: flash reads every stride off the
            # tensor it is given and asserts only stride(-1) == 1
            # (flash_attn_interface.py:1253-1254)
            scratch = torch.empty_like(kv_bias)
            self._scratch[self.layer] = scratch
        scratch.copy_(kv_bias)
        buf = self._deny_buf.get(self.layer)
        if buf is None or buf.shape != (B, M, block, H):
            buf = torch.empty((B, M, block, H), dtype=torch.bool, device=kv_bias.device)
            self._deny_buf[self.layer] = buf
        buf.copy_(self._deny_dev.permute(1, 2, 0).unsqueeze(2))
        scratch.view(B, M, block, H).masked_fill_(buf, MASK_BIAS)
        return scratch

    # -- reporting ---------------------------------------------------------------
    def summary(self, records=None) -> dict:
        """Ready-at-attention over the DRAFT calls only, with the denominator
        spelled out. `sel` is 63 per (layer, KV head, request, step): slot 63 is
        the tail, written locally and never fetched, and is excluded
        everywhere. Counting it would move every rate by about 1.6%, which is
        exactly the discrepancy already in this tree between
        transfer_trace.aggregate_step's `gamma` (0.9441, denominator 64) and the
        availability on the same cell (0.9432, denominator 63)."""
        recs = self.records if records is None else records
        sel = ready = phys_hit = model_hit = denied = unexp = n = 0
        match_ok = match_n = 0
        for r in recs:
            if r[3] != ROLE_DRAFT:
                continue
            n += 1
            sel += r[7]
            phys_hit += r[8]
            if r[9] >= 0:
                model_hit += r[9]
            denied += r[10]
            ready += r[11]
            unexp += r[12]
            if r[13] >= 0:
                match_n += 1
                match_ok += r[13]
        return dict(rows=n, sel_slots=sel, denominator_note="63 non-tail slots",
                    ready_slot_frac=(ready / sel) if sel else float("nan"),
                    lru_hit_frac_physical=(phys_hit / sel) if sel else float("nan"),
                    lru_hit_frac_modelled=(model_hit / sel) if sel else float("nan"),
                    denied=denied, unexpressible=unexp,
                    lru_fidelity=(match_ok / match_n) if match_n else float("nan"),
                    lru_fidelity_n=match_n)

    def per_layer_head(self, records=None, point_id: int = None) -> dict:
        """Ready-at-attention broken out per layer and per (layer, KV head), AT
        ONE AVAILABILITY POINT. THE PER-HEAD SPREAD IS A TWO-POINT SPREAD --
        NOSA-8B has 2 KV heads -- and must be reported as such, never as a
        distribution. Pooling several availability points into one per-layer
        number makes the sweep, not the layers, dominate the spread, so
        `point_id` is how a caller asks for a number that means something."""
        recs = self.records if records is None else records
        lay, head = {}, {}
        for r in recs:
            if r[3] != ROLE_DRAFT:
                continue
            if point_id is not None and r[14] != point_id:
                continue
            lay.setdefault(r[4], [0, 0])
            lay[r[4]][0] += r[11]
            lay[r[4]][1] += r[7]
            head.setdefault((r[4], r[5]), [0, 0])
            head[(r[4], r[5])][0] += r[11]
            head[(r[4], r[5])][1] += r[7]
        return dict(per_layer={k: (v[0] / v[1] if v[1] else float("nan")) for k, v in lay.items()},
                    per_layer_head={k: (v[0] / v[1] if v[1] else float("nan")) for k, v in head.items()})


POLICY: "AvailPolicy | None" = None


def install(spec: str = None) -> "AvailPolicy | None":
    """Build the process-wide policy. Returns None (and installs nothing) when
    NOSI_AVAIL is unset, in which case not one line of the guard bodies in
    cache_engine.py or nosa_llama.py executes."""
    global POLICY
    use = SPEC if spec is None else spec
    if use in ("", "0", "off"):
        POLICY = None
    else:
        POLICY = AvailPolicy(use)
    return POLICY


def uninstall():
    """Used only by the in-process 100%-identity gate, which must run one step
    with the guards inert and one with the policy installed and denying
    nothing, and require the two to be bit-identical."""
    global POLICY
    prev = POLICY
    POLICY = None
    return prev


def reinstall(policy):
    global POLICY
    POLICY = policy
