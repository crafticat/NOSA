"""Rung INT (the integration of AREA 1 J1, AREA 2 BF1 and AREA 3 ST1 / ST2 in ONE arm; KS/analysis/plan/
INTEGRATION_NOTES_2026-10-31.md; KS = /mnt/beegfs/ojerbi/kstep-cpu-runs): the classes of the two integrated arms.
  ARM L4PTFDGUDCDI   = L4PTFDGUDCD   (the best arm at C 128) + F1 + F2 + F6 + F3 (r2_j1_select: the J3 arm's classes)
                       + ST1 (the commit in one graph, by row) + ST2 (the joint twin section in one advance)
  ARM L4PTFDGUDKWCDI = L4PTFDGUDKWCD (the best arm at C 73)  + F1 + F2 + F6 (r2_j1_select.j1_wrap on the BF1 classes)
                       + ST1 + ST2 + BF1 (r2_bf1_fetch: the draft-issued background fetch into the kept store)
F3 is NOT in the BF1 arm (the notes, 3.4): F3 runs the draft residency inside the draft capture, and BF1's draft hook
(TentStateBF._resid_lean -> BgFetch.on_draft) must run eagerly per draft pass; there is no keep J3 arm either.
NO NEW KERNEL; no change to the three rungs' modules: their classes, composed.

THE TWO COLLISIONS (the notes, 3.1 and 3.2) AND THEIR FIXES:
  1. ST2 x F1: r2_j1_select's J1 state and r2_st1_commit.TwinST2 both override TentState._layer with a FULL copy; for a
     joint pass (R > 1) neither calls the other, so stacking them turns one hook OFF. FIX: IntLayer._layer = TwinST2's
     _layer with the selection line replaced by the J1 selection (the lines marked "# INT"; tests_cpu/test_int.py checks
     the copy). The state is re-classed to (IntLayer, the ST1 / ST2 state) AFTER st1_runner's re-class, so IntLayer
     owns _layer; ST2's fallback (one-row passes, the test record) is the next _layer of the MRO after TwinST2 = the J1
     state's, which falls back to the shipped one.
  2. ST1 x BF1: BF1's worker thread makes CUDA calls; a global-mode capture fails when another thread calls CUDA. BF1
     wraps the graph backend T.backend with its capture lock; ST1 captures the commit with its OWN backend
     (CudaGraphsPrivate). FIX: BgFetch.wrap_backend on the ST1 backend too (the same capture(fn, T, vs) signature).
NEGATIVE CONTROLS (tests only; ineg): "stacked" (no IntLayer: TwinST2's copy owns _layer, the J1 selection never runs
in a joint pass: J1-COMPILE and J1-PEQ must fail), "nolock" (the ST1 backend not wrapped: INT-LOCK must fail).
RECORD (read()["int"]): the arm, its hooks, the owners of _layer and commit, the J3 / BF1 classes present, the lock.
This file imports torch and the three rungs' modules only.
"""
import torch

import r2_bf1_fetch as BF
import r2_j1_select as J1
import r2_st1_commit as STC

ARMS_INT = dict(L4PTFDGUDCDI="L4PTFDGUDCD", L4PTFDGUDKWCDI="L4PTFDGUDKWCD")   # an INT arm -> its I4DC (D) arm
HOOKS = dict(L4PTFDGUDCDI=("F1", "F2", "F6", "F3", "ST1", "ST2"),
             L4PTFDGUDKWCDI=("F1", "F2", "F6", "ST1", "ST2", "BF1"))
KEEP_INT = ("L4PTFDGUDKWCDI",)
INEG = (None, "stacked", "nolock")


class IntError(RuntimeError):
    pass


class IntLayer:
    """The merged TentState._layer of an INT arm (module docstring, collision 1)."""

    def _layer(self, l, layer, hidden, pos_flat, cos_sin_cache, plan):
        """TwinST2._layer (r2_st1_commit.py) with the J1 selection (r2_j1_select: the compiled joint pooling)."""
        base = self._st2_base
        if plan.R == 1 or self.rec is not None:
            if plan.R > 1:
                self._st2["fallback"] += 1
            return base(self, l, layer, hidden, pos_flat, cos_sin_cache, plan)
        VR = None  # INT: the J1 selection below replaces the shipped _select_rows (no module lookup)
        rs = self.rs
        NL, st, lin = rs.nl, rs.st, plan.lin
        B, R = self.B, plan.R
        E = B * R
        Hkv, D = layer.num_key_value_heads, layer.head_dim
        twin = st.twins[l]
        eng = rs.cache.layers[l].cache_engine
        residual = hidden
        h = NL.layer_norm(hidden, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = lin(h, layer.wqkv)
        q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
        q = q.view(E, -1)
        k = k.view(E, -1)
        NL.apply_rope_with_cos_sin_cache_inplace(pos_flat, q, k, D, cos_sin_cache, True)
        q = q.reshape(E, layer.num_heads, D)
        k4 = k.reshape(B, R, Hkv, D)
        v4 = v.reshape(B, R, Hkv, D)
        c3 = cis.reshape(B, R, Hkv)
        pbs = self._st2_twin(l, twin, k4, c3, plan)                    # ST2: 2. the twin (and the deferred replay)
        sel = self._j1_select(plan.vs, layer, q, twin, l)  # INT: 3. the selection (J1: the compiled joint pooling)
        q4 = q.view(B, R, q.shape[1], q.shape[2])
        attn = self._rows(l, eng, twin, sel, q4, k4, v4, pbs, plan).reshape(E, 1, -1)
        hidden = residual + lin(attn, layer.wo)
        residual = hidden
        h = NL.layer_norm(hidden, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        h = lin(h, layer.gate_up_proj)
        out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
        NL.silu_and_mul(h, out)
        return residual + lin(out, layer.down_proj)


def owner(cls, name: str) -> str:
    """The name of the first class of cls's MRO that defines `name`."""
    for c in cls.__mro__:
        if name in c.__dict__:
            return c.__name__
    return None


_STATES = {}
_RUNNERS = {}
_TAILS = {}


def int_state(S):
    """(IntLayer, S): S = the ST1 / ST2 state class of an INT arm (cached)."""
    if S not in _STATES:
        _STATES[S] = type(S.__name__ + "INT", (IntLayer, S), {})
    return _STATES[S]


def base_runner(STF, arm: str):
    """The runner class of the INT arm WITHOUT ST1 / ST2 (the J1 or J3 runner on the right base classes)."""
    if arm == "L4PTFDGUDCDI":
        return J1.j1_classes(STF)["L4PTFDGUDCDJ3"][1]                  # F1 + F2 + F6 + F3 on L4PTFDGUDCD
    if arm == "L4PTFDGUDKWCDI":
        SBF, RBF = BF.bf_classes(STF)                                  # L4PTFDGUDKWCD + BF1
        return J1.j1_wrap(STF, SBF, RBF, arm)[1]                       # + F1 + F2 + F6
    raise IntError("unknown INT arm %r (one of %s)" % (arm, tuple(ARMS_INT)))


def int_runner(STF, arm: str, mode: str = "rows", ineg=None):
    """The runner class of an INT arm (cached per (STF, arm, mode, ineg)). Its __init__ takes the D arm's arguments,
    J1's (j1neg, j1draft, j1f6), ST1's (st1_backend) and, for the keep arm, BF1's (bf)."""
    if ineg not in INEG:
        raise IntError("unknown INT control %r (one of %s)" % (ineg, INEG))
    key = (id(STF), arm, mode, ineg)
    if key not in _RUNNERS:
        RS = STC.st1_runner(base_runner(STF, arm), mode, twin=True)

        class RunnerINT(RS):
            _int_arm, _int_neg = arm, ineg

            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                T = self.T
                if self._int_neg != "stacked":                         # collision 1: IntLayer owns _layer
                    T.__class__ = int_state(type(T))
                bf = getattr(self, "bf", None)
                if bf is not None and self._int_neg != "nolock":       # collision 2: the commit capture under the lock
                    bf.wrap_backend(T._st1["backend"])
                self.meta.update(int=dict(arm=self._int_arm, hooks=HOOKS[self._int_arm], ineg=self._int_neg))

            def _arm_name(self):
                return self._int_arm

            def int_record(self) -> dict:
                T = self.T
                cls = type(T)
                bf = getattr(self, "bf", None)
                be = T._st1["backend"]
                applies = (bf is not None and bf.mode != "off" and (bf.worker == "thread" or bf.packer == "gilfree"))
                return dict(arm=self._int_arm, hooks=list(HOOKS[self._int_arm]), ineg=self._int_neg,
                            layer_owner=owner(cls, "_layer"), commit_owner=owner(cls, "commit"),
                            j3=any(c.__name__ == "_J3Mix" for c in cls.__mro__),
                            bf=any(c.__name__ == "TentStateBF" for c in cls.__mro__),
                            lock_applies=bool(applies), st1_locked="capture" in vars(be))

            @torch.inference_mode()
            def read(self) -> dict:
                rec = self.int_record()                                # before read(): BF1's report() stops it
                out = super().read()
                out.update(int=rec)
                return out
        RunnerINT.__name__ = RunnerINT.__qualname__ = "TentRunner_" + arm
        _RUNNERS[key] = RunnerINT
    return _RUNNERS[key]


def with_tails(R):
    """R whose read() also carries ST1's committed-row tail checksum of every layer (r2_st1_commit.tail_checksum; the
    reference of an INT arm's TAIL-EQ). Cached."""
    if R not in _TAILS:
        class RunnerTails(R):
            def read(self) -> dict:
                out = super().read()
                with torch.inference_mode():
                    out.update(tails=STC.tail_checksum(self.T))
                return out
        RunnerTails.__name__ = RunnerTails.__qualname__ = R.__name__
        _TAILS[R] = RunnerTails
    return _TAILS[R]


# ------------------------------------------------------------------ the INT gates (pure)
def int_gates(arm: str, rd, L: int) -> list:
    """The INT MUST gates of an INT arm: [(name, must, passed, detail)].
      INT-CLASSES  _layer owned by IntLayer, commit by CommitST1; the J3 classes in the C 128 arm only; the BF1 classes
                   in the C 73 arm only; no control
      INT-LAYER    every joint layer ran BOTH the J1 selection and the ST2 advance (J1 calls = ST2 layers = joint passes
                   x L), no ST2 fallback
      INT-LOCK     the C 73 arm: when BF1's capture lock applies (a thread worker or the gilfree packer), the ST1 backend
                   is wrapped by it"""
    rd = rd or {}
    r = rd.get("int") or {}
    jp = int((((rd.get("tent") or {}).get("calls")) or {}).get("joint_passes", 0))
    keep = arm in KEEP_INT
    ok_c = (r.get("layer_owner") == "IntLayer" and r.get("commit_owner") == "CommitST1" and r.get("j3") == (not keep)
            and r.get("bf") == keep and r.get("ineg") is None and r.get("arm") == arm)
    j1c = (rd.get("j1") or {}).get("calls")
    s2 = rd.get("st2") or {}
    ok_l = jp > 0 and j1c == jp * int(L) and s2.get("layers") == jp * int(L) and s2.get("fallback") == 0
    out = [("%s-INT-CLASSES" % arm, True, bool(ok_c), "layer %s, commit %s, J3 %s, BF1 %s, control %s (arm %s)" % (
               r.get("layer_owner"), r.get("commit_owner"), r.get("j3"), r.get("bf"), r.get("ineg"), r.get("arm"))),
           ("%s-INT-LAYER" % arm, True, bool(ok_l), "joint passes %d x L %d; J1 selection calls %s; ST2 layers %s, "
            "fallback %s" % (jp, L, j1c, s2.get("layers"), s2.get("fallback")))]
    if keep:
        ok_k = (not r.get("lock_applies")) or bool(r.get("st1_locked"))
        out.append(("%s-INT-LOCK" % arm, True, bool(ok_k), "BF1's capture lock applies %s; the ST1 commit backend "
                    "wrapped %s; captures under the lock %s" % (r.get("lock_applies"), r.get("st1_locked"),
                                                               (rd.get("bf") or {}).get("captures_locked"))))
    return out
