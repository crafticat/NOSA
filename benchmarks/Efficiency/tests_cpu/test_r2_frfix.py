"""[frfix] CPU tests of the --b2-fewer-reads demand-merge selection (r2_own2.py OwnResidency._fr_merge_sel and
_fr_merge_split; _host_side's fewer-reads branch). Before the fix the branch read the FULL verify and draft records
(l, b, h, blk, flags, j / l, b, h, blk, flags) to the host in its one .tolist() and filtered them there in Python; GPU
runs measured it (B 216: median tick +150 ms, h_hs_takes +64 ms, h_hs_merges +70 ms). Now the device selects the merge
entries and only those cross, in the same ONE read.

  FR1  on random records (no V, no D, empty records, empty selection, full selection, int32 blocks): the device
       selection gives exactly the default path's merges (V["l"][m].tolist() .. in record order) and the old host
       filter's, verify part and draft part. NEGATIVE CONTROL: three planted wrong selections (DUP dropped, order
       reversed, the verify count off by one) are each seen by the same comparison; PRECONDITION per plant: the
       input where it is checked has the entries the plant changes (L34).
  FR2  the B2 round (pressure, lean and not lean): with fewer reads the round equals the default (lean: equals lean),
       and _host_side's ONE read holds exactly 7 take columns + 4 x the selected merges + the verify count + the
       recency counts (not lean: + the 6 verify record columns of the looks + the refusal count): no draft record
       column crosses, and under lean no record column at all. PRECONDITION: a tick whose records hold more entries
       than the selection (the read is smaller than the old full-record read there).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_frfix.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _shared(name, file):
    """A sibling test module as a shared module (loaded once per session under its own name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


HF = _shared("test_r2_hostfix_shared_fr", "test_r2_hostfix.py")      # the B2 round helpers (_run_hf, _diff, ...)
OWN2 = HF.OWN2
O = OWN2.O
p3, ref = HF.p3, HF.ref                                               # the fixtures, by name
OR = OWN2.OwnResidency
BITS = (O.PIN, O.READY, O.INFL, O.DUP, O.GRANT, O.EVICT, O.FIRST, O.REFUSED)


# =====================================================================================================================
# FR1 the device selection on random records
# =====================================================================================================================
def _rec(g, n, p_bit=0.3, blk_dtype=torch.int64, all_bits=None):
    """A random record part (l, b, h, blk, flags) of n entries; all_bits: every entry's flags = that value."""
    def col(hi):
        return torch.randint(0, hi, (n,), generator=g, dtype=torch.int64)
    if all_bits is not None:
        flags = torch.full((n,), all_bits, dtype=torch.int64)
    else:
        flags = torch.zeros(n, dtype=torch.int64)
        for bit in BITS:
            flags |= (torch.rand(n, generator=g) < p_bit).to(torch.int64) * bit
    return dict(l=col(4), b=col(8), h=col(4), blk=col(1000).to(blk_dtype), flags=flags)


def _default(V, D):
    """The default path's merges (_host_side without fewer reads): boolean masks on the device, four .tolist()."""
    out = []
    for X, m_of in ((V, lambda f: O.has(f, O.INFL) | O.has(f, O.DUP)), (D, lambda f: O.has(f, O.INFL))):
        if X is None:
            out.append([])
            continue
        m = m_of(X["flags"])
        out.append(list(zip(X["l"][m].tolist(), X["b"][m].tolist(), X["h"][m].tolist(), X["blk"][m].tolist())))
    return tuple(out)


def _old_host_filter(V, D):
    """The fewer-reads path before the fix: the full records on the host, filtered in Python."""
    out = []
    for X, bits in ((V, O.INFL | O.DUP), (D, O.INFL)):
        if X is None:
            out.append([])
            continue
        c = {n: X[n].to(torch.int64).tolist() for n in ("l", "b", "h", "blk", "flags")}
        out.append([(l, b, h, t) for (l, b, h, t, fl) in zip(c["l"], c["b"], c["h"], c["blk"], c["flags"]) if fl & bits])
    return tuple(out)


def _fixed(V, D, sel=OR._fr_merge_sel):
    """The fixed path: the device selection, then ONE read of the selection and its verify count (as _fr_read)."""
    msel, mnv = sel(V, D)
    if msel is None:
        return [], []
    flat = torch.cat([msel.reshape(-1).to(torch.int64), mnv.reshape(-1).to(torch.int64)]).tolist()
    return OR._fr_merge_split(flat[:-1], flat[-1])


def _cases():
    g = torch.Generator().manual_seed(17)
    out = [("none", None, None), ("empty_records", _rec(g, 0), _rec(g, 0)),
           ("v_only", _rec(g, 30), None), ("d_only", None, _rec(g, 30)),
           ("empty_sel", _rec(g, 25, all_bits=O.PIN | O.GRANT), _rec(g, 25, all_bits=O.READY | O.DUP)),
           ("full_sel", _rec(g, 25, all_bits=O.INFL | O.DUP), _rec(g, 25, all_bits=O.INFL)),
           ("int32_blk", _rec(g, 40, blk_dtype=torch.int32), _rec(g, 40, blk_dtype=torch.int32))]
    for i in range(80):
        nv = int(torch.randint(0, 60, (1,), generator=g))
        nd = int(torch.randint(0, 60, (1,), generator=g))
        p = float(torch.rand(1, generator=g))
        out.append(("rand%d" % i, _rec(g, nv, p_bit=p), _rec(g, nd, p_bit=p)))
    return out


CASES = _cases()


@pytest.mark.parametrize("name,V,D", CASES, ids=[c[0] for c in CASES])
def test_FR1a_the_device_selection_equals_the_default_and_the_old_filter(name, V, D):
    want = _default(V, D)
    assert _old_host_filter(V, D) == want                             # the two old paths agree (the reference)
    got = _fixed(V, D)
    assert (list(got[0]), list(got[1])) == (list(want[0]), list(want[1])), name
    assert all(type(x) is int for t in got[0] + got[1] for x in t)   # host ints, as .tolist() gives


def test_FR1b_the_edge_cases_are_what_they_say():
    """PRECONDITION of FR1a's edge ids: the empty selection is empty, the full one selects every entry."""
    c = {n: (V, D) for n, V, D in CASES}
    assert _default(*c["empty_sel"]) == ([], [])
    assert [len(x) for x in _default(*c["full_sel"])] == [25, 25]
    assert _default(*c["none"]) == ([], []) and _fixed(*c["none"]) == ([], [])
    assert OR._fr_merge_sel(None, None) == (None, None)
    msel, mnv = OR._fr_merge_sel(*c["empty_sel"])
    assert tuple(msel.shape) == (4, 0) and mnv.tolist() == [0]
    msel, mnv = OR._fr_merge_sel(*c["d_only"])
    assert mnv.tolist() == [0]                                        # no verify part: its count is 0
    assert sum(1 for _, V, D in CASES if _default(V, D)[0] and _default(V, D)[1]) > 40   # both parts merge often


# ------------------------------------------------------------------------------------------------- the plants
def _sel_drop_dup(V, D):
    """PLANT: the verify mask without DUP."""
    W = None if V is None else dict(V, flags=V["flags"] & ~O.DUP)    # (its count follows the planted mask)
    return OR._fr_merge_sel(W, D)


def _sel_reversed(V, D):
    """PLANT: the right entries in the reverse order."""
    msel, nv = OR._fr_merge_sel(V, D)
    return (None, None) if msel is None else (msel.flip(1), nv)


def _sel_count_off(V, D):
    """PLANT: the verify count one too small (one verify merge goes to the draft list)."""
    msel, nv = OR._fr_merge_sel(V, D)
    return (None, None) if msel is None else (msel, nv - 1)


def _has_dup_only(V, D):
    return V is not None and bool((O.has(V["flags"], O.DUP) & ~O.has(V["flags"], O.INFL)).any())


def _two_distinct(V, D):
    m = _default(V, D)
    s = m[0] + m[1]
    return len(s) >= 2 and s != s[::-1]


def _one_verify(V, D):
    return len(_default(V, D)[0]) >= 1


@pytest.mark.parametrize("plant,pre", [(_sel_drop_dup, _has_dup_only), (_sel_reversed, _two_distinct),
                                       (_sel_count_off, _one_verify)], ids=["drop_dup", "reversed", "count_off"])
def test_FR1c_negative_control_a_planted_wrong_selection_is_seen(plant, pre):
    """L34: each plant is checked on the inputs where it changes something (the precondition) and FR1a's comparison
    sees it on every one of them; the true selection passes on the same inputs."""
    hit = [(n, V, D) for n, V, D in CASES if pre(V, D)]
    assert len(hit) >= 10, "precondition: inputs where the plant changes the selection"
    for n, V, D in hit:
        want = _default(V, D)
        assert _fixed(V, D) == (list(want[0]), list(want[1])), n
        assert _fixed(V, D, sel=plant) != (list(want[0]), list(want[1])), n


# =====================================================================================================================
# FR2 the B2 round: the same values, and the ONE read holds the selection only
# =====================================================================================================================
def _read_sizes(eng, svc, box):
    """Records per _host_side call: the entries of its ONE _fr_read and the entries it must hold (from P and takes)."""
    hs0, rd0 = svc._host_side, svc._fr_read
    state = dict(inside=False)

    def host_side(steps, info, P, ev, takes, *a, **k):
        V, D = P["V"], P["D"]
        mv = int((O.has(V["flags"], O.INFL) | O.has(V["flags"], O.DUP)).sum()) if V is not None else 0
        md = int(O.has(D["flags"], O.INFL).sum()) if D is not None else 0
        nv_rec = int(V["l"].numel()) if V is not None else 0
        nd_rec = int(D["l"].numel()) if D is not None else 0
        want = 7 * int(takes["l"].numel()) + 4                       # the take columns, the recency counts
        if V is not None or D is not None:
            want += 4 * (mv + md) + 1                                 # the selection and its verify count
        if not svc.lean:
            want += 6 * nv_rec + (1 if D is not None else 0)          # the looks' verify columns, the refusal count
        old = 7 * int(takes["l"].numel()) + 4 + 6 * nv_rec + 5 * nd_rec   # the read before the fix
        box.append(dict(want=want, old=old, sel=4 * (mv + md), got=None))
        state["inside"] = True
        try:
            return hs0(steps, info, P, ev, takes, *a, **k)
        finally:
            state["inside"] = False

    def fr_read(cols):
        if state["inside"]:
            box[-1]["got"] = sum(int(c.numel()) for c in cols)
        return rd0(cols)
    svc._host_side, svc._fr_read = host_side, fr_read


@pytest.mark.parametrize("lean", [True, False], ids=["lean", "full"])
def test_FR2a_the_round_is_unchanged_and_reads_the_selection_only(p3, ref, lean):
    toks, _ = ref
    base = HF._run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False, lean=lean))
    box = []
    few = HF._run_hf(p3, toks, cfg="pressure", svc_kw=dict(checked=False, lean=lean, fewer_reads=True),
                     setup=lambda eng, svc: _read_sizes(eng, svc, box))
    assert HF._diff(base, few, HF.ALL_FIELDS + ("kept", "queue", "queued")) is None
    HF._same_end(base, few)
    assert len(box) == few.svc.b2_applied > 10
    assert all(r["got"] == r["want"] for r in box), [r for r in box if r["got"] != r["want"]][:3]
    assert any(r["got"] < r["old"] for r in box), "precondition: a tick whose records hold more than the selection"
    print("REPORT %s: host_side read entries per tick %s (before the fix %s); merges selected %d in total"
          % ("lean" if lean else "full", [r["got"] for r in box[:6]], [r["old"] for r in box[:6]],
             sum(r["sel"] for r in box) // 4))
