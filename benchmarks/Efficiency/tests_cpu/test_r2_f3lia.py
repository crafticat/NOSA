"""[f3lia] CPU tests of the merged tree r2-f3lia: flean 665ac69 (fused resolver v3 + --res-lean / --res-dev) + admitfill
c714fb2 (--inflight-array) + acc216 76758f4 (--store-committed-tokens and the ACC gates). The merge adds no mechanism:
the array poll is OwnResidency._poll_arrivals (the take path's issue parts), the fused resolve is FusedResolver.resolve,
and the ACC gates read only the finished points. These tests prove that the parts still work TOGETHER. The kernels run
under the Triton interpreter (TRITON_INTERPRET=1) in a FRESH process (L36), kernel v3 (as test_r2_flean.py's child).

  F3A  BOTH FLAGS REACH THE TICK (child): on test_r2_resfix.py's RL2_CASES, fused v3 + --res-lean --res-dev with the
       array poll: FusedResolver._lean_on() is True, the fused layers run (layers > 0, kernel v3), the array poll runs
       and admits (ia_stats polls / parts_due / admitted > 0; the old poll never runs: a counter on it stays 0).
  F3B  THE ARRAY POLL CHANGES NOTHING ON THE FUSED PATH (child): fused v3 + flags + --inflight-array == fused v3 + flags
       (the evalfast2f3l path) at every tick and layer (RF._rounds_diff), the same end (HF._same_end), the same arrival
       forced / held counts, the same fused report (launches, layers, stats, err).
  F3C  THE C3L EXACTNESS PATH IS KEPT (child): check mode + flags + --inflight-array compares every layer, 0 diffs, no
       error (the evalfast2c3lia gate's CPU form).
  F3D  NEGATIVE CONTROLS (child; L34: a precondition that the plant acts; L49: a plant on a module global can hit a
       sibling function and cancel out):
       (a) r2_resdev.excl_scan_rows made inclusive (bind_lean's ranks), with r2_resdev.sorted_free kept on the REAL
           scan (else its compaction shifts too and the two shifts cancel: test_r2_flean.py FL6b): with the array poll
           ON, the CHECK mode still reports xpool.* differences (its shadow pool binds with the REFERENCE bind), and the
           fused round differs from the clean one. PRECONDITION: the plant changed a rank in both rounds.
       (b) b2_mut ia_due_early (an issue part is due one tick early) on the fused + flags path: the round differs from
           the clean round AT the plant's tick. PRECONDITION: ia_mut_done names the plant. REPORT: the CHECK mode's diffs
           with the same plant (the CHECK compares the resolve only; a poll defect is the F3LIA-SAME / IA-SAME job).
  F3E  THE FLAGS IN PROCESS: the harness parses every flag set of the new cells; the refusals of both sides stay; the
       harness passes both flags to OwnResidency; OwnResidency's signature holds fused_kernel and inflight_array.
  F3F  STATIC GATE LISTS (T21 style): one_point's gates.extend helpers are fused_gates and stack_acted_gates; main runs
       eval_gates and acc_gates; preflight and one_point call note_fused_warm (flean); OwnResidency keeps the array poll
       helpers (admitfill); the step pins the modules of both sides as tracked.
  F3G  THE CELLS (static): evalfast2f3lia / evalfast2c3lia / evalacc216f3lia; the controls evalfast2f3l and evalfast2ia
       unchanged; the REGISTERED F3LIA block has no stamp and every PREDICTION is "TBD by lead" (after the lead stamps
       it: no TBD is left and the bands equal F3LIA_BANDS); outside DRYRUN the step refuses F3LIA_BANDS=TBD.
  F3H  THE DRYRUN of the three new cells on a clean checkout of this ROOT, parsed into the flags.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_f3lia.py
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _shared(name, file):
    """A sibling test module as a shared module (loaded once per session under its own name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / file))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


RF = _shared("test_r2_resfix_shared_rd", "test_r2_resfix.py")         # RL2's rounds and cases (shared name)
HF, TT, OWN2 = RF.HF, RF.TT, RF.OWN2
p3 = RF.p3                                                            # the fixture, by name
F3_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3ldx"                          # [f3ldx] (was nosi-r2-f3ld / -f3lia)
LAG = 2                                                               # the eval cells' --arrival-lag
STANDIN_BANDS = "1:2:1:2:1:2"                                         # syntax only (no prediction): the parse test


def _need_triton():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")


# =====================================================================================================================
# the child process (TRITON_INTERPRET=1 from its start)
# =====================================================================================================================
F3_CHILD = r'''
import importlib.util, json, os, sys
sys.dont_write_bytecode = True
assert os.environ.get("TRITON_INTERPRET") == "1", "F3 child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "F3 child: triton was imported before the interpreter flag was read"
import torch
HERE, PART = sys.argv[1], sys.argv[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


RF = load("test_r2_resfix_f3child", os.path.join(HERE, "test_r2_resfix.py"))
TT = RF.TT
FR = TT._load("r2_fusedres")
assert FR.triton is not None, "F3 child: triton did not import"
FR.DEFAULT_KERNEL = "v3"
FR.KERNELS = ("v3",)                                    # a resolver with any other kernel raises
RD = TT._load("r2_resdev")
OWNC = RF.OWN2.OwnResidency
OUT = {}
OLD_POLLS = [0]
_old_arrived = OWNC._arrived                            # the per-transfer arrival rule: the OLD poll's call only


def _count_old(self, tr):                               # (the array poll never calls it: test_r2_admitfill.py IA6)
    OLD_POLLS[0] += 1
    return _old_arrived(self, tr)
OWNC._arrived = _count_old


def p3_toks(path):
    P3 = load("piece3_v2_f3child", path)
    toks, _ = TT.serial_ref(TT.make_engine(TT.make_ops(P3)))
    return P3, toks


def engine(P3, layout, cfg, seed, mode, ia, checked=False, extra=None):
    kw = dict(resolve=mode, res_dev=True, arrival_lag=2, inflight_array=ia)
    kw.update(extra or {})
    return RF._engine(P3, layout, cfg, seed, True, checked, svc_kw=kw)


def fused_rep(svc):
    r = svc.fusedres.report()
    return dict(kernel=r["kernel"], launches=r["launches"], layers=r["layers"], stats=r["stats"], err=r["err"],
                compared=r["compared"], diffs=r["diffs"])


def run_info(r):
    s = r.svc
    return dict(err=r.err, ticks=len(r.ticks), ia=bool(s.inflight_array), ia_stats=dict(s.ia_stats),
                lean_on=s.fusedres._lean_on(), res_lean=bool(s.res_lean), dev=bool(s._lean_c.dev),
                forced=s.arrival_forced, held=s.arrival_held, rep=fused_rep(s),
                check_diffs=[list(x) for x in s.fusedres.diffs[:8]], n_check_diffs=len(s.fusedres.diffs),
                ia_mut_done=None if s.ia_mut_done is None else list(s.ia_mut_done))


if PART == "pool":
    i, P3, toks = int(sys.argv[3]), *p3_toks(sys.argv[4])
    layout, cfg, forced, seed, checked = RF.RL2_CASES[i]
    base = RF._round(engine(P3, layout, cfg, seed, "fused", False, checked), toks, forced)
    old_polls_base = OLD_POLLS[0]
    OLD_POLLS[0] = 0
    arr = RF._round(engine(P3, layout, cfg, seed, "fused", True, checked), toks, forced)
    chk = RF._round(engine(P3, layout, cfg, seed, "check", True, checked), toks, forced)
    old_polls_ia = OLD_POLLS[0]
    d = RF._rounds_diff(base, arr)
    try:
        RF.HF._same_end(base, arr)
        end = None
    except AssertionError as e:
        end = "same_end: " + repr(e)[:300]
    OUT["pool"] = dict(case=[layout, cfg, forced, seed, checked], base=run_info(base), arr=run_info(arr),
                       chk=run_info(chk), diff=None if d is None else repr(d)[:400], end=end,
                       old_polls=[old_polls_base, old_polls_ia],
                       inflight_same=list(dict.keys(base.svc.inflight)) == list(dict.keys(arr.svc.inflight)))

elif PART == "plant_scan":
    P3, toks = p3_toks(sys.argv[3])
    clean = RF._round(engine(P3, "pool", "pressure", 1, "fused", True), toks, True)
    real, real_sf = RD.excl_scan_rows, RD.sorted_free
    changed = []

    def inclusive(x):                                   # PLANT: every rank one too high where x is 1
        out = real(x) + x
        if not torch.equal(out, real(x)):
            changed.append(1)
        return out

    def sf_real_scan(*a, **k):                          # sorted_free keeps the REAL scan (L49: else the shifts cancel)
        RD.excl_scan_rows = real
        try:
            return real_sf(*a, **k)
        finally:
            RD.excl_scan_rows = inclusive
    RD.excl_scan_rows, RD.sorted_free = inclusive, sf_real_scan
    try:
        bad = RF._round(engine(P3, "pool", "pressure", 1, "fused", True), toks, True)
        n_round = len(changed)
        chk = RF._round(engine(P3, "pool", "pressure", 1, "check", True), toks, True)
    finally:
        RD.excl_scan_rows, RD.sorted_free = real, real_sf
    d = RF._rounds_diff(clean, bad)
    OUT["plant_scan"] = dict(clean=run_info(clean), bad=run_info(bad), chk=run_info(chk), changed_round=n_round,
                             changed_check=len(changed) - n_round, diff=None if d is None else repr(d)[:400],
                             restored=RD.excl_scan_rows is real and RD.sorted_free is real_sf)

elif PART == "plant_due":
    P3, toks = p3_toks(sys.argv[3])
    mut = dict(b2_mut={"ia_due_early": True})
    clean = RF._round(engine(P3, "stream", "default", 1, "fused", True, True), toks, True)
    bad = RF._round(engine(P3, "stream", "default", 1, "fused", True, True, extra=mut), toks, True)
    chk = RF._round(engine(P3, "stream", "default", 1, "check", True, True, extra=mut), toks, True)
    d = RF._rounds_diff(clean, bad)
    OUT["plant_due"] = dict(clean=run_info(clean), bad=run_info(bad), chk=run_info(chk),
                            diff=None if d is None else [d[0], repr(d)[:400]])
print("F3_JSON " + json.dumps(OUT))
'''


def _child(part, *extra):
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", F3_CHILD, str(HERE), part] + [str(x) for x in extra], env=env,
                       cwd=str(ROOT), capture_output=True, text=True, timeout=5400)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("F3_JSON ")]
    assert r.returncode == 0 and last, ("F3 child %s exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                                        % (part, r.returncode, r.stdout[-4000:], r.stderr[-8000:]))
    return json.loads(last[-1][len("F3_JSON "):])


_POOL = {}


def _pool(p3, i):
    _need_triton()
    if i not in _POOL:
        _POOL[i] = _child("pool", i, p3.__file__)["pool"]
    return _POOL[i]


CASE_IDS = ["-".join(str(x) for x in c) for c in RF.RL2_CASES]


# =====================================================================================================================
# F3A both flags reach the tick
# =====================================================================================================================
@pytest.mark.parametrize("i", range(len(RF.RL2_CASES)), ids=CASE_IDS)
def test_F3A_both_flags_reach_the_tick(p3, i):
    r = _pool(p3, i)
    for k in ("arr", "chk"):
        x = r[k]
        assert x["err"] is None and x["ticks"] > 10, (k, x["err"])
        assert x["ia"] and x["lean_on"] and x["res_lean"] and x["dev"], (k, "the flags did not reach the service")
        assert x["rep"]["kernel"] == "v3" and x["rep"]["layers"] > 0, (k, "precondition: the fused layers ran")
        st = x["ia_stats"]
        assert st["polls"] > 0 and st["parts_due"] > 0 and st["admitted"] > 0, (k, "the array poll did not admit", st)
    b = r["base"]
    assert not b["ia"] and b["ia_stats"]["polls"] == 0 and b["lean_on"], "the reference: flean's path, the old poll"
    assert r["old_polls"][0] > 0, "precondition: the counter on the old poll sees the reference's polls"
    assert r["old_polls"][1] == 0, ("the old poll ran with --inflight-array", r["old_polls"])


# =====================================================================================================================
# F3B the array poll changes nothing on the fused path
# =====================================================================================================================
@pytest.mark.parametrize("i", range(len(RF.RL2_CASES)), ids=CASE_IDS)
def test_F3B_the_array_poll_equals_the_old_poll_on_the_fused_lean_path(p3, i):
    r = _pool(p3, i)
    assert r["base"]["err"] is None and r["base"]["ticks"] > 10, ("precondition: the reference ran", r["base"]["err"])
    assert r["diff"] is None, ("fused + flags + --inflight-array differs from fused + flags", r["diff"], r["arr"]["err"])
    assert r["end"] is None, r["end"]
    assert (r["base"]["forced"], r["base"]["held"]) == (r["arr"]["forced"], r["arr"]["held"])
    assert r["inflight_same"]
    a, b = r["base"]["rep"], r["arr"]["rep"]
    assert (a["launches"], a["layers"], a["stats"], a["err"]) == (b["launches"], b["layers"], b["stats"], b["err"])


def test_F3B_precondition_the_cases_force_and_hold(p3):
    """L34: over the cases, the compared rounds force due transfers and hold early ones (both arrival branches)."""
    runs = [_pool(p3, i) for i in range(len(RF.RL2_CASES))]           # cached: F3A / F3B / F3C ran the same children
    assert sum(r["base"]["forced"] for r in runs) > 0, "precondition: no due transfer was forced"
    assert sum(r["base"]["held"] for r in runs) > 0, "precondition: no transfer landed before it was due"


# =====================================================================================================================
# F3C the C3L exactness path is kept
# =====================================================================================================================
@pytest.mark.parametrize("i", range(len(RF.RL2_CASES)), ids=CASE_IDS)
def test_F3C_check_mode_with_the_array_poll_compares_every_layer(p3, i):
    c = _pool(p3, i)["chk"]
    assert c["err"] is None, c["err"]
    assert c["rep"]["compared"] == c["rep"]["layers"] > 0, (c["rep"]["compared"], c["rep"]["layers"])
    assert c["rep"]["diffs"] == 0 and c["check_diffs"] == [] and c["rep"]["err"] == 0, c["check_diffs"]


# =====================================================================================================================
# F3D negative controls
# =====================================================================================================================
def test_F3Da_a_wrong_lean_scan_is_caught_with_the_array_poll_on(p3):
    _need_triton()
    r = _child("plant_scan", p3.__file__)["plant_scan"]
    assert r["restored"], "the plant was not removed"
    assert r["clean"]["err"] is None and r["clean"]["ia"] and r["clean"]["lean_on"], r["clean"]["err"]
    assert r["changed_round"] > 0 and r["changed_check"] > 0, ("precondition (L34): the plant changed a rank", r)
    assert r["bad"]["ia"] and r["chk"]["ia"] and r["chk"]["ia_stats"]["admitted"] > 0, "the array poll was not on"
    assert r["diff"] is not None or r["bad"]["err"] is not None, "the wrong scan was not caught by the round compare"
    assert r["chk"]["n_check_diffs"] > 0, ("the CHECK mode did not see the wrong lean bind", r["chk"])
    assert any(str(d[2]).startswith("xpool.") for d in r["chk"]["check_diffs"]), r["chk"]["check_diffs"]


def test_F3Db_an_early_due_issue_is_caught_on_the_fused_path(p3):
    _need_triton()
    r = _child("plant_due", p3.__file__)["plant_due"]
    assert r["clean"]["err"] is None and r["clean"]["ticks"] > 10 and r["clean"]["ia_mut_done"] is None, r["clean"]
    done = r["bad"]["ia_mut_done"]
    assert done is not None and done[1] == "ia_due_early", ("precondition (L34): the plant acted", done)
    assert r["bad"]["lean_on"] and r["bad"]["rep"]["layers"] > 0, "precondition: the plant ran on the fused path"
    assert r["diff"] is not None, "the early due issue was not seen"
    assert r["diff"][0] == done[0], ("seen at another tick", r["diff"], done)
    print("REPORT F3Db: the CHECK mode with the same plant: %d diffs, err %s (the CHECK compares the resolve only)"
          % (r["chk"]["n_check_diffs"], r["chk"]["err"]))


# =====================================================================================================================
# F3E the flags in process
# =====================================================================================================================
SG5_BASE = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x"]
B2 = SG5_BASE + ["--backends", "b2", "--X", "64", "--arrival-lag", "2"]
COMBO = ["--b2-resolve", "fused", "--fused-kernel", "v3", "--res-lean", "--res-dev", "--inflight-array"]


def test_F3E_the_flags_parse_together_and_reach_OwnResidency():
    G = HF._sg5()
    for res in ("fused", "check"):
        a = G.parse_args(B2 + ["--b2-resolve", res] + COMBO[2:])
        assert (a.b2_resolve, a.fused_kernel, a.res_lean, a.res_dev, a.inflight_array, a.arrival_lag) == \
            (res, "v3", True, True, True, 2), res
    acc = G.parse_args(B2 + COMBO + ["--modes", "fast,bi_c_strict", "--layout", "pool", "--store-committed-tokens"])
    assert acc.store_committed_tokens and acc.inflight_array and acc.b2_resolve == "fused" and acc.res_dev
    for bad in (SG5_BASE + ["--backends", "b2", "--X", "64"] + COMBO,          # --inflight-array: no fixed rule
                B2 + ["--b2-resolve", "fused", "--res-dev", "--inflight-array"],   # --res-dev without --res-lean
                B2 + ["--fused-kernel", "v3", "--inflight-array"]):           # --fused-kernel with the torch resolve
        with pytest.raises(SystemExit):
            G.parse_args(bad)
    me = inspect.getsource(G.make_engine)
    assert 'inflight_array=bool(getattr(a, "inflight_array", False))' in me
    assert 'res_lean=bool(getattr(a, "res_lean", False))' in me and 'res_dev=bool(getattr(a, "res_dev", False))' in me
    assert 'fused_kernel=(None if getattr(a, "b2_resolve", "torch") == "torch"' in me
    ps = inspect.signature(OWN2.OwnResidency.__init__).parameters
    assert ps["fused_kernel"].default is None and ps["inflight_array"].default is False
    src = inspect.getsource(OWN2.OwnResidency.__init__)
    assert "kernel=fused_kernel" in src and "self.inflight_array = bool(inflight_array)" in src
    assert "res_lean needs resolve" not in src                        # [flean] the refusal stays gone


# =====================================================================================================================
# F3F static gate lists (T21 style)
# =====================================================================================================================
def test_F3F_the_static_gate_lists_hold_both_sides():
    G = HF._sg5()
    op = inspect.getsource(G.one_point)
    assert sorted(set(re.findall(r"gates\.extend\((\w+)\(", op))) == ["fused_gates", "stack_acted_gates"]
    assert "note_fused_warm(svc)" in op and "note_fused_warm(svc)" in inspect.getsource(G.preflight)   # [fr3]
    main = inspect.getsource(G.main)
    for s in ("eval_gates(a, meta, res, gates)", "acc_gates(a, meta, res, gates, acc[", "af.inflight_array = False",
              "meta.update(inflight_array=True)", 'meta["fused_warm_mem_gb"] = _FUSED_WARM[0]'):
        assert s in main, s
    for name in ("fused_gates", "note_fused_warm", "eval_gates", "acc_gates", "accuracy_scope", "serial_reference",
                 "dup_pairs", "load_link"):
        assert callable(getattr(G, name, None)), name                 # the helpers of flean / acc216
    for name in ("_poll_arrivals_array", "_ia_add", "_ia_force", "_ia_raise", "_ia_forget"):
        assert callable(getattr(OWN2.OwnResidency, name, None)), name   # the helpers of admitfill
    for k in ("ia_skip_one", "ia_due_early", "ia_bad_gen"):
        assert k in OWN2.B2_MUTS, k
    step = HF.STEP.read_text()
    for f in ("nosi/nosi/r2_own2.py", "nosi/nosi/r2_residency_sc.py", "nosi/nosi/r2_transport_gilfree.py",
              "nosi/nosi/r2_fusedres.py", "benchmarks/Efficiency/r2_evalacc.py", "nosi/nosi/r2_xpool.py"):
        assert f in step, f                                           # tracked at the pinned commit (the step checks)


# =====================================================================================================================
# F3G the cells (static)
# =====================================================================================================================
def _bands(text):
    ln = next(x for x in text.splitlines() if x.strip().startswith("F3LIA_BANDS="))
    return ln.split("=", 1)[1].split()[0]


def _reg(text):
    reg = text[text.index("# REGISTERED for evalfast2f3lia / evalfast2c3lia / evalacc216f3lia"):]
    return reg[:reg.index("\n#\n")]


def test_F3G_the_cells_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=" + F3_ROOT
    assert F3_ROOT == HF.HF_ROOT
    assert "|evalfast2f3l|evalfast2ia|evalacc216m|evalacc216|evalfast2f3lia|evalfast2c3lia|evalacc216f3lia" \
           "|evalfast2f3d|evalfast2c3d|evalfast2f3t|evalfast2c3t|evalfast2f3c|evalfast2c3c] [gilfree|gather]" in text  # [f3ld] [f3ldx]
    assert "        evalfast2f3lia|evalfast2c3lia) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "        evalacc216f3lia) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    i = text.index("    evalfast2f3lia) MODES=fast")
    f3 = text[i:text.index(" ;;\n", text.index("EVALARG=(", i)) + 3]
    for w in ("BATCH=216", "PX=1512 ; PRV=1464", "VARIANTS=fused/batched", "NOSI=$NOSI_EVAL",
              "--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped",
              "--eval-role fast-time --mem-cap-gb 84", '--eval-bands "$F3LIA_BANDS" --b2-resolve fused --fused-kernel v3',
              "--wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array", '[ "${DRYRUN:-0}" = 1 ]'):
        assert w in f3, w
    assert "STEP_MIN=" not in f3 and "STEPS=" not in f3
    i = text.index("    evalfast2c3lia) MODES=fast")
    c3 = text[i:text.index(" ;;\n", i) + 3]
    for w in ("STEPS=8 ; DTICKS=4", "BATCH=216", "--b2-resolve check --fused-kernel v3 --res-lean --res-dev --inflight-array"):
        assert w in c3, w
    for w in ("--bare", "--eval-role", "--accuracy-record", "--eval-bands"):
        assert w not in c3, w
    i = text.index("    evalacc216f3lia) MODES=fast,bi_c_strict")
    ac = text[i:text.index(" ;;\n", i) + 3]
    for w in ("BATCH=216", "ACC=1", 'LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2f3lia/summary.json")',
              "--b2-resolve fused --fused-kernel v3", "--res-lean --res-dev --inflight-array --store-committed-tokens "
              "--serial-rows 8 --gold-workload \"$GOLD\"", '"${LINKARG[@]}"'):
        assert w in ac, w
    for w in ("--eval-role", "--accuracy-record", "--mem-cap-gb", "--eval-bands", "--bare", "STEP_MIN=", "STEPS="):
        assert w not in ac, w
    # the same-job controls keep their arguments (their own tests: test_r2_flean.py FL7, test_r2_admitfill.py IA7)
    assert "FLFLAGS=(--res-lean --res-dev) ; BANDS=335:375:258:292:220:248" in text
    assert "--eval-bands 328:362:305:338:245:282 --b2-resolve torch" in text and "--b2-fewer-reads --res-lean " \
           "--inflight-array) ;;" in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def test_F3G_the_registration_is_left_to_the_lead():
    text = HF.STEP.read_text()
    reg, bands = _reg(text), _bands(text)
    for tag in ("C3LIA", "F3LIA-MED", "F3LIA-MEAN", "F3LIA-EFF", "F3LIA-POLL", "F3LIA-SAME", "F3LIA-FALS", "F3LIA-MEM",
                "ACC3LIA-BI", "ACC3LIA-ETD", "ACC3LIA-AE", "ACC3LIA-LINK", "ACC3LIA-MEM", "nosi-r2-f3lia", "665ac69",
                "c714fb2", "evalfast2f3l", "evalfast2ia", "0 diffs"):
        assert tag in reg, tag
    preds = [ln for ln in reg.splitlines() if "PREDICTION:" in ln]
    assert len(preds) >= 11, preds
    if not re.search(r"STAMP \d{4}-\d\d-\d\dT", reg):              # the builder's state: nothing predicted
        assert bands == "TBD", bands
        assert all(ln.rstrip().endswith("PREDICTION: TBD by lead") for ln in preds), preds
    else:                                                             # the lead filled and stamped it
        assert "TBD" not in reg and bands != "TBD", "a stamped block with a TBD left"
        assert re.fullmatch(r"[0-9.]+(:[0-9.]+){5}", bands) and bands in reg, bands


def test_F3G_the_step_refuses_TBD_bands_outside_dryrun(tmp_path):
    text = HF.STEP.read_text()
    if _bands(text) != "TBD":
        pytest.skip("F3LIA_BANDS is filled (test_F3G_the_registration_is_left_to_the_lead checks the stamp)")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_", "DRYRUN"))}
    env.update(SG5_HOLDERS=str(tmp_path))                             # a second barrier: refused outside DRYRUN too
    r = subprocess.run(["bash", str(HF.STEP), "--nosi", "0" * 40, "418", "t1", "evalfast2f3lia"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, (r.returncode, r.stderr[-300:])
    assert "cell evalfast2f3lia: --eval-bands TBD" in r.stderr, r.stderr[-300:]
    assert "r2_tick_sg5.py" not in r.stdout


# =====================================================================================================================
# F3H the DRYRUN of the new cells
# =====================================================================================================================
F3_CELLS = dict(
    evalfast2f3lia=dict(res="fused", time=True, acc=False),
    evalfast2c3lia=dict(res="check", time=False, acc=False),
    evalacc216f3lia=dict(res="fused", time=False, acc=True))


@pytest.mark.parametrize("cell", sorted(F3_CELLS))
def test_F3H_the_cells_dryrun(tmp_path, cell):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    r = HF._dry(tmp_path, cell)
    if r.returncode != 0:
        assert Path(F3_ROOT).resolve() != HF.REPO.resolve(), r.stderr[-400:]
        pytest.skip("DRYRUN stopped outside the f3lia ROOT: %s" % r.stderr.strip()[-200:])
    line = r.stdout.strip().splitlines()[-1]
    got = line.split("r2_tick_sg5.py ", 1)[1].strip().replace("\\,", ",")
    w = F3_CELLS[cell]
    for s in ("--backends b2", "--batch 216", "--C 96 --X 1512 --R-ver 1464", "--layout pool --X-pool 1512 --arrival-lag 2",
              "--repeat-docs --log-slots -1 --skip-shipped", "--variants fused/batched", "--mem=240G", "--time=120",
              "--b2-resolve %s --fused-kernel v3" % w["res"], "--res-lean --res-dev --inflight-array",
              "PYTHONPATH=%s/nosi" % F3_ROOT, "--job-name=sg5-t1-418-%s " % cell, "/r2-runs/sg5/sg5_t1_418_%s " % cell):
        assert s in got or s in line, (cell, s)
    assert "torch/per_call" not in got
    bands = _bands(HF.STEP.read_text())
    if w["time"]:
        assert "--eval-bands %s" % bands in got, (cell, bands)
        if bands == "TBD":
            with pytest.raises(SystemExit):                           # the harness refuses the placeholder too
                HF._sg5().parse_args(got.split())
            got = got.replace("--eval-bands TBD", "--eval-bands " + STANDIN_BANDS)
    pa = HF._sg5().parse_args(got.split())
    assert (pa.batch, pa.X_pool, pa.b2_resolve, pa.fused_kernel) == (216, 1512, w["res"], "v3"), cell
    assert pa.res_lean and pa.res_dev and pa.inflight_array and pa.arrival_lag == 2, cell
    if w["time"]:
        assert pa.modes == ["fast"] and pa.eval_role == "fast-time" and pa.bare and pa.steps == 48
        assert pa.mem_cap_gb == 84 and pa.host_profile and pa.gc_freeze and pa.lean_time
        assert pa.wait_index and pa.b2_fewer_reads and not pa.store_committed_tokens
        assert "--accuracy-record /mnt/beegfs/ojerbi/r2-runs/evalacc/eacc_t1_418_evalacc/summary.json" in got
    elif w["acc"]:
        assert pa.modes == ["fast", "bi_c_strict"] and pa.store_committed_tokens and pa.serial_rows == 8
        assert pa.eval_role == "" and not pa.bare and pa.mem_cap_gb == 0 and not pa.accuracy_record and pa.steps == 48
        assert pa.host_profile and pa.gc_freeze and pa.lean_time and pa.wait_index and pa.b2_fewer_reads
        assert pa.gold_workload == "/mnt/beegfs/ojerbi/kstep-cpu-runs/workload/pg19-test-ctx65280"
        assert pa.link_record == "/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_t1_418_evalfast2f3lia/summary.json"
        assert "[acc216] modes fast,bi_c_strict" in r.stdout
    else:
        assert pa.modes == ["fast"] and pa.eval_role == "" and pa.steps == 8 and pa.draft_ticks == 4 and not pa.bare
        for s in ("--accuracy-record", "--eval-bands", "--host-profile", "--wait-index", "--b2-fewer-reads",
                  "--store-committed-tokens"):
            assert s not in got, (cell, s)
