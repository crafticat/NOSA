"""[f3tall] CPU tests of the merged tree r2-f3tall: f3all 3a7f51a (f3lia + --end-batch + --apply-batched) merged with
f3ldx 3769b64 (f3lia + --worker cpp --b2-issue desc + --no-transfer-mirror + the descriptor's lane row on the X pool),
and of the cells evalfast2tall / evalfast2c3tall / evalacc216tall of r2_tick_sg5_step.sh.

THE JOINING LINE. Both sides changed the end of a request: b2-desc added to OwnResidency.end_request the drop of the
request's write-back marks (_hm[b] = -1, E's T4 marks of the desc issue); endbatch added OwnResidency.end_requests (ALL
the ends of a tick in one call), written before b2-desc. The merge adds the same drop to end_requests (one row per
ended batch slot). Nothing else joins the two sides: the batched apply (TickEngine._apply_batched) runs after
backend.decisions and writes only the twins' draft banks and the draft ring rows; the desc issue and the table poll are
the service's.

The packer is BUILT by the tests (packer v3 of this tree: test_r2_b2_desc.py's v3 fixture, the OpenMP build): run
them on a compute node, never on the login node. No Triton (the torch resolve of make_res_engine).

  TL1  STATE: on the desc path (stream and pool layouts; the Transfer mirror on, and --no-transfer-mirror with lean),
       --end-batch == the per-request ends at EVERY tick (test_r2_b2_desc.py FIELDS: rows, tokens, logits, pages,
       state, table, counters, readers, memory, arrival, store) and _hm at every tick and at the end. The batched run
       ends requests only through end_requests (end_request never).
  TL1z PRECONDITIONS over the TL1 runs: end_requests ran; some ended batch slot held a write-back mark (_hm >= 0) when
       its request ended (else the joining line is not exercised).
  TL2  NEGATIVE CONTROL (L34, L49): end_requests without the joining line (the plant restores the ended rows of _hm
       after the call). Precondition: a planted call restored a row with a mark >= 0. Detected: _hm differs from the
       per-request reference, first at the record after that call.
  TL3  the harness: the tall flag set parses (desc + --no-transfer-mirror + --end-batch + --apply-batched); make_engine
       passes all of them; OwnResidency.end_requests has the joining line.
  TL4  the cells (static): evalfast2tall / evalfast2c3tall / evalacc216tall; the controls evalfast2f3t, evalfast2all,
       evalfast2f3lia keep their arguments; the REGISTERED TALL block is NOT STAMPED (every PREDICTION "TBD by lead",
       TALL_BANDS=TBD until the lead stamps it); outside DRYRUN the step refuses TALL_BANDS=TBD.
  TL5  the DRYRUN argv on a clean checkout of this ROOT: evalfast2tall = evalfast2f3t + --end-batch --apply-batched,
       evalfast2c3tall = evalfast2c3t + the two flags, evalacc216tall = evalacc216all with the f3t flags in place of
       --inflight-array; each parses into the flags.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_r2_f3tall.py
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import re
import subprocess
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


BD = _shared("test_r2_b2_desc_shared_f3ld", "test_r2_b2_desc.py")      # the desc world (_run, _first_diff, v3)
F3 = _shared("test_r2_f3ld_shared_f3ldx", "test_r2_f3ld.py")          # _cell (shared name with test_r2_f3ldx.py)
FA = _shared("test_r2_f3all_shared_f3tall", "test_r2_f3all.py")       # _argv, _norm (DRYRUN helpers)
HF = F3.HF                                                            # the step helpers (_dry, _clean_head, STEP)
OWN2 = BD.OWN2
p3, ref, v3 = BD.p3, BD.ref, BD.v3                                    # the fixtures, by name
TALL_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-f3tall"
NEW = ["--end-batch", "--apply-batched"]
F3T = ["--worker", "cpp", "--b2-issue", "desc", "--no-transfer-mirror"]
JOIN = "self._hm[b] = -1"


# =====================================================================================================================
# TL1 / TL1z / TL2 the joining line on the desc path
# =====================================================================================================================
CASES = [dict(name="%s_%s" % (lay, "nm" if nm else "mirror"), pool=(lay == "pool"),
              kw=dict(lean=True, no_transfer_mirror=True) if nm else {})
         for lay in ("stream", "pool") for nm in (False, True)]
_COV = {}


def _round(p3, ext, toks, case, batched, plant=False):
    """One desc round; batched: OwnResidency(end_batch=True) and TickEngine.end_batch (make_engine's order). plant:
    end_requests without the joining line (TL2). Returns the round, the _hm records (one per tick, then the final),
    the end_requests calls (record index, uids, a mark >= 0 in an ended row before the call) and end_request calls."""
    hm, calls, singles = [], [], []
    kw = dict(case["kw"], end_batch=bool(batched))

    def hooks(tp, svc, ticks, world, probes):
        probes.append(lambda n, eng: hm.append(svc._hm.clone()))

    def setup(eng, svc, ticks):
        eng.end_batch = bool(batched)
        er0, one0 = svc.end_requests, svc.end_request

        def end_requests(uids):
            uids = list(uids)
            bs = [svc.uid_b[u] for u in uids]
            before = svc._hm[bs].clone()
            calls.append((len(ticks), uids, bool((before >= 0).any())))
            er0(uids)
            if plant:
                svc._hm[bs] = before                                 # the plant: the joining line undone
        svc.end_requests = end_requests

        def end_request(uid):
            singles.append(uid)
            one0(uid)
        svc.end_request = end_request
    res = BD._run(p3, ext, toks, "desc", svc_kw=kw, plant=hooks, setup=setup, pool=case["pool"])
    hm.append(res.svc._hm.clone())
    return res, hm, calls, singles


def _hm_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if not torch.equal(x, y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_TL1_end_batch_on_the_desc_path_equals_the_per_request_ends(p3, ref, v3, case):
    toks, _ = ref
    base, hm0, c0, s0 = _round(p3, v3, toks, case, batched=False)
    got, hm1, c1, s1 = _round(p3, v3, toks, case, batched=True)
    assert base.err is None and got.err is None, (base.err, got.err)
    assert base.svc.issue == "desc" and base.svc._hm is not None
    assert base.svc.no_transfer_mirror == bool(case["kw"]) and got.svc.end_batch and not base.svc.end_batch
    assert c0 == [] and len(s0) == base.eng.B                         # the reference: one end_request per request
    assert s1 == [] and sorted(u for (_, us, _) in c1 for u in us) == list(range(got.eng.B))
    assert BD._first_diff(base, got) is None
    assert all(torch.equal(x, y) for x, y in zip(base.final, got.final))
    assert _hm_diff(hm0, hm1) is None
    _COV[case["name"]] = (len(c1), max(len(us) for (_, us, _) in c1), any(m for (_, _, m) in c1))


def test_TL1z_preconditions_over_the_runs():
    if len(_COV) != len(CASES):
        pytest.skip("TL1 did not run in this session (run the whole file)")
    assert all(n >= 1 for (n, _, _) in _COV.values()), _COV
    assert any(m for (_, _, m) in _COV.values()), ("no ended slot held a write-back mark: the joining line is not "
                                                   "exercised", _COV)
    print("TL1z coverage (calls, most ends in one call, a mark at the end): %s" % _COV)


def test_TL2_without_the_joining_line_the_marks_differ_at_the_next_record(p3, ref, v3):
    toks, _ = ref
    seen = 0
    for case in CASES:
        base, hm0, _, _ = _round(p3, v3, toks, case, batched=False)
        got, hm1, c1, _ = _round(p3, v3, toks, case, batched=True, plant=True)
        acted = [n for (n, _, m) in c1 if m]
        if not acted:
            continue                                                  # this case's plant is a no-op (L34)
        seen += 1
        assert base.err is None and got.err is None, (base.err, got.err)
        assert _hm_diff(hm0, hm1) == acted[0], (case["name"], _hm_diff(hm0, hm1), acted)
    assert seen >= 1, "the plant never acted (no ended slot held a mark): TL2 proves nothing"


# =====================================================================================================================
# TL3 the harness and the joining line (static)
# =====================================================================================================================
def test_TL3_the_flag_set_parses_and_reaches_the_service():
    G = HF._sg5()
    base = ["--out", "o", "--workload", "w", "--biattn", "b", "--gilfree", "g", "--gilfree-build", "/x", "--backends",
            "b2", "--arrival-lag", "2", "--lean-time", "--variants", "fused/batched", "--b2-resolve", "fused"]
    a = G.parse_args(base + F3T + NEW)
    assert a.b2_issue == "desc" and a.worker == "cpp" and a.no_transfer_mirror and not a.inflight_array
    assert a.end_batch and a.apply_batched
    with pytest.raises(SystemExit):
        G.parse_args(base + F3T + NEW + ["--inflight-array"])         # f3ldx's refusal stays
    me = inspect.getsource(G.make_engine)
    for s in ('issue=getattr(a, "b2_issue", "object")', 'no_transfer_mirror=bool(getattr(a, "no_transfer_mirror", False))',
              'end_batch=bool(getattr(a, "end_batch", False)), **kw)', 'eng.end_batch = bool(getattr(a, "end_batch", False))',
              'eng.apply_batched = bool(getattr(a, "apply_batched", False))'):
        assert s in me, s
    ps = inspect.signature(OWN2.OwnResidency.__init__).parameters
    assert ps["end_batch"].default is False and ps["issue"].default == "object"
    assert ps["no_transfer_mirror"].default is False
    er = inspect.getsource(OWN2.OwnResidency.end_requests)
    assert "if self._hm is not None:" in er and "for b in bs:" in er and JOIN in er
    assert JOIN in inspect.getsource(OWN2.OwnResidency.end_request)


# =====================================================================================================================
# TL4 the cells (static)
# =====================================================================================================================
def _reg(text):
    reg = text[text.index("# REGISTERED for evalfast2tall / evalfast2c3tall / evalacc216tall"):]
    return reg[:reg.index("\n#\n")]


def _bands(text):
    ln = next(x for x in text.splitlines() if x.strip().startswith("TALL_BANDS="))
    return ln.split("=", 1)[1].split()[0]


def test_TL4_the_cells_static():
    text = HF.STEP.read_text()
    lines = text.splitlines()
    assert next(ln for ln in lines if ln.startswith("NOSI_EVAL=")).split()[0] == "NOSI_EVAL=" + TALL_ROOT
    assert TALL_ROOT == HF.HF_ROOT
    assert "|evalfast2f3c|evalfast2c3c|evalfast2tall|evalfast2c3tall|evalacc216tall] [gilfree|gather]" in text
    assert "        evalfast2tall|evalfast2c3tall) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    assert "        evalacc216tall) [ -z \"$CELL\" ] || usage; CELL=$w ;;" in text
    ft = F3._cell(text, "    evalfast2tall) MODES=fast")
    for w in ("BATCH=216", "PX=1512 ; PRV=1464", "VARIANTS=fused/batched", "NOSI=$NOSI_EVAL", "GF3=1", "TALL_BANDS=",
              '[ "${DRYRUN:-0}" = 1 ] || [ "$TALL_BANDS" != TBD ]', "--eval-bands TBD (the lead fills and stamps",
              "--layout pool --X-pool 1512 --arrival-lag 2 --bare --repeat-docs --log-slots -1 --skip-shipped",
              "--eval-role fast-time --mem-cap-gb 84", '--eval-bands "$TALL_BANDS" --b2-resolve fused --fused-kernel v3',
              "--wait-index --b2-fewer-reads --res-lean --res-dev --worker cpp --b2-issue desc --no-transfer-mirror",
              "--end-batch --apply-batched) ;;"):
        assert w in ft, w
    assert "--inflight-array" not in ft and "STEP_MIN=" not in ft and "STEPS=" not in ft
    c3 = F3._cell(text, "    evalfast2c3tall) MODES=fast")
    for w in ("STEPS=8 ; DTICKS=4", "BATCH=216", "GF3=1",
              "--b2-resolve check --fused-kernel v3 --res-lean --res-dev --lean-time --worker cpp --b2-issue desc",
              "--no-transfer-mirror --end-batch --apply-batched) ;;"):
        assert w in c3, w
    for w in ("--bare", "--eval-role", "--accuracy-record", "--eval-bands", "--inflight-array"):
        assert w not in c3, w
    ac = F3._cell(text, "    evalacc216tall) MODES=fast,bi_c_strict")
    for w in ("BATCH=216", "ACC=1", "GF3=1", 'LINKARG=(--link-record "$OUTROOT/sg5_${RUN}_${JOB}_evalfast2tall/summary.json")',
              "--b2-resolve fused --fused-kernel v3", "--res-lean --res-dev --worker cpp --b2-issue desc --no-transfer-mirror "
              "--end-batch --apply-batched", "--store-committed-tokens --serial-rows 8 --gold-workload \"$GOLD\" "
              "\"${LINKARG[@]}\""):
        assert w in ac, w
    for w in ("--eval-role", "--accuracy-record", "--mem-cap-gb", "--eval-bands", "--bare", "STEP_MIN=", "STEPS=",
              "--inflight-array"):
        assert w not in ac, w
    # the same-job controls keep their arguments (their own tests: test_r2_f3ld.py TO5, test_r2_f3all.py FA7,
    # test_r2_f3lia.py F3G)
    assert "F3FLAGS=(--worker cpp --b2-issue desc --no-transfer-mirror)" in text
    assert "C3FLAGS=(--lean-time --worker cpp --b2-issue desc --no-transfer-mirror)" in text
    assert "--wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array --end-batch --apply-batched) ;;" in text
    assert "--wait-index --b2-fewer-reads --res-lean --res-dev --inflight-array) ;;" in text
    for word in ("awk", "perl", "python3", " bc ", "node "):
        for ln in lines:
            if not ln.lstrip().startswith("#"):
                assert word not in ln, (word, ln)


def test_TL4_the_registration_is_left_to_the_lead():
    text = HF.STEP.read_text()
    reg, bands = _reg(text), _bands(text)
    for tag in ("C3TALL", "TALL-ACT", "TALL-MED", "TALL-MEAN", "TALL-EFF", "TALL-HEND", "TALL-SAME", "TALL-FALS",
                "TALL-MEM", "ACCTALL", "nosi-r2-f3tall", "3a7f51a", "3769b64", "evalfast2f3t", "evalfast2all",
                "evalfast2f3lia", "0 diffs", "_AB_acted", "desc acted", "desc_jobs", "VOID", "_hm[b] = -1"):
        assert tag in reg, tag
    preds = [ln for ln in reg.splitlines() if "PREDICTION:" in ln]
    assert len(preds) >= 10, preds
    if not re.search(r"STAMP \d{4}-\d\d-\d\dT", reg):              # the builder's state: nothing predicted
        assert bands == "TBD", bands
        assert all(ln.rstrip().endswith("PREDICTION: TBD by lead") for ln in preds), preds
    else:                                                             # the lead filled and stamped it
        assert "TBD" not in reg and bands != "TBD", "a stamped block with a TBD left"
        assert re.fullmatch(r"[0-9.]+(:[0-9.]+){5}", bands) and bands in reg, bands


def test_TL4_the_step_refuses_TBD_bands_outside_dryrun(tmp_path):
    text = HF.STEP.read_text()
    if _bands(text) != "TBD":
        pytest.skip("TALL_BANDS is filled (test_TL4_the_registration_is_left_to_the_lead checks the stamp)")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_", "DRYRUN"))}
    env.update(SG5_HOLDERS=str(tmp_path))
    r = subprocess.run(["bash", str(HF.STEP), "--nosi", "0" * 40, "418", "t1", "evalfast2tall"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, (r.returncode, r.stderr[-300:])
    assert "cell evalfast2tall: --eval-bands TBD" in r.stderr, r.stderr[-300:]
    assert "r2_tick_sg5.py" not in r.stdout


# =====================================================================================================================
# TL5 the DRYRUN argv of the new cells
# =====================================================================================================================
TL_CELLS = dict(evalfast2tall="evalfast2f3t", evalfast2c3tall="evalfast2c3t", evalacc216tall="evalacc216all")


def _norm(got, cell):
    toks = [t.replace("_evalfast2tall/", "_LINK/").replace("_evalfast2all/", "_LINK/") for t in FA._norm(got, cell)]
    return toks


def _drop_run(toks, run):
    """toks without the ONE contiguous occurrence of run (it must occur once)."""
    at = [i for i in range(len(toks) - len(run) + 1) if toks[i:i + len(run)] == run]
    assert len(at) == 1, (run, at)
    return toks[:at[0]] + toks[at[0] + len(run):]


@pytest.mark.parametrize("cell", sorted(TL_CELLS))
def test_TL5_the_cells_dryrun(tmp_path, cell):
    if HF._clean_head() is None:
        pytest.skip("not a clean git checkout (tracked changes or untracked files in nosi/): the DRYRUN needs one")
    base = TL_CELLS[cell]
    r, line, got = FA._argv(tmp_path / "a", cell)
    _, _, bgot = FA._argv(tmp_path / "b", base)
    toks, btoks = _norm(got, cell), _norm(bgot, base)
    assert [toks.count(f) for f in NEW] == [1, 1], (cell, toks)
    if cell == "evalacc216tall":                                      # base evalacc216all: --inflight-array -> f3t
        assert "--inflight-array" not in toks and btoks.count("--inflight-array") == 1
        assert _drop_run(toks, F3T) == [t for t in btoks if t != "--inflight-array"], cell
    else:
        assert not any(f in btoks for f in NEW)
        assert [t for t in toks if t not in NEW] == btoks, (cell, [t for t in toks if t not in btoks],
                                                            [t for t in btoks if t not in toks])
    for s in ("PYTHONPATH=%s/nosi" % TALL_ROOT, "--job-name=sg5-t1-418-%s " % cell, "/r2-runs/sg5/sg5_t1_418_%s " % cell,
              "--time=120", "--mem=240G"):
        assert s in line, (cell, s)
    assert "[f3ld] packer copied: %s (packer v3 of --nosi, DESC 2)" % HF._clean_head() in r.stdout
    if cell == "evalfast2tall":
        bands = _bands(HF.STEP.read_text())
        assert "--eval-bands %s" % bands in got, (cell, bands)
        if bands == "TBD":
            with pytest.raises(SystemExit):                           # the harness refuses the placeholder too
                HF._sg5().parse_args(got.split())
            got = got.replace("--eval-bands TBD", "--eval-bands " + FA.STANDIN_BANDS)
    pa = HF._sg5().parse_args(got.split())
    assert pa.end_batch and pa.apply_batched and pa.rebuild == "batched", cell
    assert pa.b2_issue == "desc" and pa.worker == "cpp" and pa.no_transfer_mirror and not pa.inflight_array, cell
    assert pa.res_lean and pa.res_dev and pa.fused_kernel == "v3" and pa.arrival_lag == 2 and pa.lean_time, cell
    assert (pa.batch, pa.X_pool, pa.layout) == (216, 1512, "pool"), cell
    if cell == "evalfast2tall":
        assert pa.b2_resolve == "fused" and pa.eval_role == "fast-time" and pa.bare and pa.steps == 48
        assert pa.mem_cap_gb == 84 and pa.host_profile and pa.gc_freeze and not pa.store_committed_tokens
    elif cell == "evalacc216tall":
        assert pa.b2_resolve == "fused" and pa.modes == ["fast", "bi_c_strict"] and pa.store_committed_tokens
        assert pa.eval_role == "" and not pa.bare and pa.mem_cap_gb == 0 and not pa.accuracy_record
        assert pa.link_record == "/mnt/beegfs/ojerbi/r2-runs/sg5/sg5_t1_418_evalfast2tall/summary.json"
        assert "[acc216] modes fast,bi_c_strict" in r.stdout
    else:
        assert pa.b2_resolve == "check" and pa.eval_role == "" and pa.steps == 8 and pa.draft_ticks == 4
        assert not pa.bare and not pa.accuracy_record
