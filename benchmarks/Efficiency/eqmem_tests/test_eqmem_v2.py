"""CPU tests of the equal-memory harness v2 (2026-10-20): per-step timings, host memory, the agreement reference, the
holder GPU uuid file, capacity scans. No GPU, no model, no engine import.

  V1  per-step split: split_step_windows / rewarm_unit_windows / step_record (refusals); decode_shipped's real marking
      code driven by a fake model and a fake torch: every decode_update window and its blocks land in their own step,
      re-warm steps flagged; NEGATIVE CONTROL: a step that skips a layer's window is refused (steps_error, the cell's
      other outputs intact)
  V2  host memory: /proc/self/status parsing, host_record (refused without VmHWM), the live reader on this node
  V3  the report keys the token agreement on the FIRST COMPLETED original repetition (rep 0 failed, rep 1 done);
      NEGATIVE CONTROL: no completed original -> no reference, no agreement
  V4  eqmem_step.sh: bash -n; read_holder_uuid (the function of the script, run by bash) on a good file, a missing
      file, an empty file, a GPU- prefixed and a malformed uuid; no uuid pinned in the script; the child's check stays
  V5  capacity scans: expand_cells (lists, ranges, refusals), plan mode with the step commands, capacity_scan on
      measured cells (interior vs edge), the report's scan table

From the worktree root:
    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -rA \\
        "benchmarks/Efficiency/eqmem_tests/test_eqmem_v2.py::<node>"
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

EFF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EFF))
sys.dont_write_bytecode = True

import eqmem_bench as EB  # noqa: E402
import eqmem_core as EC  # noqa: E402

STEP = EFF / "eqmem_step.sh"


# ------------------------------------------------------------------------------------------------------------- V1
def test_V1_split_step_windows_and_step_record():
    assert EC.split_step_windows([1, 2, 3, 4, 5, 6], [0, 2, 4, 6], 2) == [3.0, 7.0, 11.0]
    for bad in ([0, 2, 3, 6], [0, 2, 4], [1, 3, 5, 6], [0, 2, 4, 6, 8]):        # off by one, short, late start, past the end
        with pytest.raises(ValueError):
            EC.split_step_windows([1, 2, 3, 4, 5, 6], bad, 2)
    # spec: the pilot's warm-up step leads, then one step per re-warm round
    assert EC.rewarm_unit_windows([1, 1, 2, 2, 3, 3], 2, 2) == [4.0, 6.0]
    with pytest.raises(ValueError):
        EC.rewarm_unit_windows([1, 1, 2, 2, 3, 3], 1, 2)                          # a window left over
    rec = EC.step_record([20.0, 10.0, 12.0, 11.0], [5.0, 2.0, 3.0, 2.5], [10, 1, 2, 1], [1, 0, 0, 0])
    s = rec["summary"]
    assert rec["rest_ms"] == [15.0, 8.0, 9.0, 8.5] and rec["rewarm"] == [1, 0, 0, 0]
    assert s["n"] == 4 and s["rewarm_units"] == 1 and s["rewarm_ms"] == 20.0 and s["total_ms"] == 53.0
    assert s["steady_total"]["median"] == 11.0 and s["steady_wait"]["n"] == 3
    assert s["steady_wait_share"] == pytest.approx(7.5 / 33.0) and s["wait_share"] == pytest.approx(12.5 / 53.0)
    with pytest.raises(ValueError):
        EC.step_record([1.0], [2.0], [0], [0])                                    # the wait above its step
    with pytest.raises(ValueError):
        EC.step_record([1.0, 2.0], [0.5], [0, 0], [0, 0])


class _Clock:
    t = 0.0


class _Ev:
    def __init__(self, enable_timing=True):
        self.t = None

    def record(self, stream=None):
        self.t = _Clock.t

    def elapsed_time(self, other):
        return other.t - self.t


def _fake_torch():
    return types.SimpleNamespace(
        int=torch.int, int64=torch.int64, inference_mode=torch.inference_mode, cat=torch.cat, stack=torch.stack,
        arange=lambda *a, device=None, **k: torch.arange(*a, **k), zeros=lambda *a, device=None, **k: torch.zeros(*a, **k),
        cuda=types.SimpleNamespace(Event=_Ev, synchronize=lambda: None, mem_get_info=lambda: (10 ** 9, 8 * 10 ** 10),
                                   memory_reserved=lambda: 0, memory_allocated=lambda: 0))


class _Eng:
    WINDOW_MS = 2.0

    def __init__(self, layer):
        self.layer = layer
        self._load_mask = torch.full((2, 3, 4), -1, dtype=torch.int64)

    def decode_update(self, step):
        _Clock.t += self.WINDOW_MS * (1 + self.layer)       # layer l's window: 2 (l + 1) ms
        self._load_mask.fill_(-1)
        self._load_mask.view(-1)[: (step + self.layer) % 5] = 7  # (step + l) % 5 blocks fetched
        return None


class _Model:
    REST_MS = 10.0

    def __init__(self, cache, skip=None):
        self.cache, self.has_buffers, self.skip, self.step = cache, False, skip, 0

    def decode_inference(self, tok, cu, pos, cache):
        for l, lay in enumerate(cache.layers):
            if self.skip == (self.step, l):
                continue
            lay.cache_engine.decode_update(self.step)
        _Clock.t += self.REST_MS
        self.has_buffers = True
        self.step += 1
        B = tok.shape[0]
        lg = torch.zeros((B, 1, 5))
        lg[:, 0, (self.step % 5)] = 1.0
        return lg


def _run_decode(monkeypatch, N=7, nL=3, skip=None):
    _Clock.t = 0.0
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(cache_engine=_Eng(l)) for l in range(nL)])
    model = _Model(cache, skip)
    FT = _fake_torch()
    probe = EB.EngineProbe(FT, cache)
    probe.install()
    calls = []

    def live_rewarm(m, c, look):
        calls.append(m.step)
        return (m.step in (0, 3)), ("first" if m.step == 0 else "out_len")
    monkeypatch.setattr(EB, "live_rewarm", live_rewarm)
    t0 = torch.zeros((2, 1), dtype=torch.int64)
    dec = EB.decode_shipped(FT, model, cache, t0, torch.zeros((2, 1), dtype=torch.int64), N, [], probe)
    return dec, nL


def test_V1b_decode_shipped_charges_every_window_to_its_own_step(monkeypatch):
    dec, nL = _run_decode(monkeypatch)
    st = dec["steps"]
    assert dec["steps_error"] is None and st["n"] == 6 and st["unit"] == "step"
    want_wait = sum(_Eng.WINDOW_MS * (1 + l) for l in range(nL))                     # 2 + 4 + 6 = 12 ms per step
    assert st["wait_ms"] == [want_wait] * 6 and st["step_ms"] == [want_wait + _Model.REST_MS] * 6
    assert st["rest_ms"] == [_Model.REST_MS] * 6
    assert st["blocks"] == [sum((s + l) % 5 for l in range(nL)) for s in range(6)]
    assert st["rewarm"] == [1, 0, 0, 1, 0, 0] and st["summary"]["rewarm_units"] == 2
    assert dec["unit_ms"] == st["step_ms"] and [r["step"] for r in dec["rewarms"]] == [0, 3]


def test_V1c_negative_control_a_missing_window_is_refused_not_misattributed(monkeypatch):
    dec, nL = _run_decode(monkeypatch, skip=(2, 1))                                # step 2 skips layer 1's update
    assert dec["steps"] is None and "step 2 holds 2 windows, expected 3" in dec["steps_error"]
    assert len(dec["unit_ms"]) == 6 and dec["seq"].shape == (2, 7)                  # the cell's other outputs stand


# ------------------------------------------------------------------------------------------------------------- V2
def test_V2_host_memory_record():
    txt = "Name:\tpython\nVmPeak:\t 9000 kB\nVmHWM:\t  204800000 kB\nVmRSS:\t  150000000 kB\nVmPin:\t 0 kB\nVmLck:\t 1 kB\n"
    st = EC.parse_proc_status(txt)
    assert st == dict(VmRSS=150000000, VmHWM=204800000, VmPin=0, VmLck=1)
    samples = [dict(label="start", host_rss_gb=1.0), dict(label="end", host_rss_gb=150.0)]
    h = EC.host_record(samples, st, 204700000, 166 * 10 ** 9)
    assert h["rss_peak_gb"] == pytest.approx(204.8) and h["ru_maxrss_gb"] == pytest.approx(204.7)
    assert h["pinned_kv_gb"] == pytest.approx(166.0) and h["vm_pin_gb"] == 0.0 and h["rss_samples"][1]["rss_gb"] == 150.0
    with pytest.raises(ValueError):
        EC.host_record(samples, EC.parse_proc_status("VmRSS:\t 5 kB\n"), None, 0)    # no peak: refused
    live = EB.proc_status()                                                          # this node's /proc
    assert live["VmHWM"] is not None and live["VmHWM"] >= live["VmRSS"] > 0


# ------------------------------------------------------------------------------------------------------------- V3
def _result(cell, family, rep, status, e2e, B, P=0):
    return dict(cell=cell, family=family, rep=rep, status=status, plan_key=[75.0, 1.0, 512, 16128],
                plan=dict(budget_gb=75.0, N=512, P=P), memory=dict(verdict="VALID", used_peak_est_gb=70.0),
                metrics=dict(batch=B, e2e_tok_s=e2e, decode_tok_s=e2e * 6, decode_tok_s_gpu_units=e2e * 6, e2e_s=900.0,
                             transferred=dict(bytes_per_committed_token=1e6), data_wait=dict(share_of_decode=0.1),
                             latency_s=dict(end_to_end=900.0)))


def _write_run(run, cells):
    for name, r, seq in cells:
        d = run / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(r))
        if seq is not None:
            torch.save(dict(seq=seq, rows=list(range(seq.shape[0]))), d / "tokens.pt")


def test_V3_the_agreement_reference_is_the_first_completed_original(tmp_path):
    seq = torch.arange(4 * 6).view(4, 6) % 7
    other = seq.clone()
    other[2, 4] += 1
    run = tmp_path / "run"
    _write_run(run, [("original_rep0", _result("original", "original", 0, "error", 0.0, 4), None),       # OOM-killed: no tokens
                     ("original_rep1", _result("original", "original", 1, "done", 160.0, 4), seq),
                     ("cached_p33_rep0", _result("cached_p33", "cached", 0, "done", 170.0, 3, P=33), other[:3])])
    out = tmp_path / "rep"
    assert EB.main(["--mode", "report", "--results", str(run), "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["agreement_reference"] == "original_rep1" and "agreement_vs_original_rep0" not in rep
    ag = rep["agreement_vs_original"]
    assert set(ag) == {"cached_p33_rep0"} and ag["cached_p33_rep0"]["full_agreement"] == 2 and ag["cached_p33_rep0"]["prefix_min"] == 4
    assert rep["first_completed_rep"] == {"original": 1, "cached_p33": 0}
    assert "token agreement reference (the first completed original repetition): original_rep1" in (out / "report.md").read_text()
    # NEGATIVE CONTROL: no completed original repetition -> no reference and no agreement (rep 0 is not assumed)
    run2 = tmp_path / "run2"
    _write_run(run2, [("original_rep0", _result("original", "original", 0, "error", 0.0, 4), None),
                      ("cached_p33_rep0", _result("cached_p33", "cached", 0, "done", 170.0, 3, P=33), other[:3])])
    out2 = tmp_path / "rep2"
    assert EB.main(["--mode", "report", "--results", str(run2), "--out", str(out2)]) == 0
    rep2 = json.loads((out2 / "report.json").read_text())
    assert rep2["agreement_reference"] is None and rep2["agreement_vs_original"] == {}
    assert EC.agreement_reference([("original", 2, "done", True), ("original", 1, "done", False), ("original", 3, "done", True)]) == ("original", 2)


# ------------------------------------------------------------------------------------------------------------- V4
def _holder_fn():
    src = STEP.read_text()
    return src[src.index("read_holder_uuid() {"):src.index("\nmain() {")]


def _read_uuid(d, job):
    p = subprocess.run(["bash", "-c", _holder_fn() + '\nread_holder_uuid "$1" "$2"', "t", str(d), job],
                       capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr


def test_V4_the_step_script_reads_the_holder_uuid_file_and_stops_without_it(tmp_path):
    src = STEP.read_text()
    assert subprocess.run(["bash", "-n", str(STEP)]).returncode == 0
    # the merged harness (feature/nosi-ctxsweep-v2) runs 4.1 and 4.2 from one worktree
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-ctxsweep-v2 " in src and "HOLDERS=/mnt/beegfs/ojerbi/kstep-cpu-runs/holders " in src
    assert 'GPU_UUID=$(read_holder_uuid "$HOLDERS" "$JOB") || exit 1' in src
    assert "f31fc723" not in src and "GPU_UUID=f" not in src                       # no uuid pinned in the script
    assert '[ "$seen" = "GPU-$U" ]' in src and '"$GPU_UUID" "$APPTAINER"' in src      # the child still checks the visible GPU
    assert src.index('GPU_UUID=$(read_holder_uuid') < src.index('if [ "$DRY" = 1 ]')  # before DRYRUN and before any claim
    u = "f31fc723-9dfe-407a-9947-28173556232a"
    (tmp_path / "123.gpu_uuid").write_text(u + "\n")
    assert _read_uuid(tmp_path, "123") == (0, u, "")
    (tmp_path / "124.gpu_uuid").write_text(u)                                       # no trailing newline: accepted
    assert _read_uuid(tmp_path, "124")[:2] == (0, u)
    rc, out, err = _read_uuid(tmp_path, "999")                                      # NEGATIVE CONTROLS
    assert rc == 1 and out == "" and "STOP: no" in err
    for job, text in (("125", ""), ("126", "GPU-" + u + "\n"), ("127", "f31fc723-9dfe\n"), ("128", u.upper() + "\n")):
        (tmp_path / ("%s.gpu_uuid" % job)).write_text(text)
        rc, out, err = _read_uuid(tmp_path, job)
        assert rc == 1 and out == "" and "STOP" in err, (job, rc, out, err)


# ------------------------------------------------------------------------------------------------------------- V5
def test_V5_capacity_scan_cells_plans_commands_and_the_measured_optimum(tmp_path):
    names = [c.name for c in EC.expand_cells("original, cached:10/16/22, scan:30-40:5, spec:2")]
    assert names == ["original", "cached_p10", "cached_p16", "cached_p22", "cached_p30", "cached_p35", "cached_p40", "spec_k2_r62"]
    for bad in ("", "scan:10-5:2", "scan:0-10:2", "scan:10-20:0", "scan:10-200:5", "cached:10/x", "cached:10,cached:10",
                "cached:10/10", "cached:161"):
        with pytest.raises(ValueError):
            EC.expand_cells(bad)
    out = tmp_path / "plan.json"
    sha = "f" * 40
    assert EB.main(["--mode", "plan", "--budget-gb", "75", "--cells", "original,cached:10/16", "--step-nosi", sha,
                    "--step-holder", "2190001", "--step-run", "h50", "--step-reps", "0,1", "--out", str(out)]) == 0
    pl = json.loads(out.read_text())
    assert pl["cells"] == ["original", "cached_p10", "cached_p16"] and len(pl["plans"]) == 3
    assert pl["commands"][0] == "bash benchmarks/Efficiency/eqmem_step.sh --nosi %s 2190001 h50 75 original 0" % sha
    assert pl["commands"][5] == "bash benchmarks/Efficiency/eqmem_step.sh --nosi %s 2190001 h50 75 cached:16 1" % sha
    assert pl["commands"][-1].endswith(" h50 75 report") and len(pl["commands"]) == 7
    assert EB.main(["--mode", "plan", "--budget-gb", "75", "--cells", "spec:2:30", "--step-nosi", sha, "--step-holder", "1",
                    "--step-run", "h50", "--out", str(tmp_path / "p2.json")]) == 4                  # R 30: not a step cell
    assert EB.main(["--mode", "plan", "--budget-gb", "75", "--cells", "original", "--step-nosi", "abc", "--step-holder", "1",
                    "--step-run", "h50", "--out", str(tmp_path / "p3.json")]) == 4                  # a short sha
    assert EB.main(["--mode", "plan", "--budget-gb", "75", "--cells", "scan:9-5:1", "--out", str(tmp_path / "p4.json")]) == 4
    # the scan on measured cells: interior optimum, then an edge optimum (extend the scan)
    res = [_result("original", "original", 0, "done", 160.0, 304), _result("cached_p10", "cached", 0, "done", 165.0, 270, P=10),
           _result("cached_p16", "cached", 0, "done", 175.0, 256, P=16), _result("cached_p22", "cached", 0, "done", 170.0, 240, P=22)]
    cs = EC.capacity_scan(res)
    assert [t["P"] for t in cs["table"]] == [0, 10, 16, 22] and cs["best"]["P"] == 16 and cs["position"] == "interior"
    assert cs["best_vs_original"] == pytest.approx(175.0 / 160.0 - 1)
    cs2 = EC.capacity_scan(res[:2] + [_result("cached_p16", "cached", 0, "done", 150.0, 256, P=16)])
    assert cs2["best"]["P"] == 10 and cs2["position"] == "edge_low"
    run = tmp_path / "run"
    _write_run(run, [("%s_rep0" % r["cell"], r, None) for r in res])
    rep_out = tmp_path / "rep"
    assert EB.main(["--mode", "report", "--results", str(run), "--out", str(rep_out)]) == 0
    md = (rep_out / "report.md").read_text()
    assert "## capacity scan" in md and "measured best P 16 (cached_p16" in md and "interior of the scan" in md
