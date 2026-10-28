"""CPU test of WS-C with the REAL fork kernel (benchmarks/Efficiency/fork_vllm_unified_attention.py, the byte copy of
fia6's) under the Triton interpreter. It runs in a FRESH child process with TRITON_INTERPRET=1 (Triton reads the flag
at its first import, lessons L36), on CPU float32, as test_r2_fia4.py / test_r2_fia5.py of nosi-r2-fia6 did. This file
needs ITS OWN srun (minutes of interpreter time); tests_cpu/test_wsc.py runs the same claims with the torch stand-in.

THE GEOMETRY. The real launcher needs BLOCK_M = rows x query heads per KV head >= 16 (tl.dot's minimum, fork :862) and
a head dim the tl.dot accepts: the child sets ragged_world's HQ 8 and D 16 (fia5's GEOM4: Hq 8, Hkv 2, D 16) before it
builds anything (2 query heads per KV head, the default world, gives BLOCK_M 8: J0 shows the refusal).

J0  the real launcher refuses BLOCK_M 8 (the default CPU world): why test_wsc.py uses the stand-in.
J1  the real fork and the stand-in (make_fake_fork(16)) on the SAME inputs of union fork passes (TeeFork): the union
    tables (entries below the length) and the lengths equal; the outputs within 1e-4 relative (fia5's interpreter
    tolerance).
J2  the REAL fork in the joint pass against SH (B 4, 3 passes): every row's selection set = SH's; dropped 0, unserved 0;
    VIS and VB 0 from the tables the REAL kernel wrote; G_block0; the "F" (request alone) and "RA" (row alone) probes
    bitwise with the real kernel; argmax = SH's, delta <= 1e-4; preconditions: rows past a block edge, F4 rows.
J3  the plant fork_union_rows with the real fork: VB and VIS fail (DETECTED).
J4  R 5 (two fork sequences per request: 4 + 1 rows) with the real fork: SH's selections, VB 0, the probes bitwise.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_wsc_interp.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FORK = ROOT / "benchmarks/Efficiency/fork_vllm_unified_attention.py"

CHILD = r'''
import importlib.util, os, sys, types
assert os.environ.get("TRITON_INTERPRET") == "1", "child: TRITON_INTERPRET must be 1 from process start"
assert "triton" not in sys.modules, "child: triton imported before the flag was read"
sys.dont_write_bytecode = True
import torch

HERE = sys.argv[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


W = load("ragged_world", os.path.join(HERE, "ragged_world.py"))
W.HQ, W.D = 8, 16                                    # fia5's GEOM4: BLOCK_M = 4 x 4 = 16, head dim 16
W.CFG = types.SimpleNamespace(num_hidden_layers=W.L, num_key_value_heads=W.HKV, head_dim=W.D)
Z = load("wsc_world", os.path.join(HERE, "wsc_world.py"))
F = load("fork_real", sys.argv[2])
real = F.unified_attention
fails, ran = [], []


def check(name, ok, detail=""):
    ran.append(name)
    if not ok:
        fails.append((name, str(detail)[:400]))


S4 = [4857, 4923, 4860, 4795]                       # block edges inside 3 passes of 4 rows (F4 rows, past-edge rows)

# J0 the real launcher refuses BLOCK_M 8 (2 query heads per KV head)
q = torch.zeros(4, 4, 16)
k = torch.zeros(8, 64, 2, 16)
tab = torch.zeros((2, 1, 256), dtype=torch.int32)
lists = dict(page_key=torch.zeros((4, 2, 64), dtype=torch.int32), page_table=torch.zeros((4, 2, 64), dtype=torch.int32),
             num_pages=torch.zeros((4, 2), dtype=torch.int32), last_len=torch.zeros(4, dtype=torch.int32),
             seq_idx=torch.zeros(4, dtype=torch.int32), slots_per_entry=1)
try:
    real(q, k, k, torch.zeros_like(q), torch.tensor([0, 4], dtype=torch.int32), 4, torch.zeros(1, dtype=torch.int32), 64,
         0.25, True, (-1, -1), tab, 0, None, None, None, union_ext=torch.zeros_like(tab),
         union_len=torch.zeros((2, 1), dtype=torch.int32), cis_bias=torch.zeros(8, 64, 2), build_union=lists,
         rows_per_q_block=4, num_warps=4, num_stages=3, tile_size=64)
    check("J0_block_m_8_refused", False, "no refusal")
except AssertionError:
    check("J0_block_m_8_refused", True)

# J1 the real fork against the stand-in on the same inputs (two union fork passes)
tee = Z.TeeFork(Z.make_fake_fork(16), real)
ev1 = Z.run_fork_vs_sh(S4, 2, fork=tee, probe=False)
check("J1_calls", len(tee.diffs) >= 2 * W.L, len(tee.diffs))
check("J1_tables_equal", all(tee.tables_equal), tee.tables_equal)
check("J1_outputs_close", max(tee.diffs) <= 1e-4, max(tee.diffs))

# J2 the REAL fork in the joint pass against SH
ev = Z.run_fork_vs_sh(S4, 3, fork=real)
st = ev["stats"]
check("J2_preconditions", st["past_edge"] > 0 and st["f4_rows"] > 0 and bool(ev["fills"]), st)
check("J2_sel", ev["changed"] == [], ev["changed"][:8])
check("J2_present", st["dropped"] == 0 and st["unserved"] == 0, st)
check("J2_vis_vb", st["vis_bad"] == 0 and st["vb_bad"] == 0 and st["vb_rows"] == 3 * W.L * W.HKV * 4 * 4, st)
check("J2_block0", st["block0_bad"] == 0, st)
fa = ev["vs"].fa_check
check("J2_probe_F", bool(fa["F"]["entries"]) and all(fa["F"]["equal"]), fa)
check("J2_probe_RA", bool(fa["RA"]["entries"]) and all(fa["RA"]["equal"]), fa)
check("J2_logits", ev["argmax_bad"] == [] and ev["max_delta"] <= 1e-4, (ev["argmax_bad"][:5], ev["max_delta"]))
check("J2_calls", st["calls"]["attn"] == 3 * W.L and st["calls"]["fills"] == 3 * W.L, st["calls"])
check("J2_handle_none", F.FORK_LAST_LAUNCH["kernel"] is None)          # the interpreter gives no compiled kernel

# J3 the union plant with the real fork
ev3 = Z.run_fork_vs_sh(S4, 3, fork=real, neg="fork_union_rows", probe=False)
st3 = ev3["stats"]
check("J3_precondition", st["union_max"] > W.TOPK, st["union_max"])
check("J3_union_rows_DETECTED", st3["vb_bad"] > 0 and st3["vis_bad"] > 0, st3)

# J4 R 5: two fork sequences per request (4 + 1 rows)
ev4 = Z.run_fork_vs_sh(S4, 2, K=4, fork=real)
st4 = ev4["stats"]
check("J4_chunks", ev4["vs"].fork_sizes == [4, 1] and ev4["vs"].fork_ns == 8, (ev4["vs"].fork_sizes, ev4["vs"].fork_ns))
check("J4_sel_vb", ev4["changed"] == [] and st4["vb_bad"] == 0 and st4["vis_bad"] == 0 and st4["dropped"] == 0, st4)
fa4 = ev4["vs"].fa_check
check("J4_probes", all(fa4["F"]["equal"]) and all(fa4["RA"]["equal"]), fa4)
check("J4_logits", ev4["argmax_bad"] == [] and ev4["max_delta"] <= 1e-4, ev4["max_delta"])

print("RAN %d: %s" % (len(ran), " ".join(ran)))
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("WSC-INTERP OK")
'''


def test_wsc_real_fork_under_the_triton_interpreter(tmp_path):
    """J0-J4 (module docstring), in a fresh child process with TRITON_INTERPRET=1 (skipped without triton)."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", CHILD, str(HERE), str(FORK)], env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["WSC-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
