"""CPU test of rung I3 with the REAL fork kernel (benchmarks/Efficiency/fork_vllm_unified_attention.py, the byte copy of
fia6's) under the Triton interpreter, as tests_cpu/test_wsc_interp.py does for WS-C. It runs in a FRESH child process with
TRITON_INTERPRET=1 (Triton reads the flag at its first import, lessons L36), on CPU float32. This file needs ITS OWN srun
(the interpreter runs many minutes); tests_cpu/test_i3.py runs the same claims with the torch stand-in.

THE GEOMETRY (test_wsc_interp.py's): the child sets ragged_world's HQ 8 and D 16 before it builds anything, so BLOCK_M =
4 rows x 4 query heads per KV head = 16 (the real launcher's minimum, fork :862).
THE BIAS CONVENTION: the real kernels compute (q.k + bias) x scale; the CPU world's stand-ins q.k x scale + bias. The child
switches SH (wsc_world.use_kernel_bias) AND TentState's own one-row attention (i3_world.use_kernel_entry) to the kernel
convention, so every attention of a run uses one convention.

K1  the real fork and the stand-in (wsc_world.make_fake_fork(16, "kernel")) on the SAME inputs of L4PTF teacher passes
    (wsc_world.TeeFork): the union tables (entries below the length) and the lengths equal; outputs within 1e-4.
K2  the REAL fork in L4PTF's teacher-forced joint passes against SH (B 4, S 4925: pass 0 holds an F4 row and two rows past
    the block edge at 4928): every row's selection set = SH's; dropped 0, unserved 0; VIS, VB 0 from the tables the REAL
    kernel wrote; TAIL 0; CHAIN 0; the "F" / "RA" probes bitwise; argmax = SH's, delta <= 1e-4.
K3  the plant window_tail with the real fork: TAIL and VB fail (DETECTED).
K4  L4PTF on the schedule of contract 4 with the REAL fork (B 4, N 24 from S 4910: rejects, a block edge): SH's tokens,
    exactly N per request, logits within 1e-4; VB, TAIL, CHAIN, PRESENT 0.

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_i3_interp.py
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
I3 = load("i3_world", os.path.join(HERE, "i3_world.py"))
Z = I3.Z
Z.use_kernel_bias()                                 # SH's stand-in with the REAL kernels' (q.k + bias) x scale
I3.use_kernel_entry()                               # TentState's one-row attention too
F = load("fork_real", sys.argv[2])
real = F.unified_attention
fails, ran = [], []
TOL = 1e-4


def check(name, ok, detail=""):
    ran.append(name)
    if not ok:
        fails.append((name, str(detail)[:400]))


def zero_counters(st, keys):
    return all(st[k] == 0 for k in keys)


# K1 the real fork against the stand-in on the same inputs (two L4PTF teacher passes)
tee = Z.TeeFork(Z.make_fake_fork(16, "kernel"), real)
ev1 = I3.run_teacher_f(4, G=2, S=4925, fork=tee, probe=False)
check("K1_calls", len(tee.diffs) >= 2 * W.L, len(tee.diffs))
check("K1_tables_equal", all(tee.tables_equal), tee.tables_equal)
check("K1_outputs_close", max(tee.diffs) <= TOL, max(tee.diffs))

# K2 the REAL fork in L4PTF's teacher passes against SH
ev = I3.run_teacher_f(4, G=2, S=4925, fork=real)
st = ev["stats"]
check("K2_preconditions", st["past_edge"] > 0 and st["f4_rows"] > 0 and ev["past_rows"] > 0, st)
check("K2_sel", ev["changed"] == [], ev["changed"][:8])
check("K2_present", st["dropped"] == 0 and st["unserved"] == 0, st)
check("K2_vis_vb", st["vis_bad"] == 0 and st["vb_bad"] == 0 and st["vb_rows"] == 2 * W.L * W.HKV * 4 * 4, st)
check("K2_tail_chain", st["tail_reads"] == 0 and st["chain_bad"] == 0 and st["chain_rows"] > 0, st)
check("K2_block0", st["block0_bad"] == 0, st)
fa = ev["fa"]
check("K2_probe_F", bool(fa["F"]["entries"]) and all(fa["F"]["equal"]), fa)
check("K2_probe_RA", bool(fa["RA"]["entries"]) and all(fa["RA"]["equal"]), fa)
check("K2_logits", ev["argmax_bad"] == [] and ev["max_delta"] <= TOL, (ev["argmax_bad"][:5], ev["max_delta"]))
check("K2_calls", st["calls"]["attn"] == 2 * W.L and st["calls"]["fills"] == 2 * W.L, st["calls"])
check("K2_handle_none", F.FORK_LAST_LAUNCH["kernel"] is None)          # the interpreter gives no compiled kernel

# K3 the window_tail plant with the real fork
ev3 = I3.run_teacher_f(4, G=2, S=4925, fork=real, fneg="window_tail", probe=False)
st3 = ev3["stats"]
check("K3_window_tail_DETECTED", st3["tail_reads"] > 0 and st3["vb_bad"] > 0, st3)

# K4 the schedule of contract 4 with the REAL fork
FILTER = I3.SW.corrupt(4, 1, kinds=("draft",), phase=2)
ev4 = I3.run_tentf(4, build=I3.tentf_builder(fork=real), draft_filter=FILTER, with_reference=False)
u = ev4["rd"]["union"]
f4 = ev4["facts"]
check("K4_preconditions", f4["rejects"] > 0 and f4["full"] > 0 and f4["crossed"] > 0 and u["chain_reads"] > 0, (f4, u))
check("K4_count", ev4["rd"]["count"] == [I3.N0] * 4, ev4["rd"]["count"])
check("K4_tokens", ev4["bad_tokens"] == [] and ev4["max_logit_delta"] <= TOL, (ev4["bad_tokens"][:3],
                                                                                 ev4["max_logit_delta"]))
check("K4_counters", zero_counters(u, ("dropped", "unserved", "vis_bad", "vb_bad", "tail_reads", "chain_bad")), u)
check("K4_calls", u["calls"]["attn"] == W.L * u["passes"] and u["passes"] > 0, u["calls"])

print("RAN %d: %s" % (len(ran), " ".join(ran)))
if fails:
    print("FAILS", fails)
    sys.exit(1)
print("I3-INTERP OK")
'''


def test_i3_real_fork_under_the_triton_interpreter(tmp_path):
    """K1-K4 (module docstring), in a fresh child process with TRITON_INTERPRET=1 (skipped without triton)."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton is not installed")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", CHILD, str(HERE), str(FORK)], env=env, cwd=str(ROOT),
                       capture_output=True, text=True, timeout=10800)
    print(r.stdout[-3000:])
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1:] == ["I3-INTERP OK"], (
        "child exit %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
