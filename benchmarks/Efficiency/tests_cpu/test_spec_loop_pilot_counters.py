"""R2 milestone 1 g1 (step 2188818.23): the pilot's final counters. PC1: final_counter_sums sums the per-(layer, head,
request) counter tensors into ints that equal the per-round convention (v.sum()); PC2 (negative control): the
pre-fix expression {k: int(v)} raises on the same store. Loads the pilot module by path with a stub engine import
guard (the pilot's top level needs no GPU for these helpers).

    python -m pytest -q -p no:cacheprovider -rA benchmarks/Efficiency/tests_cpu/test_spec_loop_pilot_counters.py
"""
import ast
from pathlib import Path

import pytest
import torch

PILOT = Path(__file__).resolve().parents[1] / "spec_loop_pilot.py"


def _helper():
    """Extract final_counter_sums from the pilot's source and execute it alone (the pilot module runs a benchmark on
    import)."""
    tree = ast.parse(PILOT.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "final_counter_sums")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(PILOT), "exec"), ns)
    return ns["final_counter_sums"]


class _Store:
    def __init__(self):
        self.c = {"prefetch_blocks": torch.arange(24).reshape(2, 3, 4), "late_fetch_blocks": torch.ones(2, 3, 4,
                                                                                                       dtype=torch.long)}

    def counters(self):
        return {k: v.clone() for k, v in self.c.items()}


def test_PC1_final_counter_sums():
    out = _helper()(_Store())
    assert out == {"prefetch_blocks": 276, "late_fetch_blocks": 24}
    assert all(type(v) is int for v in out.values())


def test_PC3_run_spec_arm_keeps_its_inference_mode_decorator():
    """The helper was first inserted between run_spec_arm and its @torch.inference_mode() decorator (found by PC1)."""
    tree = ast.parse(PILOT.read_text())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert [ast.unparse(d) for d in fns["run_spec_arm"].decorator_list] == ["torch.inference_mode()"]
    assert fns["final_counter_sums"].decorator_list == []


def test_PC2_the_prefix_expression_raises():
    with pytest.raises(ValueError):
        {k: int(v) for k, v in _Store().counters().items()}
