"""CPU tests of the R2 capture audit (docs/R2_CAPTURE_AUDIT.md): the source lint of the decode-step functions against
the retroinfer-eval AGENTS.md law 3 (inside a captured region: fixed shapes, fixed device addresses with in-place
writes only, no allocation, no host branch on a device value, no tensor[scalar_tensor] indexing, cannot raise), and
the bitwise proof that the capture-safe twin step equals the validated one.

Run on a CPU node, inside a job (never on the login node):

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2-capture python -m pytest -q tests/test_r2_capture.py

LINT CLASSES (lint_function; AST, per function, no imports of the linted code):
  SYNC        a host value of a device tensor: bool(), float(), .item(), .tolist(), .cpu(), .numpy(), synchronize(),
              int() of anything but static metadata (.shape, .stride(), block_size, topk, the tail slot, lk_max, n_max)
  SHAPE       a data-dependent shape: .nonzero(), .unique(), .masked_select(), .argwhere(), indexing with a boolean
              tensor (a name assigned from a comparison, &, |, ~ or a mask method); .numel() is metadata and counts
              only as a BRANCH when a test calls it
  BRANCH      a host branch (if / while / assert / conditional expression) whose test calls anything but static
              metadata, or reads a boolean / 0-d tensor name
  ALLOC       an explicit allocation: a torch factory (empty, zeros, ones, full, arange, tensor, as_tensor, *_like,
              rand*, randperm, linspace) or .clone()
  SCALARIDX   tensor[0-d tensor]: a reduction without dim (max, min, sum, argmax, argmin, prod, mean, any, all) as an
              index
  HOSTSTATE   host state written per step: an augmented assignment or an attribute rebinding
  CALLS_EAGER a call of a function that is not capture-safe: .step(, .decode(, .advance(, .update(, pooling_lengths(,
              body_update(, layer_decode(, decode_inference(
CAPTURED (must be clean): the functions that the captured step runs (CAPTURED below). EAGER (documented): the
validated eager functions, which must show AT LEAST the classes the audit lists for them (EAGER below). POSITIVE
CONTROLS: a planted function with one instance of every class, and planted regressions inside the real captured
functions, are each found.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import textwrap
from pathlib import Path

import pytest
import torch

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
N = NOSI_REPO / "nosi/nosi"
FILES = {name: N / (name + ".py") for name in ("r2_positional_twin", "r2_decode_body", "r2_writeback",
                                                "max_pooling_per_request", "stage1_seqused", "r2_decode_path",
                                                "r2_capture")}
if not all(p.is_file() for p in FILES.values()):
    pytest.skip("the R2 modules are not all under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)

CLASSES = ("SYNC", "SHAPE", "BRANCH", "ALLOC", "SCALARIDX", "HOSTSTATE", "CALLS_EAGER")
STATIC_CALLS = {"tuple", "isinstance", "len", "callable", "getattr", "hasattr", "type", "dim", "stride",
                "is_contiguous", "is_cuda", "is_pinned", "is_current_stream_capturing", "is_tensor", "size"}
INT_OK = re.compile(r"\.shape\b|\.stride\(|block_size|\btopk\b|_tail_block_idx_on_gpu|lk_max|n_max")
FACTORIES = {"empty", "zeros", "ones", "full", "arange", "tensor", "as_tensor", "empty_like", "zeros_like",
             "ones_like", "full_like", "rand", "randn", "randint", "randperm", "empty_strided", "linspace"}
SYNC_ATTRS = {"item", "tolist", "cpu", "numpy", "synchronize"}
SHAPE_ATTRS = {"nonzero", "unique", "masked_select", "argwhere"}   # .numel() is metadata: only a BRANCH on it counts
BOOL_ATTRS = {"_key_due", "_cis_due", "full_mask", "isnan", "isinf", "isfinite", "eq", "ne", "lt", "le", "gt", "ge",
              "logical_and", "logical_or", "logical_not"}
REDUCE_ATTRS = {"max", "min", "sum", "argmax", "argmin", "prod", "mean", "any", "all"}
EAGER_ATTRS = {"step", "decode", "advance", "update"}
EAGER_NAMES = {"pooling_lengths", "body_update", "layer_decode", "decode_inference"}


def _callname(c):
    f = c.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)


def _is_bool_expr(v, boolnames):
    if isinstance(v, ast.Compare):
        return True
    if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Invert):
        return True
    if isinstance(v, ast.BinOp) and isinstance(v.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return any(_is_bool_expr(s, boolnames) or (isinstance(s, ast.Name) and s.id in boolnames)
                   for s in (v.left, v.right))
    return isinstance(v, ast.Call) and _callname(v) in BOOL_ATTRS


def _is_scalar_reduction(v):
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr in REDUCE_ATTRS
            and not v.args and not v.keywords)


def lint_function(fn):
    """[(class, lineno, source)] for one FunctionDef node (nested functions included)."""
    found = []
    boolnames, scalarnames = set(), set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if _is_bool_expr(node.value, boolnames):
                boolnames.add(node.targets[0].id)
            if _is_scalar_reduction(node.value):
                scalarnames.add(node.targets[0].id)

    def add(cls, node):
        found.append((cls, getattr(node, "lineno", 0), ast.unparse(node)[:100]))

    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = _callname(node)
            literal = bool(node.args) and all(isinstance(x, ast.Constant) for x in node.args)
            if isinstance(node.func, ast.Name) and name in ("bool", "float") and not literal:
                add("SYNC", node)
            elif isinstance(node.func, ast.Name) and name == "int" and not literal and not (
                    node.args and INT_OK.search(ast.unparse(node.args[0]))):
                add("SYNC", node)
            if isinstance(node.func, ast.Attribute) and name in SYNC_ATTRS:
                add("SYNC", node)
            if isinstance(node.func, ast.Attribute) and name in SHAPE_ATTRS:
                add("SHAPE", node)
            if isinstance(node.func, ast.Attribute) and name in FACTORIES and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "torch":
                add("ALLOC", node)
            if isinstance(node.func, ast.Attribute) and name == "clone":
                add("ALLOC", node)
            if (isinstance(node.func, ast.Attribute) and name in EAGER_ATTRS) or \
                    (isinstance(node.func, ast.Name) and name in EAGER_NAMES):
                add("CALLS_EAGER", node)
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            for c in ast.walk(node.test):
                if (isinstance(c, ast.Call) and _callname(c) not in STATIC_CALLS) or \
                        (isinstance(c, ast.Name) and c.id in boolnames | scalarnames):
                    add("BRANCH", node.test)
                    break
        if isinstance(node, ast.Subscript):
            parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for e in parts:
                if isinstance(e, ast.Name) and e.id in boolnames:
                    add("SHAPE", node)
                if (isinstance(e, ast.Name) and e.id in scalarnames) or _is_scalar_reduction(e):
                    add("SCALARIDX", node)
        if isinstance(node, ast.AugAssign) or (isinstance(node, ast.Assign) and
                                               any(isinstance(t, ast.Attribute) for t in node.targets)):
            add("HOSTSTATE", node)
    return found


def _tree(name):
    return ast.parse(FILES[name].read_text(encoding="utf-8"))


def _fn(name, qual):
    """The FunctionDef `qual` ('func' or 'Class.method') of module `name`."""
    tree = _tree(name)
    if "." in qual:
        cls, meth = qual.split(".")
        c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == meth)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == qual)


CAPTURED = [
    ("r2_positional_twin", "PositionalTwin.step_fixed"), ("r2_positional_twin", "PositionalTwin.key_window"),
    ("r2_positional_twin", "PositionalTwin._key_due"), ("r2_positional_twin", "PositionalTwin._cis_due"),
    ("r2_decode_body", "DecodeBody.decode_fixed"), ("r2_decode_body", "DecodeBody.write_tail"),
    ("r2_decode_body", "DecodeBody.select"), ("r2_decode_body", "DecodeBody.full_mask"),
    ("r2_decode_body", "DecodeBody.cache_lens_value"), ("r2_decode_body", "DecodeBody.positions"),
    ("r2_decode_body", "DeviceWritebackBody.writeback"),
    ("r2_writeback", "masked_writeback"), ("r2_writeback", "_launch"), ("r2_writeback", "check_args"),
    ("max_pooling_per_request", "nosa_pooling_per_request"), ("max_pooling_per_request", "check_lengths"),
    ("stage1_seqused", "stage1_seqused"), ("stage1_seqused", "check_seqused"),
    ("r2_capture", "layer_decode_fixed"), ("r2_capture", "decode_step_fixed"),
]

EAGER = {   # the classes the audit documents for each validated eager function (the lint must find AT LEAST these)
    ("r2_positional_twin", "PositionalTwin.step"): {"SYNC", "SHAPE", "BRANCH", "ALLOC"},
    ("r2_decode_body", "DecodeBody.decode"): {"SYNC", "BRANCH"},
    ("r2_decode_body", "DecodeBody.writeback"): {"SYNC", "SHAPE"},
    ("r2_decode_path", "layer_decode"): {"ALLOC", "HOSTSTATE", "CALLS_EAGER"},
    ("r2_decode_path", "decode_inference"): {"HOSTSTATE", "CALLS_EAGER"},
    ("r2_decode_path", "pooling_lengths"): {"ALLOC", "BRANCH", "HOSTSTATE"},
    ("r2_decode_path", "body_update"): {"SYNC", "BRANCH", "CALLS_EAGER"},
    ("r2_decode_path", "R2Body.update"): {"CALLS_EAGER"},
    # the driver of the captured step: it runs on the host, OUTSIDE the captured region, by design
    ("r2_capture", "prepare_step"): {"ALLOC"},
    ("r2_capture", "finish_step"): {"HOSTSTATE"},
    ("r2_capture", "errors"): {"SYNC"},
}


@pytest.mark.parametrize("mod,qual", CAPTURED, ids=["%s.%s" % c for c in CAPTURED])
def test_L1_captured_functions_are_clean(mod, qual):
    found = lint_function(_fn(mod, qual))
    assert found == [], "%s.%s: %s" % (mod, qual, found)


@pytest.mark.parametrize("mod,qual", sorted(EAGER), ids=["%s.%s" % c for c in sorted(EAGER)])
def test_L2_eager_functions_show_their_documented_classes(mod, qual):
    classes = {c for c, _, _ in lint_function(_fn(mod, qual))}
    assert EAGER[(mod, qual)] <= classes, "%s.%s: documented %s, found %s" % (mod, qual, EAGER[(mod, qual)], classes)


PLANTED = textwrap.dedent('''
def planted(self, x, mask_src, st):
    y = x.sum().item()                     # SYNC
    idx = x.nonzero()                      # SHAPE
    m = x > 0
    z = x[m]                               # SHAPE (boolean mask)
    if x.any():                            # BRANCH
        pass
    t = torch.empty(3)                     # ALLOC
    s = x.max()
    w = x[s]                               # SCALARIDX
    st.steps += 1                          # HOSTSTATE
    self.twin.step(x, x, mask_src)         # CALLS_EAGER
''')


def test_L3_positive_control_every_class_is_found():
    fn = ast.parse(PLANTED).body[0]
    assert {c for c, _, _ in lint_function(fn)} == set(CLASSES)


@pytest.mark.parametrize("mod,qual,line,cls", [
    ("r2_positional_twin", "PositionalTwin.step_fixed", "b = active.nonzero()", "SHAPE"),
    ("r2_positional_twin", "PositionalTwin.step_fixed", "x = bool(self.pos.any())", "SYNC"),
    ("r2_positional_twin", "PositionalTwin.step_fixed", "w = torch.empty(3)", "ALLOC"),
    ("r2_decode_body", "DecodeBody.decode_fixed", "self.steps += 1", "HOSTSTATE"),
    ("r2_decode_body", "DecodeBody.decode_fixed", "q = pos[pos.max()]", "SCALARIDX"),
    ("r2_decode_body", "DecodeBody.decode_fixed", "self.decode(eng, key_states)", "CALLS_EAGER"),
    ("r2_writeback", "masked_writeback", "assert full.any()", "BRANCH"),
])
def test_L4_planted_regressions_in_the_captured_functions_are_found(mod, qual, line, cls):
    fn = _fn(mod, qual)
    fn.body.insert(1, ast.parse(line).body[0])
    assert cls in {c for c, _, _ in lint_function(fn)}


# ---------------------------------------------------------------------------
# the capture-safe twin step == the validated step, bitwise (fix X1)
# ---------------------------------------------------------------------------
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tw = _load(FILES["r2_positional_twin"], "nosi_r2_twin_for_capture_test")
H_, D_ = 2, 8


def _pair(S_list, extra, seed, data):
    g = torch.Generator().manual_seed(seed)
    B = len(S_list)
    total = max(S_list) + extra
    if data == "ints":
        keys = torch.randint(-8, 9, (B, total, H_, D_), generator=g).float()
        cis = torch.randint(-8, 9, (B, total, H_), generator=g).float()
    else:
        keys = torch.randn((B, total, H_, D_), generator=g)
        cis = torch.randn((B, total, H_), generator=g)
    a = tw.PositionalTwin(B, H_, D_, total // 16, total)
    f = tw.PositionalTwin(B, H_, D_, total // 16, total)
    for b, S in enumerate(S_list):
        a.prefill(b, keys[b, :S], cis[b, :S])
        f.prefill(b, keys[b, :S], cis[b, :S])
    return a, f, keys, cis


TABLES = ("pos", "n_comp", "n_ccis", "comp_k", "ring", "total_cis", "ccis")


def _assert_twins_equal(a, f, where):
    for k in TABLES:
        assert torch.equal(getattr(a, k), getattr(f, k)), "%s: %s differs" % (where, k)


@pytest.mark.parametrize("data", ["ints", "normals"])
def test_X1_step_fixed_equals_step_bitwise_on_ragged_runs(data):
    """step_fixed and step on the same inputs (B = 4, prompts with S % 16 = 0, 1, 15, 4, random activity, 250 steps):
    EVERY table (full tensors, not only the live prefixes) and every counter equal after every step; err stays 0."""
    S_list = [4032, 4033, 4047, 4100]
    a, f, keys, cis = _pair(S_list, 400, 3, data)
    g = torch.Generator().manual_seed(4)
    bi = torch.arange(4)
    for u in range(250):
        act = torch.rand(4, generator=g) < 0.6
        if u == 0:
            act = torch.tensor([True, False, True, False])
        p = a.pos.clone()
        k, c = keys[bi, p], cis[bi, p]
        a.step(k, c, active=act)
        f.step_fixed(k, c, act)
        _assert_twins_equal(a, f, "step %d" % (u + 1))
    assert int(f.err) == 0
    f.check_invariants()


def test_X1_step_fixed_refusals_change_nothing_and_set_err():
    S = 64
    g = torch.Generator().manual_seed(5)
    keys = torch.randn((2, 200, H_, D_), generator=g)
    cis = torch.randn((2, 200, H_), generator=g)
    t = tw.PositionalTwin(2, H_, D_, lk_max=S // 16 - 1, n_max=200)
    t.prefill(0, keys[0, :S], cis[0, :S])                                 # request 1 has no prefill
    snap = {k: getattr(t, k).clone() for k in TABLES}
    t.step_fixed(keys[:, S], cis[:, S], torch.tensor([True, True]))        # request 1 active, no prefill
    assert int(t.err) == tw.ERR_NO_PREFILL
    for k in TABLES:
        assert torch.equal(getattr(t, k), snap[k]), k
    with pytest.raises(RuntimeError, match="prefill"):
        t.check_error()
    t.clear_error()
    one = torch.tensor([True, False])
    for p in range(S, 79):                                                # the cis summary due at p = 79 has no row
        t.step_fixed(keys[:, p], cis[:, p], one)
    assert int(t.err) == 0
    snap = {k: getattr(t, k).clone() for k in TABLES}
    t.step_fixed(keys[:, 79], cis[:, 79], one)
    assert int(t.err) == tw.ERR_LK_MAX
    for k in TABLES:
        assert torch.equal(getattr(t, k), snap[k]), k
    with pytest.raises(tw.TwinCapacityError):
        t.check_error()


def test_X1_step_fixed_static_checks():
    a, f, keys, cis = _pair([64, 64], 64, 6, "ints")
    with pytest.raises(ValueError):
        f.step_fixed(keys[:, 64], cis[:, 64], torch.tensor([1, 1]))       # active not bool
    with pytest.raises(ValueError):
        f.step_fixed(keys[:1, 64], cis[:1, 64], torch.tensor([True]))      # wrong batch
