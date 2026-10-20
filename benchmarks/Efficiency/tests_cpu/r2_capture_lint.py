"""The capture lint (retroinfer-eval AGENTS.md law 3), SHARED: lint_function is e5fc694's tests/test_r2_capture.py
lint, copied unchanged (the plain decode step's audit, docs/R2_CAPTURE_AUDIT.md on feature/nosi-r2-m2-capture);
the tick's capture tests (test_r2_tick_capture.py, C7) import it from here. Not a test module (no test_ prefix).

LINT CLASSES (AST, per function, no import of the linted code):
  SYNC        a host value of a device tensor: bool(), float(), .item(), .tolist(), .cpu(), .numpy(), synchronize(),
              int() of anything but static metadata (.shape, .stride(), block_size, topk, the tail slot, lk_max, n_max)
  SHAPE       a data-dependent shape: .nonzero(), .unique(), .masked_select(), .argwhere(), indexing with a boolean
              tensor (a name assigned from a comparison, &, |, ~ or a mask method)
  BRANCH      a host branch (if / while / assert / conditional expression) whose test calls anything but static
              metadata, or reads a boolean / 0-d tensor name
  ALLOC       an explicit allocation: a torch factory or .clone()
  SCALARIDX   tensor[0-d tensor]
  HOSTSTATE   host state written per step: an augmented assignment or an attribute rebinding
  CALLS_EAGER a call of a function that is not capture-safe (.step(, .decode(, .advance(, .update(, ...)
allowed(findings, patterns): the findings that NO documented (class, regex) pattern explains; a callee's documented
findings are TRANSIENT allocations (the allocation rule, lead 2026-10-20), IN-PLACE augmented assignments on tensor
attributes (Tensor.__iadd__ / __ior__ rebind the attribute to the same object), or Python-scalar conversions of
non-tensor arguments.
"""
from __future__ import annotations

import ast
import re
import textwrap

CLASSES = ("SYNC", "SHAPE", "BRANCH", "ALLOC", "SCALARIDX", "HOSTSTATE", "CALLS_EAGER")
STATIC_CALLS = {"tuple", "isinstance", "len", "callable", "getattr", "hasattr", "type", "dim", "stride",
                "is_contiguous", "is_cuda", "is_pinned", "is_current_stream_capturing", "is_tensor", "size"}
INT_OK = re.compile(r"\.shape\b|\.stride\(|block_size|\btopk\b|_tail_block_idx_on_gpu|lk_max|n_max")
FACTORIES = {"empty", "zeros", "ones", "full", "arange", "tensor", "as_tensor", "empty_like", "zeros_like",
             "ones_like", "full_like", "rand", "randn", "randint", "randperm", "empty_strided", "linspace"}
SYNC_ATTRS = {"item", "tolist", "cpu", "numpy", "synchronize"}
SHAPE_ATTRS = {"nonzero", "unique", "masked_select", "argwhere"}
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


def fn_from_source(src: str, qual: str):
    """The FunctionDef `qual` ('func' or 'Class.method') of a module's source (top level only)."""
    tree = ast.parse(src)
    if "." in qual:
        cls, meth = qual.split(".")
        c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == meth)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == qual)


def unexplained(found, patterns):
    """The findings that no (class, regex) pattern explains (re.search on the finding's source)."""
    return [f for f in found if not any(f[0] == c and re.search(rx, f[2]) for c, rx in patterns)]


# e5fc694's positive control: one instance of every class
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
