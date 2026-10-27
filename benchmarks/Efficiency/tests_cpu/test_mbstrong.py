"""CPU tests of r2-mbstrong: --recap-nogc / --ab-record / --ab-ref / --fit-probe (benchmarks/Efficiency/r2_base_cell.py,
benchmarks/Efficiency/r2_recap_nogc.py) and the step script benchmarks/Efficiency/r2_mbstrong_step.sh. No CUDA, no
Triton launch. Notes: docs/NOTES_mbstrong.md.

THE CPU WORLD (MBS4): nosi/nosi/nosa_llama.py, cache_engine.py and stage1_seqused.py load through a synthetic package
(the GPU libraries stubbed; the fakes and the world of the graphstep tree's test_graph_step.py). torch.cuda.graph stays
the REAL class of the installed torch; only its CUDA primitives are faked (Stream, stream, synchronize, empty_cache,
CUDAGraph) and gc.collect is counted, so graph.__init__ / __enter__ / __exit__ run as shipped. Every call is logged in
LOG. The raw-pointer kernels are faked on FLAT ADDRESSES (L28): the gathers and the pool swap form element addresses
from the tensor's own strides and storage offset, as flash_h2d_mask.py:41-56 and flash_pool_swap.py:144-176 do;
diff_offload reads its tensors as the contiguous (H, B, M) block at the data pointer (diff_offload_kernel.cu:125-128);
pool_update refuses a non-contiguous tensor (pool_update.cpp:27-28).

MBS1 test_flag_off_is_shipped: the flags default off; r2_recap_nogc is imported only inside main, under `if ab_on`;
     main calls run_shipped_ab only under `if ab_on` and run_shipped otherwise; run_shipped_ab + shipped_loop_ab equal
     run_shipped by AST once the three documented ab calls are removed; every r2-mbstrong summary key and gate sits
     under `if mbs_on`; run_shipped and the other shipped harness functions are the AST of ded9d41; nosi/ and
     r2_ragged_gate.py are the bytes of ded9d41 (skipped without that commit).
MBS2 test_graph_no_release_contract: torch.cuda.graph(g) logs sync, gc, empty, stream enter, capture_begin, capture_end,
     stream exit; GraphNoRelease(g) logs the same sequence without the first three, on the same stream object, with the
     same capture_begin arguments (also with a pool and another capture_error_mode); check_contract passes on the
     installed torch and refuses three planted variants; capture_swap releases once, restores torch.cuda.graph after an
     exception and refuses a nested swap.
MBS3 test_gc_disable_does_not_stop_an_explicit_collect: the reason why "gc disabled around the loop" was not used.
MBS4 test_recap_nogc_equals_shipped_on_cpu_fakes[pool 4, 0]: three worlds of one seed, S 6128, 100 steps (they cross
     the first, out_len and width re-warms): the loop of run_shipped (ref), shipped_loop_ab plain and shipped_loop_ab
     --recap-nogc. torch.equal logits at every step and equal final state; the captured with-bodies of every re-warm
     and layer are the same code with the same capture_begin arguments and equal signatures (the scalars, the buffer
     layouts, the values of the inputs); per re-warm step the log is NLAYERS x (sync, gc, empty, stream, capture) in
     ref / plain and ONE (sync, gc, empty) + NLAYERS x (stream, capture) under --recap-nogc: gc.collect once per
     re-warm (counted by the monkeypatched gc.collect); steady steps log nothing; RN-GC and RN-EQ pass; the digests
     equal the digests of the ref logits.
MBS5 test_ab_compare_gates_and_verdicts: compare_ab, rn_gc_gate, logit_digest, rewarm_stats, effective_excl_first,
     is_oom and mbs_verdicts on planted values; the PRED_MBS state (TBD or stamped).
MBS6 test_batch_arithmetic: int32_batch_limit against cache_engine.py:216 / :219 (and p0prof's 128K B 63), fit_batch at
     82.04 and 84.0 GB for C 63 / 73 / 81 / 128, the peak model against three measured peaks.
MBS7 test_step_script: the static text (ROOT, BASE pin, claim per holder, cells = fit_batch), malformed cells STOP
     before any pin; at a clean checkout of the ROOT, DRYRUN prints every sub-cell (C, pool, B, mode, role, STEP_MIN,
     --mem) and the python arguments, the pair order of an even and an odd holder (skipped elsewhere, as MB4).

    env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_mbstrong.py
"""
from __future__ import annotations

import ast
import gc
import importlib
import inspect
import math
import os
import re
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.cuda import graphs as torch_graphs

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EFF = ROOT / "benchmarks/Efficiency"
NOSI = ROOT / "nosi/nosi"
BASE = "ded9d41"
PKG = "mbscpu"
STEP_ROOT = "/mnt/beegfs/ojerbi/nosi-r2-mbstrong"
UUID = "f31fc723-9dfe-407a-9947-28173556232a"
B, HQ, HKV, D, HID, INTER, VOCAB, NLAYERS = 2, 4, 2, 8, 32, 48, 50, 2
LOG = []


def _bc():
    sys.path.insert(0, str(EFF))
    import r2_base_cell as BC
    return BC


def _rn():
    sys.path.insert(0, str(EFF))
    import r2_recap_nogc as RN
    return RN


def _git_show(path: str):
    r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (BASE, path)], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _fn(src: str, name: str):
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


def _body(fn):
    b = fn.body
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
        b = b[1:]
    return b


def _dumps(stmts):
    return [ast.dump(s) for s in stmts]


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SRUN_", "SALLOC_", "SBATCH_"))}


def _clean_head():
    """The checkout's HEAD if this checkout IS the step script's ROOT and has no tracked change, else None."""
    if str(ROOT) != STEP_ROOT:
        return None
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if head.returncode != 0 or dirty.stdout.strip():
        return None
    return head.stdout.strip()


# ------------------------------------------------------------------ CUDA capture primitives on the CPU (logged)
class FakeStream:
    pass


class FakeStreamCtx:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        LOG.append(("stream_enter", id(self.s)))
        return self

    def __exit__(self, *exc):
        LOG.append(("stream_exit", id(self.s)))
        return False


_BODY, _NAMES = {}, {}
VALUE_TENSORS = ("self.score_buf", "self.compressed_cis_buf", "max_pooling_buf", "max_pooling_buf_cis")


def _with_body(code):
    """The compiled body of the `with torch.cuda.graph(...)` block of a function's code object, and the names it reads
    (locals and self attributes)."""
    if code not in _BODY:
        tree = ast.parse(textwrap.dedent(inspect.getsource(code)))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.With)
                    and "cuda.graph" in ast.unparse(n.items[0].context_expr))
        mod = ast.Module(body=node.body, type_ignores=[])
        ast.fix_missing_locations(mod)
        _BODY[code] = compile(mod, code.co_filename, "exec")
        names = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - {"self"})
        attrs = sorted({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                        and isinstance(n.value, ast.Name) and n.value.id == "self"})
        _NAMES[code] = (names, attrs)
    return _BODY[code]


def _canon(name, v):
    """What a capture freezes of one name: a tensor's layout (dtype, shape, strides, storage offset) and, for the
    body's INPUTS (VALUE_TENSORS), its values; a scalar's value; another object's type."""
    if torch.is_tensor(v):
        meta = (str(v.dtype), tuple(v.shape), tuple(v.stride()), v.storage_offset())
        return ("T", meta, v.detach().clone() if name in VALUE_TENSORS else None)
    if isinstance(v, (bool, int, float, str, type(None))):
        return ("V", v)
    return ("O", type(v).__name__)


def _signature(code, loc):
    names, attrs = _NAMES[code]
    sig = {n: _canon(n, loc[n]) for n in names if n in loc}
    s = loc["self"]
    sig.update({"self." + a: _canon("self." + a, getattr(s, a)) for a in attrs})
    return sig


def _sig_diff(a, b):
    bad = sorted(set(a) ^ set(b))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x[0] != y[0] or x[1] != y[1]:
            bad.append(k)
        elif x[0] == "T" and ((x[2] is None) != (y[2] is None) or (x[2] is not None and not torch.equal(x[2], y[2]))):
            bad.append(k)
    return bad


class FakeCUDAGraph:
    """torch.cuda.CUDAGraph on the CPU. capture_begin logs its arguments and freezes the with-body of the calling
    decode_forward_warmup frame (its code, its globals, its locals: the frozen scalars and buffer objects a real
    capture records) and the signature of what that body reads; capture_end logs; replay() re-runs the frozen body."""

    def __init__(self, *a, **k):
        self.body, self.args, self.sig, self.ended = None, None, None, False

    def capture_begin(self, *args, **kw):
        self.args = (args, tuple(sorted(kw.items())))
        LOG.append(("capture_begin",) + self.args)
        fr = sys._getframe(1)
        while fr is not None and fr.f_code.co_name != "decode_forward_warmup":
            fr = fr.f_back
        if fr is not None:
            self.body = (_with_body(fr.f_code), fr.f_globals, dict(fr.f_locals))
            self.sig = _signature(fr.f_code, fr.f_locals)

    def capture_end(self):
        LOG.append(("capture_end",))
        self.ended = True

    def replay(self):
        assert self.ended and self.body is not None, "replay before a complete capture"
        code, glb, loc = self.body
        exec(code, glb, dict(loc))


@pytest.fixture
def cuda_fakes(monkeypatch):
    """The capture primitives on the CPU, every call logged in LOG; torch.cuda.graph stays the REAL class."""
    RN = _rn()
    LOG.clear()
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: FakeStreamCtx(s))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: LOG.append(("sync",)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: LOG.append(("empty",)))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeCUDAGraph)
    monkeypatch.setattr(gc, "collect", lambda *a, **k: LOG.append(("gc",)) or 0)
    monkeypatch.setattr(RN.BASE, "default_capture_stream", None)
    assert torch.cuda.graph is RN.BASE is torch_graphs.graph
    return RN


# ------------------------------------------------------------------ MBS1
class _Unab(ast.NodeTransformer):
    """run_shipped's statements from shipped_loop_ab's: unwrap `with ab.step(...)`, drop `ab.<x>(...)` statements,
    the parameter `device` -> "cuda"."""

    @staticmethod
    def ab_call(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "ab")

    def visit_With(self, node):
        self.generic_visit(node)
        if len(node.items) == 1 and self.ab_call(node.items[0].context_expr):
            return node.body
        return node

    def visit_Expr(self, node):
        return None if self.ab_call(node.value) else node

    def visit_Name(self, node):
        if node.id == "device" and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(value="cuda", kind=None), node)
        return node


def test_MBS1_flag_off_is_shipped():
    BC = _bc()
    base = ["--out", "x", "--workload", "w", "--batch", "216", "--arms", "SH", "--pool", "65"]
    a = BC.parse_args(base)
    assert (a.recap_nogc, a.ab_record, a.ab_ref, a.fit_probe) == (False, False, None, False)
    a = BC.parse_args(base + ["--recap-nogc", "--ab-record", "--ab-ref", "d", "--fit-probe"])
    assert (a.recap_nogc, a.ab_record, a.ab_ref, a.fit_probe) == (True, True, "d", True)
    with pytest.raises(SystemExit):
        BC.parse_args(base + ["--ab-ref", "d"])                          # RN-EQ needs a record of its own
    for flag in ("--recap-nogc", "--ab-record", "--fit-probe"):         # the SH arm only
        with pytest.raises(SystemExit):
            BC.parse_args(["--out", "x", "--workload", "w", "--batch", "64", flag])
    src = (EFF / "r2_base_cell.py").read_text()
    tree = ast.parse(src)
    top = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not [s for s in top if "r2_recap_nogc" in s], top
    importers = sorted(n.name for n in tree.body if isinstance(n, ast.FunctionDef) and any(
        isinstance(x, ast.Import) and any(al.name == "r2_recap_nogc" for al in x.names) for x in ast.walk(n)))
    assert importers == ["main"], importers
    main = _fn(src, "main")
    assigns = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in main.body if isinstance(n, ast.Assign)}
    assert assigns["ab_on"] == "a.recap_nogc or a.ab_record" and assigns["mbs_on"] == "ab_on or a.fit_probe"

    def calls(nodes, name):
        return [c for n in nodes for c in ast.walk(n)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name]
    gated = [n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "ab_on"]
    assert any(calls(n.body, "run_shipped_ab") and calls(n.orelse, "run_shipped") and not calls(n.body, "run_shipped")
               for n in gated)
    assert len(calls([main], "run_shipped")) == 1 and len(calls([main], "run_shipped_ab")) == 1
    imp = [n for n in gated if any(isinstance(x, ast.Import) for x in ast.walk(n))]
    assert imp and all(not n.orelse for n in imp)                        # the import is under `if ab_on` only
    inside = {id(x) for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("mbs_on")
              for x in ast.walk(n)}
    for x in ast.walk(main):
        u = ast.unparse(x) if isinstance(x, (ast.Expr, ast.Assign)) else ""
        if u.startswith(("meta.update(", "out['AB']", "torch.save(", "preds = preds + mbs_verdicts")) or "'RN-" in u:
            assert id(x) in inside, u
    # run_shipped_ab + shipped_loop_ab = run_shipped once the three ab calls are removed
    ship = _body(_fn(src, "run_shipped"))
    rab = _body(_fn(src, "run_shipped_ab"))
    raw = _fn(src, "shipped_loop_ab")
    assert sorted(ast.unparse(c) for c in ast.walk(raw) if _Unab.ab_call(c)) == [
        "ab.after(t, need, lg)", "ab.finish(lg)", "ab.result()", "ab.step(t, need)"]
    loop = _body(_Unab().visit(_fn(src, "shipped_loop_ab")))
    i_cu = next(i for i, s in enumerate(ship) if ast.unparse(s).startswith("cu = "))
    i_fr = next(i for i, s in enumerate(ship) if re.match(r"\(?fr, tot\)? = ", ast.unparse(s)))
    j = next(i for i, s in enumerate(rab) if ast.unparse(s).startswith("out = shipped_loop_ab("))
    assert _dumps(rab[:j]) == _dumps(ship[:i_cu])                       # the prefill lines
    assert ast.unparse(loop[0]) == "import torch"
    assert _dumps(loop[1:-1]) == _dumps(ship[i_cu:i_fr])                 # the loop and the timer read-back
    assert _dumps(rab[j + 1:j + 3]) == _dumps(ship[i_fr:i_fr + 2])       # mem_get_info, del cache
    ret_ship = {k.arg: ast.dump(k.value) for k in ship[-1].value.keywords}
    ret_loop = {k.arg: ast.dump(k.value) for k in _body(raw)[-1].value.keywords}
    upd = {k.arg: ast.dump(k.value) for k in rab[j + 3].value.keywords}
    assert ast.unparse(rab[j + 3]).startswith("out.update(") and ast.unparse(rab[-1]) == "return out"
    assert set(ret_loop) | set(upd) == set(ret_ship) | {"ab"}
    for k in ret_ship:
        assert {**ret_loop, **upd}[k] == ret_ship[k], k
    old = _git_show("benchmarks/Efficiency/r2_base_cell.py")
    if old is None:
        pytest.skip("the base commit %s is not reachable from this checkout" % BASE)
    old = old.decode()
    for fn in ("run_shipped", "run_r2", "r2_teacher_forced", "rewarm_need", "warm_state", "expected_rewarms",
               "period_stats", "rates", "agreement", "free", "mbase_peak_model", "mbase_batch", "verdicts_mbase"):
        assert ast.dump(_fn(src, fn)) == ast.dump(_fn(old, fn)), fn
    for p in ("nosi/nosi/nosa_llama.py", "nosi/nosi/cache_engine.py", "nosi/nosi/max_pooling_fused.py",
              "nosi/nosi/nosa_linear.py", "nosi/nosi/stage1_seqused.py", "nosi/nosi/__init__.py",
              "nosi/nosi/flash_cache_engine/pool_update.cpp", "nosi/nosi/flash_cache_engine/pool_update_kernel.cu",
              "nosi/nosi/flash_cache_engine/diff_offload_kernel.cu", "nosi/nosi/flash_cache_engine/flash_pool_swap.py",
              "nosi/nosi/flash_cache_engine/flash_h2d_mask.py", "nosi/nosi/flash_cache_engine/flash_h2d_mask_bias.py",
              "benchmarks/Efficiency/r2_ragged_gate.py"):
        assert (ROOT / p).read_bytes() == _git_show(p), p


# ------------------------------------------------------------------ MBS2
class PlantedEnter(torch_graphs.graph):
    """A torch whose graph.__enter__ has one statement more than torch 2.6.0 (negative control of check_contract)."""

    def __enter__(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.stream_ctx.__enter__()
        self.cuda_graph.capture_begin(*self.pool, capture_error_mode=self.capture_error_mode)


def test_MBS2_graph_no_release_contract(cuda_fakes):
    RN = cuda_fakes
    info = RN.check_contract()                                          # the installed torch
    assert tuple(info["graph_enter"]) == RN.SHIPPED_ENTER
    assert tuple(info["no_release_enter"]) == RN.SHIPPED_ENTER[RN.N_RELEASE:]
    g0 = FakeCUDAGraph()
    RN.GraphNoRelease(g0)                                               # first: it creates the stream on the BASE class
    assert isinstance(RN.BASE.default_capture_stream, FakeStream)
    assert "default_capture_stream" not in vars(RN.GraphNoRelease)
    s = id(RN.BASE.default_capture_stream)
    for kw in ({}, dict(pool=("pool", 7), capture_error_mode="relaxed")):
        LOG.clear()
        g1 = FakeCUDAGraph()
        with torch.cuda.graph(g1, **kw):
            LOG.append(("body",))
        shipped = list(LOG)
        LOG.clear()
        g2 = FakeCUDAGraph()
        made = RN.GraphNoRelease.made
        with RN.GraphNoRelease(g2, **kw):
            LOG.append(("body",))
        ours = list(LOG)
        assert RN.GraphNoRelease.made == made + 1
        assert shipped[:3] == [("sync",), ("gc",), ("empty",)]
        assert ours == shipped[3:], (shipped, ours)                     # the same calls minus the release
        assert ours[0] == ("stream_enter", s) and ours[-1] == ("stream_exit", s)
        pool = (kw["pool"],) if "pool" in kw else ()
        assert ours[1] == ("capture_begin", pool, (("capture_error_mode", kw.get("capture_error_mode", "global")),))
        assert ours[2:4] == [("body",), ("capture_end",)]
        assert g1.args == g2.args and g1.ended and g2.ended

    class KeepsGc(RN.GraphNoRelease):
        def __enter__(self):
            gc.collect()
            self.stream_ctx.__enter__()
            self.cuda_graph.capture_begin(*self.pool, capture_error_mode=self.capture_error_mode)

    class OwnExit(RN.GraphNoRelease):
        def __exit__(self, *exc):
            return None
    for base, ours, why in ((PlantedEnter, RN.GraphNoRelease, "__enter__ of this torch"),
                            (RN.BASE, KeepsGc, "KeepsGc.__enter__"), (RN.BASE, OwnExit, "its own __exit__")):
        with pytest.raises(RuntimeError, match=re.escape(why)):
            RN.check_contract(base=base, ours=ours)
    row = {}
    LOG.clear()
    with pytest.raises(ZeroDivisionError):
        with RN.capture_swap(row):
            assert torch.cuda.graph is RN.GraphNoRelease
            raise ZeroDivisionError
    assert torch.cuda.graph is RN.BASE and row["release_ms"] is not None
    assert LOG == [("sync",), ("gc",), ("empty",)]                      # ONE release, in the order of graph.__enter__
    LOG.clear()
    with RN.capture_swap({}):
        with pytest.raises(RuntimeError, match="swap is active"):
            with RN.capture_swap({}):
                pass
        assert torch.cuda.graph is RN.GraphNoRelease
    assert torch.cuda.graph is RN.BASE
    assert LOG == [("sync",), ("gc",), ("empty",)]                      # the refused swap released nothing
    ab = RN.ABRecorder(True, True, NLAYERS)
    LOG.clear()
    with ab.step(5, False):                                             # a steady step passes through
        assert torch.cuda.graph is RN.BASE
    assert LOG == [] and ab.rows == {}


# ------------------------------------------------------------------ MBS3
def test_MBS3_gc_disable_does_not_stop_an_explicit_collect():
    class Cyc:
        pass
    was = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        a = Cyc()
        a.me = a
        del a
        assert not gc.isenabled()
        assert gc.collect() >= 1          # the cycle is collected although the automatic collector is disabled
    finally:
        if was:
            gc.enable()


# ------------------------------------------------------------------ the CPU world (MBS4)
def _flat(t):
    """The address space a raw-pointer kernel sees: a 1-D view over t's whole storage, in elements (L28)."""
    return torch.as_strided(t, (t.untyped_storage().nbytes() // t.element_size(),), (1,), 0)


def _raw(t, shape):
    """What a data_ptr kernel that assumes a contiguous layout reads: `shape` elements from t's data pointer."""
    st, n = [], 1
    for s in reversed(shape):
        st.insert(0, n)
        n *= s
    return torch.as_strided(t, tuple(shape), tuple(st), t.storage_offset())


def _rows_addr(t, b, rows, h):
    """Element addresses of rows `rows` of (b, :, h[, :]) as the Triton kernels form them: storage offset + b x
    stride_b + row x stride_m + h x stride_h (+ d x stride_d), the strides of the tensor itself."""
    s = t.stride()
    a = t.storage_offset() + b * s[0] + rows[:, None] * s[1] + h * s[2]
    if t.dim() == 4:
        a = a + torch.arange(t.shape[3])[None, :] * s[3]
    return a


class FakeExt:
    """The load() result (diff_offload, pool_update) and the stage-1 extension (varlen_fwd_stage1), on the CPU. The
    contracts of diff_offload_kernel.cu, pool_update_kernel.cu and the stage-1 op (as test_graph_step.py)."""

    def __init__(self):
        self.hits = self.parks = 0

    def diff_offload(self, old, new, new_map, load):
        H, Bn, M = old.shape
        O, N, NM, LD = (_raw(x, (H, Bn, M)) for x in (old, new, new_map, load))
        for h in range(H):
            for b in range(Bn):
                o, n = O[h, b].clone(), N[h, b].clone()
                oh, nh = torch.isin(o, n), torch.isin(n, o)
                NM[h, b][oh] = o[oh]
                LD[h, b][oh] = -1
                free, want = (~oh).nonzero().flatten(), (~nh).nonzero().flatten()
                k = min(free.numel(), want.numel())
                NM[h, b][free[:k]] = n[want[:k]]
                LD[h, b][free[:k]] = n[want[:k]]

    def pool_update(self, bm, lm, pmap, page, ptgt, pact, stamp, tail):
        for t in (bm, lm, pmap, page, ptgt, pact):
            assert t.is_contiguous(), "pool_update.cpp:28: must be contiguous"
        H, Bn, M = bm.shape
        P = pmap.shape[2]
        BM, LM, PM, PA = bm.tolist(), lm.tolist(), pmap.tolist(), page.tolist()
        TG = [[[-1] * M for _ in range(Bn)] for _ in range(H)]
        AC = [[[0] * M for _ in range(Bn)] for _ in range(H)]
        for h in range(H):
            for b in range(Bn):
                X, Y, pm, pa = LM[h][b], BM[h][b], PM[h][b], PA[h][b]
                claim, hitq, want = [-1] * P, [-1] * M, [False] * M
                for m in range(M):
                    if m == tail or X[m] < 0:
                        continue
                    q = next((j for j in range(P) if pm[j] == X[m]), -1)
                    if q >= 0 and claim[q] < 0:
                        claim[q], hitq[m] = m, q
                    elif Y[m] >= 0:
                        want[m] = True
                unclaimed = [j for j in range(P) if claim[j] < 0]
                order = sorted(unclaimed, key=lambda j: (pa[j], j))
                npm, npa = list(pm), list(pa)
                rank = 0
                for m in range(M):
                    if m == tail or X[m] < 0:
                        continue
                    q = hitq[m]
                    if q >= 0:
                        self.hits += 1
                        X[m] = -1
                        TG[h][b][m] = q
                        if Y[m] >= 0:
                            AC[h][b][m], npm[q], npa[q] = 1, Y[m], stamp + m
                            self.parks += 1
                        else:
                            AC[h][b][m], npm[q], npa[q] = 2, -1, -P + q
                    elif want[m]:
                        r, rank = rank, rank + 1
                        if r < len(order):
                            q = order[r]
                            TG[h][b][m], AC[h][b][m], npm[q], npa[q] = q, 3, Y[m], stamp + m
                            self.parks += 1
                PM[h][b], PA[h][b] = npm, npa
        lm.copy_(torch.tensor(LM, dtype=lm.dtype))
        pmap.copy_(torch.tensor(PM, dtype=pmap.dtype))
        page.copy_(torch.tensor(PA, dtype=page.dtype))
        ptgt.copy_(torch.tensor(TG, dtype=ptgt.dtype))
        pact.copy_(torch.tensor(AC, dtype=pact.dtype))

    def varlen_fwd_stage1(self, q, k, v, out, cu_q, cu_k, seqused, leftpad, block_table, alibi, max_q, max_k, p_drop,
                          scale, zero, causal, wl, wr, softcap, ret, gen):
        Bn, H = cu_q.numel() - 1, k.shape[1]
        W = (int(max_k) + 127) // 128 * 128
        p = torch.zeros((H, Bn, W), dtype=q.dtype)
        for b in range(Bn):
            q0, q1, k0 = int(cu_q[b]), int(cu_q[b + 1]), int(cu_k[b])
            L = int(seqused[b]) if seqused is not None else int(cu_k[b + 1]) - k0
            s = torch.einsum("ghd,lhd->hgl", q[q0:q1], k[k0:k0 + L]) * scale
            p[:, b, :L] = torch.softmax(s, dim=-1).sum(dim=1)
        return (p, None, None, None)


def f_rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def f_silu_and_mul(x, out):
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])


def f_rope(positions, q, k, head_size, cache, is_neox):
    p = positions.to(q.dtype)[:, None]
    q.add_(p * 1e-3)
    k.add_(p * 2e-3)


def f_nosa_linear(qkv, delta_w, A, q_size, kv_size, kv_heads):
    q = qkv[..., :q_size].contiguous()
    k = qkv[..., q_size:q_size + kv_size].contiguous()
    v = qkv[..., q_size + kv_size:].contiguous()
    cis = (A * F.softplus(v @ delta_w.t())).contiguous()
    return q, k, v, cis


def f_pooling(q_score, cis_score, max_seqlen, q_pooled, cis_pooled, block_size=32, stride=16, padding=1, set_size=5,
              init_blocks=1, local_blocks=16):
    """nosa_pool_kernel (max_pooling_fused.py:7-75): window m reads entries 4m - 1 .. 4m + 3 below max_seqlen; the
    first init_blocks and the last local_blocks + 1 windows are +inf."""
    H, Bn, M = q_pooled.shape
    m = torch.arange(M)
    j = torch.arange(8)
    idx = m[:, None] * (set_size - 1) - padding + j[None, :]
    valid = (idx < max_seqlen) & (idx >= 0) & (j[None, :] < set_size)
    boundary = (m < init_blocks) | (M - m <= local_blocks + 1)
    for src, dst in ((q_score, q_pooled), (cis_score, cis_pooled)):
        g = src[:, :, idx.clamp(0, src.shape[-1] - 1).reshape(-1)].reshape(H, Bn, M, 8)
        acc = g.masked_fill(~valid, float("-inf")).max(dim=-1).values
        dst.copy_(torch.where(boundary, torch.full_like(acc, float("inf")), acc))


def f_attn(q, k_cache, v_cache, kv_bias, cache_seqlens=None, num_splits=0):
    Bn, _, Hq, Dh = q.shape
    G = Hq // k_cache.shape[2]
    out = torch.empty_like(q)
    for b in range(Bn):
        L = int(cache_seqlens[b])
        for h in range(Hq):
            g = h // G
            s = (k_cache[b, :L, g] @ q[b, 0, h]) / math.sqrt(Dh) + kv_bias[b, :L, g]
            out[b, 0, h] = torch.softmax(s, dim=0) @ v_cache[b, :L, g]
    return out


def f_pool_swap(k_gpu, v_gpu, bias, target, action, topk, block_size=64):
    """pool_swap_kernel (flash_pool_swap.py:89-176) on flat addresses: both sides loaded before either is stored."""
    Bn, S_GPU, H, _ = k_gpu.shape
    M = target.shape[2]
    P = S_GPU // block_size - topk
    A, T = _raw(action, (H, Bn, M)), _raw(target, (H, Bn, M))
    off = torch.arange(block_size)
    for h, b, m in (A != 0).nonzero().tolist():
        act, q = int(A[h, b, m]), int(T[h, b, m])
        if q < 0 or q >= P:
            continue
        read_att, read_pool = act in (1, 3), act in (1, 2)
        att_r, pool_r = m * block_size + off, (topk + q) * block_size + off
        ma, mp = att_r < S_GPU, pool_r < S_GPU
        for t in (k_gpu, v_gpu, bias):
            fl = _flat(t)
            aa, pa = _rows_addr(t, b, att_r, h), _rows_addr(t, b, pool_r, h)
            zero = torch.zeros((), dtype=fl.dtype)
            xa = torch.where((ma & read_att)[:, None], fl[aa.clamp(0, fl.numel() - 1)], zero)
            xp = torch.where((mp & read_pool)[:, None], fl[pa.clamp(0, fl.numel() - 1)], zero)
            if read_att:
                fl[pa[mp]] = xa[mp]
            if read_pool:
                fl[aa[ma]] = xp[ma]


def _h2d(gpu, cpu, load, bs, bias_buf=None, bias=None):
    """flash_h2d_from_mask{,_bias}_kernel on flat addresses: masked source rows (row >= S_CPU) load 0 (Triton's masked
    load without `other` is undefined; never reached here), masked destination rows (row >= S_GPU) are not stored."""
    Bn, S_GPU, H, _ = gpu.shape
    S_CPU = cpu.shape[1]
    M = load.shape[2]
    L = _raw(load, (H, Bn, M))
    pairs = [(gpu, cpu)] + ([(bias_buf, bias)] if bias_buf is not None else [])
    off = torch.arange(bs)
    for h, b, m in (L >= 0).nonzero().tolist():
        X = int(L[h, b, m])
        src_r, dst_r = X * bs + off, m * bs + off
        ms, md = src_r < S_CPU, dst_r < S_GPU
        for dst_t, src_t in pairs:
            G, C = _flat(dst_t), _flat(src_t)
            sa, da = _rows_addr(src_t, b, src_r, h), _rows_addr(dst_t, b, dst_r, h)
            vals = torch.where(ms[:, None], C[sa.clamp(0, C.numel() - 1)], torch.zeros((), dtype=C.dtype))
            G[da[md]] = vals[md]


def f_h2d(gpu, cpu, load, block_size=64):
    _h2d(gpu, cpu, load, block_size)


def f_h2d_bias(gpu, cpu, bias_buf, bias, load, block_size=64):
    _h2d(gpu, cpu, load, block_size, bias_buf, bias)


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _gpu_only(*a, **k):
    raise RuntimeError("a GPU library function was called in a CPU test (stub)")


def _load_package():
    """nosa_llama, cache_engine and stage1_seqused as mbscpu.*, the GPU libraries stubbed, cpp_extension.load
    returning FakeExt, NOSI_POOL_BLOCKS 4 at import. Cached in sys.modules."""
    if PKG + ".nosa_llama" in sys.modules:
        return sys.modules[PKG + ".nosa_llama"], sys.modules[PKG + ".cache_engine"], sys.modules[PKG + ".stage1_seqused"]
    pytest.importorskip("transformers")
    import torch.utils.cpp_extension as cpp
    mp = pytest.MonkeyPatch()
    try:
        stubs = {
            "flashinfer": _stub("flashinfer"),
            "flashinfer.norm": _stub("flashinfer.norm", rmsnorm=_gpu_only),
            "flashinfer.activation": _stub("flashinfer.activation", silu_and_mul=_gpu_only),
            "flashinfer.rope": _stub("flashinfer.rope", apply_rope_with_cos_sin_cache_inplace=_gpu_only),
            "infllm_v2": _stub("infllm_v2", infllmv2_attn_stage1=_gpu_only, infllmv2_attn_stage1_fast=_gpu_only,
                               infllmv2_attn_varlen_func=_gpu_only, infllmv2_attn_with_kvcache=_gpu_only,
                               max_pooling_1d=_gpu_only, max_pooling_1d_varlen=_gpu_only),
            "flash_attn": _stub("flash_attn", flash_attn_with_kvcache=_gpu_only),
            "flash_attn_nosa": _stub("flash_attn_nosa", flash_attn_with_kvcache=_gpu_only),
        }
        try:
            import triton                      # noqa: F401  (a real import works on a CPU node; launches never happen)
            import triton.language             # noqa: F401
        except Exception:
            tr, tl = _stub("triton"), _stub("triton.language")
            tr.jit = lambda fn=None, **k: fn if fn is not None else (lambda f: f)
            tr.cdiv = lambda a, b: -(-a // b)
            tr.language = tl
            tl.constexpr = object
            stubs.update({"triton": tr, "triton.language": tl})
        for name, mod in stubs.items():
            mp.setitem(sys.modules, name, mod)
        mp.setattr(cpp, "load", lambda *a, **k: FakeExt())
        mp.setenv("NOSI_POOL_BLOCKS", "4")
        mp.setenv("NOSI_ATTN_SPLITS", "4")
        for v in ("NOSI_AVAIL", "NOSI_TRANSFER_TRACE", "NOSI_R2_DECODE"):
            mp.delenv(v, raising=False)
        if PKG not in sys.modules:
            pkg = types.ModuleType(PKG)
            pkg.__path__ = [str(NOSI)]
            sys.modules[PKG] = pkg
        try:
            NL = importlib.import_module(PKG + ".nosa_llama")
            S1 = importlib.import_module(PKG + ".stage1_seqused")
        except Exception:
            for k in [k for k in sys.modules if k == PKG or k.startswith(PKG + ".")]:
                del sys.modules[k]
            raise
    finally:
        mp.undo()
    return NL, sys.modules[PKG + ".cache_engine"], S1


@pytest.fixture
def W(monkeypatch, cuda_fakes):
    """The loaded modules with every kernel replaced by its CPU fake and the capture primitives logged (undone after
    the test)."""
    NL, CE, S1 = _load_package()
    ext = FakeExt()

    def shipped_stage1(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False):
        return S1.stage1_seqused(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, seqused_k=None,
                                 causal=causal, ext=ext)
    for name, fn in (("rmsnorm", f_rmsnorm), ("silu_and_mul", f_silu_and_mul),
                     ("apply_rope_with_cos_sin_cache_inplace", f_rope), ("nosa_linear", f_nosa_linear),
                     ("nosa_pooling", f_pooling), ("flash_attn_nosa_with_kvcache", f_attn),
                     ("infllmv2_attn_stage1_fast", shipped_stage1)):
        monkeypatch.setattr(NL, name, fn)
    monkeypatch.setattr(NL, "nvtx", types.SimpleNamespace(range_push=lambda *a, **k: None,
                                                          range_pop=lambda *a, **k: None))
    monkeypatch.setattr(CE, "diff", ext)
    monkeypatch.setattr(CE, "_pool", ext)
    monkeypatch.setattr(CE, "_flash_pool_swap", f_pool_swap)
    monkeypatch.setattr(CE, "flash_h2d_from_mask", f_h2d)
    monkeypatch.setattr(CE, "flash_h2d_from_mask_bias", f_h2d_bias)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    return types.SimpleNamespace(NL=NL, CE=CE, ext=ext, mp=monkeypatch, RN=cuda_fakes)


def _cache_class(CE):
    class FakeCache:
        """InfLLMv2Cache's own delegating methods on a plain object (no transformers DynamicCache constructor)."""

        def __init__(self, layers):
            self.layers = layers
            self._seen_tokens = 0
    for n in ("prefill_update_kv", "decode_update_kv", "update_cis", "update_uncompressed_cis",
              "update_compress_k_prefill", "update_compress_k_decode", "update_no_compress_k_prefill",
              "update_no_compress_k_decode", "get_seq_length"):
        setattr(FakeCache, n, CE.InfLLMv2Cache.__dict__[n])
    return FakeCache


def build_world(W, S, steps, seed):
    """A NOSA-like model (B 2, 4 query heads, 2 KV heads, head 8, 2 layers, vocabulary 50) and its cache after the
    SHIPPED prefill updates (nosa_llama.prefill_forward's order, the cache-layer methods themselves)."""
    NL, CE = W.NL, W.CE
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.5):
        return torch.randn(shape, generator=g) * scale
    with torch.inference_mode():
        model = NL.Llama.__new__(NL.Llama)
        model.embed_tokens, model.lm_head = rnd(VOCAB, HID), rnd(VOCAB, HID)
        model.norm_weight, model.norm_variance_epsilon = 1 + rnd(HID, scale=0.1), 1e-6
        model.cos_sin_cache = torch.zeros(1)
        model.has_buffers, model.block_size, model.num_key_value_heads = False, 64, HKV
        model.hidden_size, model.vocab_size, model.dtype, model.device = HID, VOCAB, torch.float32, "cpu"
        model.layers = []
        for i in range(NLAYERS):
            L = NL.LlamaLayer(i, topk_blocks=64, topk_q_blocks=16, init_blocks=1, window_blocks=16, block_size=64,
                              pooling_stride=16, pooling_block_size=32)
            L.wqkv, L.wo = rnd(HQ * D + 2 * HKV * D, HID), rnd(HID, HQ * D)
            L.gate_up_proj, L.down_proj = rnd(2 * INTER, HID), rnd(HID, INTER)
            L.delta, L.A = types.SimpleNamespace(weight=rnd(HKV, HKV * D)), rnd(HKV)
            L.q_size, L.kv_size = HQ * D, HKV * D
            L.input_layernorm_weight, L.input_layernorm_variance_epsilon = 1 + rnd(HID, scale=0.1), 1e-6
            L.post_attention_layernorm_weight, L.post_attention_layernorm_variance_epsilon = 1 + rnd(HID, scale=0.1), 1e-6
            L.num_heads, L.num_key_value_heads, L.head_dim = HQ, HKV, D
            model.layers.append(L)
        model.num_layers = NLAYERS
        cfg = types.SimpleNamespace(num_hidden_layers=NLAYERS, num_key_value_heads=HKV, head_dim=D)
        layers = [CE.InfLLMv2CacheLayer(cfg, True) for _ in range(NLAYERS)]
        for L in layers:
            L.cache_engine.device = torch.device("cpu")
            L.cache_engine.dtype = torch.float32            # the model dtype here (bf16 on the GPU, every world)
            L.cache_engine.max_gen_len = steps + 64
        cache = _cache_class(CE)(layers)
        cu = torch.arange(0, B * S + 1, S, dtype=torch.int32)
        for l in range(NLAYERS):
            keys, vals, cis = rnd(B, S, HKV, D), rnd(B, S, HKV, D), rnd(B, S, HKV, scale=0.2)
            ucis = cache.update_uncompressed_cis(cis, l, 0, B)
            cache.update_cis(cis.permute(2, 0, 1), l, 0, B)
            cache.prefill_update_kv(keys, vals, ucis, l, 0, B)
            ck, cuc = NL.compress_k(keys.reshape(B * S, HKV, D), cu, HKV, D)
            m0 = int((cuc[1:] - cuc[:-1]).max())
            cache.update_compress_k_prefill(ck.contiguous(), l, cuc, max_seqlen=m0, current_batch_pos=0, total_bsz=B)
            cache.update_no_compress_k_prefill(keys[:, m0 * 16:], l, 32, 16, current_batch_pos=0, total_bsz=B)
        tok = torch.randint(0, VOCAB, (B, 1), generator=g)
        pos = torch.full((B, 1), S, dtype=torch.int32)
    return model, cache, tok, pos


class TestClock:
    """r2_base_cell.HostClock with an index per event; record() logs ("step", index). The shipped loop records the
    event of step t BEFORE its decode call, so the log segment of step t holds all of step t, the release included."""
    __test__ = False

    def __init__(self):
        self.n = 0

    def event(self):
        self.n += 1
        return [0.0, self.n - 1]

    def record(self, e):
        LOG.append(("step", e[1]))
        e[0] = time.perf_counter()

    def elapsed(self, a, b):
        return (b[0] - a[0]) * 1e3

    def sync(self):
        pass


def _state(cache):
    """The fully defined state after the run (the memory that the prefill leaves undefined is not compared)."""
    out = {}
    for l, L in enumerate(cache.layers):
        e = L.cache_engine
        names = ["_block_map", "_new_block_map_buf", "_load_mask", "_cache_lens"]
        if e.pool_blocks > 0:
            names += ["_pool_map", "_pool_age", "_pool_target", "_pool_action"]
        out.update({"eng%d.%s" % (l, n): getattr(e, n).clone() for n in names})
        out.update({"layer%d.%s" % (l, n): getattr(L, n).clone()
                    for n in ("compressed_cis", "cached_compressed_cu_seqlens", "compress_k_cache_varlen")})
        out.update({"layer%d.%s" % (l, n): getattr(L, n)
                    for n in ("seq_length", "no_compress_k_len", "cached_compressed_max_seqlen", "cis_len",
                              "tail_cis_len")})
        out.update({"eng%d.%s" % (l, n): getattr(e, n)
                    for n in ("seq_length", "_tail_block_len_on_gpu", "_pool_stamp_base", "_tail_block_idx_on_gpu")})
    return out


def _eq(a, b):
    return torch.equal(a, b) if torch.is_tensor(a) else a == b


def _segments(log, steps):
    """LOG split at the ("step", t) markers: {t: [the events of step t]}."""
    seg, cur = {}, None
    for e in log:
        if e[0] == "step":
            cur = e[1]
            if cur < steps:
                seg[cur] = []
            else:
                cur = None
        elif cur is not None:
            seg[cur].append(e)
    return seg


def run_world(W, S, steps, pool, mode, seed=0):
    """One world: mode "ref" = the loop of run_shipped (its statements, device "cpu"); "plain" / "nogc" =
    BC.shipped_loop_ab with an ABRecorder (record on). Returns the logits of every step, the after-pooling graphs of
    every re-warm step, the log segments, the final state and the A/B record."""
    BC, RN = _bc(), W.RN
    W.mp.setattr(W.CE, "POOL_BLOCKS", pool)
    W.mp.setattr(RN.BASE, "default_capture_stream", None)
    model, cache, tok, pos = build_world(W, S, steps, seed)
    logits, caps = [], {}
    shipped = model.decode_inference

    def recording(*a, **k):
        warm = not model.has_buffers
        lg = shipped(*a, **k)
        if warm:
            caps[len(logits)] = [L.after_pooling_graph for L in model.layers]
        logits.append(lg.detach().clone())
        return lg
    model.decode_inference = recording
    LOG.clear()
    out = dict(mode=mode)
    with torch.inference_mode():
        if mode == "ref":
            clock = TestClock()
            cu = torch.arange(0, B + 1, dtype=torch.int, device="cpu")
            model.has_buffers = False
            warm, rewarms = None, []
            ev = [clock.event() for _ in range(steps + 1)]
            for t in range(steps):
                l0 = cache.layers[0]
                need, why = BC.rewarm_need(cache.get_seq_length(0), l0.no_compress_k_len,
                                           l0.cached_compressed_max_seqlen, warm)
                if need:
                    warm = BC.warm_state(cache.get_seq_length(0), l0.no_compress_k_len, l0.cached_compressed_max_seqlen)
                    rewarms.append((t, why))
                    model.has_buffers = False
                clock.record(ev[t])
                lg = model.decode_inference(tok, cu, pos, cache)
                tok = lg[:, -1, :].argmax(dim=-1, keepdim=True)
                pos = pos + 1
            clock.record(ev[steps])
            out.update(rewarms=rewarms, ab=None, record=None)
        else:
            ab = RN.ABRecorder(mode == "nogc", True, NLAYERS)
            res = BC.shipped_loop_ab(model, cache, tok, pos, B, steps, TestClock(), ab, "cpu")
            meta = dict(batch=B, steps=steps, S0=S, capacity_C=63 + pool, pool_blocks=pool, attn_splits=4,
                        workload_sha256="cpu", doc_rows=list(range(B)))
            out.update(rewarms=res["rewarms"], ab=res["ab"], tokens=res["inputs"],
                       record=dict(tokens=res["inputs"].clone(), rewarms=[list(x) for x in res["rewarms"]],
                                   nogc=mode == "nogc", meta=meta, **ab.record_tensors()))
    assert torch.cuda.graph is RN.BASE, "the swap must be undone after the loop"
    out.update(logits=logits, caps=caps, seg=_segments(list(LOG), steps), state=_state(cache),
               stream=id(RN.BASE.default_capture_stream))
    return out


PLAIN_LAYER = ["sync", "gc", "empty", "stream_enter", "capture_begin", "capture_end", "stream_exit"]
NOGC_LAYER = ["stream_enter", "capture_begin", "capture_end", "stream_exit"]


@pytest.mark.parametrize("pool", [4, 0])
def test_MBS4_recap_nogc_equals_shipped_on_cpu_fakes(W, pool):
    RN = W.RN
    S, steps = 6128, 100
    worlds = {m: run_world(W, S, steps, pool, m) for m in ("ref", "plain", "nogc")}
    ref = worlds["ref"]
    whys = {w for _, w in ref["rewarms"]}
    assert {"first", "out_len", "width"} <= whys, ref["rewarms"]     # precondition: every re-warm kind is crossed
    if pool:
        assert W.ext.hits > 0 and W.ext.parks > 0, (W.ext.hits, W.ext.parks)   # precondition: the pool acted
    rw = [t for t, _ in ref["rewarms"]]
    assert sorted(ref["caps"]) == rw and len(ref["logits"]) == steps
    for m in ("plain", "nogc"):
        w = worlds[m]
        assert [tuple(x) for x in w["rewarms"]] == [tuple(x) for x in ref["rewarms"]], m
        bad = [t for t in range(steps) if not torch.equal(w["logits"][t], ref["logits"][t])]
        assert not bad, "%s: logits differ first at step %s" % (m, bad[:5])
        sd = [k for k in ref["state"] if not _eq(ref["state"][k], w["state"][k])]
        assert not sd, (m, sd)
        assert sorted(w["caps"]) == rw, m
        for t in rw:
            for l in range(NLAYERS):
                g, r = w["caps"][t][l], ref["caps"][t][l]
                assert g is not r and g.ended and r.ended
                assert g.body[0] is r.body[0], (m, t, l)                 # the same captured code
                assert g.args == r.args == ((), (("capture_error_mode", "global"),)), (m, t, l, g.args)
                assert not _sig_diff(g.sig, r.sig), (m, t, l, _sig_diff(g.sig, r.sig))
        toks = torch.stack([lg[:, -1, :].argmax(dim=-1) for lg in ref["logits"]], dim=1)
        assert torch.equal(w["tokens"][:, 1:], toks), m
        for t in rw:
            assert torch.equal(w["record"]["digests"][t], RN.logit_digest(ref["logits"][t])), (m, t)
        assert torch.equal(w["record"]["final"], RN.logit_digest(ref["logits"][-1])), m
    # the capture calls of every step: gc.collect NLAYERS times per re-warm as shipped, ONCE under --recap-nogc
    for m, w in worlds.items():
        assert sorted(w["seg"]) == list(range(steps)), m
        for t in range(steps):
            kinds = [e[0] for e in w["seg"][t]]
            if t not in rw:
                assert kinds == [], (m, t, kinds)
            elif m == "nogc":
                assert kinds == ["sync", "gc", "empty"] + NOGC_LAYER * NLAYERS, (m, t, kinds)
            else:
                assert kinds == PLAIN_LAYER * NLAYERS, (m, t, kinds)
            assert {e[1] for e in w["seg"][t] if e[0] in ("stream_enter", "stream_exit")} <= {w["stream"]}, (m, t)
        n_gc = sum(1 for t in rw for e in w["seg"][t] if e[0] == "gc")
        assert n_gc == len(rw) * (1 if m == "nogc" else NLAYERS), (m, n_gc)
    for m in ("plain", "nogc"):
        ab = worlds[m]["ab"]
        want = (1, NLAYERS) if m == "nogc" else (NLAYERS, 0)
        assert sorted(int(t) for t in ab["rows"]) == rw
        assert all((r["n_gc"], r["n_nogc_ctx"]) == want for r in ab["rows"].values()), (m, ab["rows"])
        assert all((r["release_ms"] is not None) == (m == "nogc") for r in ab["rows"].values()), m
        ok, detail = RN.rn_gc_gate(ab, rw)
        assert ok, detail
    c = RN.compare_ab(worlds["plain"]["record"], worlds["nogc"]["record"])
    assert c["passed"], c
    assert not RN.compare_ab(worlds["plain"]["record"], worlds["plain"]["record"])["passed"]   # not an A/B pair


# ------------------------------------------------------------------ MBS5
def test_MBS5_ab_compare_gates_and_verdicts():
    RN, BC = _rn(), _bc()

    def rec(nogc):
        return dict(tokens=torch.randint(0, 9, (3, 7), generator=torch.Generator().manual_seed(1)),
                    rewarms=[[0, "first"], [4, "out_len"]], digests={0: torch.arange(3), 4: torch.arange(3) + 10},
                    final=torch.arange(3) + 20, nogc=nogc,
                    meta=dict(batch=3, steps=6, S0=16128, capacity_C=128, pool_blocks=65, attn_splits=4,
                              workload_sha256="x", doc_rows=[0, 1, 2]))
    a = rec(False)
    c = RN.compare_ab(a, rec(True))
    assert c["passed"] and c["token_mismatches"] == 0 and c["first_token_mismatch"] is None and c["modes"] == [False, True]
    assert not RN.compare_ab(a, rec(False))["passed"]                  # two plain records are not an A/B pair
    b = rec(True)
    b["tokens"][1, 5] += 1
    c = RN.compare_ab(a, b)
    assert not c["passed"] and c["token_mismatches"] == 1 and c["first_token_mismatch"] == [1, 5]
    b = rec(True)
    b["digests"][4] = b["digests"][4] + 1
    c = RN.compare_ab(a, b)
    assert not c["passed"] and c["digest_mismatch_steps"] == [4]
    b = rec(True)
    b["final"] = b["final"] + 1
    assert not RN.compare_ab(a, b)["final_equal"] and not RN.compare_ab(a, b)["passed"]
    b = rec(True)
    b["meta"]["doc_rows"] = [0, 1, 1]
    assert RN.compare_ab(a, b)["meta_diff"] == ["doc_rows"] and not RN.compare_ab(a, b)["passed"]
    b = rec(True)
    b["rewarms"] = [[0, "first"], [4, "width"]]
    assert not RN.compare_ab(a, b)["rewarms_equal"] and not RN.compare_ab(a, b)["passed"]
    b = rec(True)
    del b["digests"][4]
    assert not RN.compare_ab(a, b)["digest_keys_ok"] and not RN.compare_ab(a, b)["passed"]
    b = rec(True)
    b["tokens"] = b["tokens"][:, :6]
    c = RN.compare_ab(a, b)
    assert c["token_mismatches"] is None and not c["passed"]
    # RN-GC
    row = lambda g, n, ms: dict(n_gc=g, n_nogc_ctx=n, release_ms=ms)   # noqa: E731
    ok_nogc = dict(nogc=True, n_layers=32, rows={"0": row(1, 32, 170.0), "64": row(1, 32, 160.0)})
    assert RN.rn_gc_gate(ok_nogc, [0, 64])[0]
    assert not RN.rn_gc_gate(ok_nogc, [0, 64, 128])[0]                  # a re-warm without a row
    for r in (row(2, 32, 170.0), row(1, 31, 170.0), row(1, 32, None)):
        assert not RN.rn_gc_gate(dict(ok_nogc, rows={"0": r, "64": row(1, 32, 160.0)}), [0, 64])[0], r
    ok_plain = dict(nogc=False, n_layers=32, rows={"0": row(32, 0, None)})
    assert RN.rn_gc_gate(ok_plain, [0])[0] and not RN.rn_gc_gate(dict(ok_plain, rows={"0": row(1, 32, 1.0)}), [0])[0]
    assert not RN.rn_gc_gate(dict(ok_plain, rows={}), [])[0]
    # the digest: the int64 sum of the int32 bit patterns; one ULP moves it by one
    g = torch.Generator().manual_seed(5)
    lg = torch.randn((5, 1, 11), generator=g)
    ref = lg[:, -1, :].contiguous().view(torch.int32).to(torch.int64).sum(dim=1)
    assert torch.equal(RN.logit_digest(lg, chunk=2), ref) and torch.equal(RN.logit_digest(lg), ref)
    lg2 = lg.clone()
    lg2[3, 0, 4] = torch.nextafter(lg2[3, 0, 4], torch.tensor(float("inf")))
    d2 = RN.logit_digest(lg2)
    assert int((d2 != ref).sum()) == 1 and abs(int(d2[3] - ref[3])) == 1
    with pytest.raises(TypeError):
        RN.logit_digest(lg.to(torch.bfloat16))
    # re-warm statistics, effective tok/s without step 0, the OOM test
    periods = [6000.0, 64.0, 64.0, 500.0, 64.0, 400.0]
    rs = BC.rewarm_stats(periods, [(0, "first"), (3, "out_len"), (5, "width")])
    assert rs == dict(n=3, first_ms=6000.0, later_median_ms=450.0, later_mean_ms=450.0, sum_ms=6900.0), rs
    assert BC.effective_excl_first(2, periods, [(0, "first")]) == round(1e3 * 2 * 5 / 1092.0, 1)
    assert BC.effective_excl_first(2, periods, [(1, "first")]) is None

    class OutOfMemoryError(RuntimeError):
        pass
    assert BC.is_oom(OutOfMemoryError("Tried to allocate 2.00 GiB")) and BC.is_oom(RuntimeError("CUDA error: out of memory"))
    assert not BC.is_oom(RuntimeError("an illegal memory access was encountered"))
    # the verdicts: UNREGISTERED without values, CONFIRMED / REFUTED with planted bands
    unreg = {k: None for k in BC.PRED_MBS}
    nogc_sh = dict(rates=dict(steady_step_ms=64.0, steady_tok_s=3375.0, effective_tok_s=2500.0), gpu_peak_alloc_GB=82.0,
                   rewarm_stats=dict(first_ms=6000.0, later_median_ms=450.0))
    plain_sh = dict(rates=dict(steady_step_ms=64.2, steady_tok_s=3364.5, effective_tok_s=1210.0), gpu_peak_alloc_GB=82.0)
    v = {n: x for n, x, _ in BC.mbs_verdicts(128, 216, nogc_sh, True, plain_sh, None, unreg)}
    assert v == {k: "UNREGISTERED" for k in ("MX-TOK", "MX-PEAK", "RN-RW-LATER", "RN-RW-FIRST", "RN-STEADY", "RN-EFF")}, v
    pred = dict(unreg, rn_steady_ratio=(0.99, 1.01), rn_eff_ratio={"128": (1.8, 2.2)}, rn_rewarm_later_ms=(100, 400),
                rn_rewarm_first_ms=(4000, 8000), mx_steady_tok_s={"128_216": (3300, 3450)}, mx_peak_ratio=(0.98, 1.02),
                mx_fits={"128_222": True})
    v = {n: x for n, x, _ in BC.mbs_verdicts(128, 216, nogc_sh, True, plain_sh, None, pred)}
    assert v == {"MX-TOK": "CONFIRMED", "MX-PEAK": "CONFIRMED", "RN-RW-LATER": "REFUTED", "RN-RW-FIRST": "CONFIRMED",
                 "RN-STEADY": "CONFIRMED", "RN-EFF": "CONFIRMED"}, v
    d = {n: x for n, _, x in BC.mbs_verdicts(128, 216, plain_sh, False, nogc_sh, None, pred)}   # the plain half
    assert "2.0661" in d["RN-EFF"] and "0.9969" in d["RN-STEADY"] and "RN-RW-LATER" not in d
    assert BC.mbs_verdicts(128, 222, None, True, None, "oom", pred) == [
        ("MX-FIT", "REFUTED", "C 128 B 222: oom (registered True)")]
    assert BC.mbs_verdicts(73, 369, None, True, None, "oom", pred)[0][1] == "UNREGISTERED"
    if BC.PRED_MBS["stamp"] is None:                                   # TBD: the lead registers the values
        assert all(x is None for x in BC.PRED_MBS.values())
    else:
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", BC.PRED_MBS["stamp"])
        assert all(x is not None for x in BC.PRED_MBS.values())


# ------------------------------------------------------------------ MBS6
def test_MBS6_batch_arithmetic():
    BC = _bc()
    lines = (NOSI / "cache_engine.py").read_text().splitlines()
    assert "_gpu_slots = self.topk + self.pool_blocks" in lines[201]                   # cache_engine.py:202
    assert "assert total_bsz * (S + self.max_gen_len) * _hd < 2**31" in lines[215]     # :216
    assert "assert total_bsz * _gpu_slots * self.block_size * _hd < 2**31" in lines[218]   # :219
    X = (16128 + 512 + 64) * 256
    assert 502 * X < 2 ** 31 <= 503 * X
    assert BC.int32_batch_limit(16128, 512, 128) == BC.int32_batch_limit(16128, 512, 63) == 502
    assert BC.int32_batch_limit(130816, 512, 128) == 63          # p0prof: 128K C 128 allows B 63 and refuses 64
    assert BC.int32_batch_limit(16128, 512, 4095) == (2 ** 31 - 1) // (4096 * 64 * 256)   # the GPU side can bind
    want = {63: (406, 419), 73: (358, 369), 81: (326, 336), 128: (216, 222)}
    got = {C: (BC.fit_batch(C, BC.MBS_BUDGET_GB["matched"]), BC.fit_batch(C, BC.MBS_BUDGET_GB["edge"])) for C in want}
    assert got == want, got
    for C, (bm, be) in want.items():
        assert BC.mbase_peak_model(C, bm) <= 82.04 < BC.mbase_peak_model(C, bm + 1), C
        assert BC.mbase_peak_model(C, be) <= 84.0 < BC.mbase_peak_model(C, be + 1), C
    assert BC.fit_batch(128, 82.04) == BC.mbase_batch(128) == 216        # mb1's B (2175791: 216 fits, 224 OOM)
    for C, Bm, peak in ((73, 318, 74.962), (81, 290, 74.923), (128, 216, 82.042)):   # dr73, dr81 (2195573), mb1
        assert abs(BC.mbase_peak_model(C, Bm) / peak - 1) < 0.005, (C, Bm, BC.mbase_peak_model(C, Bm))
    with pytest.raises(ValueError):
        BC.fit_batch(128, 84.0, S=130816)                               # the model holds at 16K only


# ------------------------------------------------------------------ MBS7
def _step(cell, holder, holders, sha="0" * 40):
    e = dict(_env(), DRYRUN="1", MBS_HOLDERS=str(holders))
    return subprocess.run(["bash", str(EFF / "r2_mbstrong_step.sh"), "--nosi", sha, holder, "t1", cell], env=e,
                          capture_output=True, text=True, timeout=300)


def _subcells(stdout):
    """DRYRUN: {sub-cell: (its description line, its printed command line)}."""
    lines = stdout.splitlines()
    out = {}
    for i, ln in enumerate(lines):
        m = re.match(r"R2 mbstrong step: sub-cell (\S+): (.*)$", ln)
        if m:
            out[m.group(1)] = (m.group(2), lines[i + 1] if i + 1 < len(lines) else "")
    return out


def test_MBS7_step_script(tmp_path):
    BC = _bc()
    txt = (EFF / "r2_mbstrong_step.sh").read_text()
    assert "NOSI=/mnt/beegfs/ojerbi/nosi-r2-mbstrong " in txt and "OUTROOT=/mnt/beegfs/ojerbi/r2-runs/mbstrong " in txt
    assert "BASE_SHA=ded9d412d81aef76e2c1b6983673e6eaa7010adc " in txt and "SELF=benchmarks/Efficiency/r2_mbstrong_step.sh" in txt
    assert "CLAIM=$OUTROOT/mbs_active_$JOB" in txt and 'mkdir "$CLAIM"' in txt          # one claim per holder
    assert 'git -C "$NOSI" diff --quiet "$BASE_SHA" "$NSHA" -- nosi benchmarks/Efficiency/r2_ragged_gate.py' in txt
    assert '--warmup 2 --r2 fused --arms SH' in txt and '--pool "$POOL" --mbase "${RX[@]}"' in txt
    assert "benchmarks/Efficiency/r2_recap_nogc.py" in txt
    old = (EFF / "r2_base_fused_step.sh").read_text()
    assert "nosi-r2-mbase" in old and "mbstrong" not in old                           # the copied script is untouched
    f = BC.fit_batch
    for C, paper, probes in ((73, 318, (f(73, 82.04), f(73, 84.0))), (81, 290, (f(81, 82.04), f(81, 84.0))),
                             (63, 344, (f(63, 82.04),))):
        assert "PAIR=(c%db%d c%db%dn)" % (C, paper, C, paper) in txt, C
        assert "PROBES=(%s)" % " ".join("c%db%dn" % (C, b) for b in probes) in txt, (C, probes)
    assert "mbs1) PAIR=(mb1 mb1n)" in txt and "mbx128) PROBES=(c128b%dn)" % f(128, 84.0) in txt
    assert "mbs0) PAIR=(c128b16 c128b16n)" in txt
    for bad in ("mbs2", "mbx74", "c73b318", "c74b318n", "c73b0n", "c73b503n", "c73b31an", "cb8n", "c73bn", "mb1n"):
        r = _step(bad, "116", tmp_path)
        assert r.returncode == 1 and "cell '%s'" % bad in r.stderr, (bad, r.returncode, r.stderr[-300:])
    sha = _clean_head()
    if sha is None:
        pytest.skip("not a clean checkout of %s: the full DRYRUN needs one" % STEP_ROOT)
    for h in ("116", "117"):
        (tmp_path / ("%s.gpu_uuid" % h)).write_text(UUID + "\n")
    r = _step("mbx73", "116", tmp_path, sha)
    if r.returncode != 0:
        pytest.skip("DRYRUN stopped on a pin outside the cell stage: %s" % r.stderr.strip()[-200:])
    assert "cells c73b318 c73b318n c73b358n c73b369n; STEP_MIN total 191" in r.stdout, r.stdout[-600:]
    sc = _subcells(r.stdout)
    want = {"c73b318": "C 73 (pool 10) B 318 mode plain role first steps 512 STEP_MIN 46 mem 340G",
            "c73b318n": "C 73 (pool 10) B 318 mode nogc role second steps 512 STEP_MIN 46 mem 340G",
            "c73b358n": "C 73 (pool 10) B 358 mode nogc role probe steps 512 STEP_MIN 49 mem 340G",
            "c73b369n": "C 73 (pool 10) B 369 mode nogc role probe steps 512 STEP_MIN 50 mem 340G"}
    assert {k: v[0] for k, v in sc.items()} == want, sc
    for name, (_, cmd) in sc.items():
        assert "NOSI_POOL_BLOCKS=10" in cmd and "--pool 10 --mbase" in cmd and "--steps 512 --warmup 2" in cmd
        assert ("--recap-nogc" in cmd) == name.endswith("n") and ("--fit-probe" in cmd) == (name in ("c73b358n", "c73b369n"))
        assert ("--ab-record" in cmd) == (name in ("c73b318", "c73b318n")) and "mbs_t1_116_%s " % name in cmd
    assert "--ab-ref /mnt/beegfs/ojerbi/r2-runs/mbstrong/mbs_t1_116_c73b318" in sc["c73b318n"][1]
    assert "--ab-ref" not in sc["c73b318"][1] and "--time=50" in sc["c73b369n"][1]
    r = _step("mbs1", "117", tmp_path, sha)                              # odd holder: the --recap-nogc half first
    assert r.returncode == 0 and "cells mb1n mb1; STEP_MIN total 76" in r.stdout, r.stdout[-400:]
    sc = _subcells(r.stdout)
    assert sc["mb1n"][0] == "C 128 (pool 65) B 216 mode nogc role first steps 512 STEP_MIN 38 mem 240G"
    assert sc["mb1"][0] == "C 128 (pool 65) B 216 mode plain role second steps 512 STEP_MIN 38 mem 240G"
    assert "--ab-ref /mnt/beegfs/ojerbi/r2-runs/mbstrong/mbs_t1_117_mb1n" in sc["mb1"][1]
    r = _step("c128b222n", "116", tmp_path, sha)
    assert r.returncode == 0 and "cells c128b222n; STEP_MIN total 38" in r.stdout
    assert _subcells(r.stdout)["c128b222n"][0] == "C 128 (pool 65) B 222 mode nogc role probe steps 512 STEP_MIN 38 mem 240G"
    r = _step("mbs0", "116", tmp_path, sha)                              # the preflight miniature
    assert r.returncode == 0 and "cells c128b16 c128b16n; STEP_MIN total 42" in r.stdout, r.stdout[-400:]
    sc = _subcells(r.stdout)
    assert sc["c128b16"][0] == "C 128 (pool 65) B 16 mode plain role first steps 512 STEP_MIN 21 mem 240G"
    assert "--ab-ref /mnt/beegfs/ojerbi/r2-runs/mbstrong/mbs_t1_116_c128b16" in sc["c128b16n"][1]
