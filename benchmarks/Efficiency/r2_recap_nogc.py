"""r2-mbstrong --recap-nogc: the shipped re-warm captures its 32 pooling + top-k graphs WITHOUT the release that
torch.cuda.graph runs on entry, after ONE release per re-warm. Also the A/B instrument of the shipped decode loop
(r2_base_cell.shipped_loop_ab). Notes: docs/NOTES_mbstrong.md.

THE COST (kstep-cpu-runs/analysis/plan/SCORING_p0all.md section 5, MEASURED): one shipped re-warm takes 5.53-5.75 s,
and gc takes 93-97 % of it. nosa_llama.py:459-460 opens `with torch.cuda.graph(self.after_pooling_graph)` once per
layer, so one re-warm opens 32 contexts. In torch 2.6.0, graph.__enter__ runs torch.cuda.synchronize(), gc.collect()
and torch.cuda.empty_cache() before capture_begin (torch/cuda/graphs.py:171-183). Each gc.collect() is a full
collection (gc_count [0, 0, 32] per re-warm in the record).

THE CHANGE (existing tools only, no kernel, no engine edit):
- GraphNoRelease is torch.cuda.graph with an __enter__ that omits its first three statements. It keeps the shipped
  __init__ (the shipped default side stream, no pool, capture_error_mode "global") and the shipped __exit__
  (capture_end, then the stream context). So the capture uses the shipped calls: the stream context and the raw
  CUDAGraph.capture_begin / capture_end (the pattern of the graphstep tree, nosi/nosi/graph_step.py capture_all).
- check_contract() reads the INSTALLED torch: the shipped __enter__ must be exactly the five statements SHIPPED_ENTER,
  GraphNoRelease.__enter__ exactly the last two, and GraphNoRelease must not define __exit__. Another torch refuses.
- capture_swap() runs ONE release (the same three calls, in the order of graph.__enter__), then makes torch.cuda.graph
  name GraphNoRelease for ONE decode call (the re-warm step), and restores the shipped class in a finally. nosi/ keeps
  the bytes of ded9d41 (the engine-findings test pins its line numbers; the step script pins its bytes).

WHY NOT gc.disable() AROUND THE LOOP: gc.disable() stops the AUTOMATIC collector only. An explicit gc.collect() runs a
full collection also when the collector is disabled (the Python documentation of gc.disable and gc.collect; the CPU
test MBS2 shows it). The 32 collections of a re-warm are explicit calls in graph.__enter__, so gc.disable() does not
remove them.

WHAT STAYS EQUAL: the three release calls enqueue no work on the capture stream and write no tensor that the captured
body reads. The replay runs on the current stream after the context closes, so the stream order of the eager work
before it does not change. The captured body allocates no memory (INFERRED: its outputs are the buffers that
nosa_llama.py:819-828 allocates before the captures). The CPU test MBS3 compares the captured bodies, every logit and
the state; the GPU A/B pair (RN-EQ) compares every token and the logits of every re-warm step and of the last step.
"""
from __future__ import annotations

import ast
import contextlib
import gc
import inspect
import json
import os
import textwrap
import time

import torch
from torch.cuda import graphs as _graphs

BASE = _graphs.graph            # the shipped context class (bound here: torch.cuda.graph can be swapped later)
SHIPPED_ENTER = ("torch.cuda.synchronize()", "gc.collect()", "torch.cuda.empty_cache()",
                 "self.stream_ctx.__enter__()",
                 "self.cuda_graph.capture_begin(*self.pool, capture_error_mode=self.capture_error_mode)")
N_RELEASE = 3                   # the first three statements of SHIPPED_ENTER are the release
INIT_PARAMS = ("self", "cuda_graph", "pool", "stream", "capture_error_mode")
META_KEYS = ("batch", "steps", "S0", "capacity_C", "pool_blocks", "attn_splits", "workload_sha256", "doc_rows")


class GraphNoRelease(BASE):
    """torch.cuda.graph without the release on entry. __init__ forces the stream that the shipped class uses (its
    lazy default_capture_stream, created on the SHIPPED class, graphs.py:159-160); __exit__ is the shipped one."""
    made = 0                     # contexts made (one per `with`): the RN-GC count

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="global"):
        if BASE.default_capture_stream is None:
            BASE.default_capture_stream = torch.cuda.Stream()
        GraphNoRelease.made += 1
        super().__init__(cuda_graph, pool=pool,
                         stream=stream if stream is not None else BASE.default_capture_stream,
                         capture_error_mode=capture_error_mode)

    def __enter__(self):
        self.stream_ctx.__enter__()
        self.cuda_graph.capture_begin(*self.pool, capture_error_mode=self.capture_error_mode)


def statements(fn) -> tuple:
    """The statements of a function body as ast.unparse text (the docstring and the comments left out)."""
    body = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return tuple(ast.unparse(s) for s in body)


def check_contract(base=None, ours=None) -> dict:
    """Refuse (RuntimeError) unless base.__enter__ is SHIPPED_ENTER, ours.__enter__ is SHIPPED_ENTER without its first
    N_RELEASE statements, ours defines no __exit__, ours is a subclass of base and base.__init__ takes INIT_PARAMS.
    Returns the record of what it read."""
    base = BASE if base is None else base
    ours = GraphNoRelease if ours is None else ours
    got = statements(base.__enter__)
    if got != SHIPPED_ENTER:
        raise RuntimeError("recap-nogc: torch.cuda.graph.__enter__ of this torch is %s, not the torch 2.6.0 "
                           "statements %s" % (list(got), list(SHIPPED_ENTER)))
    mine = statements(ours.__enter__)
    if mine != SHIPPED_ENTER[N_RELEASE:]:
        raise RuntimeError("recap-nogc: %s.__enter__ is %s, not %s" % (ours.__name__, list(mine),
                                                                       list(SHIPPED_ENTER[N_RELEASE:])))
    if "__exit__" in vars(ours):
        raise RuntimeError("recap-nogc: %s defines its own __exit__ (the shipped one must run)" % ours.__name__)
    if not issubclass(ours, base):
        raise RuntimeError("recap-nogc: %s is not a subclass of %s" % (ours.__name__, base.__name__))
    params = tuple(inspect.signature(base.__init__).parameters)
    if params != INIT_PARAMS:
        raise RuntimeError("recap-nogc: torch.cuda.graph.__init__ takes %s, not %s" % (params, INIT_PARAMS))
    return dict(torch=torch.__version__, graph_enter=list(got), no_release_enter=list(mine), init=list(params))


def release():
    """ONE release: the three calls of graph.__enter__ (graphs.py:173-175), in its order."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


@contextlib.contextmanager
def capture_swap(row: dict):
    """ONE release (its host ms into row["release_ms"]), then torch.cuda.graph = GraphNoRelease until exit; the
    shipped class comes back in a finally. Refuses when torch.cuda.graph is not the shipped class."""
    if torch.cuda.graph is not BASE:
        raise RuntimeError("recap-nogc: torch.cuda.graph is %r, not the shipped class (a swap is active)"
                           % (torch.cuda.graph,))
    t0 = time.perf_counter()
    release()
    row["release_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    torch.cuda.graph = GraphNoRelease
    try:
        yield
    finally:
        torch.cuda.graph = BASE


def logit_digest(lg, chunk: int = 32):
    """Per request: the int64 sum of the int32 bit patterns of its last-position fp32 logits (one ULP moves it). The
    int64 temporary is at most `chunk` rows (an int32 -> int64 cast copies, SCORING / NOTES_graphstep FIX 3)."""
    x = lg[:, -1, :]
    if x.dtype != torch.float32:
        raise TypeError("logit_digest: fp32 logits expected, got %s" % x.dtype)
    x = x.contiguous().view(torch.int32)
    out = torch.empty((x.shape[0],), dtype=torch.int64, device=x.device)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = x[i:i + chunk].to(torch.int64).sum(dim=1)
    return out


class ABRecorder:
    """The instrument of r2_base_cell.shipped_loop_ab. step(t, need) wraps one decode call. On a re-warm step it counts
    the gc.collect() calls (a counting wrapper over gc.collect for that call only) and the GraphNoRelease contexts, and
    with nogc it runs capture_swap. A steady step passes through. after(t, need, lg): the logits digest of a re-warm
    step (record only, on the device, no host sync). finish(lg): the digest of the last step (after the timed loop)."""

    def __init__(self, nogc: bool, record: bool, n_layers: int):
        self.nogc, self.record, self.n_layers = bool(nogc), bool(record), int(n_layers)
        self.rows, self.digests, self.final = {}, {}, None

    @contextlib.contextmanager
    def step(self, t: int, need: bool):
        if not need:
            yield
            return
        row = dict(n_gc=0, n_nogc_ctx=0, release_ms=None)
        orig = gc.collect

        def counting(*a, **k):
            row["n_gc"] += 1
            return orig(*a, **k)
        made0 = GraphNoRelease.made
        gc.collect = counting
        try:
            if self.nogc:
                with capture_swap(row):
                    yield
            else:
                yield
        finally:
            gc.collect = orig
            row["n_nogc_ctx"] = GraphNoRelease.made - made0
            self.rows[int(t)] = row

    def after(self, t: int, need: bool, lg):
        if need and self.record:
            self.digests[int(t)] = logit_digest(lg)

    def finish(self, lg):
        if self.record:
            self.final = logit_digest(lg)

    def result(self) -> dict:
        return dict(nogc=self.nogc, record=self.record, n_layers=self.n_layers,
                    rows={str(t): dict(r) for t, r in sorted(self.rows.items())})

    def record_tensors(self) -> dict:
        return dict(digests={int(t): d.detach().cpu() for t, d in sorted(self.digests.items())},
                    final=None if self.final is None else self.final.detach().cpu())


def rn_gc_gate(ab: dict, rewarm_steps) -> tuple:
    """RN-GC: one row per re-warm step; --recap-nogc: 1 gc.collect call, n_layers GraphNoRelease contexts and a release
    time in every row; plain: n_layers gc.collect calls (one per shipped context) and 0 GraphNoRelease contexts."""
    rows, n, nogc = ab["rows"], int(ab["n_layers"]), bool(ab["nogc"])
    steps = sorted(int(t) for t in rewarm_steps)
    keys = sorted(int(t) for t in rows)
    want = (1, n) if nogc else (n, 0)
    bad = {t: (r["n_gc"], r["n_nogc_ctx"]) for t, r in rows.items() if (r["n_gc"], r["n_nogc_ctx"]) != want}
    if nogc:
        bad.update({t: "no release time" for t, r in rows.items() if r["release_ms"] is None})
    ok = keys == steps and bool(keys) and not bad
    rel = [r["release_ms"] for _, r in sorted(rows.items(), key=lambda kv: int(kv[0]))]
    return ok, "%s: want (gc.collect calls, GraphNoRelease contexts) = %s in each of %d re-warms (rows at %s); " \
               "differ: %s; release ms %s" % ("recap-nogc" if nogc else "plain", want, len(steps), keys,
                                              bad or "none", rel if nogc else "n/a")


def compare_ab(ref: dict, mine: dict) -> dict:
    """RN-EQ: the A/B pair of one (C, B): one plain and one --recap-nogc record (tokens (B, N + 1), rewarms, digests
    {t: (B,)} at the re-warm steps, final (B,), nogc, meta). PASS: equal meta, different modes, every token equal, equal
    re-warm steps, a digest at every re-warm step in both, every digest equal, the final digest equal."""
    out = dict(meta_diff=[k for k in META_KEYS if ref["meta"].get(k) != mine["meta"].get(k)],
               modes=[bool(ref["nogc"]), bool(mine["nogc"])])
    a, b = ref["tokens"], mine["tokens"]
    if tuple(a.shape) != tuple(b.shape):
        out.update(token_mismatches=None, first_token_mismatch=None, token_shapes=[list(a.shape), list(b.shape)])
    else:
        bad = (a != b).nonzero()
        out.update(token_mismatches=int(bad.shape[0]),
                   first_token_mismatch=[int(x) for x in bad[0]] if bad.shape[0] else None)
    ra, rb = [tuple(x) for x in ref["rewarms"]], [tuple(x) for x in mine["rewarms"]]
    steps = [int(t) for t, _ in rb]
    out["rewarms_equal"] = ra == rb
    out["digest_keys_ok"] = sorted(ref["digests"]) == sorted(mine["digests"]) == steps
    out["digest_mismatch_steps"] = [t for t in steps if t in ref["digests"] and t in mine["digests"]
                                    and not torch.equal(ref["digests"][t], mine["digests"][t])]
    out["final_equal"] = (ref["final"] is not None and mine["final"] is not None
                          and torch.equal(ref["final"], mine["final"]))
    out["passed"] = bool(not out["meta_diff"] and out["modes"][0] != out["modes"][1] and out["token_mismatches"] == 0
                         and out["rewarms_equal"] and out["digest_keys_ok"] and not out["digest_mismatch_steps"]
                         and out["final_equal"])
    return out


def load_ref(path: str) -> dict:
    """The other half of the A/B pair: <path>/ab_record.pt and <path>/summary.json (it must hold an SH arm)."""
    rec = torch.load(os.path.join(path, "ab_record.pt"), map_location="cpu", weights_only=True)
    missing = [k for k in ("tokens", "rewarms", "digests", "final", "nogc", "meta") if k not in rec]
    if missing:
        raise ValueError("%s/ab_record.pt lacks %s" % (path, missing))
    with open(os.path.join(path, "summary.json")) as f:
        summ = json.load(f)
    if not summ.get("SH"):
        raise ValueError("%s/summary.json has no SH arm" % path)
    return dict(record=rec, summary=summ)
