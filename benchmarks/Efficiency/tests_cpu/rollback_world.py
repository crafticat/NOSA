"""The CPU WORLD of tests_cpu/test_rollback.py (Task 1.5; not a test file). Load it by file path:

    spec = importlib.util.spec_from_file_location("rollback_world", <tests_cpu>/rollback_world.py)

It builds on tests_cpu/ragged_world.py (Task 1.3, read its docstring first): the SHIPPED engine runs from its own
source, the ragged engine and the R2 modules from theirs, and every CUDA op is a torch stand-in that copies the kernel's
FLAT-ADDRESS arithmetic (lesson L28): diff_offload, pool_update (per (head, request) row), flash_pool_swap (loads before
stores, P from the tensor rows), the two host gathers (source rows masked by the host rows, the bias by its own
strides), the pooling kernels, the stage-1 op, split-KV attention, torch.topk with CUDA's sorted=False order, FakeGraph
for the captured pooling graph. The rollback module (nosi/nosi/ragged_rollback.py) and state_snapshot.py load from
the same synthetic package, so their kernels are the same stand-ins.

THIS FILE ADDS
  define_undefined(rs)   the KNOWN PITFALL of the sibling tasks: allocate / prefill_update make _new_block_map_buf and
                         _load_mask with torch.empty_like; two worlds of one seed then hold other bytes there before
                         step 0. They are per-step scratch: one value for both. Every data row is defined already
                         (ragged_world poisons the unwritten window / tail / host rows with NaN; the pool rows are zero).
                         The tests assert "two worlds bitwise equal before step 0" on the WHOLE state.
  state(rs), diff()      every state tensor of a RaggedState (engine, pool, host rows, twin, host mirror, err) and the
                         LRU clock; diff compares bitwise (NaN bits included).
  rows_of(st, b)         the rows of request b (the global entries dropped).
  Driver                 feeds one world: request b at local position k[b]; step u of a group reads tok[b, k[b] + u] at
                         position S_b + k[b] + u (ragged_world.run_solo's teacher forcing).
  journal_facts          host facts of an open group (the preconditions of the tests).
  DeferredStream         a CPU executor of commit()'s work that runs it only when an event is waited or drained.
  live_slot_diff         the LIVE bytes of the window and pool slots (the journal-overflow test).
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(HERE / (name + ".py")))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


W = _load("ragged_world")
RE, RB, TW = W.RE, W.RB, W.TW
RR = W.pkg_load("ragged_rollback")
SS = W.pkg_load("state_snapshot")
L, B_MAX, BS, TOPK, P, V, HKV = W.L, W.B_MAX, W.BS, W.TOPK, W.P, W.V, W.HKV
K = 3
CAP = 1024                          # journal slots per layer and step of the exact tests (overflow 0 is asserted)
NONE_, SWAP, MOVE_IN, MOVE_OUT = RR.NONE_, RR.SWAP, RR.MOVE_IN, RR.MOVE_OUT
MAPS = ("_block_map", "_new_block_map_buf", "_load_mask", "_pool_map", "_pool_age", "_pool_target", "_pool_action")
ROWS = ("_k_gpu", "_v_gpu", "_kv_bias_gpu", "_k_cpu", "_v_cpu", "_cache_lens")
ENGINE = ROWS + MAPS
TWIN = ("pos", "n_comp", "n_ccis", "epoch", "prefilled", "comp_k", "ring", "total_cis", "ccis", "overflow")
SCRATCH = ("_new_block_map_buf", "_load_mask", "_pool_target", "_pool_action")
SLOTS = ("_k_gpu", "_v_gpu", "_kv_bias_gpu")
FAMILY = dict(slots=SLOTS, meta=("_block_map", "_new_block_map_buf", "_load_mask", "_cache_lens"),
              pool=("_pool_map", "_pool_age", "_pool_target", "_pool_action"), twin=tuple("twin." + n for n in TWIN),
              host=("_k_cpu", "_v_cpu"))
NEG_FAMILY = dict(no_slot_restore="slots", no_tail_restore="slots", no_meta_restore="meta", no_pool_restore="pool",
                  no_twin_restore="twin")


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def define_undefined(rs, value: int = -1) -> None:
    """Give the per-step scratch _new_block_map_buf / _load_mask one value (module docstring, KNOWN PITFALL)."""
    for lay in rs.cache.layers:
        e = lay.cache_engine
        e._new_block_map_buf.fill_(value)
        e._load_mask.fill_(value)


def state(rs) -> dict:
    out = {}
    for l, lay in enumerate(rs.cache.layers):
        e = lay.cache_engine
        for n in ENGINE:
            t = getattr(e, n, None)
            if t is not None:
                out["L%d.%s" % (l, n)] = t.detach().clone()
        out["L%d._pool_stamp_base" % l] = getattr(e, "_pool_stamp_base", None)
        tw = rs.st.twins[l]
        for n in TWIN:
            out["L%d.twin.%s" % (l, n)] = getattr(tw, n).detach().clone()
    out["pos_host"] = rs.st.pos_host.clone()
    out["err"] = rs.err.clone()
    return out


def same(x, y) -> bool:
    if torch.is_tensor(x) or torch.is_tensor(y):
        return (torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape
                and torch.equal(W.bits(x), W.bits(y)))
    return x == y


def diff(a: dict, b: dict, skip=()) -> list:
    """The names whose values differ (bitwise for tensors, NaN bits included); a name that ends with one of skip is not
    compared."""
    skip = tuple(skip)
    return [n for n in sorted(set(a) | set(b)) if not (skip and n.endswith(skip)) and not same(a.get(n), b.get(n))]


def in_family(names, fam: str) -> bool:
    return any(n.endswith(FAMILY[fam]) for n in names)


def rows_of(st: dict, b: int) -> dict:
    """The rows of request b in a state() dict. Dropped: the clocks and the other ints, err, the twin overflow flag."""
    out = {}
    for n, t in st.items():
        base = n.split(".")[-1]
        if not torch.is_tensor(t) or n == "err" or base == "overflow":
            continue
        out[n] = t[:, b] if (base in MAPS or base == "ccis") else t[b]
    return out


def snapshot_equal(c0, c1) -> list:
    """state_snapshot.CacheSnapshot of two takes: the names (layer, kind, field) that differ; [] = equal."""
    bad = []
    for i, (s0, s1) in enumerate(zip(c0.layers, c1.layers)):
        for kind in ("engine", "layer"):
            for n in sorted(set(s0[kind]) | set(s1[kind])):
                if not same(s0[kind].get(n), s1[kind].get(n)):
                    bad.append("%d.%s.%s" % (i, kind, n))
    if c0.top != c1.top:
        bad.append("top")
    return bad


def live_slot_diff(a: dict, b: dict) -> list:
    """The LIVE bytes of the window and pool slots of two states with EQUAL maps (checked first): a window slot that
    _block_map names (slot 63: its rows below pos % 64), a pool slot that _pool_map names. Returns the names whose live
    bytes differ."""
    bad = []
    for l in range(L):
        bm, pm = a["L%d._block_map" % l], a.get("L%d._pool_map" % l)
        assert same(bm, b["L%d._block_map" % l]) and same(pm, b.get("L%d._pool_map" % l)), "live_slot_diff: maps differ"
        live = torch.cat((bm >= 0, pm >= 0), 2) if pm is not None else bm >= 0          # (H, B, S)
        Bn, Sn = live.shape[1], live.shape[2]
        live[:, :, TOPK - 1] = False
        rows = live.permute(1, 2, 0).repeat_interleave(BS, dim=1)                       # (B, S * 64, H)
        tail = torch.arange(BS).view(1, BS) < (a["L%d.twin.pos" % l] % BS).view(Bn, 1)   # (B, 64)
        rows[:, (TOPK - 1) * BS:TOPK * BS, :] = tail.view(Bn, BS, 1).expand(Bn, BS, rows.shape[2])
        for n in SLOTS:
            x, y = a["L%d.%s" % (l, n)], b["L%d.%s" % (l, n)]
            if not torch.equal(W.bits(x)[rows], W.bits(y)[rows]):
                bad.append("L%d.%s" % (l, n))
    return bad


# ---------------------------------------------------------------------------
# worlds and drivers
# ---------------------------------------------------------------------------
def build(S_list, horizon: int = 96, seed: int = 0, ops=None, nopool: bool = False):
    """A ragged world (ragged_engine.build; the CPU EXACT ops unless ops is given), poisoned and defined."""
    model = W.make_model(seed)
    rows = W.prompt_rows(seed + 1, S_list)
    cache = W.new_cache(0)
    if nopool:
        for lay in cache.layers:
            lay.cache_engine.pool_blocks = 0
    rs = RE.build(model, cache, S_list, horizon, ops if ops is not None else W.ragged_ops(), lambda b, l: rows[(b, l)],
                  pin=False)
    W.poison_built(rs, S_list)
    define_undefined(rs)
    return model, rs


def tokens(seed: int, B: int, n: int = 96):
    g = torch.Generator().manual_seed(seed + 2)
    return torch.randint(0, V, (B, n), generator=g)


class Driver:
    """Feeds one world (module docstring)."""

    def __init__(self, model, rs, S_list, tok):
        self.model, self.rs, self.tok = model, rs, tok
        self.S = torch.tensor(S_list, dtype=torch.int64)
        self.B = len(S_list)
        self.k = torch.zeros(self.B, dtype=torch.int64)

    def positions(self, u: int = 0):
        return self.S + self.k + u

    def step(self, u: int = 0, act=None, wrong=None):
        """wrong (B,) bool: these requests read another token (the true one + 1): a rejected draft token, whose K / V
        differ from the rows that the kept step at that position writes later."""
        act = torch.ones(self.B, dtype=torch.bool) if act is None else act
        ids = self.tok[torch.arange(self.B), self.k + u]
        if wrong is not None:
            ids = torch.where(wrong, (ids + 1) % V, ids)
        pos = self.positions(u).view(self.B, 1).to(torch.int32)
        RE.set_activity(self.rs, act, act)
        return RE.decode_step(self.model, self.rs, ids.view(self.B, 1), pos)[:, -1].clone()

    def plain(self, n: int):
        out = []
        for _ in range(n):
            out.append(self.step())
            self.k += 1
        return out

    def group(self, rb, n_acc, steps: int = K, after_step=None, before_commit=None, wrong: bool = False):
        """A rollback group: begin, `steps` tentative steps (every request active), commit(n_acc). wrong: a step that
        request b rejects (u >= n_acc[b]) reads a wrong token (Driver.step), as a rejected draft does."""
        rb.begin()
        lg = []
        for u in range(steps):
            lg.append(self.step(u, wrong=(u >= n_acc) if wrong else None))
            if after_step is not None:
                after_step(u)
        if before_commit is not None:
            before_commit()
        rb.commit(n_acc.clone(), n_acc.clone())
        self.k += n_acc.clamp(0, steps)
        return lg

    def masked(self, n_acc, steps: int = K):
        """The reference of a group: `steps` plain steps, request b active iff u < n_acc[b]: it runs only the kept
        steps."""
        lg = []
        for u in range(steps):
            lg.append(self.step(u, u < n_acc))
        self.k += n_acc.clamp(0, steps)
        return lg


def kept_logits_equal(lg_a, lg_b, n_acc) -> list:
    """[(u, b)] of the kept rows (u < n_acc[b]) whose logits differ bitwise."""
    bad = []
    for u, (x, y) in enumerate(zip(lg_a, lg_b)):
        for b in range(x.shape[0]):
            if u < int(n_acc[b]) and not torch.equal(W.bits(x[b]), W.bits(y[b])):
                bad.append((u, b))
    return bad


def fills(S_list, k, n_acc, steps: int = K):
    """Host facts: [(request, step u, kept?)] of the tentative steps that fill a block ((p + 1) % 64 == 0)."""
    out = []
    for b, S in enumerate(S_list):
        for u in range(steps):
            if (S + int(k[b]) + u + 1) % BS == 0:
                out.append((b, u, u < int(n_acc[b])))
    return out


def journal_facts(rb, n_acc, steps: int = K) -> dict:
    """Host facts of an OPEN group (call before commit): over (layer, step u), for the requests that reject u: the pool
    actions, the lost slots in the journal, the MOVE_OUTs whose victim was a LIVE block, the unsaved lost slots that held
    a block."""
    f = dict(swap=0, move_in=0, move_out=0, lost=0, live_victims=0, unsaved_live=0)
    for l in range(rb.L):
        for u in range(steps):
            j = rb.J[l][u]
            rej = n_acc <= u
            r3 = rej.view(1, -1, 1)
            if rb.pool:
                act = j["plan_action"].long()
                for a, key in ((SWAP, "swap"), (MOVE_IN, "move_in"), (MOVE_OUT, "move_out")):
                    f[key] += int(((act == a) & r3).sum())
                mo = (act == MOVE_OUT) & r3
                q = torch.where(mo, j["plan_target"], torch.zeros_like(j["plan_target"]))
                f["live_victims"] += int((mo & (j["_pool_map"].gather(2, q) >= 0)).sum())
                owner = torch.cat((j["_block_map"], j["_pool_map"]), 2)
            else:
                owner = j["_block_map"]
            ent = j["ent"][:rb.N]
            e = ent[ent >= 0]
            f["lost"] += int(rej[(e // rb.S) % rb.B].sum())
            f["unsaved_live"] += int((j["unsaved"] & (owner >= 0) & r3).sum())
    return f


# ---------------------------------------------------------------------------
# the deferred executor (test_rollback_frees_after_last_reader)
# ---------------------------------------------------------------------------
class Pending:
    """An event of DeferredStream: done once the work queued before it has run."""

    def __init__(self, ex, upto: int):
        self.ex, self.upto, self.done, self.ran_by = ex, upto, False, None

    def query(self) -> bool:
        return self.done

    def synchronize(self, by: str = "wait") -> None:
        if not self.done:
            self.ex.run_to(self.upto, by)


class DeferredStream:
    """A CPU executor of commit()'s device work: run() queues it; record() returns a Pending that completes only when it
    is waited (wait / synchronize) or the queue is drained. It models a stream whose work has not run yet when the host
    goes on."""

    def __init__(self):
        self.jobs, self.ran, self.events = [], 0, []

    def run(self, fn) -> None:
        self.jobs.append(fn)

    def record(self, device) -> Pending:
        ev = Pending(self, len(self.jobs))
        self.events.append(ev)
        return ev

    def wait(self, ev, device) -> None:
        ev.synchronize("wait")

    def run_to(self, n: int, by: str) -> None:
        while self.ran < n:
            fn = self.jobs[self.ran]
            self.ran += 1
            fn()
        for ev in self.events:
            if not ev.done and ev.upto <= self.ran:
                ev.done, ev.ran_by = True, by

    def drain(self) -> None:
        self.run_to(len(self.jobs), "drain")

    def pending(self) -> int:
        return len(self.jobs) - self.ran


# ---------------------------------------------------------------------------
# host syncs: the static scan and the run-time trap (test_no_host_sync_in_commit_path)
# ---------------------------------------------------------------------------
SYNC_ATTRS = {"item", "tolist", "cpu", "numpy", "nonzero", "synchronize", "masked_select", "unique",
              "unique_consecutive", "argwhere", "repeat_interleave", "equal", "allclose", "is_nonzero"}
SYNC_BUILTINS = {"int", "bool", "float", "complex", "any", "all", "sum", "min", "max", "list", "tuple", "sorted",
                 "next", "iter"}


def scan_source(src: str, names):
    """Static scan of the functions `names` in `src` for constructs that read a device value on the host. Rules: a call of
    .item / .tolist / .cpu / .numpy / .nonzero / .synchronize / ... (SYNC_ATTRS); a call of int / bool / float / any /
    sum / list / ... (SYNC_BUILTINS); torch.where with one argument; .to("cpu"); a Call or a Subscript inside the test
    of if / while / assert / a conditional expression (a tensor in a host branch); a for loop (or comprehension) over
    anything but range(...) or an ALL-CAPS constant. Returns (the functions found, [(function, line, rule)])."""
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in set(names)}
    hits = []
    for name, fn in fns.items():
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute) and f.attr in SYNC_ATTRS:
                    hits.append((name, n.lineno, "." + f.attr))
                elif isinstance(f, ast.Name) and f.id in SYNC_BUILTINS:
                    hits.append((name, n.lineno, f.id + "()"))
                elif isinstance(f, ast.Attribute) and f.attr == "where" and len(n.args) == 1 and not n.keywords:
                    hits.append((name, n.lineno, "where(cond)"))
                elif isinstance(f, ast.Attribute) and f.attr == "to" and any(
                        isinstance(x, ast.Constant) and x.value == "cpu" for x in list(n.args) + [k.value for k in n.keywords]):
                    hits.append((name, n.lineno, '.to("cpu")'))
            elif isinstance(n, (ast.If, ast.While, ast.Assert, ast.IfExp)):
                if any(isinstance(x, (ast.Call, ast.Subscript)) for x in ast.walk(n.test)):
                    hits.append((name, n.lineno, "branch on a call or subscript"))
            elif isinstance(n, (ast.For, ast.comprehension)):
                it = n.iter
                ok = (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range") or \
                    (isinstance(it, ast.Name) and it.id.isupper())
                if not ok:
                    hits.append((name, getattr(n, "lineno", fn.lineno), "for over a non-range"))
    return sorted(fns), hits


class SyncTrap:
    """Run-time trap: every call of a tensor-to-host method (item, tolist, cpu, numpy, nonzero, __bool__, __int__,
    __float__, __index__) whose DIRECT caller is the file `path` is recorded (function, line, method). Calls from other
    files (the kernel stand-ins of the CPU world model GPU kernels and read on the host) are not recorded."""

    NAMES = ("item", "tolist", "cpu", "numpy", "nonzero", "__bool__", "__int__", "__float__", "__index__")

    def __init__(self, path):
        self.path, self.hits, self.saved = str(path), [], {}

    def __enter__(self):
        for n in self.NAMES:
            orig = getattr(torch.Tensor, n)
            self.saved[n] = (n in torch.Tensor.__dict__, orig)

            def wrap(t, *a, _n=n, _o=orig, **kw):
                fr = sys._getframe(1)
                if fr.f_code.co_filename == self.path:
                    self.hits.append((fr.f_code.co_name, fr.f_lineno, _n))
                return _o(t, *a, **kw)
            setattr(torch.Tensor, n, wrap)
        return self

    def __exit__(self, *exc):
        for n, (own, orig) in self.saved.items():
            if own:
                setattr(torch.Tensor, n, orig)
            else:
                delattr(torch.Tensor, n)
        return False
