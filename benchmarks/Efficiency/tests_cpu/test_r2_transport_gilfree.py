"""CPU tests of the "gilfree" transport (nosi/nosi/r2_transport_gilfree.py): agent E's GIL-free packer v2 behind K3X's
transport interface. The extension is BUILT from its pinned blobs (nosi-sustained-gilfree2 f2b981a: gilfree_pack.py and
gilfree_pack.cpp, copied out of the shared object store; the OpenMP build, as its own CPU tests do) and compared with
K3X's s0 transport (S0Transport over cpupack_core at 91e6e47, CPU backend) on the same plans.
THE C++ WORKER (feature/nosi-r2-gilfree-worker, Q-C2): GT1-GT6b run twice, parametrized by the worker: "python" = the
pinned v2 build and the Python feeder (the default; these tests' meaning before the worker), "cpp" = packer v3 from
this tree (benchmarks/Efficiency/gilfree_pack.{py,cpp}) and its C++ worker (asynchronous: the tests' sync point is
flush(), a no-op in the synchronous mode). GT4b: the worker's board publication, and a planted missing one. GT7: the
worker runs while the main thread HOLDS the GIL (negative control: the Python feeder cannot). GTG1: GPU-only (the later
GPU gate), S-C's exact wait without its submission wait, more than 1024 launches behind the spin.

    python -m pytest -q -p no:cacheprovider benchmarks/Efficiency/tests_cpu/test_r2_transport_gilfree.py
"""
from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
NOSI = ROOT / "nosi/nosi"
PKG = "r2gf"
PINS = {"gilfree_pack.py": ("f2b981ac4ea450a1fc9d993a2c5d2c9a10a7db43", "benchmarks/Efficiency/gilfree_pack.py",
                            "45c56f679b017436810bea8e16fa2719d642ba54"),
        "gilfree_pack.cpp": ("f2b981ac4ea450a1fc9d993a2c5d2c9a10a7db43", "benchmarks/Efficiency/gilfree_pack.cpp",
                             "df3e3c8026fb0a48cc2fe2597013c3f20ab1338b"),
        "cpupack_core.py": ("91e6e47", "benchmarks/Efficiency/cpupack_core.py", "54b071f7fe8746076bcfe1274e27629d517e70a1")}
L, B, H, D, BS, NB, NPOS = 2, 2, 2, 4, 64, 12, 10
WORKERS = ("python", "cpp")


def _load(mod):
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(NOSI)]
        sys.modules[PKG] = pkg
    full = PKG + "." + mod
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, str(NOSI / (mod + ".py")))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules[full]


GF = _load("r2_transport_gilfree")
TR = sys.modules[PKG + ".kstep.transfer"]
SCG = _load("r2_residency_sc")                                # the generation board (GT4)


def _by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def pinned(tmp_path_factory):
    """The pinned blobs written to a fresh directory (each checked by its blob id), the extension built (OpenMP) and
    cpupack_core loaded."""
    src = tmp_path_factory.mktemp("gilfree_src")
    for name, (rev, path, blob) in PINS.items():
        got = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "%s:%s" % (rev, path)], capture_output=True, text=True)
        assert got.returncode == 0 and got.stdout.strip() == blob, (name, got.stdout, got.stderr[-300:])
        body = subprocess.run(["git", "-C", str(ROOT), "show", blob], capture_output=True)
        assert body.returncode == 0
        (src / name).write_bytes(body.stdout)
    gp = _by_path(src / "gilfree_pack.py", "gilfree_pack_pinned")
    ext = gp.load(str(tmp_path_factory.mktemp("gilfree_build")))
    core = _by_path(src / "cpupack_core.py", "cpupack_core_pinned")
    return types.SimpleNamespace(gp=gp, ext=ext, core=core)


@pytest.fixture(scope="module")
def v3(tmp_path_factory):
    """Packer v3 of THIS tree (benchmarks/Efficiency/gilfree_pack.{py,cpp}: v2 + the C++ worker), built (OpenMP)."""
    gp = _by_path(ROOT / "benchmarks/Efficiency/gilfree_pack.py", "gilfree_pack_v3_tree")
    ext = gp.load(str(tmp_path_factory.mktemp("gilfree3_build")))
    assert int(ext.VERSION) == 3 and int(ext.SLOT_EXTRA_PER_GROUP) == 16 and int(ext.OPENMP) == 1
    return types.SimpleNamespace(gp=gp, ext=ext)


@pytest.fixture(params=WORKERS)
def pk(request, pinned):
    """The packer of one worker: "python" = the pinned v2 build, the Python feeder (the default; it does not need the
    v3 build); "cpp" = packer v3 and its C++ worker (kw: the transport's worker option)."""
    if request.param == "python":
        return types.SimpleNamespace(worker="python", ext=pinned.ext, gp=pinned.gp, kw={})
    v3 = request.getfixturevalue("v3")
    return types.SimpleNamespace(worker="cpp", ext=v3.ext, gp=v3.gp, kw=dict(worker="cpp"))


class Ev:
    def __init__(self, done=False):
        self.done = done


def _world(seed=0):
    g = torch.Generator().manual_seed(seed)
    host_k = [torch.randn(B, NB * BS, H, D, generator=g) for _ in range(L)]
    host_v = [torch.randn(B, NB * BS, H, D, generator=g) for _ in range(L)]
    store = lambda: [torch.zeros(B, NPOS * BS, H, D) for _ in range(L)]   # noqa: E731
    return host_k, host_v, store(), store(), store(), store()


def _plans(seed=1):
    """Three issues: (layer, [(b, h, block, slot)]), the destinations distinct over ALL issues (a later issue never
    overwrites an earlier one, so every transfer's host truth stays checkable), random blocks."""
    rnd = random.Random(seed)
    dst = rnd.sample([(b, h, s) for b in range(B) for h in range(H) for s in range(NPOS)], 27)
    out = []
    for i, layer in enumerate((0, 1, 0)):
        out.append((layer, [(b, h, rnd.randrange(NB), s) for (b, h, s) in dst[9 * i: 9 * (i + 1)]]))
    return out


def _transfers(layer, groups, seq0, hist=None):
    return [TR.Transfer(seq0 + i, "u%d" % b, layer, b, h, blk, s, 1, "draft", ("u%d" % b, 1, 1), hist_ev=hist)
            for i, (b, h, blk, s) in enumerate(groups)]


def _truth_ok(store_k, store_v, host_k, host_v, plans):
    for layer, groups in plans:
        for (b, h, blk, s) in groups:
            if not (torch.equal(store_k[layer][b, s * BS:(s + 1) * BS, h], host_k[layer][b, blk * BS:(blk + 1) * BS, h])
                    and torch.equal(store_v[layer][b, s * BS:(s + 1) * BS, h],
                                    host_v[layer][b, blk * BS:(blk + 1) * BS, h])):
                return False
    return True


def _raises_at_issue_or_force(gf, layer, trs, match):
    """A release's failure: raised at issue (the synchronous mode) or, asynchronous (the C++ worker), at force."""
    with pytest.raises(RuntimeError, match=match):
        gf.issue(layer, trs)
        gf.flush()
        gf.force(trs[0])


def test_GT1_gilfree_bytes_equal_s0(pinned, pk):
    """GT1: the gilfree transport (the OpenMP build, a team of 2, 4 groups per chunk so every issue spans chunks) and
    K3X's s0 transport land BITWISE the same store rows on the same three plans, and those rows are the host truth of
    every transfer; every transfer completes (done) and the team used is the team asked. Negative control: a plan with
    one block changed lands other bytes, and the same comparison sees it."""
    assert int(pk.ext.OPENMP) == 1
    host_k, host_v, gk, gv, sk, sv = _world()
    gf = GF.GilfreeTransport(pk.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                             cap_groups=4, **pk.kw)
    be = pinned.core.CpuBackend()
    s0 = TR.S0Transport(pinned.core, be, host_k, host_v, sk, sv, BS, D, H, B, NPOS, NB * BS, cap_groups=4, ring=2,
                        dtype=torch.float32)
    plans = _plans()
    seq = 0
    trs = []
    for layer, groups in plans:
        a = _transfers(layer, groups, seq)
        b = _transfers(layer, groups, seq)
        seq += len(groups)
        gf.issue(layer, a)
        s0.issue(layer, b)
        trs += a
    gf.flush()
    be.synchronize()
    assert all(gf.done(tr) for tr in trs)
    for l in range(L):
        assert torch.equal(gk[l], sk[l]) and torch.equal(gv[l], sv[l]), l
    assert _truth_ok(gk, gv, host_k, host_v, plans)
    assert gf.stats["threads_min"] == gf.stats["threads_max"] == 2 and gf.stats["releases"] == 3
    assert gf.threads_ok() and pk.gp.threads_ok(2, gf.stats)              # the adapter's rule is agent E's rule
    gf.close()
    # negative control: one block of the first issue changed
    host_k2, host_v2, gk2, gv2, _, _ = _world()
    gf2 = GF.GilfreeTransport(pk.ext, host_k2, host_v2, gk2, gv2, BS, D, threads=2, host_wait=lambda ev: None,
                              cap_groups=4, **pk.kw)
    bad = [(layer, list(groups)) for layer, groups in plans]
    b0, h0, k0, d0 = bad[0][1][0]
    bad[0][1][0] = (b0, h0, (k0 + 1) % NB, d0)
    for i, (layer, groups) in enumerate(bad):
        gf2.issue(layer, _transfers(layer, groups, 100 * (i + 1)))
    gf2.flush()
    gf2.close()
    assert not all(torch.equal(gk2[l], sk[l]) for l in range(L))


def test_GT2_order_rules_and_the_team(pk):
    """GT2: T4 -- every transfer's hist_ev (the tick's write-back of its host rows) is host-waited before the release
    reads the rows; the team rule -- more than one thread from an extension without OpenMP is refused. Negative
    control: the adapter with the T4 wait skipped leaves the event pending."""
    host_k, host_v, gk, gv, _, _ = _world()
    waited = []

    def host_wait(ev):
        waited.append(ev)
        ev.done = True
    gf = GF.GilfreeTransport(pk.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=host_wait, cap_groups=4,
                             **pk.kw)
    ev = Ev()
    gf.issue(0, _transfers(0, [(0, 0, 3, 1), (1, 1, 4, 2)], 0, hist=ev))
    gf.flush()
    assert ev.done and waited == [ev]
    gf.close()
    gf2 = GF.GilfreeTransport(pk.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=host_wait, cap_groups=4,
                              skip_hist_wait=True, **pk.kw)
    ev2 = Ev()
    gf2.issue(0, _transfers(0, [(0, 0, 5, 3)], 10, hist=ev2))
    gf2.flush()
    gf2.close()
    assert not ev2.done
    with pytest.raises(RuntimeError, match="without OpenMP"):
        GF.GilfreeTransport(types.SimpleNamespace(OPENMP=0), host_k, host_v, gk, gv, BS, D, threads=2,
                            host_wait=host_wait, **pk.kw)


def test_GT3_a_release_on_another_team_is_refused(pk, tmp_path_factory):
    """GT3 (FIX_DESIGN.md: packer v1 ran on one thread, unseen until its stats were read): the adapter (the C++ worker:
    the worker) refuses a release whose packing team differs from the requested team. Negative control (agent E's own):
    the extension built WITHOUT OpenMP (a serial loop that reports a team of 1), presented as an OpenMP build, with a
    team of 2 requested: the first release raises (at issue, or at force for the worker). The same serial build asked
    for its own team of 1 lands the host truth."""
    noomp = pk.gp.load(str(tmp_path_factory.mktemp("gilfree_build_noomp")), openmp=False)
    assert int(noomp.OPENMP) == 0
    liar = types.SimpleNamespace(OPENMP=1, NativePipe=noomp.NativePipe,
                                 SLOT_EXTRA_PER_GROUP=int(getattr(noomp, "SLOT_EXTRA_PER_GROUP", 0)))
    host_k, host_v, gk, gv, _, _ = _world()
    gf = GF.GilfreeTransport(liar, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None, cap_groups=4,
                             **pk.kw)
    layer, groups = _plans()[0]
    _raises_at_issue_or_force(gf, layer, _transfers(layer, groups, 0), "!= requested 2")
    gf.close()
    host_k, host_v, gk1, gv1, _, _ = _world()
    gf1 = GF.GilfreeTransport(noomp, host_k, host_v, gk1, gv1, BS, D, threads=1, host_wait=lambda ev: None,
                              cap_groups=4, **pk.kw)
    plans = _plans()
    for i, (layer, groups) in enumerate(plans):
        gf1.issue(layer, _transfers(layer, groups, 10 * i))
    gf1.flush()
    assert gf1.threads_ok() and _truth_ok(gk1, gv1, host_k, host_v, plans)
    gf1.close()


def test_GT4_a_release_publishes_its_slot_generations(pk):
    """GT4 (S-C's exact device wait, residency note 12f): with a generation board attached, every release publishes
    (slot, generation) of its transfers into the board after its rows (CPU mode: at once): exactly the destinations of
    the issued transfers hold their generations, every other entry stays 0, the rows are the host truth, and
    stats["published"] counts every packed transfer (FG1's published == groups). Negative control: the same issues
    without attach_board leave the board at 0 and publish nothing."""
    for attach in (True, False):
        host_k, host_v, gk, gv, _, _ = _world()
        gf = GF.GilfreeTransport(pk.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                 cap_groups=4, **pk.kw)
        board = SCG.GenBoard(L, B, H, NPOS, "cpu")
        if attach:
            gf.attach_board(board)
        want = torch.zeros_like(board.t)
        seq = 0
        plans = _plans()
        for layer, groups in plans:
            trs = _transfers(layer, groups, seq)
            for i, tr in enumerate(trs):
                tr.gen = seq + i + 2
                want[layer, tr.b, tr.h, tr.slot] = tr.gen
            seq += len(groups)
            gf.issue(layer, trs)
        gf.flush()
        assert _truth_ok(gk, gv, host_k, host_v, plans)
        assert torch.equal(board.t, want if attach else torch.zeros_like(want)), attach
        assert gf.stats["published"] == (seq if attach else 0) and gf.stats["groups"] == seq   # every packed transfer
        gf.close()


def test_GT4b_the_cpp_worker_publishes_and_a_missing_publication_is_seen(v3):
    """GT4b (the C++ worker's board publication: the pairs ride in the staging slot, one index_put_ per chunk after its
    rows): over 3 issues of 9 transfers in chunks of 4 groups, the board holds exactly every transfer's generation and
    the rows are the host truth. Negative control: the worker's publication planted off (set_skip_publish) lands the
    same rows, and the board comparison sees the missing publication (all 0, published 0)."""
    for planted in (False, True):
        host_k, host_v, gk, gv, _, _ = _world()
        gf = GF.GilfreeTransport(v3.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                 cap_groups=4, worker="cpp")
        board = SCG.GenBoard(L, B, H, NPOS, "cpu")
        gf.attach_board(board)
        if planted:
            gf.npipe.set_skip_publish(True)                    # NEGATIVE CONTROL
        want = torch.zeros_like(board.t)
        seq = 0
        plans = _plans()
        for layer, groups in plans:
            trs = _transfers(layer, groups, seq)
            for i, tr in enumerate(trs):
                tr.gen = 3 * (seq + i) + 7
                want[layer, tr.b, tr.h, tr.slot] = tr.gen
            seq += len(groups)
            gf.issue(layer, trs)
        gf.flush()
        gf.close()
        assert _truth_ok(gk, gv, host_k, host_v, plans) and gf.stats["chunks"] == 9 and gf.stats["groups"] == seq
        assert torch.equal(board.t, want) == (not planted), planted
        assert gf.stats["published"] == (0 if planted else seq) and int(board.t.count_nonzero()) == (0 if planted else seq)


@pytest.mark.parametrize("s_host, n_slots", [(16128 + 119 + 64, 80 + 32 + 4), (16128 + 160 + 64, 80 + 32 + 4),
                                             (16128 + 160 + 64, 80 + 32 + 5), (16128 + 64, 80 + 32 + 5),
                                             (16128 + 119 + 64, 96 + 32 + 4)])
def test_GT5_the_production_geometry(pk, s_host, n_slots):
    """GT5 (the FG / SG5 run on holder 2190876: "S and S_dst must be multiples of R"): the PRODUCTION geometry. The host
    rows per request are build_ragged's n_host = S_max + horizon + 64 (r2_decode_path.ragged_widths): SG5 p1 16128 +
    119 + 64 = 16311 (horizon 48 + LAMBDA 7 + 64), FG 16128 + 160 + 64 = 16352 (horizon 64 + 16 + 80); both are not
    multiples of 64; and 16128 + 64, a multiple. The store holds C + X + 4 slots per request (C 80 / X 32; + 5 with the
    trash slot; C 96). The adapter lands every transfer's rows equal to the host truth, the last complete host block,
    the first and the last slot of a request included; a block past the complete host blocks is refused by the adapter,
    before the packer. Negative control: the packer called as the adapter called it before (one batched (B, S_host, H,
    D) source per layer) refuses that geometry ("S and S_dst must be multiples of R") whenever S_host is not a multiple
    of 64."""
    Bp, Hp, Dp = 2, 2, 4
    g = torch.Generator().manual_seed(11)
    host_k = [torch.randn(Bp, s_host, Hp, Dp, generator=g) for _ in range(L)]
    host_v = [torch.randn(Bp, s_host, Hp, Dp, generator=g) for _ in range(L)]
    sk = [torch.zeros(Bp, n_slots * BS, Hp, Dp) for _ in range(L)]
    sv = [torch.zeros(Bp, n_slots * BS, Hp, Dp) for _ in range(L)]
    gf = GF.GilfreeTransport(pk.ext, host_k, host_v, sk, sv, BS, Dp, threads=2, host_wait=lambda ev: None,
                             cap_groups=4, **pk.kw)
    nblk = s_host // BS
    assert (gf.nblk, gf.n_slots, gf.S_floor) == (nblk, n_slots, nblk * BS)
    groups = [(0, 0, nblk - 1, n_slots - 1), (1, 1, nblk - 1, 0), (1, 0, 0, n_slots - 2), (0, 1, 7, 5)]
    gf.issue(1, _transfers(1, groups, 0))
    gf.flush()
    assert _truth_ok(sk, sv, host_k, host_v, [(1, groups)]) and gf.threads_ok()
    with pytest.raises(ValueError, match="outside the geometry"):
        gf.issue(0, _transfers(0, [(0, 0, nblk, 1)], 10))
    gf.close()
    if s_host % BS:
        elem = host_k[0].element_size()
        nb = 4 * BS * 8 + 4 * int(getattr(pk.ext, "SLOT_EXTRA_PER_GROUP", 0)) + 2 * 4 * BS * Dp * elem
        npipe = pk.ext.NativePipe(torch.zeros((2, nb), dtype=torch.uint8), torch.zeros((2, nb), dtype=torch.uint8),
                                  4, BS, [], [])
        npipe.set_sources(host_k, host_v)
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64)   # noqa: E731
        with pytest.raises(RuntimeError, match="multiples of R"):
            npipe.release(i64([0]), i64([0]), i64([1]), i64([3]), i64([2]), sk[1].view(-1, Dp), sv[1].view(-1, Dp))


def test_GT5b_a_geometry_outside_the_contract_is_refused_at_construction(pk):
    """GT5b (E's R contract, FIX_DESIGN.md at f2b981a): the adapter refuses, at construction and naming the side, every
    geometry the packer would refuse or misread: a HEAD-MAJOR (B, H, S, D) host (the packer would read S = H), a store
    whose rows per request are not whole 64-row slots (DESTINATION), host rows without one complete block (SOURCE), a
    store of another B. Positive control: a token-major host whose rows are not a multiple of 64 (5 x 64 + 55) and a
    store of C + X + 5 whole slots are accepted (GT5 checks the bytes of such geometries)."""
    Bp, Hp, Dp, s_host, n_slots = 2, 2, 4, 5 * BS + 55, 80 + 32 + 5

    def build(host_shape, store_shape):
        hk = [torch.zeros(*host_shape) for _ in range(L)]
        hv = [torch.zeros(*host_shape) for _ in range(L)]
        sk = [torch.zeros(*store_shape) for _ in range(L)]
        sv = [torch.zeros(*store_shape) for _ in range(L)]
        return GF.GilfreeTransport(pk.ext, hk, hv, sk, sv, BS, Dp, threads=2, host_wait=lambda ev: None,
                                   cap_groups=4, **pk.kw)
    gf = build((Bp, s_host, Hp, Dp), (Bp, n_slots * BS, Hp, Dp))
    assert (gf.nblk, gf.n_slots) == (5, n_slots)
    gf.close()
    for host_shape, store_shape, match in (
            ((Bp, Hp, s_host, Dp), (Bp, n_slots * BS, Hp, Dp), "must both be TOKEN-MAJOR"),
            ((Bp, s_host, Hp, Dp), (Bp, n_slots * BS + 32, Hp, Dp), "DESTINATION: the store's rows per request"),
            ((Bp, BS - 1, Hp, Dp), (Bp, n_slots * BS, Hp, Dp), "SOURCE: the host rows .* hold no complete block"),
            ((Bp, s_host, Hp, Dp), (Bp + 1, n_slots * BS, Hp, Dp), "must both be TOKEN-MAJOR")):
        with pytest.raises(ValueError, match=match):
            build(host_shape, store_shape)


@pytest.mark.parametrize("where", ["built_in_inference_mode", "inference_tensors_only"])
def test_GT6_the_feeder_thread_enters_the_inference_mode(pk, where):
    """GT6 (SG5 p2 on holder 2190876: "Inplace update to inference tensor outside InferenceMode is not allowed", raised
    by the packer in the feeder thread): torch's inference mode is THREAD-LOCAL. The host rows and the store are made
    under torch.inference_mode() on the main thread (as the tick makes them); built_in_inference_mode: the adapter too;
    inference_tensors_only: the adapter is built outside inference mode from those inference tensors. The releases run
    in the BACKGROUND feeder thread (the C++ worker: the worker thread): it enters inference mode (grad off), and a
    two-request plan lands the host truth. Negative control: thread_modes=False (the feeder keeps a new thread's
    defaults: inference off, grad on) reproduces the run's error, re-raised at force()."""
    def build(thread_modes):
        with torch.inference_mode():
            host_k, host_v, gk, gv, _, _ = _world()
            gf = None
            if where == "built_in_inference_mode":
                gf = GF.GilfreeTransport(pk.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                         cap_groups=4, background=True, thread_modes=thread_modes, **pk.kw)
        assert host_k[0].is_inference() and gk[0].is_inference()
        if gf is None:
            gf = GF.GilfreeTransport(pk.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                     cap_groups=4, background=True, thread_modes=thread_modes, **pk.kw)
        return gf, host_k, host_v, gk, gv
    groups = [(0, 0, 3, 1), (1, 1, 4, 2), (1, 0, NB - 1, NPOS - 1), (0, 1, 0, 0), (1, 1, 7, 5)]
    gf, host_k, host_v, gk, gv = build(True)
    assert gf.thread_inference and not gf.thread_grad
    with torch.inference_mode():
        trs = _transfers(0, groups, 0)
        gf.issue(0, trs)
        for tr in trs:
            gf.force(tr)
    gf.close()
    assert gf.feeder_seen == dict(inference=True, grad=False)
    assert _truth_ok(gk, gv, host_k, host_v, [(0, groups)]) and gf.threads_ok() and gf.stats["groups"] == len(groups)
    gf2, _, _, _, _ = build(False)
    with torch.inference_mode():
        trs2 = _transfers(0, groups, 10)
        gf2.issue(0, trs2)
        with pytest.raises(RuntimeError, match="Inplace update to inference tensor outside InferenceMode"):
            gf2.force(trs2[0])
    gf2.close()
    assert gf2.feeder_seen == dict(inference=False, grad=True)


def test_GT6b_the_feeder_publishes_into_an_inference_board(pk):
    """GT6b (FG1's exact arms: the PUBLICATION path, residency note 12f): the feeder's second in-place write is the
    board publication (_publish: board.t index_put_; the C++ worker: its index_put_). Here the host rows, the store and
    the adapter are NOT inference tensors (built outside inference mode), only the generation board is (S-C builds it
    in the tick's inference mode). attach_board turns the feeder's inference mode on; the background feeder lands the
    host truth and publishes every transfer's (slot, generation). Negative control: the feeder without the board's mode
    (thread_inference forced off after attach_board, as 7ab1ca2 had it) packs and lands the rows, then the publication
    raises the run's error, re-raised at force(); the Python feeder's traceback names the frames _feeder, _release and
    _publish (the C++ worker's error is its message). The modes checked are the feeder thread's own: the Python
    feeder records them per release; the C++ worker reports its thread-local state (NativePipe.modes_seen, read in
    the worker at its start and before each job); None before the first release."""
    import traceback
    for neg in (False, True):
        host_k, host_v, gk, gv, _, _ = _world()
        gf = GF.GilfreeTransport(pk.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                 cap_groups=4, background=True, **pk.kw)
        assert not gf.thread_inference and not gk[0].is_inference()
        with torch.inference_mode():
            board = SCG.GenBoard(L, B, H, NPOS, "cpu")
        assert board.t.is_inference()
        gf.attach_board(board)
        assert gf.thread_inference and not gf.thread_grad
        if neg:
            gf.thread_inference, gf.thread_grad = False, True     # NEGATIVE CONTROL: the board's mode ignored
        groups = [(0, 0, 3, 1), (1, 1, 4, 2), (1, 0, NB - 1, NPOS - 1)]
        trs = _transfers(1, groups, 0)
        want = torch.zeros_like(board.t)
        for i, tr in enumerate(trs):
            tr.gen = i + 2
            want[1, tr.b, tr.h, tr.slot] = tr.gen
        assert gf.feeder_seen is None                           # no release ran yet (cpp: the worker not started)
        gf.issue(1, trs)
        if neg:
            with pytest.raises(RuntimeError, match="Inplace update to inference tensor outside InferenceMode") as ei:
                gf.force(trs[0])
            if pk.worker == "python":
                tb = "".join(traceback.format_exception(ei.type, ei.value, ei.value.__traceback__))
                assert all(f in tb for f in ("_feeder", "_release", "_publish")), tb[-600:]
            assert gf.feeder_seen == dict(inference=False, grad=True) and gf.stats["published"] == 0
        else:
            for tr in trs:
                gf.force(tr)
            assert gf.feeder_seen == dict(inference=True, grad=False)
            assert torch.equal(board.t, want) and gf.stats["published"] == len(groups)
        gf.close()
        assert _truth_ok(gk, gv, host_k, host_v, [(1, groups)]), neg


@pytest.mark.parametrize("worker", ["cpp", "python"])
def test_GT7_the_cpp_worker_runs_while_the_main_thread_holds_the_gil(v3, worker):
    """GT7 (Q-C2: the captured tick's host must not wait for a pack's submission): packer v3's TEST call
    hold_gil_until runs issue() and then HOLDS THE GIL while it waits up to 3 s for the release to complete. The C++
    worker completes it meanwhile (no Python, no GIL): the wait ends early, and the rows are the host truth. Negative
    control: the Python feeder (the same v3 extension, worker="python", background) cannot dequeue the release while
    the GIL is held: the wait times out with nothing completed; the release completes once the GIL is free. (The
    switch interval is raised meanwhile, so the interpreter does not hand the GIL to the feeder during issue().)"""
    host_k, host_v, gk, gv, _, _ = _world()
    gf = GF.GilfreeTransport(v3.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                             cap_groups=4, background=True, worker=worker)
    assert gf.submits_without_gil == (worker == "cpp")
    groups = [(0, 0, 3, 1), (1, 1, 4, 2), (1, 0, NB - 1, NPOS - 1)]
    trs = _transfers(0, groups, 0)
    old = sys.getswitchinterval()
    sys.setswitchinterval(10.0)
    try:
        reached, seen, ms = v3.ext.hold_gil_until(gf.npipe, lambda: gf.issue(0, trs), 1, 3000)
    finally:
        sys.setswitchinterval(old)
    for tr in trs:
        gf.force(tr)
    gf.close()
    assert _truth_ok(gk, gv, host_k, host_v, [(0, groups)]) and int(gf.npipe.completed) == 1
    if worker == "cpp":
        assert (reached, seen) == (1, 1) and ms < 3000, (reached, seen, ms)
    else:
        assert (reached, seen) == (0, 0) and ms >= 3000, (reached, seen, ms)   # NEGATIVE CONTROL


def _takes(seed=3, per_layer=9):
    """Q-B3 takes over both layers: (l, b, h, block, slot, gen), the destinations distinct per layer, slot < NPOS - 1
    (room for the planted shift), the generations distinct."""
    rnd = random.Random(seed)
    out = []
    for l in range(L):
        dst = rnd.sample([(b, h, s) for b in range(B) for h in range(H) for s in range(NPOS - 1)], per_layer)
        out += [(l, b, h, rnd.randrange(NB), s, 3 + len(out) + i) for i, (b, h, s) in enumerate(dst)]
    return out


def _desc(takes):
    return torch.tensor([list(col) for col in zip(*takes)], dtype=torch.int64)       # (6, n)


def _cpp(v3, host_wait=lambda ev: None, board=True):
    host_k, host_v, gk, gv, _, _ = _world()
    gf = GF.GilfreeTransport(v3.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=host_wait, cap_groups=4,
                             worker="cpp")
    bd = None
    if board:
        bd = SCG.GenBoard(L, B, H, NPOS, "cpu")
        gf.attach_board(bd)
    return gf, host_k, host_v, gk, gv, bd


def test_GT8_issue_desc_lands_what_issue_lands(v3):
    """GT8 (Q-B3, the CPU LOCKSTEP): the same takes over both layers, issued as per-layer Transfer objects
    (issue(layer, transfers), unchanged) and as ONE (6, n) descriptor (issue_desc), land BITWISE the same store rows
    and the same generation board; the rows are the host truth; groups and publications are counted alike; one job.
    Negative control: the descriptor path with a planted wrong slot mapping (set_plant_slot_shift(1): every take one
    slot off) does not land the same store."""
    takes = _takes()
    ga, hk, hv, ak, av, aboard = _cpp(v3)
    seq = 0
    for l in range(L):
        mine = [t for t in takes if t[0] == l]
        trs = [TR.Transfer(seq + i, "u%d" % b, l, b, h, blk, s, gen, "draft", ("u%d" % b, 1, 1))
               for i, (_, b, h, blk, s, gen) in enumerate(mine)]
        seq += len(trs)
        ga.issue(l, trs)
    ga.flush()
    ga.close()
    gb, _, _, bk, bv, bboard = _cpp(v3)
    s1 = gb.issue_desc(_desc(takes))
    gb.flush()
    assert s1 == 1 and gb.landed_upto() == 1
    gb.close()
    for l in range(L):
        assert torch.equal(ak[l], bk[l]) and torch.equal(av[l], bv[l]), l
    assert torch.equal(aboard.t, bboard.t) and int(bboard.t.count_nonzero()) == len(takes)
    by_layer = [(l, [(b, h, blk, s) for (ll, b, h, blk, s, _) in takes if ll == l]) for l in range(L)]
    assert _truth_ok(bk, bv, hk, hv, by_layer)
    assert gb.stats["groups"] == ga.stats["groups"] == len(takes) and gb.stats["published"] == len(takes)
    assert gb.stats["releases"] == 1 and ga.stats["releases"] == L and gb.threads_ok()
    gc, _, _, ck, cv, _ = _cpp(v3)
    gc.npipe.set_plant_slot_shift(1)                           # NEGATIVE CONTROL: a wrong slot mapping
    gc.issue_desc(_desc(takes))
    gc.flush()
    gc.close()
    assert not all(torch.equal(ak[l], ck[l]) and torch.equal(av[l], cv[l]) for l in range(L))


class _Stream:
    """The CPU stand-in of ONE device stream: its events complete in record order (waiting one completes the earlier)."""

    def __init__(self):
        self.done_upto, self.n = -1, 0

    def record(self):
        ev = _StreamEv(self, self.n)
        self.n += 1
        return ev


class _StreamEv:
    def __init__(self, stream, k):
        self.stream, self.k = stream, k

    def query(self):
        return self.k <= self.stream.done_upto

    def synchronize(self):
        self.stream.done_upto = max(self.stream.done_upto, self.k)


def test_GT9_reader_and_write_back_marks(v3):
    """GT9 (Q-B3, T3 / T4 as marks): three ticks of per-layer reader events and write-back events on ONE stream
    (completing in record order), registered by mark (mark_reader(tick * L + l), mark_hist(tick)). Every take's slot
    has a last-reader tick; issue_desc gets per layer the newest mark of its takes and the newest write-back tick.
    Then, before the rows land, every take's last-reader event is complete, and the job lands the host truth. A mark
    that was never registered is refused. Negative control: an OLDER reader mark for layer 1 (one tick early) leaves a
    take's last reader pending when its rows land."""
    takes = _takes()
    rt = []                                                    # each take's last-reader tick
    for i, t in enumerate(takes):
        rt.append(i % 2 if t[0] == 0 else i % 3)              # layer 0: ticks 0 / 1; layer 1: ticks 0 / 1 / 2
    assert max(r for r, t in zip(rt, takes) if t[0] == 1) == 2 and max(r for r, t in zip(rt, takes) if t[0] == 0) == 1

    def run(older):
        st = _Stream()
        gf, hk, hv, gk, gv, _ = _cpp(v3, host_wait=lambda ev: ev.synchronize(), board=False)
        evs = {}
        for t in range(3):
            for l in range(L):
                evs[(t, l)] = st.record()
                gf.mark_reader(t * L + l, evs[(t, l)])
            gf.mark_hist(t, st.record())
        marks = [max(r * L + l for r, tk in zip(rt, takes) if tk[0] == l) for l in range(L)]
        if older:
            marks[1] -= L                                      # NEGATIVE CONTROL: one tick early for layer 1
        gf.issue_desc(_desc(takes), torch.tensor(marks), hist_mark=1)
        pending = [i for i, (r, tk) in enumerate(zip(rt, takes)) if not evs[(r, tk[0])].query()]
        gf.flush()
        with pytest.raises(ValueError, match="never registered"):
            gf.issue_desc(_desc(takes[:1]), torch.tensor([-1, 99]), hist_mark=-1)
        gf.close()
        by_layer = [(l, [(b, h, blk, s) for (ll, b, h, blk, s, _) in takes if ll == l]) for l in range(L)]
        return pending, _truth_ok(gk, gv, hk, hv, by_layer)
    pending, ok = run(False)
    assert pending == [] and ok, pending
    pending2, ok2 = run(True)
    assert pending2 and ok2, pending2                          # detected: a take landed before its last reader


def test_GT10_the_job_watermark_grows_and_a_failed_job_never_lands(v3):
    """GT10 (Q-B3, host admission): landed_upto() over a run of descriptor jobs and one object job: it never shrinks,
    never passes the submitted jobs, and reaches the last job after flush. A failed job (a block past the host rows)
    never lands: the watermark stops below it (also after later good jobs), and its error is raised at the next poll
    (negative control of a watermark that would count jobs, not landings)."""
    gf, _, _, _, _, _ = _cpp(v3)
    takes = _takes()
    seen, seqs = [], []
    for k in range(6):
        if k == 3:
            gf.issue(0, [TR.Transfer(900 + i, "u%d" % b, 0, b, h, blk, s, gen, "draft", ("u%d" % b, 1, 1))
                         for i, (_, b, h, blk, s, gen) in enumerate(takes[:4])])
            seqs.append(gf._last_seq)
        else:
            seqs.append(gf.issue_desc(_desc(_takes(seed=10 + k))))
        seen.append(gf.landed_upto())
    gf.flush()
    seen.append(gf.landed_upto())
    assert seqs == list(range(1, 7)), seqs
    assert seen == sorted(seen) and seen[-1] == 6 and all(0 <= v <= s for v, s in zip(seen, seqs)), seen
    bad = gf.issue_desc(_desc([(0, 0, 0, NB, 1, 99)]))         # block NB: past the complete host blocks
    gf.flush()
    assert bad == 7 and gf.landed_upto() == 6
    with pytest.raises(RuntimeError, match="outside the geometry"):
        gf.poll()
    good = gf.issue_desc(_desc(takes))
    gf.flush()
    assert good == 8 and gf.landed_upto() == 6                 # a failed job never lands, nor what follows it
    gf.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only: the later GPU gate runs it")
def test_GTG1_gpu_exact_wait_without_the_submission_wait(v3):
    """GTG1 (GPU-ONLY; the later GPU gate): S-C's exact wait WITHOUT its submission wait (CudaDeviceOps._submitted,
    r2_residency_sc.py:675-683) on the C++ worker. The release's T4 event completes only after a 200M-cycle GPU sleep on
    a side stream, so the worker's job is still waiting when the main stream starts to spin on the generation board
    (r2_residency_sc._wait_kernel, compiled before); then 1500 launches go behind the spin (more than the ~1024-entry
    launch queue: the main thread blocks in a launch while the spin runs). The worker packs, lands and publishes without
    the GIL: the spin ends with err 0 after it waited (> 10 ms), every launch ran, the rows are the host truth, and
    torch's Stream.wait_event accepts the job's event (_JobEv.wait: S-C's superset path). Negative control: the worker's
    publication planted off (set_skip_publish): the spin ends by its bound with err 1, the launches still run."""
    dev = torch.device("cuda")
    Bp, Hp, Dp, nblk, npos = 2, 2, 64, 16, 12
    g = torch.Generator().manual_seed(5)

    def run(planted, max_spin):
        host_k = [torch.randn(Bp, nblk * BS, Hp, Dp, generator=g) for _ in range(L)]
        host_v = [torch.randn(Bp, nblk * BS, Hp, Dp, generator=g) for _ in range(L)]
        sk = [torch.zeros(Bp, npos * BS, Hp, Dp, device=dev) for _ in range(L)]
        sv = [torch.zeros(Bp, npos * BS, Hp, Dp, device=dev) for _ in range(L)]
        gf = GF.GilfreeTransport(v3.ext, host_k, host_v, sk, sv, BS, Dp, threads=2,
                                 host_wait=lambda ev: ev.synchronize(), cap_groups=4, worker="cpp")
        board = SCG.GenBoard(L, Bp, Hp, npos, dev)
        gf.attach_board(board)
        if planted:
            gf.npipe.set_skip_publish(True)                    # NEGATIVE CONTROL
        groups = [(b, h, (3 * b + h) % nblk, 2 * h + b) for b in range(Bp) for h in range(Hp)]
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.cuda._sleep(200_000_000)
            hist = torch.cuda.Event()
            hist.record(side)
        trs = _transfers(1, groups, 0, hist=hist)
        for i, tr in enumerate(trs):
            tr.gen = i + 5
        idx = torch.tensor([board.flat(1, tr.b, tr.h, tr.slot) for tr in trs], dtype=torch.int64, device=dev)
        gen = torch.tensor([tr.gen for tr in trs], dtype=torch.int64, device=dev)
        err = torch.zeros((), dtype=torch.int32, device=dev)
        x = torch.zeros((), device=dev)
        kern = SCG._wait_kernel()
        kern[(1,)](board.t, torch.full_like(idx, -1), gen, idx.shape[0], err, max_spin, BLOCK=1024)   # compile
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gf.issue(1, trs)                                       # queued: the host does not wait for the submission
        e0.record()
        kern[(1,)](board.t, idx, gen, idx.shape[0], err, max_spin, BLOCK=1024)
        e1.record()
        for _ in range(1500):
            x.add_(1.0)
        torch.cuda.current_stream().wait_event(trs[0].ev)      # _JobEv.wait (the superset protocol)
        torch.cuda.synchronize()
        gf.close()
        ok = all(torch.equal(sk[1][b, s * BS:(s + 1) * BS, h].cpu(), host_k[1][b, k * BS:(k + 1) * BS, h])
                 and torch.equal(sv[1][b, s * BS:(s + 1) * BS, h].cpu(), host_v[1][b, k * BS:(k + 1) * BS, h])
                 for (b, h, k, s) in groups)
        return int(err.item()), float(x.item()), e0.elapsed_time(e1), ok
    err, x, spin_ms, ok = run(False, 1 << 24)
    assert err == 0 and x == 1500.0 and ok and spin_ms > 10.0, (err, x, spin_ms, ok)
    err2, x2, _, ok2 = run(True, 1 << 16)
    assert err2 == 1 and x2 == 1500.0 and ok2, (err2, x2, ok2)
