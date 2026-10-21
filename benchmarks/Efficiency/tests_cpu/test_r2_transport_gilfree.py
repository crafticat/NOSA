"""CPU tests of the "gilfree" transport (nosi/nosi/r2_transport_gilfree.py): agent E's GIL-free packer v2 behind K3X's
transport interface. The extension is BUILT from its pinned blobs (nosi-sustained-gilfree2 f2b981a: gilfree_pack.py and
gilfree_pack.cpp, copied out of the shared object store; the OpenMP build, as its own CPU tests do) and compared with
K3X's s0 transport (S0Transport over cpupack_core at 91e6e47, CPU backend) on the same plans.

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


def test_GT1_gilfree_bytes_equal_s0(pinned):
    """GT1: the gilfree transport (the OpenMP build, a team of 2, 4 groups per chunk so every issue spans chunks) and
    K3X's s0 transport land BITWISE the same store rows on the same three plans, and those rows are the host truth of
    every transfer; every transfer completes (done) and the team used is the team asked. Negative control: a plan with
    one block changed lands other bytes, and the same comparison sees it."""
    assert int(pinned.ext.OPENMP) == 1
    host_k, host_v, gk, gv, sk, sv = _world()
    gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                             cap_groups=4)
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
    be.synchronize()
    assert all(gf.done(tr) for tr in trs)
    for l in range(L):
        assert torch.equal(gk[l], sk[l]) and torch.equal(gv[l], sv[l]), l
    assert _truth_ok(gk, gv, host_k, host_v, plans)
    assert gf.stats["threads_min"] == gf.stats["threads_max"] == 2 and gf.stats["releases"] == 3
    assert gf.threads_ok() and pinned.gp.threads_ok(2, gf.stats)          # the adapter's rule is agent E's rule
    # negative control: one block of the first issue changed
    host_k2, host_v2, gk2, gv2, _, _ = _world()
    gf2 = GF.GilfreeTransport(pinned.ext, host_k2, host_v2, gk2, gv2, BS, D, threads=2, host_wait=lambda ev: None,
                              cap_groups=4)
    bad = [(layer, list(groups)) for layer, groups in plans]
    b0, h0, k0, d0 = bad[0][1][0]
    bad[0][1][0] = (b0, h0, (k0 + 1) % NB, d0)
    for i, (layer, groups) in enumerate(bad):
        gf2.issue(layer, _transfers(layer, groups, 100 * (i + 1)))
    assert not all(torch.equal(gk2[l], sk[l]) for l in range(L))


def test_GT2_order_rules_and_the_team(pinned):
    """GT2: T4 -- every transfer's hist_ev (the tick's write-back of its host rows) is host-waited before the release
    reads the rows; the team rule -- more than one thread from an extension without OpenMP is refused. Negative
    control: the adapter with the T4 wait skipped leaves the event pending."""
    host_k, host_v, gk, gv, _, _ = _world()
    waited = []

    def host_wait(ev):
        waited.append(ev)
        ev.done = True
    gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=host_wait, cap_groups=4)
    ev = Ev()
    gf.issue(0, _transfers(0, [(0, 0, 3, 1), (1, 1, 4, 2)], 0, hist=ev))
    assert ev.done and waited == [ev]
    gf2 = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=host_wait, cap_groups=4,
                              skip_hist_wait=True)
    ev2 = Ev()
    gf2.issue(0, _transfers(0, [(0, 0, 5, 3)], 10, hist=ev2))
    assert not ev2.done
    with pytest.raises(RuntimeError, match="without OpenMP"):
        GF.GilfreeTransport(types.SimpleNamespace(OPENMP=0), host_k, host_v, gk, gv, BS, D, threads=2,
                            host_wait=host_wait)


def test_GT3_a_release_on_another_team_is_refused(pinned, tmp_path_factory):
    """GT3 (FIX_DESIGN.md: packer v1 ran on one thread, unseen until its stats were read): the adapter refuses a
    release whose packing team differs from the requested team. Negative control (agent E's own): the extension built
    WITHOUT OpenMP (a serial loop that reports a team of 1), presented as an OpenMP build, with a team of 2 requested:
    the first release raises. The same serial build asked for its own team of 1 lands the host truth."""
    noomp = pinned.gp.load(str(tmp_path_factory.mktemp("gilfree_build_noomp")), openmp=False)
    assert int(noomp.OPENMP) == 0
    liar = types.SimpleNamespace(OPENMP=1, NativePipe=noomp.NativePipe)
    host_k, host_v, gk, gv, _, _ = _world()
    gf = GF.GilfreeTransport(liar, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None, cap_groups=4)
    layer, groups = _plans()[0]
    with pytest.raises(RuntimeError, match="!= requested 2"):
        gf.issue(layer, _transfers(layer, groups, 0))
    host_k, host_v, gk1, gv1, _, _ = _world()
    gf1 = GF.GilfreeTransport(noomp, host_k, host_v, gk1, gv1, BS, D, threads=1, host_wait=lambda ev: None,
                              cap_groups=4)
    plans = _plans()
    for i, (layer, groups) in enumerate(plans):
        gf1.issue(layer, _transfers(layer, groups, 10 * i))
    assert gf1.threads_ok() and _truth_ok(gk1, gv1, host_k, host_v, plans)


def test_GT4_a_release_publishes_its_slot_generations(pinned):
    """GT4 (S-C's exact device wait, residency note 12f): with a generation board attached, every release publishes
    (slot, generation) of its transfers into the board after its rows (CPU mode: at once): exactly the destinations of
    the issued transfers hold their generations, every other entry stays 0, the rows are the host truth, and
    stats["published"] counts every packed transfer (FG1's published == groups). Negative control: the same issues
    without attach_board leave the board at 0 and publish nothing."""
    for attach in (True, False):
        host_k, host_v, gk, gv, _, _ = _world()
        gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                 cap_groups=4)
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
        assert _truth_ok(gk, gv, host_k, host_v, plans)
        assert torch.equal(board.t, want if attach else torch.zeros_like(want)), attach
        assert gf.stats["published"] == (seq if attach else 0) and gf.stats["groups"] == seq   # every packed transfer


@pytest.mark.parametrize("s_host, n_slots", [(16128 + 119 + 64, 80 + 32 + 4), (16128 + 160 + 64, 80 + 32 + 4),
                                             (16128 + 160 + 64, 80 + 32 + 5), (16128 + 64, 80 + 32 + 5),
                                             (16128 + 119 + 64, 96 + 32 + 4)])
def test_GT5_the_production_geometry(pinned, s_host, n_slots):
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
    gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, sk, sv, BS, Dp, threads=2, host_wait=lambda ev: None,
                             cap_groups=4)
    nblk = s_host // BS
    assert (gf.nblk, gf.n_slots, gf.S_floor) == (nblk, n_slots, nblk * BS)
    groups = [(0, 0, nblk - 1, n_slots - 1), (1, 1, nblk - 1, 0), (1, 0, 0, n_slots - 2), (0, 1, 7, 5)]
    gf.issue(1, _transfers(1, groups, 0))
    assert _truth_ok(sk, sv, host_k, host_v, [(1, groups)]) and gf.threads_ok()
    with pytest.raises(ValueError, match="outside the geometry"):
        gf.issue(0, _transfers(0, [(0, 0, nblk, 1)], 10))
    if s_host % BS:
        elem = host_k[0].element_size()
        nb = 4 * BS * 8 + 2 * 4 * BS * Dp * elem
        npipe = pinned.ext.NativePipe(torch.zeros((2, nb), dtype=torch.uint8), torch.zeros((2, nb), dtype=torch.uint8),
                                      4, BS, [], [])
        npipe.set_sources(host_k, host_v)
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64)   # noqa: E731
        with pytest.raises(RuntimeError, match="multiples of R"):
            npipe.release(i64([0]), i64([0]), i64([1]), i64([3]), i64([2]), sk[1].view(-1, Dp), sv[1].view(-1, Dp))


def test_GT5b_a_geometry_outside_the_contract_is_refused_at_construction(pinned):
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
        return GF.GilfreeTransport(pinned.ext, hk, hv, sk, sv, BS, Dp, threads=2, host_wait=lambda ev: None,
                                   cap_groups=4)
    gf = build((Bp, s_host, Hp, Dp), (Bp, n_slots * BS, Hp, Dp))
    assert (gf.nblk, gf.n_slots) == (5, n_slots)
    for host_shape, store_shape, match in (
            ((Bp, Hp, s_host, Dp), (Bp, n_slots * BS, Hp, Dp), "must both be TOKEN-MAJOR"),
            ((Bp, s_host, Hp, Dp), (Bp, n_slots * BS + 32, Hp, Dp), "DESTINATION: the store's rows per request"),
            ((Bp, BS - 1, Hp, Dp), (Bp, n_slots * BS, Hp, Dp), "SOURCE: the host rows .* hold no complete block"),
            ((Bp, s_host, Hp, Dp), (Bp + 1, n_slots * BS, Hp, Dp), "must both be TOKEN-MAJOR")):
        with pytest.raises(ValueError, match=match):
            build(host_shape, store_shape)


@pytest.mark.parametrize("where", ["built_in_inference_mode", "inference_tensors_only"])
def test_GT6_the_feeder_thread_enters_the_inference_mode(pinned, where):
    """GT6 (SG5 p2 on holder 2190876: "Inplace update to inference tensor outside InferenceMode is not allowed", raised
    by the packer in the feeder thread): torch's inference mode is THREAD-LOCAL. The host rows and the store are made
    under torch.inference_mode() on the main thread (as the tick makes them); built_in_inference_mode: the adapter too;
    inference_tensors_only: the adapter is built outside inference mode from those inference tensors. The releases run
    in the BACKGROUND feeder thread: it enters inference mode (grad off), and a two-request plan lands the host truth.
    Negative control: thread_modes=False (the feeder keeps a new thread's defaults: inference off, grad on) reproduces
    the run's error, re-raised at force()."""
    def build(thread_modes):
        with torch.inference_mode():
            host_k, host_v, gk, gv, _, _ = _world()
            gf = None
            if where == "built_in_inference_mode":
                gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                         cap_groups=4, background=True, thread_modes=thread_modes)
        assert host_k[0].is_inference() and gk[0].is_inference()
        if gf is None:
            gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                     cap_groups=4, background=True, thread_modes=thread_modes)
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


def test_GT6b_the_feeder_publishes_into_an_inference_board(pinned):
    """GT6b (FG1's exact arms: the PUBLICATION path, residency note 12f): the feeder's second in-place write is the
    board publication (_publish: board.t index_put_). Here the host rows, the store and the adapter are NOT inference
    tensors (built outside inference mode), only the generation board is (S-C builds it in the tick's inference mode).
    attach_board turns the feeder's inference mode on; the background feeder lands the host truth and publishes every
    transfer's (slot, generation). Negative control: the feeder without the board's mode (thread_inference forced off
    after attach_board, as 7ab1ca2 had it) packs and lands the rows, then the publication raises the run's error,
    re-raised at force(); its traceback names the frames _feeder, _release and _publish."""
    import traceback
    for neg in (False, True):
        host_k, host_v, gk, gv, _, _ = _world()
        gf = GF.GilfreeTransport(pinned.ext, host_k, host_v, gk, gv, BS, D, threads=2, host_wait=lambda ev: None,
                                 cap_groups=4, background=True)
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
        gf.issue(1, trs)
        if neg:
            with pytest.raises(RuntimeError, match="Inplace update to inference tensor outside InferenceMode") as ei:
                gf.force(trs[0])
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
