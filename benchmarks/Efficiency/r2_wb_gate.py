"""R2-M2, step 8 prerequisite: the GPU gate of the DEVICE-DRIVEN MASKED WRITE-BACK (nosi/nosi/r2_writeback.py), on
one A100 in the holder (benchmarks/Efficiency/r2_wb_step.sh runs it). Design: docs/R2_WRITEBACK.md.

Shapes of NOSA-8B's layer: B 64 requests, H 2 KV heads, D 128, bf16; the GPU slot cache has 64 slots of 64 rows
(tail slot rows 4032 .. 4095); the host tables have N 24384 rows (16128 + 8192 + 64), PINNED. Seeded.

Every comparison is BITWISE (int16 views of bf16) over the WHOLE host buffers, which are NaN-poisoned before every
case: an extra write replaces a NaN, a missing write leaves one, a wrong row shows as both. MUST gates (exit code =
the number of failed MUST gates; summary.json records every gate):
  W0  the body's TORCH backend (r2_decode_body.DecodeBody.writeback: per-request copy_(non_blocking=True) from the GPU
      into the pinned rows) == masked_writeback_torch (the reference, run on CPU copies), every case of W1.
  W1  EAGER: the kernel == the reference: n_full 0, 1, 7, 64 of 64 (positions that end a block; the other requests at
      random positions), and a guard case (a full request at p = 30, one at p = N + 40: only in-range rows written).
  W2  CAPTURE: one kernel launch captured in torch.cuda.graph (after an eager warm-up that compiles it), replayed
      --replays times; before each replay a new mask, new positions and new tail rows are written INTO the captured
      device tensors and both host buffer pairs are re-poisoned; each replay == an eager launch on a second pair of
      pinned host buffers == the reference.
  W3  WRITE THEN READ ON ONE STREAM: the kernel, then the shipped gather flash_h2d_from_mask (flash_cache_engine/
      flash_h2d_mask.py, as cache_engine.py:381 calls it) naming each full request's just-written block id, into a
      NaN scratch slot: the scratch rows == the GPU tail rows for full requests, NaN for the others. Eager, and the two
      kernels captured together in one graph and replayed with a new case.
  W4  NEGATIVE CONTROLS: the kernel with row shift -1, with row shift +1 and unmasked: the W1 comparison must FAIL for
      each (3 of 3 caught).
REPORT (no claim is made from it): W5 timing with CUDA events, median of 30: the kernel, the graph replay, and the
torch backend's per-request copies, for n_full 1, 8, 64.
Run (inside the image, one GPU):
    python benchmarks/Efficiency/r2_wb_gate.py --out <new dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NOSI_PKG = os.path.join(HERE, "..", "..", "nosi", "nosi")
B, H, D, BS, TOPK = 64, 2, 128, 64, 64
TAIL0 = (TOPK - 1) * BS
N = 16128 + 8192 + 64


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(NOSI_PKG, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits(t):
    return t.contiguous().view(torch.int16)


def beq(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--replays", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("r2 wb gate: --out %s exists" % a.out, file=sys.stderr)
        return 2
    os.makedirs(a.out)
    wb = _load("nosi_r2_writeback_gate", "r2_writeback.py")
    rb = _load("nosi_r2_decode_body_wb_gate", "r2_decode_body.py")
    h2d = _load("nosi_flash_h2d_mask_wb_gate", os.path.join("flash_cache_engine", "flash_h2d_mask.py"))
    if wb.triton is None:
        print("r2 wb gate: triton did not import", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    bf = torch.bfloat16
    gcpu = torch.Generator().manual_seed(a.seed)
    gdev = torch.Generator(device=dev).manual_seed(a.seed)
    rec = dict(t0=time.time(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, shapes=dict(
        B=B, H=H, D=D, block=BS, topk=TOPK, tail0=TAIL0, n_host=N, dtype="bf16"), gates={}, report={})
    nfail = 0

    def gate(name, ok, detail, must=True):
        nonlocal nfail
        rec["gates"][name] = dict(passed=bool(ok), must=must, detail=detail)
        print("%s %s %s" % (name, "PASS" if ok else ("FAIL" if must else "REPORT"), detail), flush=True)
        if must and not ok:
            nfail += 1

    # the captured tensors: their addresses never change
    k_gpu = torch.randn((B, TOPK * BS, H, D), generator=gdev, device=dev).to(bf)
    v_gpu = torch.randn((B, TOPK * BS, H, D), generator=gdev, device=dev).to(bf)
    full = torch.zeros(B, dtype=torch.bool, device=dev)
    pos = torch.zeros(B, dtype=torch.int64, device=dev)
    k_cpu = torch.empty((B, N, H, D), dtype=bf, pin_memory=True)
    v_cpu = torch.empty((B, N, H, D), dtype=bf, pin_memory=True)
    k_cpu2 = torch.empty((B, N, H, D), dtype=bf, pin_memory=True)
    v_cpu2 = torch.empty((B, N, H, D), dtype=bf, pin_memory=True)

    def case(n_full, guard=False):
        perm = torch.randperm(B, generator=gcpu).tolist()
        f = [False] * B
        for i in perm[:n_full]:
            f[i] = True
        blocks = torch.randint(0, N // BS, (B,), generator=gcpu).tolist()
        other = torch.randint(0, N, (B,), generator=gcpu).tolist()
        p = [blocks[i] * BS + BS - 1 if f[i] else other[i] for i in range(B)]
        if guard:
            f[perm[0]], p[perm[0]] = True, 30
            f[perm[1]], p[perm[1]] = True, N + 40
        return f, p

    def set_inputs(f, p, fresh_tail=True):
        full.copy_(torch.tensor(f, dtype=torch.bool))
        pos.copy_(torch.tensor(p, dtype=torch.int64))
        if fresh_tail:
            k_gpu[:, TAIL0:TAIL0 + BS].copy_(torch.randn((B, BS, H, D), generator=gdev, device=dev).to(bf))
            v_gpu[:, TAIL0:TAIL0 + BS].copy_(torch.randn((B, BS, H, D), generator=gdev, device=dev).to(bf))
        torch.cuda.synchronize()

    def poison(*bufs):
        torch.cuda.synchronize()
        for t in bufs:
            t.fill_(float("nan"))

    def reference():
        """masked_writeback_torch on CPU copies of the current inputs, into NaN host buffers."""
        rk = torch.full((B, N, H, D), float("nan"), dtype=bf)
        rv = torch.full((B, N, H, D), float("nan"), dtype=bf)
        wb.masked_writeback_torch(k_gpu.cpu(), v_gpu.cpu(), rk, rv, full.cpu(), pos.cpu(), TAIL0, BS)
        return rk, rv

    # ---- W0 / W1: eager
    w0_bad, w1_bad, cases = [], [], [(0, False), (1, False), (7, False), (64, False), (7, True)]
    for n_full, guard in cases:
        f, p = case(n_full, guard)
        set_inputs(f, p)
        rk, rv = reference()
        tag = "n_full %d%s" % (n_full, " guard" if guard else "")
        if not guard:                                    # the torch backend has no row guard: in-range cases only
            poison(k_cpu2, v_cpu2)
            e = types.SimpleNamespace(block_size=BS, _tail_block_idx_on_gpu=TOPK - 1, _k_gpu=k_gpu, _v_gpu=v_gpu,
                                      _k_cpu=k_cpu2, _v_cpu=v_cpu2)
            rb.DecodeBody(False).writeback(e, full, full.cpu(), pos, pos.cpu())
            torch.cuda.synchronize()
            if not (beq(k_cpu2, rk) and beq(v_cpu2, rv)):
                w0_bad.append(tag)
        poison(k_cpu, v_cpu)
        wb.masked_writeback(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, TAIL0, BS)
        torch.cuda.synchronize()
        if not (beq(k_cpu, rk) and beq(v_cpu, rv)):
            w1_bad.append(tag)
    gate("W0", not w0_bad, "the body's torch backend == the reference on %d in-range cases; failing: %s"
         % (len([c for c in cases if not c[1]]), w0_bad))
    gate("W1", not w1_bad, "the kernel (eager) == the reference on %d cases (n_full 0/1/7/64 of %d, and the row guard); "
         "failing: %s" % (len(cases), B, w1_bad))

    # ---- W2: capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wb.masked_writeback(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, TAIL0, BS)
    torch.cuda.synchronize()
    w2_bad = []
    for i in range(a.replays):
        f, p = case([0, 1, 3, 8, 21, 64][i % 6], guard=(i % 5 == 4))
        set_inputs(f, p)
        rk, rv = reference()
        poison(k_cpu, v_cpu, k_cpu2, v_cpu2)
        graph.replay()
        wb.masked_writeback(k_gpu, v_gpu, k_cpu2, v_cpu2, full, pos, TAIL0, BS)
        torch.cuda.synchronize()
        ok = beq(k_cpu, rk) and beq(v_cpu, rv) and beq(k_cpu2, rk) and beq(v_cpu2, rv)
        if not ok:
            w2_bad.append(i)
    gate("W2", not w2_bad, "%d replays of one captured launch with new masks, positions and tail rows == the eager "
         "launch == the reference; failing replays: %s" % (a.replays, w2_bad))

    # ---- W3: write, then the shipped gather on the same stream
    scratch = torch.empty((B, BS, H, D), dtype=bf, device=dev)
    load_ids = torch.full((H, B, 1), -1, dtype=torch.int64, device=dev)

    def chain():
        wb.masked_writeback(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, TAIL0, BS)
        h2d.flash_h2d_from_mask(scratch, k_cpu, load_ids, BS)

    def w3_check():
        bad = []
        for b in range(B):
            if bool(full[b]):
                if not beq(scratch[b], k_gpu[b, TAIL0:TAIL0 + BS]):
                    bad.append(b)
            elif not bool(torch.isnan(scratch[b].float()).all()):
                bad.append(b)
        return bad

    def w3_inputs(n_full):
        f, p = case(n_full)
        set_inputs(f, p)
        load_ids.copy_(torch.where(full, pos // BS, torch.full_like(pos, -1)).view(1, B, 1).expand(H, B, 1))
        scratch.fill_(float("nan"))
        poison(k_cpu, v_cpu)

    w3_inputs(7)
    chain()
    torch.cuda.synchronize()
    bad_e = w3_check()
    graph3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph3):
        chain()
    torch.cuda.synchronize()
    w3_inputs(21)
    graph3.replay()
    torch.cuda.synchronize()
    bad_g = w3_check()
    gate("W3", not bad_e and not bad_g, "write-back then flash_h2d_from_mask of the written block on one stream: "
         "eager mismatching requests %s; captured chain replayed with a new case: %s" % (bad_e, bad_g))

    # ---- W4: negative controls
    f, p = case(7)
    set_inputs(f, p)
    rk, rv = reference()
    caught = {}
    for name, shift, masked in (("row-1", -1, True), ("row+1", 1, True), ("unmasked", 0, False)):
        poison(k_cpu, v_cpu)
        wb.check_args(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, TAIL0, BS)
        wb._launch(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, TAIL0, BS, shift, masked)
        torch.cuda.synchronize()
        caught[name] = not (beq(k_cpu, rk) and beq(v_cpu, rv))
    gate("W4", all(caught.values()), "the W1 comparison catches each planted defect: %s" % caught)

    # ---- W5: timing (REPORT)
    def timed(fn, reps=30):
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0)
        return round(statistics.median(ts), 1)

    eng = types.SimpleNamespace(block_size=BS, _tail_block_idx_on_gpu=TOPK - 1, _k_gpu=k_gpu, _v_gpu=v_gpu,
                                _k_cpu=k_cpu2, _v_cpu=v_cpu2)
    torch_body = rb.DecodeBody(False)
    for n_full in (1, 8, 64):
        f, p = case(n_full)
        set_inputs(f, p, fresh_tail=False)
        fh, ph = full.cpu(), pos.cpu()
        rec["report"]["n_full_%d" % n_full] = dict(
            kernel_us=timed(lambda: wb.masked_writeback(k_gpu, v_gpu, k_cpu, v_cpu, full, pos, TAIL0, BS)),
            graph_us=timed(graph.replay),
            torch_backend_us=timed(lambda: torch_body.writeback(eng, full, fh, pos, ph)))
    gate("W5", True, "REPORT median us (kernel, graph replay, torch backend): %s" % rec["report"], must=False)

    rec.update(failed=nfail, t1=time.time())
    with open(os.path.join(a.out, "summary.json"), "w") as out_f:
        json.dump(rec, out_f, indent=1)
    print("r2 wb gate: %d MUST gate(s) failed" % nfail, flush=True)
    return nfail


if __name__ == "__main__":
    sys.exit(main())
