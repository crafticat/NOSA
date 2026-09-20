"""Host-side pack throughput: scattered 32 KB blocks of a large CPU window gathered into a
contiguous staging buffer by torch's multi-threaded index_select (the copy a CPU worker pool
would do before one contiguous cudaMemcpyAsync per layer -- E3d's decision, ledger 2026-09-20).
Reports GB/s vs thread count for the per-layer pack size of a decode step at B requests and
D blocks per (head, request). CPU only. Env: HP_WINDOW_GB (8), HP_B (128), HP_D (4 8), HP_THREADS ("1 2 4 8 16"), HP_REPS (5).
"""
import os, time, json, sys
import torch

WINDOW_GB = float(os.environ.get("HP_WINDOW_GB", "8"))
B = int(os.environ.get("HP_B", "128")); H = 2; BLOCK_BYTES = 32768
D_LIST = tuple(int(x) for x in os.environ.get("HP_D", "4 8").split())
THREADS = tuple(int(x) for x in os.environ.get("HP_THREADS", "1 2 4 8 16").split())
REPS = int(os.environ.get("HP_REPS", "5"))
OUT = os.environ.get("HP_OUT", "")

n_blocks = int(WINDOW_GB * 2**30) // BLOCK_BYTES
window = torch.empty((n_blocks, BLOCK_BYTES // 2), dtype=torch.bfloat16)      # the pinned window's block-major view (contiguous 32 KB per block)
window.view(torch.int16).random_()
rows = []
g = torch.Generator().manual_seed(0)
for D in D_LIST:
    n_sel = B * H * D * 32                                                     # blocks per step over 32 layers (K only; V doubles it)
    idx = torch.randint(0, n_blocks, (n_sel,), generator=g)
    dst = torch.empty((n_sel, BLOCK_BYTES // 2), dtype=torch.bfloat16)
    for t in THREADS:
        torch.set_num_threads(t)
        torch.index_select(window, 0, idx, out=dst)                           # warm
        best = None
        for r in range(REPS):
            t0 = time.perf_counter(); torch.index_select(window, 0, idx, out=dst); dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        gbps = n_sel * BLOCK_BYTES / best / 1e9
        rows.append(dict(D=D, threads=t, blocks=n_sel, MB=n_sel * BLOCK_BYTES / 1e6, best_ms=1000 * best, GBps=gbps))
        print("[host_pack] D=%d threads=%2d: %d blocks (%.0f MB) in %.1f ms = %.1f GB/s (per step over 32 layers, K only)" % (D, t, n_sel, n_sel * BLOCK_BYTES / 1e6, 1000 * best, gbps), flush=True)
if OUT:
    json.dump(dict(window_gb=WINDOW_GB, B=B, rows=rows, cpu=os.uname().nodename), open(OUT, "w"), indent=1)
