"""Stage N0 of the Strata / HiCache experiment: SGLang's NATIVE HiCache load path beside the decode-step proxy of
hisparse_repro.py (HR_PAYLOAD=hicache). The harness (proxy, gated bracket, modes, regimes, sizing, per-launch side
events, tables) is hisparse_repro.py's, unchanged; this module only builds the buffers, the plan, the arms and the
correctness hooks.

THE NATIVE PATH (sglang 87db743, pinned in retroinfer-eval scripts/sglang_hicache/; file:line of that commit)
  host pool   MHATokenToKVPoolHost (mem_cache/pool_host/mha.py :162-203): K and V each (size, L, H, Dh) bf16 for
              page_first (the 87db743 default, arg_groups/memory.py :132-144) or (L, size, H, Dh) for layer_first
              (the Strata-era default and NOSI's host form); a CPU tensor registered with cudaHostRegister(ptr, size, 0)
              (pool_host/common.py :124-160) -- HR_HC_HOST_ALLOC=register (default) | pin (torch pin_memory).
  item        page_size 1 (arg_groups/overrides.py :1397-1418): one token row of both KV heads = H x Dh x 2 = 512 B,
              K and V at the same index (mha.py :187).
  plan        one plan per load op, reused by all L per-layer launches (l2_transfer.py :88-110); host slots of a prefix
              are consecutive runs, device rows scattered (plan.build_native). Per request 3.5 runs of 64 tokens on
              average (the nosi16 miss volume): 16 x 224 = 3,584 items per layer, 117,440,512 B per pass (K + V, 32 layers).
  launches    32 per-layer launches on the side stream, an event after each (LAUNCH_LOG = SGLang's load_events[layer]).
ARMS (identical bytes, identical destinations except dma2d, which writes each request's prefix)
  strata<W>_t1024_<lf|pf>      AOT transfer_kv_per_layer / _pf_lf, W blocks x 32 warps (CLOSEST RELEASED IMPLEMENTATION
                               of Strata's IO kernel; the paper kernel is unavailable)
  hicachejit<W>_t1024_<lf|pf>  JIT hicache_transfer_per_layer, W blocks x 1024 threads (SGLang's default; NOT Strata)
  hisparse<W>_i512_t1024       HiSparse copy_cache_planned_kernel on the SAME lf host buffers (control; NOT Strata)
  dma2d                        cudaMemcpy2DAsync of the same bytes per layer and tensor, request prefixes (copy engine)
ITEM-SIZE CONTROL (HR_HC_ITEM_CONTROL=1; a mechanism control, NOT SGLang's item for NOSA): a second seeded plan whose
device rows are 64-row runs, copied by the AOT kernel as 512-B items (strata<W>_t1024_i512runs) and as 2048-B items of four
consecutive tokens (strata<W>_t1024_i2048runs; Strata Fig. 5's Llama-3.1-8B K row is 2 KB), identical bytes and
destinations. The SASS of transfer_item_warp batches four 8-B loads per lane only when a lane has >= 4 chunks (items
>= 1 KB); this pair tests whether the item size, not a different kernel, sets the AOT kernel's bandwidth.
PLAN COST: EXCLUDED from every timed side (prebuilt index tensors, as SGLang's per-layer loop); the released caller's
cost (host build + pageable index upload, once per load op) is timed separately (meta.plans.native). Every row is a
'prebuilt-plan copy microbenchmark, NOT an integrated LRU getter baseline'.
CORRECTNESS: every destination row is NaN-poisoned before a transfer rep; after it every item's K and V rows must be
torch.equal to the source rows (all layers) AND every other element of both device buffers must still hold the canary
value (no stray write). NEGATIVE CONTROLS (each must be DETECTED by the check): omit the last layer's launch, and a
destination index permutation (two items swapped) -- for both kernels; HiSparse SkipIO (the harness's own control).
GATES: cudaDevAttrCanUseHostPointerForRegisteredMem == 1 and device pointer == host pointer for every host buffer.
"""
import ctypes
import json
import time
from typing import Dict

import torch

CANARY = 0.8125          # exactly representable in bf16


def _load_hc(H):
    d = H.os.path.join(H.FCE, "sglang_hicache")
    return H._load_by_path("sglang_hicache", H.os.path.join(d, "__init__.py"), d)


def hisparse_plan_from_native(hs, p, host_records: int, dev_records: int):
    """The SAME items as a HiSparse plan (plan row = request): src int64 [B, P], dst int32 [B, P], pad -1."""
    per = p.per_request_items()
    B, P = len(per), max(per)
    src = torch.full((B, P), -1, dtype=torch.int64)
    dst = torch.full((B, P), -1, dtype=torch.int32)
    o = 0
    for b, n in enumerate(per):
        src[b, :n] = p.host_idx[o:o + n]
        dst[b, :n] = p.dev_idx[o:o + n].to(torch.int32)
        o += n
    return hs.Plan(src=src, dst=dst, counts=torch.tensor(per, dtype=torch.int32), num_real=torch.tensor([B], dtype=torch.int32),
                   kind="row", item_size_bytes=p.item_size, n_items=p.n_items, host_records=host_records, dev_records=dev_records)


def setup(cfg, hs, side, cudart, H) -> Dict:
    """H = the hisparse_repro module (helpers). Returns the env dict run() consumes."""
    HC = _load_hc(H)
    t0 = time.time()
    aot, jit = HC.load()
    print("[repro] sglang_hicache extensions ready in %.0fs (%s)" % (time.time() - t0, HC.UPSTREAM_COMMIT), flush=True)
    m = H.MODEL
    Hh, Dh, L, B = m["n_kv"], m["head_dim"], cfg.layers, cfg.B
    item = Hh * Dh * 2
    E = item // 2
    fails, controls, skipped = 0, {}, []
    p, tm = HC.timed_native_build(B=B, host_tokens_per_request=cfg.s_cpu, dev_rows_per_request=cfg.window_rows, runs_per_request=cfg.miss_per_head,
                                  block_rows=cfg.block_rows, item_size=item, seed=cfg.seed, dev_mode=cfg.hc_dev_mode)
    hidx, didx = tm.pop("host_idx_gpu"), tm.pop("dev_idx_gpu")
    N = p.n_items
    host_records, dev_records = B * cfg.s_cpu, B * cfg.window_rows
    HC.validate_indices(p.host_idx, p.dev_idx, host_records, dev_records, item)
    pass_bytes = L * p.bytes_per_layer
    plan_meta = {"native": dict(tm, item_size_bytes=item, k_only=False, label=HC.PLAN_COST_LABEL, counts=p.counts, dev_mode=p.dev_mode)}
    print("[repro] payload hicache: %d items of %d B per layer (runs %s), %.4f GB per pass (K+V); plan %s"
          % (N, item, p.counts, pass_bytes / 1e9, json.dumps(plan_meta["native"])), flush=True)

    # ---- host buffers: layer_first (L, S, H, Dh) always (the controls read it), page_first (S, L, H, Dh) when asked
    can_host = int(aot.can_use_host_pointer_for_registered_mem(0))
    controls["can_use_host_pointer_for_registered_mem"] = can_host
    fails += int(can_host != 1)

    def alloc(shape):
        if cfg.hc_host_alloc == "register":
            t = torch.empty(shape, dtype=torch.bfloat16)
            HC.host_register(t)
            return t
        return torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
    t0 = time.time()
    lf_k, lf_v = alloc((L, host_records, Hh, Dh)), alloc((L, host_records, Hh, Dh))
    H._fill_pinned(lf_k, cfg.seed + 2)
    H._fill_pinned(lf_v, cfg.seed + 3)
    bufs = dict(lf_k=lf_k, lf_v=lf_v)
    if "pf" in cfg.hc_layouts:
        pf_k, pf_v = alloc((host_records, L, Hh, Dh)), alloc((host_records, L, Hh, Dh))
        for l in range(L):                                      # the same logical values as layer_first
            pf_k.view(host_records, L, item // 2)[:, l].copy_(lf_k[l].view(host_records, item // 2))
            pf_v.view(host_records, L, item // 2)[:, l].copy_(lf_v[l].view(host_records, item // 2))
        bufs.update(pf_k=pf_k, pf_v=pf_v)
    ptr_ok = {k: int(aot.host_ptr_is_device_ptr(v)) for k, v in bufs.items()}
    controls["host_ptr_is_device_ptr"] = ptr_ok
    fails += sum(1 for v in ptr_ok.values() if v != 1)
    vmas = H.NM.parse()
    nb = lf_k.numel() * 2
    numa = dict(buffers={k: H.numa_range(v.data_ptr(), nb, vmas) for k, v in bufs.items()}, proc=H.proc_numa(),
                bind=H.os.environ.get("HR_NUMA_BIND", ""), gpu_numa_node=H.os.environ.get("HR_GPU_NUMA_NODE", ""), bytes_each=nb,
                host_alloc=cfg.hc_host_alloc)
    print("[repro] hicache host buffers %s, %.2f GB each, %s + filled in %.0fs; NUMA %s"
          % (sorted(bufs), nb / 1e9, cfg.hc_host_alloc, time.time() - t0, json.dumps(numa)), flush=True)

    # ---- device buffers (layer_first, as MHATokenToKVPool: per layer (rows, H, Dh)), canary everywhere
    dev_k = torch.full((L, dev_records, Hh, Dh), CANARY, dtype=torch.bfloat16, device="cuda")
    dev_v = torch.full_like(dev_k, CANARY)
    ref_k = lf_k[:, p.host_idx].cuda()
    ref_v = lf_v[:, p.host_idx].cuda()
    w_dma = N // B * item                                   # the same bytes as one prefix per request and layer tensor
    if N % B:
        raise ValueError("items per layer %d do not split into %d equal request rows" % (N, B))
    w_el = w_dma // 2
    dma_ref_k = lf_k.view(L, B, -1)[:, :, :w_el].cuda()
    dma_ref_v = lf_v.view(L, B, -1)[:, :, :w_el].cuda()
    dma_bytes = 2 * L * B * w_dma
    stats = dict(canary_fail=0)
    nan = float("nan")

    def region(kind):
        if kind == "dma2d":
            return [(dev_k.view(L, B, -1)[:, :, :w_el], dma_ref_k), (dev_v.view(L, B, -1)[:, :, :w_el], dma_ref_v)]
        return None

    def rows_of(kind):
        if kind == "runs":
            return runs["didx"], runs["ref_k"], runs["ref_v"]
        return didx, ref_k, ref_v

    def poison(kind):
        if kind == "dma2d":
            for t, _ in region(kind):
                t.fill_(nan)
        else:
            di_, _, _ = rows_of(kind)
            dev_k[:, di_] = nan
            dev_v[:, di_] = nan

    def canary_intact() -> bool:
        return all(not bool(t[l].ne(CANARY).any()) for t in (dev_k, dev_v) for l in range(L))

    def check(kind) -> bool:
        if kind == "dma2d":
            ok = all(torch.equal(t, r) for t, r in region(kind))
            for t, _ in region(kind):
                t.fill_(CANARY)
        else:
            di_, rk, rv = rows_of(kind)
            ok = torch.equal(dev_k[:, di_], rk) and torch.equal(dev_v[:, di_], rv)
            dev_k[:, di_] = CANARY
            dev_v[:, di_] = CANARY
        intact = canary_intact()
        stats["canary_fail"] += int(not intact)
        return bool(ok and intact)

    # ---- per-layer views, built once (nothing is allocated inside a timed side)
    views = {}
    for lay in cfg.hc_layouts:
        if lay == "lf":
            views[lay] = [(lf_k[l].view(-1, E), lf_v[l].view(-1, E)) for l in range(L)]
        else:
            views[lay] = [(bufs["pf_k"].view(host_records, L, E)[:, l], bufs["pf_v"].view(host_records, L, E)[:, l]) for l in range(L)]
    dviews = [(dev_k[l].view(-1, E), dev_v[l].view(-1, E)) for l in range(L)]
    layer_bytes = p.bytes_per_layer

    def side_hc(kern, lay, W, di=None, layers=None):
        dix = didx if di is None else di
        lays = list(range(L)) if layers is None else layers

        def mk(passes):
            def f():
                for _ in range(passes):
                    for l in lays:
                        if kern == "aot" and lay == "lf":
                            aot.aot_per_layer(lf_k[l], dev_k[l], lf_v[l], dev_v[l], hidx, dix, item, W, 32)
                        elif kern == "aot":
                            aot.aot_per_layer_pf_lf(bufs["pf_k"], dev_k[l], bufs["pf_v"], dev_v[l], hidx, dix, l, item, L * item, W, 32)
                        else:
                            ks, vs = views[lay][l]
                            jit.jit_per_layer(dviews[l][0], dviews[l][1], dix, ks, vs, hidx, item, W)
                        H.SZ.LAUNCH_LOG.mark(layer_bytes)
            return f
        return mk

    hplan = hisparse_plan_from_native(hs, p, host_records, dev_records).to("cuda")
    hs.validate_plan(hplan)
    if hplan.bytes_per_launch != layer_bytes:
        raise AssertionError("HiSparse control plan bytes %d != %d" % (hplan.bytes_per_launch, layer_bytes))

    def side_hisparse(W, T, skip_io=False):
        def mk(passes):
            def f():
                for _ in range(passes):
                    for l in range(L):
                        hs.copy_plan(hplan, lf_k[l], lf_v[l], dev_k[l], dev_v[l], W, T, skip_io)
                        H.SZ.LAUNCH_LOG.mark(hplan.bytes_per_launch)
            return f
        return mk

    def side_dma():
        spitch, dpitch = cfg.s_cpu * item, cfg.window_rows * item

        def mk(passes):
            def f():
                st = ctypes.c_void_p(side.cuda_stream)
                for _ in range(passes):
                    for l in range(L):
                        for src, dst in ((lf_k[l], dev_k[l]), (lf_v[l], dev_v[l])):
                            rc = cudart.cudaMemcpy2DAsync(ctypes.c_void_p(dst.data_ptr()), dpitch, ctypes.c_void_p(src.data_ptr()), spitch, w_dma, B, 1, st)
                            if rc != 0:
                                raise RuntimeError("cudaMemcpy2DAsync rc=%d" % rc)
                        H.SZ.LAUNCH_LOG.mark(2 * B * w_dma)
            return f
        return mk

    arms, grids = [], {}
    for lay in cfg.hc_layouts:
        for kern in cfg.hc_kernels:
            for W in cfg.hc_blocks:
                fam = "strata" if kern == "aot" else "hicachejit"
                name = "%s%d_t1024_%s" % (fam, W, lay)
                grid = HC.aot_grid(N, W, 32)[1] if kern == "aot" else HC.jit_grid(N, W)
                grids[name] = dict(grid=grid, block=1024, kernel=("transfer_kernel_impl" if kern == "aot" else "hicache_transfer_per_layer"),
                                   label=(HC.STRATA_LABEL if kern == "aot" else HC.JIT_LABEL), layout=lay)
                arms.append(dict(arm=name, family=fam, W=W, threads=1024, item="i512_" + lay, kind="scatter", mk=side_hc(kern, lay, W),
                                 bytes_pass=pass_bytes))
    for T in cfg.hs_threads:
        for W in H.blocks_for(cfg, T):
            arms.append(dict(arm="hisparse%d_i512_t%d" % (W, T), family="hisparse", W=W, threads=T, item="i512_lf", kind="scatter",
                             mk=side_hisparse(W, T), bytes_pass=pass_bytes))
    if cfg.dma:
        arms.append(dict(arm="dma2d", family="dma2d", W=None, threads=None, item="prefix", kind="dma2d", mk=side_dma(), bytes_pass=dma_bytes))

    # ---- item-size control (module docstring): a second plan with device runs, AOT at 512 B and at 2048 B
    runs = None
    if cfg.hc_item_control:
        pr, tr = HC.timed_native_build(B=B, host_tokens_per_request=cfg.s_cpu, dev_rows_per_request=cfg.window_rows, runs_per_request=cfg.miss_per_head,
                                       block_rows=cfg.block_rows, item_size=item, seed=cfg.seed + 1, dev_mode="runs")
        hr_, dr_ = tr.pop("host_idx_gpu"), tr.pop("dev_idx_gpu")
        HC.validate_indices(pr.host_idx, pr.dev_idx, host_records, dev_records, item)
        g4 = 4
        h4, d4 = pr.host_idx.view(-1, g4), pr.dev_idx.view(-1, g4)
        if not (bool(((h4 - h4[:, :1]) == torch.arange(g4)).all()) and bool(((d4 - d4[:, :1]) == torch.arange(g4)).all())
                and bool((h4[:, 0] % g4 == 0).all()) and bool((d4[:, 0] % g4 == 0).all())):
            raise AssertionError("the runs plan does not group into aligned 4-token items")
        h4i, d4i = (h4[:, 0] // g4).cuda(), (d4[:, 0] // g4).cuda()
        runs = dict(didx=dr_, ref_k=lf_k[:, pr.host_idx].cuda(), ref_v=lf_v[:, pr.host_idx].cuda())
        plan_meta["runs"] = dict(tr, item_size_bytes=item, label=HC.PLAN_COST_LABEL, counts=pr.counts, dev_mode="runs",
                                 note="item-size control only; the i2048 items are 4 consecutive tokens of this plan")

        def side_runs(W, big):
            def mk(passes):
                def f():
                    for _ in range(passes):
                        for l in range(L):
                            if big:
                                aot.aot_per_layer(lf_k[l], dev_k[l], lf_v[l], dev_v[l], h4i, d4i, g4 * item, W, 32)
                            else:
                                aot.aot_per_layer(lf_k[l], dev_k[l], lf_v[l], dev_v[l], hr_, dr_, item, W, 32)
                            H.SZ.LAUNCH_LOG.mark(layer_bytes)
                return f
            return mk
        for W in cfg.hc_blocks:
            for big in (False, True):
                name = "strata%d_t1024_%s" % (W, "i2048runs" if big else "i512runs")
                n_it = pr.n_items // g4 if big else pr.n_items
                grids[name] = dict(grid=HC.aot_grid(n_it, W, 32)[1], block=1024, kernel="transfer_kernel_impl", label=HC.STRATA_LABEL, layout="lf",
                                   control="item size")
                arms.append(dict(arm=name, family="strata", W=W, threads=1024, item=("i2048" if big else "i512") + "_runs", kind="runs",
                                 mk=side_runs(W, big), bytes_pass=pass_bytes))

    # ---- negative controls: each must make check() fail
    wrong = didx.clone()
    wrong[0], wrong[1] = didx[1].item(), didx[0].item()
    neg = []
    for kern in cfg.hc_kernels:
        lay = cfg.hc_layouts[0]
        W = max(cfg.hc_blocks)
        neg.append(("omit_last_layer_%s_%s" % (kern, lay), side_hc(kern, lay, W, layers=list(range(L - 1))), "scatter"))
        neg.append(("swapped_dst_index_%s_%s" % (kern, lay), side_hc(kern, lay, W, di=wrong), "scatter"))
    skip_mk = side_hisparse(max(cfg.hs_blocks or (2,)), 1024, skip_io=True) if cfg.hs_blocks else None
    if cfg.triton_w:
        skipped.append("triton%s: NOSI's per-head tile gather does not apply to SGLang's token-row items" % (list(cfg.triton_w),))
    return dict(arms=arms, poison=poison, check=check, fails=fails, controls=controls, skipped=skipped, skip_io=("i512_lf", skip_mk),
                neg_controls=neg, stats=stats,
                meta=dict(payload=dict(kind="hicache", item_size_bytes=item, n_items_per_layer=N, counts=p.counts, block_rows=p.block_rows,
                                       dev_mode=p.dev_mode, host_alloc=cfg.hc_host_alloc, layouts=list(cfg.hc_layouts), kernels=list(cfg.hc_kernels),
                                       item_control=bool(cfg.hc_item_control),
                                       same_plan_every_layer=True, bench_label=HC.BENCH_LABEL, plan_cost=HC.PLAN_COST_LABEL),
                          pass_bytes=pass_bytes, dma_bytes=dma_bytes, dma_width_bytes=w_dma, numa=numa, plans=plan_meta, expected_grids=grids,
                          strata_label=HC.STRATA_LABEL, jit_label=HC.JIT_LABEL, upstream_sglang=HC.UPSTREAM_COMMIT))
