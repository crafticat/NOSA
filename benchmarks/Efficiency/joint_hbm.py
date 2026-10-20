"""Paper section 4.5, "Does joint execution reduce GPU-memory traffic?": SEPARATE vs JOINT execution of the SAME
draft and verification work of one K-step block (K = 3) at L = 16128 on the 94-document workload, on NOSA-8B with the
engine's own kernels. The pure parts (forms, layouts, write plans, accounting, parsers, summary) are in
joint_hbm_plan.py; read its docstring for the work, the four forms and the accounting levels.

MODES (python joint_hbm.py --mode <mode> ...):
  orchestrate   (inside the image; no CUDA in this process) runs, in order: the unprofiled cell U (a child process);
                the profiled cell P_none under Nsight Compute with --cache-control none; the profiled cell P_all with
                --cache-control all (a cold-L2 control: every kernel starts with a flushed L2); the ncu CSV exports;
                summarize. Exit = the number of failed MUST gates; 2 = refused (--out exists, bad arguments).
  cell          (U) capture + the four forms, unprofiled: gates, CUDA-event latency per form and per call (median /
                p95 over --reps, forms interleaved, rotated start), the host enqueue time, one torch.profiler pass per
                form (kernel count, busy, idle = launch / synchronization gaps), the exact accounting and the analytic
                bytes. Writes U/result.json, U/capture.pt, U/trace_<form>.json.
  ncu_cell      (P, run under ncu by orchestrate) capture + the forms; per form: a warm run, an L2 scrub (a write of
                2 x the 40 MB L2, outside the window), then ONE profiler window (cudaProfilerStart, k spin_kernel
                markers, the form, cudaProfilerStop); window k = the k-th form in the recorded order. Writes
                P_<cc>/windows.json.
  summarize     (CPU) U/result.json + P_*/windows.json + P_*/raw.csv -> summary.json, summary.md.

CAPTURE (shared by U and P; deterministic): prefill B documents (c16k_tokens.bin rows 0 .. B-1, sha256 checked against
SHA256SUMS), greedy shipped decode for W + K + 1 steps; at steps W .. W + K a wrapper around InfLLMv2Cache.
decode_update_kv records each layer's topk_idx and the _block_map after the update (the engine is not edited). After
step W + K: the tail slot rows of every layer (the committed rows and the shipped K / V of the block's positions) are
kept, the engine's GPU windows are freed (never used again), and the forms run from the host KV, the kept tail rows,
total_cis (the bias rows), the compressed keys and compressed cis of the engine (no compress event inside the block:
checked).

THE FORWARD (run_call, one call = one batched forward of R rows; every op is the engine's decode op on R rows, in the
decode order; NVTX range jh|<form>|<call>|<part> around every part): embed; per layer: rmsnorm + wqkv + nosa_linear
(norm_qkv), rope at the rows' positions (rope), stage 1 (the op call of infllmv2_attn_stage1_fast without its NVTX
ranges, joint_hbm_plan.stage1_direct; the rows of a request are ONE batch entry of the op through cu_seqlens_q) + the
compressed-cis rows + nosa_pooling + the two topk + the mask (score; its selection is stored and
compared with the capture, REPORT; the attention uses the captured selection), the rows' own K / V / bias writes
(kvwrite), flash_attn_nosa_with_kvcache with cache_batch_idx, per-row cache_seqlens and the per-row bias, num_splits =
NOSI_ATTN_SPLITS (attn), wo + residual + rmsnorm + gate_up + silu_and_mul + down + residual (wo_ffn); final rmsnorm +
lm_head (head).

GATES (U; MUST unless marked): G1 capture (no fill / compress in the block, tail slot = T, block map = selection minus
T, ids < T); G2 window content (sampled windows equal the host KV / the kept tail rows, the masked bias where the
plan says); G3 logits (every form, every row: max |logit - shipped logit| <= 4 x tau_ctrl, tau_ctrl = 0.4844 at 16128;
argmax agreement reported); G4 mask control (joint_shared with the masks off must DIFFER from masked); G5 accounting
(sum >= union_sep >= union_request per version and layer; private layouts attend no masked token); R1 selection
agreement (report); R2 cross-form logits (report). Orchestrator: U ran (exit 0), each P variant ran and every window
was found, P_none single-pass (MUST); untagged kernels (report).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_hbm_plan as JP  # noqa: E402

TAU = 0.4844                     # tau_ctrl at 16128 (job 2175374)
TOL = 4 * TAU
NCU_METRICS = ("gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_aperture_device_op_read.sum,"
               "lts__t_sectors_aperture_device_op_write.sum,lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum,"
               "pcie__read_bytes.sum,pcie__write_bytes.sum")       # the "full" set of job 2175791 (single-pass there)
SCRUB_BYTES = 2 * 40 * 1024 * 1024                               # 2 x the A100 L2


def log(msg: str) -> None:
    print("[joint_hbm %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("orchestrate", "cell", "ncu_cell", "summarize"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("NOSI_MODEL_PATH", ""))
    ap.add_argument("--workload", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--L", type=int, default=16128)
    ap.add_argument("--W", type=int, default=20)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--forms", default=",".join(JP.FORMS))
    ap.add_argument("--ncu", default="/opt/nvidia/nsight-compute/2025.1.0/ncu")
    ap.add_argument("--variants", default="none,all", help="ncu --cache-control variants (empty: no profiled cell)")
    ap.add_argument("--cc", default="none", help="(ncu_cell) the variant this process runs under")
    ap.add_argument("--child-timeout", type=int, default=2700)
    a = ap.parse_args(argv)
    a.forms = [f for f in a.forms.split(",") if f]
    bad = [f for f in a.forms if f not in JP.FORMS]
    if bad:
        ap.error("unknown forms %s" % bad)
    a.variants = [v for v in a.variants.split(",") if v]
    if any(v not in ("none", "all") for v in a.variants):
        ap.error("variants must be none / all")
    return a


# =====================================================================================================================
# workload and capture (CUDA)
# =====================================================================================================================
def load_workload(path: str, B: int, L: int):
    import numpy as np
    import torch
    man = json.load(open(os.path.join(path, "c16k_manifest.json")))
    n, ctx = man["shape"]
    if ctx != L or man.get("dtype") != "int32-le":
        raise SystemExit("REFUSED: workload %s is %s x %d (%s), not L = %d int32-le" % (path, n, ctx, man.get("dtype"), L))
    if not 1 <= B <= n:
        raise SystemExit("REFUSED: batch %d outside 1 .. %d distinct documents" % (B, n))
    sums = {}
    for line in open(os.path.join(path, "SHA256SUMS")):
        h, name = line.split()
        sums[name] = h
    raw = open(os.path.join(path, "c16k_tokens.bin"), "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sums.get("c16k_tokens.bin") != sha:
        raise SystemExit("REFUSED: c16k_tokens.bin sha256 %s is not SHA256SUMS's %s" % (sha, sums.get("c16k_tokens.bin")))
    arr = np.frombuffer(raw, dtype="<i4")
    if arr.size != n * ctx:
        raise SystemExit("REFUSED: c16k_tokens.bin holds %d ids, not %d x %d" % (arr.size, n, ctx))
    ids = torch.from_numpy(arr.reshape(n, ctx)[:B].astype("int64"))
    return ids, [d["index"] for d in man["docs"][:B]], sha


def engine_state(cache):
    out = []
    for lay in cache.layers:
        e = lay.cache_engine
        out.append(dict(tail=int(e._tail_block_len_on_gpu), seq=int(e.seq_length), ncl=int(lay.no_compress_k_len),
                        max_k=int(lay.cached_compressed_max_seqlen)))
    return out


def setup(a):
    """Load, prefill, capture. Returns a dict with the model, the engine pieces the forms need and the capture."""
    import torch
    import nosi.cache_engine as CE
    import nosi.nosa_llama as NL
    from nosi import NOSALlama as Llama
    from nosi.cache_engine import InfLLMv2Cache
    if CE.POOL_BLOCKS != 0:
        raise SystemExit("REFUSED: NOSI_POOL_BLOCKS = %d (the forms assume the shipped 64-slot window)" % CE.POOL_BLOCKS)
    if NL._ATTN_SPLITS <= 0:
        raise SystemExit("REFUSED: NOSI_ATTN_SPLITS must be pinned (> 0) so every form and the shipped decode share the "
                         "split count (is %d)" % NL._ATTN_SPLITS)
    T, t0 = JP.check_block(a.L, a.W, a.K)
    ids, docs, sha = load_workload(a.workload, a.batch, a.L)
    B, K, W = a.batch, a.K, a.W
    t_load = time.time()
    model = Llama(model_name=a.model, device="cuda", offload=True)
    cache = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    log("model loaded in %.1f s" % (time.time() - t_load))
    t_pre = time.time()
    with torch.inference_mode():
        logits, pos = model.batch_prefill(ids.to("cuda"), cache)
    torch.cuda.synchronize()
    log("prefill %d x %d in %.1f s" % (B, a.L, time.time() - t_pre))
    rec = {}
    step = [0]
    orig = cache.decode_update_kv

    def hook(k, v, kv_bias, layer_idx, topk_idx):
        out = orig(k, v, kv_bias, layer_idx, topk_idx)
        if step[0] >= W:
            rec[(step[0], layer_idx)] = (topk_idx.clone(), cache.layers[layer_idx].cache_engine._block_map.clone())
        return out

    cache.decode_update_kv = hook
    inputs, shipped = {}, {}
    pre = None
    with torch.inference_mode():
        tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        position_ids = pos[:, -1:] + 1
        cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
        for s in range(W + K + 1):
            step[0] = s
            if s == W:
                torch.cuda.synchronize()
                pre = engine_state(cache)
                pos_W = position_ids.clone()
            inputs[s] = tok.clone()
            lg = model.decode_inference(tok, cu, position_ids, cache)
            if s >= W:
                shipped[s] = lg[:, -1, :].float().clone()
            tok = lg[:, -1, :].argmax(dim=-1, keepdim=True)
            position_ids = position_ids + 1
        torch.cuda.synchronize()
    del cache.decode_update_kv
    post = engine_state(cache)
    nL = model.num_layers
    H = model.layers[0].num_key_value_heads
    g1 = []
    if any(x["tail"] != t0 for x in pre):
        g1.append("tail length at step %d is %s, not t0 = %d" % (W, sorted({x["tail"] for x in pre}), t0))
    if any(x["tail"] != t0 + K + 1 for x in post):
        g1.append("tail length after step %d is %s, not %d (a fill?)" % (W + K, sorted({x["tail"] for x in post}), t0 + K + 1))
    if any(p["max_k"] != q["max_k"] for p, q in zip(pre, post)) or any(x["ncl"] + K > 31 for x in pre):
        g1.append("a compress event inside the block (ncl at step %d %s, max_k %s -> %s)"
                  % (W, sorted({x["ncl"] for x in pre}), sorted({x["max_k"] for x in pre}), sorted({x["max_k"] for x in post})))
    if not torch.equal(pos_W.view(-1).cpu(), torch.full((B,), a.L + W, dtype=pos_W.dtype)):
        g1.append("positions at step %d are not L + W = %d" % (W, a.L + W))
    bms = []                                                   # per layer: (K + 1, H, B, 64)
    tks = []
    for l in range(nL):
        bm = torch.stack([rec[(W + o, l)][1] for o in range(K + 1)])
        tk = torch.stack([rec[(W + o, l)][0] for o in range(K + 1)])
        if not bool((bm[..., JP.SLOTS - 1] == T).all()):
            g1.append("layer %d: the tail slot does not hold T = %d" % (l, T))
        hist = bm[..., :JP.SLOTS - 1]
        if not bool((hist < T).all()) or not bool((hist >= 0).all()):
            g1.append("layer %d: a window id outside [0, T)" % l)
        want = torch.sort(torch.cat([hist, torch.full_like(hist[..., :1], T)], -1), -1).values
        if not torch.equal(torch.sort(tk, -1).values, want):
            g1.append("layer %d: block map slots 0..62 + T differ from the selection" % l)
        if not bool((torch.sort(hist, -1).values.diff(dim=-1) > 0).all()):
            g1.append("layer %d: repeated historical ids" % l)
        bms.append(bm)
        tks.append(tk)
    tailK, tailV, eng_pieces = [], [], []
    with torch.inference_mode():
        for l in range(nL):
            lay = cache.layers[l]
            e = lay.cache_engine
            base = (JP.SLOTS - 1) * JP.BS
            tk_ = e._k_gpu[:, base:base + JP.BS].clone()
            tv_ = e._v_gpu[:, base:base + JP.BS].clone()
            tk_[:, t0 + K + 1:].zero_()
            tv_[:, t0 + K + 1:].zero_()
            tailK.append(tk_)
            tailV.append(tv_)
            ck = lay.compress_k_cache_varlen.contiguous().flatten(0, 1)
            Nk = int(lay.cached_compressed_max_seqlen)
            cc = lay.compressed_cis
            Hc, Bc, Mc = cc.shape
            cis_pad = torch.full((Hc, Bc, Nk), float("-inf"), dtype=cc.dtype, device=cc.device)
            cis_pad[..., :min(Mc, Nk)].copy_(cc[..., :min(Mc, Nk)])
            eng_pieces.append(dict(ck=ck, cu_k=lay.cached_compressed_cu_seqlens, max_k=Nk, cis_pad=cis_pad, Mc=Mc,
                                   total_cis=lay.total_cis, k_cpu=e._k_cpu, v_cpu=e._v_cpu))
            e._k_gpu = e._v_gpu = e._kv_bias_gpu = None                # never used again: the forms own their windows
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    meta = dict(B=B, L=a.L, W=W, K=K, T=T, t0=t0, layers=nL, kv_heads=H, docs=docs, workload_sha256=sha,
                attn_splits=NL._ATTN_SPLITS, qk_select=NL.QK_SELECT, out_len=int(model.pooling_buf_all.shape[-1]),
                compressed_keys=eng_pieces[0]["max_k"], compressed_cis_width=eng_pieces[0]["Mc"],
                nosi_commit=os.environ.get("JH_COMMIT", "unknown"), forms=a.forms, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), limits=list(JP.LIMITS), ncl_at_W=sorted({x["ncl"] for x in pre}))
    log("capture: T %d t0 %d, %d layers, compressed keys %d (cis width %d), G1 issues %d"
        % (T, t0, nL, meta["compressed_keys"], meta["compressed_cis_width"], len(g1)))
    s1ext = sys.modules[NL.infllmv2_attn_stage1_fast.__module__].infllm_cuda    # the extension the decode's wrapper calls
    return dict(model=model, cache=cache, NL=NL, B=B, K=K, W=W, T=T, t0=t0, H=H, nL=nL, inputs=inputs, shipped=shipped,
                bms=bms, tks=tks, tailK=tailK, tailV=tailV, eng=eng_pieces, g1=g1, meta=meta, s1ext=s1ext)


# =====================================================================================================================
# plans (CUDA tensors per call and layer)
# =====================================================================================================================
def token_store(S, l):
    import torch
    e = S["eng"][l]
    n = S["T"] * JP.BS
    k = torch.cat([e["k_cpu"][:, :n].to("cuda", non_blocking=False), S["tailK"][l]], dim=1)
    v = torch.cat([e["v_cpu"][:, :n].to("cuda", non_blocking=False), S["tailV"][l]], dim=1)
    return k, v, e["total_cis"][:, :n + JP.BS]


def hist_of(S, l):
    """(B, K + 1, H, 63) historical ids of row (b, o) in the engine's slot order (block map slots 0 .. 62)."""
    return S["bms"][l][..., :JP.SLOTS - 1].permute(2, 0, 1, 3).contiguous()


def common_call(S, form, name, rows):
    import torch
    m = S["model"]
    B, K, W, H = S["B"], S["K"], S["W"], S["H"]
    R = len(rows)
    U = R // B
    lay0 = m.layers[0]
    out_len = int(m.pooling_buf_all.shape[-1])
    dev = "cuda"
    dt = m.pooling_buf_all.dtype
    return dict(form=form, name=name, rows=list(rows), R=R, U=U,
                tokens=torch.cat([S["inputs"][W + o][b:b + 1] for b, o in rows]).view(R, 1),
                positions=torch.tensor([S["meta"]["L"] + W + o for _, o in rows], dtype=torch.int32, device=dev),
                cu_q=(torch.arange(B + 1, dtype=torch.int32, device=dev) * U),
                req=torch.tensor([b for b, _ in rows], dtype=torch.int64, device=dev),
                pq=torch.empty((H, R, out_len), dtype=dt, device=dev), pc=torch.empty((H, R, out_len), dtype=dt, device=dev),
                tv_q=torch.empty((H, R, S["NL"].QK_SELECT), dtype=dt, device=dev),
                ti_q=torch.empty((H, R, S["NL"].QK_SELECT), dtype=torch.int64, device=dev),
                tv=torch.empty((H, R, lay0.topk_blocks), dtype=dt, device=dev),
                ti=torch.empty((H, R, lay0.topk_blocks), dtype=torch.int64, device=dev),
                mask=torch.empty((H, R, out_len), dtype=torch.bool, device=dev),
                sel_out=torch.empty((S["nL"], H, R, lay0.topk_blocks), dtype=torch.int64, device=dev),
                layers=[])


def _idx(x):
    import torch
    return torch.tensor(x, dtype=torch.int64, device="cuda")


def build_group(S, group, forms):
    """Build every call of the forms of one group (private or shared). Returns {form: [call plan]}, the layout
    metrics and the G2 issues of the sampled checks."""
    import torch
    B, K, T, t0, H, nL = S["B"], S["K"], S["T"], S["t0"], S["H"], S["nL"]
    plans = {f: [] for f in forms}
    calls = []
    for f in forms:
        for name, kinds, layout in JP.FORM_CALLS[f]:
            cp = common_call(S, f, name, JP.call_rows(B, K, kinds))
            cp["layout"] = layout
            cp["metrics"] = dict(attended_tokens=0, window_tokens=0, useful_tokens=0, masked_tokens=0)
            plans[f].append(cp)
            calls.append(cp)
    issues = []
    E = B * (K + 1)
    req_p = [b for b in range(B) for _ in range(K + 1)]
    priv = {}
    for cp in calls:
        if cp["layout"] == "private":
            pc = JP.private_call(cp["rows"], K, t0)
            cp["static"] = pc
            cp["entry"] = torch.tensor(pc["entry"], dtype=torch.int32, device="cuda")
            cp["extent_t"] = torch.tensor(pc["extent"], dtype=torch.int32, device="cuda")
            cp["w"] = tuple(_idx(pc[k]) for k in ("kv_src", "kv_dst", "b_src", "b_dst"))
    for l in range(nL):
        kst, vst, bst = token_store(S, l)
        hist = hist_of(S, l)
        hist_cpu = hist.cpu()
        if any(cp["layout"] == "private" for cp in calls):
            tok = JP.private_tok(S["bms"][l])
            kw = JP.gather_window(kst, tok, req_p)
            vw = JP.gather_window(vst, tok, req_p)
            priv[l] = (kw, vw)
            if l in (0, nL - 1):
                issues += check_window(S, l, kw, [(e, req_p[e], [S["bms"][l][e % (K + 1), h, req_p[e]].tolist()
                                                                 for h in range(H)]) for e in (0, E - 1)])
        for cp in calls:
            ext_rows = None
            if cp["layout"] == "private":
                pc = cp["static"]
                kw, vw = priv[l]
                rid_rows = torch.tensor([JP.rid(b, o, K) for b, o in cp["rows"]], device="cuda")
                store = torch.zeros((E, pc["S_e"], H), dtype=bst.dtype, device="cuda")
                store[:cp["R"]] = JP.gather_window(bst, tok[rid_rows], [b for b, _ in cp["rows"]])
                lay = dict(kwin=kw, vwin=vw, bias_store=store, bias=store, extent=cp["extent_t"], w=cp["w"])
                ext_rows = pc
            else:
                sc = JP.shared_call(hist, cp["rows"], K, t0, T)
                kw = JP.gather_window(kst, sc["tok"])
                vw = JP.gather_window(vst, sc["tok"])
                vals = JP.gather_window(bst, sc["tok"])                       # (B, S_e, H)
                rb = torch.tensor([b for b, _ in cp["rows"]], device="cuda")
                store = torch.where(sc["visible"], vals[rb], torch.full_like(vals[rb], JP.MASK))
                lay = dict(kwin=kw, vwin=vw, bias_store=store, bias=store[:B],
                           extent=torch.tensor(sc["extent"], dtype=torch.int32, device="cuda"),
                           w=tuple(_idx(sc[k]) for k in ("kv_src", "kv_dst", "b_src", "b_dst")),
                           unmasked=vals[rb].clone() if cp["form"] == "joint_shared" else None,     # G4 control
                           n_hist=sc["n_hist"])
                cp.setdefault("entry", torch.tensor(sc["entry"], dtype=torch.int32, device="cuda"))
                ext_rows = sc
                if l in (0, nL - 1) and cp["form"] == "joint_shared":
                    order = sc["order"].cpu()
                    issues += check_window(S, l, kw, [(b, b, [order[b, h].tolist() for h in range(H)]) for b in (0, B - 1)],
                                           tails=sc["tails"], n_hist=sc["n_hist"])
                    issues += check_bias(S, l, store, sc, cp["rows"], bst)
            e = S["eng"][l]
            lay.update(ck=e["ck"], cu_k=e["cu_k"], max_k=e["max_k"], cis_pad=e["cis_pad"])
            lay["kflat"] = lay["kwin"].view(-1, H, lay["kwin"].shape[-1])
            lay["vflat"] = lay["vwin"].view(-1, H, lay["vwin"].shape[-1])
            lay["bflat"] = lay["bias_store"].view(-1, H)
            cp["layers"].append(lay)
            useful = JP.call_useful_tokens(hist_cpu, cp["rows"], K, t0, T)
            mt = JP.layout_metrics(ext_rows, useful, H)
            for k_ in cp["metrics"]:
                cp["metrics"][k_] += mt[k_]
        del kst, vst
        if l == 0 or l == nL - 1:
            log("group %s: layer %d built, allocated %.1f GB" % (group, l, torch.cuda.memory_allocated() / 1e9))
    torch.cuda.synchronize()
    return plans, issues


def check_window(S, l, kw, entries, tails=("t",), n_hist=None):
    """G2: entry e of the window equals the HOST rows of its blocks (slot order as given) and the kept tail rows."""
    import torch
    issues = []
    e_ = S["eng"][l]
    H = S["H"]
    for e, b, blocks_per_head in entries:
        got = kw[e].cpu()
        nh = len(blocks_per_head[0]) if n_hist is None else n_hist
        for h in range(H):
            for s in range(nh):
                blk = int(blocks_per_head[h][s])
                if blk == S["T"]:
                    want = S["tailK"][l][b, :, h].cpu()
                else:
                    want = e_["k_cpu"][b, blk * JP.BS:(blk + 1) * JP.BS, h]
                if not torch.equal(got[s * JP.BS:(s + 1) * JP.BS, h], want):
                    issues.append("layer %d entry %d head %d slot %d (block %d) differs from its source" % (l, e, h, s, blk))
                    break
            if n_hist is not None:
                for j in range(len(tails)):
                    s = nh + j
                    if not torch.equal(got[s * JP.BS:(s + 1) * JP.BS, h], S["tailK"][l][b, :, h].cpu()):
                        issues.append("layer %d entry %d head %d tail slot %d differs from the kept tail" % (l, e, h, s))
    return issues


def check_bias(S, l, store, sc, rows, bst):
    """G2: the first and the last row of the joint_shared call: visible rows carry total_cis of their token, the rest
    MASK."""
    import torch
    issues = []
    tok = sc["tok"].cpu()
    vis = sc["visible"].cpu()
    cis = bst.cpu()
    for p in (0, len(rows) - 1):
        b, _ = rows[p]
        vals = cis[b][tok[b], torch.arange(S["H"]).view(1, -1)]                 # (S_e, H)
        want = torch.where(vis[p], vals, torch.full_like(vals, JP.MASK))
        if not torch.equal(store[p].cpu(), want):
            issues.append("layer %d row %d: the bias differs from total_cis / MASK by the plan" % (l, p))
    return issues


# =====================================================================================================================
# the forward
# =====================================================================================================================
def run_call(S, cp, unmasked=False):
    """One call: R rows through the 32 layers and the head; returns float logits (R, V)."""
    import torch
    import torch.nn.functional as F
    from torch.cuda import nvtx
    NL = S["NL"]
    m = S["model"]
    R = cp["R"]
    tag = "jh|%s|%s|" % (cp["form"], cp["name"])
    nvtx.range_push(tag + "embed")
    x = F.embedding(cp["tokens"], m.embed_tokens)
    nvtx.range_pop()
    for l, layer in enumerate(m.layers):
        P = cp["layers"][l]
        Hkv, D = layer.num_key_value_heads, layer.head_dim
        nvtx.range_push(tag + "norm_qkv")
        residual = x
        h = NL.layer_norm(x, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        qkv = F.linear(h, layer.wqkv)
        q, k, v, cis = NL.nosa_linear(qkv, layer.delta.weight, layer.A, layer.q_size, layer.kv_size, Hkv)
        nvtx.range_pop()
        nvtx.range_push(tag + "rope")
        q = q.view(R, -1)
        k = k.view(R, -1)
        NL.apply_rope_with_cos_sin_cache_inplace(cp["positions"], q, k, D, m.cos_sin_cache, True)
        q = q.reshape(R, layer.num_heads, D)
        k = k.reshape(R, Hkv, D)
        v = v.reshape(R, Hkv, D)
        nvtx.range_pop()
        nvtx.range_push(tag + "score")
        score = JP.stage1_direct(S["s1ext"], q, P["ck"], cp["cu_q"], P["cu_k"], cp["U"], P["max_k"])
        cis_rows = torch.index_select(P["cis_pad"], 1, cp["req"])
        NL.nosa_pooling(score, cis_rows, P["max_k"], cp["pq"], cp["pc"], stride=layer.pooling_stride,
                        block_size=layer.block_size, local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
        torch.topk(cp["pq"], NL.QK_SELECT, dim=-1, sorted=False, out=(cp["tv_q"], cp["ti_q"]))
        cp["mask"].zero_()
        cp["mask"].scatter_(2, cp["ti_q"], True)
        cp["pc"].masked_fill_(cp["mask"], float("inf"))
        torch.topk(cp["pc"], layer.topk_blocks, dim=-1, sorted=False, out=(cp["tv"], cp["ti"]))
        cp["sel_out"][l].copy_(cp["ti"])
        nvtx.range_pop()
        nvtx.range_push(tag + "kvwrite")
        kv_src, kv_dst, b_src, b_dst = P["w"]
        P["kflat"].index_copy_(0, kv_dst, k.index_select(0, kv_src))
        P["vflat"].index_copy_(0, kv_dst, v.index_select(0, kv_src))
        own = cis.view(R, Hkv)
        P["bflat"].index_copy_(0, b_dst, own.index_select(0, b_src))
        if unmasked:
            P["unmasked"].view(-1, Hkv).index_copy_(0, b_dst, own.index_select(0, b_src))
        nvtx.range_pop()
        nvtx.range_push(tag + "attn")
        bias = P["unmasked"][:P["bias"].shape[0]] if unmasked else P["bias"]
        o = NL.flash_attn_nosa_with_kvcache(q.unsqueeze(1), P["kwin"], P["vwin"], bias, cache_seqlens=P["extent"],
                                            cache_batch_idx=cp["entry"], num_splits=NL._ATTN_SPLITS)
        nvtx.range_pop()
        nvtx.range_push(tag + "wo_ffn")
        o = o.view(R, 1, -1)
        x = residual + F.linear(o, layer.wo)
        residual = x
        h = NL.layer_norm(x, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        h = F.linear(h, layer.gate_up_proj)
        out = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
        NL.silu_and_mul(h, out)
        x = residual + F.linear(out, layer.down_proj)
        nvtx.range_pop()
    nvtx.range_push(tag + "head")
    h = NL.layer_norm(x, w=m.norm_weight, eps=m.norm_variance_epsilon)
    logits = F.linear(h, m.lm_head).float().view(R, -1)
    nvtx.range_pop()
    return logits


def run_form(S, plans, form, events=None):
    outs = []
    for i, cp in enumerate(plans[form]):
        outs.append(run_call(S, cp))
        if events is not None:
            events[i].record()
    return outs


# =====================================================================================================================
# gates and analytic bytes (U)
# =====================================================================================================================
def logits_gate(S, plans, form, outs):
    import torch
    worst, agree, n = 0.0, 0, 0
    per_kind = {}
    for cp, lg in zip(plans[form], outs):
        for p, (b, o) in enumerate(cp["rows"]):
            ref = S["shipped"][S["W"] + o][b]
            d = float((lg[p] - ref).abs().max())
            worst = max(worst, d)
            a = bool(int(lg[p].argmax()) == int(ref.argmax()))
            agree += a
            n += 1
            k = JP.kind(o, S["K"])
            w, ag, nn = per_kind.get(k, (0.0, 0, 0))
            per_kind[k] = (max(w, d), ag + a, nn + 1)
    return dict(max_abs=worst, argmax_agree=agree / max(n, 1), rows=n,
                per_kind={k: dict(max_abs=w, argmax_agree=ag / nn) for k, (w, ag, nn) in per_kind.items()})


def selection_agreement(S, plans, form):
    import torch
    tks = torch.stack(S["tks"])                                                  # (L, K + 1, H, B, 64)
    eq, n = 0, 0
    for cp in plans[form]:
        o = torch.tensor([o for _, o in cp["rows"]], device=tks.device)
        b = torch.tensor([b for b, _ in cp["rows"]], device=tks.device)
        ref = torch.sort(tks[:, o, :, b], -1).values.permute(1, 2, 0, 3)           # (R, L, H, 64) -> (L, H, R, 64)
        same = (torch.sort(cp["sel_out"], -1).values == ref).all(-1)
        eq += int(same.sum())
        n += same.numel()
    return dict(equal_sets=eq, compared=n, fraction=eq / max(n, 1))


def analytic(S, plans, form, acc_tot):
    m = S["model"]
    per_fwd = m.lm_head.numel() * m.lm_head.element_size() + m.norm_weight.numel() * m.norm_weight.element_size()
    act_row = m.lm_head.shape[1] + m.lm_head.shape[0]
    for lay in m.layers:
        for t in (lay.wqkv, lay.wo, lay.gate_up_proj, lay.down_proj, lay.delta.weight, lay.A, lay.input_layernorm_weight,
                  lay.post_attention_layernorm_weight):
            per_fwd += t.numel() * t.element_size()
        for wt in (lay.wqkv, lay.wo, lay.gate_up_proj, lay.down_proj):
            act_row += wt.shape[0] + wt.shape[1]
    calls = plans[form]
    rows = sum(cp["R"] for cp in calls)
    emb = m.embed_tokens.shape[1] * m.embed_tokens.element_size()
    ck = sum(int(S["eng"][l]["ck"].numel()) * S["eng"][l]["ck"].element_size() for l in range(S["nL"]))
    lay_m = dict(attended_tokens=0, window_tokens=0, useful_tokens=0, masked_tokens=0)
    for cp in calls:
        for k in lay_m:
            lay_m[k] += cp["metrics"][k]
    shared = any(cp["layout"] == "shared" for cp in calls)
    joint = len(calls) == 1
    kv_form = acc_tot["sum"] if not shared else (acc_tot["union_request"] if joint else acc_tot["union_sep"])
    return dict(calls=[cp["name"] for cp in calls], rows=rows, forwards=len(calls),
                weight_bytes=len(calls) * per_fwd + rows * emb, ck_bytes=len(calls) * ck, act_bytes=rows * act_row * 2,
                kv_sum=acc_tot["sum"], kv_form=kv_form, kv_request=acc_tot["union_request"], layout=lay_m,
                kv_form_rule="private: sum (own copies); shared separate: union within each call; shared joint: union "
                             "within the request")


def gate(name, passed, must, detail):
    return dict(name=name, passed=bool(passed), must=bool(must), detail=detail)


# =====================================================================================================================
# modes
# =====================================================================================================================
def mode_cell(a):
    import torch
    os.makedirs(a.out, exist_ok=False)
    S = setup(a)
    gates = [gate("G1_capture", not S["g1"], True, "; ".join(S["g1"][:6]) or "block invariants hold")]
    acc = [JP.account_layer(hist_of(S, l).cpu(), S["K"], S["t0"], S["T"]) for l in range(S["nL"])]
    acc_tot = {lvl: dict(c=0, x=0, d=0, hist_blocks=0) for lvl in ("sum", "union_sep", "union_request")}
    bad_acc = []
    for l, x in enumerate(acc):
        for lvl in acc_tot:
            JP._add(acc_tot[lvl], x[lvl])
        for v in JP.VERSIONS:
            if not x["sum"][v] >= x["union_sep"][v] >= x["union_request"][v]:
                bad_acc.append("layer %d version %s: %d / %d / %d" % (l, v, x["sum"][v], x["union_sep"][v], x["union_request"][v]))
    torch.save(dict(bms=[b.cpu() for b in S["bms"]], tks=[t.cpu() for t in S["tks"]],
                    inputs={s: t.cpu() for s, t in S["inputs"].items()}, meta=S["meta"]), os.path.join(a.out, "capture.pt"))
    timing, kineto, analytic_b, logit_g, sel_g, cross, issues2, mask_ctl = {}, {}, {}, {}, {}, {}, [], None
    ref_out = {}
    order_log = []
    scrub = torch.empty(SCRUB_BYTES, dtype=torch.uint8, device="cuda")
    with torch.inference_mode():
        for group, gforms in JP.GROUPS:
            forms = [f for f in gforms if f in a.forms]
            if not forms:
                continue
            plans, iss = build_group(S, group, forms)
            issues2 += iss
            for f in forms:
                for cp in plans[f]:
                    if cp["layout"] == "private" and cp["metrics"]["masked_tokens"] != 0:
                        bad_acc.append("form %s call %s: a private layout attends %d masked tokens" % (f, cp["name"], cp["metrics"]["masked_tokens"]))
                    if cp["metrics"]["masked_tokens"] < 0:
                        bad_acc.append("form %s call %s: negative masked tokens" % (f, cp["name"]))
            for f in forms:                                              # warm (JIT, cuBLAS) and the gate outputs
                for _ in range(max(1, a.warm)):
                    outs = run_form(S, plans, f)
                torch.cuda.synchronize()
                logit_g[f] = logits_gate(S, plans, f, outs)
                sel_g[f] = selection_agreement(S, plans, f)
                ref_out[f] = [o.cpu() for o in outs]
                analytic_b[f] = analytic(S, plans, f, acc_tot)
            if "joint_shared" in forms:                                  # G4: the masks matter
                cp = plans["joint_shared"][0]
                un = run_call(S, cp, unmasked=True)
                torch.cuda.synchronize()
                ms = ref_out["joint_shared"][0]
                d = float((un.cpu() - ms).abs().max())
                mask_ctl = dict(max_abs_unmasked_vs_masked=d, differs=not torch.equal(un.cpu(), ms))
            ms_rec = {f: [] for f in forms}
            host_rec = {f: [] for f in forms}
            call_rec = {f: [] for f in forms}
            for rep in range(a.reps):
                k0 = rep % len(forms)
                for f in forms[k0:] + forms[:k0]:
                    scrub.add_(1)
                    torch.cuda.synchronize()
                    evs = [torch.cuda.Event(enable_timing=True) for _ in range(len(plans[f]) + 1)]
                    h0 = time.perf_counter()
                    evs[0].record()
                    run_form(S, plans, f, events=evs[1:])
                    h1 = time.perf_counter()
                    torch.cuda.synchronize()
                    ms_rec[f].append(evs[0].elapsed_time(evs[-1]))
                    call_rec[f].append([evs[i].elapsed_time(evs[i + 1]) for i in range(len(evs) - 1)])
                    host_rec[f].append((h1 - h0) * 1e3)
                    order_log.append(f)
            for f in forms:
                xs = sorted(ms_rec[f])
                timing[f] = dict(median_ms=statistics.median(xs), p95_ms=xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))],
                                 min_ms=xs[0], reps=len(xs), host_median_ms=statistics.median(host_rec[f]),
                                 per_call_median_ms=[statistics.median(c) for c in zip(*call_rec[f])], group=group, all_ms=ms_rec[f])
                scrub.add_(1)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run_form(S, plans, f)
                    torch.cuda.synchronize()
                tp = os.path.join(a.out, "trace_%s.json" % f)
                prof.export_chrome_trace(tp)
                kineto[f] = JP.trace_busy(json.load(open(tp)))
                log("form %s: median %.2f ms (p95 %.2f), host %.2f ms, kernels %d busy %.2f idle %.2f ms, max |dlogit| %.3f"
                    % (f, timing[f]["median_ms"], timing[f]["p95_ms"], timing[f]["host_median_ms"], kineto[f]["n"],
                       kineto[f]["busy_ms"], kineto[f]["idle_ms"], logit_g[f]["max_abs"]))
            del plans
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    forms = [f for f in JP.FORMS if f in a.forms]
    if len(forms) > 1:
        base = forms[0]
        rb = {r: i for i, r in enumerate(rows_of(base, S))}
        xb = torch.cat(ref_out[base])
        for f in forms[1:]:
            mx, ag, n = 0.0, 0, 0
            xf = torch.cat(ref_out[f])
            for i, r in enumerate(rows_of(f, S)):
                d = float((xf[i] - xb[rb[r]]).abs().max())
                mx = max(mx, d)
                ag += int(int(xf[i].argmax()) == int(xb[rb[r]].argmax()))
                n += 1
            cross[f] = dict(base=base, max_abs=mx, argmax_agree=ag / max(n, 1))
    gates.append(gate("G2_window_content", not issues2, True, "; ".join(issues2[:6]) or "sampled windows and biases equal their sources"))
    worst = max((g["max_abs"] for g in logit_g.values()), default=float("nan"))
    gates.append(gate("G3_logits", all(g["max_abs"] <= TOL for g in logit_g.values()) and len(logit_g) == len(forms), True,
                      "tol %.4f; " % TOL + "; ".join("%s max %.4f argmax %.4f" % (f, g["max_abs"], g["argmax_agree"]) for f, g in logit_g.items())))
    if "joint_shared" in forms:
        gates.append(gate("G4_mask_control", bool(mask_ctl and mask_ctl["differs"]), True, json.dumps(mask_ctl)))
    gates.append(gate("G5_accounting", not bad_acc, True, "; ".join(bad_acc[:6]) or "sum >= union_sep >= union_request; private masked 0"))
    gates.append(gate("R1_selection", True, False, json.dumps(sel_g)))
    gates.append(gate("R2_cross_form", True, False, json.dumps(cross)))
    hb = {lvl: acc_tot[lvl]["hist_blocks"] for lvl in acc_tot}
    meta = dict(S["meta"], reps=a.reps, warm=a.warm, timing_order=order_log[:64],
                hist_block_ratio=dict(union_request_over_sum=hb["union_request"] / max(hb["sum"], 1),
                                      union_sep_over_sum=hb["union_sep"] / max(hb["sum"], 1)),
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, worst_logit=worst)
    res = dict(meta=meta, forms=forms, gates=gates, timing=timing, kineto=kineto, analytic=analytic_b,
               accounting=dict(total=acc_tot, per_layer=acc), logits=logit_g, selection=sel_g, cross=cross)
    write_json(os.path.join(a.out, "result.json"), res)
    nfail = sum(1 for g in gates if g["must"] and not g["passed"])
    log("cell done: %d MUST gate(s) failed" % nfail)
    return 0


def rows_of(form, S):
    """The rows of a form in output order (the calls' rows concatenated)."""
    out = []
    for name, kinds, layout in JP.FORM_CALLS[form]:
        out += JP.call_rows(S["B"], S["K"], kinds)
    return out


def mode_ncu_cell(a):
    import torch
    os.makedirs(a.out, exist_ok=False)
    S = setup(a)
    cudart = torch.cuda.cudart()
    scrub = torch.empty(SCRUB_BYTES, dtype=torch.uint8, device="cuda")
    wins = []
    k = 0
    with torch.inference_mode():
        for group, gforms in JP.GROUPS:
            forms = [f for f in gforms if f in a.forms]
            if not forms:
                continue
            plans, _ = build_group(S, group, forms)
            for f in forms:
                for _ in range(max(1, a.warm)):
                    run_form(S, plans, f)
            torch.cuda.synchronize()
            for f in forms:
                k += 1
                run_form(S, plans, f)                                   # the identical warm-up of every window
                torch.cuda.synchronize()
                scrub.add_(1)                                           # identical L2 state: the scrub buffer
                torch.cuda.synchronize()
                cudart.cudaProfilerStart()
                for _ in range(k):
                    torch.cuda._sleep(1000)
                run_form(S, plans, f)
                torch.cuda.synchronize()
                cudart.cudaProfilerStop()
                wins.append(dict(form=f, group=group, markers=k, order=k, cache_control=a.cc))
                log("profiled window %d: %s" % (k, f))
            del plans
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    write_json(os.path.join(a.out, "windows.json"), dict(windows=wins, meta=S["meta"], cache_control=a.cc,
                                                         protocol="per window: a warm run of the form, a write of 2 x L2 "
                                                                  "(scrub) outside the window, then the window"))
    return 0


def child_argv(a, mode, out, extra=()):
    argv = [sys.executable, os.path.abspath(__file__), "--mode", mode, "--out", out, "--model", a.model,
            "--workload", a.workload, "--batch", str(a.batch), "--L", str(a.L), "--W", str(a.W), "--K", str(a.K),
            "--reps", str(a.reps), "--warm", str(a.warm), "--forms", ",".join(a.forms)]
    return argv + list(extra)


def run_child(argv, logp, timeout):
    with open(logp, "w") as lf:
        lf.write("# %s\n" % " ".join(argv))
        lf.flush()
        p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            rc = "timeout"
    return rc


def mode_orchestrate(a):
    if os.path.exists(a.out):
        print("REFUSED: --out %s exists" % a.out, file=sys.stderr)
        return 2
    os.makedirs(a.out)
    write_json(os.path.join(a.out, "config.json"), dict(vars(a), env={k: v for k, v in os.environ.items()
                                                                      if k.startswith(("NOSI_", "JH_", "PYTORCH_"))}))
    og = []
    t0 = time.time()
    rc = run_child(child_argv(a, "cell", os.path.join(a.out, "U")), os.path.join(a.out, "U.log"), a.child_timeout)
    log("U exit %s after %.0f s" % (rc, time.time() - t0))
    og.append(gate("O_U_ran", rc == 0 and os.path.exists(os.path.join(a.out, "U", "result.json")), True, "exit %s" % rc))
    for cc in a.variants:
        d = os.path.join(a.out, "P_" + cc)
        rep = os.path.join(a.out, "rep_" + cc)
        ncu = [a.ncu, "--profile-from-start", "off", "--replay-mode", "kernel", "--cache-control", cc, "--clock-control",
               "none", "--nvtx", "--metrics", NCU_METRICS, "--target-processes", "application-only", "-f", "-o", rep]
        t1 = time.time()
        rc = run_child(ncu + child_argv(a, "ncu_cell", d, ("--cc", cc)), os.path.join(a.out, "P_%s.log" % cc), a.child_timeout)
        log("P_%s exit %s after %.0f s" % (cc, rc, time.time() - t1))
        exp = {}
        if os.path.exists(rep + ".ncu-rep"):
            for name, extra in (("raw.csv", ["--csv", "--page", "raw", "--print-units", "base"]),
                                ("raw_nvtx.csv", ["--csv", "--page", "raw", "--print-units", "base", "--print-nvtx-rename", "kernel"])):
                try:
                    with open(os.path.join(a.out, "%s_%s" % (cc, name)), "w") as f:
                        r = subprocess.run([a.ncu, "--import", rep + ".ncu-rep"] + extra, stdout=f, stderr=subprocess.PIPE,
                                           text=True, timeout=1800)
                    exp[name] = r.returncode
                except Exception as e:                                 # recorded, the gate below fails
                    exp[name] = str(e)[:200]
        og.append(gate("O_P_%s_ran" % cc, rc == 0 and exp.get("raw.csv") == 0, True, "exit %s, exports %s" % (rc, exp)))
    rc = mode_summarize(a, og)
    return rc


def mode_summarize(a, og=None):
    og = list(og or [])
    up = os.path.join(a.out, "U", "result.json")
    if not os.path.exists(up):
        write_json(os.path.join(a.out, "summary.json"), dict(gates=og, error="no U/result.json"))
        return max(1, sum(1 for g in og if g["must"] and not g["passed"]))
    u = json.load(open(up))
    pv = {}
    for cc in a.variants:
        wp = os.path.join(a.out, "P_" + cc, "windows.json")
        cp = os.path.join(a.out, "%s_raw.csv" % cc)
        if not (os.path.exists(wp) and os.path.exists(cp)):
            og.append(gate("O_P_%s_windows" % cc, False, True, "windows.json or raw.csv missing"))
            continue
        metas = json.load(open(wp))["windows"]
        wins = {w["markers"]: JP.window_summary(w) for w in JP.ncu_windows(JP.ncu_rows(open(cp, errors="replace").read()))}
        found = [wins.get(m["markers"]) for m in metas]
        wrong = [m["form"] for m, s in zip(metas, found) if s is None or s["forms"] != [m["form"]]]
        og.append(gate("O_P_%s_windows" % cc, not wrong and len(wins) == len(metas), True,
                       "expected %d windows, found %d; wrong or missing %s" % (len(metas), len(wins), wrong)))
        untagged = sum(s["untagged"] for s in found if s)
        og.append(gate("R_P_%s_untagged" % cc, untagged == 0, False, "%d untagged kernels in the windows" % untagged))
        passes = JP.ncu_passes(open(os.path.join(a.out, "P_%s.log" % cc), errors="replace").read()) \
            if os.path.exists(os.path.join(a.out, "P_%s.log" % cc)) else []
        single = bool(passes) and max(passes) == 1
        og.append(gate("%s_P_%s_single_pass" % ("O" if cc == "none" else "R", cc), single, cc == "none",
                       "%d profiled kernels, passes %s" % (len(passes), sorted(set(passes)))))
        pv[cc] = dict(windows=[m for m, s in zip(metas, found) if s], summaries=[s for s in found if s],
                      passes=sorted(set(passes)))
    s = JP.summarize(u, pv)
    s["gates"] = list(s["gates"]) + og
    write_json(os.path.join(a.out, "summary.json"), s)
    open(os.path.join(a.out, "summary.md"), "w").write(JP.summary_md(s))
    nfail = sum(1 for g in s["gates"] if g["must"] and not g["passed"])
    for g in s["gates"]:
        print("%s %s %s: %s" % ("GATE" if g["must"] else "REPORT", g["name"], "PASS" if g["passed"] else "FAIL", g["detail"][:300]))
    print("joint_hbm: %d MUST gate(s) failed; summary %s" % (nfail, os.path.join(a.out, "summary.md")))
    return min(nfail, 125)


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    a.out = os.path.abspath(a.out)
    if a.mode in ("cell", "ncu_cell", "orchestrate") and not (a.model and a.workload):
        print("REFUSED: --model and --workload are required", file=sys.stderr)
        return 2
    try:
        if a.mode == "orchestrate":
            return mode_orchestrate(a)
        if a.mode == "cell":
            return mode_cell(a)
        if a.mode == "ncu_cell":
            return mode_ncu_cell(a)
        return mode_summarize(a)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
