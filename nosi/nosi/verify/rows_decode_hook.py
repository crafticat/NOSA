"""rows_decode_hook: the rows mechanism (shipped decode kernel over B*U rows through
cache_batch_idx + per-row bias) timed on the ORDINARY DECODE ALLOCATION (W = 64 slots),
beside the shipped decode attention call itself, per layer -- E1c.

Why: job 2175533 measured the rows kernel over the union store (W = 128 slots) at 26.2 ms
for U = 1 vs 11.9 ms for the shipped decode over 64 slots: the kernel scans the whole
allocation (the bias masks, it does not skip), so the union store's width, not the row
count, was the dominant cost; the second row added only 3.9 ms. The paired tick keeps
V's rows on the ordinary 64-slot window (+ S's provisional slot), so its attention cost
is the rows kernel over ~64-66 slots -- measured here with U identical rows per request
(same selection: the traffic of the S row over the same window is the same).

Install with NOSI_ROWS_DECODE_BENCH=1 (verify_pilot mode decode): the module-level name
``nosi.nosa_llama.flash_attn_nosa_with_kvcache`` is wrapped; the forward keeps the shipped
output. Identity gate: row 0 of the U-row call must be torch.equal to the shipped output
at the same explicit num_splits (same allocation, same partition).
"""
from __future__ import annotations

import atexit
import json
import os
import time

import torch

STATE = {"records": [], "installed": False, "out": None, "meta": {}}


def install(out_dir: str, tag: str, u_list=(2, 3), splits: int = 4, flush_mb: int = 160):
    import nosi.nosa_llama as NL
    from flash_attn_nosa import flash_attn_with_kvcache as fa
    if STATE["installed"]:
        return
    orig = NL.flash_attn_nosa_with_kvcache
    dev = torch.device("cuda")
    flush = torch.empty(flush_mb << 20, dtype=torch.uint8, device=dev)
    STATE.update(installed=True, out=os.path.join(out_dir, "rows_decode_%s.json" % tag),
                 meta=dict(u_list=list(u_list), splits=int(splits), flush_mb=flush_mb, attn_splits_env=os.environ.get("NOSI_ATTN_SPLITS"), t0=time.time()))
    counter = {"n": 0}

    def timed(fn):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); out = fn(); e1.record()
        return out, (e0, e1)

    def hook(q, k, v, bias, cache_seqlens=None, num_splits=0, **kw):
        B = int(q.shape[0])
        rec = dict(B=B, W_slots=int(k.shape[1]) // 64, call_index=counter["n"] // 32, layer=counter["n"] % 32, shipped_splits=int(num_splits), ev={}, equal={}, max_abs={})
        counter["n"] += 1
        flush.fill_(1)
        out, rec["ev"]["shipped"] = timed(lambda: orig(q, k, v, bias, cache_seqlens=cache_seqlens, num_splits=num_splits, **kw))
        try:
            if kw:
                raise ValueError("unexpected kwargs %s" % sorted(kw))
            for U in u_list:
                qr = q.repeat_interleave(U, dim=0).contiguous()                    # (B*U, 1, Hq, D): rows of one request adjacent
                br = bias.repeat_interleave(U, dim=0).contiguous()                 # (B*U, W*bs, Hkv); the narrowed view br[:B] passes flash_api.cpp:366
                csl = cache_seqlens.repeat_interleave(U).contiguous()
                cbi = torch.arange(B, dtype=torch.int32, device=q.device).repeat_interleave(U).contiguous()
                flush.fill_(1)
                o, rec["ev"]["rows_u%d_s%d" % (U, splits)] = timed(lambda: fa(qr, k, v, br[:B], cache_seqlens=csl, cache_batch_idx=cbi, num_splits=int(splits)))
                o0 = o[0::U]
                rec["equal"]["u%d_row0_equal_shipped" % U] = bool(torch.equal(o0, out))
                rec["equal"]["u%d_rows_identical" % U] = bool(all(torch.equal(o[u::U], o0) for u in range(1, U)))
                rec["max_abs"]["u%d" % U] = float((o0.float() - out.float()).abs().max())
            rec["max_abs"]["shipped_scale"] = float(out.float().abs().max())
        except Exception as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        STATE["records"].append(rec)
        return out

    NL.flash_attn_nosa_with_kvcache = hook
    atexit.register(dump)
    print("[rows_decode] installed: U %s splits %d -> %s" % (list(u_list), splits, STATE["out"]), flush=True)


def dump():
    torch.cuda.synchronize()
    recs = []
    for r in STATE["records"]:
        rr = {k: v for k, v in r.items() if k != "ev"}
        rr["ms"] = {k: float(e0.elapsed_time(e1)) for k, (e0, e1) in r["ev"].items()}
        recs.append(rr)
    STATE["meta"]["t1"] = time.time()
    with open(STATE["out"], "w") as f:
        json.dump(dict(meta=STATE["meta"], records=recs), f)
    print("[rows_decode] wrote %d layer records to %s" % (len(recs), STATE["out"]), flush=True)
