"""rows_bench_hook: time the 'rows attention' (verify/rows_attention.py, reuse ledger
(ii-a)) BESIDE Path 1 inside the shipped verify path, per layer, with the same q /
allocation / round -- the microbench the reuse ledger section 3.5 asked for.

Install with NOSI_ROWS_BENCH=1 (verify_pilot.py, before the model loads): the
module-level name ``nosi.nosa_llama.verify_attention`` is replaced by a wrapper
that (1) runs Path 1 (timed, its output is what the forward keeps: the pilot's
numerics are unchanged), (2) builds the rows bias (timed), (3) runs the shipped
decode kernel over B*U rows for every (row order, num_splits) variant (timed,
L2 flushed before each variant), (4) records max |delta| vs Path 1, torch.equal
across row orders at equal splits, and the union size per row. Events are read
at exit after a device sync; nothing here blocks the forward.

Row orders: 'adjacent' = row b*U + u (cache_batch_idx = arange(B).repeat_interleave(U):
the two rows of a request are neighbours in grid z -> L2 reuse of the request's
K/V is possible); 'umajor' = all position-0 rows, then all position-1 rows.
"""
from __future__ import annotations

import atexit
import json
import os
import time

import torch

from . import rows_attention as RA

STATE = {"records": [], "installed": False, "out": None, "meta": {}}


def _perm(B: int, U: int, device):
    perm = torch.arange(B * U, device=device).view(B, U).t().reshape(-1).contiguous()   # u-major order of the adjacent rows
    inv = torch.empty_like(perm); inv[perm] = torch.arange(B * U, device=device)
    return perm, inv


def _reorder(rb: RA.RowsBias, perm: torch.Tensor) -> RA.RowsBias:
    return RA.RowsBias(bias=rb.bias[perm].contiguous(), selected=rb.selected[perm].contiguous(), visible_rows=rb.visible_rows[perm].contiguous(),
                       cache_batch_idx=rb.cache_batch_idx[perm].contiguous(), cache_seqlens=rb.cache_seqlens[perm].contiguous(), U=rb.U, block_size=rb.block_size)


def install(out_dir: str, tag: str, splits_list=(4, 8, 0), orders=("adjacent", "umajor"), flush_mb: int = 160):
    import nosi.nosa_llama as NL
    if STATE["installed"]:
        return
    orig = NL.verify_attention
    dev = torch.device("cuda")
    flush = torch.empty(flush_mb << 20, dtype=torch.uint8, device=dev)
    STATE.update(installed=True, out=os.path.join(out_dir, "rows_bench_%s.json" % tag),
                 meta=dict(splits_list=list(splits_list), orders=list(orders), flush_mb=flush_mb, attn_splits_env=os.environ.get("NOSI_ATTN_SPLITS"),
                           t0=time.time()))
    counter = {"layer": 0}

    def timed(fn):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); out = fn(); e1.record()
        return out, (e0, e1)

    def hook(q, k_gpu, v_gpu, kv_bias_gpu, rnd, mark=None):
        B = int(k_gpu.shape[0]); U = int(rnd.U); BU, Hq, D = q.shape
        rec = dict(U=U, B=B, rows=BU, call_index=counter["layer"] // 32, layer=counter["layer"] % 32, ev={}, max_abs={}, equal_orders={}, rolled_over=bool(rnd.tail.rolled_over))
        counter["layer"] += 1
        flush.fill_(1)
        out_p1, rec["ev"]["path1"] = timed(lambda: orig(q, k_gpu, v_gpu, kv_bias_gpu, rnd, mark=mark))
        try:
            flush.fill_(1)
            rb, rec["ev"]["bias_build"] = timed(lambda: RA.rows_bias_from_round(kv_bias_gpu, rnd, physical_tail=False))
            rec["union_slots_per_row_mean"] = float(rb.selected.sum(dim=1).float().mean())      # selected (B*U, W, Hkv) -> slots per (row, head)
            rec["union_slots_per_row_max"] = int(rb.selected.sum(dim=1).max())
            q4 = q.reshape(BU, 1, Hq, D).contiguous()
            perm, inv = _perm(B, U, q.device)
            outs = {}
            for order in orders:
                rbo, qo = (rb, q4) if order == "adjacent" else (_reorder(rb, perm), q4[perm].contiguous())
                for s in splits_list:
                    flush.fill_(1)
                    o, rec["ev"]["rows_%s_s%d" % (order, s)] = timed(lambda: RA.rows_attention(qo, k_gpu, v_gpu, rbo, s))
                    outs[(order, s)] = o if order == "adjacent" else o[inv]
            for (order, s), o in outs.items():
                rec["max_abs"]["rows_%s_s%d" % (order, s)] = float((o.float() - out_p1.float()).abs().max())
            for s in splits_list:
                if ("adjacent", s) in outs and ("umajor", s) in outs:
                    rec["equal_orders"]["s%d" % s] = bool(torch.equal(outs[("adjacent", s)], outs[("umajor", s)]))
            rec["max_abs"]["path1_self_scale"] = float(out_p1.float().abs().max())
        except Exception as e:  # never break the pilot's forward: record the failure and keep Path 1's output
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        STATE["records"].append(rec)
        return out_p1

    NL.verify_attention = hook
    atexit.register(dump)
    print("[rows_bench] installed: splits %s orders %s -> %s" % (list(splits_list), list(orders), STATE["out"]), flush=True)


def dump():
    torch.cuda.synchronize()
    recs = []
    for r in STATE["records"]:
        ms = {k: float(e0.elapsed_time(e1)) for k, (e0, e1) in r["ev"].items()}
        rr = {k: v for k, v in r.items() if k != "ev"}; rr["ms"] = ms
        recs.append(rr)
    STATE["meta"]["t1"] = time.time()
    with open(STATE["out"], "w") as f:
        json.dump(dict(meta=STATE["meta"], records=recs), f)
    print("[rows_bench] wrote %d layer records to %s" % (len(recs), STATE["out"]), flush=True)
