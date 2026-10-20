"""R2 step 7, W5 of docs/R2_TICK_CAPTURE_PLAN.md: S-C (C's r2_residency_sc.ResidencySC, 41078e8) at FIXED CAPACITY,
split for the CAPTURED tick on C's Q-T3 answers (2026-10-21). ResidencySC.resolve is unchanged and stays the
reference; CaptureSC calls the same device functions (device_verify, device_draft, the device mirror, the devops' wait
and gather) on fixed-size buffers, and leaves every host step outside the graph:

  host_part(rows)        BEFORE the replay, once per tick: the row checks of every layer (Residency._check_rows, its
                         shape check on a meta tensor), the tail checks of ResidencySC._tick_rows, sync_mirror (the ONE
                         mirror delta), on the GPU the main stream's waits on the host write-backs registered since the
                         last gather (T4; C's gather would wait them inside the graph, where an outside event cannot be
                         waited), and ONE host-to-device copy of the fixed row buffers. A host-side Plan per layer (the
                         row keys: read_done's key readers).
  device_part(l, sel)    INSIDE the region, per layer: the layer's mirror (DeviceMirror.layer), device_verify on the
                         Rv_cap verify entries, the devops' wait and gather, device_draft on the Rd_cap draft entries,
                         the page lists of the R_cap tick rows. No host read, no host-to-device copy, no host branch on a
                         device value (test S4's lint).
  after_replay(ev)       AFTER the replay: read_done of every layer with ONE reader event (C's Q-T3 (c): no slot is
                         released and taken again inside a tick), the tick's decision records (the decision objects the
                         capture created; their tensors are the graph's static outputs), then ResidencySC.end_tick (ONE
                         device read: the watched flags and every decision; a set flag refuses the tick).

FIXED CAPACITY (C's Q-T3 (d)): R_cap tick rows; Rv_cap verify and Rd_cap draft ENTRIES, the live rows first in both;
a padded entry points at the INERT ROW R_cap of the row buffers (hist_bound 0, no tail: no valid block, so no take, no
protect, no wait, no gather), reads the selection of row min(R_cap - 1, .) (every block invalid under hist_bound 0) and
writes its page list into row R_cap of the page buffer (never returned); the draft pair count (device_draft's
counters[2], Rd x H) is set from the live draft count the host copied in. A padded TICK row (R .. R_cap - 1) gets no
page (num_pages 0): the tick's region gives it its own one-page trash list (r2_tick_fixed F7).
ON THE GPU the devops must wait EXACTLY (CudaDeviceOps(wait_mode="exact"), 41078e8): the superset wait loops over host
events, which a replay cannot repeat.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .r2_residency import N_TAIL
from .r2_residency_sb import device_draft
from .r2_residency_sc import ResidencySC, device_verify


class CaptureError(ValueError):
    pass


class CaptureSC:
    """S-C at fixed capacity for the captured tick (module docstring)."""

    def __init__(self, svc: ResidencySC, R_cap: int, Rv_cap: int, Rd_cap: int, k: int):
        if not isinstance(svc, ResidencySC):
            raise CaptureError("CaptureSC needs a ResidencySC (S-C)")
        for name, v in (("R_cap", R_cap), ("Rv_cap", Rv_cap), ("Rd_cap", Rd_cap), ("k", k)):
            if isinstance(v, bool) or int(v) != v or int(v) < 1:
                raise CaptureError("%s must be a positive int (got %r)" % (name, v))
        if svc.device.type == "cuda" and getattr(svc.dev, "wait_mode", None) != "exact":
            raise CaptureError("a captured S-C needs CudaDeviceOps(wait_mode='exact'): the superset wait loops over "
                               "host events, which a replay cannot repeat")
        self.svc = svc
        self.R_cap, self.Rv_cap, self.Rd_cap, self.k = int(R_cap), int(Rv_cap), int(Rd_cap), int(k)
        self.wait = not svc.sc_mut.get("no_wait")
        dev = svc.device
        T = N_TAIL
        R1 = self.R_cap + 1                                           # row R_cap: the INERT row of padded entries
        self.n_flat = R1 * (4 + T) + self.Rv_cap + self.Rd_cap + 1
        self.flat = torch.zeros(self.n_flat, dtype=torch.int64, device=dev)
        o = 0
        self.rb = self.flat[o:o + R1]
        o += R1
        self.hb = self.flat[o:o + R1]
        o += R1
        self.nt = self.flat[o:o + R1]
        o += R1
        self.last = self.flat[o:o + R1]
        o += R1
        self.tails = self.flat[o:o + R1 * T].view(R1, T)
        o += R1 * T
        self.vix = self.flat[o:o + self.Rv_cap]
        o += self.Rv_cap
        self.dix = self.flat[o:o + self.Rd_cap]
        o += self.Rd_cap
        self.nd = self.flat[o:o + 1]
        P = self.k + T
        self.page = torch.full((R1, svc.H, P), -1, dtype=torch.int64, device=dev)
        self.npg = torch.zeros((R1, svc.H), dtype=torch.int64, device=dev)
        self.vdec = [None] * svc.L
        self.ddec = [None] * svc.L
        self.live = None
        self.plans = None
        self.stats = dict(ticks=0, uploads=0)

    # ------------------------------------------------------------------ before the replay (host)
    def host_part(self, rows: Sequence) -> None:
        svc = self.svc
        R = len(rows)
        vrows = [i for i, r in enumerate(rows) if r.kind == "v"]
        drows = [i for i, r in enumerate(rows) if r.kind == "d"]
        if R > self.R_cap or len(vrows) > self.Rv_cap or len(drows) > self.Rd_cap:
            raise CaptureError("a tick of %d rows (%d verify, %d draft) exceeds the capacity %d (%d, %d)"
                               % (R, len(vrows), len(drows), self.R_cap, self.Rv_cap, self.Rd_cap))
        if svc._tick or self.live is not None:
            raise CaptureError("a tick is in progress (host_part before the previous tick's after_replay)")
        meta = torch.empty((svc.H, R, self.k), device="meta")
        for l in range(svc.L):
            svc._check_rows(l, rows, meta)
        for i, r in enumerate(rows):                                   # ResidencySC._tick_rows' checks
            if r.hist_bound > svc.nblk:
                raise ValueError("row %d: hist_bound %d > nblk %d (the mirror names blocks < nblk only)"
                                 % (i, r.hist_bound, svc.nblk))
            if len(r.tail) > N_TAIL:
                raise ValueError("at most %d tail pages per row" % N_TAIL)
            for (_, s) in r.tail:
                if not svc.tail0 <= s < svc.tail0 + N_TAIL:
                    raise ValueError("tail slot %d outside [%d, %d)" % (s, svc.tail0, svc.tail0 + N_TAIL))
        svc.sync_mirror()
        if svc.device.type == "cuda":                                  # T4, before the replay
            main = torch.cuda.current_stream()
            for ev in svc._hist_unwaited:
                main.wait_event(ev)
            svc._hist_unwaited = []
        pad = self.R_cap + 1 - R
        host = ([r.b for r in rows] + [0] * pad
                + [r.hist_bound for r in rows] + [0] * pad
                + [len(r.tail) for r in rows] + [0] * pad
                + [r.pos % svc.bs + 1 for r in rows] + [1] * pad)
        for r in rows:
            host += [s for (_, s) in r.tail] + [-1] * (N_TAIL - len(r.tail))
        host += [-1] * (N_TAIL * pad)
        host += vrows + [self.R_cap] * (self.Rv_cap - len(vrows))
        host += drows + [self.R_cap] * (self.Rd_cap - len(drows))
        host += [len(drows)]
        h = torch.tensor(host, dtype=torch.int64)
        if svc.device.type == "cuda":
            self.flat.copy_(h.pin_memory(), non_blocking=True)
        else:
            self.flat.copy_(h)
        svc.row_uploads += 1
        self.stats["uploads"] += 1
        self.plans = [svc._new_plan(rows) for _ in range(svc.L)]
        self.live = (list(rows), vrows, drows)

    # ------------------------------------------------------------------ THE REGION (per layer)
    def device_part(self, l: int, sel: torch.Tensor):
        """sel (H, R_cap, k) int64 on the device: the tick rows' selections (padded rows included). Returns the page
        lists of the R_cap tick rows: (page_table (R_cap, H, k + 4) int32, num_pages (R_cap, H) int32, last_len
        (R_cap,) int32, seq_idx (R_cap,) int32)."""
        svc = self.svc
        if sel.dim() != 3 or sel.shape[1] != self.R_cap or sel.shape[2] != self.k:
            raise CaptureError("sel must be (H, %d, %d)" % (self.R_cap, self.k))
        m = svc.dmirror.layer(l)
        self.page.fill_(-1)
        self.npg.zero_()
        rd = self.R_cap - 1
        vix = self.vix
        vdec = device_verify(m, self.rb[vix], self.hb[vix], self.tails[vix], self.nt[vix],
                             sel.index_select(1, vix.clamp(max=rd)), svc.C, svc.X, svc.sc_mut)
        if self.wait:
            svc.dev.wait(l, vdec)
        svc.dev.gather(l, vdec)
        self.page.index_put_((vix,), vdec.page)
        self.npg.index_put_((vix,), vdec.num_pages)
        self.vdec[l] = vdec
        dix = self.dix
        ddec = device_draft(m, self.rb[dix], self.hb[dix], self.tails[dix], self.nt[dix],
                            sel.index_select(1, dix.clamp(max=rd)), svc.C, svc.X, svc.R_ver, svc.sb_mut)
        ddec.counters[2:3].copy_(self.nd * svc.H)                      # the live draft pairs, not Rd_cap x H
        self.page.index_put_((dix,), ddec.page)
        self.npg.index_put_((dix,), ddec.num_pages)
        self.ddec[l] = ddec
        R = self.R_cap
        return (self.page[:R].to(torch.int32), self.npg[:R].to(torch.int32), self.last[:R].to(torch.int32),
                self.rb[:R].to(torch.int32))

    # ------------------------------------------------------------------ after the replay (host)
    def after_replay(self, ev) -> None:
        svc = self.svc
        if self.live is None:
            raise CaptureError("after_replay without host_part")
        rows, vrows, drows = self.live
        for l in range(svc.L):
            svc.read_done(l, self.plans[l], ev)
        svc._tick = [dict(l=l, vrows=[rows[i] for i in vrows], vidx=list(vrows),
                          vdec=self.vdec[l] if vrows else None, drows=[rows[i] for i in drows], ddec=self.ddec[l]
                          if drows else None) for l in range(svc.L)]
        self.live = self.plans = None
        self.stats["ticks"] += 1
        svc.end_tick()
