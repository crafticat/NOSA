"""The OVERLAPPED restart (spec section 3 option (b); DESIGN.md section 10):
the two-position catch-up forward for the requests rejected at tick t runs
on a SIDE STREAM during tick t + 1, interleaved per layer behind the main
tick's layer events, with its OWN scoring buffers and captured graph; the
request rejoins at tick t + 2 (core.CatchupState, core.catchup_schedule).

Position 1 (tau + 1, input x_{tau+1}): layer l is launched right after the
main tick's layer l finished (event): it attends the window of layer l as the
main tick left it, the tail rows <= tau (NOT V's exact tau + 1 row: that is
the same position) and its own row at tail_len_after of the tail slot (the
row tick t + 2's V will write; tick t + 2's layer-l update waits on the side
event). Position 2 (tau + 2, input tok1 = argmax of position 1, taken on the
device: no host sync): after position 1's last layer, attends rows <= tau + 1
(V's exact row included) and the same own row. tok1 / tok2 are read by the
driver before tick t + 1's end (a sync on the side stream's argmaxes).

Rules kept: the main tick's S rows of a catching request are dummies; the
catch-up's table updates are journaled per layer on the side stream, and the
main tick t + 2 waits on the side's layer-l event before its layer-l update.
GPU-only (a second CUDA graph per layer); the state machine and the schedule
are core.py (CPU-tested).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from . import core
from . import tick as _tick


class SideScoring:
    """A second capture of the model's pooling / top-k graph per layer over
    private buffers (nosa_llama.py:465-497, the warm-up's capture, replayed
    with these buffers); cis_buf is a real copy buffer (not the table alias)."""

    def __init__(self, model, cache, stream):
        NL = _tick._gpu().NL
        self.sets: List[_tick.ScoringSet] = []
        H = model.num_key_value_heads
        B = model.topk_idx_buf.shape[1]
        out_len = model.pooling_buf_all.shape[2]
        K = model.topk_idx_buf.shape[2]
        dev, dt = model.pooling_buf_all.device, model.pooling_buf_all.dtype
        self.pooling_buf = torch.empty_like(model.pooling_buf_all)
        self.topk_val_q = torch.empty_like(model.topk_val_buf_q)
        self.topk_idx_q = torch.empty_like(model.topk_idx_buf_q)
        self.topk_val = torch.empty_like(model.topk_val_buf)
        self.topk_idx = torch.empty((H, B, K), dtype=torch.int64, device=dev)
        self.mask_buf = torch.empty_like(model.mask_buf)
        max_pooling_buf = self.pooling_buf[:H]
        max_pooling_buf_cis = self.pooling_buf[H:]
        for l, layer in enumerate(model.layers):
            clayer = cache.layers[l]
            score_buf = torch.empty_like(layer.score_buf)
            cis_buf = torch.empty_like(clayer.compressed_cis)
            max_seqlen_comp = int(clayer.cached_compressed_max_seqlen)

            def body():
                NL.nosa_pooling(score_buf, cis_buf, max_seqlen_comp, max_pooling_buf, max_pooling_buf_cis,
                                stride=layer.pooling_stride, block_size=layer.block_size, local_blocks=layer.window_blocks, init_blocks=layer.init_blocks)
                torch.topk(max_pooling_buf, NL.QK_SELECT, dim=-1, sorted=False, out=(self.topk_val_q, self.topk_idx_q))
                self.mask_buf.zero_()
                self.mask_buf.scatter_(2, self.topk_idx_q, True)
                max_pooling_buf_cis.masked_fill_(self.mask_buf, float("inf"))
                torch.topk(max_pooling_buf_cis, layer.topk_blocks, dim=-1, sorted=False, out=(self.topk_val, self.topk_idx))

            with torch.cuda.stream(stream):
                score_buf.copy_(layer.score_buf)
                cis_buf.copy_(clayer.compressed_cis)
                body()                                                  # the warm-up run before the capture, as the model does
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    body()
            self.sets.append(_tick.ScoringSet(score_buf=score_buf, cis_buf=cis_buf, graph=g, topk_idx_buf=self.topk_idx))
        torch.cuda.synchronize()


class CatchupRunner:
    """Drives one tick's catch-up: ``begin`` (before the main tick: the
    embedding and the row plan on the side stream), ``after_main_layer(l)``
    (position 1's layer l on the side stream after the main tick's layer-l
    event), ``finish`` (position 1's head, then position 2 entirely on the side
    stream; returns the events the next tick must wait on per layer), and
    ``results`` (tok1, tok2 on the host: a side-stream sync)."""

    def __init__(self, model, cache, sc: _tick.Scratch, cfg: _tick.TickConfig, side: torch.cuda.Stream, scoring: SideScoring):
        self.model, self.cache, self.sc, self.cfg, self.side, self.scoring = model, cache, sc, cfg, side, scoring
        self.G = _tick._gpu()
        self.layer_done: List[Optional[torch.cuda.Event]] = [None] * model.num_layers
        self.main_done: List[Optional[torch.cuda.Event]] = [None] * model.num_layers
        self.active = False

    def begin(self, tokens: torch.Tensor, position: torch.Tensor, req_idx: torch.Tensor):
        """tokens (n,) = the committed x_{tau+1}; position (n,) = tau + 1."""
        self.n = int(req_idx.numel())
        if self.n == 0:
            self.active = False
            return
        self.req_idx, self.pos1 = req_idx, position
        self.plan = core.row_plan(req_idx, False, True)
        with torch.cuda.stream(self.side):
            self.hidden = F.embedding(tokens.unsqueeze(1), self.model.embed_tokens)
        self.accts = []
        self.active = True

    def main_layer_done(self, l: int):
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        self.main_done[l] = ev
        if self.active:
            self._side_layer(l, self.hidden, self.pos1, pos_kind=1)

    def _tv(self, l: int, pos_kind: int) -> (core.TailView, int):
        eng = self.cache.layers[l].cache_engine
        lay = self.sc.lay
        tl_after = int(eng._tail_block_len_on_gpu)                 # after the main tick's V write at this layer
        if tl_after >= lay.bs or tl_after < 1:
            raise RuntimeError("the catch-up needs a free tail row above V's (tail_len_after=%d)" % tl_after)
        own = lay.tail_slot * lay.bs + tl_after                     # tick t + 2's V row
        prefix = tl_after - 1 if pos_kind == 1 else tl_after        # rows <= tau, or <= tau + 1
        return core.TailView(tail_len_after=tl_after, filled=False, slot_complete=False, tail_rows_for_s=prefix, content_offset=0), own

    def _side_layer(self, l: int, hidden, position, pos_kind: int):
        tv, own = self._tv(l, pos_kind)
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.main_done[l])
            out = _tick.layer_body(self.G, self.model, self.cache, l, hidden, position.unsqueeze(1), self.plan, self.req_idx, False,
                                   self.sc, self.cfg, tv_override=tv, own_row_override=own, scoring=self.scoring.sets[l])
            ev = torch.cuda.Event()
            ev.record(self.side)
        self.layer_done[l] = ev
        if pos_kind == 1:
            self.hidden = out.hidden
        return out.hidden

    def finish(self):
        """Position 1's head, then position 2 through every layer (the main tick
        is done with all layers by now: its events are all recorded)."""
        if not self.active:
            return
        NL = self.G.NL
        with torch.cuda.stream(self.side):
            h = NL.layer_norm(self.hidden, self.model.norm_variance_epsilon, self.model.norm_weight)
            logits1 = _tick.linear_rows(h, self.model.lm_head, self.cfg).float()[:, 0]
            self.tok1 = logits1.argmax(-1)
            hidden2 = F.embedding(self.tok1.unsqueeze(1), self.model.embed_tokens)
        pos2 = self.pos1 + 1
        for l in range(self.model.num_layers):
            hidden2 = self._side_layer(l, hidden2, pos2, pos_kind=2)
        with torch.cuda.stream(self.side):
            h2 = NL.layer_norm(hidden2, self.model.norm_variance_epsilon, self.model.norm_weight)
            self.tok2 = _tick.linear_rows(h2, self.model.lm_head, self.cfg).float()[:, 0].argmax(-1)

    def results(self):
        if not self.active:
            return None, None
        self.side.synchronize()
        return self.tok1, self.tok2

    def wait_before_main_layer(self, l: int):
        """The next main tick's layer l must follow the side's layer l (its
        journal restore and its reads of the window / the provisional row)."""
        ev = self.layer_done[l]
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
