import torch
from queue import Queue
import time
from .flash_cache_engine.flash_h2d_mask import flash_h2d_from_mask
from .flash_cache_engine.flash_h2d_mask_bias import flash_h2d_from_mask_bias
from . import transfer_trace as _tt   # retroinfer-eval fork: event timing, off unless NOSI_TRANSFER_TRACE
from . import avail_policy as _avail   # retroinfer-eval fork: restricted availability, off unless NOSI_AVAIL
from torch.utils.cpp_extension import load
from torch.cuda import nvtx
import argparse
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import CacheLayerMixin, DynamicLayer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import torch.nn.functional as F
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

diff = load(
    name="diff_offload",
    sources=[os.path.join(this_dir, "flash_cache_engine/diff_offload.cpp"), os.path.join(this_dir, "flash_cache_engine/diff_offload_kernel.cu")],
    verbose=False,
)

# ---------------------------------------------------------------------------
# retroinfer-eval fork: VICTIM POOL (knob NOSI_POOL_BLOCKS, default 0).
#
# NOSI's GPU block cache is exact-fit -- `topk` slots for a `topk`-block
# selection (the allocation below) -- so a block that leaves the selection and
# re-enters it a few steps later is fetched over PCIe again. Replaying the
# measured selection sequence through an LRU (scripts/nosi_cache_sweep.py,
# REPRODUCE.md experiment 1) shows misses per (layer, KV head, request, step)
# fall from 3.580 at 63 usable slots to 1.200 at 80, 0.863 at 96, 0.708 at 128
# and 0.671 at 192, flat thereafter: ~80% of the fetch is avoidable.
#
# The pool adds P extra block slots per (layer, KV head, request) at slot
# indices topk .. topk+P-1 of the SAME cache tensors, BEYOND `_cache_lens`,
# where flash_attn_nosa never reads (it takes no block table; it reads rows
# 0.._cache_lens-1 as one flat sequence). So the attended set is still exactly
# slots 0..topk-1; only WHERE a block is read from moves, never WHICH blocks
# the model attends to.
#
# THAT IS NOT ENOUGH ON ITS OWN, and job 2173299 proved it. "Attention never
# READS past _cache_lens" does not imply "attention does not DEPEND on the
# allocation": flash-attention picks its split-KV partition from
# `kcache.size(1)` (flash_api.cpp:338 -> :224 -> :235 ->
# flash_fwd_kernel.h:594), so a longer tensor recombines the partial softmax
# accumulators in a different fp32 order and the bf16 logits move by one ULP.
# At p=65 that flipped a near-tied top-k block on decode step 0 -- before the
# pool had moved a single byte -- and the run diverged. The engine therefore
# hands attention a VIEW of exactly topk*block_size rows (prefill_update
# section 3c, and the return of the pooled decode below), which makes the
# served output identical to p=0 by construction at every pool size.
# NOTE the name of that method is deliberately NOT repeated here:
# tests/test_nosi_pool_reference.py pins it to exactly two occurrences in this
# file, the guarded bind and the def, so that a second binding site cannot
# hide.
#
# P = 0 IS THE UPSTREAM PATH, BYTE FOR BYTE: nothing below is imported, no
# extension is compiled, no tensor is allocated, no kernel is launched, and
# `decode_update` is bound to the untouched upstream method at construction.
# The knob is read once, here, so a run cannot change P halfway.
#
# The one OTHER file the knob reaches is transfer_trace.py (the instrument, off
# unless NOSI_TRANSFER_TRACE). It binds its per-event recorder and its event-name
# tuple at construction for the same reason, so a P = 0 mode-"1" baseline is the
# same instrument as the baselines already on record (jobs 2170892 / 2171644).
#
# SCOPE: this engine only. cache_engine_gpu.py (offload=0, the GPU-resident
# control) never reads the knob and is untouched.
POOL_BLOCKS = int(os.environ.get("NOSI_POOL_BLOCKS", "0") or 0)
_pool = None
_flash_pool_swap = None
if POOL_BLOCKS > 0:
    # a SEPARATE extension name on purpose: adding these sources to the
    # diff_offload load() above would change that extension's build hash and
    # force every P=0 run to rebuild it
    _pool = load(
        name="nosi_pool",
        sources=[os.path.join(this_dir, "flash_cache_engine/pool_update.cpp"), os.path.join(this_dir, "flash_cache_engine/pool_update_kernel.cu")],
        verbose=False,
    )
    from .flash_cache_engine.flash_pool_swap import flash_pool_swap as _flash_pool_swap

class CacheEngine:
    def __init__(self,
        gpu_mem_usage: int = 60,
        num_layers: int = 28,
        block_size: int = 64,
        head_num: int = 2,
        head_dim: int = 128,
        cpu_gpu_mem_size_fraction: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        max_batch_size: int = 1024,
        max_seq_len: int = 131072,
        topk: int = 64,
        has_kv_bias: bool = False,
    ):
        single_block_size = block_size * head_num * head_dim * (2 if dtype in [torch.float16, torch.bfloat16] else 4)
        self.block_num = gpu_mem_usage * 1024**3 // (single_block_size * num_layers * 2)
        self.block_num_cpu = self.block_num * cpu_gpu_mem_size_fraction
        self.block_size = block_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.max_batch_size = max_batch_size

        self._k_cpu = None
        self._v_cpu = None
        self._k_gpu = None
        self._v_gpu = None
        # retroinfer-eval fork: the victim pool makes the three GPU tensors
        # longer than the attended window, so the pooled decode hands attention
        # a VIEW of the window instead of the whole allocation. Built in
        # prefill_update; None until then, and never used on the p=0 path.
        self._att_rows = None
        self._k_gpu_att = None
        self._v_gpu_att = None
        self._kv_bias_gpu_att = None


        self.seq_length = 0

        self.total_load_all = 0
        self.total_offload_all = 0
        self.total_update_time = 0
        self.is_first_decode = True
        self.max_seq_len = max_seq_len
        self.topk = topk
        self.max_gen_len = 8192

        # retroinfer-eval fork: victim pool. Dispatch is bound here, the way
        # upstream already binds the has_kv_bias variant, because this file's
        # own comment (decode_update_no_kv_bias, "用python写分支会非常慢") warns
        # that a Python branch in the decode path is very slow -- and because
        # binding is the only way P=0 provably executes the upstream body.
        self.pool_blocks = POOL_BLOCKS
        self._pool_stamp_base = 0
        # retroinfer-eval fork: the availability hook and the victim pool must
        # never run together. pool_update sits BETWEEN diff and the copy_ and may
        # PARK the outgoing block Y of a slot in anticipation of a fetch that the
        # hook would then deny: the slot would still hold Y while the pool also
        # held Y, the map would name Y twice, and diff_offload's free-slot
        # counting -- which needs pairwise distinct ids -- would break, producing
        # a plausible, finite, WRONG output with no error. The pooled decode also
        # hands attention a NARROWED VIEW, and combining a second mechanism with
        # the length the split-KV partition is derived from is the defect job
        # 2173299 measured at p=65. Refuse at construction, the way the pool
        # already refuses an inconsistent configuration.
        if _avail.SPEC not in ("", "0", "off") and POOL_BLOCKS > 0:
            raise RuntimeError(
                "NOSI_AVAIL=%r together with NOSI_POOL_BLOCKS=%d: the availability "
                "hook and the victim pool may not run in the same process "
                "(pool_update parks the victim of a fetch the hook can deny, which "
                "duplicates a block id and breaks diff_offload's slot accounting)."
                % (_avail.SPEC, POOL_BLOCKS))
        if _avail.SPEC not in ("", "0", "off") and not has_kv_bias:
            raise RuntimeError(
                "NOSI_AVAIL=%r but this CacheEngine was built with has_kv_bias=False; "
                "the availability hook is implemented only in decode_update_has_kv_bias, "
                "so this run would silently execute an un-hooked path." % (_avail.SPEC,))
        if POOL_BLOCKS > 0 and not has_kv_bias:
            # NOSA-8B always takes the has_kv_bias path (nosa_llama.py:828,850).
            # Refuse rather than silently serve an un-pooled run that a manifest
            # would then report as a pooled one.
            raise RuntimeError(
                "NOSI_POOL_BLOCKS=%d but this CacheEngine was built with "
                "has_kv_bias=False; the victim pool is implemented only in "
                "decode_update_has_kv_bias" % POOL_BLOCKS)
        if POOL_BLOCKS > 0:
            self.decode_update = self.decode_update_has_kv_bias_pool
        else:
            self.decode_update = self.decode_update_has_kv_bias if has_kv_bias else self.decode_update_no_kv_bias
        
        
    def prefill_update(self, key_states, value_states, kv_bias, current_batch_pos, total_bsz):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # return: None
        B, S, H, D = key_states.shape

        assert H == self.head_num and D == self.head_dim

        num_full_blocks = S // self.block_size
        tail_len = S % self.block_size
        self.seq_length = S


        if current_batch_pos == 0:
            # A. 开空间: CPU, GPU, block_map
            # 1. 在cpu上开空间，直接开到self.max_seq_len
            self._k_cpu = torch.empty((total_bsz, S+self.max_gen_len, H, D), dtype=self.dtype, device='cpu').pin_memory()
            self._v_cpu = torch.empty((total_bsz, S+self.max_gen_len, H, D), dtype=self.dtype, device='cpu').pin_memory()

            # 2. 在gpu上开空间，开topk * block_size大小
            # retroinfer-eval fork: + self.pool_blocks victim slots, which live
            # past _cache_lens and are therefore invisible to attention. The
            # expression is IDENTICAL to upstream when pool_blocks == 0.
            _gpu_slots = self.topk + self.pool_blocks
            if self.pool_blocks > 0:
                # THE INT32 ENVELOPE, re-derived because the pool moves one half
                # of it. flash_h2d_from_mask{,_bias} do NOT cast their program
                # ids (flash_h2d_mask.py:23-25, :48-54), so `pid_b * stride_b`
                # is int32 x int32 on BOTH sides of the copy:
                #   host side  B * (S + max_gen_len) * H * D  < 2**31
                #              (the measured wall: 64K x 112 passes, x120 aborts)
                #   GPU side   B * (topk + P) * block_size * H * D < 2**31,
                #              which the pool DIVIDES by (topk+P)/topk.
                # Guarded to the pooled path so p=0 stays upstream byte for byte;
                # env/slurm/nosi_pool.sbatch refuses out-of-envelope cells for
                # every p, before any GPU time is spent.
                _hd = H * D
                assert total_bsz * (S + self.max_gen_len) * _hd < 2**31, (
                    "host-side int32 gather overflow: B=%d S=%d -> %d elements"
                    % (total_bsz, S, total_bsz * (S + self.max_gen_len) * _hd))
                assert total_bsz * _gpu_slots * self.block_size * _hd < 2**31, (
                    "GPU-side int32 gather overflow: B=%d topk+p=%d -> %d elements"
                    % (total_bsz, _gpu_slots, total_bsz * _gpu_slots * self.block_size * _hd))
            self._k_gpu = torch.empty((total_bsz, _gpu_slots * self.block_size, H, D), dtype=self.dtype, device=self.device)
            self._v_gpu = torch.empty((total_bsz, _gpu_slots * self.block_size, H, D), dtype=self.dtype, device=self.device)
            self._kv_bias_gpu = torch.empty((total_bsz, _gpu_slots * self.block_size, H), dtype=self.dtype, device=self.device)

            # 3. block_map
            self._block_map = torch.full((H, total_bsz, self.topk), -1, dtype=torch.int64, device=self.device)
            # 开好中间buffer方便外面用cuda graph包起来
            self._new_block_map_buf = torch.empty_like(self._block_map) 
            self._load_mask = torch.empty_like(self._new_block_map_buf)

            if self.pool_blocks > 0:
                # 3b. retroinfer-eval fork: victim-pool state, one row per
                # (kv head, request). An EMPTY slot q carries age -P+q: the
                # values are pairwise distinct and all strictly below every real
                # stamp (which is >= 0), so empties are consumed first, in slot
                # order, and the LRU comparison never has to break a tie.
                _p = self.pool_blocks
                self._pool_map = torch.full((H, total_bsz, _p), -1, dtype=torch.int64, device=self.device)
                self._pool_age = (torch.arange(_p, dtype=torch.int64, device=self.device) - _p).expand(H, total_bsz, _p).contiguous()
                self._pool_target = torch.full((H, total_bsz, self.topk), -1, dtype=torch.int64, device=self.device)
                self._pool_action = torch.zeros((H, total_bsz, self.topk), dtype=torch.int8, device=self.device)
                self._pool_stamp_base = 0
                # Not required for correctness -- attention never reads past
                # _cache_lens, and a pool slot is only ever read back when
                # _pool_map[q] >= 0, which implies it was written first. COST,
                # corrected: there is one CacheEngine PER LAYER
                # (InfLLMv2CacheLayer.__init__ below), so this runs 32 times per
                # document, not once: ~0.4 ms per layer, ~13 ms and ~17 GB of
                # writes per document at p=129 B=64. That is PREFILL time; the
                # benchmark's [Speed] and every steps.csv number are measured
                # inside the decode loop (nosa_llama.batch_generate_benchmark
                # starts its clock after decode step 0), so it does not enter
                # P-3 or P-4. It makes a pool dump readable and protects any
                # future kernel that is less careful.
                self._k_gpu[:, self.topk * self.block_size:].zero_()
                self._v_gpu[:, self.topk * self.block_size:].zero_()
                self._kv_bias_gpu[:, self.topk * self.block_size:].zero_()

                # 3c. THE WINDOW ATTENTION IS ALLOWED TO SEE -- exactly topk
                # blocks, at every pool size.
                #
                # WHY. flash-attention's split-KV partition is NOT a function of
                # cache_seqlens. It is derived from the ALLOCATED length of the
                # cache tensor: flash_api.cpp:338 takes `seqlen_k` from
                # `kcache.size(1)`, :224 turns that into `num_n_blocks`, :235
                # picks `num_splits` from it, and flash_fwd_kernel.h:594 cuts
                # `n_blocks_per_split` out of it. Only :598 clamps the LAST
                # block by the real `cache_seqlens`. So handing attention the
                # padded tensor moves the seams at which the partial softmax
                # accumulators are cut and recombined in fp32, which moves the
                # bf16 logits by one ULP -- enough to flip a near-tied top-k
                # block and send the whole autoregressive trace elsewhere.
                #
                # MEASURED (job 2173299, 16128 x 16, 64 steps, 2 documents):
                # p=2 and p=4 happen to land on p=0's partition (6 splits of 6
                # n-blocks) and were bit-identical on all 128 (doc, step) logit
                # hashes; p=65 lands on 11 n-blocks per split and matched on 0
                # of 128, with a maximum logit delta of 20.22 and argmax
                # agreement 0.5938 on document 0. The pool bookkeeping was
                # correct throughout -- the divergence is already present at
                # decode step 0, where the pool has emitted nothing at all.
                #
                # WHY A VIEW IS SAFE. It is a narrow on dim 1, so stride(-1)
                # stays 1 (the only contiguity flash-attn requires,
                # flash_api.cpp:312-315) and every stride the kernel uses is
                # read off the tensor it is given (flash_api.cpp:63, :66, :76,
                # :78), batch stride included. The window is always long enough:
                # `_cache_lens` is (topk-1)*block_size + _tail_block_len_on_gpu
                # and the pooled decode asserts _tail_block_len_on_gpu <=
                # block_size, so _cache_lens <= topk*block_size = _att_rows.
                #
                # WHAT MUST NOT BE NARROWED: flash_pool_swap and the two host
                # gathers. They address the pool rows, and flash_pool_swap
                # derives P from `S_GPU // block_size - topk`
                # (flash_pool_swap.py), so a narrowed tensor would tell it the
                # pool has no slots at all.
                self._att_rows = self.topk * self.block_size
                self._k_gpu_att = self._k_gpu[:, :self._att_rows]
                self._v_gpu_att = self._v_gpu[:, :self._att_rows]
                self._kv_bias_gpu_att = self._kv_bias_gpu[:, :self._att_rows]
                # once, at prefill, so the decode path stays a bare return
                assert (self._k_gpu_att.shape[1] == self._v_gpu_att.shape[1]
                        == self._kv_bias_gpu_att.shape[1] == self._att_rows), (
                    "the attended view is %d rows, not topk*block_size = %d"
                    % (self._k_gpu_att.shape[1], self._att_rows))

            # 4. 直接把尾块在GPU上设置好
            self._tail_block_len_on_gpu = S % self.block_size

        if self._tail_block_len_on_gpu > 0: # 如果有尾块，搬到gpu上
            self._tail_block_idx_on_gpu = self.topk - 1 # 放在最后一个位置，方便用seq_len来
            _tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size

            self._k_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :, :].copy_(key_states[:, -self._tail_block_len_on_gpu:, :, :], non_blocking=True)

            self._v_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :, :].copy_(value_states[:, -self._tail_block_len_on_gpu:, :, :], non_blocking=True)

            self._kv_bias_gpu[current_batch_pos:current_batch_pos+B, _tail_block_base_pos:_tail_block_base_pos+self._tail_block_len_on_gpu, :].copy_(kv_bias[:, -self._tail_block_len_on_gpu:, :], non_blocking=True)

            self._block_map[:, current_batch_pos:current_batch_pos+B, self._tail_block_idx_on_gpu] = S // self.block_size
        else: # 如果没有尾块，在gpu上开一个块作为尾块，预备下一次decode写入
            self._tail_block_idx_on_gpu = self.topk - 1 # 放在最后一个位置，方便用seq_len来
            self._block_map[:, current_batch_pos:current_batch_pos+B, self._tail_block_idx_on_gpu] = S // self.block_size

        if current_batch_pos == 0:
            self._cache_lens = torch.full((total_bsz,), (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu, dtype=torch.int32, device=self.device)        

        # B. 把kv cache写到cpu
        self._k_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(key_states, non_blocking=True)
        self._v_cpu[current_batch_pos:current_batch_pos+B, :S, :, :].copy_(value_states, non_blocking=True)

        return
    
    def decode_update_has_kv_bias(self, key_states, value_states, kv_bias, topk_idx):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # kv_bias: (head_num, seq_len, 1)
        # topk_idx: list, [H, B, M]
        # return: (self._k_gpu, self._v_gpu, mapped_topk_idx)
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        self.seq_length += 1
        self._cache_lens[...] += 1

        # A. 首先处理尾块。尾块一定写在GPU上，且self._tail_block_idx_on_gpu指示的位置一定是可写的
        _tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu
        self._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)
        self._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)


        self._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, self.seq_length-1:self.seq_length, :], non_blocking=True)


        self._tail_block_len_on_gpu += 1
        # 先判断要不要写回。如果需要写回，在完成该层的计算后再写回 TODO: 最后弄写回的逻辑
        tail_full = self._tail_block_len_on_gpu == self.block_size
        # B. 计算load mask和新的映射 尾块的映射不会动
        _tr = _tt.TRACE
        if _tr is not None: _tr.fetch_begin()
        diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)
        # B2. RESTRICTED AVAILABILITY (knob NOSI_AVAIL, avail_policy.py). It MUST
        # sit here, between diff_offload and the copy_ below, because this is the
        # only instant at which all three facts coexist: `_block_map` still holds
        # the OLD occupant of every slot, `_new_block_map_buf` holds the REQUESTED
        # occupant, and `_load_mask[h,b,m] >= 0` names exactly the blocks the
        # selection newly reached, i.e. the ones that must come over the wire.
        # `topk_idx` is READ and never written, so the model's own selection is
        # identical at every availability level. With NOSI_AVAIL unset POLICY is
        # None and not one line of the hook body executes.
        _ap = _avail.POLICY
        if _ap is not None:
            _ap.on_diff(self._block_map, self._new_block_map_buf, self._load_mask,
                        topk_idx, self._tail_block_idx_on_gpu)
        self._block_map.copy_(self._new_block_map_buf, non_blocking=True)
        if _tr is not None: _tr.fetch_mid()

        # C. 从disk取需要load进来的块
        flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)
        flash_h2d_from_mask_bias(self._v_gpu, self._v_cpu, self._kv_bias_gpu, kv_bias, self._load_mask, self.block_size)
        if _tr is not None:
            _tr.fetch_end()
            # (b), THE LRU MISS COUNT AT THIS CAPACITY, IS READ BEFORE ANY
            # DENIAL. In mech=stale the policy rewrote _load_mask to -1 on the
            # denied slots a few lines above, so archiving the live tensor here
            # would count every SUPPRESSED fetch as a hit and understate (b) by
            # exactly the number of denials. pre_denial_mask() returns the clone
            # taken before that rewrite, and None in every other case -- mask
            # mode included, where the mask is untouched.
            _mask_for_trace = self._load_mask
            if _ap is not None:
                _pre = _ap.pre_denial_mask()
                if _pre is not None:
                    _mask_for_trace = _pre
            _tr.record_mask(_mask_for_trace, self._block_map)
            if _ap is not None: _tr.record_avail(_ap.denied())

        # D. 处理写回逻辑 TODO: 测试时CUDA Graph没有包进来这里的逻辑
        if tail_full:
            tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size
            cpu_block_base_pos = self.seq_length - self.block_size # 现在self.seq_len应该可以整除self.block_size
            self._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._k_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            self._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._v_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            # 复用self._tail_block_idx_on_gpu的位置，但是让block_map里这里指向下一个块，并且将这个块认为全部可写
            self._block_map[..., self._tail_block_idx_on_gpu] += 1
            self._tail_block_len_on_gpu = 0
            self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu
            

        return self._k_gpu, self._v_gpu, self._kv_bias_gpu, self._cache_lens 

    def decode_update_has_kv_bias_pool(self, key_states, value_states, kv_bias, topk_idx):
        """decode_update_has_kv_bias PLUS the victim pool (NOSI_POOL_BLOCKS > 0).

        A SEPARATE method, bound at construction, so that the upstream body
        above is not edited at all and a P = 0 run launches exactly the kernels
        it launches today. KEEP THE TWO BODIES IN SYNC: everything here except
        section B2 is copied verbatim from decode_update_has_kv_bias.

        What changes and what does not:
          * _cache_lens is UNCHANGED, and the tensors handed to attention are
            VIEWS of exactly _att_rows = topk*block_size rows, so both the
            length and the contents of what attention sees are identical to
            p=0 and the served logits are identical to p=0. `_cache_lens`
            alone is not sufficient -- see section 3c of prefill_update; that
            was the defect job 2173299 measured.
          * the tail slot (self._tail_block_idx_on_gpu == topk-1) never enters
            the pool. It is excluded twice over: explicitly, by the tail_slot
            argument to pool_update, and implicitly, because the tail block id
            is always inside the selection (nosa_pooling forces the local
            window), so diff_kernel takes its old_hit branch at that slot and
            _load_mask[..., tail] is always -1.
          * the tail write-back (section D) is untouched and runs AFTER the
            gathers, so it cannot race the pool; and since the pool never held
            the tail block, renaming it to T+1 leaves no stale pool entry. The
            just-completed block T is NOT captured into the pool even though its
            bytes are already in HBM. THE COST OF THAT IS ASSERTED, NOT
            MEASURED: one compulsory fetch per 64 decode steps per stream, i.e.
            ~0.4% of the traffic at the shipped 3.58 blocks/stream-step. The
            branch itself needs 64 decode steps to fire at all, which no cell
            reached before the 63-step rewrite of env/slurm/nosi_pool.sbatch;
            the cells there now cross it once, and P-8 registers the capture as
            a separate follow-up rather than leaving the number in a docstring.
        """
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        self.seq_length += 1
        self._cache_lens[...] += 1

        # A. 首先处理尾块。尾块一定写在GPU上，且self._tail_block_idx_on_gpu指示的位置一定是可写的
        _tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu
        self._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)
        self._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)


        self._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, self.seq_length-1:self.seq_length, :], non_blocking=True)


        self._tail_block_len_on_gpu += 1
        # 先判断要不要写回。如果需要写回，在完成该层的计算后再写回
        tail_full = self._tail_block_len_on_gpu == self.block_size
        # THE POOL'S SAFETY MARGIN IS EXACTLY ZERO ROWS, so pin it. _cache_lens
        # is (topk-1)*block_size + _tail_block_len_on_gpu, and the pool's first
        # row is topk*block_size (flash_pool_swap: (TOPK+q)*block_size). Those
        # two are equal the instant _tail_block_len_on_gpu reaches block_size --
        # which is precisely when `tail_full` resets it below. One token more
        # and attention would read pool row 0, i.e. a victim block from an
        # unrelated part of the document, as if it were an attended token: a
        # plausible, finite, WRONG output. Two host integer compares per decode
        # step per layer.
        assert self._tail_block_len_on_gpu <= self.block_size, (
            "tail ran past its block (%d > %d): _cache_lens would reach the pool"
            % (self._tail_block_len_on_gpu, self.block_size))
        # B. 计算load mask和新的映射 尾块的映射不会动
        _tr = _tt.TRACE
        if _tr is not None: _tr.fetch_begin()
        diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)

        # B2. THE POOL. It MUST sit between diff_offload and the copy_ below,
        # because it reads self._block_map while that still holds the OLD ids --
        # the block Y being displaced from each attended slot. After the copy_
        # that value is gone. It must also run before the two gathers, because a
        # pool hit clears _load_mask[s] to -1 and the gathers key off exactly
        # that (flash_h2d_mask.py:32-33), and because on a miss the outgoing
        # block must be parked before the gather overwrites slot s.
        #
        # STREAMS. diff_offload, pool_update and the Triton copies all use
        # PyTorch's current device stream. Keep producer and consumers together.
        # The host timestamp below still prevents replaying this entire method
        # in a CUDA graph: capture would freeze its LRU ages.
        #
        # _new_block_map_buf is deliberately NOT touched: diff already wrote
        # new_map[s] = X for every loaded slot, and after either a pool hit or a
        # host fetch slot s does hold X, so the copy_ below stays correct.
        #
        # _pool_stamp_base is a host int (free). It is correct because this
        # region is not CUDA-graph captured -- see the "TODO: 测试时CUDA
        # Graph没有包进来这里的逻辑" comment on section D. If it ever is
        # captured, move the stamp into a 0-d device tensor.
        if _tr is not None: _tr.pool_begin()
        _pool.pool_update(self._block_map, self._load_mask,
                          self._pool_map, self._pool_age,
                          self._pool_target, self._pool_action,
                          self._pool_stamp_base, self._tail_block_idx_on_gpu)
        _flash_pool_swap(self._k_gpu, self._v_gpu, self._kv_bias_gpu,
                         self._pool_target, self._pool_action,
                         self.topk, self.block_size)
        self._pool_stamp_base += self.topk
        if _tr is not None: _tr.pool_end(); _tr.record_pool(self._pool_action)

        self._block_map.copy_(self._new_block_map_buf, non_blocking=True)
        if _tr is not None: _tr.fetch_mid()

        # C. 从disk取需要load进来的块 -- now only what the pool could not serve
        flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)
        flash_h2d_from_mask_bias(self._v_gpu, self._v_cpu, self._kv_bias_gpu, kv_bias, self._load_mask, self.block_size)
        if _tr is not None: _tr.fetch_end(); _tr.record_mask(self._load_mask, self._block_map)

        # D. 处理写回逻辑 TODO: 测试时CUDA Graph没有包进来这里的逻辑
        if tail_full:
            tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size
            cpu_block_base_pos = self.seq_length - self.block_size # 现在self.seq_len应该可以整除self.block_size
            self._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._k_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            self._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._v_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            # 复用self._tail_block_idx_on_gpu的位置，但是让block_map里这里指向下一个块，并且将这个块认为全部可写
            self._block_map[..., self._tail_block_idx_on_gpu] += 1
            self._tail_block_len_on_gpu = 0
            self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu


        # THE ONLY TENSORS THAT REACH ATTENTION (nosa_llama.py:493, :610).
        # Views of the attended window, never the padded allocation -- see
        # section 3c of prefill_update for why the length of this tensor,
        # and not only its contents, changes the served logits.
        return self._k_gpu_att, self._v_gpu_att, self._kv_bias_gpu_att, self._cache_lens
    
    def decode_update_no_kv_bias(self, key_states, value_states, kv_bias, topk_idx):
        # 将 key_states 和 value_states 放入 cache
        # key_states: (batch_size, seq_len, head_num, head_dim)
        # value_states: (batch_size, seq_len, head_num, head_dim)
        # kv_bias: (head_num, seq_len, 1)
        # topk_idx: list, [H, B, M]
        # return: (self._k_gpu, self._v_gpu, mapped_topk_idx)
        B, S, H, D = key_states.shape
        assert H == self.head_num and D == self.head_dim and S == 1

        self.seq_length += 1
        self._cache_lens[...] += 1

        # A. 首先处理尾块。尾块一定写在GPU上，且self._tail_block_idx_on_gpu指示的位置一定是可写的
        _tail_write_pos = self._tail_block_idx_on_gpu * self.block_size + self._tail_block_len_on_gpu
        self._k_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(key_states, non_blocking=True)
        self._v_gpu[:, _tail_write_pos:_tail_write_pos+1, :, :].copy_(value_states, non_blocking=True)

        # 注意！用python写分支会非常慢
        if kv_bias != None:
            self._kv_bias_gpu[:, _tail_write_pos:_tail_write_pos+1, :].copy_(kv_bias[:, self.seq_length-1:self.seq_length, :], non_blocking=True)


        self._tail_block_len_on_gpu += 1
        # 先判断要不要写回。如果需要写回，在完成该层的计算后再写回 TODO: 最后弄写回的逻辑
        tail_full = self._tail_block_len_on_gpu == self.block_size
        # B. 计算load mask和新的映射 尾块的映射不会动
        _tr = _tt.TRACE
        if _tr is not None: _tr.fetch_begin()
        diff.diff_offload(self._block_map, topk_idx, self._new_block_map_buf, self._load_mask)
        self._block_map.copy_(self._new_block_map_buf, non_blocking=True)
        if _tr is not None: _tr.fetch_mid()

        # C. 从disk取需要load进来的块
        flash_h2d_from_mask(self._k_gpu, self._k_cpu, self._load_mask, self.block_size)
        flash_h2d_from_mask(self._v_gpu, self._v_cpu, self._load_mask, self.block_size)
        if _tr is not None: _tr.fetch_end(); _tr.record_mask(self._load_mask, self._block_map)

        # D. 处理写回逻辑 TODO: 测试时CUDA Graph没有包进来这里的逻辑
        if tail_full:
            tail_block_base_pos = self._tail_block_idx_on_gpu * self.block_size
            cpu_block_base_pos = self.seq_length - self.block_size # 现在self.seq_len应该可以整除self.block_size
            self._k_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._k_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            self._v_cpu[:, cpu_block_base_pos:cpu_block_base_pos+self.block_size, :, :].copy_(self._v_gpu[:, tail_block_base_pos:tail_block_base_pos+self.block_size, :, :], non_blocking=True)
            # 复用self._tail_block_idx_on_gpu的位置，但是让block_map里这里指向下一个块，并且将这个块认为全部可写
            self._block_map[..., self._tail_block_idx_on_gpu] += 1
            self._tail_block_len_on_gpu = 0
            self._cache_lens[...] = (self.topk - 1) * self.block_size + self._tail_block_len_on_gpu
            

        return self._k_gpu, self._v_gpu, self._kv_bias_gpu, self._cache_lens 
        


class InfLLMv2CacheLayer(DynamicLayer):
    def __init__(self, config, has_kv_bias):
        super().__init__()
        # Initialize any additional attributes specific to InfLLMv2CacheLayer
        self.no_rope_keys = torch.tensor([], dtype=torch.float32)
        self.compress_k_cache = []
        self.no_compress_k_cache = []
        self.cached_compressed_cu_seqlens = torch.tensor([], dtype=torch.int32)
        self.compress_k_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.tail_cis = None
        self.compressed_cis = None
        self.tail_cis_len = 0
        self.total_cis = None

        self.cache_engine = CacheEngine(
            gpu_mem_usage=12,
            num_layers=config.num_hidden_layers,
            block_size=64,
            head_num=config.num_key_value_heads,
            head_dim=config.head_dim,
            has_kv_bias=has_kv_bias
        )
        self.seq_length = 0

    def update_no_rope_key(self, key_states):
        if self.no_rope_keys.numel() == 0:
            self.no_rope_keys = key_states
        else:
            self.no_rope_keys = torch.cat([self.no_rope_keys, key_states], dim=1)
        return self.no_rope_keys

    def update_compress_k(self, key_states, cu_seqlens=None):
        if len(self.compress_k_cache) == 0:
            if cu_seqlens is not None:
                self.cached_compressed_cu_seqlens = cu_seqlens.clone()
            self.compress_k_cache_varlen = key_states
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            self.compress_k_cache = list(torch.split(key_states, split_sizes))
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    self.compress_k_cache[index] = torch.cat([self.compress_k_cache[index], k], dim=0)
            new_seq_lens = torch.tensor([tensor.shape[0] for tensor in self.compress_k_cache], dtype=torch.int32)
            new_cumsum = torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32)
            
            self.compress_k_cache_varlen = torch.cat(self.compress_k_cache, dim=0)
            self.cached_compressed_cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), new_cumsum]).to(self.compress_k_cache_varlen.device)
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens
    

    def update_compress_k_prefill(self, key_states, cu_seqlens, max_seqlen, current_batch_pos, total_bsz):
        key_shape = key_states.size()
        key_states_pad = key_states.view(-1, max_seqlen, key_shape[1], key_shape[2]).contiguous()
        if current_batch_pos == 0:
            self.compress_k_cache_varlen = torch.empty((total_bsz, max_seqlen, key_shape[1], key_shape[2]), dtype=key_states.dtype, device=key_states.device)
        self.compress_k_cache_varlen[current_batch_pos:current_batch_pos+key_states_pad.shape[0]].copy_(key_states_pad, non_blocking=True)
        if current_batch_pos == 0:
            self.cached_compressed_cu_seqlens = torch.empty((total_bsz+1,), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            self.cached_compressed_cu_seqlens[:cu_seqlens.shape[-1]].copy_(cu_seqlens)
            self.cached_compressed_max_seqlen = max_seqlen
            # cu_seqlens has total_bsz + 1 entries; one new compressed key per sequence adds [0, 1, ..., total_bsz].
            # Upstream used arange(total_bsz): a size mismatch for B > 1 (RuntimeError at the first decode step that
            # crosses the 16-token compression stride) and a silent no-op by broadcasting at B = 1. Never reached by
            # the 4-token benchmark; reached by any longer decode (retroinfer-eval fork, 2026-09-07).
            self.cached_compressed_cu_seqlens_adder = torch.arange(total_bsz + 1, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        else:
            self.cached_compressed_cu_seqlens[current_batch_pos:current_batch_pos+cu_seqlens.shape[-1]].copy_(cu_seqlens + self.cached_compressed_cu_seqlens[current_batch_pos])


    def update_compress_k_decode(self, key_states, cu_seqlens):
        if key_states is None:
            return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens, self.cached_compressed_max_seqlen
        
        self.compress_k_cache_varlen = torch.cat((self.compress_k_cache_varlen, key_states), dim=1)
        self.cached_compressed_cu_seqlens += self.cached_compressed_cu_seqlens_adder
        self.cached_compressed_max_seqlen += 1
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens, self.cached_compressed_max_seqlen

    def update_no_compress_k_prefill(self, key_states, kernel_size=32, kernel_stride=16, current_batch_pos=None, total_bsz=None):
        # key_states: (B, N, H, D)
        B, N, H, D = key_states.shape
        if current_batch_pos == 0:
            self.no_compress_k_cache = torch.empty((total_bsz, kernel_size, H, D), dtype=key_states.dtype, device=key_states.device)
        self.no_compress_k_cache[current_batch_pos:current_batch_pos+B, :N, :, :].copy_(key_states)
        self.no_compress_k_len = N
        return

    def update_no_compress_k_decode(self, key_states, kernel_size=32, kernel_stride=16):
        # key_states: (B, N, H, D)
        if self.no_compress_k_len >= kernel_size:
            to_compress = self.no_compress_k_cache.clone()
            self.no_compress_k_cache[:, :kernel_stride].copy_(self.no_compress_k_cache[:, kernel_stride:])
            self.no_compress_k_cache[:, kernel_stride:kernel_stride+1].copy_(key_states)
            self.no_compress_k_len = kernel_stride + 1
            return to_compress
        else:
            self.no_compress_k_cache[:, self.no_compress_k_len:self.no_compress_k_len+1, :, :].copy_(key_states)
            self.no_compress_k_len += 1
            return None
    

    # 从update函数拆出来的
    def prefill_update_kv(self, key_states, value_states, kv_bias, current_batch_pos, total_bsz):
        self.seq_length = key_states.shape[1]
        self.cache_engine.prefill_update(key_states, value_states, kv_bias, current_batch_pos, total_bsz)
    
    def decode_update_kv(self, key_states, value_states, kv_bias, topk_idx):
        self.seq_length += 1
        return self.cache_engine.decode_update(key_states.unsqueeze(1), value_states.unsqueeze(1), kv_bias, topk_idx)


    def update(self, key_states, value_states, cache_kwargs=None):
        is_prefill = cache_kwargs.get("is_prefill", True)
        if is_prefill: # prefill
            self.seq_length = key_states.shape[2]
            self.cache_engine.prefill_update(key_states.transpose(1, 2), value_states.transpose(1, 2))
            return key_states, value_states
        else: # decode
            topk_idx = cache_kwargs.get("topk_idx", None)
            self.seq_length += 1
            output = self.cache_engine.decode_update(key_states, value_states, topk_idx)
            return output
    
    def update_cis(self, cis, current_batch_pos, total_bsz, kernel_size=32, kernel_stride=16):
        # cis: (H, B, N)
        H, B, N = cis.shape
        if N > 1: # prefill
            if self.compressed_cis == None:
                full_block_len = N // kernel_stride
                tail_block_len = N % kernel_stride
                full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
                comp_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)
                M = comp_cis.shape[-1]
                self.comp_cis_len = M
                self.compressed_cis =  torch.empty((H, total_bsz, M), dtype=cis.dtype, device=cis.device)
                self.compressed_cis[:, :B, :].copy_(comp_cis, non_blocking=True)
                self.tail_cis = torch.empty((H, total_bsz, kernel_size), dtype=cis.dtype, device=cis.device)
                self.tail_cis_len = kernel_stride + tail_block_len
                self.tail_cis[:, :B, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * kernel_stride:], non_blocking=True)
                return comp_cis
            else:
                full_block_len = N // kernel_stride
                tail_block_len = N % kernel_stride
                full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
                comp_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)
                M = comp_cis.shape[-1]
                self.compressed_cis[:, current_batch_pos:current_batch_pos+B, :].copy_(comp_cis, non_blocking=True)
                self.tail_cis[:, current_batch_pos:current_batch_pos+B, :self.tail_cis_len].copy_(cis[:, :, (full_block_len - 1) * kernel_stride:], non_blocking=True)
                return comp_cis
            # full_block_len = N // kernel_stride
            # tail_block_len = N % kernel_stride
            # full_cis = cis[:, :, :full_block_len * kernel_stride].reshape(H * B, 1, full_block_len * kernel_stride)
            # self.compressed_cis = F.avg_pool1d(full_cis, kernel_size=kernel_size, stride=kernel_stride).reshape(H, B, -1)

            # self.tail_cis = [cis[:, :, (full_block_len - 1) * kernel_stride:]]
            # self.tail_cis_len = kernel_stride + tail_block_len
        else:
            # self.tail_cis.append(cis)
            self.tail_cis[:, :, self.tail_cis_len:self.tail_cis_len+1].copy_(cis, non_blocking=True)
            self.tail_cis_len += 1
            if self.tail_cis_len == kernel_size:
                # tail = torch.cat(self.tail_cis, dim=-1)
                tail = self.tail_cis
                tail_comp = tail.mean(dim=-1, keepdim=True)
                # self.compressed_cis = torch.cat((self.compressed_cis, tail_comp), dim=-1)
                self.compressed_cis[:, :, self.comp_cis_len:self.comp_cis_len+1].copy_(tail_comp, non_blocking=True)
                # self.tail_cis = [tail[:, :, kernel_stride:]]
                self.tail_cis[:, :, :kernel_stride].copy_(self.tail_cis[:, :, kernel_stride:])
                self.tail_cis_len = kernel_stride
            return self.compressed_cis
    
    def update_uncompressed_cis(self, cis, current_batch_pos, total_bsz):
        if self.total_cis is None:
            B, S, H = cis.shape
            self.total_cis = torch.empty((total_bsz, S+1024, H), dtype=cis.dtype, device=cis.device)
            self.total_cis[:B, :S, :].copy_(cis, non_blocking=True)
            self.cis_len = S
            self.cis_prefilled_batch = B
            return cis # prefill用的是fa接口
        elif self.cis_prefilled_batch < total_bsz:
            B, S, H = cis.shape
            assert self.cis_len == S
            self.total_cis[current_batch_pos:current_batch_pos+B, :S, :].copy_(cis, non_blocking=True)
            self.cis_prefilled_batch += B
            return cis
        else:
            self.total_cis[:, self.cis_len:self.cis_len+1, :].copy_(cis, non_blocking=True)
            self.cis_len += 1
            return self.total_cis # 用的是flash_attn_nosa的接口，为了用cuda graph保证地址不变
        
    def get_paged_kv(self):
        return self.cache_engine._k, self.cache_engine._v
    
    def get_seq_length(self):
        return self.seq_length


class InfLLMv2Cache(DynamicCache):
    def __init__(self,
                 config,num_hidden_layers: Optional[int] = None, has_kv_bias=False) -> None:
        super().__init__(config=config)
        self.layers = [InfLLMv2CacheLayer(config, has_kv_bias) for _ in range(num_hidden_layers)] if num_hidden_layers else []
        self._seen_tokens = 0

    def prefill_update_kv(self, key_states, value_states, kv_bias, layer_idx, current_batch_pos, total_bsz):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].prefill_update_kv(key_states, value_states, kv_bias, current_batch_pos, total_bsz)
    
    def decode_update_kv(self, key_states, value_states, kv_bias, layer_idx, topk_idx):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].decode_update_kv(key_states, value_states, kv_bias, topk_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
    
    def update_cis(self, cis, layer_idx, current_batch_pos, total_bsz):
        return self.layers[layer_idx].update_cis(cis, current_batch_pos, total_bsz)

    def update_uncompressed_cis(self, cis, layer_idx, current_batch_pos, total_bsz):
        return self.layers[layer_idx].update_uncompressed_cis(cis, current_batch_pos, total_bsz)
    
    def get_paged_kv(self, layer_idx):
        return self.layers[layer_idx].get_paged_kv()

    def update_no_rope_key(self, key_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update_no_rope_key(key_states)

    def update_compress_k_prefill(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None, max_seqlen=None, current_batch_pos=None, total_bsz=None):
        return self.layers[layer_idx].update_compress_k_prefill(key_states, cu_seqlens, max_seqlen, current_batch_pos, total_bsz)

    def update_compress_k_decode(self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None):
        return self.layers[layer_idx].update_compress_k_decode(key_states, cu_seqlens)

    def update_no_compress_k_prefill(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None, current_batch_pos=None, total_bsz=None):
        return self.layers[layer_idx].update_no_compress_k_prefill(key_states, kernel_size, kernel_stride, current_batch_pos=current_batch_pos, total_bsz=total_bsz)

    def update_no_compress_k_decode(self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None):
        return self.layers[layer_idx].update_no_compress_k_decode(key_states, kernel_size, kernel_stride)

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)
    
    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()
    
    def get_max_length(self):
        raise NotImplementedError
