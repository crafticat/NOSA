import time
import torch
import gc
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.cuda import nvtx
from nosi import NOSALlama as Llama
from nosi import transfer_trace as _tt   # retroinfer-eval fork: off unless NOSI_TRANSFER_TRACE

# retroinfer-eval fork (2026-09-02): the two hard-coded constants and the two
# network paths come from the environment so that every Table 5 cell is THIS
# script; defaults are the upstream values. NOSI_PG19_PARQUET points
# load_dataset at the pinned local parquet (same rows, same order).
_pq = os.environ.get("NOSI_PG19_PARQUET")
dataset = (load_dataset("parquet", data_files=_pq)['train']['text'] if _pq
           else load_dataset("emozilla/pg19")['test']['text'])

path = os.environ.get("NOSI_MODEL_PATH", "openbmb/NOSA-8B")
tokenizer = AutoTokenizer.from_pretrained(path)
model = Llama(
    model_name=path,
    device="cuda",
    offload=os.environ.get("NOSI_BENCH_OFFLOAD", "1") == "1",
    # NOSI_BENCH_MAX_LENGTH (fork, 2026-09-07): only sizes the RoPE cos/sin cache (nosa_llama.py `arange(max_length + 1024)`);
    # default = upstream default, so cells inside 128K are unchanged
    max_length=int(os.environ.get("NOSI_BENCH_MAX_LENGTH", 128 * 1024)),
)

B = int(os.environ.get("NOSI_BENCH_B", 16))
L = int(os.environ.get("NOSI_BENCH_L", 16 * 1024))
max_new_tokens = int(os.environ.get("NOSI_BENCH_NEW_TOKENS", 4))
test_n = int(os.environ.get("NOSI_BENCH_TEST_N", 4))

# NOSI_BENCH_TIMED_FROM (fork, 2026-09-09): which decode steps steps.csv marks
# `timed=1`. test_time calls batch_generate_benchmark with max_new_tokens+2,
# which loops `range(n - 1)`, i.e. max_new_tokens+1 decode steps indexed
# 0..max_new_tokens, of which 0 is the warm-up. UNSET reproduces the shipped
# default (1, 2, 3, 4) exactly, so every 4-token cell on record is unchanged.
# Set to k: steps k..max_new_tokens are timed. That is what a long decode needs
# -- a victim pool fills at ~3.6 blocks per stream-step, so scoring only steps
# 1-4 measures a cold pool whatever its size, and the offline LRU replay this is
# compared against scored 63 steps after one warm step
# (scripts/nosi_cache_sweep.lru_misses, warm_steps=1).
_n_decode_steps = max_new_tokens + 1
_timed_from = os.environ.get("NOSI_BENCH_TIMED_FROM")
_timed_steps = (tuple(range(int(_timed_from), _n_decode_steps)) if _timed_from
                else (1, 2, 3, 4))
assert _timed_steps, f"NOSI_BENCH_TIMED_FROM={_timed_from} leaves no timed step of {_n_decode_steps}"

print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")
_trace = _tt.install(model.num_layers, max_steps=max_new_tokens + 2)
_trace_out = os.environ.get("NOSI_TRACE_OUT", "nosi_trace")
_meta = dict(mode=_tt.MODE, context=L, batch=B, offload=int(model.offload), max_new_tokens=max_new_tokens)
print(f"[Trace] mode={_tt.MODE} out={_trace_out if _trace is not None else None} "
      f"decode_steps={_n_decode_steps} timed={_timed_steps[0]}..{_timed_steps[-1]} pool={os.environ.get('NOSI_POOL_BLOCKS', '0')}")


# NOSI_PROFILE=1 (retroinfer-eval stage 2, 2026-09-19): kernel-level breakdown of
# steady-state decode steps. schedule: skip the first NOSI_PROFILE_WAIT steps (the
# warm-up step and the pool-fill transient), 2 warm-up steps, then NOSI_PROFILE_ACTIVE
# profiled steps. Writes <NOSI_TRACE_OUT>/profile_doc<i>.txt (key_averages by CUDA
# time), profile_doc<i>_launches.json (kernel launch count, total CUDA kernel time,
# wall of the active window) and a Chrome trace. Profiling adds overhead: a
# profiled document's step times are NOT serving numbers and are excluded from
# steps.csv by the trace's own timed window (set NOSI_BENCH_TIMED_FROM past it) or
# by running the profile on its own document.
_profile = os.environ.get("NOSI_PROFILE", "0") == "1"
_prof_wait = int(os.environ.get("NOSI_PROFILE_WAIT", "20"))
_prof_active = int(os.environ.get("NOSI_PROFILE_ACTIVE", "6"))
_prof_doc = [0]


def _profile_ready(prof):
    import json as _json
    out = _trace_out if _trace is not None else "."
    os.makedirs(out, exist_ok=True)
    tag = f"profile_doc{_prof_doc[0]}"
    ka = prof.key_averages()
    with open(os.path.join(out, tag + ".txt"), "w") as f:
        f.write(ka.table(sort_by="cuda_time_total", row_limit=60))
    evs = [e for e in prof.events() if getattr(e, "device_type", None) is not None and str(e.device_type).endswith("CUDA")]
    kernel_us = sum(e.time_range.elapsed_us() for e in evs)
    launches = len(evs)
    t0 = min((e.time_range.start for e in evs), default=0); t1 = max((e.time_range.end for e in evs), default=0)
    with open(os.path.join(out, tag + "_launches.json"), "w") as f:
        _json.dump(dict(active_steps=_prof_active, kernel_launches=launches, kernel_launches_per_step=launches / max(1, _prof_active),
                        cuda_kernel_ms_per_step=kernel_us / 1e3 / max(1, _prof_active), window_wall_ms_per_step=(t1 - t0) / 1e3 / max(1, _prof_active)), f, indent=1)
    prof.export_chrome_trace(os.path.join(out, tag + ".json.gz"))
    print(f"[profile] doc={_prof_doc[0]} launches/step={launches / max(1, _prof_active):.0f} cuda_ms/step={kernel_us / 1e3 / max(1, _prof_active):.2f} window_ms/step={(t1 - t0) / 1e3 / max(1, _prof_active):.2f}")


def test_time(input_ids):
    if _profile:
        import torch.profiler as _tp
        import nosi.nosa_llama as _nl
        with _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
                         schedule=_tp.schedule(wait=_prof_wait, warmup=2, active=_prof_active, repeat=1),
                         on_trace_ready=_profile_ready) as prof:
            _nl.PROFILER = prof
            try:
                gen_ids, thru = model.batch_generate_benchmark(input_ids, max_new_tokens=max_new_tokens+2)
            finally:
                _nl.PROFILER = None
        _prof_doc[0] += 1
        return thru
    gen_ids, thru = model.batch_generate_benchmark(input_ids, max_new_tokens=max_new_tokens+2)
    return thru


total_t = 0
add_time = 0
first_time = True
# NOSI_BENCH_CONCAT=1 (fork, 2026-09-07): beyond PG-19's longest books, concatenate CONSECUTIVE books until
# L tokens; each concatenated document starts after the last book the previous one used, so documents never
# overlap. Off by default: the upstream loop below is byte-for-byte the shipped one when it is off.
_concat = os.environ.get("NOSI_BENCH_CONCAT", "0") == "1"
_next_i = 0
for i in range(len(dataset)):
    if i < _next_i:
        continue

    text = dataset[i]
    j = i
    if _concat:
        _ids = tokenizer(text, return_tensors="pt").input_ids
        while _ids.shape[1] < L and j + 1 < len(dataset):
            j += 1
            text = text + "\n\n" + dataset[j]
            _ids = tokenizer(text, return_tensors="pt").input_ids
        _next_i = j + 1
        if _ids.shape[1] >= L:
            print(f"[concat] books {i}..{j} -> {_ids.shape[1]} tokens for L={L}")

    input_ids = tokenizer(text, return_tensors="pt").to("cuda").input_ids
    if input_ids.shape[1] < L:
        continue
    input_ids = input_ids[:, :L].repeat(B, 1)
    
    _t0 = time.time()
    t = test_time(input_ids)
    print(f"thru: {t} tok/s")
    print(f"[doc] idx={i} n_tokens={input_ids.shape[1]} batch={input_ids.shape[0]} wall={time.time() - _t0:.1f}s "
          f"gpu_peak_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
    # host-memory evidence per document (fork addition, 2026-09-03): the 64K x 64 and 96K x 64 cells were
    # killed by the host cgroup even at 900 GB; this line records RSS/locked pages per document so growth
    # vs. steady state is measured, not inferred. NOSI_BENCH_HOST_EMPTY_CACHE=1 releases PyTorch's cached
    # pinned-host blocks between documents (torch._C._host_emptyCache); it does not touch per-document timing.
    try:
        _st = {}
        for _ln in open("/proc/self/status"):
            if _ln.startswith(("VmRSS", "VmLck", "VmHWM")):
                _k, _v = _ln.split(":", 1); _st[_k] = int(_v.split()[0]) / 1048576
        print(f"[mem] doc={i} rss_gb={_st.get('VmRSS', 0):.1f} hwm_gb={_st.get('VmHWM', 0):.1f} locked_gb={_st.get('VmLck', 0):.1f}")
        if os.environ.get("NOSI_BENCH_HOST_EMPTY_CACHE", "0") == "1" and hasattr(torch._C, "_host_emptyCache"):
            # 2171641 showed RSS 273 -> 529 GB per document with the empty-cache alone: the previous document's
            # InfLLMv2Cache (32 layers of pinned _k_cpu/_v_cpu) is cyclic garbage that CPython's refcount does not
            # free and the generational GC has not yet visited, so the allocator had nothing to release. Collect first.
            import gc; _n = gc.collect()
            torch._C._host_emptyCache(); torch.cuda.empty_cache()
            _st2 = {}
            for _ln in open("/proc/self/status"):
                if _ln.startswith(("VmRSS", "VmHWM")):
                    _k, _v = _ln.split(":", 1); _st2[_k] = int(_v.split()[0]) / 1048576
            print(f"[mem] doc={i} host_empty_cache=ran gc_collected={_n} rss_after_gb={_st2.get('VmRSS', 0):.1f}")
    except Exception as _e:  # never let the instrument kill the cell
        print(f"[mem] doc={i} instrument_error={type(_e).__name__}: {_e}")
    if _trace is not None:
        _summ = _tt.dump(_trace_out, doc_idx=i, meta=_meta, timed_steps=_timed_steps)
        print(f"[trace] doc={i} " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in _summ.items()))
    if first_time:
        first_time = False
    else:
        total_t += t
        add_time += 1
    if add_time == test_n:
        break

n_tokens = max_new_tokens * B  
tok_per_s = total_t / add_time
decode_time = n_tokens / tok_per_s
print(f"\n[Timing] Decode {n_tokens} tokens in {decode_time:.3f} s")
print(f"[Speed] {tok_per_s:.2f} tokens/s (avg per batch)")