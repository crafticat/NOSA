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
)

B = int(os.environ.get("NOSI_BENCH_B", 16))
L = int(os.environ.get("NOSI_BENCH_L", 16 * 1024))
max_new_tokens = int(os.environ.get("NOSI_BENCH_NEW_TOKENS", 4))
test_n = int(os.environ.get("NOSI_BENCH_TEST_N", 4))

print(f"[Setup] batch={B}, input_len={L}, max_new_tokens={max_new_tokens}")
_trace = _tt.install(model.num_layers, max_steps=max_new_tokens + 2)
_trace_out = os.environ.get("NOSI_TRACE_OUT", "nosi_trace")
_meta = dict(mode=_tt.MODE, context=L, batch=B, offload=int(model.offload), max_new_tokens=max_new_tokens)
print(f"[Trace] mode={_tt.MODE} out={_trace_out if _trace is not None else None}")


def test_time(input_ids):
    gen_ids, thru = model.batch_generate_benchmark(input_ids, max_new_tokens=max_new_tokens+2)
    return thru


total_t = 0
add_time = 0
first_time = True
for i in range(len(dataset)):

    text = dataset[i]

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
        _summ = _tt.dump(_trace_out, doc_idx=i, meta=_meta)
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