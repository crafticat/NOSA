"""Phase P2 driver (retroinfer-eval fork, 2026-09-02): trace NOSA's block
selections per (step, layer, head) under TEACHER-FORCED decoding so that two
models (NOSA-8B, NOSA-1B) see identical tokens at every step.

Same prefill + decode calls as `Llama.batch_generate_benchmark` (batch_prefill,
then decode_inference per step, warm-up on the first), but the next token
comes from the document's own continuation, not from argmax -- otherwise the
two models' generations diverge after step 1 and "same tokens" is false.
Each batch row is a DIFFERENT PG-19 document (the benchmark repeats one),
truncated to L; the continuation is tokens [L, L + steps). Documents are the
first B in dataset order with >= L + steps tokens.

Outputs (NOSI_TRACE_OUT): scores_<tag>_doc<k>.pt per batch (mode "scores":
masks, maps, pooled score buffers per (step, layer)), steps.csv/layers.csv,
and run.json with per-step wall (ms) for the cost question (P2c).

Environment: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_BENCH_L, NOSI_BENCH_B,
NOSI_P2_STEPS (decode steps incl. the warm-up), NOSI_P2_DOC_OFFSET (skip the
first k qualifying documents), NOSI_TRANSFER_TRACE=scores, NOSI_TRACE_OUT.
"""
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from nosi import NOSALlama as Llama
from nosi import transfer_trace as _tt

path = os.environ["NOSI_MODEL_PATH"]
pq = os.environ["NOSI_PG19_PARQUET"]
L = int(os.environ.get("NOSI_BENCH_L", 16 * 1024))
B = int(os.environ.get("NOSI_BENCH_B", 1))
STEPS = int(os.environ.get("NOSI_P2_STEPS", 66))
OFFSET = int(os.environ.get("NOSI_P2_DOC_OFFSET", 0))
OUT = os.environ.get("NOSI_TRACE_OUT", "nosi_p2_trace")
os.makedirs(OUT, exist_ok=True)

dataset = load_dataset("parquet", data_files=pq)["train"]["text"]
tokenizer = AutoTokenizer.from_pretrained(path)
model = Llama(model_name=path, device="cuda", offload=os.environ.get("NOSI_BENCH_OFFLOAD", "1") == "1")
trace = _tt.install(model.num_layers, max_steps=STEPS)
assert trace is not None and trace.scores, "run with NOSI_TRANSFER_TRACE=scores"
print(f"[Setup] model={path} layers={model.num_layers} L={L} B={B} steps={STEPS} offset={OFFSET} mode={_tt.MODE}")

# pick B distinct documents with >= L + STEPS tokens, in dataset order
rows, ids = [], []
skipped = 0
for i in range(len(dataset)):
    t = tokenizer(dataset[i], return_tensors="pt").input_ids
    if t.shape[1] < L + STEPS:
        continue
    if skipped < OFFSET:
        skipped += 1
        continue
    rows.append(i)
    ids.append(t[0, :L + STEPS])
    if len(rows) == B:
        break
assert len(rows) == B, f"only {len(rows)} documents with >= {L + STEPS} tokens after offset {OFFSET}"
ids = torch.stack(ids).to("cuda")                    # (B, L + STEPS)
prompt, forced = ids[:, :L], ids[:, L:]
print(f"[docs] {rows}  prompt {tuple(prompt.shape)}  forced {tuple(forced.shape)}")

meta = dict(mode=_tt.MODE, context=L, batch=B, offload=int(model.offload), model=os.path.basename(path.rstrip("/")),
            steps=STEPS, docs=rows, teacher_forced=1)

@torch.inference_mode()
def run():
    from nosi.cache_engine import InfLLMv2Cache
    from nosi.cache_engine_gpu import InfLLMv2Cache as NoOffload
    trace.new_document()
    cache_cls = InfLLMv2Cache if model.offload else NoOffload
    cache_engine = cache_cls(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    t0 = time.time()
    logits, position_ids = model.batch_prefill(prompt, cache_engine)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    step_ms, argmax_match = [], []
    for it in range(STEPS):
        next_ids = forced[:, it:it + 1]                 # teacher forcing: identical tokens for every model
        torch.cuda.synchronize()
        t1 = time.time()
        logits = model.decode_inference(next_ids, cu, position_ids, cache_engine, warmup=(it == 0))
        torch.cuda.synchronize()
        step_ms.append(1e3 * (time.time() - t1))
        if it + 1 < STEPS:
            argmax_match.append(int((logits[:, -1, :].argmax(-1) == forced[:, it + 1]).sum()))
        position_ids = position_ids + 1
    return prefill_s, step_ms, argmax_match

prefill_s, step_ms, argmax_match = run()
summ = _tt.dump(OUT, doc_idx=rows[0], meta=meta, timed_steps=tuple(range(1, STEPS)))
timed = step_ms[1:]
run_json = dict(meta=meta, prefill_s=prefill_s, step_ms=step_ms,
                step_ms_mean=sum(timed) / len(timed), step_ms_p50=sorted(timed)[len(timed) // 2],
                argmax_matches_forced=argmax_match, gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9, trace_summary=summ)
with open(os.path.join(OUT, "run.json"), "w") as f:
    json.dump(run_json, f, indent=1)
print(f"[P2] prefill {prefill_s:.1f}s  step mean {run_json['step_ms_mean']:.2f} ms p50 {run_json['step_ms_p50']:.2f} ms over {len(timed)} steps  "
      f"argmax==forced {sum(argmax_match)}/{B * len(argmax_match)}  peak {run_json['gpu_peak_gb']:.2f} GB")
print("[trace] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in summ.items()))
