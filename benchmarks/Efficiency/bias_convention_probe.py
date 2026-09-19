"""Does NOSI's decode serve NOSA's tokens?  A one-prompt, teacher-forced probe.

Three runs of this script (separate processes, because NOSI reads its knobs at
import) and one comparison:

  NOSI_PROBE_MODE=hf        the checkpoint's own forward (the reference: the
                            training-time sparse attention with V * exp(cis))
                            over prompt + N forced tokens; saves the logits at
                            positions L-1 .. L+N-1.
  NOSI_PROBE_MODE=nosi      NOSI prefill of the prompt, then N teacher-forced
                            decode steps; saves the prefill logits and each
                            step's logits.  Run once with NOSI_KV_BIAS_SCALE
                            unset (shipped) and once with sqrt(128).
  NOSI_PROBE_MODE=compare   per-step argmax agreement, max |dlogit|, KL against
                            the reference; the prefill row is the control (both
                            use the varlen kernel + exp(cis) trick).

Alignment: reference position L-1 predicts forced[0] and equals NOSI's prefill
logits; reference position L+it predicts forced[it+1] and equals NOSI decode
step it (input forced[it]).

Env: NOSI_MODEL_PATH, NOSI_PG19_PARQUET, NOSI_PROBE_OUT, NOSI_PROBE_L (default
4032: off the 128 boundary, see the score_buf aliasing note in the ledger),
NOSI_PROBE_N (default 16, < 64 so no tail rollover), NOSI_PROBE_DOCS (default 2).
"""
import json
import os
import sys
import time

import torch

MODE = os.environ.get("NOSI_PROBE_MODE", "compare")
OUT = os.environ.get("NOSI_PROBE_OUT", "nosi_bias_probe")
L = int(os.environ.get("NOSI_PROBE_L", "4032"))
N = int(os.environ.get("NOSI_PROBE_N", "16"))
NDOCS = int(os.environ.get("NOSI_PROBE_DOCS", "2"))
TAG = os.environ.get("NOSI_PROBE_TAG", MODE)
os.makedirs(OUT, exist_ok=True)


def load_docs(path):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    dataset = load_dataset("parquet", data_files=os.environ["NOSI_PG19_PARQUET"])["train"]["text"]
    ids, rows = [], []
    for i in range(len(dataset)):
        t = tokenizer(dataset[i], return_tensors="pt").input_ids
        if t.shape[1] < L + N + 1:
            continue
        rows.append(i)
        ids.append(t[0, :L + N + 1])
        if len(rows) == NDOCS:
            break
    assert len(rows) == NDOCS, f"only {len(rows)} documents with >= {L + N + 1} tokens"
    return torch.stack(ids), rows


@torch.inference_mode()
def run_hf(path, ids):
    """Reference logits at positions L-1 .. L+N-1 (N+1 rows) from one forward
    over the first L+N tokens: the checkpoint's own modeling, remote code."""
    try:
        from transformers import AutoModelForCausalLM
        # the sparse CIS path lives in LlamaSdpaAttention (modeling_llama_long_infllmv2.py:855,
        # :958 _sparse_attention_forward, :1095 sparse_forward with V * exp(cis) at :1135-1175)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        how = "AutoModelForCausalLM(trust_remote_code, sdpa)"
    except Exception as e:  # the fork's own HF benchmark path
        print(f"[hf] AutoModel failed ({type(e).__name__}: {e}); falling back to proxy_modeling_nosa", flush=True)
        from proxy_modeling_nosa import SparseLlamaForCausalLM
        model = SparseLlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
        how = "proxy_modeling_nosa.SparseLlamaForCausalLM"
    model.eval()
    attn = model.model.layers[0].self_attn
    assert type(attn).__name__ == "LlamaSdpaAttention" and hasattr(attn, "_sparse_attention_forward"), (
        f"reference attention is {type(attn).__name__}: not the sparse CIS path")
    print(f"[hf] loaded via {how}; attn_implementation={getattr(model.config, '_attn_implementation', '?')}; attention class {type(attn).__name__}", flush=True)
    x = ids[:, :L + N].to("cuda")
    t0 = time.time()
    out = model(input_ids=x, use_cache=False, num_logits_to_keep=N + 1)
    torch.cuda.synchronize()
    logits = out.logits.float().cpu()                      # (B, N+1, V): positions L-1 .. L+N-1
    assert logits.shape[1] == N + 1, logits.shape
    print(f"[hf] forward over {L + N} tokens: {time.time() - t0:.1f}s; logits {tuple(logits.shape)}", flush=True)
    torch.save(dict(logits=logits, L=L, N=N, how=how), os.path.join(OUT, "hf_logits.pt"))


@torch.inference_mode()
def run_nosi(path, ids):
    from nosi import NOSALlama as Llama
    from nosi import cache_engine as _ce
    from nosi.cache_engine import InfLLMv2Cache
    scale = _ce.KV_BIAS_SCALE
    print(f"[nosi] KV_BIAS_SCALE={scale} ATTN_SPLITS={os.environ.get('NOSI_ATTN_SPLITS', '0')}", flush=True)
    model = Llama(model_name=path, device="cuda", offload=True)
    B = ids.shape[0]
    x = ids.to("cuda")
    prompt, forced = x[:, :L], x[:, L:L + N]
    cache_engine = InfLLMv2Cache(config=model.config, num_hidden_layers=model.config.num_hidden_layers, has_kv_bias=True)
    logits, position_ids = model.batch_prefill(prompt, cache_engine)
    rows = [logits[:, -1, :].float().cpu()]                 # predicts forced[0]  <-> reference position L-1
    position_ids = position_ids[:, -1:] + 1
    cu = torch.arange(0, B + 1, dtype=torch.int, device="cuda")
    for it in range(N):
        lg = model.decode_inference(forced[:, it:it + 1], cu, position_ids, cache_engine, warmup=(it == 0))
        rows.append(lg[:, -1, :].float().cpu())            # predicts forced[it+1] <-> reference position L+it
        position_ids = position_ids + 1
    torch.cuda.synchronize()
    out = torch.stack(rows, dim=1)                          # (B, N+1, V)
    torch.save(dict(logits=out, L=L, N=N, kv_bias_scale=scale), os.path.join(OUT, f"nosi_{TAG}_logits.pt"))
    print(f"[nosi] saved {tuple(out.shape)} as nosi_{TAG}_logits.pt", flush=True)


def compare():
    ref = torch.load(os.path.join(OUT, "hf_logits.pt"))["logits"].double()
    arms = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("nosi_") and f.endswith("_logits.pt"):
            d = torch.load(os.path.join(OUT, f))
            arms[f[len("nosi_"):-len("_logits.pt")]] = (d["logits"].double(), d.get("kv_bias_scale"))
    assert arms, "no NOSI arms found"
    B, R, V = ref.shape
    lines = ["| arm | kv_bias scale | row | argmax agree (of %d) | max abs dlogit | mean KL(ref || nosi) |" % B, "|---|---|---|---|---|---|"]
    summary = {}
    for name, (lg, scale) in arms.items():
        assert lg.shape == ref.shape, (name, lg.shape, ref.shape)
        agree_dec, kl_dec, mx_dec = [], [], []
        for r in range(R):
            a = ref[:, r, :]; b = lg[:, r, :]
            agree = int((a.argmax(-1) == b.argmax(-1)).sum())
            mx = float((a - b).abs().max())
            pa = torch.log_softmax(a, -1); pb = torch.log_softmax(b, -1)
            kl = float((pa.exp() * (pa - pb)).sum(-1).mean())
            label = "prefill (control)" if r == 0 else f"decode step {r - 1}"
            lines.append(f"| {name} | {scale} | {label} | {agree} | {mx:.3f} | {kl:.4f} |")
            if r > 0:
                agree_dec.append(agree); kl_dec.append(kl); mx_dec.append(mx)
        summary[name] = dict(kv_bias_scale=scale, prefill_argmax_agree=int((ref[:, 0].argmax(-1) == lg[:, 0].argmax(-1)).sum()),
                             decode_argmax_agree_frac=sum(agree_dec) / (B * len(agree_dec)), decode_max_abs_dlogit=max(mx_dec), decode_mean_kl=sum(kl_dec) / len(kl_dec))
    lines.append("\n| arm | prefill agree | decode argmax agreement | decode max abs dlogit | decode mean KL |\n|---|---|---|---|---|")
    for name, s in summary.items():
        lines.append(f"| {name} (scale {s['kv_bias_scale']}) | {s['prefill_argmax_agree']}/{B} | {100 * s['decode_argmax_agree_frac']:.1f}% | {s['decode_max_abs_dlogit']:.3f} | {s['decode_mean_kl']:.4f} |")
    if len(arms) == 2:
        (n1, (l1, _)), (n2, (l2, _)) = arms.items()
        ag = float((l1[:, 1:].argmax(-1) == l2[:, 1:].argmax(-1)).double().mean())
        lines.append(f"\nNOSI arm vs NOSI arm ({n1} vs {n2}): decode argmax agreement {100 * ag:.1f}%, max abs dlogit {float((l1[:, 1:] - l2[:, 1:]).abs().max()):.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "compare.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT, "compare.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    if MODE == "compare":
        compare(); sys.exit(0)
    path = os.environ["NOSI_MODEL_PATH"]
    ids, rows = load_docs(path)
    print(f"[docs] {rows}  L={L} N={N}", flush=True)
    if MODE == "hf":
        run_hf(path, ids)
    elif MODE == "nosi":
        run_nosi(path, ids)
    else:
        raise SystemExit(f"unknown NOSI_PROBE_MODE {MODE}")
