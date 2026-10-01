"""Post-run gate for the treecap group-mean capture (retroinfer-eval fork, 2026-10-01). CPU only.

    python treecap_check.py NEW_DIR REF_DIR L EXPECT_SCORES_SHA256 [EXPECT_LAYERS]

Trajectory: the run's own selection trace (scores_/selection_/logits_ .pt of the unchanged 64 decode calls)
must equal the archive's (REF_DIR, job 2173290) byte for byte; if the bytes differ, every tensor is
compared bitwise and the result is reported, but the gate fails. Negative control: one flipped mask
element must be detected by the same tensor comparison.
Export: shape [n_layers, B, H, L//64, 128] fp32, ids, finite values, the npz sha in its JSON, and the
in-job checks (spot and compressed-key pair) all ok.
Writes NEW_DIR/CHECK.json; exit 0 only if every gate passes.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def diff_tensors(a, b, path=""):
    """Bitwise comparison of nested dict/list/tensor objects. Returns the list of differing paths."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return [path + ":keys"]
        return [d for k in sorted(a) for d in diff_tensors(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + ":len"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff_tensors(x, y, f"{path}[{i}]")]
    if torch.is_tensor(a):
        same = torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and \
            torch.equal(a.reshape(-1).view(torch.uint8) if a.numel() else a, b.reshape(-1).view(torch.uint8) if b.numel() else b)
        return [] if same else [path]
    return [] if a == b else [path]


def first_int_tensor(o):
    if torch.is_tensor(o):
        return o if not o.is_floating_point() and o.dtype != torch.bool and o.numel() else None
    for v in (o.values() if isinstance(o, dict) else o if isinstance(o, (list, tuple)) else []):
        t = first_int_tensor(v)
        if t is not None:
            return t
    return None


def main(new, ref, L, expect_sha, expect_layers="0,4,8,12,16,20,24,28"):
    L = int(L)
    tag = f"ctx{L}_b4_off1_doc0"
    out = dict(new=new, ref=ref, L=L, expect_scores_sha256=expect_sha, gates={})
    g = out["gates"]

    files = {}
    for kind in ("scores", "selection", "logits"):
        a, b = os.path.join(new, f"{kind}_{tag}.pt"), os.path.join(ref, f"{kind}_{tag}.pt")
        files[kind] = dict(new=sha(a), ref=sha(b))
        files[kind]["byte_identical"] = files[kind]["new"] == files[kind]["ref"]
    out["files"] = files
    g["scores_sha_is_archive"] = files["scores"]["new"] == expect_sha
    g["trajectory_bytes"] = all(f["byte_identical"] for f in files.values())
    for name in ("logits.csv", "layers.csv", "steps.csv"):
        a, b = os.path.join(new, name), os.path.join(ref, name)
        out.setdefault("info_csv_identical", {})[name] = os.path.exists(a) and os.path.exists(b) and sha(a) == sha(b)

    new_sc = torch.load(os.path.join(new, f"scores_{tag}.pt"), map_location="cpu")
    ref_sc = torch.load(os.path.join(ref, f"scores_{tag}.pt"), map_location="cpu")
    d = diff_tensors(new_sc, ref_sc)
    out["scores_tensor_diffs"] = d
    g["trajectory_tensors"] = not d
    t = first_int_tensor(new_sc)
    t.view(-1)[0] += 1                                        # negative control: one flipped element
    g["negative_control_detected"] = bool(diff_tensors(new_sc, ref_sc))
    t.view(-1)[0] -= 1

    rj_new = json.load(open(os.path.join(new, "run.json")))
    rj_ref = json.load(open(os.path.join(ref, "run.json")))
    out["argmax_matches_forced"] = dict(new=sum(rj_new["argmax_matches_forced"]), ref=sum(rj_ref["argmax_matches_forced"]))
    g["argmax_same"] = rj_new["argmax_matches_forced"] == rj_ref["argmax_matches_forced"]

    npz_p, js_p = os.path.join(new, f"treecap_groupmean_L{L}.npz"), os.path.join(new, f"treecap_groupmean_L{L}.json")
    z, meta = np.load(npz_p), json.load(open(js_p))
    G = z["group_mean_key"]
    layers = [int(x) for x in expect_layers.split(",")]
    n = L // 64
    want = (len(layers), 4, 2, n, 128)
    out["export"] = dict(shape=list(G.shape), dtype=str(G.dtype), nbytes=int(G.nbytes), npz_bytes=os.path.getsize(npz_p),
                         npz_sha256=sha(npz_p), in_job_checks=meta["checks"])
    g["export_shape"] = G.shape == want and G.dtype == np.float32 and G.nbytes == int(np.prod(want)) * 4
    g["export_ids"] = (z["layer_ids"].tolist() == layers and z["request_ids"].tolist() == [0, 1, 2, 3]
                       and z["kv_head_ids"].tolist() == [0, 1] and z["group_ids"].tolist() == list(range(n))
                       and z["group_start_token"].tolist() == [64 * i for i in range(n)]
                       and z["doc_index"].tolist() == rj_ref["meta"]["docs"])
    g["export_finite"] = bool(np.isfinite(G).all())
    g["export_sha_matches_json"] = meta["npz_sha256"] == out["export"]["npz_sha256"]
    g["export_in_job_checks"] = bool(meta["checks"]["ok"])

    out["ok"] = all(g.values())
    with open(os.path.join(new, "CHECK.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[check] " + " ".join(f"{k}={v}" for k, v in g.items()) + f"  => ok={out['ok']}")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
