"""Known-answer checks of the acceptance pilot's MASKED DRAFT PATH (request relayed by Codex, 2026-09-23; ledger
'MASK CHECK REGISTERED'). It runs the exact archived code (fork commit 6e6f80a: acceptance_pilot.py, avail_policy.py,
nosa_llama.py, cache_engine.py are imported UNCHANGED; this file is the only addition) with the model and settings of
jobs 2174689 / 2175453 / 2175629 (NOSA-8B, PG-19, L = 16128, warm 8, mech mask, T = 1.0, top-p 1.0, top-k 0).

At every (document, probe site, availability point) the state is snapshotted once and every arm below starts from the
restored state (engine CacheSnapshot incl. the GPU window; the policy's virtual caches are cloned by begin_rollout),
so every arm runs the SAME decode call with the SAME tensor shapes (B = 1, the same 64-slot window, the same explicit
kernel configuration). Only the draft position j = 1 is used, so no arm reads another arm's side effects.

  T      target step (role TARGET, full availability)                         -> reference logits
  ALL    draft step whose deny mask is forced ALL-FALSE but routed through the scratch-bias path of mask_bias
         (the path real denials take)                                      CHECK 1: torch.equal(ALL, T)
  M      the actual masked draft step (the archived measurement)
  P      M plus, inside the layer, AFTER mask_bias and BEFORE attention: the K rows of every denied slot overwritten
         with +/-64 (bounded so the scores stay << the 3e4 mask margin) and the V rows with +/-1e4 (large finite), and
         every tail-slot row at or beyond the call's cache_seqlens overwritten likewise
                                                                          CHECK 2: torch.equal(P, M)
  C      M plus the same garbage written into ONE visible, non-denied, non-tail slot per (layer, request) (head 0)
                                                                          positive control: C must differ from M
  R      matched-count RANDOM masking: per (layer, head, request) exactly as many slots as M denies, drawn uniformly
         (seeded) from the non-tail slots                                 CHECK 3: TV(T, M) vs TV(T, R)
  T2     the target step again after all arms                             hygiene: torch.equal(T2, T)

Also recorded per round: denied slots, how many denied slots are slot 0 / slot 62 (adjacent to the tail), how many
denied blocks are NEWLY SELECTED this step (the engine's load mask names them), TV / overlap / expected acceptance
probability at j = 1 for M and R. Exit code = number of failed checks (1, 2, positive control, hygiene).
Env: MC_DOCS (8), MC_PROBES (8), plus the pilot's own env (NOSI_ACC_POINTS via APPTAINERENV_, NOSI_AVAIL,
NOSI_BENCH_L, NOSI_ACC_WARM, NOSI_ACC_OUT, NOSI_REPO, ...).
"""
import json
import os
import sys
import time

os.environ.setdefault("NOSI_ACC_MODE", "rollout")
os.environ.setdefault("NOSI_ACC_K", "1")
if "MC_DOCS" in os.environ:
    os.environ["NOSI_ACC_DOCS"] = os.environ["MC_DOCS"]
if "MC_PROBES" in os.environ:
    os.environ["NOSI_ACC_PROBES"] = os.environ["MC_PROBES"]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import acceptance_pilot as AP  # noqa: E402   (the archived pilot, unchanged)

K_POISON, V_POISON = 64.0, 1.0e4


def main():
    am = AP.am
    probes = list(range(AP.WARM + 1, AP.WARM + 1 + AP.NPROBES))
    AP.budget_or_die(1 + AP.WARM + AP.NPROBES, 1)
    pilot = AP.Pilot()
    pol = pilot.policy
    ap_mod = pilot.ap
    need = 1 + AP.WARM + AP.NPROBES + 1 + 2
    ids, doc_rows = AP.load_documents(pilot.model.tokenizer, need)
    ids = ids.to("cuda")
    orig_mask_bias = pol.mask_bias
    arm = {"name": None, "seed": 0, "doc": 0, "probe": 0}
    stat = {}

    def engine_of(layer):
        return pilot.cache.layers[layer].cache_engine

    def poison_rows(eng, b, h, r0, r1, gen):
        n = r1 - r0
        if n <= 0:
            return
        ks = torch.randint(0, 2, (n, eng._k_gpu.shape[-1]), generator=gen, device="cpu").to(eng._k_gpu.device) * 2 - 1
        vs = torch.randint(0, 2, (n, eng._v_gpu.shape[-1]), generator=gen, device="cpu").to(eng._v_gpu.device) * 2 - 1
        eng._k_gpu[b, r0:r1, h, :] = (ks * K_POISON).to(eng._k_gpu.dtype)
        eng._v_gpu[b, r0:r1, h, :] = (vs * V_POISON).to(eng._v_gpu.dtype)

    def hooked_mask_bias(kv_bias):
        name = arm["name"]
        layer = pol.layer
        if name == "ALL" and pol.role == ap_mod.ROLE_DRAFT:
            # force the scratch path with NOTHING denied (the tail is never denied)
            H, B, M = engine_of(layer)._block_map.shape
            pol._deny_dev = torch.zeros((H, B, M), dtype=torch.bool, device=kv_bias.device)
            return orig_mask_bias(kv_bias)
        if name == "R" and pol.role == ap_mod.ROLE_DRAFT and pol._deny_dev is not None:
            eng = engine_of(layer)
            actual = pol._deny_dev
            H, B, M = actual.shape
            gen = torch.Generator().manual_seed(arm["seed"] * 1000003 + layer)
            rnd = torch.zeros_like(actual, device="cpu")
            valid = (eng._block_map.cpu() >= 0)
            for h in range(H):
                for b in range(B):
                    n = int(actual[h, b].sum())
                    if n == 0:
                        continue
                    cand = [m for m in range(M - 1) if bool(valid[h, b, m])]         # non-tail slots holding a block
                    pick = torch.randperm(len(cand), generator=gen)[:n].tolist()
                    for i in pick:
                        rnd[h, b, cand[i]] = True
            pol._deny_dev = rnd.to(actual.device)
            return orig_mask_bias(kv_bias)
        out = orig_mask_bias(kv_bias)
        if name in ("M", "P", "C") and pol.role == ap_mod.ROLE_DRAFT:
            eng = engine_of(layer)
            H, B, M = eng._block_map.shape
            bs = eng.block_size
            deny = pol._deny_dev.cpu() if pol._deny_dev is not None else torch.zeros((H, B, M), dtype=torch.bool)
            lm = eng._load_mask.cpu()
            if name == "M":
                s = stat.setdefault((arm["doc"], arm["probe"], arm["pid"]), dict(denied=0, slot0=0, slot62=0, newly_selected=0))
                s["denied"] += int(deny.sum()); s["slot0"] += int(deny[..., 0].sum()); s["slot62"] += int(deny[..., M - 2].sum())
                s["newly_selected"] += int((deny & (lm >= 0)).sum())
            gen = torch.Generator().manual_seed(arm["seed"] * 7919 + layer)
            if name == "P":
                for h in range(H):
                    for b in range(B):
                        for m in range(M - 1):
                            if bool(deny[h, b, m]):
                                poison_rows(eng, b, h, m * bs, (m + 1) * bs, gen)
                        # invisible tail rows: at or beyond this call's cache_seqlens
                        vis = int(eng._cache_lens[b])
                        poison_rows(eng, b, h, vis, M * bs, gen)
            elif name == "C":
                for b in range(B):
                    h = 0
                    ok = [m for m in range(M - 1) if not bool(deny[h, b, m]) and int(eng._block_map[h, b, m]) >= 0]
                    if ok:
                        m = ok[len(ok) // 2]
                        poison_rows(eng, b, h, m * bs, (m + 1) * bs, gen)
        return out

    pol.mask_bias = hooked_mask_bias           # the model calls avail_policy.POLICY.mask_bias -> this instance attribute
    assert ap_mod.POLICY is pol

    rows, fails = [], dict(check1=0, check2=0, positive_control=0, hygiene=0)
    t_start = time.time()
    for d in range(AP.NDOCS):
        prompt, forced = ids[d:d + 1, :AP.L], ids[d:d + 1, AP.L:]
        pol.new_document(d)
        pilot.open_document(prompt)
        pilot.step(forced[:, 0:1], ap_mod.ROLE_WARM)
        trans = pilot.ss.transient_ids(pilot.model)
        for t in range(1, 1 + AP.WARM):
            pilot.step(forced[:, t:t + 1], ap_mod.ROLE_WARM)
        for probe in probes:
            first = forced[:, probe:probe + 1]
            pilot.snap.take()
            pos0 = pilot.pos.clone()

            def restore():
                pilot.snap.restore(); pilot.pos = pos0.clone()
                pilot.ss.assert_transients_intact(pilot.model, trans)

            arm.update(name="T", doc=d, probe=probe, pid=-1, seed=d * 1009 + probe)
            lg_T = pilot.step(first, ap_mod.ROLE_TARGET).clone()
            restore()
            p_T = pilot.dist(lg_T[0])
            for point in pilot.points:
                if point["d"] == 0:
                    continue
                res = {}
                for name in ("ALL", "M", "P", "C", "R"):
                    arm.update(name=name, pid=point["id"])
                    pol.set_site(probe)
                    pol.begin_rollout(point["id"])
                    lg = pilot.step(first, ap_mod.ROLE_DRAFT).clone()
                    pol.end_rollout()
                    restore()
                    res[name] = lg
                arm.update(name=None)
                ok1 = bool(torch.equal(res["ALL"], lg_T))
                ok2 = bool(torch.equal(res["P"], res["M"]))
                okc = not bool(torch.equal(res["C"], res["M"]))
                fails["check1"] += int(not ok1); fails["check2"] += int(not ok2); fails["positive_control"] += int(not okc)
                q_M, q_R = pilot.dist(res["M"][0]), pilot.dist(res["R"][0])
                ovM, tvM = am.overlap_and_tv(p_T, q_M)
                ovR, tvR = am.overlap_and_tv(p_T, q_R)
                s = stat.get((d, probe, point["id"]), dict(denied=0, slot0=0, slot62=0, newly_selected=0))
                row = dict(doc=d, doc_row=doc_rows[d], probe=probe, point=point["spec"], C=point["C"], d=("inf" if point["d"] == float("inf") else point["d"]),
                           check1_all_visible_scratch_equal_target=ok1, check2_poison_denied_equal_masked=ok2, positive_control_visible_poison_differs=okc,
                           max_abs_all_vs_target=float((res["ALL"] - lg_T).abs().max()), max_abs_poison_vs_masked=float((res["P"] - res["M"]).abs().max()),
                           max_abs_control_vs_masked=float((res["C"] - res["M"]).abs().max()),
                           tv_masked=float(tvM), tv_random_matched=float(tvR), overlap_masked=float(ovM), overlap_random_matched=float(ovR),
                           argmax_target=int(lg_T.argmax()), argmax_masked=int(res["M"].argmax()), argmax_random=int(res["R"].argmax()), **s)
                rows.append(row)
                print("[maskcheck] doc %d probe %d %-10s denied %4d (new %4d, slot0 %d, slot62 %d)  check1 %s  check2 %s  control %s  TV masked %.5f random %.5f"
                      % (d, probe, point["spec"], s["denied"], s["newly_selected"], s["slot0"], s["slot62"], ok1, ok2, okc, tvM, tvR), flush=True)
            arm.update(name="T")
            lg_T2 = pilot.step(first, ap_mod.ROLE_TARGET).clone()
            restore()
            okh = bool(torch.equal(lg_T2, lg_T))
            fails["hygiene"] += int(not okh)
            pilot.step(first, ap_mod.ROLE_WARM)        # advance the verified trajectory, as the pilot does
        print("[maskcheck] doc %d done (%.0fs)" % (d, time.time() - t_start), flush=True)
    by_point = {}
    for r in rows:
        a = by_point.setdefault(r["point"], dict(n=0, tv_m=[], tv_r=[], denied=0, new=0, c1=0, c2=0, pc=0))
        a["n"] += 1; a["tv_m"].append(r["tv_masked"]); a["tv_r"].append(r["tv_random_matched"]); a["denied"] += r["denied"]; a["new"] += r["newly_selected"]
        a["c1"] += int(r["check1_all_visible_scratch_equal_target"]); a["c2"] += int(r["check2_poison_denied_equal_masked"]); a["pc"] += int(r["positive_control_visible_poison_differs"])
    summary = {}
    for k, a in by_point.items():
        tvm, tvr = np.array(a["tv_m"]), np.array(a["tv_r"])
        summary[k] = dict(rounds=a["n"], check1_pass=a["c1"], check2_pass=a["c2"], positive_control_pass=a["pc"],
                          tv_masked_mean=float(tvm.mean()), tv_random_mean=float(tvr.mean()), tv_paired_diff_mean=float((tvm - tvr).mean()),
                          tv_masked_gt_random_rounds=int((tvm > tvr).sum()), denied_per_round=a["denied"] / a["n"],
                          newly_selected_share_of_denied=(a["new"] / a["denied"]) if a["denied"] else None)
    out = dict(rows=rows, summary=summary, fails=fails, probes=probes, docs=doc_rows, points=[p["spec"] for p in pilot.points],
               poison=dict(K=K_POISON, V=V_POISON), commit_note="fork 6e6f80a + benchmarks/Efficiency/mask_check.py only",
               seconds=time.time() - t_start)
    os.makedirs(AP.OUT, exist_ok=True)
    json.dump(out, open(os.path.join(AP.OUT, "mask_check.json"), "w"), indent=1)
    L = ["# Mask-path known-answer checks (acceptance pilot, fork 6e6f80a)", "",
         "| point | rounds | check 1: all-visible scratch == target | check 2: poisoned denied == masked | positive control differs | denied blocks per round | newly selected share of denied | TV masked | TV random matched | rounds TV masked > random |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in summary.items():
        L.append("| %s | %d | %d/%d | %d/%d | %d/%d | %.1f | %s | %.5f | %.5f | %d/%d |" % (k, s["rounds"], s["check1_pass"], s["rounds"], s["check2_pass"], s["rounds"],
                 s["positive_control_pass"], s["rounds"], s["denied_per_round"], ("%.3f" % s["newly_selected_share_of_denied"]) if s["newly_selected_share_of_denied"] is not None else "-",
                 s["tv_masked_mean"], s["tv_random_mean"], s["tv_masked_gt_random_rounds"], s["rounds"]))
    L += ["", "Failures: %s (hygiene = target logits torch.equal after all arms)" % json.dumps(fails)]
    open(os.path.join(AP.OUT, "mask_check.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return sum(fails.values())


if __name__ == "__main__":
    sys.exit(min(main(), 200))
