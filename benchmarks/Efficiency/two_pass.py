"""TWO-PASS RE-DRAFT: same-position K=1 accuracy (single-token overlap, 1-TV) of a plain draft vs a draft that
gathers the KV its first pass requested and RE-DRAFTS THE SAME POSITION with selections recomputed (request relayed
by Codex, 2026-09-23; ledger 'TWO-PASS RE-DRAFT REGISTERED'). This is NOT an accepted-prefix / K-round measurement
and NOT the one-step-arrival (d = 1) data: the d = 1 curve lets draft position j+1 use what position j requested;
here the second pass redoes position 1 itself.

It runs the exact archived code (fork commit 6e6f80a: acceptance_pilot.py, avail_policy.py, nosa_llama.py,
cache_engine.py imported UNCHANGED; this file and two_pass_plan.py are the only additions) with the model and settings
of jobs 2174689 / 2175453 / 2175629 and mask check 2175726 (NOSA-8B, PG-19, L = 16128, warm 8, mech mask, T = 1.0,
top-p 1.0, top-k 0, B = 1).

Per (document, probe site) the committed state S0 is snapshotted ONCE: the engine (CacheSnapshot incl. the GPU window)
AND every availability point's draft LRU (two_pass_plan.clone_streams, BEFORE the target runs). Arms, all at draft
position 1 with the same input token and the same tensor shapes:

  T          the exact target (shipped engine, full availability) from S0 -> p. The target neither denies nor
             advances any draft LRU (avail_policy ROLE_TARGET); gate T-ISO checks that every point's LRU after T equals
             the S0 snapshot by content AND recency order, and the S0 snapshot (never the post-T state) is what the
             draft arms start from. T's fetched KV cannot enter a draft arm: the engine is restored to S0 after T, and
             checks 1/2 below run on P1 AND P2.
  M          the pilot's own plain draft (begin_rollout -> one draft step), the reference for P1.
  P1         plain draft from S0 at budget C  -> q1  (sim_mode 'plain_draft_step1'); records per (layer, KV head,
             request) the selected ids and the requested (= denied, not resident) ids.
  gather 1   admit exactly P1's requested ids into the P1 working LRU at capacity C with the pilot's arrival rule,
             recording every eviction (two_pass_plan.admit). Nothing else enters.
  P2         the SAME position re-drafted from the restored engine state with the post-gather LRU; selections are
             recomputed by the model at every layer; whatever P2 selects that is not resident is DENIED (masked) and is
             a NEW MISS; no further fetch  -> q2  (sim_mode 'two_pass_redraft').
  P3..Pmax   FIXED-POINT known-answer arm: gather the previous pass's new misses only, re-draft again; stops at the
             first pass with zero misses (whose logits MUST equal T bitwise) or at TP_MAX_PASSES.
  ALL_n      (n = 1, 2) pass n with its deny mask forced all-False through the scratch-bias path   check 1: == T
  POISON_n   (n = 1, 2) pass n with every denied slot's K (+/-64) and V (+/-1e4) and every tail row at or beyond
             cache_seqlens overwritten after mask_bias                                            check 2: == pass n
  T2         the target again after every arm                                                    hygiene: == T

PRIMARY METRIC: overlap = 1 - TV(p, q), the single-token speculative acceptance probability, for P1 and P2 on every
round, and the paired difference overlap(P2) - overlap(P1); per-document means (8 probes each) for all documents and a
document cluster bootstrap (>= 10,000 resamples, fixed seed, 95% percentile). SECONDARY (labelled): sampled accept
draws with common random numbers (the pilot's seeds, identical for every arm).

Raw export (for plotting elsewhere): raw/two_pass_raw_p<id>.npz per point + raw/manifest.json (dtypes, shapes, codes);
the NPZ round-trip is verified exactly before the manifest is written.
Env: TP_DOCS (8), TP_PROBES (8), TP_MAX_PASSES (4), TP_NBOOT (10000), plus the pilot's env. Exit code = failures.
"""
import json
import os
import sys
import time

os.environ.setdefault("NOSI_ACC_MODE", "rollout")
os.environ.setdefault("NOSI_ACC_K", "1")
if "TP_DOCS" in os.environ:
    os.environ["NOSI_ACC_DOCS"] = os.environ["TP_DOCS"]
if "TP_PROBES" in os.environ:
    os.environ["NOSI_ACC_PROBES"] = os.environ["TP_PROBES"]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import acceptance_pilot as AP  # noqa: E402   (the archived pilot, unchanged)
import two_pass_plan as TP  # noqa: E402

K_POISON, V_POISON = 64.0, 1.0e4      # the mask-check values (mask_check.py:48)
MAXP = int(os.environ.get("TP_MAX_PASSES", "4"))
NBOOT = int(os.environ.get("TP_NBOOT", "10000"))
BOOT_SEED = 20260923
if MAXP < 2:
    raise SystemExit("TP_MAX_PASSES must be >= 2 (P1 and P2 always run)")
if NBOOT < 10000:
    raise SystemExit("TP_NBOOT must be >= 10000")

GATES = ("p1_equals_plain_draft", "t_isolation_lru", "check1_all_visible_eq_target_p1", "check1_all_visible_eq_target_p2",
         "check2_poison_denied_eq_pass_p1", "check2_poison_denied_eq_pass_p2", "layer0_no_new_miss",
         "zero_p1_denial_p2_eq_p1", "zero_denial_pass_eq_target", "d0_identity", "requests_eq_sel_minus_resident",
         "no_stray_inflight", "category_consistent", "finite_logits", "hygiene")


def sim_mode(n):
    return "plain_draft_step1" if n == 1 else ("two_pass_redraft" if n == 2 else "iterated_redraft_pass%d" % n)


def family(point):
    d = point["d"]
    if d == 0:
        return "d0_identity_shipped_engine"
    if d == float("inf"):
        return "start_dinf_fixed_residency"
    return "start_d%g_rollout" % d


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
    NL = int(pilot.model.config.num_hidden_layers)
    orig_mask_bias = pol.mask_bias
    orig_on_diff = pol.on_diff
    arm = {"name": None, "seed": 0}
    cap = {"on": False, "keys": {}}

    def engine_of(layer):
        return pilot.cache.layers[layer].cache_engine

    # ---- hooks (instance attributes; the model calls avail_policy.POLICY.<hook>) ------------------------------
    def hooked_on_diff(block_map, new_block_map_buf, load_mask, topk_idx, tail_slot):
        new_map = new_block_map_buf.cpu().tolist() if cap["on"] else None     # read BEFORE the policy runs
        orig_on_diff(block_map, new_block_map_buf, load_mask, topk_idx, tail_slot)
        if cap["on"]:
            deny = pol._deny_cpu if pol.role == ap_mod.ROLE_DRAFT else None
            cap["keys"].update(TP.selection_and_requests(new_map, deny, int(tail_slot), int(pol.layer)))

    def poison_rows(eng, b, h, r0, r1, gen):                                    # = mask_check.py:68-75
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
        if name == "ALL" and pol.role == ap_mod.ROLE_DRAFT:                    # = mask_check.py:80-84
            H, B, M = engine_of(layer)._block_map.shape
            pol._deny_dev = torch.zeros((H, B, M), dtype=torch.bool, device=kv_bias.device)
            return orig_mask_bias(kv_bias)
        out = orig_mask_bias(kv_bias)
        if name == "POISON" and pol.role == ap_mod.ROLE_DRAFT:                 # = mask_check.py:104-123 (arm P)
            eng = engine_of(layer)
            H, B, M = eng._block_map.shape
            bs = eng.block_size
            deny = pol._deny_dev.cpu() if pol._deny_dev is not None else torch.zeros((H, B, M), dtype=torch.bool)
            gen = torch.Generator().manual_seed(arm["seed"] * 7919 + layer)
            for h in range(H):
                for b in range(B):
                    for m in range(M - 1):
                        if bool(deny[h, b, m]):
                            poison_rows(eng, b, h, m * bs, (m + 1) * bs, gen)
                    vis = int(eng._cache_lens[b])
                    poison_rows(eng, b, h, vis, M * bs, gen)
        return out

    pol.on_diff = hooked_on_diff
    pol.mask_bias = hooked_mask_bias
    assert ap_mod.POLICY is pol

    H0, B0, M0 = None, None, None
    fails = {g: 0 for g in GATES}
    rounds = []                      # one paired record per (doc, probe, point)
    raw = {}                         # pid -> lists of per-round arrays
    t_start = time.time()

    for d in range(AP.NDOCS):
        prompt, forced = ids[d:d + 1, :AP.L], ids[d:d + 1, AP.L:]
        pol.new_document(d)
        pilot.open_document(prompt)
        pilot.step(forced[:, 0:1], ap_mod.ROLE_WARM)
        trans = pilot.ss.transient_ids(pilot.model)
        for t in range(1, 1 + AP.WARM):
            pilot.step(forced[:, t:t + 1], ap_mod.ROLE_WARM)
        if H0 is None:
            H0, B0, M0 = (int(x) for x in engine_of(0)._block_map.shape)
        W = M0 - 1                                                              # non-tail slots
        for probe in probes:
            first = forced[:, probe:probe + 1]
            pilot.snap.take()
            pos0 = pilot.pos.clone()

            def restore():
                pilot.snap.restore(); pilot.pos = pos0.clone()
                pilot.ss.assert_transients_intact(pilot.model, trans)

            def run(role, name, seed):
                arm.update(name=name, seed=seed)
                lg = pilot.step(first, role).clone()
                arm.update(name=None)
                restore()
                if not bool(torch.isfinite(lg).all()):
                    fails["finite_logits"] += 1
                return lg

            def captured(role, name, seed):
                cap["on"], cap["keys"] = True, {}
                try:
                    lg = run(role, name, seed)
                finally:
                    cap["on"] = False
                return lg, dict(cap["keys"])

            seed = d * 1009 + probe
            # S0: every point's draft LRU, snapshotted BEFORE the target runs
            s0_lru = {pt["id"]: TP.clone_streams(pol.points[pt["id"]]["streams"]) for pt in pilot.points}
            s0_fp = {pid: TP.lru_fingerprint(s) for pid, s in s0_lru.items()}
            lg_T, cap_T = captured(ap_mod.ROLE_TARGET, "T", seed)
            p_T = pilot.dist(lg_T[0])
            x_T = am.sample(p_T, np.random.default_rng(am.seed_token(d, probe, 1)))
            u = float(np.random.default_rng(am.seed_accept(d, probe, 1)).random())

            for point in pilot.points:
                pid = point["id"]
                dpt = point["d"]
                g = {k: True for k in GATES}
                # T-ISO: the target left this point's LRU exactly as S0 had it; then start from the S0 SNAPSHOT
                g["t_isolation_lru"] = TP.lru_fingerprint(pol.points[pid]["streams"]) == s0_fp[pid]
                pol.points[pid]["streams"] = TP.clone_streams(s0_lru[pid])
                pol.set_site(probe)
                pol.begin_rollout(pid)
                lg_M = run(ap_mod.ROLE_DRAFT, "M", seed)
                pol.end_rollout()

                pol.set_site(probe)
                pol.begin_rollout(pid)
                entering_fp = TP.lru_fingerprint(pol.streams)
                passes = []                        # per pass: dict
                s0res = None
                gathered_before = {}               # key -> set (union over gathers so far)
                evicted_before = {}
                last_evicted = None                # key -> list, the gather that preceded the current pass
                n_stray = n_badreq = n_badcat = 0
                for n in range(1, MAXP + 1):
                    pre = TP.clone_streams(pol.streams)
                    lg_n, rec = captured(ap_mod.ROLE_DRAFT, "PASS", seed)
                    post = pol.streams
                    info = dict(n=n, logits=lg_n, rec=rec)
                    if n <= 2:
                        pol.streams = TP.clone_streams(pre)
                        lg_all = run(ap_mod.ROLE_DRAFT, "ALL", seed)
                        pol.streams = TP.clone_streams(pre)
                        lg_poi, rec_poi = captured(ap_mod.ROLE_DRAFT, "POISON", seed)
                        pol.streams = post
                        info["check1"] = bool(torch.equal(lg_all, lg_T))
                        info["check2"] = bool(torch.equal(lg_poi, lg_n)) and rec_poi == rec
                        info["max_abs_all_vs_target"] = float((lg_all - lg_T).abs().max())
                        info["max_abs_poison_vs_pass"] = float((lg_poi - lg_n).abs().max())
                    before = {k: set(vc.res.keys()) for k, vc in post.items()}
                    if n == 1:
                        s0res = before
                    info["denied_total"] = sum(len(r) for _, r in rec.values())
                    info["denied_per_layer"] = [0] * NL
                    for (l, h, b), (_, r) in rec.items():
                        info["denied_per_layer"][l] += len(r)
                    if dpt != 0:
                        for key, (sel, req) in rec.items():
                            if not TP.check_requests(sel, req, before.get(key, set())):
                                n_badreq += 1
                    # categories of this pass's misses (n >= 2) and of the evictions of the gather that preceded it
                    if n >= 2:
                        cats, ecats = {}, {}
                        for key, (sel, req) in rec.items():
                            c, bad = TP.categorize_new_misses(req, s0res.get(key, set()), gathered_before.get(key, set()),
                                                              evicted_before.get(key, set()))
                            cats[key] = c
                            n_badcat += bad if dpt != 0 else 0
                            ecats[key] = TP.categorize_evictions(last_evicted.get(key, []), req, gathered_before.get(key, set()))
                        info["newmiss_cats"] = cats
                        info["evicted_cats"] = ecats
                    # common random numbers (SECONDARY): the pilot's seeds, identical for every arm
                    q = pilot.dist(lg_n[0])
                    x = am.sample(q, np.random.default_rng(am.seed_token(d, probe, 1)))
                    a, ratio = am.acceptance(p_T, q, x)
                    ov, tv = am.overlap_and_tv(p_T, q)
                    info.update(q_x=float(q[x]), p_x=float(p_T[x]), x=int(x), accept_prob=float(a), accept_draw=int(u < a),
                                overlap=float(ov), tv=float(tv), argmax=int(np.argmax(q)), argmax_agree=int(np.argmax(q) == np.argmax(p_T)))
                    passes.append(info)
                    if info["denied_total"] == 0 and not torch.equal(lg_n, lg_T):
                        g["zero_denial_pass_eq_target"] = False
                    if n >= 2 and info["denied_per_layer"][0] != 0:
                        g["layer0_no_new_miss"] = False
                    if n >= 2 and info["denied_total"] == 0:
                        break
                    if n == MAXP:
                        break
                    # GATHER after pass n: exactly this pass's requested ids, with evictions recorded
                    gathered, last_evicted = {}, {}
                    for key, vc in post.items():
                        req = rec.get(key, ((), ()))[1]
                        ev, stray = TP.admit(vc, req)
                        n_stray += len(stray)
                        gathered[key] = sorted(set(req))
                        last_evicted[key] = ev
                        gathered_before.setdefault(key, set()).update(req)
                        evicted_before.setdefault(key, set()).update(ev)
                    info["gathered"] = gathered
                    info["evicted"] = dict(last_evicted)
                    info["units"] = TP.transfer_units(gathered)
                pol.end_rollout()

                P1, P2 = passes[0], passes[1]
                g["p1_equals_plain_draft"] = bool(torch.equal(P1["logits"], lg_M))
                g["check1_all_visible_eq_target_p1"] = P1["check1"]
                g["check1_all_visible_eq_target_p2"] = P2["check1"]
                g["check2_poison_denied_eq_pass_p1"] = P1["check2"]
                g["check2_poison_denied_eq_pass_p2"] = P2["check2"]
                if P1["denied_total"] == 0:
                    g["zero_p1_denial_p2_eq_p1"] = bool(torch.equal(P2["logits"], P1["logits"]))
                if dpt == 0:
                    g["d0_identity"] = (P1["denied_total"] == 0 and bool(torch.equal(P1["logits"], lg_T))
                                        and bool(torch.equal(P2["logits"], lg_T)))
                g["requests_eq_sel_minus_resident"] = n_badreq == 0
                g["no_stray_inflight"] = n_stray == 0
                g["category_consistent"] = n_badcat == 0
                for k in GATES:
                    if k in ("hygiene", "finite_logits"):
                        continue
                    fails[k] += int(not g[k])
                conv = next((p["n"] for p in passes if p["denied_total"] == 0), -1)
                nm_cat_counts = {1: 0, 2: 0, 3: 0}
                for key, c in P2["newmiss_cats"].items():
                    for _, code in c:
                        nm_cat_counts[code] += 1
                ev_cat_counts = {2: 0, 3: 0, 4: 0}
                for key, c in P2["evicted_cats"].items():
                    for _, code in c:
                        ev_cat_counts[code] += 1
                rnd = dict(label=TP.LABEL, doc=d, doc_row=int(doc_rows[d]), probe=probe, point=point["spec"], point_id=pid,
                           C=point["C"], d=("inf" if dpt == float("inf") else dpt), start_family=family(point),
                           overlap_p1=P1["overlap"], overlap_p2=P2["overlap"], tv_p1=P1["tv"], tv_p2=P2["tv"],
                           overlap_diff_p2_minus_p1=P2["overlap"] - P1["overlap"],
                           argmax_agree_p1=P1["argmax_agree"], argmax_agree_p2=P2["argmax_agree"],
                           secondary_sampled_accept_p1=P1["accept_draw"], secondary_sampled_accept_p2=P2["accept_draw"],
                           secondary_accept_prob_p1=P1["accept_prob"], secondary_accept_prob_p2=P2["accept_prob"], u=u,
                           p1_denied=P1["denied_total"], p2_new_miss=P2["denied_total"],
                           p2_new_miss_per_layer=P2["denied_per_layer"], p1_denied_per_layer=P1["denied_per_layer"],
                           p2_new_miss_a_never_resident=nm_cat_counts[1], p2_new_miss_b_s0_evicted_by_gather=nm_cat_counts[2],
                           p2_new_miss_c_gathered_then_evicted=nm_cat_counts[3],
                           gather1_evicted_b=ev_cat_counts[2], gather1_evicted_c=ev_cat_counts[3], gather1_evicted_unused=ev_cat_counts[4],
                           gather1_units=P1.get("units"), passes_run=len(passes), passes_to_converge=conv,
                           overlap_by_pass=[p["overlap"] for p in passes], tv_by_pass=[p["tv"] for p in passes],
                           denied_by_pass=[p["denied_total"] for p in passes],
                           max_abs_all_vs_target_p1=P1["max_abs_all_vs_target"], max_abs_all_vs_target_p2=P2["max_abs_all_vs_target"],
                           max_abs_poison_vs_pass_p1=P1["max_abs_poison_vs_pass"], max_abs_poison_vs_pass_p2=P2["max_abs_poison_vs_pass"],
                           entering_p1_lru_fingerprint_hash=hash(entering_fp) & 0xFFFFFFFF,
                           gates=g)
                rounds.append(rnd)
                raw.setdefault(pid, []).append(export_round(TP, passes, cap_T, s0res, point, d, doc_rows[d], probe, u, x_T,
                                                            NL, H0, B0, W))
                print("[twopass] doc %d probe %d %-11s P1 denied %4d ov %.5f | P2 new %4d (a %d b %d c %d) ov %.5f | diff %+.5f | "
                      "conv %d | gates %s" % (d, probe, point["spec"], P1["denied_total"], P1["overlap"], P2["denied_total"],
                                              nm_cat_counts[1], nm_cat_counts[2], nm_cat_counts[3], P2["overlap"],
                                              P2["overlap"] - P1["overlap"], conv,
                                              "ok" if all(g.values()) else [k for k, v in g.items() if not v]), flush=True)
            lg_T2 = run(ap_mod.ROLE_TARGET, "T", seed)
            fails["hygiene"] += int(not bool(torch.equal(lg_T2, lg_T)))
            pilot.step(first, ap_mod.ROLE_WARM)        # advance the verified trajectory, as the pilot does
        print("[twopass] doc %d done (%.0fs)" % (d, time.time() - t_start), flush=True)

    return finish(TP, rounds, raw, fails, pilot, doc_rows, probes, t_start)


def export_round(TP_, passes, cap_T, s0res, point, d, doc_row, probe, u, x_T, NL, H, B, W):
    """One round's raw arrays. Arm axis A = [T, P1 .. P_MAXP]; gather axis G = [after P1 .. after P_{MAXP-1}];
    new-miss axis = [P2 .. P_MAXP]. -1 = padding / not run."""
    A, G = 1 + MAXP, MAXP - 1
    sel_ids = np.full((A, NL, H, B, W), -1, np.int16)
    sel_state = np.full((A, NL, H, B, W), -1, np.int8)           # 1 visible, 2 denied
    req_ids = np.full((MAXP, NL, H, B, W), -1, np.int16)
    t_in_s0 = np.full((NL, H, B, W), -1, np.int8)                # T's ids: 1 resident in the P1 draft LRU, 0 not
    gath = np.full((G, NL, H, B, W), -1, np.int16)
    evid = np.full((G, NL, H, B, W), -1, np.int16)
    evcat = np.full((G, NL, H, B, W), -1, np.int8)
    nmid = np.full((MAXP - 1, NL, H, B, W), -1, np.int16)
    nmcat = np.full((MAXP - 1, NL, H, B, W), -1, np.int8)
    denied_pl = np.full((A, NL), -1, np.int16)
    nm_pl_cat = np.full((MAXP - 1, NL, 3), -1, np.int16)
    units = np.full((G, 6), -1, np.int64)
    ran = np.zeros(A, np.int8)
    ov = np.full(A, np.nan); tv = np.full(A, np.nan); qx = np.full(A, np.nan); px = np.full(A, np.nan); acc = np.full(A, np.nan)
    xs = np.full(A, -1, np.int32); am_ = np.full(A, -1, np.int32); agree = np.full(A, -1, np.int8); draw = np.full(A, -1, np.int8)
    # T
    ran[0] = 1; ov[0] = 1.0; tv[0] = 0.0; acc[0] = 1.0; xs[0] = x_T; draw[0] = 1; agree[0] = 1; denied_pl[0] = 0
    for (l, h, b), (sel, _) in cap_T.items():
        sel_ids[0, l, h, b] = TP_.pad_ids(sel, W)
        sel_state[0, l, h, b, :len(sel)] = 1
        res = s0res.get((l, h, b), set()) if s0res is not None else set()
        t_in_s0[l, h, b, :len(sel)] = [1 if s in res else 0 for s in sel]
    for p in passes:
        n = p["n"]
        ran[n] = 1; ov[n] = p["overlap"]; tv[n] = p["tv"]; qx[n] = p["q_x"]; px[n] = p["p_x"]; acc[n] = p["accept_prob"]
        xs[n] = p["x"]; am_[n] = p["argmax"]; agree[n] = p["argmax_agree"]; draw[n] = p["accept_draw"]
        denied_pl[n] = p["denied_per_layer"]
        for (l, h, b), (sel, req) in p["rec"].items():
            sel_ids[n, l, h, b] = TP_.pad_ids(sel, W)
            rs = set(req)
            sel_state[n, l, h, b, :len(sel)] = [2 if s in rs else 1 for s in sel]
            req_ids[n - 1, l, h, b] = TP_.pad_ids(req, W)
        if "gathered" in p:
            gi = n - 1
            gath[gi] = TP_.key_grid(p["gathered"], NL, H, B, W)
            units[gi] = [p["units"][k] for k in ("per_head_blocks", "both_heads_blocks", "bytes_k_per_head_unit",
                                                  "bytes_kv_per_head_unit", "bytes_k_both_heads_unit", "bytes_kv_both_heads_unit")]
        if n >= 2:
            ni = n - 2
            nm_pl_cat[ni] = 0
            for (l, h, b), c in p["newmiss_cats"].items():
                nmid[ni, l, h, b] = TP_.pad_ids([i for i, _ in c], W)
                nmcat[ni, l, h, b] = TP_.pad_ids([k for _, k in c], W, dtype=np.int8)
                for _, k in c:
                    nm_pl_cat[ni, l, k - 1] += 1
            gi = n - 2                                  # the gather that preceded pass n
            for (l, h, b), c in p["evicted_cats"].items():
                evid[gi, l, h, b] = TP_.pad_ids([i for i, _ in c], W)
                evcat[gi, l, h, b] = TP_.pad_ids([k for _, k in c], W, dtype=np.int8)
    return dict(doc=np.int32(d), doc_row=np.int32(doc_row), probe=np.int32(probe), u=np.float64(u), ran=ran,
                sel_ids=sel_ids, sel_state=sel_state, requested_ids=req_ids, t_resident_in_p1_draft_lru=t_in_s0,
                gathered_ids=gath, evicted_ids=evid, evicted_cat=evcat, newmiss_ids=nmid, newmiss_cat=nmcat,
                denied_per_layer=denied_pl, newmiss_per_layer_by_cat=nm_pl_cat, gather_units=units,
                overlap=ov, tv=tv, q_of_x=qx, p_of_x=px, accept_prob=acc, x=xs, argmax_q=am_, argmax_agree=agree,
                accept_draw=draw)


def finish(TP_, rounds, raw, fails, pilot, doc_rows, probes, t_start):
    os.makedirs(AP.OUT, exist_ok=True)
    rawdir = os.path.join(AP.OUT, "raw")
    os.makedirs(rawdir, exist_ok=True)
    # ---- raw export + exact round trip ------------------------------------------------------------------------
    files, arrays_meta = [], {}
    for pid, lst in raw.items():
        arrays = {k: np.stack([r[k] for r in lst]) for k in lst[0]}
        path = os.path.join(rawdir, "two_pass_raw_p%d.npz" % pid)
        arrays_meta[os.path.basename(path)] = TP_.write_export(path, arrays)
        bad = TP_.verify_roundtrip(path, arrays)
        if bad:
            fails.setdefault("export_roundtrip", 0)
            fails["export_roundtrip"] += len(bad)
        files.append(path)
    pts = {p["id"]: p for p in pilot.points}
    manifest = dict(
        label=TP_.LABEL, not_this="NOT an accepted-prefix or K-round measurement; NOT the d = 1 one-step-arrival data",
        files={os.path.basename(f): dict(point=pts[int(os.path.basename(f).split("_p")[-1].split(".")[0])]["spec"])
               for f in files},
        arrays=arrays_meta,
        axes=dict(round="(doc, probe) in run order; doc/doc_row/probe arrays name it",
                  arm="A = [T, P1 .. P%d]" % MAXP, gather="G = [after P1 .. after P%d]" % (MAXP - 1),
                  newmiss_pass="[P2 .. P%d]" % MAXP, layer="0..31", kv_head="0..1", request="0 (B = 1)",
                  slot="the non-tail entries in slot order, -1 padded (width 63)"),
        codes=dict(sel_state={"-1": "padding / arm not run", "1": "visible (resident in the draft LRU)", "2": "denied (masked, requested)"},
                   t_resident_in_p1_draft_lru={"-1": "padding", "0": "T's id not resident in the P1 draft LRU", "1": "resident"},
                   category={str(k): v for k, v in TP_.CATEGORY_NAMES.items()},
                   newmiss_per_layer_by_cat="last axis = categories (a), (b), (c)",
                   gather_units="[per_head_blocks, both_heads_blocks, bytes_k_per_head_unit, bytes_kv_per_head_unit, "
                                "bytes_k_both_heads_unit, bytes_kv_both_heads_unit]"),
        tail="slot 63 (the in-progress tail block) is excluded from every id set and is never denied (avail_policy.py:82-87)",
        units=dict(per_head_block_k_bytes=TP_.PER_HEAD_BLOCK_K_BYTES, per_head_block_kv_bytes=TP_.PER_HEAD_BLOCK_KV_BYTES,
                   both_heads_block_k_bytes=TP_.BOTH_HEADS_BLOCK_K_BYTES, both_heads_block_kv_bytes=TP_.BOTH_HEADS_BLOCK_KV_BYTES,
                   note="per-head unit = one (layer, KV head, request, block); both-heads unit = one (layer, request, block) moved "
                        "for both heads (NOSI's contiguous host block), counted once even if only one head needed it"),
        scalars="overlap = 1 - TV(p_T, q) (PRIMARY, float64); tv; q_of_x, p_of_x, accept_prob, accept_draw, u, x = SECONDARY "
                "sampled acceptance with common random numbers (am.seed_token / am.seed_accept at j = 1, identical for every arm)",
        size_mb=None, roundtrip="np.array_equal per array after reload (dtype, shape, values)")
    manifest["size_mb"] = TP_.export_size_mb(files)
    TP_.write_manifest(os.path.join(rawdir, "manifest.json"), manifest)

    # ---- summaries: per point, document is the unit ------------------------------------------------------------
    summary = {}
    for pt in pilot.points:
        rs = [r for r in rounds if r["point_id"] == pt["id"]]
        if not rs:
            continue
        by_doc = {}
        for r in rs:
            by_doc.setdefault(r["doc"], []).append(r)
        docs = sorted(by_doc)
        o1 = [[r["overlap_p1"] for r in by_doc[k]] for k in docs]
        o2 = [[r["overlap_p2"] for r in by_doc[k]] for k in docs]
        df = [[r["overlap_diff_p2_minus_p1"] for r in by_doc[k]] for k in docs]
        b1 = TP_.doc_bootstrap(o1, NBOOT, BOOT_SEED)
        b2 = TP_.doc_bootstrap(o2, NBOOT, BOOT_SEED)
        bd = TP_.doc_bootstrap(df, NBOOT, BOOT_SEED)
        den = sum(r["p1_denied"] for r in rs)
        nm = sum(r["p2_new_miss"] for r in rs)
        ca = sum(r["p2_new_miss_a_never_resident"] for r in rs)
        cb = sum(r["p2_new_miss_b_s0_evicted_by_gather"] for r in rs)
        cc = sum(r["p2_new_miss_c_gathered_then_evicted"] for r in rs)
        with_den = [r for r in rs if r["p1_denied"] > 0]
        conv = {}
        for r in rs:
            conv[str(r["passes_to_converge"])] = conv.get(str(r["passes_to_converge"]), 0) + 1
        units = [r["gather1_units"] for r in rs if r["gather1_units"]]
        summary[pt["spec"]] = dict(
            label=TP_.LABEL, start_family=family(pt), rounds=len(rs), docs=docs,
            overlap_p1=b1, overlap_p2=b2, overlap_diff_p2_minus_p1=bd,
            diff_sign_counts=TP_.sign_counts([r["overlap_diff_p2_minus_p1"] for r in rs]),
            tv_p1_mean=float(np.mean([r["tv_p1"] for r in rs])), tv_p2_mean=float(np.mean([r["tv_p2"] for r in rs])),
            rounds_with_p1_denials=len(with_den),
            rounds_with_p1_denials_tv_improved=sum(1 for r in with_den if r["tv_p2"] < r["tv_p1"]),
            p1_denied_per_round=den / len(rs), p2_new_miss_per_round=nm / len(rs),
            p2_new_miss_share_of_p1_denied=(nm / den) if den else None,
            p2_new_miss_categories=dict(a_never_resident=ca, b_s0_evicted_by_gather=cb, c_gathered_then_evicted=cc),
            bc_share_of_new_miss=((cb + cc) / nm) if nm else None,
            p2_new_miss_layer0=sum(r["p2_new_miss_per_layer"][0] for r in rs),
            p2_new_miss_per_layer=[sum(r["p2_new_miss_per_layer"][l] for r in rs) for l in range(len(rs[0]["p2_new_miss_per_layer"]))],
            passes_to_converge_hist=conv,
            gather1_mean_per_head_blocks=float(np.mean([u["per_head_blocks"] for u in units])) if units else 0.0,
            gather1_mean_both_heads_blocks=float(np.mean([u["both_heads_blocks"] for u in units])) if units else 0.0,
            gather1_mean_bytes_kv_per_head_unit=float(np.mean([u["bytes_kv_per_head_unit"] for u in units])) if units else 0.0,
            gather1_mean_bytes_kv_both_heads_unit=float(np.mean([u["bytes_kv_both_heads_unit"] for u in units])) if units else 0.0,
            secondary_sampled_accept_p1=float(np.mean([r["secondary_sampled_accept_p1"] for r in rs])),
            secondary_sampled_accept_p2=float(np.mean([r["secondary_sampled_accept_p2"] for r in rs])),
            argmax_agree_p1=float(np.mean([r["argmax_agree_p1"] for r in rs])),
            argmax_agree_p2=float(np.mean([r["argmax_agree_p2"] for r in rs])))
    predictions = evaluate_predictions(summary, fails)
    out = dict(label=TP_.LABEL, rounds=rounds, summary=summary, fails=fails, predictions=predictions, probes=probes,
               docs=[int(x) for x in doc_rows[:AP.NDOCS]], points=[p["spec"] for p in pilot.points],
               max_passes=MAXP, n_boot=NBOOT, boot_seed=BOOT_SEED, poison=dict(K=K_POISON, V=V_POISON),
               commit_note="fork 6e6f80a + benchmarks/Efficiency/two_pass.py + two_pass_plan.py only",
               raw_manifest="raw/manifest.json", seconds=time.time() - t_start)
    json.dump(out, open(os.path.join(AP.OUT, "two_pass.json"), "w"), indent=1, default=str)
    L = render_md(TP_, summary, fails, predictions, manifest)
    open(os.path.join(AP.OUT, "two_pass.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return sum(fails.values())


def _ci(b):
    return "%.5f [%.5f, %.5f]" % (b["mean"], b["lo"], b["hi"])


def evaluate_predictions(summary, fails):
    """The registered predictions, evaluated mechanically (ledger 'TWO-PASS RE-DRAFT REGISTERED')."""
    ev = {}
    ev["1_all_gates_pass"] = sum(fails.values()) == 0
    delayed = {k: s for k, s in summary.items() if s["start_family"] != "d0_identity_shipped_engine"}
    ev["2_mean_diff_positive_every_C"] = all(s["overlap_diff_p2_minus_p1"]["mean"] > 0 for s in delayed.values())
    c63 = summary.get("C=63,d=1")
    ev["2_ci_excludes_0_at_C63_d1"] = (c63["overlap_diff_p2_minus_p1"]["lo"] > 0) if c63 else None
    ev["2_FALSIFIED_ci_at_or_below_0_at"] = [k for k, s in delayed.items() if s["overlap_diff_p2_minus_p1"]["hi"] <= 0]
    ev["3_new_miss_share_in_3_to_25pct_every_C"] = all(s["p2_new_miss_share_of_p1_denied"] is not None
                                                       and 0.03 <= s["p2_new_miss_share_of_p1_denied"] <= 0.25
                                                       for s in delayed.values())
    ev["3_layer0_zero"] = all(s["p2_new_miss_layer0"] == 0 for s in summary.values())
    ev["4_bc_share_le_1pct"] = all((s["bc_share_of_new_miss"] or 0.0) <= 0.01 for s in delayed.values())
    if c63:
        tv1 = c63["tv_p1_mean"]
        gain = c63["overlap_diff_p2_minus_p1"]["mean"]
        ev["5_gain_over_tv_p1_at_C63_d1"] = gain / tv1 if tv1 else None
        ev["5_in_0.2_to_0.7"] = (0.2 <= gain / tv1 <= 0.7) if tv1 else None
    return ev


def render_md(TP_, summary, fails, predictions, manifest):
    L = ["# Two-pass re-draft: %s" % TP_.LABEL, "",
         "Plain draft (P1) vs gather P1's requested KV and re-draft the SAME position with recomputed selections (P2). "
         "Fork 6e6f80a (archived pilot unchanged) + two_pass.py. NOT accepted-prefix / K-round, NOT the d = 1 data.", "",
         "## Primary: %s, per point (mean [95%% document-bootstrap interval, %d resamples])" % (TP_.LABEL, NBOOT), "",
         "| point | start family | rounds | P1 overlap | P2 overlap | P2 - P1 | rounds +/0/- | P1 denied / round | P2 new misses / round (share of P1) | new misses (a)/(b)/(c) | layer-0 new misses | converged at pass (hist) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in summary.items():
        sc = s["diff_sign_counts"]
        cat = s["p2_new_miss_categories"]
        L.append("| %s | %s | %d | %s | %s | %s | %d/%d/%d | %.1f | %.1f (%s) | %d/%d/%d | %d | %s |" % (
            k, s["start_family"], s["rounds"], _ci(s["overlap_p1"]), _ci(s["overlap_p2"]), _ci(s["overlap_diff_p2_minus_p1"]),
            sc["positive"], sc["zero"], sc["negative"], s["p1_denied_per_round"], s["p2_new_miss_per_round"],
            ("%.3f" % s["p2_new_miss_share_of_p1_denied"]) if s["p2_new_miss_share_of_p1_denied"] is not None else "-",
            cat["a_never_resident"], cat["b_s0_evicted_by_gather"], cat["c_gathered_then_evicted"], s["p2_new_miss_layer0"],
            json.dumps(s["passes_to_converge_hist"], sort_keys=True)))
    L += ["", "## %s: per-document mean of overlap(P2) - overlap(P1) (8 probes each)" % TP_.LABEL, "",
          "| point | " + " | ".join("doc %d" % i for i in range(len(next(iter(summary.values()))["overlap_diff_p2_minus_p1"]["per_doc_mean"]))) + " |",
          "|---|" + "---|" * len(next(iter(summary.values()))["overlap_diff_p2_minus_p1"]["per_doc_mean"])]
    for k, s in summary.items():
        L.append("| %s | " % k + " | ".join("%+.5f" % v for v in s["overlap_diff_p2_minus_p1"]["per_doc_mean"]) + " |")
    L += ["", "## Gather 1 transfer volume (per round, B = 1, all 32 layers)", "",
          "| point | per-head blocks | both-heads blocks | bytes K+V per-head unit (32 KiB) | bytes K+V both-heads unit (64 KiB) |",
          "|---|---|---|---|---|"]
    for k, s in summary.items():
        L.append("| %s | %.1f | %.1f | %.0f | %.0f |" % (k, s["gather1_mean_per_head_blocks"], s["gather1_mean_both_heads_blocks"],
                                                        s["gather1_mean_bytes_kv_per_head_unit"], s["gather1_mean_bytes_kv_both_heads_unit"]))
    L += ["", "## SECONDARY (labelled): sampled single-token accept with common random numbers; argmax agreement", "",
          "| point | sampled accept P1 | sampled accept P2 | argmax agree P1 | argmax agree P2 |", "|---|---|---|---|---|"]
    for k, s in summary.items():
        L.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (k, s["secondary_sampled_accept_p1"], s["secondary_sampled_accept_p2"],
                                                        s["argmax_agree_p1"], s["argmax_agree_p2"]))
    L += ["", "Gate failures: %s" % json.dumps(fails), "", "Registered predictions (mechanical): %s" % json.dumps(predictions),
          "", "Raw export: raw/manifest.json (%.1f MB)" % (manifest["size_mb"] or 0.0)]
    return L


if __name__ == "__main__":
    sys.exit(min(main(), 200))
