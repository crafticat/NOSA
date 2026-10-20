"""CPU tests of the M-P3 profiling cell's pure parts (benchmarks/Efficiency/r2_p3_profile.py; docs/R2_BASE_P3.md 4).

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-base-p3 python -m pytest -q tests/test_r2_p3.py

P1  classify: the kernel names of both steps (cuBLAS / CUTLASS GEMMs, the stage-1 split-KV kernel BEFORE the attention
    split-KV kernels, the pooling, top-k / sort, the Triton gathers and write-back, the diff kernel, flashinfer's fused
    ops, PyTorch's small kernels), memcpy / memset as COPY, an unknown name as OTHER; the REAL names of run 2190876,
    whose flash kernels name cutlass::bfloat16_t, as STAGE1 / ATTN (not GEMM). NEGATIVE CONTROL: that run's own rules
    (GEMM first) put exactly the three flash kernels in GEMM.
P2  parse_trace on a synthetic Kineto trace: a kernel is attributed to the innermost sh. / r2. range of its LAUNCH (the
    cuda_runtime or cuda_driver event with its correlation id) on the launching thread; NEGATIVE CONTROLS: a launch
    outside every range, a range of another thread, a range whose name lacks the prefixes, and a graph launch outside
    the ranges give no range; a kernel without a launch record gives no range.
P3  aggregate: per-step counts and ms per class, per range ('-' none), the names by device time.
P4  q_verdicts: every Q CONFIRMED on a constructed aggregate that meets the registered limits; REFUTED one by one when
    a limit is broken; R2's side falls back to the eager steps when the graph's kernels were not seen; UNDETERMINED
    when SH or R2E saw nothing.
P5  the windows: 16 steps each, disjoint and in order; SH's windows hold no shipped re-warm at S 16128
    (r2_base_cell.expected_rewarms); R2's capture step precedes its timed replays.
P6  the offline re-analysis (benchmarks/Efficiency/r2_p3_reclass.py), pure parts: rules_of / reclassified (the
    recorded rules reproduce the recorded classes), same_profile (equal -> []; NEGATIVE CONTROLS: a changed value and
    a missing class are named), name_delta, seq_variants (instances grouped by sequence, per-position medians),
    cut_steps, transfer_ranges (matched kernels take the eager step's range, an extra graph kernel is '?unmatched',
    a kernel before the first mark '?pre'; NEGATIVE CONTROLS: a match below MATCH_MIN and a wrong step count are
    UNDETERMINED).
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

NOSI_REPO = Path(os.environ.get("NOSI_REPO") or Path(__file__).resolve().parents[1])
EFF = NOSI_REPO / "benchmarks/Efficiency"
if not (EFF / "r2_p3_profile.py").is_file():
    pytest.skip("no r2_p3_profile.py under NOSI_REPO=%s" % NOSI_REPO, allow_module_level=True)
sys.path.insert(0, str(EFF))
import r2_p3_profile as P   # noqa: E402
import r2_base_cell as BC   # noqa: E402


def test_P1_classify():
    cases = {
        "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn": "GEMM",
        "sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel": "GEMM",
        "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8>(Params)": "GEMM",
        "void gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, float, false, true>": "GEMM",
        "void flash::flash_fwd_splitkv_stage1_kernel<Flash_fwd_kernel_traits<128, 16, 64, 1>, false>": "STAGE1",
        "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4>, false, false>(Flash_fwd_params)": "ATTN",
        "void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128>, 4, 2, true>(Flash_fwd_params)": "ATTN",
        "nosa_pool_kernel_per_request": "POOL",
        "nosa_pool_kernel": "POOL",
        "void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(...)": "SELECT",
        "void at::native::bitonicSortKVInPlace<2, -1, 16, 16, float, long>": "SELECT",
        "flash_h2d_from_mask_bias_kernel": "H2D",
        "flash_h2d_from_mask_kernel": "H2D",
        "diff_kernel": "DIFF",
        "_wb_kernel": "WB",
        "void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat16>": "FUSED",
        "void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheKernel<true, 128u>": "FUSED",
        "void vllm::act_and_mul_kernel<c10::BFloat16, &silu_kernel>": "FUSED",
        "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<long>>": "SMALL",
        "void at::native::index_elementwise_kernel<128, 4, ...>": "SMALL",
        "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, MeanOps>>": "SMALL",
        "void at::native::(anonymous namespace)::CatArrayBatchedCopy<int, unsigned int, 3, 64, 64>": "SMALL",
        "some_new_kernel": "OTHER",
    }
    for name, cls in cases.items():
        assert P.classify(name) == cls, (name, P.classify(name), cls)
    assert P.classify("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy") == "COPY"
    assert P.classify("Memset (Device)", "gpu_memset") == "COPY"
    # the names of run 2190876 (summary.json, cut at 200 characters as recorded): the flash kernels' template
    # arguments name cutlass::bfloat16_t; cuBLAS's split-K reduction stays GEMM
    for name, cls in REAL_2190876.items():
        assert P.classify(name) == cls, (name, P.classify(name), cls)
    # NEGATIVE CONTROL: the rules of run 2190876 (GEMM first, the order of fca556f) put the three flash kernels in GEMM
    # (Q4 REFUTED with ATTN 0 / 0); the rules argument reproduces a run's own classes
    old = {n: P.classify(n, rules=RULES_2190876) for n in REAL_2190876}
    assert [n for n, c in old.items() if c != REAL_2190876[n]] == [n for n in REAL_2190876 if "flash_fwd" in n]
    assert set(old[n] for n in REAL_2190876 if "flash_fwd" in n) == {"GEMM"}


RULES_2190876 = (                                     # summary.json meta.class_rules of run 2190876 (fca556f)
    ("GEMM", r"gemm|gemv|xmma|cutlass|cublas"),
    ("STAGE1", r"stage1|infllm"),
    ("ATTN", r"flash_fwd|flash_attn|splitkv"),
    ("POOL", r"pool"),
    ("SELECT", r"topk|sort|radix"),
    ("H2D", r"h2d"),
    ("DIFF", r"diff"),
    ("WB", r"_wb_kernel|writeback"),
    ("FUSED", r"rmsnorm|rotary|rope|act_and_mul|silu|nosa_linear"),
    ("SMALL", r"elementwise|index|reduce|catarray|copy|fill|where|gather|scatter|arange|clamp|compare"),
)
REAL_2190876 = {
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t, "
    "Flash_kernel_traits<128, 64, 128, 4, cutlass::bfloat16_t> >, false, false, false, false, true,": "ATTN",
    "void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t, "
    "Flash_kernel_traits<128, 64, 128, 4, cutlass::bfloat16_t> >, 4, 2, true>(Flash_fwd_par": "ATTN",
    "void flash_fwd_splitkv_stage1_kernel<Flash_fwd_kernel_traits<128, 16, 64, 1, false, false, cutlass::bfloat16_t, "
    "Flash_kernel_traits<128, 16, 64, 1, cutlass::bfloat16_t> >, false, false, false, false, ": "STAGE1",
    "void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, true, false, false>"
    "(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, __nv_bfloat16*,": "GEMM",
    "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn": "GEMM",
    "ampere_s16816gemm_bf16_128x64_ldg8_stages_64x4_tn": "GEMM",
    "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn": "GEMM",
    "void at::native::sbtopk::gatherTopK<c10::BFloat16, unsigned int, 2, false>(at::cuda::detail::TensorInfo<c10::"
    "BFloat16 const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int": "SELECT",
    "nosa_pool_kernel_per_request": "POOL",
    "flash_h2d_from_mask_bias_kernel": "H2D",
    "void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, true>"
    "(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, at::cuda::detail::TensorInfo": "SMALL",
}


def _x(cat, name, ts, dur, tid=1, pid=1, corr=None):
    e = dict(ph="X", cat=cat, name=name, ts=ts, dur=dur, tid=tid, pid=pid)
    if corr is not None:
        e["args"] = dict(correlation=corr)
    return e


def _trace():
    ev = [
        _x("user_annotation", "ProfilerStep#7", 0, 1000),                 # not a sh. / r2. range: ignored
        _x("user_annotation", "r2.twin", 100, 100),
        _x("user_annotation", "r2.body", 300, 100),
        _x("user_annotation", "r2.stage1", 320, 20),                      # nested in r2.body: the innermost wins
        _x("user_annotation", "sh.kv", 100, 100, tid=2),                  # another thread
        _x("cuda_runtime", "cudaLaunchKernel", 110, 2, corr=1),            # in r2.twin
        _x("cuda_runtime", "cudaLaunchKernel", 150, 2, corr=2),            # in r2.twin
        _x("cuda_driver", "cuLaunchKernel", 310, 2, corr=3),               # Triton, in r2.body
        _x("cuda_runtime", "cudaLaunchKernel", 325, 2, corr=4),            # in r2.stage1 (inside r2.body)
        _x("cuda_runtime", "cudaLaunchKernel", 500, 2, corr=5),            # outside every range
        _x("cuda_runtime", "cudaGraphLaunch", 600, 5, corr=6),             # a graph replay: 2 kernels
        _x("cuda_runtime", "cudaMemcpyAsync", 120, 2, corr=7),             # in r2.twin
        _x("cuda_runtime", "cudaLaunchKernel", 150, 2, tid=1, pid=1, corr=8),
        _x("kernel", "void at::native::vectorized_elementwise_kernel<4>", 1000, 3.0, tid=7, pid=0, corr=1),
        _x("kernel", "void at::native::reduce_kernel<512>", 1010, 5.0, tid=7, pid=0, corr=2),
        _x("kernel", "flash_h2d_from_mask_kernel", 1020, 7.0, tid=7, pid=0, corr=3),
        _x("kernel", "void flash::flash_fwd_splitkv_stage1_kernel<T>", 1030, 11.0, tid=7, pid=0, corr=4),
        _x("kernel", "ampere_bf16_s16816gemm_bf16_128x64", 1040, 13.0, tid=7, pid=0, corr=5),
        _x("kernel", "diff_kernel", 1050, 2.0, tid=7, pid=0, corr=6),
        _x("kernel", "_wb_kernel", 1060, 4.0, tid=7, pid=0, corr=6),
        _x("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", 1070, 1.0, tid=8, pid=0, corr=7),
        _x("kernel", "orphan_kernel", 1080, 1.0, tid=7, pid=0, corr=99),   # no launch record
        _x("kernel", "nosa_pool_kernel", 1090, 6.0, tid=7, pid=0, corr=8),
    ]
    ev.append(dict(ph="i", cat="instant", name="noise", ts=5))
    return dict(traceEvents=ev)


def test_P2_parse_trace_attribution():
    ops = P.parse_trace(_trace())
    got = {o["name"]: (o["cls"], o["range"], o["dur_us"]) for o in ops}
    assert len(ops) == 10
    assert got["void at::native::vectorized_elementwise_kernel<4>"] == ("SMALL", "r2.twin", 3.0)
    assert got["void at::native::reduce_kernel<512>"] == ("SMALL", "r2.twin", 5.0)
    assert got["flash_h2d_from_mask_kernel"] == ("H2D", "r2.body", 7.0)                    # a driver launch
    assert got["void flash::flash_fwd_splitkv_stage1_kernel<T>"] == ("STAGE1", "r2.stage1", 11.0)   # innermost
    assert got["Memcpy HtoD (Pinned -> Device)"] == ("COPY", "r2.twin", 1.0)
    assert got["nosa_pool_kernel"] == ("POOL", "r2.twin", 6.0)
    # NEGATIVE CONTROLS: no range for a launch outside the ranges, a graph launch, a kernel without a launch record
    assert got["ampere_bf16_s16816gemm_bf16_128x64"] == ("GEMM", None, 13.0)
    assert got["diff_kernel"] == ("DIFF", None, 2.0) and got["_wb_kernel"] == ("WB", None, 4.0)
    assert got["orphan_kernel"][1] is None
    # the range INSTANCE, the device start and the correlation id (the offline sequence analysis)
    by = {o["name"]: o for o in ops}
    assert by["void flash::flash_fwd_splitkv_stage1_kernel<T>"]["rid"] == (1, 1, 320.0)
    assert by["void at::native::reduce_kernel<512>"]["rid"] == by["nosa_pool_kernel"]["rid"] == (1, 1, 100.0)
    assert by["diff_kernel"]["rid"] is None and by["orphan_kernel"]["rid"] is None
    assert by["diff_kernel"]["corr"] == by["_wb_kernel"]["corr"] == 6 and by["_wb_kernel"]["ts"] == 1060.0
    # the range of ANOTHER thread does not capture tid 1's launches; a range without the prefixes captures nothing
    tr = _trace()
    for e in tr["traceEvents"]:
        if e.get("name") == "r2.twin":
            e["tid"] = 3
    got2 = {o["name"]: o["range"] for o in P.parse_trace(tr)}
    assert got2["void at::native::reduce_kernel<512>"] is None
    tr = _trace()
    for e in tr["traceEvents"]:
        if e.get("name") == "r2.body":
            e["name"] = "body"
    got3 = {o["name"]: o["range"] for o in P.parse_trace(tr)}
    assert got3["flash_h2d_from_mask_kernel"] is None and got3["void flash::flash_fwd_splitkv_stage1_kernel<T>"] == "r2.stage1"
    assert P.parse_trace({"traceEvents": []}) == [] and P.parse_trace([]) == []


def test_P3_aggregate_per_step():
    agg = P.aggregate(P.parse_trace(_trace()), steps=2)
    # 10 GPU ops of 3 + 5 + 7 + 11 + 13 + 2 + 4 + 1 + 1 + 6 = 53 us over 2 steps
    assert agg["total"] == dict(kernels=5.0, ms=round(53.0 / 1e3 / 2, 4))
    assert agg["classes"]["SMALL"] == dict(kernels=1.0, ms=round(8.0 / 1e3 / 2, 4))
    # r2.twin: the two small kernels, the memcpy and the pooling kernel (3 + 5 + 1 + 6 us)
    assert agg["ranges"]["r2.twin"] == dict(kernels=2.0, ms=round(15.0 / 1e3 / 2, 4))
    assert agg["ranges"]["-"] == dict(kernels=2.0, ms=round(20.0 / 1e3 / 2, 4))
    assert agg["names"][0]["name"] == "ampere_bf16_s16816gemm_bf16_128x64" and agg["names"][0]["cls"] == "GEMM"
    assert [n["ms"] for n in agg["names"]] == sorted((n["ms"] for n in agg["names"]), reverse=True)


def _agg(kernels, ms, classes, ranges):
    return dict(steps=16, total=dict(kernels=kernels, ms=ms), names=[],
                classes={k: dict(kernels=0.0, ms=v) for k, v in classes.items()},
                ranges={k: dict(kernels=n, ms=v) for k, (n, v) in ranges.items()})


def _good():
    sh = _agg(4000.0, 46.0, dict(GEMM=20.0, ATTN=8.0, STAGE1=2.0, POOL=1.0, SELECT=1.0, SMALL=6.0),
              {"sh.k": (100.0, 0.3), "sh.cis": (100.0, 0.3), "sh.kv": (300.0, 1.0)})
    r2g = _agg(7200.0, 52.0, dict(GEMM=20.4, ATTN=8.2, STAGE1=2.3, POOL=1.2, SELECT=1.0, SMALL=12.0), {})
    r2e = _agg(7300.0, 52.5, dict(GEMM=20.4, ATTN=8.2, STAGE1=2.3, POOL=1.2, SELECT=1.0, SMALL=12.0),
               {"r2.twin": (2300.0, 4.0), "r2.body": (1500.0, 2.5), "r2.stage1": (32.0, 2.3), "r2.pool": (32.0, 1.2)})
    return dict(SH=sh, R2G=r2g, R2E=r2e)


def _v(agg):
    return {n: v for n, v, _ in P.q_verdicts(agg)}


def test_P4_q_verdicts():
    assert _v(_good()) == {q: "CONFIRMED" for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")}
    breaks = [("Q1", lambda a: a["R2G"]["total"].update(kernels=5000.0)),
              ("Q2", lambda a: a["R2E"]["ranges"]["r2.twin"].update(ms=2.0)),
              ("Q2", lambda a: a["SH"]["ranges"]["sh.cis"].update(ms=0.9)),
              ("Q3", lambda a: a["R2E"]["ranges"]["r2.body"].update(ms=1.5)),
              ("Q4", lambda a: a["R2G"]["classes"]["GEMM"].update(ms=22.0)),
              ("Q5", lambda a: a["R2G"]["classes"]["POOL"].update(ms=3.0)),
              ("Q6", lambda a: a["R2G"]["total"].update(ms=55.0))]
    for q, brk in breaks:
        a = copy.deepcopy(_good())
        brk(a)
        v = _v(a)
        assert v[q] == "REFUTED", (q, v)
        assert sum(x == "REFUTED" for x in v.values()) == 1, (q, v)
    a = copy.deepcopy(_good())                                     # the graph's kernels not seen: R2E is R2's side
    a["R2G"]["total"].update(kernels=0.0)
    det = {n: d for n, _, d in P.q_verdicts(a)}
    assert _v(a)["Q1"] == "CONFIRMED" and "R2E" in det["Q1"]
    a = copy.deepcopy(_good())
    a["SH"]["total"].update(kernels=0.0)
    assert set(_v(a).values()) == {"UNDETERMINED"}
    assert set(_v(dict(SH=_good()["SH"])).values()) == {"UNDETERMINED"}


def test_P5_windows():
    wins = [P.SH_TIMED, P.SH_PROF]
    assert all(b - a == P.N_WIN == 16 for a, b in wins) and P.SH_TIMED[1] == P.SH_PROF[0]
    r2 = [P.R2G_TIMED, P.R2G_PROF, P.R2E_TIMED, P.R2E_PROF]
    assert all(b - a == 16 for a, b in r2) and all(r2[i][1] == r2[i + 1][0] for i in range(3))
    assert P.R2_WARM < P.R2G_TIMED[0] and P.R2E_PROF[1] == P.R2_STEPS and P.SH_STEPS == P.R2_STEPS + 1
    rew = [t for t, _ in BC.expected_rewarms(BC.S0, P.SH_STEPS)]
    assert rew == [0, 64] and not [t for t in rew if P.SH_TIMED[0] <= t < P.SH_PROF[1]]
    a = P.parse_args(["--out", "x", "--workload", "w"])
    assert a.batch == 64
    with pytest.raises(SystemExit):
        P.parse_args(["--out", "x", "--workload", "w", "--batch", "0"])


def _op(name, ts, dur, rng=None, rid=None, cls=None, cat="kernel"):
    return dict(name=name, cat=cat, cls=cls or P.classify(name, cat), dur_us=float(dur), range=rng, ts=float(ts),
                rid=rid, corr=None)


def test_P6_offline_reanalysis_pure_parts():
    import r2_p3_reclass as R
    flash = next(n for n in REAL_2190876 if n.startswith("void flash_fwd_splitkv_kernel"))
    # rules_of / reclassified: the recorded rules (JSON lists) reproduce the recorded class
    summ = dict(meta=dict(class_rules=[list(r) for r in RULES_2190876]))
    assert R.rules_of(summ) == RULES_2190876
    ops = [_op(flash, 0, 5.0), _op("ampere_s16816gemm_bf16_128x64_ldg8_stages_64x4_tn", 10, 3.0)]
    assert [o["cls"] for o in ops] == ["ATTN", "GEMM"]
    assert [o["cls"] for o in R.reclassified(ops, RULES_2190876)] == ["GEMM", "GEMM"]
    assert [o["cls"] for o in ops] == ["ATTN", "GEMM"]                                   # a copy: ops unchanged
    # same_profile
    agg = P.aggregate(P.parse_trace(_trace()), steps=2)
    assert R.same_profile(agg, copy.deepcopy(agg)) == []
    bad = copy.deepcopy(agg)
    bad["classes"]["SMALL"]["ms"] += 0.001
    assert R.same_profile(agg, bad) == ["classes SMALL ms: %s / %s" % (agg["classes"]["SMALL"]["ms"],
                                                                         bad["classes"]["SMALL"]["ms"])]
    bad = copy.deepcopy(agg)
    del bad["ranges"]["r2.twin"]
    assert len(R.same_profile(agg, bad)) == 1 and R.same_profile(agg, bad)[0].startswith("ranges r2.twin:")
    # name_delta
    a = {"x": [2, 30.0, "SMALL"], "y": [1, 10.0, "H2D"]}
    b = {"y": [1, 4.0, "H2D"], "z": [1, 1.0, "COPY"]}
    d = R.name_delta(a, b, steps=1)
    assert [(r["name"], r["kernels"], r["ms"]) for r in d] == [("x", 2.0, 0.03), ("y", 0.0, 0.006), ("z", -1.0, -0.001)]
    # seq_variants: two instances of one sequence, one shorter instance; device order inside an instance
    body = [_op("q", 11, 3.0, "r2.body", "i1"), _op("p", 10, 1.0, "r2.body", "i1"),
            _op("p", 20, 3.0, "r2.body", "i2"), _op("q", 21, 5.0, "r2.body", "i2"),
            _op("p", 30, 7.0, "r2.body", "i3"), _op("p", 40, 9.0, "r2.twin", "i4")]
    sv = R.seq_variants(body, "r2.body", 3)
    assert (sv["instances"], sv["variants"]) == (3, 2)
    top = sv["top"][0]
    assert (top["instances"], top["ops"], top["median_us"]) == (2, 2, 6.0)
    assert [(s["name"], s["us"]) for s in top["seq"]] == [("p", 2.0), ("q", 4.0)]
    assert sv["top"][1]["instances"] == 1 and sv["top"][1]["seq"][0]["us"] == 7.0
    # cut_steps / transfer_ranges
    emb = "void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::BFloat16, long>"

    def eager_steps(n):
        out, t = [], 0
        for _ in range(n):
            for name, rng in ((emb, None), ("tw", "r2.twin"), ("bd", "r2.body"), ("mm", None)):
                out.append(_op(name, t, 1.0, rng, rid=(1, 1, t) if rng else None))
                t += 1
        return out

    def graph_steps(n, extra=False, pre=False):
        out, t = ([_op("cp", -5, 1.0)] if pre else []), 0
        for _ in range(n):
            names = [emb, "tw", "bd"] + (["xx"] if extra else []) + ["mm"]
            for name in names:
                out.append(_op(name, t, 2.0))
                t += 1
        return out
    steps_, before = R.cut_steps(graph_steps(2, pre=True))
    assert len(steps_) == 2 and [o["name"] for o in before] == ["cp"] and len(steps_[0]) == 4
    st, det, g = R.transfer_ranges(graph_steps(2, pre=True), eager_steps(2), 2)
    assert st == "OK" and "matched 8 of 8" in det
    assert [o["range"] for o in g] == ["?pre"] + ["-", "r2.twin", "r2.body", "-"] * 2
    gagg = P.aggregate(g, 2)
    assert gagg["ranges"]["r2.body"] == dict(kernels=1.0, ms=round(2.0 / 1e3, 4))
    # NEGATIVE CONTROLS: an extra graph kernel per step (8 of 10 matched < MATCH_MIN), a wrong step count
    st, det, g = R.transfer_ranges(graph_steps(2, extra=True), eager_steps(2), 2)
    assert st == "UNDETERMINED" and "matched 8 of 10" in det
    assert [o["range"] for o in g][:5] == ["-", "r2.twin", "r2.body", "?unmatched", "-"]
    st, det, g = R.transfer_ranges(graph_steps(2), eager_steps(3), 2)
    assert st == "UNDETERMINED" and g is None and "R2E 3" in det
    assert R.cut_steps([_op("tw", 0, 1.0)]) == ([], [_op("tw", 0, 1.0)])
    with pytest.raises(SystemExit):
        R.parse_args(["--run", "r", "--out", "o", "--top", "0"])
