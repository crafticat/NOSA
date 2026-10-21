# R2 shared ready-KV reads, part 4: fixes after the review of 9a501d8 (SOURCE ONLY, NOT RUN)

Branch `r2-sharedkv4`, worktree `/mnt/beegfs/ojerbi/nosi-r2-sharedkv4`, base `9a501d8` (frozen: its six torch arms
are queued). Review: `kstep-cpu-runs/paper-notes/review-sharedkv3-9a501d8.md`. No CPU test and no GPU cell ran.

| finding | fix | where |
|---|---|---|
| MAJOR-1 no GPU check fused == build_union | the fused probe runs build_union on the SAME inputs of EVERY keyed call (untimed, after the profiler window), counts differing elements per field on the device, reads them once (`union_checked`, `union_diff`); the recorded tables keep the fused union. MUST `SKV_fused_union_equals_torch` (gate, fused probe arms) and `O_cell_fused_union_equals_torch` (S3 driver, fused cells) | SK `_check_union`, probe body, `report`, `report_tables`; GATE `sharedkv_gates`; NCU `mode_summarize` |
| MINOR-1 interpreter root cause | DIAGNOSTIC node `test_D1_...`: a360d86's masked [TE, SP] tl.min / tl.max on the U7 hand case in variants (exact construct TE 2 / 8; without the constexpr if; one tile; the int32 reduction alone); it prints and warns a `REPRO {cause, variants}` line naming the first failing layer | test_r2_sharedkv4.py |
| MINOR-2 empty-group programs do O(N^2) | pass 0 writes the empty group's final outputs (ut / rowpos / uidx -1, ucnt / win 0); `r0 = GROW[g, 0]`; passes A, B, C only under `if r0 >= 0:` (uniform over the CTA; the 3 barriers inside) | SK `_union_kernel` |
| MINOR-3 repeated (key, slot) within a row | every selector arm (probe* too) refuses `--no-check`; the precondition is stated in the kernel's docstring | GATE `parse_args`; SK |
| MINOR-4 fused band vs F5'; whole fused band without an arm | split fused band [0.60, 1.10] (its top IS F5'); the whole fused band is withdrawn (no arm: "no band registered") | GATE `SKV_PRED`; NCU `S3_PRED`, `band`; DOC3 section 5 |
| MINOR-5 what the build range holds | selector counter `union_builds` {torch, fused}; MUST `SKV_union_build_kind` (the requested build ran, the other never); S3: per marked call the `_union_kernel` count and other kernels in `bd`, MUST `O_build_kernel_<cc>` (exactly 1 per call) in fused cells; the fused build band uses only the calls whose build range holds ONLY the union kernel (`build_us_union_only`: not a tick's first keyed layer) | SK `_build`; GATE; NCU |
| MINOR-7 claim hygiene | both steps: existence check BEFORE the claim; an owner file in the claim; on 124 / 137 a bounded wait (10 min) for the holder's steps, no freeze of an output whose writer may run, the claim kept; the release removes the owner file. NEW run ids: the fused S1 arms use `s1f` (not s1a) | r2_sharedkv_step.sh, r2_sharedkv_ncu_step.sh |
| MINOR-8 doc drift | DOC3 U3 row (token cache; the in-place guard is gone), U6 (3 barriers), SK comment (token cache) | DOC3, SK |
| MINOR-9 supplied union unchecked | `bi_paged_decode_shared` refuses a supplied union whose grow / ut / rowpos / win are not contiguous int32 on q's device | SK |
| MINOR-11 bf16 under the interpreter | one bf16 case: shared kernel = piece 3's kernel, both schedules (skips with the reason if the interpreter cannot run piece 3's bf16 kernel) | test_r2_sharedkv4.py |

Not addressed here (not in the lead's list): MINOR-6 (the per_row miniature inside the chain), MINOR-10 (the CPU
log's skip reasons): the lead's chain / wrapper.

## CPU nodes

`benchmarks/Efficiency/tests_cpu/test_r2_sharedkv4.py`: `test_F1_probe_checks_the_fused_union_against_build_union`,
`test_F2_tables_keep_the_fused_union`, `test_D1_interpreter_repro_of_the_a360d86_window_reduction` (diagnostic),
`test_B1_bf16_shared_kernel_under_the_interpreter`, `test_S1_union_kernel_exits_early_for_empty_groups`,
`test_S2_step_claim_hygiene[r2_sharedkv_step.sh-skv_active]`, `test_S2_step_claim_hygiene[r2_sharedkv_ncu_step.sh-skv2_active]`.
Changed: `test_bi_paged_attention_shared.py::test_gate_selector_default_is_piece3_and_choices` (probe* refuse
--no-check; the two new MUST gates with positive and negative fakes), `::test_step_dryrun_refusals_and_pins` and
`test_r2_sharedkv_ncu.py::test_N5_step_dryrun_refusals_and_pins` (NOSI sharedkv4). Rerun:
`test_r2_sharedkv3_union.py` (the kernel changed: U5 / U7 cover the empty groups and the windows),
`test_r2_sharedkv2_keys.py`, `test_r2_sharedkv_ncu.py`, `test_bi_paged_attention_shared.py`.

## The fused S1 cell (holder step, after the CPU suite passes at this commit)

    bash benchmarks/Efficiency/r2_sharedkv_step.sh --nosi <sharedkv4 sha> <holder> s1f probe-plant-fused
    bash benchmarks/Efficiency/r2_sharedkv_step.sh --nosi <sharedkv4 sha> <holder> s1f probe-fused
    bash benchmarks/Efficiency/r2_sharedkv_step.sh --nosi <sharedkv4 sha> <holder> s1f shared-fused

in this order, stopping at the first nonzero exit (probe-fused is VOID unless probe-plant-fused passed). MUST in the
probe arms: SKV_fused_union_equals_torch, SKV_union_build_kind, SKV_bitwise_shared_equals_per_row (probe) /
SKV_plant_detected (plant), SKV_rows_max4_calls, SKV_tile_identity.
