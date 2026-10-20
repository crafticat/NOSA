# R2 capture audit: the decode step against AGENTS.md law 3

Branch `feature/nosi-r2-m2-capture` (from `feature/nosi-r2-m2-wb` 257d7b3 =
the integration line 54ea1d6 + the device write-back). Status: SOURCE ONLY.
Nothing here comes from a GPU run.

THE LAW (retroinfer-eval `AGENTS.md`, law 3). Inside a captured region:

- fixed shapes;
- fixed device addresses, with in-place writes only;
- no allocation;
- no host branch on a device value;
- no `tensor[scalar_tensor]` indexing;
- the region cannot raise.

THE REGION. C's step-7 tick (`docs/R2_STEP7_SPEC_ROUND.md` in
`nosi-r2-m2-spec`, f1f9f56, section 2.4) will be captured. Per layer it runs the
positional tables (`PositionalTwin` semantics per bank), stage 1, the pooling,
the top-k, the body's tail write / selection / gathers / write-back, attention,
FFN. Today those steps are `r2_decode_path.layer_decode` and its callees. This
audit covers them: `r2_decode_path.py`, `r2_positional_twin.py`,
`r2_decode_body.py`, `stage1_seqused.py`, `max_pooling_per_request.py`,
`r2_writeback.py`.

METHOD. Read every line, then a source lint (`tests/test_r2_capture.py`) that
finds each violation class per function. CAPTURED functions must be clean. The
validated EAGER functions must show at least the classes listed below, so this
document cannot go stale silently. Planted violations are found (positive
controls).

## Violation classes (the lint's)

| Class | What it catches | Why it breaks capture |
|---|---|---|
| SYNC | `bool()`, `float()`, `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, `synchronize()`, `int()` of a non-static value | a host read of a device value: a sync, and frozen at capture |
| SHAPE | `.nonzero()`, `.unique()`, `.masked_select()`, `.argwhere()`, boolean-mask indexing | data-dependent shape: the captured kernel keeps the capture's shape |
| BRANCH | `if` / `while` / `assert` / ternary whose test calls more than static metadata | the branch taken at capture is replayed forever |
| ALLOC | torch factories, `.clone()` | a new address per call; the replay keeps the capture's |
| SCALARIDX | `x[t]` with a 0-d tensor `t` | a hidden sync |
| HOSTSTATE | `+=` and attribute rebinding | host state that the replay does not update |
| CALLS_EAGER | `.step(`, `.decode(`, `.advance(`, `.update(`, `pooling_lengths(`, `body_update(`, `layer_decode(`, `decode_inference(` | a call into a function that is not capture-safe |

A `raise` after a static check (shapes, dtypes, devices) runs only when the
region is captured, never at replay. It is allowed. A value-dependent raise
is SYNC + BRANCH.

## Table A: violations in the validated eager path

Lines are those of this tree. "X1" .. "X3" are fixed in this commit; "X4" is
planned (next section); "OP" needs a decision.

| File:line | Function | Class | What | Fix |
|---|---|---|---|---|
| `r2_positional_twin.py:222` | `PositionalTwin.step` | ALLOC | `torch.ones` (default `active`) | X1: `active` is required |
| `r2_positional_twin.py:223` | `step` | ALLOC | `torch.as_tensor(active)` | X1: dtype / device checked statically |
| `r2_positional_twin.py:226` | `step` | ALLOC | `written = torch.full(...)` | X1: returns None |
| `r2_positional_twin.py:227` | `step` | SHAPE (+ sync) | `active.nonzero()` | X1: every request runs; masks |
| `r2_positional_twin.py:228` | `step` | BRANCH | `if b.numel() == 0` | X1 |
| `r2_positional_twin.py:230-231` | `step` | SYNC, BRANCH | `bool(prefilled[b].all())` -> raise | X1: bit `ERR_NO_PREFILL` in `twin.err`, the step changes nothing |
| `r2_positional_twin.py:238-240` | `step` | SYNC, BRANCH, SHAPE | capacity checks -> raise | X1: bits `ERR_N_MAX`, `ERR_LK_MAX`, the step changes nothing |
| `r2_positional_twin.py:250` | `step` | ALLOC | `torch.arange(B)` | X1: `self._all_b`, allocated in `__init__` |
| `r2_positional_twin.py:252-254`, `:263-265` | `step` | SHAPE, BRANCH | `b[due_k]`, `n_k[due_k]`, `if bk.numel()` | X1: `torch.where` over all B rows, clamped indices |
| `r2_decode_body.py:175-176` | `DecodeBody.decode` | SYNC, BRANCH | a host-mirror range check -> `IndexError` (no device sync, but it would run only at capture) | X3: bit `ERR_POS` in the caller's `err`, the step changes nothing |
| `r2_decode_body.py:272-273` | `DecodeBody.writeback` | SHAPE, SYNC | host loop over `full_host.nonzero().tolist()` | X3: `decode_fixed` needs `DeviceWritebackBody` (`r2_writeback.masked_writeback`) |
| `r2_decode_body.py:249` (was `torch.arange` per call at 257d7b3) | `DecodeBody.write_tail` | ALLOC | the row index | X2: `_index()` cache, filled by the eager warm-up |
| `r2_decode_path.py:312` | `layer_decode` | ALLOC | `pos_before = twin.pos.clone()` | X4: a static buffer, `copy_` |
| `r2_decode_path.py:313` | `layer_decode` | CALLS_EAGER | `twin.step` | X4: `twin.step_fixed` |
| `r2_decode_path.py:314`, `:118`, `:121-122` | `layer_decode`, `pooling_lengths` | CALLS_EAGER, ALLOC, BRANCH, HOSTSTATE | lengths by `.clone()`; negative-control hooks keyed on the host step counter | X4: static int32 `n_used` / `M_b`, `copy_`; the hooks stay EAGER-only (gates) |
| `r2_decode_path.py:315-316` | `layer_decode` | HOSTSTATE, ALLOC | `st.M0` set on the first step | X4: not in the captured step (the "frozen" control is eager) |
| `r2_decode_path.py:327` | `layer_decode` | ALLOC (+ rebinding) | `st.topk_log[l] = topk_idx.clone()` | X4: a static `(L, H, B, 64)` log, `copy_` |
| `r2_decode_path.py:328`, `:289`, `:292`, `:155` | `layer_decode`, `body_update`, `R2Body.update` | CALLS_EAGER, SYNC, BRANCH | the adapter calls `decode`; body "shipped" checks `bool(st.active.all())` | X4: body "r2" only, `decode_fixed` (the shipped body is not capturable: host `tail_full` branch, `cache_engine.py:359`, `:401`) |
| `r2_decode_path.py:336` | `layer_decode` | ALLOC | `out = torch.empty(...)` (FFN) | X4: a static buffer |
| `r2_decode_path.py:354-355` | `decode_inference` | HOSTSTATE | `st.steps += 1`, `st.pos_host += ...` | X4: the driver, after each replay |
| `r2_decode_path.py:223` | `R2State.set_active` | (sync) | pageable host -> device copy | outside the region: the driver writes the static mask before a replay |

## Table B: op-internal allocations (DECISION NEEDED: "OP")

These allocate inside an op. Under `torch.cuda.graph` they come from the
graph's private pool and keep their addresses across replays. Law 3 read
strictly forbids them. Either the law accepts graph-pool allocations, with the
capture gate proving captured == eager bitwise, or each needs an `out=`
variant.

| Site | Allocation | out= available |
|---|---|---|
| `r2_decode_path.py:303`, `:334`, final norm | flashinfer `rmsnorm` output | NOT CHECKED (the NL wrapper `nosa_llama.py:47` passes none) |
| `r2_decode_path.py:305` -> `nosa_linear.py:102-105` | q, k, v, cis `torch.empty` | no (shipped op; an `out=` would be a small change) |
| `r2_decode_path.py:318` -> `stage1_seqused.py:83`, `:85`, IV2 `csrc/flash_attn/flash_api.cpp:936` | a reshaped q copy, `cu_q * groups`, the score `torch::full` | the score: no (a C++ change); the other two: precomputable per run |
| `r2_decode_path.py:329` | attention output | NOT CHECKED (`flash_attn_nosa` wrapper) |
| `F.linear` (wqkv, wo, gate_up, down, lm_head) | outputs | `torch.matmul(out=)` |
| `step_fixed`, `decode_fixed`, `write_tail`, `select` | `torch.where` / arithmetic temporaries | in-place rewrites are possible but change nothing numerically |

## Table C: the capture-safe functions (the lint finds nothing)

`PositionalTwin.step_fixed`, `.key_window`, `._key_due`, `._cis_due`;
`DecodeBody.decode_fixed`, `.write_tail`, `.select`, `.full_mask`,
`.cache_lens_value`, `.positions`; `DeviceWritebackBody.writeback`;
`r2_writeback.masked_writeback`, `._launch`, `.check_args` (the pinned check
is skipped while capturing); `max_pooling_per_request.nosa_pooling_per_request`,
`.check_lengths`; `stage1_seqused.stage1_seqused`, `.check_seqused`.
`DecodeBody._index` is not in the list on purpose: it allocates once, on the
first call, which the eager warm-up makes before the capture.

## The fixes in this commit (no numerical change)

* X1 `PositionalTwin.step_fixed(keys, cis, active)`:
  - Every request runs every operation. A request that writes nothing rewrites
    its own old values (`torch.where`; clamped indices keep the reads in range).
  - The key and cis means are computed over all B requests, as `step()` already
    does since step 6.
  - A step that `step()` refuses changes nothing and ORs a bit into `twin.err`.
    `check_error()` (one sync) raises the same exception later.
  - `_all_b` and `err` are allocated in `__init__`.
  - Proof:
    - `tests/test_r2_capture.py` X1: every table equal to `step()` after every
      step, 250 ragged steps, integer and normal data.
    - `tests/test_r2_positional_twin.py` `*_step_fixed*`: gate 1 against the
      shipped tables, S % 16 in {0, 1, 15} at B = 1 and 4, plus bf16; gate 2
      ragged with rollbacks; the capacity refusals.
    - `step()` itself is unchanged: C's R1 / R7 pin its text.
* X2 `DecodeBody._index`: the row index of `write_tail` is cached per
  (B, device), not allocated per call. It is the same tensor values; every
  existing body test covers it.
* X3 `DecodeBody.decode_fixed(eng, ..., pos, active, err, kernels)`:
  - The same writes in the same order as `decode()`, without host mirrors.
  - It needs a device-driven write-back (`DeviceWritebackBody`; refused at
    capture time otherwise).
  - An out-of-range active request ORs `ERR_POS` into the caller's `err`, and
    the step changes nothing.
  - Proof: `tests/test_r2_decode_body.py` `*_fixed*`:
    - G-EQ against the AST-extracted shipped body, S % 64 in {0, 1, 63} at
      B = 1 and 4;
    - G-RAG against independent B = 1 engines (eager `decode`);
    - G-POISON;
    - the inactive request;
    - the cross-boundary rollback;
    - the device refusal.

## X4 and the capture gate (plan; the next milestone)

`nosi/nosi/r2_capture.py`, a new module (C's `r2_decode_path.py` is not
edited, so it cannot collide with step 7):

* `CaptureBuffers`, built once after `build_ragged`:
  - input ids (B, 1), positions (B, 1), the active mask (B,);
  - `pos_before` (B,) int64; `n_used`, `M_b` (B,) int32;
  - the top-k log (L, H, B, 64); the FFN output (B, 1, I); the logits (B, 1, V);
  - `err` int32 (0-d).
* `layer_decode_fixed`: `layer_decode` with X1 / X3 and the buffers above. Body
  "r2" with `DeviceWritebackBody(f4_fix, r2_writeback.masked_writeback)`.
  Negative-control hooks are not supported inside it.
* `decode_step_fixed`: the embedding from the static ids, the layers, the final
  norm, the lm_head, `logits.copy_`.
* The driver, per step:
  - write ids / positions / the active mask into the buffers;
  - replay;
  - then on the host: `pos_host += active_host`, `steps += 1`;
  - read `err` and every twin's `err` at the sync where it reads the logits.
* Warm-up: two EAGER `decode_step_fixed` steps on the same buffers before the
  capture. They compile every Triton kernel with the capture's specialisation
  and fill the `_index` cache. A warm-up step is a real step: the arms below
  count it.

Gate `benchmarks/Efficiency/r2_capture_gate.py` (to write; one process, the
model loaded once, `build_ragged` per arm, the ragged gate's documents,
activity table and deltas):

| Arm | What | Gate |
|---|---|---|
| CE | the validated eager path (`layer_decode`, body "r2") | the reference |
| CF | eager `decode_step_fixed` | MUST: bitwise CE at every step: logits, every layer's selection, twin counters, `cache_lens`, the tail id; the final tables |
| CG | 2 eager warm-up steps, then ONE captured `decode_step_fixed`, replayed for the remaining steps with the ragged activity | MUST: bitwise CE at every step, as CF; every `err` 0 |
| NG1 | CG with `M_b` taken from a host value frozen at capture | MUST differ from CE, first at a request's first block boundary after the capture |
| NG2 | CG with the active mask not rewritten before a replay | MUST differ from CE, first at the first step whose activity differs |
| T | step time CF vs CG | REPORT |

Size: B 8, S0 16128, deltas 0,1,17,63,64,65,777,4097, 192 steps (every
request crosses at least two block boundaries), p-active 0.7, seed 6.

Step script (committed now; it STOPs until the gate file is tracked at the
pinned sha):

    DRYRUN=1 bash benchmarks/Efficiency/r2_capture_step.sh --nosi <sha> <holder job> <run id>
    bash benchmarks/Efficiency/r2_capture_step.sh --nosi <sha> <holder job> <run id>

It is the pattern of `r2_ragged_step.sh`:

- pins: the image, the model size, the workload sha256;
- the GPU uuid read from `/mnt/beegfs/ojerbi/kstep-cpu-runs/holders/<job>.gpu_uuid`
  before any claim and recorded in `command.txt`; the child still checks it
  with `nvidia-smi`;
- claim `/mnt/beegfs/ojerbi/r2-runs/capture/capture_active`;
- `STEP_MIN` 30.

## CPU command (inside a CPU job)

    env NOSI_REPO=/mnt/beegfs/ojerbi/nosi-r2-m2-capture python -m pytest -q tests/test_r2_capture.py tests/test_r2_positional_twin.py tests/test_r2_decode_body.py tests/test_r2_writeback.py tests/test_r2_engine_findings.py

and C's `benchmarks/Efficiency/tests_cpu` (R1 / R2 / R7 inspect and run the
unchanged `step()`).
