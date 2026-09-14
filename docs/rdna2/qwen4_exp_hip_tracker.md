# Qwen4Exp HIP path — work tracker

**Scope**: track what needs to be tested and optimized for the HIP
scaffolding landed 2026-09-14 (`hc_rdna2.cu`, `qsa_rdna2.cu`,
`ple_short_conv_rdna2.cu` + bindings + dispatchers + env-var gates).

The matching test specification lives at
[`qwen4_exp_hip_tests.md`](qwen4_exp_hip_tests.md); the matching
path overview lives at
[`qwen4_exp_hip_path.md`](qwen4_exp_hip_path.md). Update this tracker
whenever a kernel moves between stages.

## Status legend

- **[ ]** open
- **[~]** in progress (someone is looking at it now)
- **[x]** done
- **[!]** blocked / wrong — needs investigation before continuing

## Stage gates

| Stage | Description | Exit criteria |
|---|---|---|
| **S0 build** | Code compiles, ops registered | `test_rdna_hip_build_smoke` passes |
| **S1 parity** | HIP output matches Triton reference (fp16 noise) | Parity test passes for each kernel |
| **S2 dispatch** | Env-var gate works, dispatcher routes correctly | `test_*_env_gate` tests pass |
| **S3 integration** | Real forward pass completes without raising | `test_qwen4_exp_rdna_hip_smoke` passes |
| **S4 cudagraph** | Captured graph replays without fault | `test_vllm_serve_smoke` passes |
| **S5 bench** | HIP throughput ≥ Triton (or justified slower) | Bench numbers logged in `qwen4_exp_hip_path.md` |
| **S6 default-on** | Flip env-var default to "1" + remove `VLLM_DISABLED_KERNELS` parallel | All tests pass with default-on |

---

## HC prefill HIP

Code: `csrc/rocm/hc_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/hc_rdna2.py` · Gate:
`VLLM_RDNA_HC_PREFILL_HIP=1`

### hc_grouped_gemma_rmsnorm_rdna2

- [ ] **S0 build** — rebuild `_rocm_C.abi3.so`, run op-presence probe
- [ ] **S1 parity** — write
  `tests/kernels/rocm/qwen4_exp/test_hc_rdna2.py::test_grouped_gemma_rmsnorm_parity`
  against `torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm`. Cover
  `N ∈ {1, 8, 64, 1024, 8192}`, `DIM ∈ {512, 4096, 8192}`,
  `W_SHARED ∈ {True, False}`.
- [ ] **S2 dispatch** — `test_hc_dispatcher_env_gate` (§1.4 of test plan).
- [ ] **S3 integration** — runs in the Qwen4Exp PLE forward path with
  `VLLM_RDNA_HC_PREFILL_HIP=1`. Specifically check the pre-mix path in
  `hyperconnection.py:mix()` (the `_rdna_fused_ok(xn)` branch is
  currently gated on M <= 8 — verify prefill (M > 8) routes to the new
  Triton-or-HIP path).
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, NG=4)`. Target
  ≥ +20% over Triton.
- [ ] **S6 default-on** — once S1-S5 pass, flip env-var default.

### hc_silu_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — straightforward elementwise; `atol=1e-3` should
  hold.
- [ ] **S5 bench** — production shape `(B=2048, DIM=320)`. Silu + 1/HC
  scale is bandwidth-bound; expect +20-40% over Triton (vec8 loads,
  no Triton block overhead).
- [ ] **S6 default-on**

### hc_gate_mix_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — reduction over HC streams; `atol=5e-3`. Cover
  `HC ∈ {1, 2, 4, 8}` (template-specialized). Add `test_hc_unroll_factor`
  for `hc_count=3, 5` → expect `TORCH_CHECK` "hc_count in {1,2,4,8} only".
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`. The
  constexpr unroll over `HC` should win over Triton's `static_range`;
  target +20%.
- [ ] **S6 default-on**

### hc_combine_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — elementwise affine; `atol=1e-3`.
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`.
- [ ] **S6 default-on**

### hc_combine_norm_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — round-trip boundary test (§1.3 of test plan):
  verify `hc_combine + grouped_gemma_rmsnorm == hc_combine_norm(out, y)`
  elementwise.
- [ ] **S5 bench** — production shape `(B=2048, DIM=5120, HC=4)`.
  Single-pass over the data should win over two launches; target
  +30-40% over the `(hc_combine, rmsnorm)` Triton sequence.
- [ ] **S6 default-on**

---

## QSA decode HIP

Code: `csrc/rocm/qsa_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/qsa_rdna2.py` · Gate:
`VLLM_RDNA_QSA_HIP=1`

### qsa_store_cache_rows_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — pure scatter (no math). Cover `num_rows ∈ {1, 8,
  64, 1024}`, `WIDTH ∈ {32, 64, 128}`, `PAGE_SIZE ∈ {16, 784}`,
  `num_blocks ∈ {1, 4, 64}`.
- [ ] **S2 dispatch** — `test_qsa_dispatcher_env_gate`.
- [ ] **S3 integration** — runs in `Qwen4ExpQSAAttention`'s indexer
  write path. Verify shape conversion `[rows, 1, width]` → `[rows,
  width]` happens in the dispatcher (squeeze the head=1 dim).
- [ ] **S5 bench** — production shape `(B=512, WIDTH=128)`. Target
  ≥ +30% (pure scatter, vec8 loads).
- [ ] **S6 default-on**

### qsa_compress_groups_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — sum with raw/state switch + first-position tail.
  Cover `compress_ratio ∈ {1, 2, 4, 8, 16}`,
  `LOAD_ROPE_POSITIONS ∈ {False, True}`, all-NULL `state_table`,
  early-token `end_position < COMPRESS_RATIO - 1`.
- [ ] **S2 dispatch**
- [ ] **S3 integration**
- [ ] **S5 bench** — production shape `(B=512, head_dim=128, ratio=4)`.
  Reduction-over-K is the long pole; target ≥ +15%.
- [ ] **S6 default-on**

### qsa_mqa_paged_rdna2

- [ ] **S0 build**
- [ ] **[!]** **S1 parity** — current wrapper delegates to
  `paged_mqa_logits_decode_rdna2`. The wrapper signature
  (`q_fp16, kv_cache, weights, context_lens, block_tables,
  max_model_len`) does NOT match the Triton
  `qsa_mqa_paged(q, k_cache, page_table, token_to_req,
  query_positions, sequence_lengths, compress_ratio, num_columns,
  score_scale)`. Test reduced to a direct-GEMV sanity check (§2.4 of
  test plan) until the wrapper grows the missing args.
- [ ] **S3 integration** — once the wrapper signature matches,
  integrate with `qsa_mqa_paged` callers.
- [ ] **S5 bench** — once parity matches Triton; otherwise skip.
- [ ] **S6 default-on** — deferred until wrapper signature lands.

---

## PLE dilated short-conv HIP

Code: `csrc/rocm/ple_short_conv_rdna2.cu` · Dispatcher:
`vllm/models/qwen4_exp/amd/ops/ple_conv_rdna2.py` · Gate:
`VLLM_RDNA_PLE_CONV_HIP=1`

### ple_short_conv_decode_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — see §3.2 of test plan. Cover `B ∈ {1, 8, 64}`,
  `D ∈ {2560, 5120}`, `state_len ∈ {2, 4, 8}`, `dilation ∈ {1, 2}`,
  `silu ∈ {True, False}`, `has_init mix`, `null_block slots`.
- [ ] **S2 dispatch** — `test_ple_conv_env_gate`.
- [ ] **S3 integration** — already wired into
  `Qwen4ExpPLELayer._short_conv_dilated_decode_batched`. Smoke
  pass through the PLE forward path.
- [ ] **S5 bench** — production shape `(B=8, D=5120, K=5, dilation=2)`.
  Target ≥ +50% vs `F.conv1d` (depthwise over D=5120 channels with
  vec8 inner-loop, vs PyTorch's generic dilated conv path).
- [ ] **S6 default-on** — flip when S5 holds.

### ple_short_conv_prefill_rdna2

- [ ] **S0 build**
- [ ] **S1 parity** — see §3.3 of test plan. Cover `num_prefills ∈
  {1, 4, 8}`, `max_len ∈ {128}` (capped at 128 in current HIP build),
  state_len / dilation / silu matrices.
- [ ] **S3 integration** — wired into
  `_short_conv_dilated_prefill_batched`. Watch the early-return
  guards (`if ... pass  # fall through to torch`) — they should NOT
  trigger in normal traffic.
- [ ] **S5 bench** — `F.conv1d` is GPU-overhead-bound at small
  `max_len`; expect a smaller win than decode (target +20-30%).
- [ ] **S6 default-on**

---

## Open questions / blockers

- **Q1 (hc_combine_norm)**: the HIP wrapper writes `out` (post-combine)
  AND `y` (post-norm) in one launch. The Triton reference has the same
  contract. Verify on a checkpoint-loaded model that
  `hc_combine_norm(...).out` equals `hc_combine(...)` and `.y` equals
  `grouped_gemma_rmsnorm(out, w, eps, hc_count)`. If the model's next
  step is the `attn_hc.mix(...)` GEMV (the fused decode path), check
  that the `.y` produced here matches what `attn_hc._rdna_fused_ok`
  expects.
- **Q2 (ple_conv state write-back)**: the current decode HIP path uses
  `state[i] = history[dilation + i]` (left-shift by `dilation`). The
  eager torch reference uses `next_state = history[..., -state_len:]`.
  These differ when `dilation < state_len`. Pick the eager torch
  semantics as the source of truth and update the kernel.
- **Q3 (qsa_mqa_paged wrapper)**: signature gap noted in
  `qwen4_exp_hip_path.md` "Follow-up work" — needs a follow-up PR
  before S1 parity is meaningful here.
- **Q4 (HC `W_SHARED` layout)**: the current HIP path infers `W_SHARED`
  from `weight.numel()`. If a future checkpoint stores weights with a
  reshape (`[HC, GROUP_DIM]` flattened to `[HC * GROUP_DIM]` in some
  shape), the inference breaks. Add a TODO when we see a second
  checkpoint.
- **Q5 (cudagraph capture)**: none of the new HIP kernels allocate
  scratch or use Triton JIT, so they should be cudagraph-safe
  out-of-the-box. Confirm by capturing/replaying
  `_short_conv_dilated_decode_batched` with the dispatcher on.

---

## Next actions (priority-ordered)

1. **[S0 build]** Rebuild `_rocm_C.abi3.so` against the new
   `csrc/rocm/*_rdna2.cu` files. Verify with the
   `test_rdna_hip_build_smoke` probe.
2. **[S1 parity]** Land the HC parity tests first (smallest kernels,
   fastest signal). Use a low-shape parametrize matrix for the first
   CI run, expand to production shapes after.
3. **[S1 parity]** Land the QSA `qsa_store_cache_rows` parity test
   (pure scatter — no math, only a stride contract to verify).
4. **[S1 parity]** Land the PLE conv decode parity test (most
   complex of the decode paths; non-trivial state write-back
   semantics).
5. **[S5 bench]** Run a small set of production-shape benches
   against the current Triton reference on `.176`. Target: ≥ +20%
   on every kernel; if a kernel is slower, leave it gated
   (`VLLM_DISABLED_KERNELS` analog) and document why.
6. **[S6 default-on]** Flip env-var defaults from "0" to "1" one
   family at a time, on a release branch.

---

## Change log

- **2026-09-14** — scaffolding shipped:
  - `csrc/rocm/hc_rdna2.cu` (5 kernels)
  - `csrc/rocm/qsa_rdna2.cu` (3 ops)
  - `csrc/rocm/ple_short_conv_rdna2.cu` (2 kernels)
  - `vllm/models/qwen4_exp/amd/ops/{hc,qsa,ple_conv}_rdna2.py` dispatchers
  - `csrc/rocm/torch_bindings.cpp` (10 new ops under `#ifdef VLLM_ROCM_GFX1030`)
  - `csrc/rocm/ops.h` (10 host-wrapper declarations)
  - `CMakeLists.txt` (3 new files in the gfx1030 `EXT_SRC` list)
  - `vllm/envs.py` (3 env vars: `VLLM_RDNA_HC_PREFILL_HIP`,
    `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_PLE_CONV_HIP`; default "0")
  - `vllm/models/qwen4_exp/amd/ops/{hc,qsa}.py` (dispatcher early-returns)
  - `vllm/models/qwen4_exp/amd/ple_layer.py` (early-return branches in
    `_short_conv_dilated_decode_batched` /
    `_short_conv_dilated_prefill_batched`)
  - `docs/rdna2/qwen4_exp_hip_path.md` (overview + opt-in recipe)
  - `docs/rdna2/qwen4_exp_hip_tests.md` (test specification)

---

## 2026-09-14 — QSA/attention abort diagnostic (pre-HIP-path blocker)

The `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` checkpoint now **loads, shards,
EP-splits, registers all 4 PLE workers, allocates KV (313k tokens) and captures
its CUDA graphs** after the PLE/offload/cache fixes in `rdna_extras`
(`e7ce54178`, `469c456d6`, `7259f0f44`). It then dies with:

- Worker `SIGABRT` (exit -6), no Python traceback; `GPU core dump failed`.
- dmesg: `amdgpu ... Trap debug id already reserved` (×3) — a GPU **runtime**
  trap, **not** a Triton compile abort (no `Fatal Python error`/`make_amdgcn`).

Observations:
- MoE already on RDNA2 HIP (`CompressedTensorsWNA16RDNA2MoEMethod`).
- `Using FlashAttention version None` (no flash-attn lib on ROCm).
- QSA forward routes via `qwen4_exp_qsa_with_output` (registered as a
  `direct_register_custom_op`) -> Triton `qsa_sparse_paged_attention`
  (`forward_qsa` in `vllm/models/qwen4_exp/amd/qsa.py`).
- `Op 'sparse_attn_indexer' doesn't exist` is a **harmless no-op** (that's the
  CDNA/AITER config name; the AMD QSA uses `paged_mqa_logits_decode_rdna2`).
- **QSA tile env overrides did NOT fix it**: `VLLM_RDNA_QSA_BLOCK_N=16
  VLLM_RDNA_QSA_WARPS=2 VLLM_RDNA_QSA_SPLITS=1` still SIGABRTs. So it is not
  the prefill tile profile.

This is the `[!]`-status item to address when finalising the HIP path: the QSA
`forward_qsa` / `qwen4_exp_qsa_with_output` runtime (the `qsa_rdna2.cu` ops
under `VLLM_RDNA_QSA_HIP` are the target replacement).

**Standalone-probe refinement (2026-09-14):** `qsa_sparse_paged_attention`
(the `forward_qsa` sparse-attention kernel) **passes in isolation** with the
model's actual geometry (24 heads / 2 KV / head_dim 256 / group 12, up to 512
tokens) — no trap. So the SIGABRT is **not** the sparse-attention kernel; it is
upstream of it in `_run_qsa`: either the **MQA paged indexer**
(`self.indexer(...)` -> `_qsa_mqa_paged_kernel`) or the `do_kv_cache_update` /
FlashAttention-forward integration. Probe: `/tmp/qsa_probe.py` on `.176`.

**Refinement 2 (2026-09-14):** `qsa_mqa_paged` (the indexer, `BLOCK_N=128 /
BLOCK_D=256 / num_warps=8`) **also passes standalone** with the model config
(24 heads, head_dim 256, up to 64 tokens). So neither QSA kernel is the trap;
the SIGABRT is in the **forward integration** — the `qwen4_exp_qsa_with_output`
op -> `_run_qsa`'s `do_kv_cache_update` / FlashAttention `forward` path, or a
different kernel in the model's warmup forward (the trace shows
`Using FlashAttention version None`). Probes: `/tmp/{qsa,mqa}_probe.py`.
