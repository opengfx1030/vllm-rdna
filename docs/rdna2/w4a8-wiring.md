# W4A8 sdot4 wiring — opt-in drop-in for the dense W4A16 prefill on gfx1030

Branch: `w4a8-wiring` (= `rdna_extras` + the production wrapper around the
explore kernel from PR #9). Default OFF — the dispatcher takes the existing
`gptq_gemm_rdna2_prefill` path unless `VLLM_RDNA2_W4A8_SDOT4=1`.

The kernel body (`w4a8_gemm_kernel`, `w4a8_act_quant_kernel`, `Cfg`)
lives in `csrc/rocm/w4a8_sdot4_rdna2.cuh`. Production instantiates only
`a8_lds_k32_ag` (group sizes 32, 64, and 128). The explore source of truth
stays untouched per the "do not modify `csrc/rocm/explore/*` in place" rule —
the fork never had an `explore/` directory and this branch does not create one.

## What changed

| File | Purpose |
|---|---|
| `csrc/rocm/w4a8_sdot4_rdna2.cu` | Production TU. `w4a8_gemm_rdna2` is a self-contained drop-in for `gptq_gemm_rdna2_prefill` with the same signature: allocates the int8 A buffer, A scales, A group sums, and the output internally; tries the `a8_lds_k32_ag` W4A8 fast path; falls back to `gptq_gemm_rdna2_prefill` whenever the shape/LDS budget is not eligible. `w4a8_act_quant_rdna2` is kept for tests (returns an empty tensor on ineligibility). One-time `TORCH_WARN_ONCE` marker when the fast path fires. |
| `csrc/rocm/w4a8_sdot4_rdna2.cuh` | Production GEMM and activation-quant kernels. Live launch is `a8_lds_k32_ag`. |
| `csrc/rocm/ops.h` | Declarations for the two ops. `w4a8_gemm_rdna2` mirrors `gptq_gemm_rdna2_prefill`'s signature; `w4a8_act_quant_rdna2` keeps the standalone-test signature. |
| `csrc/rocm/torch_bindings.cpp` | Registers both ops with the new signatures; the C++ Meta stubs are gone (the fake path is `register_fake` in `_custom_ops.py`). |
| `vllm/_custom_ops.py` | Wrappers `w4a8_act_quant_rdna2` / `w4a8_gemm_rdna2` with `register_fake` paths that match the new contract. |
| `vllm/model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py` | The variant selector `_rdna2_w4a16_select_kernel` gains the `"w4a8_prefill"` arm (gated on `W4A8_MIN_ROWS`, `k % 32`, `k % group_size`, `_w4a8_lds_fits`). `process_weights_after_loading` resolves `self._w4a8` once from the env + op-availability. The forward's elif chain calls `ops.w4a8_gemm_rdna2(...)` directly — no post-call sentinel test, no env access, no logging in the traced region. |
| `tests/kernels/quantization/test_rdna2_w4a8.py` | Updated for the fused gemm signature + the new fused-entry env-on/off A/B. |
| `docs/rdna2/w4a8-wiring.md` | This file. |

The kernel body, the W4A16/MoE/attention TUs, `rdna_extras`, and the launchers
are not touched.

## Design

`w4a8_gemm_rdna2` is **self-contained, Tensor-in -> Tensor-out**, with the
same signature as `gptq_gemm_rdna2_prefill`. It allocates the int8 A, A
scales, A group sums, and the output internally, then either fires the W4A8
fast path or falls back **internally** to `gptq_gemm_rdna2_prefill`. The
Python forward is branch-free: the variant selector picks `"w4a8_prefill"`
from shape-derived ints, the dispatcher calls `ops.w4a8_gemm_rdna2(...)`
directly, and the result is always a populated tensor.

The Python pre-check is only an **optimisation**: it skips crossing the C ABI
for shapes the W4A8 path could never service. The C++ entry always returns a
correct, populated `[M, N]` fp16 tensor — the final eligibility decision
lives in C++ (mirror of `pick_split_k`, plus shape/contiguity/group/g_idx
gates).

## Op contract

```
w4a8_act_quant_rdna2(x, group_size, a_i8, a_scale, a_asum) -> Tensor
  x       [M, K]            fp16, contiguous on dim 1
  a_i8    [T, K/8, 8, 8]    int8   (T = ceil(M / 8))
  a_scale [T, K/G, 8]       fp32   (per-(token, group), A_GROUP variant)
  a_asum  [T, K/G, 8]       int32
  Returns the int8 buffer on success or an empty tensor when not eligible.
  Used by the standalone test; the wired gemm op allocates its own
  intermediates.

w4a8_gemm_rdna2(a, b_q_weight, b_qzeros, b_scales, b_g_idx, use_v2_format) -> Tensor
  Drop-in for gptq_gemm_rdna2_prefill. Allocates the int8 A buffer, A
  scales, A group sums, and the output internally; tries the W4A8 sdot4
  fast path; falls back to gptq_gemm_rdna2_prefill when the shape/LDS
  budget is not eligible. Always returns a populated [M, N] fp16 tensor.
  use_v2_format selects zero_offset (0 for AWQ uint4, 1 for GPTQv1 uint4b8).
```

## Selection (Python pre-check, optimisation only)

`_rdna2_w4a16_select_kernel(m, k, n, is_awq, w4a8, group_size)` returns
`"w4a8_prefill"` only when **all** of:

- `w4a8` (i.e. `self._w4a8` is True: env var set + ops built),
- `m >= W4A8_MIN_ROWS` (33; prefill-only — decode keeps `rdna2_decode`),
- `k % 32 == 0` (K-step 32 alignment),
- `k % group_size == 0` (group alignment),
- `_w4a8_lds_fits(k, group_size)` (mirror of `pick_split_k`: some
  group-aligned split <= 16 fits `M_TILE=8` rows of K plus per-group
  scales in 64 KiB LDS).

When the selector returns `"w4a8_prefill"`, the forward calls
`ops.w4a8_gemm_rdna2(x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format)` — the
same call shape as `ops.gptq_gemm_rdna2_prefill` — and the C++ side owns
the rest.

## Flag (resolved at construction)

`process_weights_after_loading` reads the env var once and stores
`self._w4a8`:

```python
try:
    torch.ops._rocm_C.w4a8_gemm_rdna2
    w4a8_ops_built = True
except AttributeError:
    w4a8_ops_built = False
self._w4a8 = os.environ.get(W4A8_ENV_VAR) == "1" and w4a8_ops_built
```

The forward reads `self._w4a8` and never touches `os.environ`. The selector
is fully int-derived from `m, k, n, group_size` and the `w4a8` flag — dynamo
sees no data-dependent branch on an op result.

## C++ correctness gate

`w4a8_gemm_rdna2` is the canonical entry used by the dispatcher. When
invoked, it:

1. Validates the shape / contiguity / group_size / g_idx-empty / gfx1030 /
   fp16 gates on the wire (the same gates the Python selector uses, plus
   finer C++-only checks).
2. On any ineligible shape, falls back **internally** to
   `gptq_gemm_rdna2_prefill` and returns its output.
3. Otherwise, allocates `a_i8`/`a_scale`/`a_asum`/`out` internally, runs
   `w4a8_act_quant_kernel` then `w4a8_gemm_kernel` (config `a8_lds_k32_ag`,
   MT=8, split_k=1).
4. On any launch failure, falls back to `gptq_gemm_rdna2_prefill`.

A one-time `TORCH_WARN_ONCE` fires the first time the fast path succeeds, so
a serve log confirms it landed:

```
W ... RDNA2 W4A8 sdot4 prefill fast path active (config a8_lds_k32_ag)
```

## Build

```bash
cd <local-home>/Projects/infrastructure/gfx1030_optimized/vllm-rdna-0.28.0
git switch w4a8-wiring
rsync -avz --exclude='.git/' -e "ssh -i ~/.ssh/id_ed25519_ansible" \
    ./ <bench-host>:~/vllm-rdna-0.28.0/

ssh <bench-host>
source <bench-home>/Apps/vllm/venv-7.14.0_0.28.0/bin/activate
cd <bench-home>/vllm-rdna-0.28.0
export SETUPTOOLS_SCM_PRETEND_VERSION=0.20.1.dev99
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH='gfx1030'
export VLLM_PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python
export MAX_JOBS=16
export CMAKE_BUILD_TYPE=RelWithDebInfo
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
export ROCM_HOME=/opt/rocm/core-7.14
export ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14
export HIP_ROOT_DIR=/opt/rocm/core-7.14
export CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$PATH
pip install -e . --no-build-isolation --no-deps
```

Verify the ops are registered:

```bash
python -c "
import torch
schemas = torch._C._jit_get_all_schemas()
names = sorted({str(s) for s in schemas if 'w4a8' in str(s)})
for n in names: print(n)
"
# Expect:
#   _rocm_C::w4a8_act_quant_rdna2(Tensor x, int group_size, Tensor(a!) a_i8,
#                                  Tensor(a!) a_scale, Tensor(a!) a_asum) -> Tensor
#   _rocm_C::w4a8_gemm_rdna2(Tensor a, Tensor b_q_weight, Tensor b_qzeros,
#                             Tensor b_scales, Tensor b_g_idx,
#                             bool use_v2_format) -> Tensor
```

## Validate

### Pytest (unit, no model)

```bash
cd <bench-home>/vllm-rdna-0.28.0
.venv/bin/python -m pytest tests/kernels/quantization/test_rdna2_w4a8.py -v
```

The suite covers:
- `test_w4a8_act_quant_matches_numpy[M/g]` — act_quant vs the NumPy
  reference (per-(token, group) quant + tiled layout, bit-exact).
- `test_w4a8_act_quant_rejects_bad_group_size` — invalid group_size (16)
  returns an empty tensor.
- `test_w4a8_gemm_matches_w4a16_prefill[g]` — the fused gemm vs
  `_w4a16_reference` on the same packed weight buffer; rel-L2 < 0.1.
- `test_w4a8_dispatcher_uses_w4a16_when_env_var_unset` — with the env var
  absent (even on an eligible shape), `apply_weights` matches
  `gptq_gemm_rdna2_prefill`.
- `test_w4a8_fused_entry_env_on_off` — the new fused-entry A/B: eligible
  shape (M=64) is close-but-not-identical with env on vs off (rel-L2 < 0.1,
  > 1e-6 so the fast path provably fired); ineligible shape (M=16) is
  byte-identical.

All skip cleanly when the ops are absent (no GPU, partial build, or
non-gfx1030).

### In-model A/B + matrix

The serve launcher `scripts/serve_gfx1030_27b_dense.sh` exports
`VLLM_RDNA2_W4A8_SDOT4` from the `W4A8` knob (default `1`). The matrix
driver `tools/rdna2_028/m27b_matrix.sh` runs 16k/1k and 1k/512 at c=1 and
c=8 for both `MTP=0` and `MTP=2`. Confirm the fast path fired by looking for
the C++ `TORCH_WARN_ONCE` in the serve log:

```bash
grep "W4A8 sdot4 prefill fast path active" <serve log>
```

A meaningful A/B compares prefill throughput and TTFT at the dense-FFN
prefill shapes. Decode (M < `W4A8_MIN_ROWS`) routes to `gptq_gemm_rdna2`
and is unaffected.

## Env gates

| Env | Effect |
|---|---|
| `VLLM_RDNA2_W4A8_SDOT4=1` | Opt in to the W4A8 path. Default OFF (unset = W4A16). |
| (anything else) | No change vs `rdna_extras`. |

This env var is **experimental**. No new env vars in the platform selector
(per AGENTS.md — config-propagated only). No default behavior change.

## Known limits

- **split_k > 1 is OFF in the wired path.** The pk4 fp16 CAS epilogue is
  order-dependent (see `w4a8_sdot4_rdna2.cuh:atomic_add_f16x2/f16x4`); the
  wired `w4a8_gemm_rdna2` uses split_k=1 with plain stores.
- **Act-order (`g_idx`) is rejected.** The W4A8 GEMM reads a contiguous A;
  an act-order pack would read out of permutation. The C++ entry checks
  `has_g_idx` and falls back to `gptq_gemm_rdna2_prefill` (which handles
  g_idx).
- **MT=8 hardcoded.** The wired path uses the `a8_lds_k32_ag` config (MT=8)
  hard-coded in the C++ entry.
- **Pre-fill only.** `W4A8_MIN_ROWS=33` is the floor: decode (M <= 32) stays
  on the `rdna2_decode` arm. The C++ side enforces its own shape gates and
  falls back to the W4A16 prefill kernel whenever any finer check fails —
  the selector is an optimisation, never a correctness dependency.

## Pointer back to the explore work

The explore-only tree lives at `/tmp/pr9_branch/` (extracted copy of upstream
branch `explore/w4a8-sdot4`):

- Explore kernel body (source of truth, unchanged):
  `/tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4.cuh`
  (copied verbatim into `csrc/rocm/w4a8_sdot4_rdna2.cuh` here).
- Explore C ABI (replaced by this branch's `.cu`):
  `/tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4_capi.cu`
- Explore ISA shim (CPU compilation without ROCm):
  `/tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4_isa_shim.h`
- Explore NumPy reference (CPU checks, no torch/GPU):
  `/tmp/pr9_branch/benchmarks/kernels/w4a8_sdot4_explore/reference.py`
- Explore G2 correctness tests:
  `/tmp/pr9_branch/benchmarks/kernels/w4a8_sdot4_explore/test_reference.py`
