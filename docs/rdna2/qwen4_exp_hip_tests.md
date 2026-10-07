# Qwen4Exp / Qwen3.8-Flash-Next HIP path — test plan

**Status**: scaffolding shipped 2026-09-14; tests not yet run on `.176`.
This document is the test specification someone picks up the next time
the build server is reachable. The HIP code is **default-off** (Triton
path stays the source of truth) so any new test should be guarded by
the matching `VLLM_RDNA_*_HIP=1` env var.

## Conventions

- Test files live under `tests/kernels/rocm/<area>/` (one file per HIP
  family — mirrors `tests/kernels/quantization/test_gdn_decode_rdna2.py`
  and `tests/kernels/attention/test_fa_rdna2_writer_layout.py`).
- Each test skips at module level when the kernel is not built /
  registered: `current_platform.is_rocm()` + `on_gfx10x()` +
  `_op_exists(name)` check.
- Numerical tolerance for fp16 elementwise: `atol=1e-3, rtol=1e-3`. For
  reductions (`sum_sq` in RMSNorm) loosen to `atol=1e-2`. For tiled
  reductions (QSA compress, conv1d with K=16) loosen further to
  `atol=5e-3`.
- Each test produces a short log line `atol=… maxdiff=… passes=N` so we
  can spot regressions across commits without reading diffs.

```python
# Standard module guard template
import pytest, torch
from vllm.platforms import current_platform
if not current_platform.is_rocm():
    pytest.skip("RDNA-only", allow_module_level=True)
from vllm.platforms.rocm import on_gfx10x
if not on_gfx10x():
    pytest.skip("gfx1030-only", allow_module_level=True)

# Op-presence guard (the .so may not have been rebuilt against the
# scaffold yet — keeps the test green while the build is pending).
def _op(name: str) -> bool:
    return any(str(s) == name for s in torch._C._jit_get_all_schemas())
```

---

## 1. HC prefill HIP

Test file: `tests/kernels/rocm/qwen4_exp/test_hc_rdna2.py`
Reference: `vllm/models/qwen4_exp/amd/ops/hc.py`
HIP source: `csrc/rocm/hc_rdna2.cu`

### 1.1 Op-presence probe

```python
@pytest.mark.parametrize("op", [
    "_rocm_C::hc_grouped_gemma_rmsnorm_rdna2",
    "_rocm_C::hc_silu_rdna2",
    "_rocm_C::hc_gate_mix_rdna2",
    "_rocm_C::hc_combine_rdna2",
    "_rocm_C::hc_combine_norm_rdna2",
])
def test_hc_op_registered(op):
    assert _op(op), f"{op} not registered (rebuild _rocm_C.abi3.so)"
```

### 1.2 Per-kernel parity vs Triton reference

For each of the 5 kernels, parametrize the shape matrix below and
compare HIP output to the existing Triton kernel
(`torch.ops.vllm.qwen4_exp_<name>`).

#### Shape matrix

| param | values | why |
|---|---|---|
| `N` (rows) | `[1, 8, 64, 1024, 8192]` | decode single-token → chunked-prefill |
| `DIM` | `[512, 4096, 8192]` | Qwen3.8 hidden_size variants |
| `HC` (for hc_*) | `[1, 2, 4]` | qwen4_exp checkpoints use `hc_count=4` |
| `num_groups` | `[2, 4, 8]` | matches `hc_count` (Gemma layout) |
| `W_SHARED` | `[True, False]` | `weight.numel() == GROUP_DIM` vs `== DIM` |
| `dtype` | `[float16]` | no bf16 on gfx1030 (RDNA2 lacks it) |

#### Per-kernel tests

```python
@pytest.mark.parametrize("N,DIM,NG,W_SHARED", SHAPE_MATRIX)
def test_grouped_gemma_rmsnorm_rdna2_parity(N, DIM, NG, W_SHARED):
    # Triton ref
    torch.manual_seed(0)
    weight = (torch.randn(DIM // NG) if W_SHARED else torch.randn(DIM))
    x = torch.randn(N, DIM, dtype=torch.float16, device="cuda")
    y_ref = torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm(
        x, weight, 1e-6, NG,
    )
    # HIP
    y = torch.empty_like(x)
    torch.ops._rocm_C.hc_grouped_gemma_rmsnorm_rdna2(
        x, weight, y, NG, 1e-6,
    )
    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2), (
        f"maxdiff={(y - y_ref).abs().max().item()}"
    )
```

Repeat the same pattern for:
- `hc_silu_rdna2` — exact fp16 elementwise, `atol=1e-3`
- `hc_gate_mix_rdna2` — reduction over HC streams, `atol=5e-3`
- `hc_combine_rdna2` — elementwise affine, `atol=1e-3`
- `hc_combine_norm_rdna2` — combined op; test that
  `hc_combine + grouped_gemma_rmsnorm` matches `hc_combine_norm`'s
  `out` and `y` outputs respectively (`atol=1e-2` for the combined op
  because of the intermediate fp16 round).

### 1.3 Round-trip boundary

`hc_combine_norm` writes an intermediate `out` value rounded to fp16
**before** the RMSNorm pass (mirroring the unfused combine → RMSNorm
boundary). Verify that calling `hc_combine` then `grouped_gemma_rmsnorm`
on the output matches `hc_combine_norm`. This is what the model relies
on at line `ple_layer.py:1192` when a combine is pending into the next
HC mix.

### 1.4 Dispatcher env-var gate

```python
def test_hc_dispatcher_env_gate(monkeypatch):
    from vllm.models.qwen4_exp.amd.ops import hc_rdna2
    monkeypatch.setenv("VLLM_RDNA_HC_PREFILL_HIP", "0")
    assert hc_rdna2.hc_use_rdna2() is False
    monkeypatch.setenv("VLLM_RDNA_HC_PREFILL_HIP", "1")
    monkeypatch.setattr(hc_rdna2, "on_gfx10x", lambda: True)
    assert hc_rdna2.hc_use_rdna2() is True
    monkeypatch.setattr(hc_rdna2, "on_gfx10x", lambda: False)
    assert hc_rdna2.hc_use_rdna2() is False
```

### 1.5 HC unroll factor edge cases

The HIP template is hard-coded to `HC in {1, 2, 4, 8}`. `hc_count=3` and
`hc_count=5` should raise a `TORCH_CHECK` from the C++ wrapper. Add a
test that asserts the error message includes "hc_count in {1,2,4,8} only".

---

## 2. QSA decode HIP

Test file: `tests/kernels/rocm/qwen4_exp/test_qsa_rdna2.py`
Reference: `vllm/models/qwen4_exp/amd/ops/qsa.py`
HIP source: `csrc/rocm/qsa_rdna2.cu`

### 2.1 Op-presence probe (same template as §1.1)

```python
@pytest.mark.parametrize("op", [
    "_rocm_C::qsa_store_cache_rows_rdna2",
    "_rocm_C::qsa_compress_groups_rdna2",
    "_rocm_C::qsa_mqa_paged_rdna2",
])
def test_qsa_op_registered(op): ...
```

### 2.2 `qsa_store_cache_rows_rdna2` parity

Shape matrix:
- `num_rows`: `[1, 8, 64, 1024]` (decode batches)
- `WIDTH`: `[32, 64, 128]` (QSA head_dim variants)
- `PAGE_SIZE`: `[16, 784]` (Qwen3.5 hybrid forced by GDN page alignment)
- `num_blocks`: `[1, 4, 64]`

```python
def test_store_cache_rows_parity(N, width, page_size, num_blocks):
    torch.manual_seed(N + width + num_blocks)
    rows = torch.randn(N, width, dtype=torch.float16, device="cuda")
    slots = torch.randint(0, num_blocks * page_size, (N,),
                          dtype=torch.int32, device="cuda")
    # Mix valid and invalid slots (NULL_BLOCK_ID convention).
    slots[::5] = -1
    cache_ref = torch.zeros(num_blocks, page_size, width,
                           dtype=torch.float16, device="cuda")
    cache_hip = torch.zeros_like(cache_ref)

    # Triton ref
    qsa_store_cache_rows(cache_ref, slots, rows)  # writes in-place
    # HIP path (from the dispatcher)
    qsa_rdna2.qsa_store_cache_rows(rows, slots, cache_hip,
                                   page_size, width)
    assert torch.equal(cache_ref, cache_hip)
```

### 2.3 `qsa_compress_groups_rdna2` parity

Shape matrix:
- `rows`: `[1, 8, 64]`
- `head_dim`: `[32, 64, 128]`
- `compress_ratio`: `[1, 2, 4, 8]`
- `compressor_state_size`: `[compress_ratio * 2]` (always > ratio)
- `num_requests`: `[1, 4]`
- `LOAD_ROPE_POSITIONS`: `[False, True]`
- `*_BLOCK_TABLE valid mix*: half valid slots, half `NULL_BLOCK_ID`

```python
def test_compress_groups_parity(rows, head_dim, compress_ratio,
                                num_requests, load_rope):
    # Triton reference call
    pooled_ref, first_ref = qsa_compress_groups_with_ratio(
        raw_keys, raw_positions, comp_state_cache,
        comp_state_table, token_to_req, query_start_loc,
        logical_positions, compressed_slots, compress_ratio,
        rope_cache=rope_cache,
    )
    # HIP path (via the dispatcher, which handles the [rows, 1, head_dim]
    # <-> [rows, head_dim] reshape + first_positions return).
    pooled_hip, first_hip = qsa_rdna2.qsa_compress_groups(...)  # see dispatcher
    assert torch.allclose(pooled_hip, pooled_ref, atol=5e-3, rtol=5e-3)
    assert torch.equal(first_hip, first_ref)
```

Edge cases worth pinning:
- All-zero `compressor_state_table` (every request falls back to raw
  rows — exercises the `state_ok = False` branch).
- `end_position < COMPRESS_RATIO - 1` (early tokens that don't form a
  group yet — `valid_row` should be False; output stays zero).
- `compressed_slot < 0` (caller hasn't assigned a slot yet — same).

### 2.4 `qsa_mqa_paged_rdna2` parity

The HIP wrapper currently delegates to
`paged_mqa_logits_decode_rdna2`. Validate the shape contract matches
(`q [N, 1, head_dim]`, `kv_cache [num_blocks, page, 1, head_dim]`):

```python
def test_qsa_mqa_paged_parity(num_rows, num_blocks, page, head_dim):
    q = torch.randn(num_rows, 1, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(num_blocks, page, 1, head_dim,
                    dtype=torch.float16, device="cuda")
    weights = torch.ones(num_blocks, dtype=torch.float32, device="cuda")
    context_lens = torch.full((num_rows,), page, dtype=torch.int32,
                               device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32,
                                device="cuda").repeat(num_rows, 1)
    # HIP call
    logits_hip = torch.ops._rocm_C.qsa_mqa_paged_rdna2(
        q, k, weights, context_lens, block_tables, page * 2,
    )
    assert logits_hip.shape == (num_rows, page)
    # Numerical parity: ref = direct sum (sanity check for the wrapper
    # at this point; full parity vs the Triton _qsa_mqa_paged_kernel is
    # deferred until the wrapper's signature translation is complete).
    ref = (q.squeeze(1).unsqueeze(1) @ k.squeeze(2).transpose(1, 2)).sum(-1)
    assert torch.allclose(logits_hip, ref, atol=2e-2, rtol=2e-2)
```

---

## 3. PLE dilated short-conv HIP

Test file: `tests/kernels/rocm/qwen4_exp/test_ple_short_conv_rdna2.py`
Reference: `Qwen4ExpPLELayer._short_conv_dilated_decode_batched` /
`_short_conv_dilated_prefill_batched` in
`vllm/models/qwen4_exp/amd/ple_layer.py`
HIP source: `csrc/rocm/ple_short_conv_rdna2.cu`

### 3.1 Op-presence probe (template as §1.1)

### 3.2 Decode batched parity

Shape matrix (Qwen4Exp PLE defaults; parametrize for variations once we
see other checkpoints):
- `B` (batch): `[1, 8, 64]`
- `D` (hidden_size): `[2560, 5120]` (Qwen3.8 hidden_size variants)
- `state_len`: `[2, 4, 8]` (PLE conv kernel sizes)
- `dilation`: `[1, 2]` (no dilation → 1)
- `silu`: `[True, False]`
- `has_init mix`: half True, half False (zero-init vs cached state)
- `null_block slots`: half the rows remapped to `NULL_BLOCK_ID`

```python
def test_ple_short_conv_decode_parity(B, D, state_len, dilation, silu):
    K = state_len + dilation + 1
    torch.manual_seed(B * 100 + D + state_len + dilation)
    x = torch.randn(B, D, dtype=torch.float16, device="cuda")
    state = torch.randn(2 * B, D, state_len, dtype=torch.float16, device="cuda")
    state[0:B] = 0  # first half is zero-init, second half is cached
    weight = torch.randn(D, K, dtype=torch.float16, device="cuda")
    state_idx = torch.tensor([0] * (B // 2) + [B] * (B - B // 2),
                              dtype=torch.int32, device="cuda")
    state_idx[::3] = NULL_BLOCK_ID  # one-third nulled out
    has_init = torch.tensor([0] * (B // 2) + [1] * (B - B // 2),
                            dtype=torch.uint8, device="cuda")
    out_hip = torch.empty_like(x)
    ple_rdna2.ple_short_conv_decode(
        x, state, weight, out_hip, dilation, state_len, silu,
        bias=None, state_idx=state_idx, has_init=has_init,
        null_block=NULL_BLOCK_ID,
    )
    # Reference: torch F.conv1d + state shift.
    history = torch.cat([
        state[state_idx.clamp_min(0)][..., :state_len],  # gather cached/0
        x.unsqueeze(-1),
    ], dim=-1)
    ref = F.conv1d(history, weight.unsqueeze(1),
                   groups=D, dilation=dilation).squeeze(-1)
    if silu: ref = F.silu(ref)
    valid = (state_idx != NULL_BLOCK_ID).view(-1, 1).to(ref.dtype)
    ref = ref * valid
    # State write-back is left-shift by dilation (matches the HIP path).
    state_ref = state.clone()
    next_state = history[..., -state_len:]  # last state_len taps
    for b in range(B):
        if state_idx[b] == NULL_BLOCK_ID:
            continue
        for d in range(D):
            state_ref[state_idx[b], d, :] = next_state[b, d, :]
    assert torch.allclose(out_hip, ref, atol=1e-3, rtol=1e-3)
    # Compare state write-back exactly on the valid rows.
    cmp_rows = (state_idx != NULL_BLOCK_ID).nonzero().squeeze(-1)
    for b in cmp_rows.tolist():
        slot = state_idx[b].item()
        if slot < 0: continue
        assert torch.equal(state[slot], state_ref[slot])
```

### 3.3 Prefill batched parity

Shape matrix:
- `num_prefills`: `[1, 4, 8]`
- `max_len`: `[128, 512, 2048]` (constrained to <= 128 in current HIP
  build — extend once we add the larger variant)
- `D`: `[2560, 5120]`
- `state_len`, `dilation`, `silu`: same matrix
- `lengths`: random in `[1, max_len]`, sorted
- `valid_state mix`: half True, half False (zero-init vs cached state)
- padding tokens: positions beyond each request's length should be
  zero in the output

```python
def test_ple_short_conv_prefill_parity(num_prefills, max_len, D,
                                      state_len, dilation, silu):
    # same setup as decode but with [B, D, max_len] tensors; compare
    # against F.conv1d applied on [B, D, state_len + max_len] input.
    # Padding tokens: output[b, d, t>=lengths[b]] == 0.
```

### 3.4 K bound

`K = state_len + dilation + 1` must be `<= 64` (per-CTA register buffer
size). Test that `K=65` raises a `TORCH_CHECK`.

### 3.5 Dispatcher gate

Same env-var template as §1.4:
```python
def test_ple_conv_env_gate(monkeypatch): ...
```

### 3.6 PLE layer end-to-end (smoke)

Larger integration test that mounts a `Qwen4ExpPLELayer` on synthetic
weights and validates that with `VLLM_RDNA_PLE_CONV_HIP=1` the layer's
`_short_conv_dilated_decode_batched` matches the torch `F.conv1d`
reference up to fp16 noise. This catches dispatcher + bookkeeping
regressions (state shift, NULL_BLOCK remap, valid_mask zeroing).

---

## 4. Integration tests

Test file: `tests/v1/spec_decode/test_qwen4_exp_rdna_hip.py`
(alongside `tests/v1/spec_decode/test_qwen4_exp.py`)

### 4.1 Smoke

```python
def test_qwen4_exp_rdna_hip_smoke():
    # Build a tiny Qwen4ExpConfig (1 layer, hc_count=4, hidden_size=64,
    # 4 heads, head_k_dim=8; PLE layer at layer 0 with embed_dim=8).
    # Load the model on CPU and run a forward pass with the env-var
    # gates ON. We only need "did the model run to completion without
    # raising" — exact numerics are covered by the parity tests above.
```

### 4.2 vLLM smoke

```python
def test_vllm_serve_smoke():
    # Construct VllmConfig with the right flags, call
    # `model_executor.models.qwen4_exp.amd.model.Qwen4ExpForCausalLM`
    # through the standard forward path. Asserts that the dispatcher
    # routes to HIP for at least one of the new kernels (probe via a
    # counter or via `VLLM_RDNA_HC_PREFILL_HIP=1` + a known forward call).
```

This is the right place to pin a small Cudagraph capture loop if we
ever decide to put the new HIP paths inside a captured graph. The
HIP kernels all write into caller-provided outputs and use no JIT
scratch, so they should be cudagraph-safe out of the box; a single
test that captures and replays a representative forward pass is
sufficient.

---

## 5. Performance tests (deferred — run on `.176`)

Once the parity tests pass, add a thin bench module:

Test file: `tests/kernels/rocm/qwen4_exp/bench_rdna_hip.py`

```bash
pytest tests/kernels/rocm/qwen4_exp/bench_rdna_hip.py --benchmark-only
```

Measures (production shapes only — Qwen3.8-Flash-Next-AWQ-W4A16 on
4×V620, TP=4, c=8):

| Kernel | Production shape | Triton (tok/s) | HIP target |
|---|---|---|---|
| `hc_combine_norm` | (B=2048, DIM=5120, HC=4) | baseline | ≥ +20% |
| `hc_silu` | (B=2048, DIM=320) | baseline | ≥ +20% |
| `qsa_store_cache_rows` | (B=512, WIDTH=128) | baseline | ≥ +30% (pure scatter) |
| `qsa_compress_groups` | (B=512, head_dim=128, ratio=4) | baseline | ≥ +15% |
| `ple_short_conv_decode` | (B=8, D=5120, K=5) | baseline F.conv1d | ≥ +50% |

Tracked separately from the parity tests; the parity tests are the
gate, the bench is the signal.

---

## 6. Build / register smoke (run first after rebuilding)

```python
def test_rdna_hip_build_smoke():
    import torch
    schemas = torch._C._jit_get_all_schemas()
    new_ops = [
        "_rocm_C::hc_grouped_gemma_rmsnorm_rdna2",
        "_rocm_C::hc_silu_rdna2",
        "_rocm_C::hc_gate_mix_rdna2",
        "_rocm_C::hc_combine_rdna2",
        "_rocm_C::hc_combine_norm_rdna2",
        "_rocm_C::qsa_store_cache_rows_rdna2",
        "_rocm_C::qsa_compress_groups_rdna2",
        "_rocm_C::qsa_mqa_paged_rdna2",
        "_rocm_C::ple_short_conv_decode_rdna2",
        "_rocm_C::ple_short_conv_prefill_rdna2",
    ]
    registered = {str(s) for s in schemas}
    missing = [op for op in new_ops if op not in registered]
    assert not missing, f"rebuild _rocm_C.abi3.so, missing: {missing}"
```

If this fails, the rebuild is incomplete — re-check that
`csrc/rocm/hc_rdna2.cu`, `csrc/rocm/qsa_rdna2.cu`,
`csrc/rocm/ple_short_conv_rdna2.cu` are in the gfx1030 `EXT_SRC` list
(see `CMakeLists.txt:1537`) and that the new `rocm_ops.def` / `impl`
calls in `csrc/rocm/torch_bindings.cpp` are inside the
`#ifdef VLLM_ROCM_GFX1030` block.

---

## 7. Follow-up checklist (out of scope for this PR)

- **Sparse splitk attention** (`_qsa_sparse_paged_gqa_splitk_kernel` +
  `_qsa_merge_splitk_kernel`) — port when the prefill cudagraph story
  is solid; current focus is decode-side.
- **`qsa_mqa_paged`** signature alignment — once the wrapper grows the
  missing `query_positions` / `sequence_lengths` / `compress_ratio` /
  `score_scale` args, the parity test in §2.4 should compare against
  the Triton reference instead of the direct GEMV.
- **HC weight layout flexibility** — add a `stride` parameter and
  explicit mode (vs `W_SHARED` from `numel()`) once a non-trivial
  checkpoint appears.
- **PLE conv state-write semantics** — switch from the left-shift to
  a `next_state = history[..., -state_len:]` gather once we see the
  prefill opt-in path exercised in production (matches the torch
  reference exactly).
- **Bench module** (§5) — once the parity tests are green, run on
  `.176` with the canonical env vars and lock in the row in
  `docs/rdna2/qwen4_exp_hip_path.md`.
