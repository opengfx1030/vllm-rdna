# Explore: INT8 KV cache on the RDNA2 backend — wiring, the measured decode path, and INT8 + FP16 block scales

## The question

Three questions, in the order they have to be answered:

1. Can the RDNA2 backend (FA-RDNA2) serve the `int8_per_token_head` KV dtype
   that vLLM already ships on the Triton path?
2. If it can, what is the cheapest way to read an INT8 cache on gfx1030 — the
   dequantise-to-FP16 pattern the tree ships today, or a native INT8 dot?
3. Is the shipped INT8 format (symmetric INT8, one FP32 scale per token-head)
   the right format, or should the scale become an FP16 block scale?

Status: **this is a design document.** Nothing in this PR changes the serving
path. Every V620 cell in `TESTPLAN.md` is empty on purpose, and every number
below is either read out of the tree or produced by a script that is committed
here or in PR #134. No hardware measurement was taken for this PR.

All `file:line` references are against `origin/rdna_extras` at `30d6e82dcd`.

---

## What already exists

### The dtype ships. It ships on Triton

`int8_per_token_head` is not a new capability:

| Piece | Where |
| --- | --- |
| the dtype literal | `vllm/config/cache.py:54` |
| dtype to torch dtype | `vllm/utils/torch_utils.py:44` (`torch.int8`) |
| the quant mode | `vllm/v1/kv_cache_interface.py:48` (`KVQuantMode.INT8_PER_TOKEN_HEAD`), mapped at `:88` |
| "does this dtype need per-token-head scales" | `vllm/v1/kv_cache_interface.py:132` |
| the Triton backend serves it | `vllm/v1/attention/backends/triton_attn.py:313-321` |
| the writer | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:272` (`triton_reshape_and_cache_flash_per_token_head_quant`) |
| the scale | `absmax / 127` per (token, head), computed in the kernel (`:207`, `:239`), values clamped to `[127, -128]` (`:267`) |
| the scale cache | `vllm/v1/attention/backends/triton_attn.py:405` (`_ensure_scale_caches`) carves FP32 scale views out of the packed cache |
| the cache geometry | per (head, slot) the content dim is `[K(hs) \| K_scale(4 B) \| V(hs) \| V_scale(4 B)]` (`vllm/v1/attention/backends/triton_attn.py:425-427`); the scale views are physical `(num_blocks, block_size, num_kv_heads)` (`:451`, `:460` of the same file) |
| the tests | `tests/quantization/test_per_token_kv_cache.py` (the Triton path) |

So the gap is narrow and specific: **the RDNA2 backend cannot serve a dtype
that the Triton backend already serves.**

### The FA-RDNA2 INT8 kernels are already in the tree

`csrc/rocm/fa_rdna2.cu` carries the whole INT8 FA-RDNA2 stack:

| Op | Where |
| --- | --- |
| `fa_rdna2_decode_paged_int8` (host wrapper) | `:4457` |
| the decode kernel it launches | `fa_decode_paged_splitk_kernel<int8_t, false, true>` — the `IS_INT8` branch of the shared decode template |
| the INT8 load in that kernel | `:443` (D=128) and `:689` (D=256) |
| `fa_rdna2_prefill_paged_varlen_int8` (host wrapper) | `:4270` |
| the INT8 split-K prefill kernels | `fa_prefill_paged_varlen_splitk_kernel_int8_128` / `_256` |
| `reshape_and_cache_int8_rdna2` (the writer) | `:4699` |

**None of those kernels is a native INT8 dot.** The comment at `:2429` calls
the prefill kernel a "native int8 prefill kernel"; it is native in the
*storage* sense (the cache holds int8), not in the *dot* sense:

- The decode path takes the `IS_INT8` branch of the shared FP16-shaped kernel.
  At `:443` it does, per element, `(float)*k_ptr * k_s` and then
  `__float2half_rn`, and writes the result into an FP16 shared tile. The score
  is then a `v_dot2_f32_f16` (`fdot2`) against an FP16 Q (`:472-477`) and the PV
  product is a scalar FP32 FMA over that FP16 tile (`:508-517`).
- The prefill path is the same idea: a packed dword load of K, a per-element
  `v_cvt_f32_i32` + `v_mul_f32` + `v_cvt_f16_f32` promote, and `fdot2` against
  an FP16 Q (`:2588-2621`).

That is worth stating plainly, because PR #134 describes this work as
"native INT8 prefill and decode kernels". It is not. The cache is INT8; the
arithmetic is FP16.

### The primitive is available

gfx1030 has `v_dot4c_i32_i8` — a 4-wide signed int8 dot into int32 — and this
tree already uses it: `__builtin_amdgcn_sdot4` is wrapped at
`csrc/rocm/w4a8_sdot4_rdna2.cuh:125-128` and used by the W4A8 dense and MoE
kernels (PR #93, already on `rdna_extras`).

So an INT8 dot needs no new ISA support. The question is only *where* to use
it, and the answer is not obvious: see the ISA study below.

---

## What is missing or broken

This is the part that was mapped against this tree. It is the difference
between "a backlog item" and "a plan".

### 1. The two read ops are not registered

`csrc/rocm/torch_bindings.cpp` registers only the writer
(`:543-546`), and the comment above it already says why:

> *"Not wired up yet: RDNA_ATTN still rejects quantized KV caches, and
> fa_rdna2_decode_paged_int8 is not registered."*

`fa_rdna2_prefill_paged_varlen_int8` and `fa_rdna2_decode_paged_int8` are
absent. The legacy tree reached these kernels through a `load_inline` JIT
build; this tree loads them through `torch.ops._rocm_C`, so
`csrc/rocm/torch_bindings.cpp` and the shim in
`vllm/v1/attention/ops/fa_rdna2_backend.py` both have to change.

### 2. The backend gate rejects every quantised KV dtype

`vllm/v1/attention/backends/rdna_attn.py:283`:

```python
if get_kv_quant_mode(self.kv_cache_dtype) != KVQuantMode.NONE:
    return False
```

and `supported_kv_cache_dtypes` (`:601-605`) lists `auto`, `float16`,
`bfloat16`. Both have to open for `int8_per_token_head`.

### 3. The writer writes the scale bytes at the wrong address

`csrc/rocm/fa_rdna2.cu:4681` (K) and `:4683` (V):

```c
k_dst[HEAD_DIM * block_size + slot_in_block + d] = ...
```

The declared layout is `[num_blocks, H_kv, HEAD_DIM+4, block_size]` with
`stride_d = block_size` and `stride_s = 1`, so plane `p` occupies
`[p*block_size, (p+1)*block_size)` and the byte of plane `p`, slot `s` lives
at `p*block_size + s`. The four bytes of one slot's FP32 scale must therefore
land in four *planes* of one slot:

```c
k_dst[(HEAD_DIM + d) * block_size + slot_in_block] = ...
```

As written, byte `d` of slot `s` lands in plane `HEAD_DIM`, slot `s + d`.
With `block_size = 16`:

| slot, byte | written to | should be |
| --- | --- | --- |
| s=0, j=0 | plane 128, slot 0 | plane 128, slot 0 |
| s=0, j=1 | plane 128, slot 1 | plane 129, slot 0 |
| s=0, j=2 | plane 128, slot 2 | plane 130, slot 0 |
| s=0, j=3 | plane 128, slot 3 | plane 131, slot 0 |

So 17 of the 19 distinct addresses in a page's scale region are written by
more than one thread (a race), and planes `HEAD_DIM+1 .. HEAD_DIM+3` are never
written at all.

### 4. The writer's cache geometry is not vLLM's geometry

This is the largest unknown in the port, and it is not visible from the op
list alone.

The FA-RDNA2 writer expects **two separate tensors** with a legacy geometry:

```text
key_cache   : [num_blocks, H_kv, D+4, block_size] int8
value_cache : [num_blocks, H_kv, D+4, block_size] int8
kv_cache    : [2, num_blocks, H_kv, D+4, block_size] int8
```

What vLLM allocates for `int8_per_token_head` is **one** packed tensor whose
content dim carries the scale inline, per (head, slot):

```text
[ K(hs) | K_scale(4 B) | V(hs) | V_scale(4 B) ]
```

(`vllm/v1/attention/backends/triton_attn.py:405-427`). The two geometries are
not the same thing, and the RDNA2 backend's own cache plumbing assumes the
FP16 geometry: `vllm/v1/attention/backends/rdna_attn.py:306` calls
`PagedAttention.split_kv_cache(kv_cache.transpose(0, 1), ...)` and
`split_kv_cache` (`vllm/v1/attention/ops/paged_attn.py:25-29`) does

```python
key_cache = kv_cache[0].view(num_blocks, num_kv_heads, head_size // x, -1, x)
```

which only resolves when the content dim is exactly `head_size`. With the
per-token-head scale pad the content dim is `head_size + 4`, and the view does
not resolve.

So the port has to decide, explicitly:

- **(a) Adapt the writer to vLLM's geometry** — write the int8 data into the
  two data planes and the FP32 scale into the inline scale bytes, in the same
  packed tensor, and hand the kernel an int8-aware cache split that strips the
  pad. The Triton backend already does exactly this with `as_strided` views
  over the same allocation (`vllm/v1/attention/backends/triton_attn.py:451`, `:460`), so the pattern
  exists to copy.
- **(b) Give the RDNA2 backend its own cache** for quantised dtypes, and keep
  the legacy geometry. This is a bigger change and it splits the allocator.

(a) is the smaller change and keeps one KV cache for both backends. It is the
one this document assumes.

### 5. The scale index convention differs three ways — and the fix is local

| Party | Scale index |
| --- | --- |
| the FA-RDNA2 int8 kernels | **logical** sequence position — `k_scale_per_tok[n_global * H_kv + h_kv]` (`:443`, `:689`) |
| the legacy Python bridge | **physical** `(block, slot)` |
| vLLM's canonical scale cache (this tree) | **physical** `(num_blocks, block_size, num_kv_heads)` (`vllm/v1/attention/backends/triton_attn.py:451-467`) |

Physical and logical indexing differ whenever a block is partially filled,
which is the normal case under chunked prefill.

The good news: **the decode kernel already knows the physical slot.** At the
load point it addresses the cache with `s_blk[n_local]` (block) and
`s_slot[n_local]` (slot), so the scale lookup becomes

```text
scale_cache[(block * block_size + slot) * H_kv + h_kv]
```

instead of `scale_per_tok[n_global * H_kv + h_kv]`. That is a one-line change
per lookup site and it makes the kernel agree with vLLM's canonical layout by
construction.

### 6. The legacy Python reader is not the inverse of the writer

The legacy bridge did:

```python
k_scale_per_slot = kv_cache[0, :, :, _hs:_hs + 4, :].view(torch.float32)
```

`.view(torch.float32)` reinterprets the **slot** dimension as bytes, not the
byte dimension: it packs four consecutive slots' byte 0 into one FP32. Even
with the writer fixed, that does not recover the scales. Under vLLM's
geometry the scale is read with an `as_strided` view of the packed tensor
(`vllm/v1/attention/backends/triton_attn.py:451`), which is the correct shape of the answer.

### 7. The legacy evidence is a throughput number

The legacy tree's end-to-end bench reports 2.93 output tok/s against a 2.91
tok/s FP16 baseline at 5.89 GiB of KV cache. That measurement cannot
distinguish a correct scale from a corrupted one — and with the writer defect
above, the scales *are* corrupted, and a dequantised K of ~0 still produces
plausible tokens per second.

Upstream has a writer-to-reader test for the **Triton** path
(`tests/quantization/test_per_token_kv_cache.py`). Nothing exercises the
FA-RDNA2 writer-to-reader path, so the numerical correctness of this path has
never been established. The port has to bring a correctness test that has never
existed, not re-apply something that worked.

### 8. Smaller items

- The int8 ops return a `Tensor` while every other `fa_rdna2` op writes into
  an `out` parameter. Worth normalising while the binding is written.
- The writer takes an int32 `slot_mapping` (`TORCH_CHECK` at `:4710`); vLLM
  passes int64.
- `csrc/rocm/ops.h` declares `reshape_and_cache_int8_rdna2` twice (`:458` and
  `:763`).
- The legacy int8 work is a seven-commit chain, not one commit, so it should
  be reviewed as a series.

---

## The native INT8 implementation

### What the ISA study already measured

There is a measured answer to most of this question, in the parent project's
ISA study (`.omo/plans/rdna2-int8-isa-benchmark/isa-report.md`, V620). It
benchmarked six dequant/dot patterns for the decode QK product at D=128 and
D=256:

| Variant | What it does | D=256, seq=16384 | D=128, seq=1024 |
| --- | --- | ---: | ---: |
| `fp16_ref` | the FP16 ceiling | 86.3 out/s (1.000×) | 7679 (1.000×) |
| `v0` | the `__hmul` baseline | 83.7 (0.970×) | 5022 (0.654×) |
| `v1` | int8→fp16 then `v_dot2` | 75.7 (0.877×) | 7184 (0.935×) |
| `v2` | **`v_dot4c_i32_i8`** (`sdot4`) | 134.5 (1.560×) | 8548 (**1.113×**) |
| `v3` | packed dword load + VGPR dequant + `v_dot2` | 190.4 (**2.207×**) | 7618 (0.992×) |
| `v3_128` | v3 with 128 threads/CTA | 7715 (89.4×) | 178618 (23.3×) |

Three things follow, and they are the most useful thing in this document:

1. **The ISA study's implementation choice is `v3`, not the native int8 dot.**
   Its verdict: "the wiki's 'live contract' pattern (v3) is the implementation
   choice for downstream Todos 2 and 4", with `v2` (sdot4) "rejected as the
   implementation choice — it wins the single-thread microbench at D=128 but
   loses at D=256 and we lose SDOT4's single-thread advantage when we
   parallelize across threads (which production does with 128 thr/CTA)".
2. **The pattern the fork ships today is the measured loser.** The fork's
   `IS_INT8` decode branch (`:443`) is a per-element scalar dequantise into an
   FP16 shared tile — the `v1`/`v0` family. The study measured that family at
   0.877× of the FP16 ceiling (v1, D=256) and 0.654× (v0, D=128), and says
   of the legacy baseline: "The current `fa_rdna2_decode_paged_int8` is
   leaving 30%+ on the table at the typical Qwen3.x head dim (128)."
3. **The one native-dot opportunity nobody has measured is the PV product.**
   The study says so twice: "The PV (output) update is scalar in all variants.
   A future optimization could quantize P to int8 and use `v_dot4c_i32_i8` for
   PV too — not measured here."

A caveat the study states itself, which this document carries: the microbench
is **single-thread and compute-bound**, so those ratios are per-CU ISA
efficiency, not the memory saving. Its own expectation for production is
"somewhere between the single-thread and 128-thread microbench numbers, with
the int8-vs-fp16 ratio holding around 1.1×".

### So "native INT8" splits into three pieces

**N1 — port the measured `v3_128` pattern into the fork's decode kernel.**
This is not an int8 dot; it is the opposite: keep the dequantise in VGPRs
(packed dword load → `v_cvt_f32_i32` → `v_mul_f32` → `v_cvt_f16_f32`), feed
`fdot2`, and do not materialise an FP16 shared tile at all. It is the study's
own implementation choice, and against the pattern the fork ships it is the
difference between the measured 0.654× and the measured 2.207× at the study's
primary shape.

The fork's decode kernel has a *third* structure (128 threads, one KV row per
thread, an FP16 shared tile, a per-thread D slice for PV), so the study's
ratios do not transfer directly — the pattern has to be re-measured in this
structure. That re-measurement is H2 below.

**N2 — a native INT8 dot for the PV product.** The open item. The shape of
it:

- The per-(token, head) V scale is per *k*, so it folds into P rather than
  into the cache: `P'[k] = P[k] * s_V[k] / s_Vmax` over the block, with
  `s_Vmax` the block maximum, applied once per block at the end. Folding the
  raw scale into P instead would underflow FP16 for small scales, so the
  normalisation by the block maximum is required, not cosmetic.
- P is then quantised to int8 against the block's running softmax maximum
  (which the online softmax already tracks), and the PV product becomes
  `v_dot4c_i32_i8` with an int32 accumulator, scaled per block.
- The structural cost: the decode PV loop gives each thread one output
  dimension `d` and walks `k`, so a 4-wide int8 dot needs four *consecutive k*
  for one `d`. In the current `[k][d]` shared layout that is a strided read;
  the tile has to be transposed (or the dot done as a strided
  `v_dot4c` per lane with a shuffle). This is a real kernel-design item, not
  a one-liner.
- The accuracy cost: P drops from FP16 to INT8. That is a probability-precision
  question, and it has to be measured on attention output error (H3).

**N3 — `v_dot4c_i32_i8` for the QK product.** Recorded here so that it is not
re-litigated: it was implemented and measured, and it was rejected for the
QK dot at production occupancy. The primitive is real (`v_dot4c_i32_i8` is
what `__builtin_amdgcn_sdot4` lowers to — no scalar `s_dot4` is emitted) and
it is the right primitive for the *PV* product (N2), where the operands can
both be int8 without quantising the query.

### How the block scale changes the native path

With a per-(token, head) scale the QK score is one scaled dot:

```text
score[k] = s_Q * s_K[k] * dot_int32(Q_q, K_q[k])
```

With a block scale along the head axis it becomes one dot per block, each
scaled on its own:

```text
score[k] = s_Q * sum_b s_K[k][b] * dot_int32(Q_q[b], K_q[k][b])
```

The dot instruction count is unchanged (32 `v_dot4c` per 128 elements either
way); what changes is that the per-block scale has to be applied to a partial
result. That is 3 extra multiplies and 3 extra adds per score, i.e. about
0.06 ops per element — noise next to the 0.25 ops per element the dot itself
costs. **A block scale is essentially free for the native path**, which is
what makes the format change in the next section compatible with the speed
work rather than in tension with it.

---

## The INT8 + FP16 block scale format

### The format

| Field | Value |
| --- | --- |
| Element | INT8, symmetric, `q` in `[-127, 127]` |
| Block scale | FP16 (E5M10), one per `k` values along the head dimension |
| Zero-point | none (see below) |
| Global scale | none |
| Bits/value | `8 + 16/k`; at `k=32` that is **8.500** |
| Dequant | `x_hat = q * scale` |

### What it buys, measured

The numbers below come from PR #134's `reference_codec.py`, run for this
document on the same synthetic rows it uses (64 rows of `N(0, 0.5)`,
`head_size=128`, two x40 outlier channels for K and two x40 outlier tokens for
V). **They are arithmetic on synthetic data, not V620 measurements.**

| Format | bits/value | K-side dB | V-side dB |
| --- | ---: | ---: | ---: |
| `int8_per_token_head` (shipped: symmetric, one FP32 scale per row of 128) | 8.250 | 33.415 | 44.022 |
| `int8_fp16_block` k=128 (same granularity, FP16 scale) | **8.125** | 33.415 | 44.027 |
| `int8_fp16_block` k=64 | 8.250 | **36.425** | **44.764** |
| `int8_fp16_block` k=32 | 8.500 | **39.475** | **45.594** |

Reading it:

- **At the same granularity, FP16 is free.** Swapping the shipped FP32 scale
  for an FP16 scale at one scale per row costs 0.000 dB on the K side and
  0.005 dB on the V side, and saves 0.125 bits/value — 1.5 % less KV memory
  for nothing. This is the same result PR #134 reached for INT4 ("FP32 buys
  between 0.000 and 0.05 dB and costs 0.5 bits/value").
- **At `k=64`, the block scale pays for itself.** One FP16 scale per 64
  values along the head axis costs exactly what the shipped format costs
  (8.250 bits/value) and buys **+3.01 dB on the K side and +0.74 dB on the V
  side**, because the scale is now finer than one-per-row.
- **At `k=32` it costs 0.25 bits/value for +6.06 dB on K and +1.57 dB on V.**
  That is the same trade the INT4 proposal makes at its own budget.

The K side is where the granularity matters: a per-row scale gives one loud
channel the whole row, which is exactly the channel-outlier failure KIVI
describes for K. The V side is much less sensitive (44.022 → 45.594 across
the whole range) because whole-token outliers, not channels, dominate it.

### Why the head-dimension axis and not the token axis

PR #134 measured both axes for INT4 and the crossover is sharp:

| Grouping | K-side dB | V-side dB |
| --- | ---: | ---: |
| head-dimension, k=32 | 20.302 | 21.814 |
| token-axis, k=32 tokens | **21.778** | 4.829 |

Token-axis grouping is 1.48 dB better on K and catastrophic on V, and it is
also unworkable for a paged cache: a token-axis scale spans `k` tokens, so it
cannot be finalised until all `k` have been written, and chunked prefill can
land a chunk boundary inside a page — at which point the scale would have to
be recomputed over already-quantised data. Head-dimension grouping has no
such problem: a token's row is complete the moment it is written. That is why
every per-token-head format upstream is head-dimension granular.

### What does not port from INT4: the packed zero-point

PR #134's headline trick is that INT4's 4-bit zero-point rides in the low
mantissa bits of the FP16 block scale, so the format gets a second level of
range for free. **That trick does not port to INT8.** An INT8 zero-point
needs 8 bits; an FP16 mantissa has 10, so packing it would leave 2 bits of
mantissa for the scale itself.

So `int8_fp16_block` is symmetric, and the question becomes: how much is the
asymmetry worth at 8 bits? The codec's answer for INT4 was that it was worth
2.0 dB (14.570 → 20.302 dB), because INT4's 16 codes are coarse enough that
wasting half of them on a symmetric range is expensive. At 8 bits the
symmetric range is already tight — the shipped symmetric INT8 measures
33.415 dB against FP8 e4m3's 32.968 dB — so the headroom for asymmetry is
much smaller. A separate uint8 zero-point per group would cost +0.25
bits/value (8.750 total).

**This is the open question of the format**, and it is measurable: add an
asymmetric INT8 variant to the reference codec and compare it against the
symmetric one at `k=32` and `k=64`. If the gain is under ~0.5 dB, the format
stays symmetric and the 0.25 bits/value is never spent.

### The budget question, stated honestly

PR #134's conclusion for the 8-bit end of the table applies here too: FP8
e4m3 with a per-tensor scale already ships, costs 8.000 bits/value, and is the
only KV dtype with a measurement on the V620 (about 2× the memory saved and
+0.1 PPL). A new 8-bit format that costs *more* than that has to earn its
place on something other than memory.

What it can offer instead:

- **1.94× the context of FP16** at 8.250 bits/value, against FP8's 2.0× —
  nearly the same, so memory is not the argument.
- **A decode path that is cheaper to read than the FP8 one.** FP8 on gfx1030
  has no hardware decoder: the shipped kernel decodes e4m3 in software
  (sign/exponent/mantissa, a subnormal branch, a select — about 7 ops per
  element, `csrc/rocm/fa_rdna2.cu:203-226`). INT8 with a native dot has no
  such tax. That is the real argument for INT8 over FP8 on this chip, and it
  is the one this document is about.
- **Better accuracy than FP8 at a similar budget**: 36.4 dB (k=64) or 39.5 dB
  (k=32) against FP8's 33.0 dB, on the same synthetic rows.

So the INT8 case on RDNA2 is *not* "more context than FP8". It is "FP8-class
context with a cheaper read path and more accuracy", which only pays off if
the read path is actually made cheap — which is what N1 and N2 are for.

---

## Where it would plug in

Nothing below exists yet. This is the surface an implementation would have to
touch.

| Piece | File | What changes |
| --- | --- | --- |
| op registration | `csrc/rocm/torch_bindings.cpp:543-546` | register `fa_rdna2_prefill_paged_varlen_int8` and `fa_rdna2_decode_paged_int8` |
| binding shim | `vllm/v1/attention/ops/fa_rdna2_backend.py` | the `fa_rdna2_*` wrappers the backend calls |
| backend gate | `vllm/v1/attention/backends/rdna_attn.py:283`, `:601-605` | open for `int8_per_token_head` |
| cache split | `vllm/v1/attention/backends/rdna_attn.py:306` + `vllm/v1/attention/ops/paged_attn.py:25-29` | an int8-aware split that strips the scale pad and exposes the scale views |
| writer | `csrc/rocm/fa_rdna2.cu:4699` | write into vLLM's packed geometry; fix the scale offset at `:4681`/`:4683` |
| scale lookup | `csrc/rocm/fa_rdna2.cu:443`, `:689` | physical `(block, slot, head)` instead of logical position |
| decode kernel | `csrc/rocm/fa_rdna2.cu:443` | the `v3_128` pattern (N1), then the native PV (N2) |
| new dtype (format) | `vllm/config/cache.py`, `vllm/utils/torch_utils.py`, `vllm/v1/kv_cache_interface.py:48` | a `KVQuantMode` for `int8_fp16_block`, and the page-size arithmetic for the scale bytes |

---

## Hypotheses and kill criteria

| # | Hypothesis | Check | Holds if | If not |
| --- | --- | --- | --- | --- |
| H1 | The FA-RDNA2 INT8 path is numerically correct once the writer offset and the scale index are fixed | write K/V through `reshape_and_cache_int8_rdna2`, read the scale back out of the cache, assert it equals the per-(token, head) `amax/127` the writer computed; then compare the attention output against an FP16 reference | the scale round-trips exactly and the attention output matches FP16 within the tolerance the Triton path already meets | stop: the path is not usable and the dtype stays Triton-only |
| H2 | The `v3_128` pattern is faster in *this* kernel structure than the shipped per-element dequantise | A/B the decode kernel on the V620 at D=128 and D=256, seq 4k/16k, against the FP16 path | decode tok/s at c=1 improves against the FP16 baseline and against the shipped INT8 path | keep the shipped pattern and record the negative result |
| H3 | A native INT8 PV product does not cost accuracy | quantise P to int8 per block and compare attention output error against the FP16 reference and the FP16-P path | the output error stays within the band the FP16-P path has | drop N2; the native dot stays on the shelf and INT8 keeps its FP16 dequantise |
| H4 | The FP16 block scale buys accuracy at no memory cost | compare `int8_fp16_block` k=64 against the shipped format on captured V620 activations, at equal bits/value | k=64 improves the attention output error against `int8_per_token_head` at 8.250 bits/value | the format is not worth a new dtype; keep the shipped per-token-head scale |
| H5 | The INT8 asymmetry is not worth 0.25 bits/value | extend the reference codec with an asymmetric INT8 block variant and compare | the asymmetric variant is within ~0.5 dB of the symmetric one | the zero-point earns its bytes, or it does not — decide before any kernel work |

---

## Order of work

1. **Fix the writer offsets** (`:4681`, `:4683`) and write the writer-to-reader
   correctness test (H1). This is the cheapest step and it unblocks everything
   else; it is also the step that has never been done.
2. **Pick the cache geometry** (item 4 above) and make the writer, the reader
   and the cache spec agree on it. The scale index convention falls out of
   this for free: physical `(block, slot, head)` everywhere.
3. **Register the two ops**, wire the binding shim, open the backend gate.
4. **A/B `int8_per_token_head` against FP16 on the V620**: NIAH / long-context
   probes, PPL, KV capacity, the 16k/1k and 32k cells — with output-token
   parity against the FP16 baseline, not just throughput (H2).
5. **Port the `v3_128` pattern** into the decode kernel and re-measure (H2).
6. **Only then** decide whether the native PV (N2) and the `int8_fp16_block`
   format (H4, H5) are worth building, and in that order: the format needs a
   format decision, the native PV needs a kernel decision, and the format
   decision is cheaper.

---

## Prior work, and why this is not a duplicate

Three things already exist in this area and none of them is superseded here:

- **The shipped Triton INT8 path.** `int8_per_token_head` on the Triton
  backend (`vllm/v1/attention/backends/triton_attn.py:321`). This document is about the RDNA2 backend
  serving the same dtype, not about replacing the Triton path.
- **PR #134, "[explore] INT4 + FP16 block scales for the RDNA2 KV cache".**
  A different element width at a different budget (4.500 bits/value). This
  document's format is its INT8 sibling and deliberately reuses its codec,
  its measurements and its reasoning about grouping axes; it does not fork
  them. Where the two disagree — the packed zero-point, which does not port
  to INT8 — this document says so.
- **The INT8 ISA study** in the parent project
  (`.omo/plans/rdna2-int8-isa-benchmark/isa-report.md`), and the legacy
  `humanwork-legacy` commit `3baecdb516` it studied. This document does not
  re-run that study; it uses its verdict (v3 is the implementation choice,
  sdot4 is rejected for QK, the PV product is unmeasured) as the starting
  point and extends it.

And one thing that is *not* prior work but might look like it:

- **PR #93, the W4A8 `sdot4` dense and MoE kernels.** Same instruction
  (`v_dot4c_i32_i8`), same tree, different tensor: that PR quantises
  *activations* to INT8 for a weight GEMM; this one reads an INT8 *cache*.
  The reusable part is the primitive and the helper at
  `csrc/rocm/w4a8_sdot4_rdna2.cuh:125-128`.

---

## What this PR contains

- `docs/explore/kv-int8-native/README.md` — this document.
- `docs/explore/kv-int8-native/TESTPLAN.md` — the measurement plan, with the
  V620 cells left empty.
- `docs/explore/kv-int8-native/check_citations.py` — resolves every
  `file:line` citation in this document against a base commit and prints the
  cited source, so a stale reference is visible. Run it with
  `python3 docs/explore/kv-int8-native/check_citations.py`; it exits non-zero
  if a citation does not resolve.

## What this PR does not contain

- No dtype, no backend change, no kernel. Nothing in the serving path changes.
- No performance claim. Every V620 cell is empty; the only throughput numbers
  quoted are from the ISA study, which measured dequant/dot *patterns* in a
  different harness and said so.
- No accuracy claim for the new format. The SNR figures are arithmetic on
  synthetic rows; the format's acceptance criterion is attention output error
  on captured activations, which is H4 and is unmeasured.
- No decision on the zero-point. It is H5, and it is measurable before any
  kernel work.

---

## Reproducing the numbers

The INT8 SNR rows come from PR #134's codec, which is not on this branch.
From a checkout that has it:

```bash
python3 - <<'PY'
import random
import sys

sys.path.insert(0, "docs/explore/kv-int4-fp16-scales")
import reference_codec as rc

for outlier, name in (("channel", "K-side"), ("token", "V-side")):
    rng = random.Random(42)
    rows = rc._make_rows(rng, 64, 128, outlier)
    for label, quant, bits in (
        ("int8_per_token_head (shipped)",
         lambda r: rc.quantize_int8_fp32_block(r, k=128),
         rc.bits_per_value(8, 32, 128)),
        ("int8_fp16_block k=128",
         lambda r: rc.quantize_int8_fp16_block(r, k=128),
         rc.bits_per_value(8, 16, 128)),
        ("int8_fp16_block k=64",
         lambda r: rc.quantize_int8_fp16_block(r, k=64),
         rc.bits_per_value(8, 16, 64)),
        ("int8_fp16_block k=32",
         lambda r: rc.quantize_int8_fp16_block(r, k=32),
         rc.bits_per_value(8, 16, 32)),
    ):
        result = quant(rows)
        snrs = [rc.snr_db(row, rec) for row, rec in zip(rows, result.reconstructed)]
        print(f"{name:8s} {label:32s} {sum(snrs)/len(snrs):8.3f} dB  {bits:.3f} bits")
PY
```

The `int8_fp16_block k=32` row it prints matches PR #134's published table
(39.475 dB K-side, 45.594 dB V-side, 8.500 bits/value), which is how the
numbers in this document were checked.

---

## AI assistance

This draft was written with AI assistance. Every `file:line` reference above
was checked against `origin/rdna_extras` at `30d6e82dcd`, and the numbers come
from the committed codec rather than from the author. No hardware measurement
was taken for this PR: the V620 cells in `TESTPLAN.md` are empty on purpose.
