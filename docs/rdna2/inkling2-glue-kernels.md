# Inkling-2-Mini glue kernels (RDNA2 / gfx1030)

Native HIP kernels for everything a layer visit of Inkling-2-Mini does that is
not a GEMM, an attention op, a short convolution or an expert GEMM.  They
replace the ~130 tiny torch launches per visit of `amd/blocks.py` /
`amd/moe.py` (block / message attention residual, message gates, RMS norms,
router + `moe_align_block_size`, softcapped SwiGLU, zero fills) by 4 to 9
CUDA-graph-capturable launches.

* Source: `csrc/rocm/inkling2_glue_rdna2.cu` (namespace `inkling2_glue`)
* Python: `vllm/_custom_ops.py` (`inkling2_glue_*`), ops in `torch.ops._rocm_C`
* Tests: `tests/kernels/core/test_inkling2_glue_rdna2.py`
* Bench: `tools/rdna/inkling2/bench_glue.py`
* Spec this implements: `docs/rdna2/inkling2-port-spec.md`

Hardware assumptions (V620, `gfx1030`, wave32): no bf16 / fp8 / fp4 dot
instructions; `V_DOT2_F32_F16` (`__builtin_amdgcn_fdot2`) is the matrix
primitive.  Activations are fp16, the residual stream and the block states are
fp32 exactly as in the torch model.  Nothing in this file touches torch,
rocBLAS, hipBLAS or Triton at run time.

## 1. Ops and what they replace

| op | replaces (torch, `blocks.py` / `moe.py`) | launches |
|---|---|---|
| `inkling2_glue_attn_res` | `BlockAttentionResidual` / `MessageAttentionResidual` (`cat`, `UnweightedRMSNorm`, `_attend_over_stack`), the lazy `hidden + attn` add, the block-state append, the per-query `Inkling2RMSNorm(...).to(fp16)` | 12-33 -> 1 |
| `inkling2_glue_msg_gate_down` + `_mix` | `MessageGate` (two norms, `cat`, fp16 `down_proj`, `.float()`, two fp32 linears, `tanh`, mix) and the layernorm after it | 29-87 -> 2 |
| `inkling2_glue_rms_norm` | `Inkling2RMSNorm` / `UnweightedRMSNorm` with optional fp32 add and fp16 cast | 8 -> 1 |
| `inkling2_glue_embed_norm` | `F.embedding` + `embed_norm` + `.float()` | 10 -> 1 |
| `inkling2_glue_moe_route` | fp32 router GEMV, `sigmoid`, `topk`, `gather`, normalisation, `routed_scaling_factor`, `moe_align_block_size` (3 kernels), `torch.zeros` of both accumulators | 15 -> 1 |
| `inkling2_glue_swiglu` | `softcapped_swiglu` (`chunk`, two `tanh`, `sigmoid`, 4 muls, casts) | 12 -> 1 |
| `inkling2_glue_copy_states` | snapshot of persistent fp32 state (section 4.3) | 1 |

(The "launches" column is measured by `bench_glue.py` with the torch profiler,
see section 6.)

Not in this file, because another kernel file already fuses them: the four
short convolutions, KV write / paged attention, the q / k / rel scaled L2 clip,
log scale, v norm and the sigmoid output gate (`inkling2_sconv_*`,
`inkling2_kv_prep`, `inkling2_attention`), the dense linears, the router-free
GEMVs and the lm_head softcap (`inkling2_linear_*`).

All ops are per-token and row independent: no cross-token state, so prefix
caching, chunked prefill, speculative decoding and TP replicas of the fp32
stream need nothing special.  Every op launches on
`at::cuda::getCurrentCUDAStream()`, does not allocate, does not synchronise,
and uses only static shapes and pointers (capturable by `torch.cuda.graph`).

## 2. Op reference

Conventions: `T` tokens, `H` hidden (2304), `R` gate rank (64), `G` gate
streams, `E` experts (128), `K` top-k (8), `I` intermediate (1152).  "aligned"
means: unit stride on the last dim, every other stride a multiple of 4 fp32
elements (16 B) and a 16-byte aligned base pointer.  Out-of-range arguments
raise `RuntimeError` from `TORCH_CHECK` (tested).

### 2.1 `inkling2_glue_attn_res`

```python
ops.inkling2_glue_attn_res(blocks, n_blocks, x0, query, out, *, x1=None,
    partial_out=None, append=None, norm_w=None, eps=1e-6, norm_eps=1e-6)
```

One CTA per token.  The stack is `blocks[:, :n_blocks]`, then `append` (if
given), then the partial state `x0 (+ x1)`.  Every state is RMS-normalised
without weight (`eps`), then for each query `q`:
`out[:, q] = softmax_s((query[q] . s_hat) * H**-0.5) @ s_hat` (normalised
states are the keys and the values, fp32 math).  With `norm_w` the result is
additionally `norm_w[q or 0] * rms_norm(out[:, q])` (`norm_eps`) and written
as fp16.

| arg | shape / dtype | notes |
|---|---|---|
| `blocks` | `[T, S, H]` fp32, aligned | `S >= n_blocks (+1 with append)`; empty tensor when `n_blocks == 0` and no `append` |
| `x0` | `[T, H]` fp32, aligned | partial state (or its first summand) |
| `x1` | `[T, H]` fp32, aligned, optional | lazy residual: partial = `x0 + x1` |
| `partial_out` | `[T, H]` fp32, aligned, optional | receives `x0 + x1` |
| `append` | `[T, H]` fp32, aligned, optional | also copied to `blocks[:, n_blocks]` |
| `query` | `[Q, H]` fp32, `Q in {1, 3, 4}` | `Q=3`: full layers, `Q=4`: messages |
| `norm_w` | `[1 or Q, H]` fp16 / fp32, optional | enables the fused norm, `out` must then be fp16 |
| `out` | `[T, Q, H]` fp32 (raw) / fp16 (with `norm_w`), aligned | **any strides**: a stream-major `[Q, T, H]` buffer can be passed as `buf.permute(1, 0, 2)`, so each stream `buf[q]` is a contiguous `[T, H]` fp16 GEMM input |

Limits: `H % 4 == 0`, `H <= 4096`, total states (`n_blocks + append + 1`)
`<= 64`.  `partial_out` and `out` must not alias any input (`x0` is read
again after `partial_out` is written; aliasing would add `x1` twice).

### 2.2 `inkling2_glue_msg_gate_down`

```python
ops.inkling2_glue_msg_gate_down(h, m, wd, d, eps=1e-6)
```

`d[g, t] = fp16( wd[g] @ fp16([rms(h[t, g]), rms(m[t, g])]) )` (unweighted
norms with `eps`, the concatenated input rounded to fp16, fp16 weights, V_DOT2
with fp32 accumulation, fp16 result): the `MessageGate.down_proj` on the same
numbers as torch.

| arg | shape / dtype | notes |
|---|---|---|
| `h` | `[T, G or 1, H]` fp32, aligned | `h.size(1) == 1` broadcasts one hidden to all gates (sliding layers); may be a view such as `raw[:, :1]` |
| `m` | `[T, G, H]` fp32, aligned | e.g. `msgs[:, :3]` of the `[T, 4, H]` messages |
| `wd` | `[G, R, 2H]` fp16, contiguous | `torch.stack([g.down_proj.weight for g in gates])` |
| `d` | `[G, T, R]` fp16, contiguous | output |

`R` even and `<= 256`, `2H % 8 == 0`, `H <= 4096` (each wave holds a whole row
of `h` / `m` in registers while it normalises it), LDS tile `<= 64 KiB`.

### 2.3 `inkling2_glue_msg_gate_mix`

```python
ops.inkling2_glue_msg_gate_mix(h, m, d, wf, wg, out, *, norm_w=None,
    scratch=None, partials=None, counters=None, norm_eps=1e-6)
```

`out = (1 - tanh(Wf d)) * h + tanh(Wg d) * m` (fp32 math on the fp16 `d`).
With an fp16 `out` it also applies the weighted RMSNorm `norm_w` (the layer
norm after the gate) and rounds to fp16; the whole-row reduction is finished
by the last-arriving CTA of each (gate, token tile) (`__threadfence` +
`atomicAdd` on `counters`, which the kernel resets to zero).

| arg | shape / dtype | notes |
|---|---|---|
| `d` | `[G, T, R]` fp16, contiguous | from `msg_gate_down` |
| `wf`, `wg` | fp16 `[G, R/2, H, 2]` (serving) or fp32 `[G, R, H]`, contiguous | repack in section 4.1 |
| `out` | `[G, T, H]`, contiguous | fp32: plain mix; fp16: mix + norm |
| `norm_w` | `[1 or G, H]` fp16 / fp32 | required for fp16 `out` |
| `scratch` | fp32, contiguous, `>= G*T*H` elements | fp16 `out` only |
| `partials` | fp32, contiguous, `>= G*T*ceil(H/128)` | fp16 `out` only |
| `counters` | int32, contiguous, `>= G*ceil(T/TB)`; `G*T` always suffices | zero on entry, left zero on exit; **zero them once at allocation** |

`TB` (tokens per CTA, 1 to 16) is chosen from `T` and `G` so that at least 216
CTAs are in flight.  The fp16 weights hold the bf16 checkpoint values exactly
for `|w| >= 2**-14`.

### 2.4 `inkling2_glue_rms_norm`

```python
ops.inkling2_glue_rms_norm(x, out, *, weight=None, x1=None, sum_out=None,
    eps=1e-6, round_fp16=False)
```

`out = weight * rms_norm(x + x1)` (fp32 math); `x` fp16 / fp32 `[T, H]`, `x1`
fp32 (lazy add), `sum_out` fp32 receives `x + x1`, `weight` `[H]` fp16 / fp32,
`out` fp16 / fp32; `round_fp16` rounds an fp32 `out` to the fp16 grid (the
reference casts the norm result to the activation dtype).  No weight: the
plain `UnweightedRMSNorm`.

### 2.5 `inkling2_glue_embed_norm`

```python
ops.inkling2_glue_embed_norm(ids, table, weight, out, eps=1e-6)
```

`out[t] = fp16(weight * rms_norm(table[ids[t]]))`; `ids` int32 / int64 (clamped
to the table), `table` fp16 `[V, H]` (row stride aligned), `out` fp16 or fp32
`[T, H]`.  Multimodal `inputs_embeds` go through `rms_norm(..., round_fp16=True)`.

### 2.5a `inkling2_glue_embed_lookup`

```python
ops.inkling2_glue_embed_lookup(ids, table, out, vocab_first, vocab_total)
```

This rank's rows of a vocabulary-sharded embedding (tensor parallelism,
`inkling2-tensor-parallel.md`):
`out[t] = table[clamp(ids[t], 0, vocab_total - 1) - vocab_first]` when that row is in
the shard (`table` holds the ids `[vocab_first, vocab_first + rows)`), zeros
otherwise.  `ids` int32 / int64, `table` and `out` fp16 with 16-byte aligned rows
(`H % 8 == 0`).  One CTA per token copies 16-byte vectors.  The rows are an exact copy,
so the all-reduce of the group (one
non-zero contributor per row, `x + 0 = x`) rebuilds the full lookup, and
`rms_norm(rows, out, weight=embed_norm, round_fp16=True)` then equals
`inkling2_glue_embed_norm` bit for bit (same code on the same values; tested by
`test_embed_lookup_then_rms_norm_is_embed_norm`).

### 2.6 `inkling2_glue_moe_route`

```python
n_pad, n_blk, n_cnt = ops.inkling2_glue_moe_route_sizes(T, E, K, BM)
ops.inkling2_glue_moe_route(x, gate_w, bias, logits, topk_weights, topk_ids,
    sorted_ids, expert_ids, post_pad, counters, BM, route_scale,
    *, zero1=None, zero2=None)
```

One launch: router logits (`x @ gate_w^T`), `sigmoid`, top-k of
`scores + bias` (ties: lowest expert id, NaN rows still yield valid ids),
`topk_weights = scores / (sum + 1e-20) * route_scale`, and the
`moe_align_block_size` layout of the `T*K` routed rows (stable order inside an
expert, `numel` as pad id, `-1` for unused blocks), plus zero fills of two fp16
buffers (the expert GEMMs accumulate into zeroed outputs).

| arg | shape / dtype | notes |
|---|---|---|
| `x` | `[T, H]` fp16, 16-B aligned rows | the MoE input |
| `gate_w` | `[E, H]` fp16 (V_DOT2) or fp32 (FMA), contiguous | |
| `bias` | `[E]` fp32 | selection bias (`e_score_correction_bias`) |
| `logits` | `[T, E]` fp32 | output, pre-sigmoid |
| `topk_weights` / `topk_ids` | `[T, K]` fp32 / int32 | outputs |
| `sorted_ids`, `expert_ids` | int32, `n_pad` / `n_blk` elements | **exactly** the sizes of `_sizes()`: slice the persistent buffers `buf[:n_pad]` |
| `post_pad` | int32 `[1]` | padded row count |
| `counters` | int32, `>= T + 1`, zero on entry, left zero | zero once at allocation |
| `zero1`, `zero2` | fp16, contiguous, 16-B aligned, `numel % 8 == 0` | `gate_up [T*K, 2I]` and the MoE output `[T, H]` |

Limits: `T*K <= 8192` (the serving `max_tokens_per_call = 1024` with `K = 8`),
`E <= 255`, `K <= 16`, `BM in {1, 2, 4, 8}`.

### 2.7 `inkling2_glue_swiglu`

```python
ops.inkling2_glue_swiglu(gate_up, out, gate_cap, up_cap)
```

`out = tanh(g/gc)*gc * sigmoid(g) * tanh(u/uc)*uc` with `[g | u] = gate_up`
(`[R, 2I]` fp16 contiguous, gate columns first), `out` `[R, I]` fp16
contiguous; `gc = 4`, `uc = 25` for this model.

### 2.8 `inkling2_glue_copy_states`

```python
ops.inkling2_glue_copy_states(src, dst)
```

Bit-exact copy of an fp32 `[T, H]` or `[T, P, H]` tensor (aligned, own strides
on each side, same shape, no aliasing).  Used for the snapshots of section
4.3.

### 2.9 How the kernels are organised

All kernels are wave32, launch on the current stream with a grid that is a
function of the shapes only, use no scratch other than what the caller passes
and accumulate in fp32.  `kMinCtas = 216` (3 per CU of a 72-CU V620) is the CTA
count the tilings aim for.

* `attn_res`: one CTA per token; every thread owns `ITER` float4 of the row.
  The first six states of the stack stay in registers between the two passes
  (pass 1: `|s|^2` and `q . s` per state; pass 2: the weighted sum of the
  *normalised* states); the `x0 + x1` partial, the append and the weighted
  output norm are done by the same CTA.
* `msg_gate_down`: CTA = (gate, tile of `TB <= 4` tokens, slice of output rows),
  4 waves.  The tile `fp16([norm(h), norm(m)])` is staged in LDS by
  `stage_unit_row` (all loads of a row are in flight before the first use); each
  wave then owns whole output rows and runs `v_dot2_f32_f16` against the fp16
  weight row.
* `msg_gate_mix`: CTA = one wave = (gate, tile of `TB <= 16` tokens, 128-column
  chunk), each lane owns 4 columns; the fp16 weights are pair-interleaved so one
  16-byte load feeds four `v_dot2` per accumulator.  With `norm_w` the CTAs of a
  tile publish per-chunk sums of squares and the last-arriving CTA (counter +
  `__threadfence`) normalises the tile's rows.
* `moe_route`: the first `min(ceil(bytes / 4096), 32)` CTAs zero the two fp16
  buffers; the other `NT x NEG` CTAs stage `TB` tokens in LDS and each wave
  computes expert rows (V_DOT2 for fp16 weights, FMA for fp32).  The last of the
  `NEG` CTAs of a tile runs the top-k (one wave per token, `K` argmax rounds,
  ties to the lowest id); the last tile runs the align, a stable counting sort:
  every wave histograms a contiguous range of the routed rows (equal experts
  found with an 8-vote ballot match), the expert block layout and the start of
  each wave's share inside an expert follow, and the scatter repeats the walk
  so the order inside an expert stays `(token, k)`.  The counters reset
  themselves.
* `swiglu`, `rms_norm`, `embed_norm`, `copy_states`: one CTA per row, 16-byte
  accesses.

Per-op timings, the effect of each tiling decision and the verification are in
sections 6 and 7.

## 3. Per-visit launch sequence

`x0 (+ x1)` is the lazily added residual stream: the previous layer returns
its residual and its post-MLP convolution output separately and the next
layer's first attention residual adds them (`partial_out` materialises the sum).

| visit | launches (glue only) |
|---|---|
| sliding / full, no messages | 1 `attn_res(norm_w=ln)` -> attention + sconv -> 1 `attn_res(norm_w=post_ln)` -> 1 `moe_route` -> `gemm1` -> 1 `swiglu` -> `gemm2` -> sconv = **4** |
| with messages (passes >= 1 of the loop) | + 1 `attn_res` (Q=4 messages) + `attn_res` without norm + `gate_down` + `gate_mix` (qkv) + `attn_res` without norm + `gate_down` + `gate_mix` (mlp) = **9** |
| layer 0 | `rms_norm(input_layernorm)` replaces the first `attn_res` |
| final readout | 1 `attn_res(norm_w = model.norm)` -> fp16 -> lm_head |
| embedding | 1 `embed_norm` |

One forward of the real model (24 layers, 48 visits, 24 with messages) is
**312 glue launches** (plus embed, readout, 8 snapshot copies) against 6216 for
the torch composition (and ~13,000 more launches when the other groups of the
profile are counted: this file removes the "blocks glue", "layer glue" and "MoE
router / align" groups, 74 ms of the 154 ms GPU time of the eager decode step).

## 4. Weights, buffers, state

### 4.1 Weight repacks (once, after loading)

```python
# message gates: gates = [q_message_gate, k_message_gate, v_message_gate]
wd  = torch.stack([g.down_proj.weight for g in gates])            # [3, R, 2H] fp16
def pack(w):                                                      # nn.Linear [H, R]
    return w.to(torch.float16).reshape(H, R // 2, 2).permute(1, 0, 2).contiguous()
wf  = torch.stack([pack(g.forget_gate_proj.weight) for g in gates])   # [3, R/2, H, 2]
wg  = torch.stack([pack(g.message_gate_proj.weight) for g in gates])
# (fp32 exact variant: w.t().contiguous() -> [G, R, H] fp32)
norm_w = torch.stack([q_ln.weight, k_ln.weight, v_ln.weight])    # [3, H] full layers
norm_w = input_layernorm.weight[None]                              # [1, H] sliding layers
router_w = gate.weight.to(torch.float16).contiguous()            # [E, H], bf16-exact checkpoint
```

Keep `router_w` in fp32 when the checkpoint is not bf16
(`torch.equal(w.to(torch.float16).float(), w)` is false).

### 4.2 Static buffers (sized once for `max_tokens`)

Slice flat buffers to the call's `T` (`flat[:G*T*H].view(G, T, H)`), because
`d`, `out`, `scratch`, `partials` must be contiguous for the exact `T`.

| buffer | shape / dtype | zero-init |
|---|---|---|
| `blk` | `[T, n_states, H]` fp32 (`n_states = ceil(L / 6) = 4`) | no |
| `hid`, `res`, `attn_c`, `mlp_c` | `[T, H]` fp32 | no |
| `prev_blk` | `[T, n_states - 1, H]` fp32 | no |
| `prev_x0`, `prev_x1`, `pre_x0`, `pre_x1` | `[T, H]` fp32 | no |
| `msgs` | `[T, 4, H]` fp32 | no |
| `raw` | `[T, 3, H]` fp32 | no |
| `d` | `[3, T, R]` fp16 | no |
| `x16` | `[3, T, H]` fp16 (stream-major) | no |
| `g_scratch` / `g_part` | `[3, T, H]` / `[3, T, ceil(H/128)]` fp32 | no |
| `gate_counters` | `[3*T]` int32 | **yes, once** |
| `logits` / `topk_w` | `[T, E]` / `[T, K]` fp32 | no |
| `topk_ids` | `[T, K]` int32 | no |
| `sorted_ids` / `expert_ids` | int32, sized for `BM = 8` | no |
| `post_pad` | `[1]` int32 | no |
| `route_counters` | `[T + 1]` int32 | **yes, once** |
| `gu` / `act` / `moe_out` | `[T*K, 2I]` / `[T*K, I]` / `[T, H]` fp16 | zero-filled by `moe_route` |

The counters are left at zero by every launch (also under graph replay); after
an aborted run (exception while a kernel is in flight) call `counters.zero_()`.
One counter buffer per stream: two streams running the same op concurrently
would share it.

### 4.3 The persistent-state hazard of the loop passes

`Inkling2Model.forward` runs the loop core (layers 6-17) three times; pass
`p >= 1` starts from the pre-loop state and every loop layer receives messages
computed from the **previous** pass:
`prev_blocks = block_states[:, num_pre + 1:]` and `prev_hidden` (the output of
layer 17).  With persistent static buffers both are overwritten while the
messages are still needed (the layer-12 block append of pass `p` rewrites the
slot that `prev_blocks` of layers 12-17 reads; the residual / mlp buffers are
rewritten by every layer).  Snapshot them at the end of each pass with
`inkling2_glue_copy_states` into dedicated buffers (one buffer each is enough:
a snapshot is consumed during pass `p + 1` and replaced at its end):

```text
after layer 5 :  copy x0 -> pre_x0, x1 -> pre_x1             (pass restart state)
after layer 17:  copy blk[:, num_pre+1 : n] -> prev_blk       (n_prev = n - num_pre - 1 states)
                 copy x0 -> prev_x0, x1 -> prev_x1
```

`blk[:, :num_pre]` (the layer-0 input state) is never rewritten by the loop and
needs no copy.  8 tiny launches per forward.

## 5. Wiring patch (exact)

`amd/model.py` and `amd/moe.py` are owned by the lead; this is the code the
fused path needs.  It reproduces `Inkling2DecoderLayer.forward` /
`Inkling2Model.forward` / `Inkling2LoopMoE.forward` with the ops above.  The
attention part (`self.self_attn(...)` and the two `_static_conv` calls) is the
`Inkling2AttnKernels` work; here it only has to deliver fp32 `[T, H]` tensors
into the given buffers.

### 5.1 `amd/glue.py` (new): static buffers

```python
import math
from types import SimpleNamespace

import torch

from vllm import _custom_ops as ops


class Inkling2GlueWorkspace:
    """Static buffers of the fused glue path for up to ``max_tokens`` tokens."""

    def __init__(self, cfg, max_tokens: int, device) -> None:
        T, H, R = max_tokens, cfg.hidden_size, cfg.loop_gate_rank
        E, K, I = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.intermediate_size
        S = -(-cfg.num_hidden_layers // cfg.n_layers_per_attn_res_block)
        C = -(-H // 128)
        f32 = dict(dtype=torch.float32, device=device)
        f16 = dict(dtype=torch.float16, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.T, self.H, self.R = T, H, R
        self.blk = torch.zeros(T, S, H, **f32)  # persistent block states
        self.prev_blk = torch.zeros(T, max(S - 1, 1), H, **f32)  # previous pass
        sizes = {  # flat fp32 buffers, bound per call by `bind`
            "embed": T * H, "hid": T * H, "res": T * H, "attn_c": T * H,
            "mlp_c": T * H, "pre_x0": T * H, "pre_x1": T * H,
            "prev_x0": T * H, "prev_x1": T * H, "msgs": T * 4 * H,
            "raw": T * 3 * H, "g_scratch": 3 * T * H, "g_part": 3 * T * C,
            "logits": T * E, "topk_w": T * K,
        }  # fmt: skip
        self.f32 = {k: torch.empty(n, **f32) for k, n in sizes.items()}
        sizes16 = {
            "d": 3 * T * R, "x16": 3 * T * H, "gu": T * K * 2 * I,
            "act": T * K * I, "moe_out": T * H,
        }  # fmt: skip
        self.f16 = {k: torch.empty(n, **f16) for k, n in sizes16.items()}
        chunk = min(T, 1024)  # Inkling2LoopMoE.max_tokens_per_call
        n_pad, n_blk, _ = ops.inkling2_glue_moe_route_sizes(chunk, E, K, 8)
        self.sorted_ids = torch.empty(n_pad, **i32)
        self.expert_ids = torch.empty(n_blk, **i32)
        self.topk_ids = torch.empty(T, K, **i32)
        self.post_pad = torch.empty(1, **i32)
        self.route_counters = torch.zeros(chunk + 1, **i32)  # zero ONCE
        self.gate_counters = torch.zeros(3 * T, **i32)  # zero ONCE
        self.no_weights = torch.empty(0, **f32)

    def bind(self, T: int) -> SimpleNamespace:
        """Contiguous per-call views (``d``, ``out``... must be contiguous)."""
        H, R = self.H, self.R
        f, h = self.f32, self.f16
        v = lambda b, *shape: b[: math.prod(shape)].view(*shape)  # noqa: E731
        return SimpleNamespace(
            embed=v(f["embed"], T, H), hid=v(f["hid"], T, H),
            res=v(f["res"], T, H), attn_c=v(f["attn_c"], T, H),
            mlp_c=v(f["mlp_c"], T, H), pre_x0=v(f["pre_x0"], T, H),
            pre_x1=v(f["pre_x1"], T, H), prev_x0=v(f["prev_x0"], T, H),
            prev_x1=v(f["prev_x1"], T, H), msgs=v(f["msgs"], T, 4, H),
            raw=v(f["raw"], T, 3, H), g_scratch=v(f["g_scratch"], 3, T, H),
            g_part=v(f["g_part"], 3, T, -(-H // 128)),
            d=v(h["d"], 3, T, R), x16=v(h["x16"], 3, T, H),
        )  # fmt: skip
```

(`logits` / `topk_w` / `gu` / `act` / `moe_out` are bound per MoE chunk in
5.4, where the chunk size and `BM` are known.)

### 5.2 Load-time repacks (`Inkling2DecoderLayer`, after loading)

```python
def process_glue_weights(self) -> None:
    """Stacked / repacked weights of the glue ops (once, after loading)."""
    H, R = self.config.hidden_size, self.config.loop_gate_rank

    def pack(w):  # nn.Linear [H, R] -> [R/2, H, 2] fp16 pairs along R
        return w.to(torch.float16).reshape(H, R // 2, 2).permute(1, 0, 2).contiguous()

    def stack_gates(gates):
        return (
            torch.stack([g.down_proj.weight for g in gates]).contiguous(),
            torch.stack([pack(g.forget_gate_proj.weight) for g in gates]),
            torch.stack([pack(g.message_gate_proj.weight) for g in gates]),
        )

    if self.is_full:  # [3, H]: q / k / v layernorm weights
        self.attn_norm_w = torch.stack(
            [self.q_input_layernorm.weight, self.k_input_layernorm.weight,
             self.v_input_layernorm.weight]
        ).contiguous()  # fmt: skip
    else:  # [1, H]
        self.attn_norm_w = self.input_layernorm.weight.detach()[None].contiguous()
    self.post_attn_norm_w = self.post_attention_layernorm.weight.detach()[None].contiguous()
    if self.is_loop:
        self.qkv_wd, self.qkv_wf, self.qkv_wg = stack_gates(
            [self.q_message_gate, self.k_message_gate, self.v_message_gate]
        )
        self.mlp_wd, self.mlp_wf, self.mlp_wg = stack_gates([self.mlp_message_gate])
```

and in `Inkling2LoopMoE.process_weights_after_loading` (end):

```python
        w = self.gate.weight.data
        w16 = w.to(torch.float16)
        # V_DOT2 router when the (bf16) checkpoint weights convert exactly
        self.router_w = (w16 if torch.equal(w16.float(), w) else w).contiguous()
```

### 5.3 `Inkling2DecoderLayer.forward_glue` (replaces `forward` in the fused path)

`hidden` is the lazy sum `x0 (+ x1)` of fp32 `[T, H]` tensors; the function
returns the lazy output of the layer and the new number of block states.

```python
def forward_glue(self, ws, x0, x1, n_blocks, msgs, positions, visit):
    cfg = self.config
    H, eps, T = cfg.hidden_size, cfg.rms_norm_eps, x0.shape[0]
    G = 3 if self.is_full else 1  # attention-input queries / layernorm rows
    v = ws.bind(T)
    blk = ws.blk[:T]
    kv_owner, conv_owner = self.kv_owners[visit], self.conv_owners[visit]
    kv_md, conv_md = get_metadata(kv_owner.prefix), get_metadata(conv_owner.prefix)

    # ---- attention input: block residual (+ message gates) + layernorm -> fp16
    if self.attention_input is None:  # layer 0: hidden = embed_norm(..), x1 is None
        hid = x0
        ops.inkling2_glue_rms_norm(
            hid, v.x16[0], weight=self.input_layernorm.weight, eps=eps
        )
    else:
        hid = v.hid
        query = self.attention_input.query.view(G, H)
        if msgs is None:
            ops.inkling2_glue_attn_res(
                blk, n_blocks, x0, query, v.x16[:G].permute(1, 0, 2), x1=x1,
                partial_out=hid, norm_w=self.attn_norm_w, eps=eps, norm_eps=eps,
            )  # fmt: skip
        else:
            raw = v.raw[:, :G]
            ops.inkling2_glue_attn_res(
                blk, n_blocks, x0, query, raw, x1=x1, partial_out=hid, eps=eps
            )
            ops.inkling2_glue_msg_gate_down(raw, msgs[:, :3], self.qkv_wd, v.d, eps)
            ops.inkling2_glue_msg_gate_mix(
                raw, msgs[:, :3], v.d, self.qkv_wf, self.qkv_wg, v.x16,
                norm_w=self.attn_norm_w, scratch=v.g_scratch, partials=v.g_part,
                counters=ws.gate_counters, norm_eps=eps,
            )  # fmt: skip
    # sliding layers without messages feed one stream to q, k and v
    one_stream = msgs is None and not self.is_full
    hq = v.x16[0]
    hk = v.x16[0] if one_stream else v.x16[1]
    hv = v.x16[0] if one_stream else v.x16[2]

    # ---- attention + post-attention short convolution -> attn_c (fp32 [T, H])
    attn = self.self_attn(hq, hk, hv, positions, kv_owner, conv_owner, kv_md, conv_md)
    self._static_conv(  # AttnKernels: sconv writes fp32 into `out`
        self.post_attention_convolution, attn, STREAM_ATTN, positions, conv_owner,
        conv_md, out=v.attn_c,
    )  # fmt: skip

    # ---- mlp input: boundary layers append their input, the others add attn_c
    query = self.mlp_input.query.view(1, H)
    if self.is_block_boundary:  # stack = states + hid + attn_c; residual = attn_c
        a0, a1, extra, new_x0 = v.attn_c, None, dict(append=hid), v.attn_c
    else:  # stack = states + (hid + attn_c); residual = hid + attn_c
        a0, a1, extra, new_x0 = hid, v.attn_c, dict(partial_out=v.res), v.res
    if msgs is None:
        ops.inkling2_glue_attn_res(
            blk, n_blocks, a0, query, v.x16[:1].permute(1, 0, 2), x1=a1,
            norm_w=self.post_attn_norm_w, eps=eps, norm_eps=eps, **extra,
        )  # fmt: skip
    else:
        raw1 = v.raw[:, :1]
        ops.inkling2_glue_attn_res(blk, n_blocks, a0, query, raw1, x1=a1, eps=eps, **extra)
        ops.inkling2_glue_msg_gate_down(raw1, msgs[:, 3:4], self.mlp_wd, v.d[:1], eps)
        ops.inkling2_glue_msg_gate_mix(
            raw1, msgs[:, 3:4], v.d[:1], self.mlp_wf, self.mlp_wg, v.x16[:1],
            norm_w=self.post_attn_norm_w, scratch=v.g_scratch, partials=v.g_part,
            counters=ws.gate_counters, norm_eps=eps,
        )  # fmt: skip

    # ---- MoE (route, gemm1, swiglu, gemm2) + post-mlp short convolution
    moe = self.mlp.forward_glue(v.x16[0], ws)  # fp16 [T, H]
    self._static_conv(
        self.post_mlp_convolution, moe, STREAM_MLP, positions, conv_owner, conv_md,
        out=v.mlp_c,
    )  # fmt: skip
    return new_x0, v.mlp_c, n_blocks + int(self.is_block_boundary)
```

### 5.4 `Inkling2LoopMoE.forward_glue`

```python
def forward_glue(self, x16, ws):
    """``x16`` [T, H] fp16 (contiguous) -> ``[T, H]`` fp16 (zero-filled and
    accumulated by the GEMMs of this call)."""
    from vllm import _custom_ops as ops

    T, E, K, I = x16.shape[0], self.num_experts, self.top_k, self.intermediate_dim
    scale = self.gate.routed_scaling_factor
    out_all = ws.f16["moe_out"][: T * self.hidden_dim].view(T, self.hidden_dim)
    for s in range(0, T, self.max_tokens_per_call):
        e = min(s + self.max_tokens_per_call, T)
        tc = e - s
        bm = self._default_block_size_m(tc, K)
        n_pad, n_blk, _ = ops.inkling2_glue_moe_route_sizes(tc, E, K, bm)
        sid, eid = ws.sorted_ids[:n_pad], ws.expert_ids[:n_blk]
        gu = ws.f16["gu"][: tc * K * 2 * I].view(tc * K, 2 * I)
        act = ws.f16["act"][: tc * K * I].view(tc * K, I)
        out = out_all[s:e]
        tw = ws.f32["topk_w"][: tc * K].view(tc, K)
        ti = ws.topk_ids[:tc]
        logits = ws.f32["logits"][: tc * E].view(tc, E)
        ops.inkling2_glue_moe_route(
            x16[s:e], self.router_w, self.gate.e_score_correction_bias, logits, tw,
            ti, sid, eid, ws.post_pad, ws.route_counters, bm, scale,
            zero1=gu, zero2=out,
        )  # fmt: skip
        self._gemm("w13", x16[s:e], gu, ws.no_weights, sid, eid, ws.post_pad,
                   K, bm, False, 0)  # fmt: skip
        ops.inkling2_glue_swiglu(gu, act, self.gate_softcap, self.up_softcap)
        self._gemm("w2", act, out, tw.reshape(-1), sid, eid, ws.post_pad,
                   1, bm, True, K)  # fmt: skip
    return out_all
```

(`ti` / `tw` of a chunk are re-used by the next chunk only after the GEMMs
that read them are stream-ordered before it, so one set of buffers suffices.)

### 5.5 `Inkling2Model.forward_glue`

```python
def forward_glue(self, input_ids, positions, inputs_embeds=None):
    cfg, ws = self.config, self.glue_ws
    H, eps, T = cfg.hidden_size, cfg.rms_norm_eps, positions.shape[0]
    ls, le = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    v = ws.bind(T)
    copy = ops.inkling2_glue_copy_states

    if inputs_embeds is None:  # fp32 stream, fp16-rounded like embed_norm(..).float()
        ops.inkling2_glue_embed_norm(
            input_ids, self.embed_tokens.weight, self.embed_norm.weight, v.embed, eps
        )
    else:
        ops.inkling2_glue_rms_norm(
            inputs_embeds, v.embed, weight=self.embed_norm.weight, eps=eps,
            round_fp16=True,
        )

    def run(layers, x0, x1, n, msgs_fn, visit):
        for j, layer in enumerate(layers):
            msgs = msgs_fn(j) if msgs_fn is not None else None
            x0, x1, n = layer.forward_glue(ws, x0, x1, n, msgs, positions, visit)
        return x0, x1, n

    x0, x1, n = run(self.layers[:ls], v.embed, None, 0, None, 0)
    num_pre = n
    loop_layers = self.layers[ls:le]
    if cfg.loop_num_passes > 1:  # state every later pass restarts from
        copy(x0, v.pre_x0)
        copy(x1, v.pre_x1)
    x0, x1, n = run(loop_layers, x0, x1, n, None, 0)
    for visit in range(1, cfg.loop_num_passes):
        n_prev = n - num_pre - 1  # block states of the previous pass's messages
        prev_blk = ws.prev_blk[:T]
        if n_prev > 0:
            copy(ws.blk[:T, num_pre + 1 : n], prev_blk[:, :n_prev])
        copy(x0, v.prev_x0)
        copy(x1, v.prev_x1)

        def msgs_fn(j, prev_blk=prev_blk, n_prev=n_prev):
            ops.inkling2_glue_attn_res(
                prev_blk, n_prev, v.prev_x0, self.message_input.query[j], v.msgs,
                x1=v.prev_x1, eps=eps,
            )  # fmt: skip
            return v.msgs

        x0, x1, n = run(loop_layers, v.pre_x0, v.pre_x1, num_pre, msgs_fn, visit)
    x0, x1, n = run(self.layers[le:], x0, x1, n, None, 0)
    final16 = v.x16[0]  # [T, H] fp16 input of the lm_head
    ops.inkling2_glue_attn_res(
        ws.blk[:T], n, x0, self.final_input.query.view(1, H), final16.unsqueeze(1),
        x1=x1, norm_w=self.norm.weight[None], eps=eps, norm_eps=eps,
    )  # fmt: skip
    return final16
```

Notes on the sequence: `msgs_fn` is evaluated by `run` before each layer of the
pass, i.e. before that layer rewrites `res` / `mlp_c` / its block slot; the
messages read only the snapshots.  `n_blocks` is a Python int per layer visit
(static for a given config), so the whole forward is capturable.  The three
`Inkling2Model` call sites of `block_states` bookkeeping
(`torch.cat(..., hidden.unsqueeze(1))`) disappear: the boundary layer's
`append=hid` writes slot `n_blocks` in place.

## 6. Measurements

One V620 (b4 box, HIP 2), real checkpoint dimensions (`H = 2304`, `R = 64`,
`E = 128`, `K = 8`, `I = 1152`), `python tools/rdna/inkling2/bench_glue.py
--tokens 1,16,128,1024`, one run.  Columns per token count: **fused** = the
fused op sequence replayed from a HIP graph (GPU time per call), **torch GPU**
= the torch composition of `blocks.py` / `moe.py` it replaces, replayed from a
HIP graph when capturable (else the sum of its kernel durations), **torch
eager** = wall time of the eager composition (what the model pays today when it
is launch bound).  All in microseconds.  Launch counts are profiler event
counts of one call at `T = 1`.

| scenario | launches torch -> fused | T=1: fused / torch GPU / torch eager | T=16 | T=128 | T=1024 |
|---|---|---|---|---|---|
| attn_res prologue Q=3 full | 33 -> 1 | 8.7 / 237 / 490 | 8.9 / 188 / 493 | 24.9 / 226 / 493 | 111.0 / 1125 / 1151 |
| attn_res prologue Q=1 sliding | 18 -> 1 | 6.3 / 116 / 312 | 6.4 / 111 / 317 | 13.2 / 167 / 316 | 62.0 / 863 / 845 |
| attn_res mid (boundary, norm) | 18 -> 1 | 6.7 / 116 / 316 | 6.7 / 115 / 320 | 14.1 / 203 / 320 | 66.2 / 1213 / 1188 |
| attn_res messages Q=4 | 12 -> 1 | 6.0 / 135 / 227 | 6.2 / 137 / 234 | 12.6 / 133 / 230 | 72.2 / 423 / 399 |
| msg gates G=3, fp16 gain/forget | 87 -> 2 | 12.2 / 643 / 1251 | 45.6 / 640 / 1265 | 150.4 / 715 / 1242 | 596.6 / 1784 / 1910 |
| msg gates G=3, fp32 gain/forget | 87 -> 2 | 15.6 / 647 / 1248 | 55.2 / 639 / 1271 | 120.6 / 715 / 1243 | 628.0 / 1787 / 1909 |
| msg gates G=1, fp16 gain/forget | 29 -> 2 | 11.8 / 206 / 415 | 14.0 / 200 / 420 | 68.8 / 238 / 415 | 259.2 / 565 / 596 |
| msg gates G=1, fp32 gain/forget | 29 -> 2 | 13.0 / 205 / 435 | 15.4 / 200 / 420 | 88.5 / 237 / 435 | 242.1 / 566 / 593 |
| moe route + align + zero, fp16 router | 15 -> 1 | 10.5 / 147 / 231 | 15.0 / 148 / 233 | 39.0 / 136 / 248 | 151.8 / 203 / 242 |
| moe route + align + zero, fp32 router | 15 -> 1 | 10.9 / 147 / 230 | 14.3 / 147 / 231 | 43.0 / 136 / 233 | 223.1 / 203 / 241 |
| softcapped swiglu | 12 -> 1 | 4.0 / 36 / 117 | 4.4 / 36 / 112 | 13.7 / 94 / 118 | 122.6 / 1134 / 1174 |
| rms_norm (+ lazy add) | 8 -> 1 | 4.0 / 37 / 88 | 4.0 / 26 / 89 | 5.0 / 30 / 95 | 26.2 / 93 / 105 |
| embed gather + norm | 10 -> 1 | 3.9 / 43 / 111 | 4.0 / 35 / 109 | 4.3 / 34 / 118 | 13.0 / 97 / 121 |

One forward of the real model (24 non-message visits: 4 full-attention + 20
sliding; 24 message visits: 4 + 20; sequence of section 3, without embed,
readout and the 8 snapshot copies), composed from the rows above:

| glue of one forward | T=1 | T=16 | T=128 | T=1024 |
|---|---|---|---|---|
| fused (312 launches), ms | 2.06 | 3.16 | 9.50 | 41.99 |
| torch GPU (6216 launches), ms | 44.49 | 43.73 | 55.34 | 232.44 |
| torch eager wall, ms | 93.72 | 94.60 | 94.81 | 237.72 |

Reading the numbers:

* At `T = 1` every fused op costs the ~4 us of a minimal launch inside a graph
  (`swiglu`, `rms_norm`, `embed` are at that floor); the torch composition
  costs 36-640 us per op because it is 8-87 tiny kernels.  The 2.06 ms of glue
  per decode step is then dominated by the 24 gate pairs (about 0.6 ms) and the
  48 routers (0.5 ms).
* Against the torch composition replayed from a graph the glue of a forward is
  21.6x (T = 1), 13.8x (T = 16), 5.8x (T = 128) and 5.5x (T = 1024) cheaper.
* The routers' align stage is a parallel counting sort: with the serial align
  that preceded it the fp16 router took 361 us at `T = 1024` (same bench), now
  152 us (torch: 203 us).  The fp32-weight router (checkpoints that are not
  bf16-exact; FMA instead of V_DOT2 on twice the weight bytes) is not faster
  than torch at `T = 1024` (223 vs 203 us).
* The gates are 2.2-3.0x faster than torch at `T = 1024` and not more.
  [INFERENCE] the fp32 round trip of the staged rows through global memory
  bounds them there; a kernel profile at `T = 1024`, `G = 3` shows `gate_down`
  237 us and `gate_mix` (with norm) 281 us.

## 7. Verification

### 7.1 Tests

`tests/kernels/core/test_inkling2_glue_rdna2.py` (gfx1030 only; every op is
compared with an fp64 torch composition of the reference semantics of
`blocks.py` / `moe.py`, and whole layer visits are compared with the reference
torch modules; the torch code is the oracle, it is not in the serving path):

```text
$ bash gpu_run.sh 600 run_tests.py -q -p no:cacheprovider --maxfail=80 \
      tests/kernels/core/test_inkling2_glue_rdna2.py      (HIP 2, b4 box)
195 passed, 14 warnings in 77.60s (0:01:17)                exit code 0
```

Covered: `attn_res` (prologue Q=3 / Q=1, boundary mid, messages Q=4, fp16 and
fp32 outputs, stream-major `out`, saturated softmax, other `eps`, strides),
the gates (30 cases: G = 1 / 3, fp16-pair and fp32 weights, `T` from 1 to 1024,
ragged `H`, `H = 2432` with an odd number of 128-column chunks, tiny / huge
activations, strided messages, raw fp32 output), the router (decode, ties, NaN
row, hot experts at `T = 300` / `T = 500` with `K = 1`, `E = 255`, fp16 and fp32
weights, `T` up to 1024), `swiglu`, `rms_norm`, `embed_norm` (strided table,
odd vocabulary), `copy_states`, the attention prologue and the MLP input of loop
visits (full / sliding, with and without messages, boundary layers) against the
reference modules, the router and `swiglu` against `blocks.py`, argument
rejection, repeated-launch race tests of the last-arriver protocols
(`test_moe_route_repeated_launches_do_not_race`,
`test_gate_mix_repeated_launches_do_not_race`) and HIP-graph capture + replay of
a whole visit sequence (`test_visit_sequence_graph_capture_and_replay`).

Measured error, final run (`GLUE_TEST_STATS=1`, 480 printed statistics):
fp16-output ops stay within one fp16 ulp of the fp64 oracle plus a tiny
absolute floor (largest excess over 1 ulp `3.0e-7`, largest mismatch fraction
`0.17 %`); router logits `<= 6.7e-7`, router weights `<= 1.5e-7`; fp32-output
gate mix `<= 1.0e-6`.  Whole visits are asserted within `VISIT_ATOL = 2e-3`
absolute (the rounding of `d` to fp16, section 8).

### 7.2 Mutation testing

`mutants.py` (a dev script kept on the box next to the build tree, not part of
the repo: it patches a copy of the `.cu`, builds it with the same JIT flags and
runs the test selection of the touched op on the mutated binary).  Every mutant
has to be killed by at least one test; 50 hand-written mutants of the final
source, 49 killed in the first run:

```text
attn_res   (7)   attn_scale_quarter attn_state_norm_no_eps attn_softmax_no_max
                 attn_norm_row_always0 attn_out_norm_uses_state_eps
                 attn_drop_x1 attn_append_wrong_slot
gate_down  (6)   gd_swap_halves gd_no_eps gd_stage_drops_last_vec
                 gd_round_toward_zero gd_rows_drop_last gd_row_stride_one_pass
gate_mix   (6)   gm_forget_sign gm_counter_not_reset gm_premature_last_arriver
                 gm_norm_row_always0 gm_gain_uses_forget_acc gm_norm_eps_x1e4
rms_norm   (2)   rms_ignores_round_flag rms_eps_x10
embed      (2)   embed_row_stride_H embed_no_upper_clamp
swiglu     (3)   swiglu_sigmoid_of_capped swiglu_hard_clip_gate
                 swiglu_up_uses_gate_cap
moe_route  (21)  route_tie_highest_index route_no_route_scale
                 route_weights_from_biased route_no_sigmoid
                 route_expert_ids_sentinel0 route_post_pad_plus_bm
                 route_sorted_sentinel route_tile_counter_not_reset
                 align_match_7bits align_wave_cursor_dropped
                 align_scatter_leader_adds_one align_key_load_swaps_pairs
                 align_waves_reversed align_hist_init_garbage
                 align_count_misses_last_wave route_zero_fill_wrong
                 route_topk_no_mask route_nan_unclamped
                 route_expert_loop_skips_last route_zero_skips_tail
                 route_premature_tile_arriver
copy       (3)   copy_dst_uses_src_tstride copy_drops_last_vec copy_tp_swapped
```

Details of the run (`49/50 killed`): each killed mutant failed 1-3 tests
(`--maxfail=3`); e.g. `gm_premature_last_arriver` is only caught by the
`T = 1024` cases (first failure `test_msg_gate[qkv_full_t1024]`),
`route_premature_tile_arriver` by `test_moe_route[t1024]`,
`align_scatter_leader_adds_one` / `align_count_misses_last_wave` by
`test_moe_route[t64]`, `attn_softmax_no_max` by
`test_attn_res[saturated_softmax]`, `route_tie_highest_index` by
`test_moe_route[ties_pick_lowest_index]`.  `copy_dst_uses_src_tstride` is killed
by a crash of the test process (`Fatal Python error: Aborted`: the mutant
writes through the wrong stride), the other two copy mutants by failing
assertions.  The one survivor of the first run, `gd_swap_halves`, was an
equivalent mutant (`which` chose both the source row and the destination half,
so flipping it only permuted the loop order); it was replaced by a mutant that
swaps the destination half only (`dst = ... + (which ^ 1) * H`), which a second
run killed: 3 failing tests, first `test_msg_gate[qkv_full_t1]`.  All 50
mutants of the final source are therefore killed.

### 7.3 Multi-architecture build

The glue source was compiled in one JIT build for six architectures
(`gfx1030,gfx1100,gfx90a,gfx942,gfx950,gfx1200`, same flags as the repo build;
`-Rpass-analysis=kernel-resource-usage` output parsed per code object):

| arch | kernels | VGPR (max) | static LDS (max) | scratch |
|---|---|---|---|---|
| gfx1030 | 59 | 239 | 12,672 B | 0 |
| gfx1100 / gfx90a / gfx942 / gfx950 / gfx1200 | 59 each | 0 | 0 | 0 |

The other arches contain the same 59 symbols as empty stubs (the device code
is behind `__HIP__RDNA2__`; the stubs live in `namespace inkling2_glue` like
the `namespace lowp_dot2` stubs of `moe_lowp_dot2_rdna2.cu`), so their code
objects hold no glue code.  Nothing here changes the other `.cu` files.

### 7.4 Build wiring

`csrc/rocm/inkling2_glue_rdna2.cu` is in the ROCm source whitelist of
`CMakeLists.txt` (line 1829), the eight ops are declared in `csrc/rocm/ops.h`,
bound in `csrc/rocm/torch_bindings.cpp` (`rocm_ops.def` + `impl` on
`torch::kCUDA`) and wrapped, with fake implementations, in
`vllm/_custom_ops.py`.  The test and bench use the same `.cu` through the JIT
path (`torch.utils.cpp_extension` with the repo flags); the full CMake build was
not run.

### 7.5 `inkling2_glue_embed_lookup` (tensor-parallel embedding)

Added for the vocabulary-sharded embedding (`inkling2-tensor-parallel.md`); built into
`_rocm_C` with the rest (CMake build of the TP tree, 157 steps, 846 s) and bound like the
other ops, no new source file.

* **Reference**: the output is an exact copy of fp16 rows, so the oracle is bit equality
  with `table[clamp(ids)]` masked by shard ownership (no tolerance).  Tests
  (`tests/kernels/core/test_inkling2_glue_rdna2.py`, 17 cases): five shards of a 4,099
  row table (first / last / single-row / middle shards, int32 and int64 ids, ids on both
  sides of every boundary, below 0 and above the vocabulary), the shards of TP = 2 / 3 / 4
  add up to the full gather (zero-padded last shard), `lookup + all-reduce + rms_norm(...,
  round_fp16=True)` is bit-identical to `inkling2_glue_embed_norm` (fp16 and fp32 `out`),
  strided table / output rows with intact padding and `T = 0`, argument checks.
* **Mutation checks** (JIT build of each mutated source, the 17 tests on HIP 4): 10 of 10
  killed: row offset + 1, `r <= rows` (reads past the shard), `r > 0` (first row lost),
  `r >= -1` (reads before the shard), no upper clamp, no lower clamp, upper clamp to
  `total - 2`, misses not zeroed, table row stride `H`, output row stride `H`.
* **Six architectures** (`gfx1030, gfx1100, gfx90a, gfx942, gfx950, gfx1200`, one JIT build
  of the whole glue source with the repo flags): compiles; gfx1030 8 VGPRs, 26 SGPRs,
  no scratch, no LDS, occupancy 16 waves per SIMD; the other five arches hold the empty
  stub of the kernel (0 VGPRs, no scratch, no LDS) like the other glue kernels.
* **Launch form** (found on the merged tree, whose CMake build stopped at step 145 with
  `inkling2_glue_rdna2.hip:1868: error: expected unqualified-id`): the kernel was the
  source's only non-template `__global__` function.  clang-format wraps its launch after
  `inkling2_glue::` and `hipify` does not parse the split qualified name; a joined line
  does not survive the repo's formatter.  It carries a dummy template parameter now
  (`template <int DUMMY = 0>`, launched as `embed_lookup_kernel<0>`) like `swiglu_kernel`
  and `copy_states_kernel`; the kernel body is unchanged.  After the change: the 212 glue
  tests and the 16 TP thread tests pass, the six-architecture compile is `compile_exit=0`
  with the same resources (gfx1030 8 VGPRs, 26 SGPRs, no scratch, no LDS, occupancy 16).

## 8. Risks and limits

* **Last-arriver protocol** (`gate_mix` with norm, `moe_route`): a CTA publishes
  its results with plain stores, `__threadfence()`, `atomicAdd` on a counter;
  the last arriver issues another `__threadfence()` and consumes them.  The
  counters are left at zero by every launch (also under graph replay).  After
  an aborted run (exception while a kernel is in flight) call
  `counters.zero_()`; use one counter buffer per stream.  Covered by the
  repeated-launch race tests and the graph-capture test, but it is a memory
  ordering argument, not a proof.
* **fp16 `d` rounding.**  The down projection is rounded to fp16 (as the
  reference does); the fp32 reduction order differs from torch, so 1-4 % of the
  `d` elements differ by one fp16 ulp from the fp64 reference, which moves the
  fused visit output by at most `VISIT_ATOL = 2e-3` absolute (tests).
* **Router near-ties.**  Top-k is exact on the kernel's own fp32 scores (lowest
  id wins exact ties) but the logits are summed in a different order than
  torch, so two experts whose biased scores differ by about 1e-6 relative may
  swap against `torch.topk`.
* **fp16 router / gate weights** are exact only for bf16-exact checkpoints with
  `|w| >= 2**-14`; smaller values round to fp16 subnormals (abs error `<= 3e-8`).
  The fp32 variants (`FMA`) are exact.
* **Snapshot hazard.**  The loop-pass snapshots (`copy_states`, section 4.3)
  are required for correctness once the buffers are persistent; skipping them
  makes the messages read overwritten states.
* **Load-time repacks** (stacked `[3, H]` layernorm weights, pair-interleaved
  gate weights, fp16 router weights) are part of the wiring patch.
* **Limits.**  `T*K <= 8192` per `moe_route` call (the serving
  `max_tokens_per_call = 1024`), `E <= 255`, `K <= 16`, `H <= 4096` for the down
  projection, at most `kMaxStates = 64` states per `attn_res` call (block states
    * append + partial).
* **Build.**  The kernels were compiled by the JIT path with the flags of the
  repo build and for six architectures (section 7); the full CMake build of the
  fork was not run by this change.
* **Tensor parallel.**  The MoE shards the intermediate dimension (`I / tp` per rank,
  `Inkling2LoopMoE(tp_size=...)`), the router and the gate weights are replicated,
  and these ops run per rank unchanged on the local shapes; the partial MoE output is
  all-reduced before the post-MLP conv (`docs/rdna2/inkling2-port-spec.md`, "tensor
  parallelism").  `Inkling2GlueWorkspace(tp_size=...)` sizes `gu` / `act` for the
  local intermediate size and gives `o_out` the fp32 `[T, H]` shape of the `o_proj`
  partial.
