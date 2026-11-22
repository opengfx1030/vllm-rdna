# Inkling-2-Mini attention and short-conv kernels on RDNA2 (`inkling2_attention`, `inkling2_sconv_*`, `inkling2_kv_prep`)

Native HIP kernels for the attention core and the paged conv / KV state of
`Inkling2LoopForConditionalGeneration` on gfx1030 (V620: wave32, no bf16 / fp8 /
fp4 dot hardware, `V_DOT2_F32_F16` is the matrix primitive). They replace the
eager torch core of `vllm/models/inkling2_loop/amd/attention.py`
(`Inkling2Attention._attend`, `_scaled_clip`, the `rel_logits` einsum, the gate
multiply) and the torch `conv_taps` / `kv_write` / `kv_read` helpers of
`amd/state.py`. They are HIP-graph capturable (no host sync, no allocation, every
pointer and length is read from device memory at run time) and need no Triton,
rocBLAS, hipBLAS or CK in the serving path.

| What | Where |
| --- | --- |
| Kernels | `csrc/rocm/inkling2_attn_rdna2.cu` |
| Registration | `csrc/rocm/ops.h`, `csrc/rocm/torch_bindings.cpp`, `CMakeLists.txt` |
| Python ops | `vllm/_custom_ops.py` (`inkling2_attention*`, `inkling2_sconv_*`, `inkling2_conv_state_write`, `inkling2_kv_prep`) |
| Tests | `tests/kernels/attention/test_inkling2_attn_rdna2.py` |
| Benchmark | `tools/rdna/inkling2/bench_attn.py` |

## 1. What is computed

| Op | Replaces (repo torch path) | Computes |
| --- | --- | --- |
| `inkling2_sconv_static` | `conv_taps` + `StaticSconv.forward_taps` | `out = (sum_lag x[t-lag] w[c, 3-lag] + x[t]) / sqrt(5)` per channel, fp32; reads the previous 3 raw inputs from the fp16 conv state ring; writes this step's inputs back only when asked (`write_state`, at most one token per request) |
| `inkling2_sconv_dynamic` | `conv_taps` + `DynamicSconv.forward_taps` | `out = (sum_lag x[t-lag] (sw[lag, c] + dw[t, h, lag]) + x[t]) / sqrt(9)` with the per-token, per-kv-head taps `dw` precomputed by the caller |
| `inkling2_conv_state_write` | the write half of `conv_taps` | stores the raw inputs of one stream into the conv state ring (the last `ring` positions of every block of the chunk), after the conv has read its history |
| `inkling2_kv_prep` | k / v conv + `l2_clip(k)` + `v * rsqrt(mean(v^2) + eps)` + `kv_write` | both short convs, the k L2 clip, the v RMS norm and the fp16 scatter into the paged K\|V cache in one launch; the conv-state write only with `write_state` |
| `inkling2_attention` | `_scaled_clip` (q and rel) + `rel_logits` + `_attend` + gate | scaled L2 clip (+ log scale on full layers) of q and of the relative features, `softmax(q.k / sqrt(128) + bias, sink)` over the paged cache with the sliding window or full causal range, `P.V`, `* sigmoid(gate)` |

Attention semantics (all per query token `t` of a request with `L` keys in the
cache, `p = L - q_len + i` its token index, `j` a key index):

```text
visible(j)  = j <= p  and  (window == 0 or p - j < window)
bias(j)     = rel[t, h] . proj_t[p - j]      if 0 <= p - j < rel_extent, else 0
score(j)    = q[t, h] . k[j] * 128^-0.5 + bias(j)
out[t, h]   = softmax([score(j) for visible j] + [sink[t, h]])[:-1] @ v  *  sigmoid(gate[t, h])
q, rel      = l2_clip(raw * scale, thr) -> fp16, then * tau(pos) -> fp16 on full layers
tau(pos)    = 1 + alpha * log(max((pos + 1) / n_floor, 1))
```

Everything inside the softmax is fp32; q, k, v, bias features and the output are
fp16. The window is one of {1024 (sliding layers), 0 (full layers)} but any
`window >= 0` is honoured.

## 2. Op reference

All ops launch on the current stream, never synchronise, allocate nothing and
read lengths / block tables / slot mappings from device tensors, so a captured
graph replays them with whatever those tensors hold at replay time. All tensors
are GPU tensors of one device. Sizes of `int` arguments are host integers fixed at
capture time.

### 2.1 `inkling2_attention`

```text
inkling2_attention(Tensor q, Tensor rel, Tensor sink, Tensor? gate,
                   Tensor q_scale, Tensor rel_scale, Tensor rel_proj_t,
                   Tensor positions, Tensor kv_cache, Tensor block_table,
                   Tensor seq_lens, Tensor query_start_loc,
                   Tensor! out, Tensor! workspace,
                   int window, int rel_extent, bool log_scale,
                   float log_alpha, float log_n_floor, float q_clip,
                   float rel_clip, int max_query_len,
                   int splits_decode, int splits_tile) -> ()
```

| Argument | Contract |
| --- | --- |
| `q` | fp16 `[T, nh * 128]` raw `q_proj` output (before the scaled clip). Unit column stride, row stride a multiple of 8 elements, 16-byte aligned base. May be a column slice of a fused projection output. |
| `rel` | fp16 `[T, nh * 16]` raw relative-feature projection (same stride rules). |
| `sink` | fp16 `[T, nh]` raw `sink_proj` output; unit column stride, even row stride. |
| `gate` | fp16 `[T, nh * 128]` raw `gate_proj` output (`sigmoid` is applied in the kernel) or `None`. |
| `q_scale` / `rel_scale` | fp32 contiguous `[nh, 128]` / `[nh, 16]` learned per-head scales (`q_proj.norm.scale`, `relative_position_bias.norm.scale`), 16-byte aligned. |
| `rel_proj_t` | fp16 contiguous `[>= rel_extent, 16]`, the learned `proj [16, rel_extent]` transposed: `rel_proj_t = proj.t().contiguous().half()` (prepared once at load). Row `d` is the bias basis of distance `d`. |
| `positions` | int32 or int64 `[>= T]`. Only used for the log scale `tau`; it must equal the token index of each query token (`seq_len - q_len + i`), as in the repo path. |
| `kv_cache` | fp16 `[num_blocks, H_kv, block_size, 256]` page array (K in `[0, 128)`, V in `[128, 256)` of the last dim); unit stride on the last dim, other strides multiples of 8 elements, 16-byte aligned. `block_size` is a power of two `>= 16` (64 in the model). The engine's block-outermost physical layout `[nb, bs, H, 256]` is passed as the permuted view `[nb, H, bs, 256]` (section 3). |
| `block_table` | int32 contiguous `[R, max_blocks]`; `block_table[r, j // block_size]` is the page of key `j`. Entries of pages outside the window of a sliding layer are never read. |
| `seq_lens` | int32 contiguous `[R]`: total keys (context + this step's tokens) of request `r`. `seq_lens[r] <= 0` marks a padded request (its output rows are written as zero). |
| `query_start_loc` | int32 contiguous `[R + 1]`; request `r` owns query rows `[qsl[r], qsl[r+1])`. Requests with `q_len == 1` use the decode kernel, `q_len >= 2` the tiled kernel, `q_len == 0` are skipped. |
| `out` | fp16 `[T, nh * 128]` (same stride rules as `q`); receives `attn * sigmoid(gate)`. Rows of padded requests are zeroed; rows `>= query_start_loc[R]` are not touched. |
| `workspace` | fp32 contiguous split-K scratch, `numel >= inkling2_attention_workspace_numel(R, T, nh, splits_decode, splits_tile)`. Any content; fully rewritten before it is read (section 5). |
| `window` | `0` = full causal attention, `> 0` = sliding window (keys with `p - j < window`). |
| `rel_extent` | rows of `rel_proj_t` that are used (1024 on the model). Distances `>= rel_extent` get bias 0. |
| `log_scale` | apply `tau(pos)` to q and rel (full layers). |
| `log_alpha`, `log_n_floor` | `0.1`, `8192`. |
| `q_clip`, `rel_clip` | `22.627416997969522`, `8.0`. |
| `max_query_len` | host upper bound of `q_len` (`1` for a pure decode batch; it must be `>=` the real maximum, otherwise longer requests are not computed). |
| `splits_decode`, `splits_tile` | split-K factors in `[1, 256]`: upper bounds on the partials per decode request / per tile CTA row (section 5). |

`nh` / `H_kv` is the GQA group size `g in {1, 2, 3, 4}`; the model has `g = 3` at
every TP degree (24/8, 12/4, 6/2).

### 2.2 `inkling2_kv_prep`

```text
inkling2_kv_prep(Tensor xk, Tensor xv, Tensor k_weight, Tensor v_weight,
                 Tensor? k_dyn, Tensor? v_dyn, Tensor! conv_cache,
                 Tensor conv_block_table, Tensor seq_idx,
                 Tensor query_start_loc, Tensor seq_lens,
                 Tensor! kv_cache, Tensor kv_slot_mapping,
                 int k_stream_offset, int v_stream_offset,
                 int conv_block_size, bool write_state,
                 float k_clip, float rms_eps) -> ()
```

| Argument | Contract |
| --- | --- |
| `xk`, `xv` | fp16 `[T, H_kv * 128]` raw `k_proj` / `v_proj` outputs (conv inputs); unit column stride, row stride a multiple of 8, 16-byte aligned. |
| `k_weight`, `v_weight` | fp32 contiguous. Static layers: the conv weights `[H_kv * 128, 1, 4]`. Dynamic layers: `static_weight [4, H_kv * 128]`. |
| `k_dyn`, `v_dyn` | `None` (static conv) or fp16 `[T, H_kv * 4]` outputs of `weight_proj` (the per-token, per-head dynamic taps; both given or both `None`, same row stride). |
| `conv_cache` | fp16 `[num_conv_blocks, H_kv, ring, W]` (section 3): `ring >= 3` rows per block, `W` columns per head; stream `s` of a head occupies `[offset, offset + 128)` of the last dim. Any strides whose block / head / row steps are multiples of 4 halfs and a base that is 8-byte aligned. |
| `conv_block_table`, `conv_block_size` | int32 `[R, max_conv_blocks]` and the tokens per conv block (a power of two; the engine's cache block size, 64). Position `p` of request `r` lives in block `conv_block_table[r, p / conv_block_size]`, ring row `p % ring`. |
| `seq_idx` | int32 `[T_md]`: request index of each token (`-1` = padding); `T_md <= T`, tokens `>= T_md` are padding. |
| `query_start_loc`, `seq_lens` | as for `inkling2_attention`. |
| `kv_cache`, `kv_slot_mapping` | the K\|V page array of section 2.1 and int64 `[>= T]` slots (`-1` = padding); K = `l2_clip(conv_k, k_clip)`, V = `conv_v * rsqrt(mean(conv_v^2) + rms_eps)`, both rounded to fp16. |
| `k_stream_offset`, `v_stream_offset` | stream offsets inside `W` (`0` and `128` of the model: `Inkling2ConvOwner.stream_ranges`). |
| `write_state` | store the raw k / v inputs in the ring after reading the history. Only valid when no request has more than one token this step; otherwise run `inkling2_conv_state_write` per stream afterwards. |
| `k_clip`, `rms_eps` | `22.627416997969522`, `1e-6`. |

The previous `k - 1 = 3` raw inputs of a token come from `x` when they belong to
the same request chunk (position `>= p0`) and from the conv state otherwise;
positions below 0 read as 0. The kernel reads the state only; the write is the
separate op (or `write_state` for one-token requests, where each thread reads its
history rows and then writes its own row, and no other thread touches them).

### 2.3 `inkling2_sconv_static` / `inkling2_sconv_dynamic` / `inkling2_conv_state_write`

```text
inkling2_sconv_static(Tensor x, Tensor weight, Tensor! out, Tensor! conv_cache,
                      Tensor block_table, Tensor seq_idx,
                      Tensor query_start_loc, Tensor seq_lens,
                      int stream_offset, int num_kv_heads, int block_size,
                      bool write_state) -> ()
inkling2_sconv_dynamic(Tensor x, Tensor static_weight, Tensor dyn,
                       Tensor! out, Tensor! conv_cache, Tensor block_table,
                       Tensor seq_idx, Tensor query_start_loc, Tensor seq_lens,
                       int stream_offset, int num_kv_heads, int block_size,
                       bool write_state) -> ()
inkling2_conv_state_write(Tensor x, Tensor! conv_cache, Tensor block_table,
                          Tensor seq_idx, Tensor query_start_loc,
                          Tensor seq_lens, int stream_offset,
                          int num_kv_heads, int block_size) -> ()
```

Generic single-stream versions (used for the attention-output and MLP-output
convs of the block, and for any k / v stream not fused in `inkling2_kv_prep`):
`x` is fp16 or fp32 `[T, C]` (`C = H_kv * width`, `width % 4 == 0`, `stream_offset
% 4 == 0`), `out` fp32 `[T, C]`; the other arguments are as in section 2.2. Tokens
with `seq_idx < 0` get a zero output row and write nothing.

`inkling2_conv_state_write` stores the raw inputs of `x` (rounded to fp16 with
round-to-nearest-even; the serving path's conv inputs are fp16 GEMM outputs, so the
state is exact) at their ring rows, for every token that is one of the **last `ring`
positions of the intersection of its chunk with its block**: `p + ring >=
min(seq_len, block_end(p))`. That rule gives every ring row one writer per step,
leaves the rows of earlier chunk positions alone (they would be overwritten anyway)
and always stores what the next reader needs: the 3 history positions of the next
chunk and the last 3 positions of every completed block (what a prefix-cache hit on
that block resumes from). It must run after the conv of the same stream (which reads
the history rows it may overwrite) and, like the conv, once per stream.

### 2.4 Helpers (pure Python, `vllm/_custom_ops.py`)

```python
inkling2_attention_workspace_numel(num_reqs, num_tokens, num_q_heads,
                                   splits_decode, splits_tile) -> int
inkling2_attention_splits(num_reqs, num_kv_heads, window, max_seq_len,
                          max_query_len) -> (splits_decode, splits_tile)
```

## 3. Cache layouts and strides

* **K\|V page** `[nb, H_kv, bs, 256]` fp16, K then V in the last dim. The engine
  hands every layer this logical `[B, H, N, C]` view for all `KVCacheLayout`s it
  can pick for the model (`vllm/v1/kv_cache_layout.py`; the KV and conv specs
  differ in shape, which restricts the choice to the block-compact layouts
  `LBHNC`, `LBNHC`, `BLHNC`, `BLNHC` [INFERENCE from `resolve_kv_cache_layout`]):
  inside a page the head-major (`HNC`) or the token-major
  (`NHC`, 4 KB per token at TP=1) order, and with the `BL*` layouts a block
  stride that skips the pages of the other layers. The kernels take the three
  leading strides from the tensor, so every one of them works without a copy; the
  tests cover head-major (`hbc`), token-major (`bhc`) and both interleaved with
  two other layers' NaN pages (block stride = 3 pages). A key row is read as four
  aligned 128-byte runs (K lo / hi, V lo / hi), so the strides must be multiples
  of 8 elements and the base 16-byte aligned. The engine's default preference
  (`_DEFAULT_LAYOUT_PREFERENCE`: `LBNHC`, `LBHNC`, `BLNHC`, `BLHNC`, ...) puts the
  token-major `LBNHC` first, which is the benchmark's default `--layout bhc`;
  `--layout hbc` is the head-major `LBHNC` (measured 5% slower at 128k, section 9).
* **Conv page** `[nb, H_kv, ring, W]` fp16 (`Inkling2ConvOwner.state`, one page per
  cache block of `bs = 64` tokens, `W` = the four stream widths back to back: 128 / 128
  / 288 / 288 = 832 at TP=1, offsets from `Inkling2ConvOwner.stream_ranges`, no power
  of two padding). Token position `p` lives in block `block_table[r, p / bs]` at ring
  row `p % ring`; `ring = 3 + num_speculative_tokens` (3 without a draft model: the
  `k - 1` history rows; with `k` draft tokens the rows of the accepted prefix of a
  step survive the rejection of the drafts behind it). The page is `H_kv * ring * W * 2`
  bytes: 39,936 B at TP=1 without drafts, 133,120 B with 7 draft tokens (ring 10). The
  engine allocates it with the block-outermost layouts (`BLNHC` / `BLHNC`), so the 48
  conv layers pack into one KV cache group next to the sliding-window and
  full-attention groups of 2 MiB pages (`docs/rdna2/inkling2-port-spec.md`, "Conv and
  KV memory").
* Slot mappings of the K\|V cache are int64 and follow vLLM's convention `slot = block
    * block_size + offset`,`-1` for padding. The conv ops take the block table and
  `seq_idx`instead (`-1` = padding) and derive the block and the row themselves.
* **What the ring writers skip.**  A step writes the ring row of a position `p` of a
  request (`p1` = its `seq_lens` entry) only if `0 <= p < p1` (`i2_ring_keeps`) and only
  into a real page: block 0 is `BlockPool.null_block`, not a page (`i2_write_ring`).
  Three kinds of token are therefore harmless although the engine builds them: the
  padding of a graph, which the engine's token -> request map (`build_qsa_metadata_rdna2`)
  gives to request 0 past the last chunk (`p >= p1`, and with a 64-token block that
  position falls into a block request 0 owns); a capture's dummy batch whose `seq_len` is
  shorter than its query (`seq_len` 1, `q_len` 6: `p < 0`, which would index the block
  table below its first row and a negative ring row and fault); and the zeroed block
  tables of the dummy batches of a capture, a warm-up and the memory profile (block 0).
  The model side also gives padded tokens `seq_idx = -1` (`SeqIdxScratch.fill`), so the
  kernel rules are a second line, not the only one.

## 4. Caller-side preparation (what the lead has to wire)

Per layer, at load time:

```python
q_scale   = self.q_proj.norm.scale.float().contiguous()                     # [nh, 128]
rel_scale = self.relative_position_bias.norm.scale.float().contiguous()     # [nh, 16]
rel_proj_t = self.relative_position_bias.proj.t().contiguous().half()       # [rel_extent, 16]
conv_w_k, conv_w_v = ...conv1d.weight.float().contiguous()   # static: [C, 1, 4]
                       # dynamic: static_weight.float().contiguous()   [4, C]
```

Per forward (fp16 GEMM outputs are passed raw, nothing is converted):

```python
q    = q_proj.linear(hq)                  # [T, nh*128]
rel  = rb.linear(hq)                      # [T, nh*16]
sink = sink_proj(hq)                      # [T, nh]
gate = o_proj.gate_proj(hq)               # [T, nh*128]
xk, xv = k_proj.linear(hk), v_proj.linear(hv)
k_dyn = v_dyn = None                      # full layers: weight_proj(hk) / weight_proj(hv)
ops.inkling2_kv_prep(xk, xv, conv_w_k, conv_w_v, k_dyn, v_dyn, conv_cache,
                     conv_md.block_table, conv_md.seq_idx,
                     conv_md.query_start_loc, conv_md.seq_lens, kv_cache,
                     kv_md.slot_mapping, 0, 128, conv_block_size,
                     write_state,         # == (max_query_len == 1)
                     qk_l2_clip_threshold, rms_norm_eps)
if not write_state:                       # chunked prefill / drafts: after the reads
    ops.inkling2_conv_state_write(xk, conv_cache, conv_md.block_table, conv_md.seq_idx,
                                  conv_md.query_start_loc, conv_md.seq_lens, 0, H_kv, conv_block_size)
    ops.inkling2_conv_state_write(xv, conv_cache, conv_md.block_table, conv_md.seq_idx,
                                  conv_md.query_start_loc, conv_md.seq_lens, 128, H_kv, conv_block_size)
ops.inkling2_attention(q, rel, sink, gate, q_scale, rel_scale, rel_proj_t,
                       positions, kv_cache, kv_md.block_table, kv_md.seq_lens,
                       kv_md.query_start_loc, attn_out, workspace,
                       window, rel_extent, not is_sliding, 0.1, 8192.0,
                       qk_l2_clip_threshold, rel_l2_clip_threshold,
                       max_query_len, splits_decode, splits_tile)
out = o_proj.linear(attn_out)
```

`inkling2_kv_prep` must run before `inkling2_attention` of the same layer visit on
the same stream (the attention reads this step's keys from the cache). The conv state
write of a multi-token step must come after the conv that reads the same stream (the
same ring rows hold the history and, for a long chunk, its tail).

## 5. Split-K, workspace and graph capture

* **Decode** (`q_len == 1`): one wave per `(request, kv head, split)` streams 16-key
  super-tiles (8 lanes per key, K / V straight from the page array, no LDS). The grid
  is head-fastest, `(H_kv, ceil(splits / 4), R)` with 4 waves per block: the waves
  that run together read the sibling heads of the same key rows, which keeps the
  token-major cache in the same DRAM pages (128k full layer: 2300 us with the
  split-fastest grid vs 1277 us, [INFERENCE] for the cause). A request with `n`
  super-tiles uses `min(splits_decode, n)` partials, so `splits_decode` is an upper
  bound that short sequences do not pay for.
* **Prefill / chunked prefill / spec-verify** (`q_len >= 2`): one 128-thread CTA per
  (16 query tokens, kv head, split) walks 64-key tiles in LDS (255 VGPRs, 55.5 KB
  LDS at `g = 3`); `splits_tile` splits the key range of one CTA row when the grid
  would otherwise be smaller than the GPU.
* **Split policy** (`inkling2_attention_splits`, measured within about 3% of the best
  swept value on one V620): `keys = window + 16` (sliding) or `max_seq_len` (full),
  `groups = R * H_kv`; `splits_decode = min(max(ceil(256 / groups), 1),
  max(ceil(keys / 16) // 2, 1), 256)` (about 256 waves, at least two super-tiles per
  wave); `splits_tile = 1` unless `max_query_len > 1` and `ceil(max_q / 16) * groups
  < 144`, then `min(ceil(144 / ctas), max(key_tiles // 4, 1), 16)`.
* Partials are merged by `i2_combine_kernel` (one wave per q head, LDS weights),
  which also adds the sink to the denominator and applies the gate. A request
  whose range fits one split is finished in the main kernel; the combine kernel
  is only launched when some split factor is `> 1`.
* **Workspace** (fp32): `rows = splits_decode * R + splits_tile * T` partial rows of
  `nh * 130` floats (128 accumulator + `(max, sum)`), decode rows first, tile rows
  after them. `inkling2_attention_workspace_numel(R, T, nh, S_A, S_B)` returns that
  size and is always sufficient; the kernel itself only needs the rows it uses
  (none for a split factor of 1, no tile rows without a tile request) and rejects a
  smaller tensor. The contents never need initialising (the tests fill the
  workspace with NaN).
* **Graph capture**: capture with the batch's padded `R`, `T`, the full-length
  bounds from `inkling2_attention_splits(R, H_kv, window, max_model_len,
  max_query_len)` and a workspace sized for them. Replay may carry any
  `seq_lens` / `block_table` / `query_start_loc` content; `max_query_len == 1`
  graphs (FULL decode) launch only the decode and combine kernels, mixed
  PIECEWISE graphs also the tiled kernel. `tests/.../test_attention_cuda_graph_replay`
  captures once and replays with new lengths and block tables.
* The grid depends on host integers only (`R`, `H_kv`, splits, `max_query_len`);
  tokens beyond `query_start_loc[R]` and empty / padded requests are inert.

## 6. Numerics and deviations from the eager torch path

| Item | Torch path | Kernel | Effect |
| --- | --- | --- | --- |
| Scores / softmax / `P V` accumulation | fp32 | fp32 accumulate; `exp` is `__expf` | ~1 ulp(fp32) |
| `P` in the tiled kernel | fp32 | fp16 pairs for `V_DOT2` (relative `2^-11` per weight) | about `5e-4` relative on the output of prefill rows; the decode kernel keeps fp32 `P` |
| Scaled clip of q / rel | fp32 clip, round to fp16, (log scale: fp32 multiply, round to fp16) | the same two roundings | the oracle's prologue is bit-identical to `Inkling2Attention._scaled_clip` (CPU test `test_scaled_clip_matches_repo`); the kernel is checked against the oracle within the tolerance below |
| `rel_logits` | fp32 einsum of the fp16 rel features and `proj.float()` (`proj` is an fp16 parameter in serving) | the same fp16 `proj` values (`rel_proj_t`), fp32 `V_DOT2` accumulate | summation order only |
| Output | `(attn.float() * sigmoid(gate.float())).to(fp16)` | the same, per element | identical rounding point |
| conv taps | fp32 `x` (from `linear(h).float()`); the reference ring state is fp16 as well | fp16 `x` is widened to fp32; the state is an fp16 ring | exact for fp16 GEMM outputs (the serving path); an fp32 `x` that is not fp16-representable is stored rounded to fp16, so only its history taps carry the rounding (relative `2^-11`), the taps inside the chunk are exact |
| `k` | fp32 `l2_clip(conv)` rounded to fp16 on the cache write | the same | |
| `v` | `v * rsqrt(mean(v^2) + eps)` in fp32 | `rsqrtf` (hardware, `~1 ulp`) | below the fp16 rounding |

The tests compare against an fp64 CPU oracle (`AttnWorld.oracle`) with
`rtol = 4e-3`, `atol = 8e-4`; the absolute floor is set by the tiled kernel's fp16
probabilities (section 8). They print `max_abs_err`, `max_rel_err(|ref| >= 1e-2)`
and `worst_err / bound` for every case.

## 7. Reproduction

Tests (gfx1030 box, built tree on `PYTHONPATH`; GPU tests skip off RDNA2 or when the
ops are not registered, the CPU tests of the oracle and of the split policy always
run):

```bash
PYTHONPATH=<tree>:<transformers-main> pytest -q -s tests/kernels/attention/test_inkling2_attn_rdna2.py
```

Benchmark (CUDA-graph replay of the kernels vs the repo's eager torch path; `--cold`
cycles distinct caches so the 128 MB L3 does not hold the K\|V rows, `--layout`
selects the page layout, `--verify-q` adds speculative-verify shapes, `--chunk-ctx`
a 1024-token chunk at the end of a long prefill):

```bash
python tools/rdna/inkling2/bench_attn.py --md --cold --tp 1 --verify-q 4 --chunk-ctx 16384 131072
python tools/rdna/inkling2/bench_attn.py --md --cold --tp 4 --only attn --ctx 16384 131072 1048576 --batch 1 --torch-iters 0
python tools/rdna/inkling2/bench_attn.py --sweep-splits --only attn --ctx 16384 131072 --batch 1 --verify-q 2 4 8 --prefill 0 --torch-iters 0
```

## 8. Verification

`tests/kernels/attention/test_inkling2_attn_rdna2.py`: 863 tests (the conv ops are the large
majority, attention decode / prefill / mixed / layouts / long context / graph replay the rest),
all passing on the real registered-ops build (no dev aliasing) on a V620 (HIP 3, merged tree,
`863 passed in 381 s`; 45 of them are the padding and null-block tests listed below).
Every GPU case compares against an fp64 CPU oracle built from the dense K / V (the
oracle's prologue is bit-identical to `Inkling2Attention._scaled_clip`) and checks the
poisoned memory: unused cache pages, padding rows and the split workspace are filled
with NaN, so a read of anything the kernel must not read fails the case.

| Group | Cases |
| --- | --- |
| CPU | oracle vs the repo modules (`_scaled_clip`, conv), split policy properties, workspace formula |
| `sconv_static`, `sconv_dynamic`, `conv_state_write`, `kv_prep` | ring geometries `(block 4, ring 3)`, `(64, 3)`, `(16, 10)` x 4 request mixes (decode, prefill, chunked with history in earlier blocks, one with an empty request; blocks wholly below the history are the null block) x widths / dtypes / kv head counts x padded token / short-metadata cases x both page layouts; after every call the **whole** conv cache is compared bit for bit with the expected ring (the rule's rows written, everything else, NaN pages included, untouched), no input argument may change, every third conv input is 4x louder so the k L2 clip engages on some rows and not on others; each compute op is first run with `write_state = False` (the state must stay bit-identical), then followed by `inkling2_conv_state_write`, and for the one-token-per-request mix also with the fused `write_state = True`; `conv_state_write` puts two streams of different dtype and width into one page and runs with the page between four other layers' pages (`block5`) |
| ring rollback | a `1 + drafts` token verify step of which 1 / 2 / 4 / 8 stay (`drafts` 4 / 7, block 4 / 16 / 64), the junk rows of the rejected tokens in the ring, then the next step reads its history from it: with `3 + drafts` rows the outputs equal the oracle, for a multi-token step and for one-token fused steps; fp32 inputs that are not fp16-representable are stored rounded (`conv_state_rounds_fp32_inputs_to_fp16`) |
| padding / null block | `conv_state_write`, the fused `sconv_static` write and `kv_prep` on the ring geometries, 45 cases: a request whose block table is the null block (block 0 is `BlockPool.null_block`, not a page; its NaN page must stay NaN), padded tokens that the engine's token -> request map gives to request 0 past the last chunk (`p >= seq_len`; the NaN-poisoned rows of request 0 must stay NaN), and rows whose `seq_len` is shorter than their query (`seq_len` 1, `q_len` 6, zeroed block tables: positions below 0, which would index the table at row -1); after each call the whole cache is compared bit for bit with the expected ring |
| `attention` decode / prefill / mixed | `q_len` 1, 2..1024, mixed batches with explicit splits, `g = 1..4`, gate / no gate, strided (fused-projection) inputs, int32 positions, small `rel_extent`, window 0 / 1024 |
| long context | TP=4 full layer with 1,048,576 tokens (three split settings) |
| cache layouts | head-major, token-major, and both interleaved with two other layers' NaN pages (block stride = 3 pages) |
| graphs / API | CUDA-graph replay with new lengths and block tables, padded requests, host-check rejections (workspace, splits), fake-tensor tracing of all four ops |

Errors against the oracle (fp16 output, `rtol = 4e-3`, `atol = 8e-4`;
`max_rel` over `|ref| >= 1e-2`; `worst / bound` is the largest `err / (atol + rtol |ref|)`
of the group, so `< 1` passes):

| Group | Cases | max abs | max rel | worst / bound |
| --- | ---: | ---: | ---: | ---: |
| decode | 12 | 9.7e-4 | 3.1e-3 | 0.11 |
| prefill | 12 | 1.26e-3 | 3.5e-2 | 0.46 |
| mixed | 10 | 1.0e-3 | 2.4e-2 | 0.29 |
| GQA groups | 6 | 9.7e-4 | 2.3e-2 | 0.38 |
| block-outermost layouts | 4 | 9.9e-4 | 3.7e-2 | 0.52 |
| 1M tokens | 3 | 3.8e-4 | 8.7e-4 | 0.07 |
| graph replay | 1 | 1.0e-3 | 2.5e-2 | 0.31 |
| padded requests | 1 | 4.6e-4 | 4.6e-4 | 0.09 |
| `rel_extent = 200` | 1 | 9.3e-4 | 4.2e-2 | 0.51 |
| gate / unstrided | 3 | 9.7e-4 | 2.2e-2 | 0.30 |

The absolute floor (`~1e-3` on prefill rows) is the tiled kernel's fp16 probabilities
(section 6); an earlier run reached `worst / bound = 0.98` at `atol = 4e-4` and the
bound was widened to `8e-4` with that justification in the test file. Real bugs move
the output by `1e-2` and more (below).

**Mutation checks** (run by hand: each mutant is the kernel source with one exact-string
change, built as its own op library, and the attention tests run with `-x`; every
mutant must fail). 24 of 24 mutants failed on the part-A source:

| Mutant | Change | First failing test |
| --- | --- | --- |
| m1a | decode window start `L - window` -> `L - window - 1` | `attention_decode[bhc-8-sliding]` |
| m1b | decode window start `L - window` -> `L - window + 1` | `attention_decode[bhc-8-sliding]` |
| m1c | tile mask `j > pq - window` -> `j >= pq - window` | `attention_prefill[bhc-8-sliding]` |
| m1d | tile key range start `p_first - window + 1` -> `+ 2` | `attention_prefill[bhc-8-sliding]` |
| m2a | decode bias row `dist` -> `dist + 1` | `attention_decode[bhc-8-sliding]` |
| m2b | tile bias row `dist` -> `dist + 1` | `attention_prefill[bhc-8-sliding]` |
| m2c | decode bias applied for `dist <= rel_extent` | `attention_decode[bhc-8-full]` |
| m2d | tile bias applied for `dist <= rel_extent` | `attention_prefill[bhc-8-full]` |
| m3a | decode epilogue drops the sink (`den = l * w`) | `attention_decode[bhc-8-sliding]` |
| m3b | tile epilogue drops the sink | `attention_prefill[bhc-8-sliding]` |
| m3c | split combine drops the sink (`den += 0`) | `attention_decode[bhc-8-sliding]` |
| m4 | tile causal mask `j <= pq` -> `j <= pq + 1` | `attention_prefill[bhc-8-sliding]` |
| m5 | log scale `pos + 1` -> `pos` | `attention_decode[bhc-4-full]` |
| m6 | decode in-page slot mask loses its top bit | `attention_decode[bhc-8-sliding]` |
| m7 | decode octet merge forgets a rescale | `attention_decode[bhc-8-sliding]` |
| m8 | tile V pair interleave corrupts one dim | `attention_prefill[bhc-8-sliding]` |
| m9 | static conv tap order of channel 0 reversed | `sconv_static[hbc-...-decode]` |
| m10 | conv history read at position 0 dropped | `sconv_static[hbc-...-decode]` |
| m11 | v RMS norm with the sum instead of the mean | `kv_prep[hbc-padding0-8-static-decode]` |
| m12 | decode partial rows of different requests collide | `attention_decode[bhc-8-sliding]` |
| m13 | tile partials overwrite the decode partial rows | `attention_mixed_splits[splits1-sliding]` |
| m14 | dynamic conv adds the lag-1 tap to lag 0 | `sconv_dynamic[hbc-...-decode]` |
| m15 | `kv_prep` drops the k L2 clip | `kv_prep[hbc-padding0-8-static-decode]` |
| m16 | dynamic conv reads static weight row 2 for lag 1 | `sconv_dynamic[hbc-...-decode]` |

The first run of the m15 mutant survived all 256 conv tests (the random conv outputs
never exceeded the clip threshold of the k L2 clip). The conv world now makes every
third input row 4x louder, and the mutant fails (after 192 passing cases).

**Conv ring mutants.** The ring rewrite replaced the conv kernels, so the conv rows above
(m9, m10, m11, m14, m15, m16) were measured on the part-A source and the ring was mutated
again, with the same method (one exact-string change per mutant, its own op library, the
conv tests `-k "test_sconv_static or test_sconv_dynamic or test_conv_state_write or
test_kv_prep or test_rejected or test_conv_state_rounds"` with `-x`). All 8 ring mutants
fail (c9 - c11, the guards of the padding and null-block rules, were built and run as
described after the table):

| Mutant | Change | First failing test |
| --- | --- | --- |
| c1 | keep rule `p + ring >= min(p1, block_end)` -> `p + ring - 1 >= ...` (the oldest kept row is never written) | `sconv_static[hbc-padding0-8-288-dtype0-4-3-prefill]` |
| c2 | history ring row `sp % ring` -> `(sp + 1) % ring` | `sconv_static[hbc-padding0-8-288-dtype0-4-3-decode]` |
| c3 | state-write ring row `p % ring` -> `(p + 1) % ring` | `sconv_static[hbc-padding0-8-288-dtype0-4-3-decode]` |
| c4 | history block `sp >> blk_shift` -> `tp.p >> blk_shift` (history taken from the block of the current token) | `sconv_static[hbc-padding0-8-288-dtype0-4-3-decode]` |
| c5 | fused one-token write placed before the history read | `sconv_static[hbc-padding0-8-288-dtype0-4-3-decode]` |
| c6 | state write ignores the stream offset (`o0` -> 0) | `sconv_static[hbc-padding0-8-288-dtype0-4-3-decode]` |
| c7 | history head stride `cs1` -> `cs2` | `sconv_static[hbc-padding0-8-288-dtype0-4-3-decode]` |
| c8 | history row hard-coded to a 3-row ring (`sp % 3`) | `sconv_static[hbc-padding0-8-288-dtype0-16-10-decode]` after 8 passing cases: only the ring-10 geometry sees it |
| c9 | keep predicate `0 <= p && p < p1` -> `p < p1` (a row below the first position of the request is written: the block-table index and the ring row go negative) | `conv_state_write_ignores_rows_shorter_than_their_query[4-3]`, 3 failures (one per geometry) |
| c10 | keep predicate `0 <= p && p < p1` -> `0 <= p` (padding that reads as request 0 writes past its last position) | `conv_state_write_ignores_padding_that_reads_as_request_zero[hbc-64-3]`, 10 failures: `conv_state_write` (both layouts), the fused `sconv_static` write and `kv_prep` (static and dynamic) in the `(64, 3)` and `(16, 10)` geometries; the `(4, 3)` geometry does not see this mutant |
| c11 | `i2_write_ring` without `if (blk <= 0) return;` (a request whose block table is the null block writes into block 0) | `conv_state_write_skips_the_null_block[hbc-4-3-decode]`, 30 failures: `conv_state_write` (12), `kv_prep` (12), the fused `sconv_static` write (3) and `conv_state_write_ignores_rows_shorter_than_their_query` (3: the zeroed tables of that test make the null block the target) |

c9 - c11 are built from the mutated `.cu` with `ninja _rocm_C` only (its own library per mutant,
the source restored after every build, the checksum of the real library verified at the end: OK) and run
on a V620 with `-k "ignores_padding_that_reads_as_request_zero or ignores_rows_shorter or
skips_the_null_block"` and no `-x`: c9 3 failed / 42 passed, c10 10 failed / 35 passed, c11 30 failed /
15 passed. The Python side of the same defect (`SeqIdxScratch.fill` giving padded tokens `seq_idx = 0`)
is killed by `test_seq_idx_of_a_graphs_padding_is_minus_one` under a plugin that restores the old fill
(`seq_idx=[0, 1, 1, 1, 2, 2, 2, 0, 0, 0]`): 1 failed.

The five mutation classes of the tap math (m9, m11, m14, m15, m16) were applied to the ring
source too, by the same method; all 5 fail:

| Mutant | Change | First failing test |
| --- | --- | --- |
| m9r | static tap order of channel 0 reversed | `sconv_static[hbc-padding0-8-288-dtype0-4-3-decode]` |
| m11r | v RMS norm with the sum instead of the mean | `kv_prep[hbc-padding0-8-static-4-3-decode]` (after 411 passing cases) |
| m14r | dynamic conv lag 0 uses the lag-1 dynamic tap | `sconv_dynamic[hbc-padding0-8-4-3-decode]` (after 144 passing cases) |
| m15r | `kv_prep` drops the k L2 clip | `kv_prep[hbc-padding0-8-static-4-3-decode]` (after 411 passing cases) |
| m16r | dynamic conv reads static weight row 2 for lag 1 | `sconv_dynamic[hbc-padding0-8-4-3-decode]` (after 144 passing cases) |

So 16 of 16 conv mutants fail on the ring source (the part-A m10, "conv history read at
position 0 dropped", is covered by c2, c3, c4 and c7).  The other 16 rows of the part-A
table mutate the attention decode, prefill and split-combine kernels, which this change
did not touch; they were not re-run.

## 9. Performance

One V620 per run (HIP 2 / 3; the two cards agree within 2% on the same kernel), real
build, CUDA-graph replay (`--reps 20`, best of 5 rounds), page layout `bhc` (the
engine's default `LBNHC`), split factors from `inkling2_attention_splits`. The torch
column is the repo's eager path (`Inkling2Attention._attend`, `_scaled_clip`,
`conv_taps` / `kv_write`) on the same data, 3 iterations. `GB/s` is the K\|V bytes of
the launch over its time; `cold` cycles distinct caches so the 128 MB Infinity Cache
does not hold the rows (the 48 layer visits of the model do the same), and a warm
figure above the ~512 GB/s DRAM peak is an L3 hit, not DRAM bandwidth [INFERENCE for
the peak]. `TFLOP/s` counts `4 * pairs * 128 * nh` useful flops.

TP=1 (24 q heads / 8 kv heads), `python tools/rdna/inkling2/bench_attn.py --md --cold
--tp 1 --verify-q 4 --chunk-ctx 16384 131072`:

| case | splits | kernel us | GB/s | cold us | cold GB/s | TFLOP/s | torch us | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attn decode sliding B=1 ctx=1024 | 32/1 | 13.6 | 308 | 29.1 | 144 | - | 983 | 72x |
| attn decode sliding B=1 ctx=16384 | 32/1 | 13.4 | 317 | 29.4 | 145 | - | 977 | 73x |
| attn decode sliding B=1 ctx=131072 | 32/1 | 13.7 | 311 | 29.4 | 145 | - | 980 | 72x |
| attn decode sliding B=4 ctx=1024 | 8/1 | 29.0 | 578 | 56.4 | 297 | - | 2657 | 92x |
| attn decode sliding B=4 ctx=16384 | 8/1 | 28.7 | 593 | 56.2 | 303 | - | 2664 | 93x |
| attn decode sliding B=4 ctx=131072 | 8/1 | 25.1 | 678 | 56.3 | 302 | - | 2683 | 107x |
| attn verify q=4 sliding B=1 ctx=1024 | 32/4 | 43.1 | 97 | 51.6 | 81 | - | 1022 | 24x |
| attn verify q=4 sliding B=1 ctx=16384 | 32/4 | 48.1 | 89 | 54.8 | 78 | - | 1029 | 21x |
| attn verify q=4 sliding B=1 ctx=131072 | 32/4 | 48.1 | 88 | 54.8 | 78 | - | 1029 | 21x |
| attn verify q=4 sliding B=4 ctx=1024 | 8/4 | 83.0 | 202 | 100.1 | 168 | 2.42 | 2849 | 34x |
| attn verify q=4 sliding B=4 ctx=16384 | 8/4 | 90.3 | 188 | 105.8 | 161 | 2.23 | 2866 | 32x |
| attn verify q=4 sliding B=4 ctx=131072 | 8/4 | 90.4 | 188 | 106.1 | 160 | 2.23 | 2853 | 32x |
| attn prefill sliding q=1024 | 32/1 | 534.7 | - | - | - | 12.06 | 5434 | 10x |
| attn chunk sliding q=1024 ctx=2048 | 32/1 | 993.6 | - | - | - | 12.97 | - | - |
| attn decode full B=1 ctx=1024 | 32/1 | 13.8 | 305 | 29.6 | 142 | - | 1171 | 85x |
| attn decode full B=1 ctx=16384 | 32/1 | 66.1 | 1015 | 181.8 | 369 | - | 5303 | 80x |
| attn decode full B=1 ctx=131072 | 32/1 | 1208.3 | 444 | 1208.3 | 444 | - | 24998 | 21x |
| attn decode full B=4 ctx=1024 | 8/1 | 28.8 | 583 | 57.0 | 294 | - | 2817 | 98x |
| attn decode full B=4 ctx=16384 | 8/1 | 635.1 | 423 | 635.1 | 423 | - | 19906 | 31x |
| attn decode full B=4 ctx=131072 | 8/1 | 4689.3 | 458 | 4689.3 | 458 | - | 98804 | 21x |
| attn verify q=4 full B=1 ctx=1024 | 32/4 | 43.6 | 96 | 52.0 | 81 | - | 1224 | 28x |
| attn verify q=4 full B=1 ctx=16384 | 32/16 | 255.6 | 263 | 293.1 | 229 | - | 5481 | 21x |
| attn verify q=4 full B=1 ctx=131072 | 32/16 | 2272.8 | 236 | 2272.8 | 236 | - | 27115 | 12x |
| attn verify q=4 full B=4 ctx=1024 | 8/4 | 86.0 | 195 | 101.1 | 166 | 2.34 | 2992 | 35x |
| attn verify q=4 full B=4 ctx=16384 | 8/5 | 1134.8 | 237 | 1134.8 | 237 | 2.84 | 20724 | 18x |
| attn verify q=4 full B=4 ctx=131072 | 8/5 | 9297.2 | 231 | 9297.2 | 231 | 2.77 | 107518 | 12x |
| attn prefill full q=1024 | 32/1 | 597.0 | - | - | - | 10.80 | 5636 | 9x |
| attn chunk full q=1024 ctx=16384 | 32/1 | 14379.1 | - | - | - | 13.89 | - | - |
| attn chunk full q=1024 ctx=131072 | 32/1 | 124469.0 | - | - | - | 13.20 | - | - |
| kv_prep static(sliding) B=1 q=1 | - | 4.9 | - | - | - | - | 3101 | 629x |
| kv_prep static(sliding) B=4 q=1 | - | 4.9 | - | - | - | - | 3149 | 637x |
| kv_prep static(sliding) B=1 q=1024 + 2 state writes | - | 24.7 | - | - | - | - | 3169 | 128x |
| kv_prep dynamic(full) B=1 q=1 | - | 4.9 | - | - | - | - | 3343 | 678x |
| kv_prep dynamic(full) B=4 q=1 | - | 5.0 | - | - | - | - | 3375 | 676x |
| kv_prep dynamic(full) B=1 q=1024 + 2 state writes | - | 23.8 | - | - | - | - | 3415 | 143x |
| sconv_static post-conv C=2304 B=1 q=1 | - | 4.2 | - | - | - | - | 1431 | 341x |
| sconv_static post-conv C=2304 B=4 q=1 | - | 4.2 | - | - | - | - | 1475 | 350x |
| sconv_static post-conv C=2304 B=1 q=1024 + 1 state write | - | 23.7 | - | - | - | - | 1508 | 64x |

Reading the table:

* **Decode** is DRAM-bound on full layers at 128k (444 GB/s cold, 87% of the ~512 GB/s
  peak [INFERENCE for the peak]) and launch / latency-bound on sliding layers (13.6 us
  = decode kernel + the 4.6 us combine; a trivial graph kernel node is 3.4 us).
  The sliding window costs the same at any context (the first row of each block).
* **Speculative verify** (`q = 4`) goes through the tiled kernel: 21-36x faster than
  torch, but 1.9-2.0x the decode time on full layers at long context (2273 vs 1208 us at
  128k) because a 16-token tile computes 16 rows for 4 live ones. A 4-token tile for
  `q <= 4` would remove most of that [INFERENCE: ~3-4x less tile compute].
* **Prefill** runs at 11-13 TFLOP/s, 13.2 TFLOP/s for a 1024-token chunk at 128k depth
  (124 ms per full-layer visit at TP=1).
* `kv_prep` / `sconv` replace 1.4-3.4 ms of torch gather / conv / scatter work with
  4.2-5.0 us (decode, state write fused) and 24-25 us (1024 tokens, including the
  separate `inkling2_conv_state_write` launches). The conv rows were re-measured with
  the ring state (block 64, 3 rows, fp16; `--only kv sconv`, HIP 3): the decode rows are
  0.4-0.5 us slower than with the previous per-token fp32 slots (4.4-4.6 / 3.8 us;
  [INFERENCE] the ring-row arithmetic and the block-table lookup per history tap), the
  1024-token rows 30-40% faster (35 / 34 / 40 us before), because only the last three
  positions of every block of the chunk are written instead of every token. The torch
  column is the reference path with the same ring semantics, so it is not comparable to
  the older
  table. The bench's attention-stream `x_post` is still fp32 (the serving path passes
  fp16), which only affects that row's load width.

TP=2 / TP=4 (`--tp 2` / `--tp 4`, 12 / 6 q heads), `kernel us / cold us` (kernel only):

| case | TP=2 | TP=4 |
| --- | ---: | ---: |
| decode sliding B=1 ctx=131072 | 12.0 / 12.9 | 11.7 / 11.4 |
| decode sliding B=4 ctx=131072 | 16.9 / 36.1 | - |
| decode full B=1 ctx=16384 | 41.0 / 98.6 | 30.1 / 61.5 |
| decode full B=1 ctx=131072 | 640.5 / 640.5 | 128.8 / 354.3 |
| decode full B=1 ctx=1048576 | 4782 / 4782 | 2498 / 2498 |
| decode full B=4 ctx=131072 | 2422 / 2422 | - |
| verify q=4 full B=1 ctx=131072 | 1123.5 / 1123.5 | 701.3 / 725.3 |
| verify q=4 full B=1 ctx=1048576 | - | 5877 / 5877 |
| prefill q=1024 sliding / full | 299 / 317 | 190 / 196 |

The 1,048,576-token full-layer decode is DRAM-bound at both TP degrees (449 and 430
GB/s). Big batches (TP=1, ctx 4096): sliding decode B=16 82.6 us, B=64 670.8 us; full
decode B=16 641.9 us, B=64 2454 us (418-438 GB/s cold).

Other measurements on the same kernel:

* **Page layout**: the head-major layout (`--layout hbc`, `LBHNC`) is 5-8% slower than
  the token-major default on full layers at 128k (1266 vs 1208 us at B=1, 5041 vs 4689
  us at B=4) and 15% faster for sliding B=4 (24 vs 29 us); both work without a copy.
* **Decode split sweep** (earlier dev-build sweeps, within 3% of the shipped policy):
  sliding B=1 best `S = 32` (13.6 us; `S = 16` 14.4 us), B=4 `S = 8`; full 128k B=1
  `S = 16` (1209 us; 1277 us at 128), 16k `S = 64`; TP=4 128k `S = 128`.
* **Verify split sweep** (this run, B=1, `Sb` = `splits_tile`): full layers best
  `Sb = 8` at 16k (254 us; the policy's 16 is 256 us) and 16 at 128k (the policy's
  choice); sliding layers prefer `Sb = 16` (q=4: 37.1 us vs 48.1 us at the policy's
  `Sb = 4`, which is capped at `key_tiles // 4`). Relaxing that cap for sliding layers is
  a Python-only change (`inkling2_attention_splits`), not applied because it was not
  re-validated on the GPU after the sweep.
* **Prefill ablation** (dev builds, `q = 1024`, sliding / full, us): base 544 / 581;
  without the bias 515 / 547; without global loads 480 / 492; without P V 405 / 409;
  without S 384 / 393: no single dominant cost.
* **Tile geometry**: a 256-thread CTA (3 rows per thread, 143 instead of 255 VGPRs,
  twice the resident waves, same LDS) was built and run on two cards: prefill q=1024
  sliding 599 / 610 us vs 533 / 533 us, full 648 / 650 vs 572 / 575 us, 131072-deep chunk
  144 / 145 ms vs 125 / 128 ms (-12 to -13%), verify q=4 sliding 45.8 vs 48.0 us (+5%).
  The shipped kernel keeps 128 threads; the geometry derives from `kI2TileThreads`.

## 10. Limits, risks and integration notes

* **Supported shapes**: `head_dim = 128`, GQA group `g = nh / H_kv` in `[1, 4]`
  (`g = 3` at every TP degree; the `g = 4` tiled kernel spills registers and is not
  used by the model), `d_rel = 16`, conv kernel size 4, KV block size a power of two
  `>= 16`, `rel_extent <= rows of rel_proj_t`, fp16 q / k / v / out. Anything else is
  rejected by a host check (`TORCH_CHECK`), never silently computed wrongly.
* **`max_query_len` must be a true upper bound** of the batch's `q_len`; longer
  requests are not computed. Capture FULL decode graphs with `max_query_len = 1` and
  mixed / spec-verify graphs with the real maximum (`1 + num_speculative_tokens`).
* **Metadata must live in persistent device buffers** for graph replay (`block_table`,
  `seq_lens`, `query_start_loc`, `slot_mapping`, `seq_idx`, `positions`).
  `Inkling2MetadataBuilder._cudagraph_support` is `NEVER` in the repo and has to
  change for FULL graphs; padded requests must carry `seq_lens = 0`, padded tokens
  `slot_mapping = -1` / `seq_idx = -1` (they produce zero rows and write nothing).
  The conv ops read neither a conv `slot_mapping` nor anything else per token than
  `seq_idx`; `write_state` is a launch constant, so a captured one-token-per-request
  decode graph uses the fused form and a verify / mixed graph the two-launch form
  (`conv_writes_in_op(md)` = `md.max_query_len == 1`).
* **Conv state**: fp16 ring rows addressed by position (`p % ring`, block
  `p / block_size`); the ring needs `3 + num_speculative_tokens` rows so that a
  rejection of draft tokens cannot replace history the next step reads, and the
  block table the ops are given must hold, for every position of the step and the
  three before it, a block that belongs to the request (the engine's forward table
  shows blocks below the window as the null block, which is only read for positions
  below the history and never written). Two requests may not write the same block in
  one step: a request that would prefix-hit blocks another request writes in the same
  step has to wait a step (`Inkling2ConvStateManager`). fp32 inputs are stored rounded
  to fp16.  Positions outside `[0, seq_len)` and block 0 are never written (section 3),
  so a graph's padding and a capture's dummy batch need no special metadata beyond
  `seq_idx = -1` for the padded tokens.
* **TP > 1**: pass each rank's slice of `q_scale` `[nh_local, 128]`,
  `rel_scale` `[nh_local, 16]`; `rel_proj_t` is replicated (it is shared by the heads).
  The serving model shards by KV-head group (`amd/loader.py:local_slice`: rank `r`
  takes heads `[r * nkv / tp, (r + 1) * nkv / tp)` and their `g = 3` q heads; the KV
  pages and the conv owners hold the local heads only).
* **Deviations** (section 6): fp16 probabilities in the tiled kernel (about `5e-4`
  relative on prefill / verify rows), `__expf` / `rsqrtf`; documented, covered by the
  tolerance of the tests, not hidden.
* **Performance tuning was done on one V620** (HIP 2); the split policy and the
  head-fastest decode grid are measured choices, not derived ones [INFERENCE for
  other parts]. Warm numbers below about 16k context are partly Infinity-Cache hits
  (128 MB L3); use the `cold` columns for DRAM-bound claims.
* **Long-context prefill is the remaining lever**: the tiled kernel reaches about 11
  TFLOP/s (27% of the `V_DOT2` peak of about 42 TFLOP/s, [INFERENCE for the peak]),
  so a 1024-token chunk on a full layer costs about 125 ms at 128k depth (TP=1) and
  full-attention prefill is quadratic in the prompt. A 256-thread CTA (3 rows per
  thread, 143 VGPRs, twice the resident waves) was built and measured: it is 12-13%
  slower on prefill (section 9), so the shipped kernel keeps 128 threads
  (`kI2TileThreads`, the tile geometry derives from it).
* **Build / repo notes**: the benchmark lives in `tools/rdna/inkling2/` as assigned
  (AGENTS.md prefers `benchmarks/kernels/`); the mutation harness used for section 8
  is a scratch tool outside the repo, the exact one-line mutations are listed there so
  they can be reproduced by hand.
