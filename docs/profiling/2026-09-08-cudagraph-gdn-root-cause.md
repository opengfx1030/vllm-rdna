# Cudagraph Garbage — Root-Cause Narrowing (2026-09-08, session 2)

**Branch**: `rdna_extras` @ `9e7184d34`
**Model**: Qwen3.8-27B-AWQ-INT4 (arch `Qwen3_5ForConditionalGeneration`, GDN hybrid), TP=2, 4× V620 gfx1030
**Status**: root cause **narrowed to the GDN hybrid recurrent-state path under piecewise capture**; not yet fixed.

## Decisive control matrix

All rows are the same 12-commit build, same env, same prompts, `temperature=0`.

| Configuration | "The capital of France is" | "1+1=" | Verdict |
|---|---|---|---|
| GDN hybrid + `--enforce-eager` | `' Paris'` | `'2'` | **correct** |
| **Dense** AWQ (Qwen2.5-0.5B-AWQ) + cudagraph PIECEWISE | `' Paris'` | `'2'` | **correct** |
| GDN hybrid + cudagraph PIECEWISE | `'oug'` | `'、Ginaxid…'` | garbage |
| GDN hybrid + cudagraph + GEMMs in `splitting_ops` | `'oug有一颗…'` | `'、Ginaxid…'` | garbage (deterministic) |
| GDN hybrid + breakable cudagraph | `'duct'` (×N, prompt-independent) | `'duct'` | garbage (input-blind) |
| GDN hybrid + cudagraph, capture sizes `[1,2,4]` | `' the'` / `' about_Theci 4'` (varies) | `'（'` | garbage (**non-deterministic**) |

**The dense-AWQ row is the key result.** It proves the piecewise cudagraph
machinery, the AWQ int4 GEMM path (`gptq_gemm_rdna2`, `awq_gemm_rdna2_prefill`),
and FA-RDNA2 are all cudagraph-safe on this tree. The corruption is specific to
the **GDN hybrid** model — i.e. the mamba/GDN recurrent state path.

## Structural bugs found in GDN cudagraph metadata

File: `vllm/v1/attention/backends/gdn_attn.py`

### 1. Static state-index buffers are skipped in PIECEWISE mode

```python
self.use_full_cuda_graph: bool = (
    self.compilation_config.cudagraph_mode.has_full_cudagraphs()   # False for PIECEWISE
)
...
if (self.use_full_cuda_graph and num_prefills == 0 and num_spec_decodes == 0
        and num_decodes <= self.decode_cudagraph_max_bs):
    self.non_spec_state_indices_tensor[:num_decodes].copy_(...)   # static buffer
```

`has_full_cudagraphs()` is False for `PIECEWISE`, so the static-buffer copy is
skipped and the model forward receives the raw per-step
`block_table_tensor[:, 0]` view. `causal_conv1d_update` is called directly in the
model forward (`qwen_gdn_linear_attn.py:1775`) and is **not** a splitting op, so
it is captured — while `qwen_gdn_attention_core` **is** a splitting op and runs
eager. The captured conv1d therefore bakes in a capture-time pointer to a
per-step tensor.

### 2. Static buffer sizing does not cover the capture sizes

```python
self.decode_cudagraph_max_bs = max_num_seqs * (num_spec + 1)      # 4 * 1 = 4
if max_cudagraph_capture_size is not None:
    self.decode_cudagraph_max_bs = min(..., max_cudagraph_capture_size)  # min(4,8) = 4
```

With `--max-num-seqs 4`, buffers are sized **4**, but the default
`cudagraph_capture_sizes` is `[1,2,4,8]` — a graph is captured at batch **8**
against size-4 static buffers.

## Attempted fix and why it was reverted

Changed the two gates from `use_full_cuda_graph` to a new
`use_static_state_buffers = has_full_cudagraphs() or has_piecewise_cudagraphs()`.

Result: still garbage, and output became **non-deterministic** (same prompt,
`temperature=0`, different first token per request). Raising `--max-num-seqs` to
8 (fixing bug 2) and bounding capture sizes to `[1,2,4]` did not help either.

Reverted, because a change that makes behaviour non-deterministic without an
understood mechanism must not be shipped. Tree restored to `9e7184d34`.

**Non-determinism at `temperature=0` for identical prompts is the signature of a
kernel reading uninitialised or racing memory.** Note the static-buffer copies
use `non_blocking=True`, and the captured graph may replay before the async copy
lands — a plausible race, not yet proven.

## Secondary findings

### Breakable cudagraph is net-harmful here, and my earlier whitelist never matched

Commit `031c1cdfc` whitelisted `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`,
but this model's architecture is **`Qwen3_5ForConditionalGeneration`** — so
breakable cudagraph never auto-enabled and the `@eager_break_during_capture`
decorator on the GDN forward never took effect in any earlier probe.

When forced on (by adding the correct arch name), output became
prompt-independent `'duct'` — total input blindness, i.e. the graph replays
without receiving real inputs. Strictly worse than the prompt-dependent garbage,
so the whitelist addition was reverted.

### HIP causal_conv1d is unreachable dead code with a proven FIR bug

- `causal_conv1d_update_rdna2` / `causal_conv1d_fwd_rdna2` exist in
  `csrc/rocm/causal_conv1d_rdna2.cu` but are **not registered** as torch ops.
  Live check: 36 `_rocm_C::` schemas, **zero** conv1d ops. No binding in
  `csrc/rocm/torch_bindings.cpp`, no `ops.h` declaration, no Python dispatch in
  `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`.
  (AGENTS.md claims commit `feb7b457e` added the binding and that 38 schemas
  register — neither is true on this tree.)
- Consequently `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=1` / `..._FWD=1` are no-ops.
- The update kernel's FIR pairing is wrong. After shift-left it does
  `acc += state_ptr[k] * w_ptr[k + 1]`, but post-shift
  `state_ptr[k] = x[t-state_len+1+k]`, so canonical `w[j]*x[t-j]` requires
  `j = state_len-1-k`. For `state_len=3` the code pairs k=0→w[1], k=1→w[2],
  k=2→w[3]; correct is k=0→w[2], k=1→w[1], k=2→w[0]. The fresh token receives
  `w[width-1]` instead of `w[0]`. The `fwd` kernel uses yet another (also
  inconsistent) pairing.

## Ruled out this session

- Piecewise cudagraph machinery — works (dense AWQ correct)
- AWQ int4 GEMM kernels — cudagraph-safe (dense AWQ correct; also Cell F)
- FA-RDNA2 — cudagraph-safe (dense AWQ correct; also Cell Afa)
- `torch::empty` uncommitted-page faults — no UTCL2/gfxhub page fault in dmesg
  during any probe
- Inductor `combo_kernels` fusion — disabled, no effect
- GEMM ops as `splitting_ops` — no effect (only made garbage deterministic)
- Capture-size/batch-size mismatch alone — bounding to `[1,2,4]` did not fix it

## Recommended next steps

1. ~~**Prove or disprove the `non_blocking=True` race.**~~ **DISPROVEN.**
   Re-applied the static-buffer change together with `non_blocking=False` on all
   8 state-index copies in `gdn_attn.py`. Output was still garbage and still
   non-deterministic (4 identical `temperature=0` requests gave
   `' thehi sn0hh'`, `' consideringCoindlagfsed'`, `' the specificoutouteObout'`,
   `' thebleeachcreas-'`). The async copy is therefore **not** the race.
   Reverted; tree restored to `c79774012`.
2. **Chase the uninitialised-memory read directly.** Non-determinism at
   `temperature=0` for identical prompts means a captured kernel reads memory
   whose contents vary between runs. The GDN static buffers in
   `gdn_attn.py:127-140` (`spec_state_indices_tensor`,
   `non_spec_state_indices_tensor`, `spec_sequence_masks`, …) are all allocated
   with `torch.empty` and only partially initialised per step — the tail fill
   happens inside the gate, and `build_for_cudagraph_capture` may leave them
   uninitialised at capture time. Switching these to `torch.zeros` and auditing
   every field the captured GDN region reads is the highest-value next step.
3. **Register and fix the HIP conv1d ops** (FIR pairing per above, plus
   `torch_bindings.cpp` + `ops.h` + Python dispatch) so the state update is an
   AOT op that can be added to `splitting_ops` — making the whole GDN layer
   eager exactly like the proven-correct eager path, while GEMMs/norms/attention
   stay captured.
4. **Write the standalone conv1d correctness test before enabling it** (the
   lesson AGENTS.md already records from the earlier false-positive).

## Config permutations tested (all still garbage)

| Permutation | Result |
|---|---|
| default PIECEWISE | garbage, deterministic |
| PIECEWISE + GEMM ops in `splitting_ops` | garbage, deterministic |
| PIECEWISE + breakable cudagraph | garbage, prompt-independent `'duct'` |
| PIECEWISE + static buffers for piecewise | garbage, **non-deterministic** |
| PIECEWISE + `--max-num-seqs 8` (buffers sized to capture size 8) | garbage, non-deterministic |
| PIECEWISE + `cudagraph_capture_sizes=[1,2,4]` (≤ `decode_cudagraph_max_bs`) | garbage, non-deterministic |
| PIECEWISE + static buffers + `non_blocking=False` | garbage, non-deterministic |
| `--enforce-eager` | **correct** |

