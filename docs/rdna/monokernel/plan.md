# MonoKernel phases (gfx1030, Flash-Next FFN-only)

Ordered work for the spec in [README.md](README.md). Each phase ends before
the next starts. No phase adds a kernel to `csrc/` until P0 has a V620 log.
Numbers below that are not in a cited run are **(unverified)**.

## P0 — probe and runner questions

Two tracks, no MonoKernel in the server.

### P0a — standalone probe

Drafter: ROCM_specialist. Sources live in
[`tools/monokernel-p0-probe/`](../../../tools/monokernel-p0-probe/README.md).

Compiled and run on a V620 (ROCm 7.14) on 2026-10-14; results in
[p0-results.md](p0-results.md).

The probe's grid is 72×256. Dynamic LDS defaults to 65536 minus the
kernel's static LDS (`hipFuncGetAttributes`), rounded down to 16 bytes.
`--lds` that would put static plus dynamic over 65536 exits 2
([probe README](../../../tools/monokernel-p0-probe/README.md)). The 64 KiB
figure in the silicon row is the per-workgroup cap, static plus dynamic. `run.sh` does all of the following, each
probe process under `timeout -s KILL 300`:

| Check | How |
| --- | --- |
| Plain vs cooperative capture | `--only graph`: stream capture, instantiate, replay, for a normal launch and for `hipLaunchCooperativeKernel` ([probe README](../../../tools/monokernel-p0-probe/README.md)). |
| Residency, WGP and `-mcumode` | `--only residency` on `build/wgp/probe` and `build/cumode/probe`. Code-object metadata is `kernel_meta.txt` and `amdhsa_directives.txt` from [`build.sh`](../../../tools/monokernel-p0-probe/build.sh). |
| Forward progress, both orders | `--only progress --order busy-first` and `--order flag-first`, both binaries. `--gap-ms` is the host gap so the GPU sees that order. |

Pass/fail reading (busy-first stall with flag-first pass means the protocol
is fine and the grid was not co-resident; a flag-first stall is a protocol
or residency bug) is the probe README, not a result. V620 logs and
results: [p0-results.md](p0-results.md#p0a--standalone-probe).

### P0b — Model Runner V2

Answer these from a serve log on dest, not from this page:

1. Does Flash-Next at TP4 stay up with `VLLM_USE_V2_MODEL_RUNNER=1`? The
   `flashnext-mtp2` recipe sets that variable
   ([recipe](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/tools/rdna/recipes/flashnext-mtp2.env)).
   A successful boot is **(unverified)** here.
2. Does `FULL_AND_PIECEWISE` capture the widths the launcher asks for?
   MTP=2 expands to 3, 6, 12, 24
   ([`rdna_mtp_args`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/tools/rdna/rdna_launcher_common.sh#L123-L136)).
   MTP off expands to 1, 2, 4, 8. Whether those graphs are the ones replayed
   on a decode step is **(unverified)**.
3. Does GDN or mamba-align refuse V2? No refuse string showed up under
   `vllm/models/qwen4_exp/` at `e245a469`. A worker log is still required
   **(unverified)**.
4. How many GPU launches does one decode step issue on the current path?
   **(unverified)**. The RFC's GLM-5.2 example (about 2,459 launches down
   to about 210) is a different model
   ([RFC](https://github.com/vllm-project/vllm/issues/60904)).
5. Does this port need Model Runner V2? The RFC says yes. #61325 dispatches
   from the uncompiled outer `forward` and its description does not mention
   V2 ([#61325](https://github.com/vllm-project/vllm/pull/61325)). If dest
   stays on V1, the hook has to be the V1 capture path. Decide after the
   log, and record it in this file.

   **Decided 2026-10-14:** no V2-specific hook. The FFN kernel is a custom
   op in the model forward and is captured into the FULL decode graph;
   production already serves under V2 with FULL widths 3/6/12/24. Answers
   to 1–5 with the evidence: [p0-results.md](p0-results.md#p0b--model-runner-v2-live-serve).

## 2a — FFN compute only

Kernel contents, in order: residual add and norm, router, shared expert
(dense, served fp16 on dest, `fdot2` with fp32 accumulate; router logits
and top-k in fp32, combine reuses the same ids/weights;
[wiki plan](https://github.com/BlivionIaG/rdna-hip-wiki/blob/e889bbd3/kernels/persistent/monokernel-gfx1030-plan.md)), routed experts (symmetric group W4A16,
`fdot2`). vLLM's all-reduce runs **before** the kernel and **after** it.
Do not change the `VLLM_RDNA_AR` protocol or its size gate
([tuning guide](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/docs/rdna2/tuning.md)).

Same-GPU flag waits only, agent scope, bounded, with the phase-id sticky
word from the probe. No cross-GPU wait inside the kernel.

Attention, GDN, causal conv, and QSA stay on the kernels dest already runs.
PIECEWISE capture is refused for the mono path, as the RFC refuses it
([RFC, routing table](https://github.com/vllm-project/vllm/issues/60904)).

Refusal from the README applies here: quantized router, shared expert, or
gated residual falls back to vLLM's layers. INT4 PLE
([#49](https://github.com/opengfx1030/vllm-rdna/pull/49)) and MTP drafter
experts ([#82](https://github.com/opengfx1030/vllm-rdna/pull/82)) are not
in this kernel.

Gate: stage numerics against vLLM's own ops on random weights in the
checkpoint layout, and a FULL-graph replay that stays bit-identical, the
same two bars the RFC lists for CI
([RFC, testing](https://github.com/vllm-project/vllm/issues/60904)). No
implementation of that test exists yet.

## 2b — in-kernel all-reduce

Start 2b only when both are true:

- 2a reduces time per output token on the batch-1 / 4 / 8 cells in phase 3.
- Blivion approves folding the collectives in.

Then both all-reduces move inside the kernel. Protocol stays the one in
[`rdna_allreduce.cuh`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh):
uncached staging, a peer announces with a posted store into the local flag,
the waiter polls locally, the wait is bounded, and the first timeout wins a
sticky abort that records the phase. Add the RFC's fail-stop on top: the
error word is copied next to the sampled tokens and `get_output()` raises
([RFC, safety](https://github.com/vllm-project/vllm/issues/60904)). The
engine-scale 120 s cap in that header
([`RDNA_AR_WAIT_MS`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L52-L62))
is not the mono-step bound. The mono-step bound has to be short enough that
one stuck peer becomes a sticky error and every later phase exits, which is
the RFC rule. The numeric bound is **(unverified)** until 2b is designed.

Do not land 2b on a timeout that still returns tokens.

## 3 — TPOT, GSM8K and long-context KL

Report time per output token, graphs against graphs, same batch cap:

| Cell | Batch | MTP | Widths |
| --- | --- | --- | --- |
| Decode, no spec | 1, 4, 8 | off | the launcher's MTP-off list is 1, 2, 4, 8 ([`rdna_mtp_args`](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/tools/rdna/rdna_launcher_common.sh#L125-L127)) |
| `flashnext-mtp2` | 1, 4, 8 | k=2 | 3, 12, 24 (batch × (1+k); the full launcher list is 3, 6, 12, 24) |

#61325 refuses spec decode
([#61325, limitations](https://github.com/vllm-project/vllm/pull/61325)).
Widths 3/12/24 are therefore past that template. They are in this plan
because the production recipe is MTP=2.

GSM8K: full 1,319 questions, graphs on, against the same build with the
kernel off, plus the RFC's paired comparison. #61325 used the model-card
recipe (thinking off, 2,048 tokens) and scored inside the stock spread.
Use that recipe as the procedure. The score itself is **(unverified)**.

Long-context KL against the BF16 reference: residual and gated-residual
activations can exceed the fp16 max (±65504), so run long contexts
([wiki plan](https://github.com/BlivionIaG/rdna-hip-wiki/blob/e889bbd3/kernels/persistent/monokernel-gfx1030-plan.md)). The overflow magnitude on Flash-Next is
**(unverified)** until measured on the V620.

- **Exact KL** needs the full logits dumped from the engine itself, on both
  the fp16 dest run and the BF16 reference.
- The OpenAI-compatible API returns only top-k logprobs, capped by
  `--max-logprobs` (default 20,
  [`ModelConfig.max_logprobs`](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py)), so a KL from it is **truncated**.
- If the API is used anyway, hit vLLM directly on both sides, not through
  LiteLLM or another gateway, so nothing rewrites `logprobs` /
  `top_logprobs`.

Phase 3 is the gate that decides whether 2b is worth opening. It is also
required again after 2b if 2b lands.

## 4 — whole layer, per layer type

One spec for a GDN layer and one for a QSA layer. #61325's Flash-Next count
is 36 GDN + 12 QSA, indexer on every fourth layer
([#61325](https://github.com/vllm-project/vllm/pull/61325)). Confirm the
AWQ checkpoint's `layer_types` before coding the split **(unverified)**.

GDN state is FP32 and is not stored in the QSA heap. The QSA heap holds KV,
compressed keys, and the indexer. That separation is the
[wiki contract](https://github.com/BlivionIaG/rdna-hip-wiki/blob/9b91d4116d24414ea2ffd9530d0caeed331ddcd0/kernels/persistent/monokernel-gfx1030-plan.md).
Re-tile any stage that would exceed 64 KiB LDS (K2 in #61325 is up to 68 KB).

The MTP drafter has no mono spec in this phase. It keeps running as vLLM
layers, including experts from the #82 quantizer.

K1 (hyper-connection, QKV, GDN or QSA front) is in this phase, not in 2a.
2a does not launch attention.
