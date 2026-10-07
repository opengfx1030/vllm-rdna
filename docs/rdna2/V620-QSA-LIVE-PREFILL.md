# QSA live-context prefill improvement

Bound AMD QSA prefill scoring to the live prompt context instead of the allocated context capacity. Round compressed columns to 64, retain enough for top-k, and carry the live bound through side metadata. Decode and missing metadata retain the existing capacity-wide path.

Clean branch from opengfx1030/rdna_extras at b33f9b66e. The QSA patches are derived from a2e47ef70/ad8cc519d and were recommitted as 9a91ff64b/de6f1c9da. The branch also folds in PR #12's remaining CPU PLE and MTP startup prerequisites. It excludes rejected graph-capture experiments, Wave-SplitK, local argmax, and INT8 shadows.

## Recorded results and limits

Historical combined deployment 82926d228: TP4, PP1, EP4, MTP2, FP16 dense, Intel AutoRound INT4 experts, original BF16 PLE in CPU RAM. No model or power changes. This combined deployment includes resident MoE and cache fixes beyond this focused PR; these are not new measurements of the clean cherry-pick.

| Suite | Context | Prompt tokens | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Regular | 16K | 16,750 | 2,011 | 63.51 |
| Coding | 16K | 18,063 | 2,036 | 68.53 |
| Regular | 32K | 33,455 | 2,024 | 55.17 |
| Regular | 64K | 66,913 | 1,974 | 59.37 |
| Coding | 64K | 71,971 | 1,986 | 70.63 |
| Regular | 128K | 133,816 | 1,859 | 53.86 |
| Coding | 128K | 143,855 | 1,832 | 72.49 |

Coding 32K was excluded for repetition, also observed in baseline. The combined stack improved valid 16K/32K prefill by 31–33% against the immediately collected 1,529–1,543 tok/s baseline. Because that deployment included changes outside this PR, the result does not isolate the QSA bound's contribution. No consistent end-to-end decode improvement was established. Later fused-draft testing retained 2,015–2,040 prefill but showed no matched-acceptance decode gain.

A later rejected QSA graph-capture experiment measured 1,369–1,389 prefill tok/s. Its launcher omitted environment values normally supplied by the default systemd unit, so it was not a controlled comparison of the graph flag alone. The cause of that slowdown remains unresolved. The Git service was rebenchmarked after restoring the deployed launcher values; it measured 1,977 at 32k, 1,927–1,929 at 64k, and 1,788–1,820 prefill tokens/s at 128k. Do not interpret the historical 2k numbers as a fresh verification of the current process.

## Validation

- Four QSA bound cases passed; selected-token sets match unbounded scoring (ordering can differ).
- Three metadata cases passed; the wider invocation recorded 16 skips and 27 deselections.
- Wider AMD suite: 11 passed, two unrelated CPU platform-dispatch failures also observed on unchanged baseline.
- Six deterministic full-model prompts matched baseline exactly.
- Ordinary follow-up reused 10,240 tokens at 1.187 s TTFT; tool follow-up reused 11,264 at 0.557 s.
- Combined startup: 373.879 s to ready; GPU worker loading approximately 193–204 s.
- Completed benchmark logs had no inference errors, overlaps or preemptions. No broad quality benchmark is claimed.

Historical focused tests, reproducible with an installed ROCm vLLM environment:

```bash
.venv/bin/python -m pytest tests/models/qwen4_exp/test_qsa_amd.py -k prefill_bounds_scoring_to_live_context -v
.venv/bin/python -m pytest tests/models/qwen4_exp/test_qsa_reference.py -k metadata_carries_live_context_bound_cpu -v
```

## Full launch and environment

Run on a Linux V620 server. These are the saved combined deployment paths and require its native resident-MoE build, model files and tuning tables. Set `V620_ROOT`, `V620_RUNTIME` and `V620_MODEL` to your own copies. This focused QSA patch alone is Python-only; the absolute benchmark rates also depend on the other deployment optimizations. The launch below explicitly includes values supplied by systemd, which must not be omitted when running manually.

Model path: `$V620_MODEL` (the measured artifact was an Intel AutoRound export). Published validation used original BF16 CPU PLE, not group16 INT4. Exact upstream model/sidecar revision identifiers have not been reverified for this publication; the local directory identifies the measured artifact.

PyTorch in the saved microbenchmark: 2.13.0+rocm10.0.0. Runtime vLLM version string: 0.26.1rc1.dev0+upstream.3e1a0e1aa; use the source commit above for provenance. TunableOp helper selects qualified rocBLAS c27e2252cc7a lookup tables, overriding initial disabled defaults while keeping live tuning off.

```bash
systemctl --user stop v620-tp4-fast.service
export VLLM_ROCM_MOE_PREFILL=0
export VLLM_RDNA_MOE_RESIDENT=1
export VLLM_RDNA_MOE_RESIDENT_SKINNY=1
export VLLM_RDNA_FUSED_SE=1
export VLLM_GDN_HIP_PREFILL=0
export VLLM_TRACE_PREFIX_CACHE=1
export VLLM_TUNED_CONFIG_FOLDER=${V620_ROOT:?set V620_ROOT}/tuned-moe
#!/usr/bin/env bash
set -euo pipefail
root=${V620_ROOT:?set V620_ROOT to the deployment checkout}
runtime=${V620_RUNTIME:?set V620_RUNTIME to the rocm10 venv root}
export PYTHONPATH=$root/source
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_PLE_CPU_OFFLOAD=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_RDNA_DENSE_INT8=0 VLLM_RDNA_DENSE_INT8_ONLY=0 VLLM_RDNA_DENSE_GEMV=0
export VLLM_RDNA_AR=1 VLLM_RDNA_AR_MAX_KB=64 VLLM_RDNA_AR_BLOCKS=0 VLLM_RDNA_AR_PACE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export VLLM_CAUSAL_CONV1D_RDNA2_FWD=0 VLLM_CAUSAL_CONV1D_RDNA2_UPDATE=0
export VLLM_ENABLE_STARTUP_PLAN=0 VLLM_ROCM_USE_AITER=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export VLLM_CACHE_ROOT=$root/cache/vllm TRITON_CACHE_DIR=$root/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$root/cache/inductor TORCH_EXTENSIONS_DIR=$root/cache/extensions
export PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
sdk=$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_core
export PATH="$runtime/.venv/bin:/opt/rocm/core-10.0/bin:/opt/rocm/core-10.0/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib/host-math/lib:/opt/rocm/core-10.0/lib"
source "$root/source/tools/rdna2/tunableop_env.sh"
configure_v620_tunableop "$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5" "$runtime/tunableop"
command=("$runtime/.venv/bin/python" -m vllm.entrypoints.openai.api_server
 --model ${V620_MODEL:?set V620_MODEL} --served-model-name active qwen3.8-flash-next
 --host 0.0.0.0 --port 8080 --tensor-parallel-size 4 --pipeline-parallel-size 1 --enable-expert-parallel --enable-ep-weight-filter
 --dtype float16 --max-model-len 262144 --block-size 1024 --max-num-seqs 4
 --max-num-batched-tokens 4096 --kv-cache-memory-bytes 4294967296
 --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,12]}'
 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
 --default-chat-template-kwargs '{"enable_thinking":false}'
 --limit-mm-per-prompt '{"image":255,"video":32}' --mm-processor-kwargs '{"max_pixels":602112}'
 --enable-prefix-caching --mamba-cache-mode align --kernel-config '{"moe_backend":"triton"}')
if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
if pgrep -u "$(id -u)" -f 'vllm.entrypoints|VLLM::EngineCore|VLLM::Worker' >/dev/null; then
 echo 'Existing vLLM process present; refusing overlap.' >&2; exit 2
fi
"$runtime/.venv/bin/python" "$root/source/tools/rdna2/check_v620_tuning.py" --rows-template "$PYTORCH_TUNABLEOP_FILENAME" "${command[@]:1}"
"$runtime/.venv/bin/python" -c 'import vllm; assert "/decode-qsa-20260920/" in vllm.__file__,vllm.__file__'
cd "$root/source"
exec "${command[@]}" "$@"
```

The reproduced deployment uses a 3.75 GiB KV allocation and a 602,112-pixel
vision cap. Its initial 128k coding timeout was caused by old Mamba state
blocks not being retired across null gaps, not by the cache allocation.
Backport commit `b619cf991` fixes the retirement cursor while preserving the
served checkpoint/replay implementation. The clean 143,855-token coding case
then completed at 1,813.9 prefill tokens/s and 78.2 generation tokens/s. Do not
use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with
CPU PLE offload: its ROCm CUDA IPC registration can fail with
`pidfd_getfd: Operation not permitted`.

## Benchmark reproduction

Saved harness: llm-context-bench 92286b24065565f4929e78c45f776029480e9939, concurrency one, streaming chat, 1,024 outputs, unique cache salts, 11% input tolerance for tokenizer drift. The historical cold-cache proxy on 8082 must be running and forwarding to 8080 for this exact command.

```bash
export PYTHONPATH=${V620_RUNTIME:?set V620_RUNTIME}/context-bench-92286b2/src
"${V620_RUNTIME}"/.venv/bin/python -m llm_context_bench \
 --base-url http://127.0.0.1:8082 --model active --engine vllm \
 --profile qsa-live-prefill --suite all --lane performance --sizes 16k 32k 64k 128k \
 --input-size-tolerance-percent 11 --chat-template-kwargs '{"enable_thinking":false}' \
 --timeout 240 --output qsa-live-prefill-results.json \
 --command "/bin/bash ${V620_ROOT:?}/serve-tp4-checkpoints.sh" \
 --system '4x V620; TP4 EP4 MTP2; Intel INT4 experts; FP16 dense; original CPU BF16 PLE'
```

## Review and scope

Open target PRs checked before publication: PR #12 concerns CPU PLE/MTP startup and does not contain this QSA bound. Upstream vLLM PR #56500 reuses a bounded NVIDIA QSA logits workspace but preserves its existing scoring width; this PR independently reduces the AMD scoring width to the live prefill context. Base is the fork's rdna_extras, not mainline vLLM.

AI assistance was used. Historical and September 23 Git-service tests are
recorded above. The live service runs the Git branch documented here.
