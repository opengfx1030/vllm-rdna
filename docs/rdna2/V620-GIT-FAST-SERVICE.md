# Reproduce the V620 TP4 fast service from Git

This branch starts from upstream `rdna_extras` at
`f3dd65fa70636566c11b249209d281c0b63c819e`. The 16k baseline and every
port stage are recorded in [V620-BASELINE-PORT-20260922.md](V620-BASELINE-PORT-20260922.md).
The served settings are in `tools/rdna2/serve_v620_baseline.sh` and
`tools/rdna2/systemd/v620-tp4-git.service`. The tuning table from
the earlier service is versioned at `tuned-moe/` with identical JSON content.

## Required machine assets

- Four Radeon Pro V620 cards with ROCm 10.0 and the existing runtime at
  `<v620-home>/v620-experiments/upstream-20260915/v620-vllm-testing/.venv`.
- The Intel AutoRound W4A16 model at
  `<v620-home>/v620-vllm/models/intel-autoround`. It retains FP16 dense
  layers and CPU-offloaded BF16 PLE weights. The relevant local model-file
  SHA-256 values are:

  | File | SHA-256 |
  | --- | --- |
  | `config.json` | `0fabfa21fab8bfe69f02234f7ae8df4ed91b785fca2e22011c1230f6a07e5329` |
  | `quantization_config.json` | `2371a9d0bed1ff619622ac57aa0e6e5b39688bb834b48d761e90d5834f4cab4b` |
  | `model.safetensors.index.json` | `35e503b758cf5441886042d315c48dfba70f947611816cbb90e02d4d5c5e58ff` |
  | `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |

The model shards and Python runtime remain machine assets; this Git branch
does not contain them. Preserve those exact assets when comparing throughput.

## Build from the Git checkout

The server gets source only through Git. Run these commands on the V620 host:

```bash
git clone --branch codex/v620-baseline-port-20260922 \
  https://github.com/GeorgeMA-Strong/vllm-rdna.git \
  <v620-home>/v620-experiments/baseline-git-20260922

root=<v620-home>/v620-experiments/baseline-git-20260922
runtime=<v620-home>/v620-experiments/upstream-20260915/v620-vllm-testing
export PATH="$runtime/.venv/bin:/opt/rocm/core-10.0/bin:/opt/rocm/core-10.0/llvm/bin:$PATH"
sdk="$runtime/.venv/lib/python3.12/site-packages/_rocm_sdk_core"
export LD_LIBRARY_PATH="$sdk/lib:$sdk/lib/host-math/lib:/opt/rocm/core-10.0/lib"

cmake -S "$root" -B "$root/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DGPU_TARGETS=gfx1030 \
  -DVLLM_PYTHON_EXECUTABLE="$runtime/.venv/bin/python" \
  -DTorch_DIR="$runtime/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch"
cmake --build "$root/build" --parallel 4
for artifact in "$root"/build/*.so; do
  ln -sfn "$artifact" "$root/vllm/$(basename "$artifact")"
done
ln -sfn "$root/build/_deps/triton_kernels-src/python/triton_kernels/triton_kernels" \
  "$root/vllm/third_party/triton_kernels"
```

The symlinks expose modules built from this Git revision; they do not transfer
source between machines. Verify `git -C "$root" rev-parse HEAD` before serving.

## Serve

The Git-tracked unit uses TP4, EP4, FP16, MTP2, 1,024-token KV blocks, a
4,096-token batch limit, resident INT4 experts, FP16 shared-expert fusion,
the versioned MoE table, CPU PLE, checkpoint replay, and the opt-in resident
skinny decode path. It listens on port 8080. It conflicts with the old fast
unit to prevent two full models using the same GPUs.

```bash
systemctl --user disable --now v620-tp4-fast.service
systemctl --user link "$root/tools/rdna2/systemd/v620-tp4-git.service"
systemctl --user daemon-reload
systemctl --user enable --now v620-tp4-git.service
curl --fail http://127.0.0.1:8080/health
```

The Git service was installed on September 23, 2026. The old fast unit is
disabled. Commit `945bba272` reached `/health` in 378 seconds, from
09:56:38 to 10:02:56 UTC. It allocated the historical 4 GiB KV cache:
284,229 tokens and 1.08x maximum concurrency at the 262,144-token limit.

## 16k measurement

Use `llm-context-bench` checkout `92286b2` with harness SHA-256
`ec553141cd5f63d962ec1bbe6fbaf65effc44d032e5d055d4e6376824138e5ae`.
Run the 16k performance lane for both regular and coding fixtures, disabled
thinking, 11% input-size tolerance, and 240-second timeout. The fixture tags
are deterministic across separate harness invocations, and this vLLM stream
does not report cached-token counts. After a warm pass, repetitions numbered
`01` may reuse prefix cache; exclude them when TTFT collapses. Compare fresh
repetitions `02` and `03`, and check the service log for Triton JIT in their
measurement window. The exact run artifacts and per-stage values are in the
baseline document.
