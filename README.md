<!-- markdownlint-disable MD001 MD041 -->

# vLLM on AMD RDNA2 (gfx1030)

💬 [Discord](https://discord.gg/n4VYYE9vAZ) · [gfx1030 wiki](https://blivioniag.github.io/gfx1030-wiki/) · [Recipes](docs/rdna/serving.md) · [Tuning](docs/rdna2/tuning.md)

This is [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna). The default branch, **`rdna_extras`**, is upstream vLLM **v0.34.0** plus the RDNA port. Serve from this checkout with [`tools/rdna/serve_rdna.sh`](tools/rdna/serve_rdna.sh) and a recipe. The logo section further down is upstream's README.

## Hardware

The recipes, TunableOp rows, and HIP kernels they turn on are for **gfx1030** (RDNA2 / Navi 21). The card those notes measure is the **Radeon PRO V620** (32 GB, 72 CUs; see [`docs/rdna/monokernel/README.md`](docs/rdna/monokernel/README.md)).

gfx1030 has no BF16 dot (`kHasDot2Bf16` is false for RDNA2 in [`csrc/rocm/rdna/common/arch.cuh`](csrc/rocm/rdna/common/arch.cuh)). The launcher always passes `--dtype float16`.

**gfx1100** (RDNA3) has a separate W4A16 dense kernel and MoE kernel, selected only when `on_gfx1100()` is true ([`rdna3_w4a16.py`](vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py), [`rdna3_moe.py`](vllm/model_executor/layers/fused_moe/experts/rdna3_moe.py)). The recipes and the `rocm7.14-rocblas5.5` TunableOp rows are gfx1030. [`docs/rdna2/tuning.md`](docs/rdna2/tuning.md) says the gfx10 HIP paths fall back on gfx11 and that nothing there is validated on RDNA3.

## What this fork adds

On top of upstream v0.34.0, for gfx1030:

- **FA-RDNA2 attention** (`--attention-backend RDNA_ATTN`), including GQA decode. An opt-in packed KV dtype, `--kv-cache-dtype int4_fp16_block`, is read by that backend ([`docs/explore/kv-int4-fp16-scales/README.md`](docs/explore/kv-int4-fp16-scales/README.md)). No recipe turns it on.
- **W4A16** HIP GEMMs for AWQ / compressed-tensors, and an opt-in **W4A8** sdot4 prefill path. `27b-awq` sets `W4A8=1`. Decode stays W4A16 ([tuning guide](docs/rdna2/tuning.md)).
- **EXL3** trellis GEMMs and the exllamav3 n-gram table, for Flash-Next (`flashnext-exl3`) and Qwen3.8-27B (`27b-exl3`).
- **HIP GDN** prefill and decode for the hybrid Qwen models, and **HIP QSA** sparse attention for Flash-Next.
- **MTP** and **DFlash2** speculative decoding. `--monokernel` is the Flash-Next MonoKernel (K1/K2, fused GDN, QSA decode). `flashnext-dflash2` is the recipe that passes it.
- **PLE** n-gram table for Flash-Next: `bf16`, `int4`, `int4-disk`, or `exl3` (the table that ships beside an EXL3 checkpoint).
- **RDNA one-shot all-reduce** for small tensor-parallel messages over PCIe (`VLLM_RDNA_AR=1`, 64 KiB; 256 KiB in `flashnext-mtp2`), with RCCL above that.
- **DeepSeek-V4-Flash** (fp8 dense, MXFP4 experts, fp8 KV) via `dsv4-flash`.
- **GGUF** Flash-Next (Unsloth `UD-Q4_K_XL`, `--load-format qwen4exp_gguf`) via `flashnext-gguf-mtp0`: PPL 4.80, 16k TTFT 8.2 s, text only, no MTP. It needs the plugin fork [`opengfx1030/vllm-gguf-plugin`](https://github.com/opengfx1030/vllm-gguf-plugin) (the container image has it); see [`docs/rdna2/gguf-flashnext-2026-11-16.md`](docs/rdna2/gguf-flashnext-2026-11-16.md).
- **Expert load balancing** for the expert-parallel MoE (W4A16 and W4A8), opt-in, no speedup measured and still owed a soak before it is called qualified (two server runs showed no fault and no added stall): `VLLM_RDNA_EPLB=1` swaps experts between ranks while serving, in small chunks at step boundaries ([`docs/rdna2/eplb-online-2026-11-17.md`](docs/rdna2/eplb-online-2026-11-17.md)); `VLLM_RDNA_EPLB_STATS_DIR` only records the loads. Both off by default and byte-identical when off. Upstream `--enable-eplb` does not start on our quantized MoE ([study](docs/rdna2/eplb-lean-design-2026-11-16.md), knobs in the [tuning guide](docs/rdna2/tuning.md)).
- **Pipeline parallel** for three gfx1030 cards (`flashnext-pp3`, `flashnext-pp3-mtp2`) and for two stages of four cards (`flashnext-tp4pp2-mtp2`). Tensor parallel in the other Flash-Next and 27B recipes is 4, 2 for `full` and the HI2 recipes, 8 for `flashnext-tp8-mtp2`, and 1, 2 or 4 for the Inkling-2-Mini recipes.
- **HI2** (`rdna_hint_flex`, HadamardINT2-FP16 3.0 bpw) Qwen3.8-27B on two cards (`27b-hi2-tp2`, `27b-hi2-tp2-mtp2`): [serving guide](docs/rdna2/hi2-27b-tp2-serving.md).
- **LeapQuant** INT8 GDN recurrent state, opt-in and lossy, for Flash-Next (`flashnext-leapquant-mtp2`): [`docs/rdna2/leapquant-gdn-2026-11-13.md`](docs/rdna2/leapquant-gdn-2026-11-13.md).
- **Inkling-2-Mini** (`thinkingmachines/Inkling-2-Mini`) on native HIP kernels at TP=1, 2 and 4, up to the model's 1,048,576-token context, with an optional DSpark draft (`inkling2-mini-*` recipes): [serving guide](docs/rdna2/inkling2-serving.md).
- **TunableOp** rocBLAS rows for the ROCm 7.14 build, under [`tunableop/rocm7.14-rocblas5.5/`](tunableop/rocm7.14-rocblas5.5/).

Knobs and the measurements already written down live in [`docs/rdna2/tuning.md`](docs/rdna2/tuning.md).

## Quick start

Four gfx1030 GPUs, a ROCm 7.14 SDK, and a local EXL3 Flash-Next checkpoint. The worked example is recipe `flashnext-exl3`.

### Prerequisites

The launcher looks for the ROCm **7.14** SDK at `/opt/rocm/core-7.14` (`ROCM_SDK_ROOT`). Your user needs the `video` and `render` groups, and `/dev/kfd` plus `/dev/dri` need to be usable. Multi-GPU hosts in the [wiki's ROCm page](https://blivioniag.github.io/gfx1030-wiki/setup/installing-rocm.html) pin 7.2.0 or 7.14.0.

This repository builds its own application images with [`docker/Dockerfile.rdna`](docker/Dockerfile.rdna) and [the image workflow](.github/workflows/docker-image.yml). Application images publish to **`ghcr.io/opengfx1030/vllm-rdna`**; the ROCm/PyTorch base remains **`docker.io/blivioniag/rocm-rdna:7.14.0`**, pinned by digest in the Dockerfile. The separate `vllm-rdna-docker` repository is not involved in these builds. For the host-venv recipe below, the [host venv](https://blivioniag.github.io/gfx1030-wiki/vllm/host-venv.html) page is the longer walkthrough.

### Build

```bash
uv venv --python 3.12 venv-7.14.0
source venv-7.14.0/bin/activate

uv pip install --pre --force-reinstall \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch==2.12.0+rocm7.14.0" \
  "torchvision==0.27.0+rocm7.14.0" \
  "torchaudio==2.11.0+rocm7.14.0" \
  "triton==3.7.1"

uv pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "rocm-sdk-device-gfx1030==7.14.0" \
  "amd-torch-device-gfx1030==2.12.0+rocm7.14.0"

git clone -b rdna_extras https://github.com/opengfx1030/vllm-rdna.git
cd vllm-rdna

# --no-build-isolation means setup.py runs in this venv. It imports
# setuptools_rust and setuptools_scm. Install those without touching torch,
# so the ROCm 7.14 pins above stay.
uv pip install \
  "cmake>=3.26.1" ninja packaging \
  "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" "setuptools-rust>=1.9.0" \
  wheel jinja2

# Leave --torch-backend off: the PyTorch rocm7.14 index has no matching
# torchaudio wheel for CPython 3.12 on x86_64.
VLLM_USE_PRECOMPILED=0 ENABLE_CK=0 \
  ROCM_HOME="${ROCM_HOME:-/opt/rocm/core-7.14}" \
  PYTORCH_ROCM_ARCH=gfx1030 \
  uv pip install -e . --no-build-isolation
```

`python -c "import torch, vllm, vllm._rocm_C; print(torch.__version__, torch.cuda.get_device_name(0))"` should print a `+rocm7.14` torch and a gfx1030 device.

### Checkpoint

`flashnext-exl3` serves an EXL3 Qwen Flash-Next checkpoint produced by **exllamav3 1.4.5 or newer**. The recipe names `turboderp/Qwen3.8-Flash-Next-exl3` and uses a local directory such as `Qwen3.8-Flash-Next-exl3-3.05bpw`. That directory must contain `ngram_embedding.safetensors`. The launcher sets `PLE=exl3`: a CPU sidecar memory-maps the table (the recipe comment: 32.6 GB at ng5, page cache). The fork decodes that file itself.

The launcher also sets `HF_HUB_OFFLINE=1`. Download the checkpoint before you serve. The same offline default applies to every recipe.

### Serve

```bash
MODEL=/path/to/Qwen3.8-Flash-Next-exl3-3.05bpw \
VENV=/path/to/venv-7.14.0 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-exl3
```

That is TP=4 on `HIP_VISIBLE_DEVICES=0,1,2,3`, served name `flash-next-exl3`, listening on `0.0.0.0:18094`. Override a recipe key on the same command (`PORT=18100`, `HIP_VISIBLE_DEVICES=4,5,6,7`). See the assembled command first:

```bash
MODEL=/path/to/Qwen3.8-Flash-Next-exl3-3.05bpw \
VENV=/path/to/venv-7.14.0 PRINT=1 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-exl3
```

The first launch compiles graphs into `<repo>/cache/`; the vLLM cache root there is per GPU set and port (`cache/vllm-<gpu ids>-<port>`), so two servers of one checkout on different cards do not stop each other. The process stays in the foreground.

### It is up when HTTP answers

```bash
curl -sf http://127.0.0.1:18094/health
curl -sS http://127.0.0.1:18094/v1/models
```

`/health` returns HTTP 200. `/v1/models` lists `"id": "flash-next-exl3"`.

```bash
curl -sS http://127.0.0.1:18094/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"flash-next-exl3","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18094/v1", api_key="not-used")
print(client.chat.completions.create(
    model="flash-next-exl3",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=64,
).choices[0].message.content)
```

Clients and gateways (OpenAI SDKs, OpenCode, LiteLLM, and the rest) get a base URL and the served model name. ROCm variables, `VLLM_*`, and serve flags stay in this launch environment. They are not client settings.

Other Flash-Next layouts (AWQ, MTP, three cards, DFlash2) are recipes in the next section. AWQ checkpoints use `PLE=bf16` unless you pass `PLE=int4` or `PLE=int4-disk` and `PLE_QUANT_DIR` pointing at a `ples_int4` directory from `primitive-ai/Qwen3.8-Flash-Next-PLE-quant`.

## Container images

The source-owned image workflow builds the checked-out commit, never a moving
branch clone. It publishes Linux x86-64 images to GHCR:

| Source event | Tags on `ghcr.io/opengfx1030/vllm-rdna` |
| --- | --- |
| Push to `rdna_extras` | `nightly-<vllm version>-rocm7.14.0`, e.g. `nightly-0.34.0-rocm7.14.0`, plus the floating `nightly` and `nightly-rocm7.14.0`; every newer build moves all three |
| Create a version tag, e.g. `v0.34.0` (Git push, GitHub UI or API) | `v0.34.0` and `v0.34.0-rocm7.14.0`, without changing nightly |
| Manual run | Same behavior for the selected `rdna_extras` branch or version tag |

The nightly's vLLM version is the nearest `v<major>.<minor>.<patch>` tag in the
history of the built commit (`git describe --tags --abbrev=0 --match 'v[0-9]*'`),
so tagging `v0.35.0` switches later builds to `nightly-0.35.0-rocm7.14.0`.
Branch builds are pushed by digest with no per-commit tag; the promote job then
moves the three nightly tags onto the newest digest. "Nightly" is a rolling channel
built on each push, not a daily cron; a multi-commit push builds its final
commit. A newer push cancels an in-flight branch build (version tag builds are
never cancelled), and a build whose commit is no longer the branch tip is not
promoted. Ref deletions do not publish images.

The previous nightly digest becomes untagged when the tags move. The daily
[package cleanup workflow](.github/workflows/package-cleanup.yml) deletes, once
older than seven days, every version that is untagged or carries only legacy
`sha-*` tags (`DISPOSABLE_TAGS`, default `sha-*`), together with its child
manifests. A version that also has `nightly`, `nightly-*` or a release tag is
never deleted, and neither is anything a retained image references (the
platform manifest and buildx provenance attestation of a tagged image are
untagged versions of this package). Manual runs are a dry run by default. Old
`nightly-<version>-rocm…` tags of an earlier vLLM version stay until deleted by
hand.

The image also carries the GGUF plugin fork (`opengfx1030/vllm-gguf-plugin`),
built for gfx1030 from the commit pinned in `docker/Dockerfile.rdna`
(`GGUF_PLUGIN_REF`), so `flashnext-gguf-mtp0` works inside it. Bump that ref to
take a plugin change.

Version builds use GitHub's `create` event, not a second tag-push trigger, so
creating a tag starts only one build. Branch creation is ignored. Create no
more than three tags at once; GitHub does not emit creation events above that
limit. Updating an existing tag requires a manual run to rebuild its image.

Publish a version from a source commit that contains the workflow:

```bash
git tag v0.33.1
git push origin v0.33.1
```

Lightweight and annotated tags are supported. Use Docker-compatible names of
at most 117 characters, such as `v0.34.0` or `v0.34.0-rc.1`; `/` and `+` are
not valid. The tag is the image name: tag `v0.34.0` builds `:v0.34.0`.
`nightly`, `nightly-*`, `sha-*` and `latest` are reserved.

### Registry and runner setup

- Publishing uses the repository's `GITHUB_TOKEN` with `packages: write`;
  no Docker Hub token, GitHub PAT or cross-repository dispatch is required.
- After the first publication, set the GHCR package's visibility to **public**
  for anonymous downloads. Existing packages must grant this repository
  Actions write access.
- Native builds use a dedicated Linux x86-64 runner with label
  **`rdna-image-builder`** by default. Promotion alone uses `ubuntu-latest`.
- The existing base is about 39 GiB unpacked, already larger than a standard
  GitHub-hosted runner's advertised 14 GB disk. Use a Linux x86-64 Docker host
  with enough space for the base, native build outputs and build cache.
- Optional repository variables: `VLLM_RDNA_RUNNER` overrides the build runner
  label; `VLLM_RDNA_MAX_JOBS` overrides native/Rust build parallelism (default 2).
  This repository sets `VLLM_RDNA_MAX_JOBS=16`. The image retains the base's
  seven RDNA architecture targets.
- Treat the build runner as privileged: its Docker access is root-equivalent.
  This workflow deliberately has no pull-request trigger. Do not route
  untrusted pull-request jobs to the same runner.

Use a generic runner display name and an isolated service hostname:
GitHub prints runner metadata before workflow steps can mask it. For a
systemd-managed runner, add this service override (requires support for
`ProtectHostname=yes:<hostname>`):

```ini
[Service]
ProtectHostname=yes:rdna-image-builder
ExecStartPre=/usr/bin/python3 -c "import socket, sys; sys.exit(0 if socket.gethostname() == 'rdna-image-builder' else 'Runner hostname isolation required')"
```

The first workflow step masks the Docker daemon's hostname and local interface
addresses before Docker actions print diagnostics. Do not publish private
hostnames, addresses or deployment paths in workflow files or PR descriptions.

### Local build and run

From a clean checkout, resolve the same native SCM version used by CI and
build against the existing Docker Hub base:

```bash
VLLM_VERSION=$(uv run --no-project --with setuptools-scm==9.2.2 python -c \
  'from setuptools_scm import get_version; print(get_version(git_describe_command=["git", "describe", "--dirty", "--tags", "--long", "--abbrev=40", "--match", "v[0-9]*"]))')
docker buildx build --platform linux/amd64 --load \
  -f docker/Dockerfile.rdna \
  --build-arg VLLM_VERSION="$VLLM_VERSION" \
  --build-arg VLLM_COMMIT="$(git rev-parse HEAD)" \
  --build-arg MAX_JOBS=2 -t vllm-rdna:local .

docker run --rm --device /dev/kfd --device /dev/dri vllm-rdna:local --help
docker run --rm --network host --ipc host --shm-size 16g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/opengfx1030/vllm-rdna:nightly \
  serve "$MODEL" --dtype float16 --host 0.0.0.0 --port 8000
```

The application remains editable at `/src/vllm`; recipes, TunableOp tables
and tuned MoE configurations are included. Git metadata, local environments,
model downloads and generated native binaries are excluded from the context.
Python's installed version comes from the checkout's native SCM metadata;
OCI labels record the exact source commit. The base torch stack is constrained
to its installed versions while application runtime dependencies are resolved.
The GDN prefill helpers use the non-compact FP16 DOT2 encoding on
gfx1200/gfx1201; gfx1030 retains its original instruction.
CLI startup on this source version queries GPU capability, including `--help`;
pass the GPU device flags shown above.
The pinned base's Triton 3.7.1 cannot load the upstream `triton_kernels`
MXFP4 backend (`is_hip_gfx1250` is missing). This existing limitation does not
affect the native RDNA2 backend; see the
[port notes](docs/rdna2/v0.31.0-port-notes.md).

## Which recipe

| You want | Recipe | Cards |
| --- | --- | --- |
| Qwen Flash-Next, EXL3, tool and reasoning parsers | `flashnext-exl3` | 4, TP=4 |
| Qwen Flash-Next, AWQ-W4A16, same parsers, no MTP | `flashnext` | 4, TP=4 |
| Flash-Next AWQ, plain decode, 262K context | `flashnext-mtp0` | 4, TP=4 |
| Flash-Next AWQ, MTP=2 | `flashnext-mtp2` | 4, TP=4 |
| Flash-Next UD-Q4_K_XL GGUF, plain decode, text only | `flashnext-gguf-mtp0` | 4, TP=4 |
| Flash-Next AWQ plus an Inco DFlash2 drafter | `flashnext-dflash2` | 4, TP=4 |
| Flash-Next AWQ, lossy LeapQuant GDN state, MTP=2 + MonoKernel | `flashnext-leapquant-mtp2` | 4, TP=4 |
| Flash-Next AWQ on three cards | `flashnext-pp3` | 3, PP=3 |
| Three cards, MTP=2, about four streams | `flashnext-pp3-mtp2` | 3, PP=3 |
| Flash-Next AWQ, MTP=2, tensor parallel over eight cards on two PCIe switches | `flashnext-tp8-mtp2` | 8, TP=8 |
| Flash-Next AWQ, MTP=2, two pipeline stages of four cards | `flashnext-tp4pp2-mtp2` | 8, TP=4 x PP=2 |
| Qwen3.8-27B AWQ, W4A8 prefill | `27b-awq` | 4, TP=4 |
| Qwen3.8-27B EXL3 (`mul1`) | `27b-exl3` | 4, TP=4 |
| Qwen3.8-27B HI2 3.0 bpw (`rdna_hint_flex`) | `27b-hi2-tp2` | 2, TP=2 |
| Qwen3.8-27B HI2 3.0 bpw, MTP=2, one to four concurrent requests | `27b-hi2-tp2-mtp2` | 2, TP=2 |
| DeepSeek-V4-Flash (K128, fp8 KV) | `dsv4-flash` | 4, TP=4 |
| HIP graphs, breakable capture, TP=2 | `full` | 2, TP=2 |
| Inkling-2-Mini, 180k context | `inkling2-mini-tp1` | 1, TP=1 |
| Inkling-2-Mini, DSpark draft | `inkling2-mini-tp1-dspark` | 1, TP=1 |
| Inkling-2-Mini, DSpark with adaptive K and a draft vocabulary | `inkling2-mini-tp1-dspark-adaptive` | 1, TP=1 |
| Inkling-2-Mini, full 1,048,576-token context | `inkling2-mini-tp2` | 2, TP=2 |
| Inkling-2-Mini, DSpark draft, 780k context | `inkling2-mini-tp2-dspark` | 2, TP=2 |
| Inkling-2-Mini, full context, 4,096-token prefill steps | `inkling2-mini-tp4` | 4, TP=4 |

Every knob those files set is in [`docs/rdna/serving.md`](docs/rdna/serving.md); the notes behind them are indexed in [`docs/rdna2/README.md`](docs/rdna2/README.md). Dry-run any of them with `PRINT=1`.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, Iluvatar CoreX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

> **RDNA / gfx1030:** start with the [quick start](#quick-start) at the top of this README and the [gfx1030 wiki](https://blivioniag.github.io/gfx1030-wiki/). `uv pip install vllm` installs upstream, which does not carry this port.

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
vllm download-kernels  # precompiled kernels for faster startup
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
