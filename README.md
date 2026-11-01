<!-- markdownlint-disable MD001 MD041 -->

# vLLM on AMD RDNA2 (gfx1030)

💬 [Discord](https://discord.gg/n4VYYE9vAZ) · [gfx1030 wiki](https://blivioniag.github.io/gfx1030-wiki/) · [Recipes](docs/rdna/serving.md) · [Tuning](docs/rdna2/tuning.md)

This is [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna). The default branch, **`rdna_extras`**, is upstream vLLM **v0.31.0** plus the RDNA port. Serve from this checkout with [`tools/rdna/serve_rdna.sh`](tools/rdna/serve_rdna.sh) and a recipe. The logo section further down is upstream's README.

## Hardware

The recipes, TunableOp rows, and HIP kernels they turn on are for **gfx1030** (RDNA2 / Navi 21). The card those notes measure is the **Radeon PRO V620** (32 GB, 72 CUs; see [`docs/rdna/monokernel/README.md`](docs/rdna/monokernel/README.md)).

gfx1030 has no BF16 dot (`kHasDot2Bf16` is false for RDNA2 in [`csrc/rocm/rdna/common/arch.cuh`](csrc/rocm/rdna/common/arch.cuh)). The launcher always passes `--dtype float16`.

**gfx1100** (RDNA3) has a separate W4A16 dense kernel and MoE kernel, selected only when `on_gfx1100()` is true ([`rdna3_w4a16.py`](vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py), [`rdna3_moe.py`](vllm/model_executor/layers/fused_moe/experts/rdna3_moe.py)). The recipes and the `rocm7.14-rocblas5.5` TunableOp rows are gfx1030. [`docs/rdna2/tuning.md`](docs/rdna2/tuning.md) says the gfx10 HIP paths fall back on gfx11 and that nothing there is validated on RDNA3.

## What this fork adds

On top of upstream v0.31.0, for gfx1030:

- **FA-RDNA2 attention** (`--attention-backend RDNA_ATTN`), including GQA decode. An opt-in packed KV dtype, `--kv-cache-dtype int4_fp16_block`, is read by that backend ([`docs/explore/kv-int4-fp16-scales/README.md`](docs/explore/kv-int4-fp16-scales/README.md)). No recipe turns it on.
- **W4A16** HIP GEMMs for AWQ / compressed-tensors, and an opt-in **W4A8** sdot4 prefill path. `27b-awq` sets `W4A8=1`. Decode stays W4A16 ([tuning guide](docs/rdna2/tuning.md)).
- **EXL3** trellis GEMMs and the exllamav3 n-gram table, for Flash-Next (`flashnext-exl3`) and Qwen3.8-27B (`27b-exl3`).
- **HIP GDN** prefill and decode for the hybrid Qwen models, and **HIP QSA** sparse attention for Flash-Next.
- **MTP** and **DFlash2** speculative decoding. `--monokernel` is the Flash-Next MonoKernel (K1/K2, fused GDN, QSA decode). `flashnext-dflash2` is the recipe that passes it.
- **PLE** n-gram table for Flash-Next: `bf16`, `int4`, `int4-disk`, or `exl3` (the table that ships beside an EXL3 checkpoint).
- **RDNA one-shot all-reduce** for small tensor-parallel messages over PCIe (`VLLM_RDNA_AR=1`, 64 KiB), with RCCL above that.
- **DeepSeek-V4-Flash** (fp8 dense, MXFP4 experts, fp8 KV) via `dsv4-flash`.
- **Pipeline parallel** for three gfx1030 cards (`flashnext-pp3`, `flashnext-pp3-mtp2`). Tensor parallel in the other recipes is 4, or 2 for `full`.
- **TunableOp** rocBLAS rows for the ROCm 7.14 build, under [`tunableop/rocm7.14-rocblas5.5/`](tunableop/rocm7.14-rocblas5.5/).

Knobs and the measurements already written down live in [`docs/rdna2/tuning.md`](docs/rdna2/tuning.md).

## Quick start

Four gfx1030 GPUs, a ROCm 7.14 SDK, and a local EXL3 Flash-Next checkpoint. The worked example is recipe `flashnext-exl3`.

### Prerequisites

The launcher looks for the ROCm **7.14** SDK at `/opt/rocm/core-7.14` (`ROCM_SDK_ROOT`). Your user needs the `video` and `render` groups, and `/dev/kfd` plus `/dev/dri` need to be usable. Multi-GPU hosts in the [wiki's ROCm page](https://blivioniag.github.io/gfx1030-wiki/setup/installing-rocm.html) pin 7.2.0 or 7.14.0.

Published images lag this branch. [`opengfx1030/vllm-rdna-docker`](https://github.com/opengfx1030/vllm-rdna-docker) builds `docker.io/blivioniag/vllm-rdna:v0.31.0-extras` (also tagged `v0.31.0-extras-rocm7.14.0`) from `rdna_extras` commit `fbe84e941`. Its bake target `base-rocm714` is PyTorch `2.12.0+rocm7.14.0`. The Dockerfiles under [`docker/`](docker/) are upstream's. For the recipe below, use a host venv on this checkout. The [host venv](https://blivioniag.github.io/gfx1030-wiki/vllm/host-venv.html) page is the longer walkthrough; the pins match that bake.

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

The first launch compiles graphs into `<repo>/cache/`. The process stays in the foreground.

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

## Which recipe

| You want | Recipe | Cards |
| --- | --- | --- |
| Qwen Flash-Next, EXL3, tool and reasoning parsers | `flashnext-exl3` | 4, TP=4 |
| Qwen Flash-Next, AWQ-W4A16, same parsers, no MTP | `flashnext` | 4, TP=4 |
| Flash-Next AWQ, plain decode, 262K context | `flashnext-mtp0` | 4, TP=4 |
| Flash-Next AWQ, MTP=2 | `flashnext-mtp2` | 4, TP=4 |
| Flash-Next AWQ plus an Inco DFlash2 drafter | `flashnext-dflash2` | 4, TP=4 |
| Flash-Next AWQ on three cards | `flashnext-pp3` | 3, PP=3 |
| Three cards, MTP=2, about four streams | `flashnext-pp3-mtp2` | 3, PP=3 |
| Qwen3.8-27B AWQ, W4A8 prefill | `27b-awq` | 4, TP=4 |
| Qwen3.8-27B EXL3 (`mul1`) | `27b-exl3` | 4, TP=4 |
| DeepSeek-V4-Flash (K128, fp8 KV) | `dsv4-flash` | 4, TP=4 |
| HIP graphs, breakable capture, TP=2 | `full` | 2, TP=2 |

Every knob those files set is in [`docs/rdna/serving.md`](docs/rdna/serving.md). Dry-run any of them with `PRINT=1`.

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
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

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
