# RDNA Fork Rules

> Domain guide for the `opengfx1030` vLLM fork (RDNA2/RDNA3 on ROCm). Linked
> from `AGENTS.md` section 0; read it before touching kernels, the platform
> layer, or a serve recipe.

## Common mistakes

- Building Hub/Dockerfile images with `VLLM_REF=main` and expecting Flash-Next:
  use a host checkout and the launcher.
- Pasting old env dumps that set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` on Flash-Next.
  Only the recipes that need it (`dsv4-flash`, `full`) set it.
- Using BF16 on RDNA2: gfx1030 has no BF16 dot instructions, serve with
  `--dtype float16`.
- Enabling AITER or CK backends: they are CDNA-only. Keep `VLLM_ROCM_USE_AITER=0`.

## RDNA code rules

- **Never break other platforms.** Gate RDNA changes with the helpers in
  `vllm/platforms/rocm.py` (`on_gfx10x()`, `on_gfx1x()`, `on_rdna()`), never
  globally and never with an exact `"gfx1030" in arch` string match. CUDA, CDNA
  (AITER), GCN and CPU paths must keep working.
- **FP16 only on gfx1030.** RDNA2 has no BF16 dot instructions; BF16 kernels
  either fail to compile or run emulated. Keep BF16 paths for other archs.
- **Wave32.** Size Triton/HIP kernels for 32-lane waves: `num_warps` <= 4 and
  RDNA-sized tiles, not CDNA (wave64, 128x128) configs.
- **Launch on the current stream.** HIP kernels capture
  `at::cuda::getCurrentCUDAStream()` and launch with `<<<grid, block, 0, stream>>>`,
  followed by `C10_CUDA_KERNEL_LAUNCH_CHECK()`. A plain `<<<grid, block>>>` goes to
  the null stream, races the model stream and fails silently.
- **Zero atomically-accumulated outputs.** Kernels that accumulate with atomics
  need `torch.zeros`, not `torch.empty`.
- **Never route to wvSplitK or LLMM1 on gfx10.** They build for gfx1030 but
  assert on the device.
- **Disable kernels through the registry.** Use `VLLM_DISABLED_KERNELS=<Class>`;
  do not add per-kernel `VLLM_DISABLE_*` env vars. Prefer flags that travel
  through the vLLM config (CLI / `--additional-config`) over env-only knobs:
  engine-core and worker processes do not always see the caller's environment.
- **Cudagraphs are the production path.** Serving runs `FULL_AND_PIECEWISE`;
  a change must stay correct both there and under `--enforce-eager`, and any
  speed change needs an A/B against the current branch.
- **Benchmarks report four numbers per cell:** prefill tok/s, decode tok/s per
  request, TTFT (s) and ITL (ms), for 16k/1k and 1k/512 at c=1 and c=8.

## Where RDNA docs live

| Topic | Location |
|---|---|
| Human wiki (setup, recipes, troubleshooting) | <https://blivioniag.github.io/gfx1030-wiki/> |
| Serve launcher and recipes | `tools/rdna/serve_rdna.sh`, `tools/rdna/recipes/` |
| Kernel and V620 notes | `docs/rdna2/` |
| TunableOp profiles | `tunableop/` |

Do not invent tok/s or hardware claims. Prefer editing a recipe over hard-coding
host paths.

---
