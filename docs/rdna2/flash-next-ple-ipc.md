# Flash-Next PLE offload on gfx1030 — CUDA-IPC failure and fix

**Status**: root-caused and fixed 2026-09-14. `expandable_segments` and the
`is_pinned()` probe were the two blockers; a third (`CpuGpuSemaphore.signal`
returning `hipError_t(709)`) remains open.

## Symptom

Serving `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` (Qwen4Exp, 512-expert W4A16 MoE)
with the PLE sidecar (`VLLM_PLE_CPU_OFFLOAD=1`,
`VLLM_PLE_QUANT_DIR=.../ples_int4`) died during PLE setup:

```
PleOffloadWorker: accept_registrations (worker.py:1152)
  -> torch rebuild_cuda_tensor -> _new_shared_cuda
  -> torch.AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)
```

The model itself loaded, sharded and EP-split correctly; only the offload
worker's cross-process handshake failed.

## Root cause

`scripts/serve_gfx1030_full.sh` exports
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. With expandable segments the
CUDA caching allocator carves large allocations straight from VMM
(`cuMemCreate`/`cuMemMap`). VMM-backed memory **cannot be exported through CUDA
IPC** on ROCm, so the receiving process's `hipIpcOpenMemHandle` fails with
`hipErrorInvalidValue` — for *every* PLE tensor, starting with the 10 MB
`[2048, 2560]` fp16 `gpu_output_buffer`.

Minimal repro (two spawned processes, one tensor):

| `PYTORCH_CUDA_ALLOC_CONF` | result |
|---|---|
| unset | shared tensor rebuilds **OK** |
| `expandable_segments:True` | **`hipErrorInvalidValue`** |

Allocation size (4 B … 10 MiB) and device (cuda:0…3) were ruled out — they all
share fine without expandable segments.

## Fix

**Turn expandable segments off for any run that uses PLE offload.**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
VLLM_PLE_CPU_OFFLOAD=1 \
VLLM_PLE_QUANT_DIR=<sidecar>/ples_int4 \
  ... --enable-expert-parallel
```

Note `serve_gfx1030_full.sh` uses
`PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"`,
so exporting an *empty* value re-selects the default: pass
`expandable_segments:False` explicitly.

Expandable segments are otherwise preferred here (they fix a cudagraph hole at
1k/1k c=8 — see the serve script), so this is a PLE-only override until the
buffers are carved out of the expandable pool (e.g. allocate the PLE output
buffers with a plain `cudaMalloc`/`torch.cuda.caching_allocator_alloc` outside
the pool, then IPC those).

## Secondary fix: `is_pinned()` false negative

After the IPC fix, `_pin_input_buffers` still failed:

```
RuntimeError: CUDA did not page-lock a PLE input buffer
```

`hipHostRegister` returned `0` (success) but `buffer.is_pinned()` reported
`False`. In isolation the same call on a `share_memory_()` CPU tensor reports
`is_pinned() == True`, so this is a torch probe quirk inside the worker context,
not a pinning failure — `hipHostRegister` returning success *is* the page-lock.
The check is now a warning.

## Verification

With the fix, all four GPU workers complete registration:

```
GPU worker 0 registered (dp_rank=0, tp_rank=0, ...)
GPU worker 3 registered (dp_rank=0, tp_rank=3, ...)
GPU worker 2 registered (dp_rank=0, tp_rank=2, ...)
GPU worker 1 registered (dp_rank=0, tp_rank=1, ...)
```

## Remaining blocker (open)

`CpuGpuSemaphore.signal failed: hipError_t(709)` (`hipErrorContextIsDestroyed`)
from `hipStreamWriteValue32` on the copy stream, raised by
`connector.py:541/677`. Only one `libamdhip64` is loaded
(`_rocm_sdk_core/lib/libamdhip64.so.7`) and every `ctypes` load form resolves the
symbol, so it is not a dual-HIP-runtime problem. Needs the stream/context
lifetime around `CpuGpuSemaphore.{reset,wait_reset,signal}` investigated next.
