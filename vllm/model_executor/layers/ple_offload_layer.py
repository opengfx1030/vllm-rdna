# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for PLE layers that support cross-process CPU offload.

Synchronization between the offload process and the GPU main stream is
handled by :class:`CpuGpuSemaphore`, which uses CUDA's
``cuStreamWaitValue32`` / ``cuStreamWriteValue32`` stream memory operations.

Typical decode-step timeline
----------------------------
Offload process (CPU)                  GPU main stream (forward)
-----------------------------          ------------------------------
forward_impl() on CPU                  ... other GPU ops ...
pinned -> GPU async H2D (copy_stream)   WaitValue32(flag==1)  <- in Graph
WriteValue32(flag=1, copy_stream)       return _gpu_output_buffer[:n]
                                       ... model consumes output ...
                                       WriteValue32(flag=0)  <- after forward
"""

import functools
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

# cuda-python is an NVIDIA-only package. The stream write/wait-value and host
# pinning primitives below have direct HIP equivalents, but no binding for them
# ships here, so import lazily: everything except the actual offload path must
# keep working on ROCm (the layer base class is imported by every qwen4_exp
# build, offloaded or not).
# On ROCm the HIP shim is always the right driver: cuda-bindings may be
# importable (a transitive dependency of a full install) but has no libcuda.
try:
    if torch.version.hip is not None:
        raise ImportError("ROCm: use the HIP driver shim")
    from cuda.bindings import driver as cuda_driver
    from cuda.bindings.driver import CUstreamWaitValue_flags

    PLE_OFFLOAD_DRIVER_AVAILABLE = True
except ImportError:  # pragma: no cover - platform dependent
    # ROCm: every primitive used here has a direct HIP equivalent, bound from
    # libamdhip64 by vllm.v1.ple_offload.hip_driver with the same surface.
    from vllm.v1.ple_offload import hip_driver as cuda_driver  # type: ignore[no-redef]
    from vllm.v1.ple_offload.hip_driver import (  # type: ignore[no-redef]
        CUstreamWaitValue_flags,
    )

    PLE_OFFLOAD_DRIVER_AVAILABLE = cuda_driver.HIP_DRIVER_AVAILABLE

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.lane import current_lane

# Module-level flag set to True inside the offload subprocess.
# Because the offload process and GPU worker processes are separate OS
# processes (spawned via multiprocessing), each has its own memory space.
# A plain module-level bool is sufficient -- no thread-local storage needed.
_offload_worker_flag = False


def is_offload_process() -> bool:
    """Return True inside the dedicated PLE CPU-offload subprocess."""
    return _offload_worker_flag


def mark_as_offload_worker() -> None:
    """Mark the current process as the dedicated PLE offload worker."""
    global _offload_worker_flag
    _offload_worker_flag = True


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


class CpuGpuSemaphore:
    """Cross-process semaphore backed by an int32 CUDA tensor.

    The flag is stored in a regular GPU tensor that is shared through
    PyTorch's CUDA IPC mechanism. Both processes obtain device pointers that
    map to the same physical GPU memory, so a stream-memory wait in the GPU
    worker observes the write issued by the offload process.

    One flag per lane: the two-lane engine runs a decode and a prefill forward
    concurrently, each with its own offload round-trip, so a single flag would
    let one lane's completion release the other lane's wait.
    """

    RESET_VALUE = 0
    DONE_VALUE = 1

    def __init__(self, device: torch.device, num_lanes: int = 1) -> None:
        self._flag_tensor = torch.zeros(
            int(num_lanes), dtype=torch.int32, device=device
        )

    @classmethod
    def from_ipc_tensor(cls, flag_tensor: torch.Tensor) -> "CpuGpuSemaphore":
        """Construct a semaphore from a CUDA tensor received through IPC."""
        semaphore = cls.__new__(cls)
        semaphore._flag_tensor = flag_tensor
        return semaphore

    @property
    def flag_tensor(self) -> torch.Tensor:
        """Return the CUDA tensor used to share the semaphore through IPC."""
        return self._flag_tensor

    def _lane_ptr(self, lane: int) -> int:
        """Device address of ``lane``'s flag (the tensor is lane-contiguous)."""
        if lane < 0 or lane >= self._flag_tensor.numel():
            raise IndexError(
                f"PLE semaphore has {self._flag_tensor.numel()} lane flag(s), "
                f"asked for lane {lane}"
            )
        return self._flag_tensor.data_ptr() + lane * self._flag_tensor.element_size()

    def reset(self, stream: torch.cuda.Stream | None = None, lane: int = 0) -> None:
        """Enqueue ``WriteValue32(flag=0)`` for ``lane`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._lane_ptr(lane)),
                self.RESET_VALUE,
                0,
            ),
            "CpuGpuSemaphore.reset",
        )

    def signal(self, stream: torch.cuda.Stream | None = None, lane: int = 0) -> None:
        """Enqueue ``WriteValue32(flag=1)`` for ``lane`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._lane_ptr(lane)),
                self.DONE_VALUE,
                0,
            ),
            "CpuGpuSemaphore.signal",
        )

    def wait_reset(
        self, stream: torch.cuda.Stream | None = None, lane: int = 0
    ) -> None:
        """Enqueue ``WaitValue32(flag==0)`` for ``lane`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWaitValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._lane_ptr(lane)),
                self.RESET_VALUE,
                CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
            ),
            "CpuGpuSemaphore.wait_reset",
        )


# ---------------------------------------------------------------------------
# Custom op: vllm::ple_offload_output
# ---------------------------------------------------------------------------
# Selects the calling lane's result rows and (on CUDA) waits for the CPU
# result. Both the lane lookup and the driver call must stay outside the
# traced graph: vLLM drops all Dynamo guards, so a host-side lane read inside
# the graph would be baked in at trace time and the two lanes would share one
# buffer. A custom-op impl runs eagerly on the calling thread, so
# ``current_lane()`` is correct for the decode lane (eager or graph replay) and
# for the prefill lane (eager segments) alike.
# ---------------------------------------------------------------------------


def _ple_offload_output_impl(
    sem_flag_tensor: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Return the calling lane's result rows, waiting for them on CUDA."""
    lane = int(current_lane())
    if torch.version.hip is None:
        stream = torch.cuda.current_stream()
        cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
        dev_ptr = cuda_driver.CUdeviceptr(
            sem_flag_tensor.data_ptr() + lane * sem_flag_tensor.element_size()
        )
        _cuda_check(
            cuda_driver.cuStreamWaitValue32(
                cuda_stream,
                dev_ptr,
                CpuGpuSemaphore.DONE_VALUE,
                CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
            ),
            "cuStreamWaitValue32(done)",
        )
    else:
        # ROCm: the connector already waits on the host for the CPU result and
        # copies it on this stream before the forward (PleOffloadConnector.
        # prepare_forward), so this device wait is redundant. hipStreamWaitValue32
        # is not recorded into HIP graphs, and in eager steps a pending
        # WAIT_REG_MEM packet cannot be preempted when KFD evicts queues: the
        # stream stalls and the engine hangs (seen under the concurrency probe).
        pass
    return gpu_output_buffer[lane][:num_tokens]


def _ple_offload_output_fake(
    sem_flag_tensor: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Represent the lane-selected result during dynamo tracing."""
    return gpu_output_buffer.new_empty((num_tokens, gpu_output_buffer.shape[-1]))


direct_register_custom_op(
    op_name="ple_offload_output",
    op_func=_ple_offload_output_impl,
    mutates_args=[],
    fake_impl=_ple_offload_output_fake,
)


class PleOffloadLayer(nn.Module, ABC):
    """Base class for embedding-like PLE layers that can run on CPU.

    Subclasses implement :meth:`forward_impl`, which contains the regular
    on-device computation. In a GPU worker with PLE offload enabled, subclass
    constructors are skipped so their large weights are never allocated. The
    CPU process constructs a complete copy and owns those weights instead.
    """

    _is_cpu_offloaded = False
    _gpu_output_buffer: torch.Tensor
    _sem: CpuGpuSemaphore

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Skip subclass initialization in cross-process GPU-worker mode."""
        super().__init_subclass__(**kwargs)
        original_init_obj = cls.__dict__.get("__init__")
        if original_init_obj is None or not callable(original_init_obj):
            return
        original_init = cast(Callable[..., None], original_init_obj)

        @functools.wraps(original_init)
        def guarded_init(
            self: "PleOffloadLayer", *args: object, **kwargs: object
        ) -> None:
            if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
                nn.Module.__init__(self)
                return
            if (
                is_offload_process()
                and not envs.VLLM_PLE_QUANT_DIR
                and not envs.VLLM_PLE_DISK_OFFLOAD_DIR
            ):
                # Only the PLE subtree owns RAM during meta model discovery.
                with torch.device("cpu"):
                    original_init(self, *args, **kwargs)
                return
            original_init(self, *args, **kwargs)

        cls.__init__ = guarded_init  # type: ignore[method-assign, assignment]

    @classmethod
    def get_target_device(cls) -> torch.device:
        """Return CPU for the offload process and the active GPU otherwise."""
        if envs.VLLM_PLE_CPU_OFFLOAD:
            return torch.device("cpu")
        return torch.device("cuda", torch.accelerator.current_device_index())

    @abstractmethod
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Execute the actual embedding computation on the owning device."""
        raise NotImplementedError

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype used by cross-process output buffers."""
        return default_dtype

    def setup_cross_process_offload(
        self,
        gpu_output_buffer: torch.Tensor,
        semaphore: CpuGpuSemaphore,
    ) -> None:
        """Configure the GPU-worker placeholder with its IPC resources.

        ``gpu_output_buffer`` is lane-major (``[num_lanes, max_tokens, dim]``)
        and ``semaphore`` holds one flag per lane, so both lanes of the
        two-lane engine share one module instance without sharing a buffer.
        """
        self._is_cpu_offloaded = True
        self._gpu_output_buffer = gpu_output_buffer
        self._sem = semaphore

    @eager_break_during_capture
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Wait for an offloaded result or delegate to ``forward_impl``."""
        if self._is_cpu_offloaded:
            out = torch.ops.vllm.ple_offload_output(
                self._sem.flag_tensor,
                self._gpu_output_buffer,
                input_ids.shape[0],
            )
            if (
                os.environ.get("VLLM_PLE_NAN_DEBUG") == "1"
                and not torch.cuda.is_current_stream_capturing()
            ):
                try:
                    if bool(torch.isnan(out).any().item()):
                        import vllm

                        vllm.logger.init_logger(__name__).warning(
                            "[ple-nan] BAD out nan=True"
                        )
                except Exception:
                    pass
            return out
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    def release_offloaded_output(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Mark the calling lane's cross-process output buffer reusable."""
        if self._is_cpu_offloaded:
            self._sem.reset(stream, lane=int(current_lane()))
