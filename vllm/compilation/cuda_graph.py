# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
import weakref
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, ClassVar
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream, weak_ref_tensors

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str


class CUDAGraphLogging:
    """Aggregate and log cudagraph metrics"""

    COLUMN_HEADERS = [
        "Unpadded Tokens",
        "Padded Tokens",
        "Num Paddings",
        "Runtime Mode",
        "Count",
    ]

    def __init__(
        self, cg_mode: CUDAGraphMode, cg_capture_sizes: list[int] | None
    ) -> None:
        self.reset()
        self.cg_mode = str(cg_mode)
        self.cg_capture_sizes = str(cg_capture_sizes or [])

        self.settings_header = (
            "**CUDAGraph Config Settings:**\n\n"
            f"- Mode: {self.cg_mode}\n"
            f"- Capture sizes: {self.cg_capture_sizes}\n\n"
            "**CUDAGraph Stats:**\n\n"
        )

    def reset(self) -> None:
        self.stats: list[CUDAGraphStat] = []

    def observe(self, cudagraph_stat: CUDAGraphStat) -> None:
        self.stats.append(cudagraph_stat)

    def generate_metric_table(self) -> str:
        stats_counts = Counter(self.stats)

        # Convert stats to rows of strings, in descending order of observed frequencies
        rows = []
        for stat, count in sorted(
            stats_counts.items(), key=lambda item: item[1], reverse=True
        ):
            rows.append(
                [
                    str(stat.num_unpadded_tokens),
                    str(stat.num_padded_tokens),
                    str(stat.num_paddings),
                    stat.runtime_mode,
                    str(count),
                ]
            )

        # Calculate column widths (max of header and data)
        col_widths = []
        for i, header_text in enumerate(self.COLUMN_HEADERS):
            max_width = len(header_text)
            for row in rows:
                max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)

        table_header_list = [
            h.ljust(w) for h, w in zip(self.COLUMN_HEADERS, col_widths)
        ]
        table_header = "| " + " | ".join(table_header_list) + " |\n"

        table_separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n"

        # Create data rows with proper alignment
        data_rows = []
        for row in rows:
            formatted_row = [
                str(val).ljust(width) for val, width in zip(row, col_widths)
            ]
            data_rows.append("| " + " | ".join(formatted_row) + " |")

        return (
            self.settings_header
            + table_header
            + table_separator
            + "\n".join(data_rows)
            + "\n"
        )

    def log(self, log_fn: Callable[..., Any] = logger.info) -> None:
        if not self.stats:
            return
        log_fn(self.generate_metric_table())
        self.reset()


@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None
    # Strong refs to capture-time input tensors so replay can copy
    # runtime values into the addresses the graph recorded.
    static_input_tensors: list[torch.Tensor] | None = None
    # Cloned arg/kwarg tree used as the captured graph's actual inputs.
    static_args: tuple[Any, ...] | None = None
    static_kwargs: dict[str, Any] | None = None


@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: on replay, runtime input tensors are copied into the capture-time
    buffers when shape/dtype/device match. If they do not match, replay is
    skipped and the underlying runnable runs eagerly. Address tracing for
    debug is still available when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    _all_instances: ClassVar[weakref.WeakSet["CUDAGraphWrapper"]] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:
        """Clear captured graphs from all CUDAGraphWrapper instances."""
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}

        CUDAGraphWrapper._all_instances.add(self)

    def __getattr__(self, key: str) -> Any:
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable[..., Any]:
        # in case we need to access the original runnable.
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "CUDAGraphWrapper":
        return self

    def clear_graphs(self) -> None:
        self.concrete_cudagraph_entries.clear()

    @staticmethod
    def _walk_tensors(obj: Any, out: list[torch.Tensor]) -> None:
        if isinstance(obj, torch.Tensor):
            out.append(obj)
        elif isinstance(obj, (tuple, list)):
            for x in obj:
                CUDAGraphWrapper._walk_tensors(x, out)
        elif isinstance(obj, dict):
            for v in obj.values():
                CUDAGraphWrapper._walk_tensors(v, out)

    @staticmethod
    def _collect_input_tensors(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> list[torch.Tensor]:
        # Piecewise FX subgraphs often pass activations as nested tuples;
        # only collecting top-level Tensor args left those stale on replay
        # (prompt-independent "duct" on eager-backend piecewise).
        tensors: list[torch.Tensor] = []
        for a in args:
            CUDAGraphWrapper._walk_tensors(a, tensors)
        for v in kwargs.values():
            CUDAGraphWrapper._walk_tensors(v, tensors)
        return tensors

    @staticmethod
    def copy_runtime_inputs_into_static(
        runtime: list[torch.Tensor], static: list[torch.Tensor]
    ) -> bool:
        """Copy runtime tensors into capture-time buffers when ptrs differ.

        Returns False if a replay-safe copy is not possible (count/shape/dtype
        mismatch); the caller should run eager instead of replaying.
        """
        if len(runtime) != len(static):
            return False
        for src, dst in zip(runtime, static):
            if src.data_ptr() == dst.data_ptr():
                continue
            if (
                src.shape != dst.shape
                or src.dtype != dst.dtype
                or src.device != dst.device
            ):
                return False
            dst.copy_(src)
        return True

    @staticmethod
    def _is_capture_static_tensor(t: torch.Tensor, num_tokens: int) -> bool:
        """Clone only small token-parallel activations for the graph.

        Full-tree clone OOMs on 32GB (KV / compile-range buffers). The
        graph must still see a distinct buffer so replay can copy runtime
        activations into the addresses capture recorded.
        """
        if t.numel() == 0:
            return False
        # ~4 MiB fp16. KV pages and 2048×hidden compile buffers stay aliased.
        # num_tokens is the capture size; activations are small either way.
        return t.numel() <= 2_000_000

    @staticmethod
    def _clone_tree(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        if isinstance(obj, tuple):
            return tuple(CUDAGraphWrapper._clone_tree(x) for x in obj)
        if isinstance(obj, list):
            return [CUDAGraphWrapper._clone_tree(x) for x in obj]
        if isinstance(obj, dict):
            return {k: CUDAGraphWrapper._clone_tree(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def _clone_activations(obj: Any, num_tokens: int) -> Any:
        if isinstance(obj, torch.Tensor):
            if CUDAGraphWrapper._is_capture_static_tensor(obj, num_tokens):
                # clone() is contiguous. Needed for inductor assert_size_stride
                # on M-RoPE positions (dummy extra column → stride 2049).
                # Do not clone packed W4A16 int tables (numel >> 2e6).
                return obj.clone()
            return obj
        if isinstance(obj, tuple):
            return tuple(
                CUDAGraphWrapper._clone_activations(x, num_tokens) for x in obj
            )
        if isinstance(obj, list):
            return [
                CUDAGraphWrapper._clone_activations(x, num_tokens) for x in obj
            ]
        if isinstance(obj, dict):
            return {
                k: CUDAGraphWrapper._clone_activations(v, num_tokens)
                for k, v in obj.items()
            }
        return obj

    @staticmethod
    def _copy_tree(src: Any, dst: Any) -> bool:
        if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
            if src.dtype != dst.dtype or src.device != dst.device:
                return False
            if src.data_ptr() == dst.data_ptr():
                return True
            if src.shape == dst.shape:
                dst.copy_(src)
                return True
            # Token-parallel activations: copy the live prefix into the
            # captured buffer (compile-range tensors are often longer).
            if (
                src.dim() > 0
                and dst.dim() > 0
                and src.shape[0] <= dst.shape[0]
                and src.shape[1:] == dst.shape[1:]
            ):
                dst[: src.shape[0]].copy_(src)
                return True
            return False
        if isinstance(src, (tuple, list)) and isinstance(dst, type(src)):
            if len(src) != len(dst):
                return False
            return all(
                CUDAGraphWrapper._copy_tree(s, d) for s, d in zip(src, dst)
            )
        if isinstance(src, dict) and isinstance(dst, dict):
            if src.keys() != dst.keys():
                return False
            return all(CUDAGraphWrapper._copy_tree(src[k], dst[k]) for k in src)
        return True

    @staticmethod
    def _nan_to_num_tree(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point() and 0 < obj.numel() <= 2_000_000:
                if obj.isnan().any():
                    obj.nan_to_num_(0.0)
            return
        if isinstance(obj, (tuple, list)):
            for x in obj:
                CUDAGraphWrapper._nan_to_num_tree(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                CUDAGraphWrapper._nan_to_num_tree(v)

    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:
        if not is_forward_context_available():
            # No forward context means we are outside the normal
            # inference path (e.g. a vision encoder forward pass).
            # Just run the underlying function without cudagraphs.
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if os.environ.get("VLLM_PIECE_IN_DEBUG") == "1":
            try:
                _pn = getattr(self, "_piece_in_n", 0)
                if _pn < 10:
                    self._piece_in_n = _pn + 1
                    _rid = (
                        getattr(self, "submod_name", None)
                        or getattr(self.runnable, "submod_name", None)
                        or repr(self.runnable)[:60]
                    )
                    _t = self._collect_input_tensors(args, kwargs)
                    with open(f"/tmp/piece_in_{torch.cuda.current_device()}.log", "a") as _f:
                        _f.write(f"\n[piece_in] call#{_pn} rank={torch.cuda.current_device()} "
                                 f"bd={batch_descriptor} rid={_rid}\n")
                        for _i, _x in enumerate(_t):
                            try:
                                if _x.is_floating_point() and 0 < _x.numel() <= 2_000_000:
                                    _nan = bool(_x.isnan().any().item())
                                    _v = _x.flatten()[:4].tolist()
                                    _f.write(f"  in[{_i}] s={tuple(_x.shape)} d={_x.dtype} "
                                             f"p=0x{_x.data_ptr():x} nan={_nan} v={_v}\n")
                                else:
                                    _f.write(f"  in[{_i}] s={tuple(_x.shape)} d={_x.dtype} "
                                             f"p=0x{_x.data_ptr():x} (big)\n")
                            except Exception as _e:
                                _f.write(f"  in[{_i}] err={_e}\n")
            except Exception:
                pass

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            # Dedicated buffers for token-parallel activations so replay
            # can copy_ into the addresses HIP recorded. Aliasing the
            # caller's tensors makes _copy_tree a no-op (same_ptr) and
            # the graph keeps warmup input_ids ([0, 1, 0, 1, ...]).
            # Weights / KV stay aliased — full-tree clone OOMs on 32GB.
            num_tokens = entry.batch_descriptor.num_tokens
            entry.static_args = self._clone_activations(args, num_tokens)
            entry.static_kwargs = self._clone_activations(kwargs, num_tokens)
            static_inputs = self._collect_input_tensors(
                entry.static_args, entry.static_kwargs
            )
            entry.static_input_tensors = static_inputs
            entry.input_addresses = [x.data_ptr() for x in static_inputs]
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(
                        patch("gc.collect", lambda *args, **kwargs: None)
                    )
                    stack.enter_context(
                        patch(
                            "torch.accelerator.empty_cache",
                            lambda *args, **kwargs: None,
                        )
                    )

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())

                # Sync offloader's copy stream before capture.
                # Ensure any pre-capture prefetches from offloader are complete.
                get_offloader().sync_prev_onload()

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(
                    cudagraph,
                    pool=self.graph_pool,
                    stream=current_stream(),
                ):
                    output = self.runnable(
                        *entry.static_args, **entry.static_kwargs
                    )
                    # Join offloader's copy stream after forward to avoid
                    # unjoined stream error. The last layer's start_prefetch
                    # forks copy_stream, but wait_prefetch only happens in
                    # the next forward pass.
                    get_offloader().join_after_forward()
                    # Strong ref on the entry so HIP replay returns the
                    # graph's output buffer, not a dead weak-ref of warmup.
                    entry.output = output
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            self._nan_to_num_tree(entry.output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        skip_replay = os.environ.get("VLLM_CG_SKIP_REPLAY") == "1"
        copied = (
            entry.static_args is not None
            and entry.static_kwargs is not None
            and self._copy_tree(args, entry.static_args)
            and self._copy_tree(kwargs, entry.static_kwargs)
        )
        # Opt-in debug scan: per-replay isnan().any() costs blocking host
        # syncs every step (~10% of c=8 GPU time). Padded rows are
        # row-parallel and arenas are zero-init, so real rows are safe.
        if os.environ.get("VLLM_CG_NAN_INPUT_CHECK") == "1":
            for t in self._collect_input_tensors(
                entry.static_args or (), entry.static_kwargs or {}
            ):
                if (
                    t.is_floating_point()
                    and 0 < t.numel() <= 2_000_000
                    and t.isnan().any()
                ):
                    t.nan_to_num_(0.0)
        log_replay = os.environ.get("VLLM_CG_REPLAY_LOG") == "1"
        if log_replay:
            rt = self._collect_input_tensors(args, kwargs)
            st = self._collect_input_tensors(
                entry.static_args or (), entry.static_kwargs or {}
            )
            same = sum(
                1
                for a, b in zip(rt, st)
                if a.data_ptr() == b.data_ptr()
            )
            ids_t = next(
                (
                    t
                    for t in st
                    if t.dtype == torch.int32 and t.dim() == 1 and 0 < t.numel() <= 32
                ),
                None,
            )
            ids = None if ids_t is None else ids_t.flatten()[:8].tolist()
            dummy = bool(
                ids is not None
                and len(ids) >= 2
                and ids == ([0, 1] * ((len(ids) + 1) // 2))[: len(ids)]
            )
            _elog = getattr(self, "_cg_embed_log_n", 0)
            _nlog = getattr(self, "_cg_replay_log_n", 0)
            if ids is not None and _elog < 24:
                self._cg_embed_log_n = _elog + 1
                logger.warning(
                    "cg-replay-embed copied=%s skip=%s n=%s same_ptr=%s "
                    "dummy=%s input_ids=%s",
                    copied,
                    skip_replay,
                    len(rt),
                    same,
                    dummy,
                    ids,
                )
            elif ids is None and _nlog < 8:
                self._cg_replay_log_n = _nlog + 1
                spec = [
                    (
                        tuple(t.shape),
                        str(t.dtype).replace("torch.", ""),
                        bool(t.isnan().any().item())
                        if t.is_floating_point() and t.numel() < 2_000_000
                        else None,
                    )
                    for t in rt
                ]
                logger.warning(
                    "cg-replay copied=%s skip=%s n=%s same_ptr=%s tensors=%s",
                    copied,
                    skip_replay,
                    len(rt),
                    same,
                    spec[:12],
                )
        if skip_replay or not copied:
            # Addresses/shapes no longer match the captured graph; do not
            # replay warmup tokens. Fall back to the underlying runnable.
            return self.runnable(*args, **kwargs)

        if self.is_debugging_mode:
            runtime_inputs = self._collect_input_tensors(args, kwargs)
            new_input_addresses = [x.data_ptr() for x in runtime_inputs]
            # After a successful copy, the graph still reads static ptrs.
            assert entry.input_addresses is not None
            assert len(new_input_addresses) == len(entry.input_addresses), (
                f"Input tensor count changed during replay. Expected "
                f"{len(entry.input_addresses)}, got {len(new_input_addresses)}"
            )

        # Sync offloader before replay - ensures any external dependencies
        # from pre-capture prefetches are satisfied.
        get_offloader().sync_prev_onload()
        entry.cudagraph.replay()
        return entry.output
