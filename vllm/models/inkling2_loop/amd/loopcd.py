# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoopCD (Loop-Contrastive Decoding, arXiv:2610.02185) for Inkling-2 on RDNA2.

Supports two zero-sync inference modes for looped-core transformers:

* ``hidden`` (LoopCD-Hidden): pre-coda representation extrapolation across the
  final two loop passes ``h_cd = h_final + alpha * (h_final - h_prev)`` (plus
  optional loop-interior AttnRes block-state extrapolation), gated to ``0`` on
  non-terminal prompt prefill tokens so KV / conv history stays exact.
* ``logits`` (LoopCD-Logits): distribution-level contrastive decoding between
  the final readout and an early-exit weak readout (penultimate pass ``P - 2``
  or Pass ``0``) with APC plausibility cutoff ``beta`` and optional uncertainty-
  adaptive scaling ``(1 - max_v p_final(v)) * alpha``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from .model import Inkling2LoopTextForCausalLM

LOOPCD_MODE_OFF = 0
LOOPCD_MODE_HIDDEN = 1
LOOPCD_MODE_LOGITS = 2
LOOPCD_MAX_WEAK_ROWS = 256

_TRUE_STRS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRS = frozenset({"0", "false", "no", "off", ""})


def _parse_bool(val: Any, name: str) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        if val in (0, 1):
            return bool(val)
        raise ValueError(f"{name} must be a boolean (0/1 or true/false), got {val!r}")
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _TRUE_STRS:
            return True
        if s in _FALSE_STRS:
            return False
    raise ValueError(f"{name} must be a boolean, got {val!r}")


def _parse_float(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float, str)):
        raise ValueError(f"{name} must be a float, got {val!r}")
    try:
        out = float(val.strip() if isinstance(val, str) else val)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {val!r}") from exc
    return out


def _parse_int(val: Any, name: str) -> int:
    if isinstance(val, bool):
        raise ValueError(f"{name} must be an integer, got {val!r}")
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {val!r}") from exc
    raise ValueError(f"{name} must be an integer, got {val!r}")


def _parse_mode(val: Any, name: str) -> Literal["hidden", "logits"]:
    if not isinstance(val, str):
        raise ValueError(f"{name} must be 'hidden' or 'logits', got {val!r}")
    s = val.strip().lower()
    if s == "hidden":
        return "hidden"
    if s == "logits":
        return "logits"
    raise ValueError(f"{name} must be 'hidden' or 'logits', got {val!r}")


@dataclass(frozen=True)
class LoopCDParams:
    alpha: float = 0.0
    mode: Literal["hidden", "logits"] = "hidden"
    adaptive: bool = False
    beta: float = 0.1
    blend_blocks: bool = True
    weak_pass: int = -1  # -1 => passes - 2 (penultimate pass); 0 => Pass 0

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ValueError(f"alpha must be a finite float >= 0, got {self.alpha!r}")
        alpha_f = float(self.alpha)
        if not math.isfinite(alpha_f) or alpha_f < 0.0:
            raise ValueError(f"alpha must be a finite float >= 0, got {self.alpha!r}")
        object.__setattr__(self, "alpha", alpha_f)

        if self.mode not in ("hidden", "logits"):
            raise ValueError(f"mode must be 'hidden' or 'logits', got {self.mode!r}")

        if not isinstance(self.adaptive, bool):
            raise ValueError(f"adaptive must be a bool, got {self.adaptive!r}")

        if isinstance(self.beta, bool) or not isinstance(self.beta, (int, float)):
            raise ValueError(f"beta must be a float in [0.0, 1.0], got {self.beta!r}")
        beta_f = float(self.beta)
        if not math.isfinite(beta_f) or not (0.0 <= beta_f <= 1.0):
            raise ValueError(f"beta must be a float in [0.0, 1.0], got {self.beta!r}")
        object.__setattr__(self, "beta", beta_f)

        if not isinstance(self.blend_blocks, bool):
            raise ValueError(f"blend_blocks must be a bool, got {self.blend_blocks!r}")

        if isinstance(self.weak_pass, bool) or not isinstance(self.weak_pass, int):
            raise ValueError(
                f"weak_pass must be -1 or an int >= 0, got {self.weak_pass!r}"
            )
        if self.weak_pass < -1:
            raise ValueError(
                f"weak_pass must be -1 or an int >= 0, got {self.weak_pass!r}"
            )

    @property
    def enabled(self) -> bool:
        return self.alpha > 0.0

    @property
    def mode_code(self) -> int:
        if not self.enabled:
            return LOOPCD_MODE_OFF
        return LOOPCD_MODE_HIDDEN if self.mode == "hidden" else LOOPCD_MODE_LOGITS

    @property
    def log_beta(self) -> float:
        return -float("inf") if self.beta <= 0.0 else math.log(self.beta)


def read_loopcd_env_defaults() -> LoopCDParams:
    """Read server-wide LoopCD defaults from ``VLLM_INKLING2_LOOPCD_*``."""
    raw_alpha = os.environ.get("VLLM_INKLING2_LOOPCD_ALPHA")
    alpha = (
        0.0
        if raw_alpha is None
        else _parse_float(raw_alpha, "VLLM_INKLING2_LOOPCD_ALPHA")
    )

    raw_mode = os.environ.get("VLLM_INKLING2_LOOPCD_MODE")
    mode: Literal["hidden", "logits"] = (
        "hidden"
        if raw_mode is None
        else _parse_mode(raw_mode, "VLLM_INKLING2_LOOPCD_MODE")
    )

    raw_adaptive = os.environ.get("VLLM_INKLING2_LOOPCD_ADAPTIVE")
    adaptive = (
        False
        if raw_adaptive is None
        else _parse_bool(raw_adaptive, "VLLM_INKLING2_LOOPCD_ADAPTIVE")
    )

    raw_beta = os.environ.get("VLLM_INKLING2_LOOPCD_BETA")
    beta = (
        0.1 if raw_beta is None else _parse_float(raw_beta, "VLLM_INKLING2_LOOPCD_BETA")
    )

    raw_blend = os.environ.get("VLLM_INKLING2_LOOPCD_BLEND_BLOCKS")
    blend_blocks = (
        True
        if raw_blend is None
        else _parse_bool(raw_blend, "VLLM_INKLING2_LOOPCD_BLEND_BLOCKS")
    )

    raw_wp = os.environ.get("VLLM_INKLING2_LOOPCD_WEAK_PASS")
    weak_pass = (
        -1 if raw_wp is None else _parse_int(raw_wp, "VLLM_INKLING2_LOOPCD_WEAK_PASS")
    )

    return LoopCDParams(
        alpha=alpha,
        mode=mode,
        adaptive=adaptive,
        beta=beta,
        blend_blocks=blend_blocks,
        weak_pass=weak_pass,
    )


def loopcd_graph_hooks_enabled() -> bool:
    """Whether LoopCD hooks are baked into captured CUDA graphs.

    Defaults to ``True`` so per-request ``vllm_xargs`` LoopCD works out of the
    box without a separate recipe. Set ``VLLM_INKLING2_LOOPCD_ENABLE=0`` (when
    ``VLLM_INKLING2_LOOPCD_ALPHA=0``) to strip the 3 graph-baked LoopCD launches
    and restore layer 17->18 ``attn_res_sconv`` fusion.
    """
    raw = os.environ.get("VLLM_INKLING2_LOOPCD_ENABLE")
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, "VLLM_INKLING2_LOOPCD_ENABLE")
    return True


def parse_loopcd_params(
    extra_args: Mapping[str, Any] | None = None,
    defaults: LoopCDParams | None = None,
) -> LoopCDParams:
    """Resolve per-request ``SamplingParams.extra_args`` over ``defaults``."""
    base = read_loopcd_env_defaults() if defaults is None else defaults
    if not extra_args:
        return base

    alpha = (
        _parse_float(extra_args["loopcd_alpha"], "loopcd_alpha")
        if extra_args.get("loopcd_alpha") is not None
        else base.alpha
    )
    mode = (
        _parse_mode(extra_args["loopcd_mode"], "loopcd_mode")
        if extra_args.get("loopcd_mode") is not None
        else base.mode
    )
    adaptive = (
        _parse_bool(extra_args["loopcd_adaptive"], "loopcd_adaptive")
        if extra_args.get("loopcd_adaptive") is not None
        else base.adaptive
    )
    beta = (
        _parse_float(extra_args["loopcd_beta"], "loopcd_beta")
        if extra_args.get("loopcd_beta") is not None
        else base.beta
    )
    blend_blocks = (
        _parse_bool(extra_args["loopcd_blend_blocks"], "loopcd_blend_blocks")
        if extra_args.get("loopcd_blend_blocks") is not None
        else base.blend_blocks
    )
    weak_pass = (
        _parse_int(extra_args["loopcd_weak_pass"], "loopcd_weak_pass")
        if extra_args.get("loopcd_weak_pass") is not None
        else base.weak_pass
    )
    return LoopCDParams(
        alpha=alpha,
        mode=mode,
        adaptive=adaptive,
        beta=beta,
        blend_blocks=blend_blocks,
        weak_pass=weak_pass,
    )


def loopcd_blend_reference(
    x0: torch.Tensor,
    x1: torch.Tensor | None,
    prev_x0: torch.Tensor,
    prev_x1: torch.Tensor,
    blk: torch.Tensor | None,
    prev_blk: torch.Tensor | None,
    omega: torch.Tensor,
    omega_blk: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Vectorized fp32 reference for pre-coda LoopCD-Hidden state blending.

    Preserves exact ``x0[t]``, ``x1[t]``, and ``blk[t]`` bits (0 ULP) whenever
    the corresponding token weight is ``0.0``.
    """
    hf = x0 + x1 if x1 is not None else x0
    hp = prev_x0 + prev_x1
    w = omega[: x0.shape[0]].to(dtype=torch.float32, device=x0.device)
    act = w != 0.0
    blended_x0 = torch.where(
        act[:, None], (hf + w[:, None] * (hf - hp)).to(x0.dtype), x0
    )
    blended_x1 = (
        None if x1 is None else torch.where(act[:, None], torch.zeros_like(x1), x1)
    )
    if blk is not None and prev_blk is not None and blk.shape[1] > 0:
        wb_src = omega_blk if omega_blk is not None else omega
        wb = wb_src[: blk.shape[0]].to(dtype=torch.float32, device=blk.device)
        b_act = wb != 0.0
        P = blk.shape[1]
        blended_blk = torch.where(
            b_act[:, None, None],
            (blk + wb[:, None, None] * (blk - prev_blk[:, :P])).to(blk.dtype),
            blk,
        )
    else:
        blended_blk = blk
    return blended_x0, blended_x1, blended_blk


def loopcd_logits_reference(
    z_final: torch.Tensor,
    z_first: torch.Tensor,
    omega: torch.Tensor,
    log_beta: torch.Tensor,
    adaptive: torch.Tensor,
) -> torch.Tensor:
    """Vectorized float64 oracle for LoopCD-Logits with APC plausibility mask.

    Exact 0-ULP no-op on rows where ``omega[b] == 0.0`` or ``max_v z_final`` is
    non-finite (e.g. all-``-inf`` rows).
    """
    B = z_final.shape[0]
    if B == 0 or z_final.shape[-1] == 0:
        return z_final.clone()
    zf = z_final.double()
    zw = z_first.double()
    w0 = omega[:B].to(dtype=torch.float64, device=z_final.device)
    lb = log_beta[:B].to(dtype=torch.float64, device=z_final.device)
    adap = adaptive[:B].to(dtype=torch.bool, device=z_final.device)

    m_b = zf.max(dim=-1).values
    finite_row = torch.isfinite(m_b) & (w0 != 0.0)
    safe_m = torch.where(torch.isfinite(m_b), m_b, torch.zeros_like(m_b))
    zf_fin = torch.isfinite(zf)
    safe_zf = torch.where(zf_fin, zf, safe_m[:, None])
    denom = torch.where(
        zf_fin, torch.exp(safe_zf - safe_m[:, None]), torch.zeros_like(zf)
    ).sum(dim=-1)
    p_max = torch.where(denom > 0.0, 1.0 / denom, torch.ones_like(denom))
    wb = torch.where(adap, w0 * (1.0 - p_max).clamp(min=0.0), w0)
    cutoff = safe_m + lb

    both_fin = zf_fin & torch.isfinite(zw)
    safe_diff = torch.where(both_fin, zf - zw, torch.zeros_like(zf))
    cd_val = torch.where(both_fin, zf + wb[:, None] * safe_diff, zf)
    masked = torch.where(
        zf_fin & (zf >= cutoff[:, None]),
        cd_val,
        torch.full_like(cd_val, -float("inf")),
    ).to(z_final.dtype)
    return torch.where(finite_row[:, None], masked, z_final)


class Inkling2ModelState(DefaultModelState):
    """Runner state driving zero-sync per-request LoopCD on ``GPUModelRunner``."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.lm = cast(
            "Inkling2LoopTextForCausalLM",
            getattr(model, "get_language_model", lambda: model)(),
        )
        self.default_params = read_loopcd_env_defaults()
        ws = self.lm.model.glue_ws
        if ws is not None:
            ws.loopcd_weak_pass = self.default_params.weak_pass

        self.req_alpha = StagedWriteTensor(self.max_num_reqs, torch.float32, device)
        self.req_mode = StagedWriteTensor(self.max_num_reqs, torch.int32, device)
        self.req_blend_blk = StagedWriteTensor(self.max_num_reqs, torch.int32, device)
        self.req_adaptive = StagedWriteTensor(self.max_num_reqs, torch.int32, device)
        self.req_log_beta = StagedWriteTensor(self.max_num_reqs, torch.float32, device)

        self._logits_stats_scratch = torch.empty(
            (max(LOOPCD_MAX_WEAK_ROWS, self.max_num_reqs * 16), 32, 2),
            dtype=torch.float32,
            device=device,
        )
        self._req_id_to_slot: dict[str, int] = {}
        self._slot_modes = np.zeros(self.max_num_reqs, dtype=np.int8)
        self._slot_dirty = np.zeros(self.max_num_reqs, dtype=np.bool_)
        self._num_active_loopcd_reqs = 0
        self._has_staged_writes = False
        self._was_hidden_active = False
        self._was_logits_active = False
        self._cur_input_batch: InputBatch | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        extra_args = (
            new_req_data.sampling_params.extra_args
            if new_req_data.sampling_params is not None
            else None
        )
        p = parse_loopcd_params(extra_args, self.default_params)
        prev_mode = int(self._slot_modes[req_index])
        if prev_mode != LOOPCD_MODE_OFF:
            self._num_active_loopcd_reqs = max(0, self._num_active_loopcd_reqs - 1)
        self._req_id_to_slot[new_req_data.req_id] = req_index
        self._slot_modes[req_index] = p.mode_code
        if p.mode_code != LOOPCD_MODE_OFF:
            self._num_active_loopcd_reqs += 1

        if p.enabled:
            self.req_alpha.stage_write(req_index, 0, [p.alpha])
            self.req_mode.stage_write_elem(req_index, p.mode_code)
            self.req_blend_blk.stage_write_elem(req_index, 1 if p.blend_blocks else 0)
            self.req_adaptive.stage_write_elem(req_index, 1 if p.adaptive else 0)
            self.req_log_beta.stage_write(req_index, 0, [p.log_beta])
            self._slot_dirty[req_index] = True
            self._has_staged_writes = True
        elif self._slot_dirty[req_index]:
            self.req_alpha.stage_write(req_index, 0, [0.0])
            self.req_mode.stage_write_elem(req_index, LOOPCD_MODE_OFF)
            self.req_blend_blk.stage_write_elem(req_index, 0)
            self.req_adaptive.stage_write_elem(req_index, 0)
            self.req_log_beta.stage_write(req_index, 0, [0.0])
            self._slot_dirty[req_index] = False
            self._has_staged_writes = True

        ws = self.lm.model.glue_ws
        if (
            p.enabled
            and p.mode == "logits"
            and p.weak_pass in (0, -1)
            and ws is not None
            and (
                self.model_config.enforce_eager
                or self.vllm_config.compilation_config.cudagraph_mode
                == CUDAGraphMode.NONE
            )
        ):
            ws.loopcd_weak_pass = p.weak_pass

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        slot = self._req_id_to_slot.pop(req_id, None)
        if slot is not None:
            if int(self._slot_modes[slot]) != LOOPCD_MODE_OFF:
                self._num_active_loopcd_reqs = max(0, self._num_active_loopcd_reqs - 1)
            self._slot_modes[slot] = LOOPCD_MODE_OFF

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        if not self._has_staged_writes:
            return
        self.req_alpha.apply_write()
        self.req_mode.apply_write()
        self.req_blend_blk.apply_write()
        self.req_adaptive.apply_write()
        self.req_log_beta.apply_write()
        self._has_staged_writes = False

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        ws = self.lm.model.glue_ws
        if ws is not None and ws.has_loop:
            assert ws.loopcd_omega is not None
            assert ws.loopcd_omega_blk is not None
            assert ws.loopcd_logits_flag is not None
            ws.loopcd_omega[:num_tokens].zero_()
            ws.loopcd_omega_blk[:num_tokens].zero_()
            ws.loopcd_logits_flag.zero_()
            ws.loopcd_hidden_active = False
            ws.loopcd_logits_active = False
            ws.loopcd_logits_indices = None
        self._was_hidden_active = False
        self._was_logits_active = False
        self._cur_input_batch = None
        return super().prepare_dummy_inputs(num_reqs, num_tokens)

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        out = super().prepare_inputs(input_batch, req_states)
        ws = self.lm.model.glue_ws
        if ws is None or not ws.has_loop:
            return out

        if (
            self._num_active_loopcd_reqs == 0
            and not self._was_hidden_active
            and not self._was_logits_active
        ):
            ws.loopcd_hidden_active = False
            ws.loopcd_logits_active = False
            ws.loopcd_logits_indices = None
            self._cur_input_batch = None
            return out

        batch_modes = self._slot_modes[input_batch.idx_mapping_np]
        has_hidden = bool(np.any(batch_modes == LOOPCD_MODE_HIDDEN))
        has_logits = bool(np.any(batch_modes == LOOPCD_MODE_LOGITS))
        ws.loopcd_hidden_active = has_hidden
        ws.loopcd_logits_active = has_logits

        assert ws.loopcd_omega is not None
        assert ws.loopcd_omega_blk is not None
        assert ws.loopcd_logits_flag is not None

        if has_hidden:
            ops.inkling2_loopcd_prepare_hidden(
                ws.loopcd_omega,
                ws.loopcd_omega_blk,
                input_batch.idx_mapping,
                input_batch.query_start_loc,
                input_batch.seq_lens,
                req_states.prefill_len.gpu,
                self.req_alpha.gpu,
                self.req_mode.gpu,
                self.req_blend_blk.gpu,
                input_batch.num_tokens_after_padding,
            )
            self._was_hidden_active = True
        elif self._was_hidden_active:
            ws.loopcd_omega.zero_()
            ws.loopcd_omega_blk.zero_()
            self._was_hidden_active = False

        if has_logits:
            ws.loopcd_logits_flag.fill_(1)
            ws.loopcd_logits_indices = input_batch.logits_indices
            self._was_logits_active = True
            self._cur_input_batch = input_batch
        else:
            if self._was_logits_active:
                ws.loopcd_logits_flag.zero_()
                self._was_logits_active = False
            ws.loopcd_logits_indices = None
            self._cur_input_batch = None

        return out

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ib = self._cur_input_batch
        self._cur_input_batch = None
        ws = self.lm.model.glue_ws
        if ib is None or ws is None or not ws.has_loop or not ws.loopcd_logits_active:
            z = self.lm._raw_compute_logits(hidden_states)
            assert z is not None
            return z

        z_final = self.lm._raw_compute_logits(hidden_states)
        assert z_final is not None
        M = hidden_states.shape[0]
        if M == 0:
            return z_final

        assert ws.loopcd_weak_h is not None
        weak_h = (
            ws.loopcd_weak_h[:M]
            if ws.loopcd_weak_is_compact
            else ws.loopcd_weak_h.index_select(0, ib.logits_indices)
        )
        z_first = self.lm._raw_compute_logits(weak_h)
        assert z_first is not None

        exp_idx = ib.expanded_idx_mapping
        modes = self.req_mode.gpu.index_select(0, exp_idx)
        alphas = self.req_alpha.gpu.index_select(0, exp_idx)
        row_omega = torch.where(
            modes == LOOPCD_MODE_LOGITS, alphas, torch.zeros_like(alphas)
        )
        row_log_beta = self.req_log_beta.gpu.index_select(0, exp_idx)
        row_adaptive = self.req_adaptive.gpu.index_select(0, exp_idx)

        if self._logits_stats_scratch.shape[0] < M:
            self._logits_stats_scratch = torch.empty(
                (M, 32, 2), dtype=torch.float32, device=z_final.device
            )
        if z_final.stride(-1) != 1:
            z_final = z_final.contiguous()
        if z_first.stride(-1) != 1:
            z_first = z_first.contiguous()
        ops.inkling2_loopcd_logits(
            z_final,
            z_first,
            row_omega,
            row_log_beta,
            row_adaptive,
            stats_scratch=self._logits_stats_scratch[:M],
        )
        return z_final


__all__ = [
    "LOOPCD_MAX_WEAK_ROWS",
    "LOOPCD_MODE_HIDDEN",
    "LOOPCD_MODE_LOGITS",
    "LOOPCD_MODE_OFF",
    "Inkling2ModelState",
    "LoopCDParams",
    "loopcd_blend_reference",
    "loopcd_logits_reference",
    "parse_loopcd_params",
    "read_loopcd_env_defaults",
]
