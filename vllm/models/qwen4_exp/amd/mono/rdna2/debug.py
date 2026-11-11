# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-serve timing (``rocm_mono_decode`` ``timing``; ``modules`` for the
today's-modules reference under the same marks).

One-thread timestamp kernels (``_rocm_C::mono_k2_rdna2_mark``) bracket the FFN
block + its all-reduce (and the HC preamble, the RowParallelLinear all-reduce)
inside captured graphs and turn on K2's per-width phase / entry stats in the
workspace tail; the per-width means are logged on eager steps.
"""

import time

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .layout import MAX_ROWS, mode

logger = init_logger(__name__)

STATS: dict[int, torch.Tensor] = {}
STAT_COLS = 36
STAT_ROWS = MAX_ROWS + 1
STAT_WORDS = (STAT_ROWS + 1) * STAT_COLS
_TICKS_PER_US = 100.0  # wall_clock64 runs at 100 MHz on V620
_last_log = [0.0]
_WRAPPED = [False]


def _corr(n, sx, sy, sxx, syy, sxy) -> float:
    """Pearson correlation from running sums (0 if degenerate)."""
    if n < 2:
        return 0.0
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    if vx <= 0 or vy <= 0:
        return 0.0
    return (sxy - sx * sy / n) / (vx * vy) ** 0.5


def log_stats():
    """Per-width means (us) of the serve timing, all devices of this process."""
    now = time.monotonic()
    if now - _last_log[0] < 30.0:
        return
    _last_log[0] = now
    for idx, st in STATS.items():
        a = st.view(-1, STAT_COLS).cpu()
        for w in range(1, STAT_ROWS):
            r = a[w].tolist()
            n, nm, ng = r[0], r[12], r[16]
            if n == 0 and nm == 0:
                continue
            us = lambda v, k: v / k / _TICKS_PER_US if k else 0.0  # noqa: E731
            ph = "/".join(f"{us(r[i], n):.1f}" for i in range(1, 8))
            logger.info(
                "mono k2_rdna2 timing dev%d w%d: region n=%d %.1f us | kernel n=%d "
                "total %.1f phases %s | entry spread %.1f wg0 %.1f | mark->first %.1f "
                "mark->last %.1f | rest n=%d %.1f us | attn n=%d %.1f us (hc %.1f) | "
                "linear AR n=%d %.1f us | core qsa n=%d %.1f gdn n=%d %.1f | "
                "sclk attn %.0f ffn %.0f MHz | load pairs %.2f "
                "corr(pairs, kernel) %.2f",
                idx,
                w,
                nm,
                us(r[13], nm),
                n,
                us(r[9], n),
                ph,
                us(r[8], n),
                us(r[10], n),
                us(r[14], n),
                us(r[15], n),
                ng,
                us(r[17], ng),
                r[18],
                us(r[19], r[18]),
                us(r[21], r[20]),
                r[22],
                us(r[23], r[22]),
                r[24],
                us(r[25], r[24]),
                r[26],
                us(r[27], r[26]),
                100.0 * r[28] / r[29] if r[29] else 0.0,
                100.0 * r[30] / r[31] if r[31] else 0.0,
                r[32] / n if n else 0.0,
                _corr(n, r[32], r[9], r[33], r[35], r[34]),
            )


def mark(kind: int, x: torch.Tensor, qsa: bool = False) -> None:
    """Serve timing mark (see mono_mark_kernel); captured graphs only."""
    m = x.shape[0]
    st = STATS.get(x.device.index)
    if st is not None and 0 < m <= MAX_ROWS and mode() != "eager":
        torch.ops._rocm_C.mono_k2_rdna2_mark(st, kind, m | (32 if qsa else 0))


def _tmark(x: torch.Tensor, kind: int) -> torch.Tensor:
    """A mark kernel, returned as a copy so compile keeps the op."""
    mark(kind, x)
    return x.clone()


def _tmark_fake(x: torch.Tensor, kind: int) -> torch.Tensor:
    return torch.empty_like(x)


def wrap_linear_all_reduce() -> None:
    """Marks 6/7 around RowParallelLinear's all-reduce (the attention o_proj /
    GDN out_proj), in both the mono and the module arm."""
    if _WRAPPED[0]:
        return
    _WRAPPED[0] = True
    import vllm.model_executor.layers.linear as lin

    orig = lin.tensor_model_parallel_all_reduce
    tmark = torch.ops.vllm.qwen4_exp_mono_mark_rdna2

    def timed_all_reduce(t):
        return tmark(orig(tmark(t, 6)), 7)

    lin.tensor_model_parallel_all_reduce = timed_all_reduce


direct_register_custom_op(
    op_name="qwen4_exp_mono_mark_rdna2",
    op_func=_tmark,
    fake_impl=_tmark_fake,
)
