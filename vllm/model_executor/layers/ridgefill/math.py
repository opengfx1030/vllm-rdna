# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-form ridge fit and fp16 range check. Standard library only.

The fit is the centered normal equation from the paper (section 2.2)::

    A = Yc^T Xc (Xc^T Xc + alpha I)^{-1}
    alpha = lambda * tr(Xc^T Xc) / din
    b = mu_y - A mu_x

``X`` is the split-layer state and ``Y`` is the later layer's input, both
with the row mean removed. Applying the map is ``y = x A^T + b``. Products
stay in fp32. fp16 is a storage decision made after the range check, not the
type the solve runs in.

Nothing here was copied from ridgefill-data. The formula is the published one.
"""

from __future__ import annotations

from dataclasses import dataclass

# Largest finite fp16 value. A larger magnitude overflows at the cast.
FP16_MAX = 65504.0


@dataclass(frozen=True)
class ApplyDtypes:
    """Dtypes chosen after an fp32 range check of one apply.

    ``input_dtype`` and ``output_dtype`` are per tensor. ``gemm_dtype`` is
    fp16 only when both fit; otherwise the GEMM is bf16 so neither side
    overflows inside the multiply. The input is still fp32 until the caller
    casts it.
    """

    input_dtype: str
    output_dtype: str
    gemm_dtype: str
    input_overflow: bool
    output_overflow: bool


def max_abs(rows: list[list[float]] | tuple[tuple[float, ...], ...]) -> float:
    """Largest absolute entry. Empty input is 0."""
    peak = 0.0
    for row in rows:
        for value in row:
            magnitude = abs(float(value))
            if magnitude > peak:
                peak = magnitude
    return peak


def choose_apply_dtypes(
    input_max: float, output_max: float, requested: str = "fp16"
) -> ApplyDtypes:
    """Pick storage after the fp32 check.

    Args:
        input_max: Max abs of the 10240-wide map input (pre-norm streams).
        output_max: Max abs of the 2560-wide prediction, computed in fp32.
        requested: ``fp16`` (default) or ``bf16`` (explicit fallback).

    Returns:
        Per-tensor dtypes. fp16 is used for a tensor only when its max abs
        is at most ``FP16_MAX``. If either tensor overflows, ``gemm_dtype``
        is bf16.

    Raises:
        ValueError: ``requested`` is not fp16 or bf16.
    """
    if requested not in ("fp16", "bf16"):
        raise ValueError(f"requested dtype must be fp16 or bf16, got {requested}")
    if requested == "bf16":
        return ApplyDtypes("bf16", "bf16", "bf16", False, False)
    input_over = input_max > FP16_MAX
    output_over = output_max > FP16_MAX
    in_dt = "bf16" if input_over else "fp16"
    out_dt = "bf16" if output_over else "fp16"
    gemm = "fp16" if in_dt == "fp16" and out_dt == "fp16" else "bf16"
    return ApplyDtypes(in_dt, out_dt, gemm, input_over, output_over)


def weight_storage(weights: list[list[float]], requested: str = "fp16") -> str:
    """Storage for a dequantized map. fp16 only when every entry fits."""
    if requested == "bf16" or max_abs(weights) > FP16_MAX:
        return "bf16"
    return "fp16"


def _transpose(rows: list[list[float]]) -> list[list[float]]:
    cols = len(rows[0])
    return [[row[col] for row in rows] for col in range(cols)]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    width = len(right[0])
    shared = len(right)
    out = []
    for row in left:
        acc = [0.0] * width
        for k in range(shared):
            scale = row[k]
            if scale == 0.0:
                continue
            right_row = right[k]
            for col in range(width):
                acc[col] += scale * right_row[col]
        out.append(acc)
    return out


def _add_diag(matrix: list[list[float]], value: float) -> None:
    for index in range(len(matrix)):
        matrix[index][index] += value


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("ridge Gram matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [entry / scale for entry in aug[col]]
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            aug[row] = [
                entry - factor * aug[col][col_index]
                for col_index, entry in enumerate(aug[row])
            ]
    return [row[size:] for row in aug]


def _mean_row(rows: list[list[float]]) -> list[float]:
    count = float(len(rows))
    width = len(rows[0])
    return [sum(row[col] for row in rows) / count for col in range(width)]


def fit_ridge(
    x_rows: list[list[float]] | tuple[tuple[float, ...], ...],
    y_rows: list[list[float]] | tuple[tuple[float, ...], ...],
    lam: float,
) -> tuple[list[list[float]], list[float]]:
    """Fit one map. Returns ``(A, b)`` with ``A`` shaped ``[dout, din]``.

    Args:
        x_rows: Split states, one row per calibration token.
        y_rows: That layer's true inputs, same row count.
        lam: The paper's lambda. ``alpha = lam * tr(G) / din`` on the
            centered Gram matrix ``G``.

    Returns:
        Weight ``A`` and bias ``b`` of ``y = x A^T + b``.

    Raises:
        ValueError: The two captures disagree in length, or ``G`` is singular.
    """
    xs = [list(map(float, row)) for row in x_rows]
    ys = [list(map(float, row)) for row in y_rows]
    if not xs or len(xs) != len(ys):
        raise ValueError("capture row counts must match and be non-empty")
    din = len(xs[0])
    if any(len(row) != din for row in xs):
        raise ValueError("split states must share din")
    dout = len(ys[0])
    if any(len(row) != dout for row in ys):
        raise ValueError("layer inputs must share dout")
    mu_x = _mean_row(xs)
    mu_y = _mean_row(ys)
    xc = [[value - mu_x[col] for col, value in enumerate(row)] for row in xs]
    yc = [[value - mu_y[col] for col, value in enumerate(row)] for row in ys]
    xt = _transpose(xc)
    gram = _matmul(xt, xc)
    trace = sum(gram[index][index] for index in range(din))
    alpha = float(lam) * trace / float(din)
    _add_diag(gram, alpha)
    # A = (Yc^T Xc) inv(G), and Yc^T Xc = (Xc^T Yc)^T.
    cross = _transpose(_matmul(xt, yc))
    weight = _matmul(cross, _invert(gram))
    bias = []
    for out in range(dout):
        acc = mu_y[out]
        row = weight[out]
        for col in range(din):
            acc -= row[col] * mu_x[col]
        bias.append(acc)
    return weight, bias


def apply_map(
    x_rows: list[list[float]] | tuple[tuple[float, ...], ...],
    weight: list[list[float]],
    bias: list[float],
) -> list[list[float]]:
    """``y = x A^T + b`` in fp32. Does not cast."""
    dout = len(weight)
    if len(bias) != dout:
        raise ValueError("bias width must match A rows")
    din = len(weight[0]) if weight else 0
    out = []
    for row in x_rows:
        if len(row) != din:
            raise ValueError("x width must match A columns")
        pred = []
        for j in range(dout):
            acc = bias[j]
            wrow = weight[j]
            for i in range(din):
                acc += float(row[i]) * wrow[i]
            pred.append(acc)
        out.append(pred)
    return out


def dequant_i8_row128(
    codes: list[list[int]],
    scales: list[list[float]],
    bias: list[float],
    group: int = 128,
) -> tuple[list[list[float]], list[float]]:
    """int8 row-group map to fp32 weights.

    Row ``i``, group ``g`` is ``codes[i, 128g:128g+128] * scales[i, g]``.
    This is the layout ``code/c5/convert.py`` documents for the Flash-Next
    release (``i8_row128``). The multiply is fp32.

    Raises:
        ValueError: Shapes are not a row-128 map.
    """
    if group <= 0:
        raise ValueError("group must be positive")
    dout = len(codes)
    if dout == 0 or len(bias) != dout or len(scales) != dout:
        raise ValueError("codes, scales and bias must share dout")
    din = len(codes[0])
    if din % group:
        raise ValueError("din must be a multiple of the group")
    groups = din // group
    weights: list[list[float]] = []
    for i in range(dout):
        if len(codes[i]) != din or len(scales[i]) != groups:
            raise ValueError(f"row {i} is not i8_row128")
        row: list[float] = []
        for g in range(groups):
            scale = float(scales[i][g])
            base = g * group
            for k in range(group):
                row.append(int(codes[i][base + k]) * scale)
        weights.append(row)
    return weights, [float(value) for value in bias]


def apply_checked(
    x_rows: list[list[float]],
    weight: list[list[float]],
    bias: list[float],
    requested: str = "fp16",
) -> tuple[list[list[float]], ApplyDtypes, float, float]:
    """fp32 apply plus the input and output range check.

    The returned predictions stay fp32. The caller casts each tensor to the
    dtype in the report, and only after this returns. A GPU GEMM uses
    ``gemm_dtype`` (fp16 with fp32 accumulate, or bf16 if either side
    overflows).

    Returns:
        Predictions, dtype decision, input max abs, output max abs.
    """
    input_peak = max_abs(x_rows)
    pred = apply_map(x_rows, weight, bias)
    output_peak = max_abs(pred)
    decision = choose_apply_dtypes(input_peak, output_peak, requested)
    return pred, decision, input_peak, output_peak
