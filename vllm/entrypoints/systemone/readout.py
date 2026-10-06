# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request check that labels encode as the loaded decision model expects.

SGLang refuses a decision request with 400 when the served tokenizer's ids
for the answer codes differ from the checkpoint readout
(sgl-project/sglang#42183). This is the same check for an encoder model:
empty encodings, structural token ids, and a fixed ``decision_config``
readout are client errors, not silent rescoring.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.systemone.errors import SystemOneError
from vllm.entrypoints.systemone.protocol import TaskPlan

_STRUCTURAL = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
)

_MISMATCH = (
    "the served tokenizer does not encode the answer codes as "
    "the token ids of the checkpoint readout"
)


@dataclass(frozen=True)
class LoadedReadout:
    """Tokenizer and optional fixed label ids captured at model load."""

    tokenizer: Any
    special_ids: frozenset[int]
    expected: Mapping[str, tuple[int, ...]] | None = None


def capture_readout(model: Any) -> LoadedReadout | None:
    """Read the tokenizer hanging off a loaded extractor or classifier.

    Returns None when the object has no tokenizer, so a test double that
    does not load weights is not forced through this check.
    """
    tokenizer = find_tokenizer(model)
    if tokenizer is None:
        return None
    return LoadedReadout(
        tokenizer=tokenizer,
        special_ids=structural_token_ids(tokenizer),
        expected=find_expected_label_ids(model),
    )


def assert_labels_match(tasks: Sequence[TaskPlan], readout: LoadedReadout) -> None:
    """Refuse the request when label ids do not match ``readout``.

    Raises:
        SystemOneError: HTTP 400, one question at a time.
    """
    for task in tasks:
        for label in task.labels:
            ids = encode_label(readout.tokenizer, label)
            if not ids:
                raise SystemOneError(
                    f"question {task.name!r}: the served tokenizer does not "
                    f"encode label {label!r}",
                    400,
                )
            if readout.special_ids.intersection(ids):
                raise SystemOneError(
                    f"question {task.name!r}: label {label!r} encodes as a "
                    "structural token of the loaded decision model",
                    400,
                )
            if readout.expected is None:
                continue
            expected = readout.expected.get(label)
            if expected is None or tuple(ids) != tuple(expected):
                raise SystemOneError(
                    f"question {task.name!r}: {_MISMATCH}",
                    400,
                )


def find_tokenizer(model: Any) -> Any | None:
    """Depth-first search for an object with ``encode``."""
    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        tokenizer = getattr(obj, "tokenizer", None)
        if tokenizer is not None and callable(getattr(tokenizer, "encode", None)):
            return tokenizer
        for name in ("processor", "model", "extractor", "data_processor"):
            child = getattr(obj, name, None)
            if child is not None and child is not obj:
                stack.append(child)
    return None


def find_expected_label_ids(model: Any) -> dict[str, tuple[int, ...]] | None:
    """Fixed readout from ``decision_config`` (``codes`` + ``token_ids``)."""
    seen: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen or isinstance(obj, dict | str | bytes):
            continue
        seen.add(id(obj))
        for attr in ("decision_config", "config"):
            found = _ids_from_config(getattr(obj, attr, None))
            if found:
                return found
        for name in ("processor", "model", "extractor", "data_processor", "config"):
            child = getattr(obj, name, None)
            if child is not None and child is not obj:
                stack.append(child)
    return None


def structural_token_ids(tokenizer: Any) -> frozenset[int]:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return frozenset()
    ids: set[int] = set()
    for token in _STRUCTURAL:
        try:
            raw = convert(token)
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(raw, int) and raw >= 0:
            ids.add(raw)
    return frozenset(ids)


def encode_label(tokenizer: Any, label: str) -> tuple[int, ...]:
    raw = tokenizer.encode(label, add_special_tokens=False)
    if raw is None:
        return ()
    return tuple(int(item) for item in raw)


def _ids_from_config(config: Any) -> dict[str, tuple[int, ...]] | None:
    if not isinstance(config, dict):
        return None
    codes = config.get("codes")
    token_ids = config.get("token_ids")
    if not isinstance(codes, list | tuple) or not isinstance(token_ids, list | tuple):
        return None
    if len(codes) != len(token_ids) or not codes:
        return None
    return {str(code): (int(token_id),) for code, token_id in zip(codes, token_ids)}
