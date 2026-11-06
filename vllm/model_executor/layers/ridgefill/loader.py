# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed check of a RidgeFill manifest against a checkpoint's pins.

The released Flash-Next maps record ``model.vocab_sha256`` and
``model.encodings`` for the MXFP4/FP8 checkpoint they were fitted on.
A W4A16, EXL3, or HI2 checkpoint will not match. This loader refuses the
bundle rather than applying the wrong map.
"""

from __future__ import annotations

import json
from pathlib import Path


class RidgeFillPinError(ValueError):
    """Manifest and checkpoint pins disagree, or the manifest is incomplete."""


def load_json(path: str | Path) -> dict:
    file = Path(path)
    if file.is_dir():
        file = file / "ridgefill.json"
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RidgeFillPinError(f"cannot read {file}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RidgeFillPinError(f"{file} is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RidgeFillPinError(f"{file} must be a JSON object")
    return data


def validate_pins(
    manifest: dict,
    *,
    vocab_sha256: str,
    encodings: dict[str, str],
) -> None:
    """Raise unless the manifest names this checkpoint.

    Args:
        manifest: Parsed ``ridgefill.json``.
        vocab_sha256: SHA-256 of the checkpoint tokenizer vocabulary.
        encodings: Per-tensor encoding strings for that checkpoint, in the
            same naming scheme as the manifest.

    Raises:
        RidgeFillPinError: A pin is missing or different. The message names
            the first mismatch.

    """
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise RidgeFillPinError("manifest has no model object")
    got_vocab = model.get("vocab_sha256")
    if not isinstance(got_vocab, str) or not got_vocab:
        raise RidgeFillPinError("manifest model.vocab_sha256 is missing")
    if got_vocab != vocab_sha256:
        raise RidgeFillPinError(
            f"vocab sha256 mismatch: manifest {got_vocab} != checkpoint {vocab_sha256}"
        )
    got_enc = model.get("encodings")
    if not isinstance(got_enc, dict):
        raise RidgeFillPinError("manifest model.encodings is missing")
    if got_enc == encodings:
        return
    keys = sorted(set(got_enc) | set(encodings))
    for key in keys:
        left = got_enc.get(key)
        right = encodings.get(key)
        if left != right:
            raise RidgeFillPinError(
                f"encoding mismatch for {key}: manifest {left!r} != "
                f"checkpoint {right!r}"
            )
    raise RidgeFillPinError("encodings mismatch")


def require_bundle(map_path: str, pins_path: str) -> None:
    """Validate ``map_path`` against ``pins_path``. Both must exist.

    ``pins_path`` is a JSON object with ``vocab_sha256`` and ``encodings``,
    produced for the checkpoint that will be served. The released MXFP4
    manifest fails this check on a W4A16, EXL3, or HI2 pin file.

    Raises:
        RidgeFillPinError: Either file is missing, or the pins disagree.

    """
    if not map_path or not pins_path:
        raise RidgeFillPinError(
            "VLLM_RDNA_RIDGEFILL_MAP and VLLM_RDNA_RIDGEFILL_PINS are both required "
            "before a bundle is applied"
        )
    manifest = load_json(map_path)
    pins = load_json(pins_path)
    vocab = pins.get("vocab_sha256")
    encodings = pins.get("encodings")
    if not isinstance(vocab, str) or not isinstance(encodings, dict):
        raise RidgeFillPinError("pins file needs vocab_sha256 and encodings")
    validate_pins(manifest, vocab_sha256=vocab, encodings=encodings)
