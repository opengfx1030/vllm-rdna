# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLiNER2.5 decision backend.

The package is imported inside :meth:`Gliner2Backend.load`. Importing vLLM
does not import ``gliner2`` and does not download weights.

``classify_text`` returns a winning label, not a full distribution. System One
needs a probability for every option, so scoring goes through
``gliner2.classification.Classifier.batch_classify``, which exposes
per-label probabilities. Requests that share a schema are scored together.
Different schemas in one micro-batch are grouped, then scored separately.

The model is moved onto a dedicated torch stream when the device is
``cuda:N``. That stream is never the default stream, and work is refused
if CUDA graph capture is active on the calling thread.
"""

import json
from argparse import Namespace
from collections.abc import Callable, Sequence

from vllm.entrypoints.systemone.devices import (
    assert_free_vram,
    assert_spare_device,
    open_decision_stream,
    parse_device,
    resolve_dtype,
)
from vllm.entrypoints.systemone.errors import (
    SystemOneError,
    SystemOneStartupError,
)
from vllm.entrypoints.systemone.protocol import (
    DecisionRequest,
    TaskPlan,
    build_local_response,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# GLiNER2.5-multi-Decide encodes at most 4096 tokens.
_MAX_LEN = 4096


class Gliner2Backend:
    """In-process GLiNER2.5 multi-decide model."""

    def __init__(
        self,
        *,
        model: str,
        device: str,
        dtype: str | None,
        vram_reserve_gb: float,
        engine_args: Namespace | None,
        check_engine_devices: bool,
    ) -> None:
        self._model = model
        self._device = device
        self._dtype = dtype
        self._vram_reserve_gb = vram_reserve_gb
        self._engine_args = engine_args
        self._check_engine_devices = check_engine_devices
        self._device_kind = "cpu"
        self._index: int | None = None
        self._stream = None
        self._clf = None
        self._loaded = False

    def load(self) -> None:
        kind, index = parse_device(self._device)
        self._device_kind = kind
        self._index = index
        dtype_name = resolve_dtype(kind, self._dtype)
        if kind == "cuda":
            if self._check_engine_devices:
                if self._engine_args is None:
                    raise SystemOneStartupError(
                        "systemone cuda device check requires engine args. "
                        "Refusing to start."
                    )
                assert_spare_device(self._device, self._engine_args)
            assert index is not None
            assert_free_vram(index, self._vram_reserve_gb)
            self._stream = open_decision_stream(index)
        auto_extractor, classifier_cls = _import_gliner()
        import torch

        torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
        logger.info(
            "Loading systemone model %s on %s (%s)",
            self._model,
            self._device,
            dtype_name,
        )
        extractor = auto_extractor.from_pretrained(self._model, map_location="cpu")
        self._clf = classifier_cls(extractor, device=self._device, dtype=torch_dtype)

        def _place() -> None:
            self._clf.to(device=self._device, dtype=torch_dtype)
            self._clf.eval()

        if kind == "cuda":
            self._guard_and_run(_place)
        else:
            _place()
        self._loaded = True

    def decide_batch(self, requests: list[DecisionRequest]) -> list[dict]:
        if not self._loaded:
            raise SystemOneError("systemone model is not loaded", 503)
        return self._guard_and_run(lambda: self._decide_unlocked(requests))

    def _decide_unlocked(self, requests: Sequence[DecisionRequest]) -> list[dict]:
        groups: dict[str, list[tuple[int, DecisionRequest]]] = {}
        order: list[str] = []
        for index, request in enumerate(requests):
            key = _fingerprint(request.tasks)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((index, request))
        answers: list[dict | None] = [None] * len(requests)
        for key in order:
            members = groups[key]
            texts = [item.state_text for _index, item in members]
            tasks = members[0][1].tasks
            rows = self._classify_texts(texts, tasks)
            if len(rows) != len(members):
                raise SystemOneError(
                    "systemone classifier returned the wrong batch size", 500
                )
            for (index, request), probs in zip(members, rows):
                answers[index] = build_local_response(
                    request.model, request.tasks, probs
                )
        return [answer for answer in answers if answer is not None]

    def _classify_texts(
        self, texts: Sequence[str], tasks: tuple[TaskPlan, ...]
    ) -> list[dict[str, dict[str, float]]]:
        schema, config = _build_schema(tasks, batch_size=len(texts))
        results = self._clf.batch_classify(list(texts), schema, config=config)
        rows: list[dict[str, dict[str, float]]] = []
        for result in results:
            row: dict[str, dict[str, float]] = {}
            for task in tasks:
                raw = dict(result.probabilities(task.name))
                row[task.name] = {label: float(raw[label]) for label in task.labels}
            rows.append(row)
        return rows

    def _guard_and_run(self, fn: Callable[[], object]) -> object:
        if self._device_kind != "cuda":
            return fn()
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise SystemOneError(
                "systemone refuses to run while CUDA graph capture is active",
                500,
            )
        if self._stream is None:
            raise SystemOneStartupError(
                "systemone CUDA stream was not created. Refusing to run."
            )
        with torch.cuda.stream(self._stream):
            result = fn()
            self._stream.synchronize()
            return result


def _import_gliner():
    try:
        from gliner2 import AutoExtractor
        from gliner2.classification import Classifier
    except ImportError as exc:
        raise ImportError(
            "systemone backend 'gliner2' requires the optional gliner2 "
            "package. Install it with `uv pip install gliner2`. "
            "Importing vLLM does not require gliner2."
        ) from exc
    return AutoExtractor, Classifier


def _build_schema(tasks: tuple[TaskPlan, ...], *, batch_size: int):
    from gliner2.classification import (
        ClassificationConfig,
        ClassificationSchema,
        SchemaError,
    )

    schema = ClassificationSchema()
    for task in tasks:
        labels = task.schema_labels()
        kwargs: dict = {}
        if task.instruction:
            kwargs["instruction"] = task.instruction
        if task.threshold is not None:
            kwargs["threshold"] = task.threshold
        try:
            if task.kind == "score":
                schema.ordinal(task.name, labels, **kwargs)
            else:
                schema.single(task.name, labels, **kwargs)
        except SchemaError as exc:
            raise SystemOneError(str(exc), 400) from exc
    config = ClassificationConfig(
        decoder="independent",
        include_confidence=True,
        batch_size=max(1, batch_size),
        max_len=_MAX_LEN,
    )
    return schema, config


def _fingerprint(tasks: tuple[TaskPlan, ...]) -> str:
    payload = [
        {
            "name": task.name,
            "kind": task.kind,
            "labels": task.labels,
            "descriptions": task.descriptions,
            "instruction": task.instruction,
            "threshold": task.threshold,
        }
        for task in tasks
    ]
    return json.dumps(payload, separators=(",", ":"))
