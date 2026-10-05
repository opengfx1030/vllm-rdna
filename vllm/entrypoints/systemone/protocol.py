# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""System One ``POST /v1/systemone`` contract.

Supported question types are ``choice``, ``score``, and ``noul``. Field names
follow the wire shared by llama.cpp ``/v1/systemone``, SGLang's System One
models, and vLLM PR #59299 where those three agree. Disagreements are listed
in ``docs/rdna2/systemone.md``. A request cannot enable the feature; that
switch is serve-side only.

Class names that #59299 also uses (``QuestionSpec``, ``StructuredDecisionError``)
live here, not under ``vllm.entrypoints.generate.structured_decisions``, so a
rebase onto a tag that contains that PR is a delete of this package.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.systemone.errors import SystemOneError

MAX_BODY_BYTES = 65536
MAX_QUESTIONS = 32
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MAX_DEPTH = 32
MAX_NAME_CHARS = 128

_MISSING = object()


@dataclass(frozen=True)
class QuestionSpec:
    """One question object. Same field names as #59299's ``QuestionSpec``."""

    type: str
    instructions: Any = ""
    criteria: Any = None


@dataclass(frozen=True)
class TaskPlan:
    """One validated question, ready to score."""

    name: str
    kind: str
    labels: tuple[str, ...]
    descriptions: tuple[str | None, ...]
    instruction: str | None
    threshold: float | None

    def schema_labels(self) -> list[str] | dict[str, str]:
        if any(item is not None for item in self.descriptions):
            return {
                label: desc if desc is not None else label
                for label, desc in zip(self.labels, self.descriptions)
            }
        return list(self.labels)


@dataclass(frozen=True)
class DecisionRequest:
    """A validated decision request."""

    model: str
    state_text: str
    tasks: tuple[TaskPlan, ...]
    body: dict[str, Any]


def parse_request(raw: bytes, *, default_model: str | None) -> DecisionRequest:
    """Parse and validate a System One request body.

    Args:
        raw: Exact UTF-8 JSON body.
        default_model: Configured decision model, used when ``model`` is omitted.

    Returns:
        The validated request. ``body`` is the parsed JSON with ``model`` filled
        in, suitable for an HTTP proxy.

    Raises:
        SystemOneError: The body is not a valid System One request (HTTP 400).
    """
    if len(raw) > MAX_BODY_BYTES:
        raise SystemOneError(f"request body exceeds {MAX_BODY_BYTES} bytes", 400)
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_dupes)
    except UnicodeDecodeError as exc:
        raise SystemOneError("request body is not UTF-8", 400) from exc
    except json.JSONDecodeError as exc:
        raise SystemOneError(f"request body is not valid JSON: {exc.msg}", 400) from exc
    if not isinstance(payload, dict):
        raise SystemOneError("request body must be a JSON object", 400)
    _check_depth(payload)

    _refuse_foreign_fields(payload)
    if "state" not in payload:
        raise SystemOneError("state is required", 400)
    state = payload["state"]
    if not _valid_state(state):
        raise SystemOneError("state must be a string, object, array, or null", 400)
    questions = payload.get("questions", _MISSING)
    if not isinstance(questions, dict) or not questions:
        raise SystemOneError("questions must be an object with 1 to 32 entries", 400)
    if len(questions) > MAX_QUESTIONS:
        raise SystemOneError(f"questions accepts at most {MAX_QUESTIONS} entries", 400)

    model = _resolve_model(payload.get("model", _MISSING), default_model)
    tasks = tuple(_parse_question(name, spec) for name, spec in questions.items())
    body = dict(payload)
    body["model"] = model
    return DecisionRequest(
        model=model,
        state_text=state_to_text(state),
        tasks=tasks,
        body=body,
    )


def state_to_text(state: Any) -> str:
    """Render shared state as encoder text."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def build_local_response(
    model: str,
    tasks: tuple[TaskPlan, ...],
    probabilities: Mapping[str, Mapping[str, float]],
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, Any]:
    """Build a System One response from per-label probabilities.

    Choice confidence is how far the top option stands above a uniform guess.
    Score confidence is one minus the spread around the top level, relative
    to a uniform spread. Both match llama.cpp and SGLang. Noul is ``P(yes)``
    and has no confidence field. ``usage`` is always present; an encoder that
    does not count tokens reports zeros.

    Raises:
        SystemOneError: A required probability is missing or not in ``[0, 1]``.
    """
    answers: dict[str, Any] = {}
    for task in tasks:
        try:
            probs = probabilities[task.name]
        except KeyError as exc:
            raise SystemOneError(
                f"decision model omitted question {task.name!r}", 502
            ) from exc
        answers[task.name] = _answer(task, probs)
    return {
        "model": model,
        "answers": answers,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def stub_probabilities(task: TaskPlan) -> dict[str, float]:
    """Deterministic distribution for the stub backend. Not a model."""
    weights = {
        1: (1.0,),
        2: (0.2, 0.8),
        3: (0.1, 0.3, 0.6),
    }.get(len(task.labels))
    if weights is None:
        share = 1.0 / len(task.labels)
        weights = tuple(share for _ in task.labels)
    return {label: weight for label, weight in zip(task.labels, weights)}


def _answer(task: TaskPlan, probs: Mapping[str, float]) -> dict[str, Any]:
    cleaned: dict[str, float] = {}
    for label in task.labels:
        if label not in probs:
            raise SystemOneError(
                f"decision model omitted probability for {task.name!r} label {label!r}",
                502,
            )
        value = float(probs[label])
        if value != value or value < 0.0 or value > 1.0:
            raise SystemOneError(
                f"probability for {task.name!r} label {label!r} is outside [0, 1]",
                502,
            )
        cleaned[label] = value
    if task.kind == "noul":
        return {"type": "noul", "noul": cleaned["yes"]}
    total = sum(cleaned.values())
    if total <= 0.0:
        raise SystemOneError(
            f"decision model returned an empty distribution for {task.name!r}",
            502,
        )
    if task.kind == "choice":
        selected = max(
            task.labels,
            key=lambda label: (cleaned[label], -task.labels.index(label)),
        )
        return {
            "type": "choice",
            "choice": selected,
            "probabilities": cleaned,
            "confidence": choice_confidence([cleaned[label] for label in task.labels]),
        }
    ordered = [cleaned[label] for label in task.labels]
    score = sum(index * value for index, value in enumerate(ordered))
    legend = {
        label: desc if desc is not None else label
        for label, desc in zip(task.labels, task.descriptions)
    }
    return {
        "type": "score",
        "score": score,
        "legend": legend,
        "probabilities": cleaned,
        "confidence": score_confidence(ordered),
    }


def choice_confidence(probs: Sequence[float]) -> float:
    """TypeSafe choice confidence used by llama.cpp and SGLang.

    ``(n * p_max - 1) / (n - 1)``, clipped to ``[0, 1]``. One option is 1.
    For two options this equals the top-two margin.
    """
    n = len(probs)
    if n < 2:
        return 1.0
    return min(1.0, max(0.0, (n * max(probs) - 1.0) / (n - 1)))


def score_confidence(probs: Sequence[float]) -> float:
    """TypeSafe score confidence used by llama.cpp and SGLang.

    One minus the probability-weighted distance to the mode, divided by the
    same distance under a uniform distribution. Clipped at 0. One level is 1.
    """
    n = len(probs)
    if n < 2:
        return 1.0
    top = max(range(n), key=lambda index: probs[index])
    spread = math.fsum(value * abs(index - top) for index, value in enumerate(probs))
    uniform = math.fsum(abs(index - (n - 1) / 2.0) for index in range(n)) / n
    if uniform <= 0.0:
        return 1.0
    return max(0.0, 1.0 - spread / uniform)


def _resolve_model(supplied: Any, default_model: str | None) -> str:
    if supplied is _MISSING or supplied is None:
        if not default_model:
            raise SystemOneError("model is required", 400)
        return _check_model(default_model, field="configured model")
    model = _check_model(supplied, field="model")
    if default_model and model != default_model:
        raise SystemOneError(
            f"model {model!r} is not the configured systemone model {default_model!r}",
            400,
        )
    return model


def _parse_question(name: Any, spec: Any) -> TaskPlan:
    if not isinstance(name, str):
        raise SystemOneError("question names must be strings", 400)
    _check_name(name, "question name")
    if not isinstance(spec, dict):
        raise SystemOneError(f"question {name!r} must be an object", 400)
    if spec.get("multi_label") is True:
        raise SystemOneError(
            f"question {name!r}: multi_label is not a System One answer. "
            "choice, score, and noul are single-valued",
            400,
        )
    if "options" in spec and "criteria" not in spec:
        raise SystemOneError(
            f"question {name!r} uses 'options'. The System One contract "
            "expects 'criteria'",
            400,
        )
    kind = spec.get("type")
    if kind not in ("choice", "score", "noul"):
        raise SystemOneError(
            f"question {name!r}: type must be 'choice', 'score', or 'noul'",
            400,
        )
    instruction = _parse_instructions(name, spec)
    threshold = _parse_threshold(name, spec)
    if kind == "choice":
        labels, descriptions = _parse_choice(name, spec)
    elif kind == "score":
        labels, descriptions = _parse_score(name, spec)
    else:
        labels, descriptions = _parse_noul(name, spec)
    if (
        kind == "noul"
        and _blank(instruction)
        and all(
            item is None or (isinstance(item, str) and not item.strip())
            for item in descriptions
        )
    ):
        raise SystemOneError(
            f"question {name!r}: a noul question needs instructions or a "
            "true or false description to decide on",
            400,
        )
    return TaskPlan(
        name=name,
        kind=kind,
        labels=labels,
        descriptions=descriptions,
        instruction=instruction,
        threshold=threshold,
    )


def _parse_choice(name: str, spec: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple]:
    if "criteria" not in spec:
        raise SystemOneError(f"question {name!r}: choice requires criteria", 400)
    criteria = spec["criteria"]
    if not isinstance(criteria, dict) or not criteria:
        raise SystemOneError(
            f"question {name!r}: choice criteria must be an object "
            f"with 1 to {MAX_CHOICE_OPTIONS} options",
            400,
        )
    if len(criteria) > MAX_CHOICE_OPTIONS:
        raise SystemOneError(
            f"question {name!r}: choice accepts at most {MAX_CHOICE_OPTIONS} options",
            400,
        )
    labels: list[str] = []
    descriptions: list[str | None] = []
    for opt, desc in criteria.items():
        if not isinstance(opt, str):
            raise SystemOneError(
                f"question {name!r}: option names must be strings", 400
            )
        _check_name(opt, f"question {name!r} option")
        labels.append(opt)
        descriptions.append(_description(desc))
    return tuple(labels), tuple(descriptions)


def _parse_score(name: str, spec: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple]:
    if "criteria" not in spec:
        raise SystemOneError(f"question {name!r}: score requires criteria", 400)
    criteria = spec["criteria"]
    if (
        not isinstance(criteria, list)
        or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS
    ):
        raise SystemOneError(
            f"question {name!r}: score criteria must be an array of "
            f"{MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} ordered levels",
            400,
        )
    labels = tuple(str(index) for index in range(len(criteria)))
    descriptions = tuple(_description(item) for item in criteria)
    return labels, descriptions


def _parse_noul(name: str, spec: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple]:
    if "criteria" not in spec or spec["criteria"] is None:
        return ("yes", "no"), (None, None)
    criteria = spec["criteria"]
    if not isinstance(criteria, dict):
        raise SystemOneError(
            f"question {name!r}: noul criteria must be omitted, null, "
            "or an object with optional 'true' and 'false' descriptions",
            400,
        )
    unknown = set(criteria) - {"true", "false"}
    if unknown:
        listed = ", ".join(sorted(repr(key) for key in unknown))
        raise SystemOneError(
            f"question {name!r}: noul criteria only accepts 'true' and "
            f"'false' (got {listed})",
            400,
        )
    yes = _description(criteria["true"]) if "true" in criteria else None
    no = _description(criteria["false"]) if "false" in criteria else None
    return ("yes", "no"), (yes, no)


def _parse_instructions(name: str, spec: Mapping[str, Any]) -> str | None:
    if "instructions" not in spec or spec["instructions"] is None:
        return None
    value = spec["instructions"]
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise SystemOneError(
        f"question {name!r}: instructions must be a string, object, array, or null",
        400,
    )


def _parse_threshold(name: str, spec: Mapping[str, Any]) -> float | None:
    has_cls = "cls_threshold" in spec
    has_thr = "threshold" in spec
    if not has_cls and not has_thr:
        return None
    if has_cls and has_thr and spec["cls_threshold"] != spec["threshold"]:
        raise SystemOneError(
            f"question {name!r}: cls_threshold and threshold disagree",
            400,
        )
    raw = spec["cls_threshold"] if has_cls else spec["threshold"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SystemOneError(
            f"question {name!r}: threshold must be a number in (0, 1)",
            400,
        )
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise SystemOneError(
            f"question {name!r}: threshold must be between 0 and 1, exclusive",
            400,
        )
    return value


def _description(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_REFUSED_FIELDS = (
    "temperature",
    "prompt_format_version",
    "return_prompt_token_ids",
)


def _refuse_foreign_fields(payload: Mapping[str, Any]) -> None:
    """Fields that would change the answer if they were silently ignored."""
    for field in _REFUSED_FIELDS:
        if field in payload and payload[field] is not None:
            raise SystemOneError(
                f"{field} is not part of this API, use /v1/decisions for it",
                400,
            )
    images = payload.get("images", _MISSING)
    if images is _MISSING or images is None or images == []:
        return
    raise SystemOneError("this decision model does not support image input", 501)


def _blank(value: str | None) -> bool:
    return value is None or not str(value).strip()


def _valid_state(state: Any) -> bool:
    if state is None or isinstance(state, str):
        return True
    return isinstance(state, dict | list)


def _check_model(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemOneError(f"{field} must be a non-blank string", 400)
    if len(value) > MAX_NAME_CHARS:
        raise SystemOneError(
            f"{field} must be at most {MAX_NAME_CHARS} characters", 400
        )
    if _has_control(value):
        raise SystemOneError(f"{field} must not contain control characters", 400)
    return value


def _check_name(name: str, what: str) -> None:
    if not name.strip():
        raise SystemOneError(f"{what} must contain a non-whitespace character", 400)
    if len(name) > MAX_NAME_CHARS:
        raise SystemOneError(f"{what} must be at most {MAX_NAME_CHARS} characters", 400)
    if _has_control(name):
        raise SystemOneError(f"{what} must not contain control characters", 400)


def _has_control(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def _reject_dupes(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise SystemOneError(f"duplicate JSON key {key!r}", 400)
        obj[key] = value
    return obj


# Same exception name as vllm-project/vllm#59299. Kept in this module so the
# generate/structured_decisions package can land later without an add/add
# conflict. Delete this package when that route is the one that serves.
StructuredDecisionError = SystemOneError


def _check_depth(value: Any, depth: int = 1) -> None:
    if depth > MAX_DEPTH:
        raise SystemOneError(f"JSON nesting depth exceeds {MAX_DEPTH}", 400)
    if isinstance(value, dict):
        for child in value.values():
            _check_depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_depth(child, depth + 1)
