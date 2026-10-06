# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve-side System One settings.

CLI flags win over ``VLLM_SYSTEMONE_*`` environment variables. The client
library's ``SYSTEM_ONE_*`` variables are a different process and are not
read here. Nothing in a request body can turn the route on.
"""

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from vllm.entrypoints.systemone.devices import parse_device, resolve_dtype

_ENV_PREFIX = "VLLM_SYSTEMONE_"


@dataclass(frozen=True)
class SystemOneConfig:
    """Resolved decision-route settings. ``backend='off'`` is the default."""

    backend: str
    model: str | None
    device: str
    max_batch: int
    max_wait_ms: float
    url: str | None
    vram_reserve_gb: float
    max_queue: int
    timeout_s: float
    dtype: str | None
    api_key: str | None


def add_systemone_cli_args(parser: ArgumentParser) -> ArgumentParser:
    """Register ``--systemone-*`` flags. Defaults stay unset so env can apply."""
    group = parser.add_argument_group(
        "System One",
        "Opt-in decision route. Off by default. The model runs in the "
        "API-server process, not in EngineCore.",
    )
    group.add_argument(
        "--systemone-backend",
        default=None,
        choices=["off", "gliner2", "http", "stub"],
        help="Decision backend. Default: off "
        "(env VLLM_SYSTEMONE_BACKEND). gliner2 loads GLiNER2 in this "
        "process. http proxies to --systemone-url. stub is for tests.",
    )
    group.add_argument(
        "--systemone-model",
        default=None,
        help="Decision model id. Required for gliner2. Env: VLLM_SYSTEMONE_MODEL.",
    )
    group.add_argument(
        "--systemone-device",
        default=None,
        help="gliner2 device: cpu (default) or cuda:N on a spare GPU. "
        "Never a tensor-parallel engine device. Env: VLLM_SYSTEMONE_DEVICE.",
    )
    group.add_argument(
        "--systemone-max-batch",
        type=int,
        default=None,
        help="Micro-batch size. Default 8. Env: VLLM_SYSTEMONE_MAX_BATCH.",
    )
    group.add_argument(
        "--systemone-max-wait-ms",
        type=float,
        default=None,
        help="How long to wait to fill a micro-batch. Default 2. "
        "Env: VLLM_SYSTEMONE_MAX_WAIT_MS.",
    )
    group.add_argument(
        "--systemone-url",
        default=None,
        help="Upstream System One URL for the http backend. A bare origin "
        "gets /v1/systemone appended. Env: VLLM_SYSTEMONE_URL.",
    )
    group.add_argument(
        "--systemone-vram-reserve-gb",
        type=float,
        default=None,
        help="Minimum free GiB on a spare cuda device before load. "
        "Default 2. Env: VLLM_SYSTEMONE_VRAM_RESERVE_GB.",
    )
    group.add_argument(
        "--systemone-max-queue",
        type=int,
        default=None,
        help="Queued decision requests before 429. Default 128. "
        "Env: VLLM_SYSTEMONE_MAX_QUEUE.",
    )
    group.add_argument(
        "--systemone-timeout-s",
        type=float,
        default=None,
        help="Per-request timeout in seconds. Default 30. "
        "Env: VLLM_SYSTEMONE_TIMEOUT_S.",
    )
    group.add_argument(
        "--systemone-dtype",
        default=None,
        help="float16, float32, or auto. CUDA defaults to float16, CPU to "
        "float32. bfloat16 is rejected. Env: VLLM_SYSTEMONE_DTYPE.",
    )
    group.add_argument(
        "--systemone-api-key",
        default=None,
        help="Bearer token for the http backend only. Env: VLLM_SYSTEMONE_API_KEY.",
    )
    return parser


def resolve_systemone_config(args: Namespace) -> SystemOneConfig:
    """Merge CLI and ``VLLM_SYSTEMONE_*`` into one config."""
    backend = _pick_str(args, "systemone_backend", "BACKEND", "off").strip().lower()
    model = _pick_optional(args, "systemone_model", "MODEL")
    if model is None and backend == "stub":
        model = "stub"
    url = _pick_optional(args, "systemone_url", "URL")
    # A bad URL must not break the default-off path. Validate it only when
    # the http backend is actually selected.
    if backend == "http" and url:
        url = normalize_upstream_url(url)
    return SystemOneConfig(
        backend=backend,
        model=model,
        device=_pick_str(args, "systemone_device", "DEVICE", "cpu"),
        max_batch=_pick_int(args, "systemone_max_batch", "MAX_BATCH", 8),
        max_wait_ms=_pick_float(args, "systemone_max_wait_ms", "MAX_WAIT_MS", 2.0),
        url=url,
        vram_reserve_gb=_pick_float(
            args, "systemone_vram_reserve_gb", "VRAM_RESERVE_GB", 2.0
        ),
        max_queue=_pick_int(args, "systemone_max_queue", "MAX_QUEUE", 128),
        timeout_s=_pick_float(args, "systemone_timeout_s", "TIMEOUT_S", 30.0),
        dtype=_pick_optional(args, "systemone_dtype", "DTYPE"),
        api_key=_pick_optional(args, "systemone_api_key", "API_KEY"),
    )


def systemone_enabled(args: Namespace) -> bool:
    """True only when the serve-side backend is not ``off``."""
    return resolve_systemone_config(args).backend != "off"


def validate_systemone_args(args: Namespace) -> None:
    """Reject impossible serve settings before the engine starts.

    Raises:
        ValueError: The selected backend is missing a required setting.
    """
    if not hasattr(args, "systemone_backend") and not os.environ.get(
        "VLLM_SYSTEMONE_BACKEND"
    ):
        return
    cfg = resolve_systemone_config(args)
    if cfg.backend == "off":
        return
    if cfg.backend not in {"off", "gliner2", "http", "stub"}:
        raise ValueError(
            "--systemone-backend must be off, gliner2, http, or stub. "
            f"Got {cfg.backend!r}."
        )
    if cfg.backend == "gliner2" and not cfg.model:
        raise ValueError(
            "--systemone-model is required when --systemone-backend=gliner2"
        )
    if cfg.backend == "http" and not cfg.url:
        raise ValueError("--systemone-url is required when --systemone-backend=http")
    if cfg.max_batch < 1:
        raise ValueError("--systemone-max-batch must be >= 1")
    if cfg.max_wait_ms < 0:
        raise ValueError("--systemone-max-wait-ms must be >= 0")
    if cfg.max_queue < 1:
        raise ValueError("--systemone-max-queue must be >= 1")
    if cfg.timeout_s <= 0:
        raise ValueError("--systemone-timeout-s must be > 0")
    if cfg.vram_reserve_gb < 0:
        raise ValueError("--systemone-vram-reserve-gb must be >= 0")
    if cfg.backend == "gliner2":
        kind, _index = parse_device(cfg.device)
        resolve_dtype(kind, cfg.dtype)


def normalize_upstream_url(url: str) -> str:
    """Require http(s), and append ``/v1/systemone`` to a bare origin."""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"--systemone-url must be an http or https URL. Got {url!r}.")
    path = parsed.path or ""
    if path in {"", "/"}:
        path = "/v1/systemone"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _pick_str(args: Namespace, attr: str, env_suffix: str, default: str) -> str:
    value = _pick_optional(args, attr, env_suffix)
    return default if value is None else value


def _pick_optional(args: Namespace, attr: str, env_suffix: str) -> str | None:
    cli = getattr(args, attr, None)
    if cli is not None:
        return str(cli)
    env = os.environ.get(_ENV_PREFIX + env_suffix)
    if env is None or env == "":
        return None
    return env


def _pick_int(args: Namespace, attr: str, env_suffix: str, default: int) -> int:
    cli = getattr(args, attr, None)
    if cli is not None:
        return int(cli)
    env = os.environ.get(_ENV_PREFIX + env_suffix)
    if env is None or env == "":
        return default
    try:
        return int(env)
    except ValueError as exc:
        raise ValueError(
            f"{_ENV_PREFIX}{env_suffix} must be an integer. Got {env!r}."
        ) from exc


def _pick_float(args: Namespace, attr: str, env_suffix: str, default: float) -> float:
    cli = getattr(args, attr, None)
    if cli is not None:
        return float(cli)
    env = os.environ.get(_ENV_PREFIX + env_suffix)
    if env is None or env == "":
        return default
    try:
        return float(env)
    except ValueError as exc:
        raise ValueError(
            f"{_ENV_PREFIX}{env_suffix} must be a number. Got {env!r}."
        ) from exc
