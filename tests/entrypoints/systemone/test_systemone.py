# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""System One route: validation, batching, and backends. No model download."""

import asyncio
import json
import sys
import threading
import types
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from vllm.entrypoints.systemone.backends.gliner2 import Gliner2Backend
from vllm.entrypoints.systemone.backends.stub import StubBackend
from vllm.entrypoints.systemone.config import (
    add_systemone_cli_args,
    resolve_systemone_config,
    systemone_enabled,
    validate_systemone_args,
)
from vllm.entrypoints.systemone.devices import (
    assert_free_vram,
    assert_spare_device,
    engine_logical_devices,
    open_decision_stream,
    resolve_dtype,
)
from vllm.entrypoints.systemone.errors import (
    SystemOneError,
    SystemOneStartupError,
)
from vllm.entrypoints.systemone.protocol import (
    build_local_response,
    parse_request,
)
from vllm.entrypoints.systemone.server import build_systemone_app
from vllm.entrypoints.systemone.service import SystemOneService

_MODEL = "fastino/GLiNER2.5-multi-Decide"


def _args(**overrides) -> Namespace:
    base = {
        "systemone_backend": None,
        "systemone_model": None,
        "systemone_device": None,
        "systemone_max_batch": None,
        "systemone_max_wait_ms": None,
        "systemone_url": None,
        "systemone_vram_reserve_gb": None,
        "systemone_max_queue": None,
        "systemone_timeout_s": None,
        "systemone_dtype": None,
        "systemone_api_key": None,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "device_ids": None,
    }
    base.update(overrides)
    return Namespace(**base)


def _body(**questions) -> bytes:
    return json.dumps(
        {"state": "I was charged twice for one order.", "questions": questions}
    ).encode()


def _spec_body() -> bytes:
    return json.dumps(
        {
            "state": "I was charged twice for one order.",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Choose the reviewing team.",
                    "criteria": {
                        "billing": "Payments and refunds",
                        "support": "Technical help",
                    },
                },
                "urgency": {
                    "type": "score",
                    "criteria": [
                        "Normal review",
                        "Timely response",
                        "Immediate human attention",
                    ],
                },
                "duplicate": {
                    "type": "noul",
                    "instructions": "Does the message report a duplicate charge?",
                },
            },
        }
    ).encode()


def _service(backend, **overrides) -> SystemOneService:
    args = _args(
        systemone_backend=overrides.pop("systemone_backend", "stub"),
        systemone_model=overrides.pop("systemone_model", _MODEL),
        **overrides,
    )
    return SystemOneService(
        resolve_systemone_config(args),
        backend=backend,
        engine_args=args,
        engine_present=True,
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"questions": {"a": {"type": "noul"}}}',
        b'{"state": "x", "questions": {}}',
        b'{"state": "x", "questions": {"a": {"type": "boolean"}}}',
        b'{"state": "x", "questions": {"a": {"type": "choice"}}}',
        b'{"state": "x", "questions": {"a": {"type": "choice", "options": ["a"]}}}',
        b'{"state": "x", "questions": {"a": {"type": "score", "criteria": ["only"]}}}',
        b'{"state": 1, "questions": {"a": {"type": "noul"}}}',
        b'{"state": "x", "questions": {" ": {"type": "noul"}}}',
        b'{"state": "x", "questions": {"a": {"type": "noul", "multi_label": true}}}',
        b'{"state": "x", "questions": {"a": {"type": "noul", "criteria": ["yes"]}}}',
        b'{"model": "other", "state": "x", "questions": {"a": {"type": "noul"}}}',
    ],
)
def test_malformed_requests_are_400(raw):
    with pytest.raises(SystemOneError) as exc:
        parse_request(raw, default_model=_MODEL)
    assert exc.value.status_code == 400


def test_duplicate_key_and_depth_and_too_many_questions():
    with pytest.raises(SystemOneError, match="duplicate"):
        parse_request(
            b'{"state": "x", "state": "y", "questions": {"a": {"type": "noul"}}}',
            default_model=_MODEL,
        )
    nested: dict = {"type": "noul"}
    cursor = nested
    for _ in range(40):
        cursor["instructions"] = {}
        cursor = cursor["instructions"]
    with pytest.raises(SystemOneError, match="nesting"):
        parse_request(
            json.dumps({"state": "x", "questions": {"a": nested}}).encode(),
            default_model=_MODEL,
        )
    questions = {f"q{i}": {"type": "noul"} for i in range(33)}
    with pytest.raises(SystemOneError, match="32"):
        parse_request(
            json.dumps({"state": "x", "questions": questions}).encode(),
            default_model=_MODEL,
        )


def test_schema_translation_and_response_shape():
    parsed = parse_request(_spec_body(), default_model=_MODEL)
    kinds = {task.name: task.kind for task in parsed.tasks}
    assert kinds == {"team": "choice", "urgency": "score", "duplicate": "noul"}
    team = next(task for task in parsed.tasks if task.name == "team")
    assert team.labels == ("billing", "support")
    assert team.schema_labels()["billing"] == "Payments and refunds"
    urgency = next(task for task in parsed.tasks if task.name == "urgency")
    assert urgency.labels == ("0", "1", "2")
    duplicate = next(task for task in parsed.tasks if task.name == "duplicate")
    assert duplicate.labels == ("yes", "no")
    assert parsed.state_text.startswith("I was charged twice")

    described = parse_request(
        _body(
            handoff={
                "type": "noul",
                "criteria": {"true": "needs a person", "false": "stay automated"},
                "cls_threshold": 0.4,
            }
        ),
        default_model=_MODEL,
    )
    noul = described.tasks[0]
    assert noul.descriptions == ("needs a person", "stay automated")
    assert noul.threshold == 0.4

    structured = parse_request(
        json.dumps(
            {
                "state": {"message": "charged twice"},
                "questions": {"a": {"type": "noul", "instructions": {"rubric": "x"}}},
            }
        ).encode(),
        default_model=_MODEL,
    )
    assert structured.state_text == '{"message":"charged twice"}'
    assert structured.tasks[0].instruction == '{"rubric":"x"}'

    response = build_local_response(
        _MODEL,
        urgency_tasks(urgency),
        {"urgency": {"0": 0.1, "1": 0.3, "2": 0.6}},
    )
    answer = response["answers"]["urgency"]
    assert answer["score"] == pytest.approx(1.5)
    assert answer["confidence"] == pytest.approx(0.3)
    assert answer["legend"]["2"] == "Immediate human attention"
    assert "usage" not in response

    choice = build_local_response(
        _MODEL,
        (team,),
        {"team": {"billing": 0.8, "support": 0.2}},
    )["answers"]["team"]
    assert choice["choice"] == "billing"
    assert choice["confidence"] == pytest.approx(0.6)
    assert set(choice) == {"type", "choice", "probabilities", "confidence"}

    noul_answer = build_local_response(
        _MODEL,
        (duplicate,),
        {"duplicate": {"yes": 0.8, "no": 0.2}},
    )["answers"]["duplicate"]
    assert noul_answer == {"type": "noul", "noul": 0.8}


def urgency_tasks(task):
    return (task,)


def test_feature_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_SYSTEMONE_BACKEND", raising=False)
    args = _args()
    assert systemone_enabled(args) is False
    assert resolve_systemone_config(args).backend == "off"
    validate_systemone_args(args)


def test_env_and_cli_resolution(monkeypatch):
    monkeypatch.setenv("VLLM_SYSTEMONE_BACKEND", "stub")
    monkeypatch.setenv("VLLM_SYSTEMONE_MODEL", "from-env")
    cfg = resolve_systemone_config(_args())
    assert cfg.backend == "stub"
    assert cfg.model == "from-env"
    cfg = resolve_systemone_config(
        _args(systemone_backend="http", systemone_url="http://127.0.0.1:8091")
    )
    assert cfg.backend == "http"
    assert cfg.url == "http://127.0.0.1:8091/v1/systemone"
    monkeypatch.delenv("VLLM_SYSTEMONE_MODEL", raising=False)
    with pytest.raises(ValueError, match="systemone-model"):
        validate_systemone_args(_args(systemone_backend="gliner2"))
    with pytest.raises(ValueError, match="bfloat16"):
        resolve_dtype("cuda", "bf16")


def test_cli_flags_do_not_default_the_backend_on():
    parser = add_systemone_cli_args(__import__("argparse").ArgumentParser())
    args = parser.parse_args([])
    assert args.systemone_backend is None
    assert systemone_enabled(args) is False


def test_engine_devices_are_not_spare(monkeypatch):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = _args(tensor_parallel_size=4, pipeline_parallel_size=1)
    assert engine_logical_devices(args) == {0, 1, 2, 3}
    with pytest.raises(SystemOneStartupError, match="spare"):
        assert_spare_device("cuda:0", args)
    with pytest.raises(SystemOneStartupError, match="spare"):
        assert_spare_device("cuda:3", args)

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6,7")
    with pytest.raises(SystemOneStartupError, match="outside the visible"):
        assert_spare_device("cuda:4", args)

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6,7,8")
    kind, index = assert_spare_device("cuda:4", args)
    assert (kind, index) == ("cuda", 4)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4")
    with pytest.raises(SystemOneStartupError, match="disagree"):
        assert_spare_device("cuda:4", args)


def test_vram_reserve_and_dedicated_stream(monkeypatch):
    fake = types.ModuleType("torch")

    class _Stream:
        def __init__(self, ident):
            self.cuda_stream = ident

    fake.cuda = types.SimpleNamespace(
        mem_get_info=lambda index: (1024**3, 16 * 1024**3),
        Stream=lambda device: _Stream(1),
        default_stream=lambda device: _Stream(1),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(SystemOneStartupError, match="systemone-vram-reserve-gb"):
        assert_free_vram(4, 2.0)
    with pytest.raises(SystemOneStartupError, match="default CUDA stream"):
        open_decision_stream(4)

    fake.cuda.mem_get_info = lambda index: (3 * 1024**3, 16 * 1024**3)
    assert_free_vram(4, 2.0)
    fake.cuda.Stream = lambda device: _Stream(9)
    stream = open_decision_stream(4)
    assert stream.cuda_stream == 9


def test_gliner_import_is_lazy():
    assert "gliner2" not in sys.modules
    from vllm.entrypoints.systemone.backends import gliner2 as gliner_mod

    assert "gliner2" not in sys.modules
    assert hasattr(gliner_mod, "Gliner2Backend")


def test_missing_gliner2_names_the_install(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gliner2" or name.startswith("gliner2."):
            raise ImportError("blocked for the test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    from vllm.entrypoints.systemone.backends.gliner2 import _import_gliner

    with pytest.raises(ImportError, match="uv pip install gliner2"):
        _import_gliner()


def test_gliner_groups_schemas_and_refuses_capture(monkeypatch):
    backend = Gliner2Backend(
        model=_MODEL,
        device="cpu",
        dtype=None,
        vram_reserve_gb=2,
        engine_args=_args(tensor_parallel_size=4),
        check_engine_devices=True,
    )
    backend._loaded = True
    seen: list[int] = []

    def classify(texts, tasks):
        seen.append(len(texts))
        rows = []
        for _text in texts:
            rows.append(
                {
                    task.name: {label: 1.0 / len(task.labels) for label in task.labels}
                    for task in tasks
                }
            )
        return rows

    backend._classify_texts = classify  # type: ignore[method-assign]
    first = parse_request(_body(a={"type": "noul"}), default_model=_MODEL)
    other = parse_request(
        _body(b={"type": "choice", "criteria": {"x": "one", "y": "two"}}),
        default_model=_MODEL,
    )
    again = parse_request(_body(a={"type": "noul"}), default_model=_MODEL)
    outs = backend.decide_batch([first, other, again])
    assert seen == [2, 1]
    assert [out["answers"].keys() for out in outs] == [{"a"}, {"b"}, {"a"}]
    assert "confidence" not in outs[0]["answers"]["a"]

    backend._device_kind = "cuda"
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_current_stream_capturing=lambda: True,
        stream=lambda _stream: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(SystemOneError, match="graph capture"):
        backend.decide_batch([first])


def test_cuda_guard_enters_the_dedicated_stream(monkeypatch):
    backend = Gliner2Backend(
        model=_MODEL,
        device="cpu",
        dtype=None,
        vram_reserve_gb=2,
        engine_args=None,
        check_engine_devices=False,
    )
    backend._device_kind = "cuda"
    synced: dict[str, bool] = {}

    class _Stream:
        def synchronize(self):
            synced["yes"] = True

    entered: dict[str, bool] = {}

    class _Ctx:
        def __enter__(self):
            entered["yes"] = True

        def __exit__(self, *args):
            return False

    backend._stream = _Stream()
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_current_stream_capturing=lambda: False,
        stream=lambda _stream: _Ctx(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert backend._guard_and_run(lambda: "ran") == "ran"
    assert entered["yes"] and synced["yes"]


@pytest.mark.asyncio
async def test_microbatch_under_concurrent_requests():
    backend = StubBackend()
    service = _service(backend, systemone_max_batch=8, systemone_max_wait_ms=200)
    await service.start()
    try:
        parsed = parse_request(_spec_body(), default_model=_MODEL)
        outs = await asyncio.gather(*[service.ask(parsed) for _ in range(4)])
    finally:
        await service.shutdown()
    assert backend.batch_sizes == [4]
    assert outs[0]["answers"]["urgency"]["score"] == pytest.approx(1.5)
    assert outs[0]["answers"]["team"]["choice"] == "support"
    assert outs[0]["model"] == _MODEL


@pytest.mark.asyncio
async def test_queue_overflow_is_429():
    started = threading.Event()
    release = threading.Event()
    backend = StubBackend(started=started, release=release)
    service = _service(
        backend,
        systemone_max_batch=1,
        systemone_max_wait_ms=0,
        systemone_max_queue=1,
        systemone_timeout_s=5,
    )
    await service.start()
    parsed = parse_request(_body(a={"type": "noul"}), default_model=_MODEL)
    first = asyncio.create_task(service.ask(parsed))
    assert await asyncio.to_thread(started.wait, 2)
    second = asyncio.create_task(service.ask(parsed))
    await asyncio.sleep(0.05)
    try:
        with pytest.raises(SystemOneError) as exc:
            await service.ask(parsed)
        assert exc.value.status_code == 429
    finally:
        release.set()
        await first
        await second
        await service.shutdown()


@pytest.mark.asyncio
async def test_request_timeout_is_504():
    started = threading.Event()
    release = threading.Event()
    backend = StubBackend(started=started, release=release)
    service = _service(
        backend,
        systemone_max_batch=1,
        systemone_timeout_s=0.2,
    )
    await service.start()
    parsed = parse_request(_body(a={"type": "noul"}), default_model=_MODEL)
    try:
        with pytest.raises(SystemOneError) as exc:
            await service.ask(parsed)
        assert exc.value.status_code == 504
        assert started.is_set()
    finally:
        release.set()
        await service.shutdown()


@pytest.mark.asyncio
async def test_http_proxy_backend_against_local_server():
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            seen.append(
                {
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": body,
                }
            )
            if body["questions"]["a"]["type"] != "noul":
                raw = b'{"error": "bad"}'
                self.send_response(400)
            else:
                raw = json.dumps(
                    {
                        "model": body["model"],
                        "answers": {"a": {"type": "noul", "noul": 0.8}},
                    }
                ).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        from vllm.entrypoints.systemone.backends.http_proxy import HTTPBackend

        backend = HTTPBackend(
            f"http://{host}:{port}/v1/systemone",
            timeout_s=5,
            api_key="test-key",
        )
        service = _service(
            backend,
            systemone_backend="http",
            systemone_url=f"http://{host}:{port}",
        )
        await service.start()
        try:
            parsed = parse_request(_body(a={"type": "noul"}), default_model=_MODEL)
            out = await service.ask(parsed)
        finally:
            await service.shutdown()
        assert out["answers"]["a"]["noul"] == 0.8
        assert seen[0]["path"] == "/v1/systemone"
        assert seen[0]["auth"] == "Bearer test-key"
        assert seen[0]["body"]["model"] == _MODEL
        assert "enable" not in seen[0]["body"]
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_route_shape_and_enable_field_is_ignored():
    app = build_systemone_app(_args(systemone_backend="stub", systemone_model=_MODEL))
    payload = json.loads(_spec_body())
    payload["enable"] = True
    with TestClient(app) as client:
        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/v1/systemone" in paths
        assert "/v1/models" not in paths
        assert "/v1/chat/completions" not in paths
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body["answers"]["duplicate"] == {"type": "noul", "noul": 0.2}
        bad = client.post("/v1/systemone", json={"state": "x"})
        assert bad.status_code == 400


def test_off_server_does_not_grow_routes():
    from fastapi import FastAPI

    app = FastAPI()
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/v1/systemone" not in paths


@pytest.mark.asyncio
async def test_init_is_a_noop_when_off():
    from vllm.entrypoints.systemone.service import init_systemone_state

    state = Namespace()
    await init_systemone_state(state, _args(), engine_present=True)
    assert not hasattr(state, "systemone_service")


def test_optional_gliner2_schema_builder_without_weights():
    pytest.importorskip("gliner2")
    try:
        from gliner2.classification import ClassificationSchema
    except ImportError as exc:
        pytest.skip(f"gliner2 classification stack is not installed: {exc}")

    parsed = parse_request(_spec_body(), default_model=_MODEL)
    schema = ClassificationSchema()
    for task in parsed.tasks:
        labels = task.schema_labels()
        kwargs = {}
        if task.instruction:
            kwargs["instruction"] = task.instruction
        if task.kind == "score":
            schema.ordinal(task.name, labels, **kwargs)
        else:
            schema.single(task.name, labels, **kwargs)
    assert schema.task_order == ("team", "urgency", "duplicate")
