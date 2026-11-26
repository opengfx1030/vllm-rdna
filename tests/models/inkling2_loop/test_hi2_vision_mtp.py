# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the HI2 kit's vision/MTP checks and the DSpark boot switches.

The tools under test are the ones the GPU runbook uses: ``validate_dir.py``
(header-only serve-dir census), ``check_real_dir.py`` (bit-exact vision tower and
MTP comparison), the HI2 recipe's DSPARK/VISION branches and the two probes'
pure helpers.  No GPU, no vllm import for the header checks; the safetensors
comparisons use tiny files written here.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
KIT = Path(os.environ.get("INKLING2_KIT_ROOT", REPO_ROOT / "tools" / "inkling_hi2"))
RECIPE = REPO_ROOT / "tools" / "rdna" / "recipes" / "inkling2-hi2-tp2.env"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


validate_dir = _load("_i2_validate_under_test", KIT / "validate_dir.py")
mtp_probe = _load("_i2_mtp_probe_under_test", KIT / "mtp_probe.py")
vision_probe = _load("_i2_vision_probe_under_test", KIT / "vision_probe.py")

VIS = "model.visual."


def _entry(dtype: str = "BF16", shape=(4, 4), size: int = 32) -> dict:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, size]}


# --- validate_dir: the vision census ----------------------------------------


def test_vision_census_accepts_the_stock_shape():
    weight_map = {VIS + "a.weight": "s.safetensors", VIS + "b.weight": "s.safetensors"}
    info, errors = validate_dir.check_vision(weight_map, lambda n: _entry())
    assert errors == []
    assert info["tensors"] == 2
    assert info["shards"] == ["s.safetensors"]
    assert info["dtypes"] == {"BF16": 2}
    assert info["bytes"] == 64
    assert info["missing"] == []


def test_a_directory_without_the_vision_tower_is_an_error():
    info, errors = validate_dir.check_vision({}, lambda n: None)
    assert info["tensors"] == 0
    assert any("vision tower is missing" in e for e in errors)


def test_a_visual_tensor_whose_shard_is_missing_is_an_error():
    weight_map = {VIS + "a.weight": "s.safetensors"}
    info, errors = validate_dir.check_vision(weight_map, lambda n: None)
    assert info["missing"] == [VIS + "a.weight"]
    assert any("1 model.visual.* tensors are missing" in e for e in errors)


def test_a_quantized_vision_tensor_is_an_error():
    weight_map = {VIS + "a.weight": "s.safetensors"}
    _, errors = validate_dir.check_vision(
        weight_map, lambda n: _entry(dtype="F8_E4M3", size=16)
    )
    assert any("non-float dtypes" in e for e in errors)


def test_fp32_vision_buffers_are_tolerated():
    weight_map = {VIS + "a.weight": "s.safetensors"}
    _, errors = validate_dir.check_vision(
        weight_map, lambda n: _entry(dtype="F32", size=64)
    )
    assert errors == []


# --- validate_dir: the MTP file ---------------------------------------------


def test_a_missing_mtp_file_is_a_warning_unless_required():
    info, errors, warnings = validate_dir.check_mtp(
        None, {}, "/x/mtp.safetensors", False
    )
    assert info["present"] is False and errors == []
    assert any("no MTP file" in w for w in warnings)

    _, errors, warnings = validate_dir.check_mtp(None, {}, "/x/mtp.safetensors", True)
    assert any("no MTP file" in e for e in errors)
    assert warnings == []


def test_the_mtp_file_must_be_bf16_and_not_overlap_the_index():
    header = {"layers.0.mlp.down_proj.weight": _entry()}
    info, errors, warnings = validate_dir.check_mtp(
        header, {"layers.0.mlp.down_proj.weight": "s.safetensors"}, "m", True
    )
    assert info["keys"] == 1 and info["dtypes"] == {"BF16": 1}
    assert any("shares" in e for e in errors)

    header = {"layers.0.mlp.down_proj.weight": _entry(dtype="F32", size=64)}
    _, errors, _ = validate_dir.check_mtp(header, {}, "m", True)
    assert any("non-bf16" in e for e in errors)


# --- check_real_dir: bit-exact comparisons on tiny files ---------------------

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
check_real_dir = _load("_i2_real_dir_under_test", KIT / "check_real_dir.py")


def _write(path: Path, tensors: dict) -> None:
    safetensors_torch.save_file(tensors, str(path))


def test_the_vision_tower_compare_passes_equal_files(tmp_path):
    names = [VIS + "blocks.0.w", VIS + "proj.w"]
    weights = {n: torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) for n in names}
    _write(tmp_path / "m.safetensors", weights)
    _write(tmp_path / "s.safetensors", weights)
    weight_map = {n: "m.safetensors" for n in names}
    smap = {n: "s.safetensors" for n in names}
    compared, bad, messages = check_real_dir.compare_vision(
        tmp_path, tmp_path, weight_map, smap
    )
    assert (compared, bad, messages) == (2, 0, [])


def test_the_vision_tower_compare_catches_one_changed_value(tmp_path):
    names = [VIS + "blocks.0.w", VIS + "proj.w"]
    mine = {n: torch.arange(8, dtype=torch.bfloat16) for n in names}
    stock = {n: torch.arange(8, dtype=torch.bfloat16) for n in names}
    stock[names[0]] = stock[names[0]].clone()
    stock[names[0]][3] = 7.0
    _write(tmp_path / "m.safetensors", mine)
    _write(tmp_path / "s.safetensors", stock)
    weight_map = {n: "m.safetensors" for n in names}
    smap = {n: "s.safetensors" for n in names}
    compared, bad, messages = check_real_dir.compare_vision(
        tmp_path, tmp_path, weight_map, smap
    )
    assert compared == 2 and bad == 1
    assert any(names[0] in m for m in messages)


def test_the_mtp_compare_requires_names_bf16_and_equal_bytes(tmp_path):
    good = {"layers.0.w": torch.arange(4, dtype=torch.bfloat16)}
    _write(tmp_path / "mtp.safetensors", good)
    compared, messages = check_real_dir.compare_mtp(
        tmp_path / "mtp.safetensors", tmp_path / "mtp.safetensors"
    )
    assert compared == 1 and messages == []

    _write(
        tmp_path / "other.safetensors",
        {"layers.0.w": torch.arange(4, dtype=torch.bfloat16) + 1},
    )
    _, messages = check_real_dir.compare_mtp(
        tmp_path / "mtp.safetensors", tmp_path / "other.safetensors"
    )
    assert any("differs" in m for m in messages)

    _write(
        tmp_path / "f32.safetensors",
        {"layers.0.w": torch.zeros(4, dtype=torch.float32)},
    )
    _, messages = check_real_dir.compare_mtp(
        tmp_path / "mtp.safetensors", tmp_path / "f32.safetensors"
    )
    assert any("non-bf16" in m for m in messages)


# --- the recipe's DSpark and vision branches --------------------------------


def _recipe_env(**env: str) -> dict[str, str]:
    script = (
        'set -e\nsource "$RECIPE"\n'
        'printf "%s\\n" "MAXLEN=$MAXLEN" "MAXBAT=$MAXBAT" '
        '"SPEC=$SPEC_CONFIG" "FEATURES=$FEATURES" "EXTRA=$EXTRA_ARGS"\n'
    )
    base = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "DSPARK",
            "VISION",
            "ADAPTIVE",
            "MAXLEN",
            "MAXBAT",
            "SPEC_CONFIG",
            "DRAFT_VOCAB",
        }
    }
    run = subprocess.run(
        ["bash", "-c", script],
        env={**base, "RECIPE": str(RECIPE), **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    out = {}
    for line in run.stdout.splitlines():
        key, _, value = line.partition("=")
        out[key] = value
    return out


def test_the_hi2_recipe_boots_dspark_by_default_with_the_stock_block():
    env = _recipe_env()
    assert env["MAXLEN"] == "720000"
    assert env["MAXBAT"] == "1024"
    assert '"method":"dspark"' in env["SPEC"]
    assert '"num_speculative_tokens":7' in env["SPEC"]
    assert "[[1,2,2],[3,8,0]]" in env["SPEC"].replace(" ", "")
    assert "language-model-only" in env["FEATURES"]
    assert '"image":0' in env["EXTRA"]


def test_the_plain_recipe_branch_is_the_full_context_without_spec():
    env = _recipe_env(DSPARK="0")
    assert env["MAXLEN"] == "1048576"
    assert env["MAXBAT"] == "320"
    assert env["SPEC"] == ""


def test_the_vision_branch_drops_language_model_only_and_allows_an_image():
    env = _recipe_env(VISION="1")
    assert "language-model-only" not in env["FEATURES"]
    assert '"image":1' in env["EXTRA"]


# --- the probes' pure helpers ----------------------------------------------


def test_the_log_parser_reads_the_last_per_position_line():
    text = "\n".join(
        [
            (
                "INFO SpecDecoding metrics: Mean acceptance length: 2.50, "
                "Per-position acceptance rate: 0.900, 0.600"
            ),
            (
                "INFO SpecDecoding metrics: Mean acceptance length: 1.75, "
                "Per-position acceptance rate: 0.700, 0.050"
            ),
        ]
    )
    parsed = mtp_probe.parse_log_metrics(text)
    assert parsed["mean_acceptance_length"] == 1.75
    assert parsed["per_position"] == [0.7, 0.05]


def test_the_vision_probe_writes_valid_pngs_and_scores_overlap(tmp_path):
    images = vision_probe.make_images(tmp_path)
    assert len(images) == 4
    for path in images:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 60
    assert vision_probe.overlap("a red square on white", "a red square on white") == 1.0
    assert vision_probe.overlap("a red square", "blue circle") == 0.0
    assert vision_probe.clean({"text": "x", "finish_reason": "stop"})
    assert not vision_probe.clean({"text": "\ufffd", "finish_reason": "stop"})


# --- serve.sh forwards the switches ----------------------------------------


def test_serve_forwards_recipe_dspark_and_vision():
    text = (KIT / "serve.sh").read_text()
    assert "RECIPE=${RECIPE:-inkling2-mini-tp2}" in text
    assert "DSPARK=${DSPARK:-0}" in text
    assert "VISION=${VISION:-0}" in text
    assert 'DSPARK="$DSPARK" VISION="$VISION"' in text
    assert "spec_args=()" in text
    assert 'spec_args=(SPEC_CONFIG="$SPEC_CONFIG")' in text
    assert '"${spec_args[@]}"' in text
    assert 'SPEC_CONFIG="${SPEC_CONFIG:-}"' not in text
    assert 'RECIPE="$RECIPE"' in text
    assert "VISION_IMAGES=${VISION_IMAGES:-1}" in text
    assert '\\"image\\":$VISION_IMAGES' in text and '"image":0' in text


# --- vision_probe: the served model name and the request URL ----------------
# Card1 failed with HTTP 404 because the probe posted "model": "default" while
# the recipe serves inkling2-hi2.  vLLM answers 404 for an unknown model name;
# these tests pin the resolved name, the exact URL and the error message.

import json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class _VisionStub(BaseHTTPRequestHandler):
    model = "inkling2-hi2"
    fail_post = False
    with_token_ids = True
    reject_token_ids = False
    posts: list = []

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._send({"data": [{"id": self.model}]})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        type(self).posts.append({"path": self.path, "payload": dict(payload)})
        if type(self).fail_post:
            self.send_error(404)
            return
        if self.path == "/tokenize":
            self._send({"count": 3, "tokens": [101, 102, 103]})
            return
        if payload.get("return_token_ids") and type(self).reject_token_ids:
            self.send_error(400)
            return
        choice: dict = {"message": {"content": "a red square"}, "finish_reason": "stop"}
        if type(self).with_token_ids and payload.get("return_token_ids"):
            choice["token_ids"] = [11, 12]
        self._send({"choices": [choice], "usage": {"completion_tokens": 2}})

    def _send(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def vision_stub():
    _VisionStub.posts = []
    _VisionStub.fail_post = False
    _VisionStub.with_token_ids = True
    _VisionStub.reject_token_ids = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VisionStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_base_url_rejects_a_path():
    with pytest.raises(ValueError, match="without a path"):
        vision_probe.base_url("http://127.0.0.1:18230/v1")
    assert vision_probe.base_url("http://127.0.0.1:18230/") == "http://127.0.0.1:18230"


def test_the_probe_posts_the_model_v1_models_reports(tmp_path, vision_stub):
    image = vision_probe.make_images(tmp_path)[0]
    model = vision_probe.served_model(vision_stub)
    assert model == "inkling2-hi2"
    got = vision_probe.describe(vision_stub, model, image, 8, 5.0)
    assert got["text"] == "a red square"
    last = _VisionStub.posts[-1]
    assert last["path"] == "/v1/chat/completions"
    assert last["payload"]["model"] == "inkling2-hi2"


def test_a_404_names_the_exact_url(tmp_path, vision_stub):
    _VisionStub.fail_post = True
    image = vision_probe.make_images(tmp_path)[0]
    with pytest.raises(RuntimeError) as e:
        vision_probe.describe(vision_stub, "default", image, 8, 5.0)
    assert "POST " + vision_stub + "/v1/chat/completions: HTTP 404" in str(e.value)


def test_mtp_complete_posts_the_served_model(vision_stub):
    mtp_probe._SERVED.clear()
    got = mtp_probe.complete(vision_stub, "hello", 8, 5.0)
    assert got["text"] == "a red square"
    last = _VisionStub.posts[-1]
    assert last["path"] == "/v1/chat/completions"
    assert last["payload"]["model"] == "inkling2-hi2"
    assert last["payload"]["return_token_ids"] is True
    assert got["token_ids"] == [11, 12]
    mtp_probe._SERVED.clear()


def test_mtp_base_url_rejects_a_path():
    with pytest.raises(ValueError, match="without a path"):
        mtp_probe.base_url("http://127.0.0.1:18230/v1")


def test_mtp_404_names_the_url_and_model(vision_stub):
    mtp_probe._SERVED.clear()
    _VisionStub.fail_post = True
    with pytest.raises(RuntimeError) as e:
        mtp_probe.complete(vision_stub, "hello", 8, 5.0)
    assert "POST " + vision_stub + "/v1/chat/completions: HTTP 404" in str(e.value)
    mtp_probe._SERVED.clear()


# --- mtp_probe: the token-level floor and the settled no-draft window -------
# The stack is not bitwise reproducible across processes, so the greedy gate is
# "identical, or within what two plain boots of the same weights already show".


def test_diff_tokens_reports_the_first_divergence_and_count():
    assert mtp_probe.diff_tokens([1, 2, 3], [1, 2, 3]) == {"first": None, "count": 0}
    assert mtp_probe.diff_tokens([1, 2, 3], [1, 2, 4]) == {"first": 2, "count": 1}
    assert mtp_probe.diff_tokens([1, 2, 3], [1, 2]) == {"first": 2, "count": 1}
    assert mtp_probe.diff_tokens([1, 9, 3, 4], [1, 2, 3]) == {
        "first": 1,
        "count": 2,
    }


def test_the_floor_verdict_passes_identity_and_within_floor_only():
    v = mtp_probe.floor_verdict([1, 2, 3], [1, 2, 3], None)
    assert v["ok"] and v["floor"] is None
    # two plain boots differ at one token; the spec differs by the same amount
    v = mtp_probe.floor_verdict([1, 2, 9], [1, 2, 3], [1, 2, 4])
    assert v["ok"]
    assert v["spec"]["first"] == 2 and v["floor"]["count"] == 1
    # beyond the floor is a finding
    v = mtp_probe.floor_verdict([1, 2, 9, 9], [1, 2, 3], [1, 2, 4])
    assert not v["ok"]
    assert v["spec"]["count"] == 2 and v["spec"]["first"] == 2
    # no floor: identity is required; a zero floor keeps the strict gate
    assert not mtp_probe.floor_verdict([1, 2, 9], [1, 2, 3], None)["ok"]
    assert not mtp_probe.floor_verdict([1, 2, 9], [1, 2, 3], [1, 2, 3])["ok"]


def test_the_no_draft_verdict_uses_the_window_delta():
    assert mtp_probe.no_draft_verdict(10, 10) == {"delta": 0, "ok": True}
    v = mtp_probe.no_draft_verdict(10, 13)
    assert v["delta"] == 3 and not v["ok"]


def test_the_no_draft_window_settles_before_reading_the_counter():
    text = (KIT / "mtp_probe.py").read_text()
    sleep_at = text.index("time.sleep(args.settle_seconds)")
    read_at = text.index(
        'window_before = metric(spec, "vllm:spec_decode_num_drafts_total")'
    )
    assert sleep_at < read_at


def test_mtp_complete_falls_back_to_the_server_tokenizer(vision_stub):
    mtp_probe._SERVED.clear()
    _VisionStub.with_token_ids = False
    got = mtp_probe.complete(vision_stub, "hello", 8, 5.0)
    assert got["token_ids"] == [101, 102, 103]
    assert any(p["path"] == "/tokenize" for p in _VisionStub.posts)
    mtp_probe._SERVED.clear()


def test_mtp_complete_retries_without_the_token_id_extension(vision_stub):
    mtp_probe._SERVED.clear()
    _VisionStub.with_token_ids = False
    _VisionStub.reject_token_ids = True
    got = mtp_probe.complete(vision_stub, "hello", 8, 5.0)
    assert got["token_ids"] == [101, 102, 103]
    chats = [p for p in _VisionStub.posts if p["path"] == "/v1/chat/completions"]
    assert "return_token_ids" in chats[0]["payload"]
    assert "return_token_ids" not in chats[1]["payload"]
    mtp_probe._SERVED.clear()


def test_the_driver_boots_two_plain_servers_and_runs_720k_first():
    text = (KIT / "validate_final_a2.sh").read_text()
    assert "STEPS=(kernel serve_dir smoke kl eval vision bench boot720 dspark)" in text
    assert text.count('boot_server plain16k 18231 "$SERVE_DIR"') == 2
    assert "--floor-json" in text and "--stock-only" in text
    assert "run_step boot720 s_boot720\nrun_step dspark s_dspark" in text
    assert 'dry_serve "stock dspark" stockspec 18231 "$STOCK"' in text
