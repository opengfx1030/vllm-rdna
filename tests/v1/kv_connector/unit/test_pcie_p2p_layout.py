# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the PCIe P2P KV layout and transfer-param contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SRC = (
    Path(__file__).resolve().parents[4]
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "pcie_p2p_layout.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("pcie_p2p_layout", _SRC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_remote_prefill_keeps_the_last_token():
    layout = _load()
    assert layout.remote_prefill_token_count(128, 0) == 127
    assert layout.remote_prefill_token_count(128, 127) == 0
    assert layout.remote_prefill_token_count(1, 0) == 0
    assert layout.remote_prefill_token_count(0, 0) == 0


def test_block_planes_kv_and_flat():
    layout = _load()
    # [2, num_blocks, block, heads, dim] -> two planes, one contiguous block.
    planes, elements = layout.block_planes((2, 100, 16, 4, 128), 100)
    assert planes == 2
    assert elements == 16 * 4 * 128
    planes, elements = layout.block_planes((80, 32, 64), 80)
    assert planes == 1
    assert elements == 32 * 64


def test_block_planes_rejects_a_missing_block_axis():
    layout = _load()
    with pytest.raises(ValueError, match="no dimension"):
        layout.block_planes((2, 16, 4), 100)


def test_finished_params_match_the_disagg_proxy_fields():
    layout = _load()
    params = layout.finished_transfer_params(
        block_ids=([1, 2], [3]),
        engine_id="prefill-0",
        request_id="req",
        handshake_host="127.0.0.1",
        handshake_port=19000,
        tp_size=2,
        remote_num_tokens=31,
    )
    assert params["do_remote_prefill"] is True
    assert params["do_remote_decode"] is False
    assert params["remote_block_ids"] == ([1, 2], [3])
    assert params["remote_host"] == "127.0.0.1"
    assert params["remote_port"] == 19000
    assert params["tp_size"] == 2
    assert params["transfer_mode"] == "pcie_p2p"


def test_connector_is_registered():
    factory = (
        Path(__file__).resolve().parents[4]
        / "vllm"
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "factory.py"
    ).read_text(encoding="utf-8")
    assert '"PcieP2pConnector"' in factory
