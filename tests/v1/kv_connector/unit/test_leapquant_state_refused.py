# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A KV connector moves state blocks and does not know the LeapQuant blob
format, so connector creation refuses quantized Mamba state.

The rule follows the safety section of the upstream LeapQuant RFC
(vllm-project/vllm#64725, lihaoyang-amd). The tests are original to this fork.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import DeviceConfig, KVTransferConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    has_kv_transfer_group,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec

pytestmark = pytest.mark.cpu_test

SERVING_FORMAT = "leapquant-int8/v1/k128v128/r4/p32/cap34/rec-float16"
CONNECTORS = {
    "example": KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_both"),
    # A connector that supports HMA: only the state format can stop it.
    "cpu-offload": KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": 1 << 30},
    ),
}


@pytest.fixture(autouse=True)
def mock_hybrid_kv_cache_supported(monkeypatch):
    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)


def make_vllm_config(kv_transfer_config):
    return VllmConfig(
        device_config=DeviceConfig("cpu"),
        scheduler_config=SchedulerConfig(
            max_model_len=16,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=False,
        ),
        kv_transfer_config=kv_transfer_config,
    )


def make_kv_cache_config(state_format):
    spec = MambaSpec(
        block_size=512,
        shapes=((5, 2560), (12, 37024)),
        dtypes=(torch.float16, torch.uint8),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=2,
        state_format=state_format,
    )
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["gdn0"], spec)],
    )


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize("connector", CONNECTORS)
def test_factory_refuses_every_connector_over_leapquant_state(connector, role):
    vllm_config = make_vllm_config(CONNECTORS[connector])
    with pytest.raises(ValueError, match="quantized Mamba state") as error:
        KVConnectorFactory.create_connector(
            vllm_config, role, make_kv_cache_config(SERVING_FORMAT)
        )
    assert SERVING_FORMAT in str(error.value)
    assert "--kv-offloading-size" in str(error.value)


def test_factory_leaves_unquantized_mamba_state_to_the_other_checks():
    # The refusal reads the spec format, so FP32 state reaches the next check.
    vllm_config = make_vllm_config(CONNECTORS["example"])
    with pytest.raises(ValueError, match="does not support HMA but HMA is enabled"):
        KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.SCHEDULER, make_kv_cache_config(None)
        )


def test_worker_setup_refuses_a_connector_over_leapquant_state():
    vllm_config = make_vllm_config(CONNECTORS["example"])
    tp_group = MagicMock()
    tp_group.broadcast_object.side_effect = lambda obj, src: obj
    with (
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp_group),
        pytest.raises(ValueError, match="quantized Mamba state"),
    ):
        ensure_kv_transfer_initialized(
            vllm_config, make_kv_cache_config(SERVING_FORMAT)
        )
    assert not has_kv_transfer_group()
