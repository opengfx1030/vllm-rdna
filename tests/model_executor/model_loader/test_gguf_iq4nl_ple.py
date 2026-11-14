# SPDX-License-Identifier: Apache-2.0
"""Bounded IQ4_NL PLE row reads, independent of vLLM imports."""

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType, quants

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm/model_executor/model_loader/gguf_iq4nl_ple.py"
)


def load_class():
    spec = importlib.util.spec_from_file_location("gguf_iq4nl_ple_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GGUFIQ4NLPLETable


def fixture_file(tmp_path, rows=7, width=160):
    blocks = np.empty((rows * width // 32, 18), dtype=np.uint8)
    scales = [0.0, 1.0, -2.0, 2**-24, -(2**-14), 0.125, 0.03125]
    for i, block in enumerate(blocks):
        block[:2] = np.frombuffer(
            struct.pack("<e", scales[i % len(scales)]), dtype=np.uint8
        )
        block[2:] = np.array(
            [((j + i) % 16) | (((15 - j + i) % 16) << 4) for j in range(16)],
            dtype=np.uint8,
        )
    path = tmp_path / "table.gguf"
    path.write_bytes(b"prefix-not-a-header" + blocks.tobytes() + b"trailer")
    expected = quants.dequantize(blocks, GGMLQuantizationType.IQ4_NL).reshape(
        rows, width
    )
    return path, blocks, expected


def test_selected_rows_match_official_oracle(tmp_path):
    path, blocks, expected = fixture_file(tmp_path)
    cls = load_class()
    with cls(path, data_offset=19, rows=7, width=160) as table:
        ids = np.array([6, 0, 3, 3, 1], dtype=np.int64)
        out = np.empty((len(ids), 160), dtype=np.float32)
        table.gather_rows_small(ids, out)
        np.testing.assert_array_equal(out, expected[ids])
        assert table.packed_bytes == blocks.nbytes
        assert table.layout == "gguf_iq4_nl_int4"


def test_hand_derived_low_then_high_nibbles(tmp_path):
    path = tmp_path / "one-block.bin"
    path.write_bytes(struct.pack("<e", 2.0) + bytes([0xF0] * 16))
    with load_class()(path, data_offset=0, rows=1, width=32) as table:
        out = np.empty((1, 32), np.float32)
        table.gather_rows_small(np.array([0]), out)
        np.testing.assert_array_equal(out[0], [-254.0] * 16 + [226.0] * 16)


@pytest.mark.parametrize(
    "ids", [np.array([-1]), np.array([7]), np.array([1.5]), np.array([[1]])]
)
def test_reject_invalid_ids_before_indexing(tmp_path, ids):
    path, _, _ = fixture_file(tmp_path)
    with (
        load_class()(path, data_offset=19, rows=7, width=160) as table,
        pytest.raises(ValueError),
    ):
        table.gather_rows_small(ids, np.empty((ids.size, 160), dtype=np.float32))


def test_reject_truncated_payload(tmp_path):
    path, _, _ = fixture_file(tmp_path)
    with pytest.raises(ValueError, match="payload"):
        load_class()(path, data_offset=19, rows=8, width=160)


def test_refuse_over_budget(tmp_path):
    path, _, _ = fixture_file(tmp_path)
    with (
        load_class()(
            path, data_offset=19, rows=7, width=160, max_gather_bytes=160 * 4
        ) as table,
        pytest.raises(ValueError, match="budget"),
    ):
        table.gather_rows_small(np.array([0, 1]), np.empty((2, 160), dtype=np.float32))


def test_torch_protocol_matches_oracle(tmp_path):
    import torch

    path, _, expected = fixture_file(tmp_path)
    with load_class()(path, data_offset=19, rows=7, width=160) as table:
        ids = torch.tensor([6, 0, 2])
        out = torch.empty((3, 160), dtype=torch.float16)
        table.gather_into(ids, out)
        torch.testing.assert_close(
            out, torch.from_numpy(expected[[6, 0, 2]]).half(), rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="shape"):
            table.gather_into(ids, torch.empty((1, 3, 160)))


def test_closed_table_refuses_lookup(tmp_path):
    path, _, _ = fixture_file(tmp_path)
    table = load_class()(path, data_offset=19, rows=7, width=160)
    table.close()
    table.close()
    with pytest.raises(ValueError, match="closed"):
        table.gather_rows_small(np.array([0]), np.empty((1, 160), dtype=np.float32))
