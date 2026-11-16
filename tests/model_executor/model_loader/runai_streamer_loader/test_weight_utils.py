# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
    safetensors_weights_iterator,
)


def test_runai_safetensors_weights_iterator_clones_reused_buffers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "0")
    weights_file = tmp_path / "model.safetensors"
    expected_tensors = {
        "first": torch.tensor([1.0, 2.0]),
        "second": torch.tensor([3.0, 4.0]),
    }
    save_file(expected_tensors, weights_file)

    actual_tensors = dict(
        runai_safetensors_weights_iterator([str(weights_file)], False)
    )

    assert actual_tensors.keys() == expected_tensors.keys()
    assert actual_tensors["first"].data_ptr() != actual_tensors["second"].data_ptr()
    for name, expected_tensor in expected_tensors.items():
        assert torch.equal(actual_tensors[name], expected_tensor)


@pytest.mark.parametrize(
    "iterator", [runai_safetensors_weights_iterator, safetensors_weights_iterator]
)
def test_manifest_selection_preserves_only_the_selected_view(
    tmp_path, monkeypatch, iterator
):
    """A mixed target/draft shard must not leak draft components into the target."""
    monkeypatch.setenv("RUNAI_STREAMER_MEMORY_LIMIT", "0")
    weights_file = tmp_path / "mixed.safetensors"
    tensors = {
        "target.weight": torch.tensor([[0, 2**31, 2**32 - 1]], dtype=torch.uint32),
        "target.scales": torch.tensor([[2**-130]], dtype=torch.bfloat16),
        "target.biases": torch.tensor([[-(2**20)]], dtype=torch.bfloat16),
        "mtp.weight": torch.tensor([[17, 29, 43]], dtype=torch.uint32),
        "mtp.scales": torch.tensor([[1]], dtype=torch.bfloat16),
        "mtp.biases": torch.tensor([[0]], dtype=torch.bfloat16),
    }
    save_file(tensors, weights_file)
    for prefix in ("target.", "mtp."):
        selected = frozenset(name for name in tensors if name.startswith(prefix))
        actual = dict(
            iterator([str(weights_file)], False, selected_tensor_names=selected)
        )
        assert actual.keys() == selected
        for name in selected:
            assert actual[name].dtype == tensors[name].dtype
            assert torch.equal(actual[name], tensors[name])


def test_manifest_selection_rejects_whole_shard_or_gpu_staging():
    with pytest.raises(ValueError, match="CPU staging"):
        list(
            runai_safetensors_weights_iterator(
                [], False, True, selected_tensor_names=frozenset({"target.weight"})
            )
        )
    with pytest.raises(ValueError, match="lazy CPU"):
        list(
            safetensors_weights_iterator(
                [], False, "eager", selected_tensor_names=frozenset({"target.weight"})
            )
        )


def test_runai_model_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
        )
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0

        runai_model_streamer_tensors = {}
        hf_safetensors_tensors = {}

        for name, tensor in runai_safetensors_weights_iterator(safetensors, True):
            runai_model_streamer_tensors[name] = tensor

        for name, tensor in safetensors_weights_iterator(safetensors, True):
            hf_safetensors_tensors[name] = tensor

        assert len(runai_model_streamer_tensors) == len(hf_safetensors_tensors)

        for name, runai_tensor in runai_model_streamer_tensors.items():
            assert runai_tensor.dtype == hf_safetensors_tensors[name].dtype
            assert runai_tensor.shape == hf_safetensors_tensors[name].shape
            assert torch.all(runai_tensor.eq(hf_safetensors_tensors[name]))


if __name__ == "__main__":
    test_runai_model_loader()
