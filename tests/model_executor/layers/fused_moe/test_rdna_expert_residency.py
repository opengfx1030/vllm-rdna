# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the default-off routed-expert DRAM residency cache."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.rdna_expert_residency import (
    SideStreamCopier,
    assign_slots,
    expert_dram_offload_enabled,
    expert_dram_offload_requested,
    experts_fit_in_vram,
    install_expert_residency,
    is_routed_expert_parameter,
    next_moe_index,
    owned_local_expert,
    read_popularity_profile,
    rebind_tensor_aliases,
    room_for_experts,
    routed_expert_cpu_weight_context,
    should_engage_expert_offload,
)


class _Experts(torch.nn.Module):
    def __init__(
        self,
        n: int = 4,
        name: str = "model.layers.0.mlp.experts",
        expert_map: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        ids = torch.arange(n, dtype=torch.float32).view(n, 1)
        self.w13_weight = torch.nn.Parameter(ids.clone())
        self.w2_weight = torch.nn.Parameter(ids.clone() + 100)
        self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(n))
        self.ngram_embedding = torch.nn.Parameter(torch.full((n, 1), -7.0))
        self.shared_experts = torch.nn.Parameter(torch.full((n, 1), -3.0))
        self.router = torch.nn.Parameter(torch.zeros(n + 3, 2))
        self.local_num_experts = n
        self.layer_name = name
        self.expert_map = expert_map
        self.alias = SimpleNamespace(scale=self.w13_weight)


def _per_expert_bytes(layer: _Experts) -> int:
    nbytes = 0
    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name)
        nbytes += param.numel() * param.element_size()
    return nbytes // layer.local_num_experts


def test_offload_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_RDNA_MOE_EXPERT_OFFLOAD", raising=False)
    assert expert_dram_offload_requested() is False
    assert expert_dram_offload_enabled() is False


def test_resident_layout_blocks_a_second_offload_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_RDNA_MOE_EXPERT_OFFLOAD", "1")
    monkeypatch.setenv("VLLM_RDNA_MOE_RESIDENT", "1")
    monkeypatch.delenv("VLLM_RDNA_MOE_RESIDENT_SKINNY", raising=False)
    assert expert_dram_offload_requested() is True
    assert expert_dram_offload_enabled() is False


def test_should_engage_only_when_enabled_and_over_budget() -> None:
    assert (
        should_engage_expert_offload(
            enabled=False, expert_bytes=100, vram_budget=1
        )
        is False
    )
    assert (
        should_engage_expert_offload(enabled=True, expert_bytes=10, vram_budget=10)
        is False
    )
    assert (
        should_engage_expert_offload(enabled=True, expert_bytes=10, vram_budget=11)
        is False
    )
    assert (
        should_engage_expert_offload(enabled=True, expert_bytes=0, vram_budget=0)
        is False
    )
    assert (
        should_engage_expert_offload(enabled=True, expert_bytes=11, vram_budget=10)
        is True
    )
    assert experts_fit_in_vram(10, 10) is True
    assert experts_fit_in_vram(11, 10) is False


def test_room_adds_back_experts_already_on_device() -> None:
    room = room_for_experts(
        total=100,
        allocated=80,
        expert_bytes_on_device=50,
        utilization=1.0,
    )
    assert room == 70


def test_assign_slots_raises_when_one_expert_per_layer_does_not_fit() -> None:
    with pytest.raises(RuntimeError, match="one expert from each MoE layer"):
        assign_slots([8, 8], [4, 4], budget=8)


def test_popularity_profile_json_and_text(tmp_path) -> None:
    text = tmp_path / "hot.txt"
    text.write_text("# comment\n0 3 10\n0 1\n", encoding="utf-8")
    assert read_popularity_profile(str(text)) == [(0, 3), (0, 1)]
    blob = tmp_path / "hot.json"
    blob.write_text('{"ranked": [[1, 2], [1, 0]]}', encoding="utf-8")
    assert read_popularity_profile(str(blob)) == [(1, 2), (1, 0)]
    missing = tmp_path / "absent.txt"
    with pytest.raises(FileNotFoundError):
        read_popularity_profile(str(missing))


def test_parameter_filter_skips_non_routed_tensors() -> None:
    weight = torch.zeros(4, 2)
    assert is_routed_expert_parameter("w13_weight", weight, 4) is True
    assert is_routed_expert_parameter("e_score_correction_bias", weight, 4) is False
    assert is_routed_expert_parameter("ngram_embedding", weight, 4) is False
    assert is_routed_expert_parameter("shared_experts", weight, 4) is False
    assert is_routed_expert_parameter("router", torch.zeros(8, 2), 4) is False
    assert is_routed_expert_parameter("ple.weight", weight, 4) is False
    assert is_routed_expert_parameter("mova_v_experts", weight, 4) is False
    assert is_routed_expert_parameter("v_experts", weight, 4) is False


def test_owned_local_expert_is_per_rank() -> None:
    expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
    assert owned_local_expert(expert_map, 1, 4) is None
    assert owned_local_expert(expert_map, 6, 4) == 2
    assert owned_local_expert(None, 2, 4) == 2
    assert owned_local_expert(None, 5, 4) is None


def test_rebind_updates_aliases() -> None:
    old = torch.zeros(2)
    new = torch.ones(2)
    root = SimpleNamespace(inner=SimpleNamespace(scale=old), items=[old])
    rebind_tensor_aliases(root, {id(old): new})
    assert root.inner.scale is new
    assert root.items[0] is new


def test_disabled_install_does_not_build_a_cache() -> None:
    layer = _Experts()
    before = layer.w13_weight.detach().clone()
    engaged = install_expert_residency(
        [(layer, 0)],
        enabled=False,
        vram_budget=1,
        profile=None,
        device=torch.device("cpu"),
    )
    assert engaged is False
    assert torch.equal(layer.w13_weight, before)
    assert layer.w13_weight.shape[0] == 4
    assert not hasattr(layer, "_rdna_expert_residency")


def test_fitting_experts_stay_dead() -> None:
    layer = _Experts()
    engaged = install_expert_residency(
        [(layer, 0)],
        enabled=True,
        vram_budget=10**9,
        profile=[(0, 3)],
        device=torch.device("cpu"),
    )
    assert engaged is False
    assert layer.w13_weight.shape[0] == 4
    assert not hasattr(layer, "_rdna_expert_residency")
    assert torch.equal(
        layer.w13_weight.view(-1),
        torch.arange(4, dtype=torch.float32),
    )


def test_profile_seeds_hot_local_experts_and_miss_fetches_from_host() -> None:
    layer = _Experts()
    copier = SideStreamCopier()
    per_expert = _per_expert_bytes(layer)
    engaged = install_expert_residency(
        [(layer, 0)],
        enabled=True,
        vram_budget=per_expert,
        profile=[(0, 3), (0, 1)],
        device=torch.device("cpu"),
        copier=copier,
    )
    assert engaged is True
    cache = layer._rdna_expert_residency
    assert cache.n_slots == 1
    assert cache.resident_experts == [3]
    assert layer.w13_weight.shape[0] == 1
    assert layer.w2_weight.shape[0] == 1
    assert float(layer.w13_weight[0]) == 3.0
    assert float(layer.w2_weight[0]) == 103.0
    assert layer.e_score_correction_bias.shape[0] == 4
    assert layer.ngram_embedding.shape[0] == 4
    assert float(layer.ngram_embedding[0]) == -7.0
    assert layer.shared_experts.shape[0] == 4
    assert layer.router.shape[0] == 7
    assert layer.alias.scale is layer.w13_weight
    seeded_copies = copier.copies
    remapped = cache.prepare(torch.tensor([[3]]), None)
    assert int(remapped[0, 0]) == 0
    assert copier.copies == seeded_copies
    remapped = cache.prepare(torch.tensor([[1]]), None)
    assert int(remapped[0, 0]) == 0
    assert copier.copies == seeded_copies + 2
    assert float(layer.w13_weight[0]) == 1.0
    assert float(layer.w2_weight[0]) == 101.0
    assert 3 not in cache.resident_experts
    assert cache.resident_experts == [1]


def test_profile_respects_rank_local_experts_and_evicts_colder() -> None:
    expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
    layer = _Experts(expert_map=expert_map)
    copier = SideStreamCopier()
    per_expert = _per_expert_bytes(layer)
    engaged = install_expert_residency(
        [(layer, 0)],
        enabled=True,
        vram_budget=per_expert * 2,
        profile=[(0, 1), (0, 4), (0, 5)],
        device=torch.device("cpu"),
        copier=copier,
    )
    assert engaged is True
    cache = layer._rdna_expert_residency
    # Global 1 is not on this rank. Globals 4 and 5 are local 0 and 1.
    assert cache.resident_experts == [0, 1]
    before = copier.copies
    remapped = cache.prepare(torch.tensor([[6]]), layer.expert_map)
    assert int(remapped[0, 0]) in (0, 1)
    assert 0 in cache.resident_experts
    assert 1 not in cache.resident_experts
    assert 2 in cache.resident_experts
    assert copier.copies == before + 2


def test_capture_does_not_fetch_host_pages() -> None:
    layer = _Experts()
    copier = SideStreamCopier()
    install_expert_residency(
        [(layer, 0)],
        enabled=True,
        vram_budget=_per_expert_bytes(layer),
        profile=[(0, 3)],
        device=torch.device("cpu"),
        copier=copier,
    )
    cache = layer._rdna_expert_residency
    before = copier.copies
    with pytest.raises(RuntimeError, match="hipGraph capture"):
        cache.prepare(torch.tensor([[1]]), None, capturing=True)
    assert copier.copies == before
    assert float(layer.w13_weight[0]) == 3.0
    assert cache.host_pages["w13_weight"][1].item() == 1.0


def test_next_moe_skips_dense_and_linear_layers() -> None:
    kinds = ["dense", "moe", "gdn", "kda", "qsa", "linear", "moe"]
    assert next_moe_index(kinds, 1) == 6
    assert next_moe_index(["moe", "mova", "shared", "moe"], 0) == 3
    assert next_moe_index(kinds, 0) is None


def test_hot_hit_does_not_wait_and_prefetch_overlaps_on_the_other_bank() -> None:
    copier = SideStreamCopier()
    first = _Experts(name="model.layers.0.mlp.experts")
    skipped = _Experts(name="model.layers.1.gdn")
    second = _Experts(name="model.layers.2.mlp.experts")
    per_expert = _per_expert_bytes(first)
    engaged = install_expert_residency(
        [(first, 0), (skipped, 1), (second, 2)],
        enabled=True,
        vram_budget=per_expert * 3,
        profile=[(0, 3), (2, 1), (2, 3)],
        device=torch.device("cpu"),
        copier=copier,
    )
    assert engaged is True
    hot = first._rdna_expert_residency
    cold = second._rdna_expert_residency
    assert hot.successor is cold
    assert cold.layer_index == 2
    assert skipped._rdna_expert_residency.successor is None
    assert hot.resident_experts == [3]
    assert cold.resident_experts == [1]
    seeded = copier.copies
    remapped = hot.prepare(torch.tensor([[3]]), None)
    assert int(remapped[0, 0]) == 0
    assert copier.copies == seeded
    assert copier.compute_waits == 0
    assert copier.stream_syncs == 0
    assert "wait" not in copier.log
    hot.prefetch_successor()
    assert copier.compute_waits == 0
    assert copier.stream_syncs == 0
    assert "wait" not in copier.log
    assert "h2d" in copier.log
    assert copier.experts[-1] == 3
    assert int(copier.dest_ptrs[-1]) not in hot.bound_storage_ids()
    assert int(copier.dest_ptrs[-1]) not in cold.bound_storage_ids()
    assert int(copier.dest_ptrs[-1]) in cold.staging_storage_ids()
    assert float(second.w13_weight[0]) == 1.0
    assert float(cold.banks[1]["w13_weight"][0]) == 3.0
    cold.prepare(torch.tensor([[3]]), None)
    assert copier.compute_waits > 0
    assert copier.stream_syncs == 0
    assert copier.log.index("wait") < copier.log.index("bind")
    assert float(second.w13_weight[0]) == 3.0
    assert float(second.w2_weight[0]) == 103.0


def test_prefetch_does_not_write_a_slot_the_gemm_is_reading() -> None:
    layer = _Experts()
    copier = SideStreamCopier()
    install_expert_residency(
        [(layer, 0)],
        enabled=True,
        vram_budget=_per_expert_bytes(layer),
        profile=[(0, 3)],
        device=torch.device("cpu"),
        copier=copier,
    )
    cache = layer._rdna_expert_residency
    live = cache.banks[0]["w13_weight"]
    assert float(live[0]) == 3.0
    cache.hold_live(cache.bound_storage_ids())
    cache.stage_prefetch(1)
    assert float(layer.w13_weight[0]) == 3.0
    assert float(live[0]) == 3.0
    assert int(copier.dest_ptrs[-1]) not in cache.bound_storage_ids()
    cache.hold_live(cache.staging_storage_ids())
    copies = copier.copies
    cache.stage_prefetch(2)
    assert copier.copies == copies
    cache.prepare(torch.tensor([[1]]), None)
    assert float(live[0]) == 3.0
    assert live.data_ptr() != layer.w13_weight.data_ptr()
    assert float(layer.w13_weight[0]) == 1.0
    assert float(layer.w2_weight[0]) == 101.0


def test_prefetch_is_this_ranks_local_routed_experts_only() -> None:
    expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3])
    layer = _Experts(expert_map=expert_map)
    copier = SideStreamCopier()
    install_expert_residency(
        [(layer, 0)],
        enabled=True,
        vram_budget=_per_expert_bytes(layer),
        profile=[(0, 1), (0, 6), (0, 7)],
        device=torch.device("cpu"),
        copier=copier,
    )
    cache = layer._rdna_expert_residency
    assert cache.resident_experts == [2]
    assert cache.prefetch_local_ids() == [3]
    cache.last_routed_local = [3, 99, -1]
    assert cache.prefetch_local_ids() == [3]
    before = list(copier.experts)
    cache.stage_prefetch(3)
    assert 99 not in copier.experts
    assert 1 not in copier.experts[len(before) :]
    assert copier.experts[-1] == 3


def test_meta_init_is_not_redirected_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_RDNA_MOE_EXPERT_OFFLOAD", "1")
    monkeypatch.delenv("VLLM_RDNA_MOE_RESIDENT", raising=False)
    monkeypatch.delenv("VLLM_RDNA_MOE_RESIDENT_SKINNY", raising=False)
    quant_method = SimpleNamespace(is_monolithic=False)
    with torch.device("meta"):
        with routed_expert_cpu_weight_context(quant_method):
            assert torch.get_default_device().type == "meta"
