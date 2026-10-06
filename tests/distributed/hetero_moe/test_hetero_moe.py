# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for default-off hetero MoE and the cold KV tier.

Run with ``--noconftest``. The root conftest is not required, and this
module stubs ``vllm.distributed`` so the package init (process groups)
is not imported.
"""

from __future__ import annotations

import ast
import json
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _namespace(name: str, path: Path) -> None:
    """Register a package without running its ``__init__``."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


# The engine package init pulls NumPy and the distributed process-group
# setup. These tests only need the hetero modules.
_namespace("vllm", ROOT / "vllm")
_namespace("vllm.distributed", ROOT / "vllm" / "distributed")

import torch  # noqa: E402

from vllm.distributed.hetero_moe.cold_ep import (  # noqa: E402
    ColdPoolRunner,
    expert_owner,
    shard_sizes,
)
from vllm.distributed.hetero_moe.config import (  # noqa: E402
    HeteroMoEConfig,
    load_config,
)
from vllm.distributed.hetero_moe.devices import (  # noqa: E402
    FORBIDDEN_COMPILE_FLAG,
    GFX1030_COMPILE,
    ROCM_PIN,
    resolve_fast_tier,
)
from vllm.distributed.hetero_moe.experts import apply_expert_tables  # noqa: E402
from vllm.distributed.hetero_moe.formulas import (  # noqa: E402
    activation_bytes,
    decode_round_trips,
    transfer_seconds,
)
from vllm.distributed.hetero_moe.gate import hetero_moe_enabled  # noqa: E402
from vllm.distributed.hetero_moe.hot_set import HotSetTracker  # noqa: E402
from vllm.distributed.hetero_moe.kernels import (  # noqa: E402
    select_cold_kernel,
    select_hot_kernel,
)
from vllm.distributed.hetero_moe.kv_tier import (  # noqa: E402
    HeteroKVTier,
    RefusedStateGroup,
    RemoteReadRefused,
)
from vllm.distributed.hetero_moe.pingpong import run_ping_pong  # noqa: E402
from vllm.distributed.hetero_moe.residency import ResidencyBridge  # noqa: E402
from vllm.distributed.hetero_moe.runtime import HeteroRuntime  # noqa: E402
from vllm.distributed.hetero_moe.schedule import run_split_moe  # noqa: E402
from vllm.distributed.hetero_moe.transport import (  # noqa: E402
    HostStagedTransport,
    LoopbackTransport,
    PeerCopyTransport,
    PeerProbe,
    UnverifiedTransport,
)
from vllm.distributed.hetero_moe.wire import (  # noqa: E402
    pack_tensors,
    read_frame,
    unpack_tensors,
    write_frame,
)


def _tables() -> torch.Tensor:
    # Values are exact in fp16 and fp64 so the wire cast does not move them.
    eye = torch.eye(4, dtype=torch.float64)
    return torch.stack([eye * (index + 1) for index in range(4)])


def _expert_fn(tables):
    def apply(hidden, expert_ids, router_weights):
        return apply_expert_tables(hidden, expert_ids, router_weights, tables)

    return apply


def test_flag_off_is_the_default(monkeypatch):
    monkeypatch.delenv("VLLM_HETERO_MOE", raising=False)
    assert hetero_moe_enabled() is False
    assert hetero_moe_enabled({}) is False
    assert hetero_moe_enabled({"VLLM_HETERO_MOE": "0"}) is False
    assert hetero_moe_enabled({"VLLM_HETERO_MOE": "1"}) is True
    monkeypatch.delenv("VLLM_HETERO_MOE_TRANSPORT", raising=False)
    config = load_config()
    assert config.enabled is False
    assert config.transport == "host_staged"
    assert config.kv_tier is False


def test_forward_modular_guard_is_lazy_and_before_apply():
    path = ROOT / "vllm/model_executor/layers/fused_moe/routed_experts.py"
    source = path.read_text(encoding="utf-8")
    module_prefix = source.split("def forward_modular", 1)[0]
    assert "hetero_moe" not in module_prefix
    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward_modular"
    )
    body = ast.get_source_segment(source, fn)
    assert body is not None
    # The docstring mentions quant_method.apply. Check the executable tail.
    code = body[body.rfind('"""') :]
    flag = 'os.environ.get("VLLM_HETERO_MOE", "0") == "1"'
    assert flag in code
    assert code.index(flag) < code.index("return self.quant_method.apply")
    assert "split_routed_forward" in code


def test_publish_does_not_force_the_flag_on():
    source = (ROOT / "vllm/engine/arg_utils.py").read_text(encoding="utf-8")
    start = source.index("def _publish_hetero_moe_env")
    body = source[start : source.index("def __post_init__")]
    assert "if self.hetero_moe:" in body
    assert 'os.environ["VLLM_HETERO_MOE"] = "1"' in body
    assert 'os.environ["VLLM_HETERO_MOE"] = "0"' not in body


def test_hot_set_respects_budget_and_skips_unseen():
    tracker = HotSetTracker(num_experts=4, alpha=1.0)
    tracker.observe(0, torch.tensor([[2, 2], [2, 0]]))
    tracker.observe(1, torch.tensor([[1, 1]]))
    placed = tracker.place(budget_bytes=20, bytes_per_expert=10)
    assert placed[0] == {2}
    assert placed[1] == {1}
    assert 0 not in placed[0]
    assert 3 not in placed[0]
    wider = tracker.place(budget_bytes=30, bytes_per_expert=10)
    assert wider[0] == {2, 0}
    assert wider[1] == {1}
    empty = tracker.place(budget_bytes=0, bytes_per_expert=10)
    assert empty[0] == set()
    assert empty[1] == set()


def test_hot_set_skips_a_layer_that_does_not_fit():
    tracker = HotSetTracker(num_experts=2, alpha=1.0)
    tracker.observe(0, torch.tensor([[0]]))
    tracker.observe(1, torch.tensor([[1]]))
    # Layer 0 is hotter (observe twice more) and too big for the budget.
    tracker.observe(0, torch.tensor([[0]]))
    placed = tracker.place(budget_bytes=50, bytes_per_expert={0: 100, 1: 10})
    assert placed[0] == set()
    assert placed[1] == {1}


def test_rebalance_refuses_capture():
    tracker = HotSetTracker(num_experts=2, alpha=1.0)
    tracker.observe(0, torch.tensor([[0]]))
    tracker.capturing = True
    try:
        tracker.place(budget_bytes=10, bytes_per_expert=10)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    try:
        tracker.observe(0, torch.tensor([[0]]))
        raised_observe = False
    except RuntimeError:
        raised_observe = True
    assert raised_observe


def test_finish_step_places_only_between_steps():
    config = HeteroMoEConfig(
        enabled=True,
        hot_budget_bytes=10,
        transport="loopback",
    )
    runtime = HeteroRuntime(
        config,
        LoopbackTransport(lambda *args: None),
        num_experts=4,
    )
    runtime.note_routing(0, torch.tensor([[1, 1]]))
    assert runtime.hot_experts(0) == set()
    placed = runtime.finish_step(10)
    assert placed[0] == {1}
    runtime.tracker.capturing = True
    try:
        runtime.finish_step(10)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_split_matches_all_local_on_loopback():
    tables = _tables()
    apply = _expert_fn(tables)
    hidden = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    ids = torch.tensor([[0, 2], [1, 3], [2, 0]])
    weights = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]],
        dtype=torch.float64,
    )
    local = apply(hidden, ids, weights)
    transport = LoopbackTransport(apply)
    split = run_split_moe(
        hidden,
        ids,
        weights,
        hot={0, 1},
        expert_fn=apply,
        transport=transport,
    )
    torch.testing.assert_close(split, local, atol=0, rtol=0)
    assert transport.sends == 1
    assert transport.recvs == 1


def test_empty_cold_set_skips_the_send():
    tables = _tables()
    apply = _expert_fn(tables)
    hidden = torch.ones(2, 4, dtype=torch.float64)
    ids = torch.tensor([[0, 1], [1, 0]])
    weights = torch.full((2, 2), 0.5, dtype=torch.float64)
    transport = LoopbackTransport(apply)
    local = apply(hidden, ids, weights)
    split = run_split_moe(
        hidden,
        ids,
        weights,
        hot={0, 1},
        expert_fn=apply,
        transport=transport,
    )
    torch.testing.assert_close(split, local, atol=0, rtol=0)
    assert transport.sends == 0
    assert transport.recvs == 0


def test_all_cold_matches_and_capture_refuses_the_split():
    tables = _tables()
    apply = _expert_fn(tables)
    hidden = torch.ones(2, 4, dtype=torch.float64)
    ids = torch.tensor([[2, 3], [3, 2]])
    weights = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=torch.float64)
    transport = LoopbackTransport(apply)
    split = run_split_moe(
        hidden,
        ids,
        weights,
        hot=set(),
        expert_fn=apply,
        transport=transport,
    )
    torch.testing.assert_close(split, apply(hidden, ids, weights), atol=0, rtol=0)
    assert transport.sends == 1
    try:
        run_split_moe(
            hidden,
            ids,
            weights,
            hot=set(),
            expert_fn=apply,
            transport=transport,
            capturing=True,
        )
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_host_staged_matches_loopback_and_counts_bytes():
    tables = _tables()
    apply = _expert_fn(tables)
    hidden = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float16,
    )
    ids = torch.tensor([[0, 2], [3, 1]])
    weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)
    loop = LoopbackTransport(apply)
    host = HostStagedTransport(apply, link="local")
    shm = HostStagedTransport(apply, link="shm")
    expected = run_split_moe(
        hidden, ids, weights, hot={1}, expert_fn=apply, transport=loop
    )
    got = run_split_moe(hidden, ids, weights, hot={1}, expert_fn=apply, transport=host)
    got_shm = run_split_moe(
        hidden, ids, weights, hot={1}, expert_fn=apply, transport=shm
    )
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    torch.testing.assert_close(got_shm, expected, atol=0, rtol=0)
    assert host.side_stream_ops == 1
    # One cold pair per token (experts 0 and 3). fp16 row is hidden*2.
    assert activation_bytes(hidden.shape[1], cold_pairs=2) == 4 * 2 * 2


def test_tcp_host_link_round_trip():
    tables = _tables()
    apply = _expert_fn(tables)
    hidden = torch.ones(1, 4, dtype=torch.float16)
    ids = torch.tensor([[2, 0]])
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    ready = threading.Event()
    port: dict[str, int] = {}

    def serve() -> None:
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        port["port"] = server.getsockname()[1]
        server.listen(1)
        ready.set()
        conn, _addr = server.accept()
        try:
            request = unpack_tensors(read_frame(conn))
            out = apply(
                request["hidden"],
                request["expert_ids"],
                request["router_weights"],
            )
            write_frame(
                conn,
                pack_tensors({"hidden": out, "token_index": request["token_index"]}),
            )
        finally:
            conn.close()
            server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(timeout=5)
    transport = HostStagedTransport(
        apply,
        link="tcp",
        addr=f"127.0.0.1:{port['port']}",
    )
    try:
        got = run_split_moe(
            hidden,
            ids,
            weights,
            hot={0},
            expert_fn=apply,
            transport=transport,
        )
    finally:
        thread.join(timeout=5)
    expected = run_split_moe(
        hidden,
        ids,
        weights,
        hot={0},
        expert_fn=apply,
        transport=LoopbackTransport(apply),
    )
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_tcp_without_address_raises():
    transport = HostStagedTransport(lambda *args: None, link="tcp", addr="")
    from vllm.distributed.hetero_moe.transport import ColdPayload

    payload = ColdPayload(
        hidden=torch.zeros(1, 2),
        expert_ids=torch.tensor([0]),
        router_weights=torch.tensor([1.0]),
        token_index=torch.tensor([0]),
    )
    try:
        transport.send(payload)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_ping_pong_overlaps_next_attention_and_skips_empty():
    def cold(mb, layer):
        return not (mb == 0 and layer == 0)

    log = run_ping_pong(
        num_microbatches=2,
        num_layers=2,
        attn=lambda mb, layer: None,
        begin_moe=lambda mb, layer: None,
        end_moe=lambda mb, layer: None,
        cold_nonempty=cold,
    )
    assert "send:0:0" not in log
    assert "recv:0:0" not in log
    send = log.index("send:0:1")
    attn = log.index("attn:1:1")
    recv = log.index("recv:0:1")
    assert send < attn < recv
    both = run_ping_pong(
        2,
        1,
        attn=lambda mb, layer: None,
        begin_moe=lambda mb, layer: None,
        end_moe=lambda mb, layer: None,
        cold_nonempty=lambda mb, layer: True,
    )
    assert both == [
        "attn:0:0",
        "send:0:0",
        "attn:1:0",
        "recv:0:0",
        "send:1:0",
        "recv:1:0",
    ]


def test_peer_stays_off_without_a_passing_probe():
    try:
        PeerCopyTransport(None, lambda *args: None)
        raised = False
    except UnverifiedTransport:
        raised = True
    assert raised
    one_way = PeerProbe(
        passed=True,
        measured=True,
        access={(0, 1): True, (1, 0): False},
        nbytes=128,
        seconds=0.5,
    )
    try:
        PeerCopyTransport(one_way, lambda *args: None)
        raised_one = False
    except UnverifiedTransport:
        raised_one = True
    assert raised_one
    # Gate fixture only. Not a measured copy from this machine.
    opened = PeerCopyTransport(
        PeerProbe(
            passed=True,
            measured=True,
            access={(0, 1): True, (1, 0): True},
            nbytes=128,
            seconds=0.5,
        ),
        lambda *args: None,
    )
    assert opened.status == "UNVERIFIED"


def test_formulas_are_symbolic():
    assert activation_bytes(hidden_size=16, cold_pairs=3) == 16 * 2 * 3
    assert decode_round_trips(num_moe_layers=48) == 48
    assert decode_round_trips(0) == 0
    text = transfer_seconds(96, "Bandwidth", "Latency")
    assert text == "(96) / (Bandwidth) + (Latency)"
    assert ROCM_PIN == "7.14.0"
    assert FORBIDDEN_COMPILE_FLAG not in GFX1030_COMPILE
    assert "gfx1030" in GFX1030_COMPILE
    assert "-mno-wavefrontsize64" in GFX1030_COMPILE


def test_fast_tier_is_gfx1100_and_gb10_is_unimplemented():
    tier = resolve_fast_tier("gfx1100")
    assert tier.arch == "gfx1100"
    assert "qsa_heap" in tier.owns
    assert "hot_routed_experts" in tier.owns
    assert "cold_routed_experts" not in tier.owns
    assert select_hot_kernel("gfx1100") == "triton_wna16"
    try:
        select_cold_kernel("gfx1100")
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "gfx1030" in str(exc)
    assert raised
    try:
        resolve_fast_tier("gb10")
        raised_gb = False
    except NotImplementedError:
        raised_gb = True
    assert raised_gb


def test_cold_pool_is_eight_way_and_uses_the_residency_bridge():
    assert shard_sizes(10, 8) == [2, 2, 1, 1, 1, 1, 1, 1]
    assert expert_owner(0, 10, 8) == 0
    assert expert_owner(2, 10, 8) == 1
    assert expert_owner(9, 10, 8) == 7
    assert select_cold_kernel("gfx1030") == "moe_gptq_gemm_rdna2"

    class Cache:
        def __init__(self):
            self.calls = []

        def prepare(self, topk_ids, expert_map, capturing=False):
            if capturing:
                raise RuntimeError("capture")
            self.calls.append("prepare")
            return topk_ids

        def after_resident_gemm(self):
            self.calls.append("after")

    class Module:
        ExpertResidencyCache = object

        def next_moe_index(self, kinds, index):
            return 4

        def should_engage_expert_offload(self, **kwargs):
            return kwargs["expert_bytes"] > kwargs["vram_budget"]

    bridge = ResidencyBridge(Module())
    assert not hasattr(bridge, "n_slots")
    assert not hasattr(bridge, "host_pages")
    assert bridge.next_moe_index(["moe", "gdn", "moe"], 0) == 4
    assert bridge.should_engage(enabled=True, expert_bytes=5, vram_budget=4)
    cache = Cache()

    class Layer:
        cold_kernel = "moe_gptq_gemm_rdna2"
        _expert_residency = cache

    seen = {}

    def apply_fn(layer, x, topk_weights, topk_ids):
        seen["ids"] = topk_ids
        return "ok"

    runner = ColdPoolRunner(
        arch="gfx1030",
        bridge=bridge,
        num_experts=8,
        num_devices=8,
        apply_fn=apply_fn,
    )
    ids = torch.tensor([[1]])
    assert runner.apply(Layer(), torch.zeros(1, 2), torch.ones(1, 1), ids) == "ok"
    assert cache.calls == ["prepare", "after"]
    assert runner.owners([0, 7]) == [0, 7]

    class WrongArch:
        cold_kernel = "moe_gptq_gemm_rdna2"

    try:
        ColdPoolRunner(
            arch="gfx1100",
            bridge=bridge,
            num_experts=8,
            num_devices=8,
            apply_fn=apply_fn,
        ).apply(WrongArch(), torch.zeros(1, 2), torch.ones(1, 1), ids)
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "gfx1030" in str(exc)
    assert raised


def test_residency_bridge_does_not_invent_a_cache():
    try:
        ResidencyBridge().module()
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "PR #37" in str(exc)
    assert raised


def test_kv_lru_evicts_and_copies_back_before_attention():
    tier = HeteroKVTier(capacity_blocks=2)
    assert tier.evict("a", "block-a")
    assert tier.evict("b", "block-b")
    assert tier.evict("c", "block-c")
    assert tier.cold_hashes() == ["b", "c"]
    restored = tier.prepare_for_attention(["b"])
    assert restored["b"] == "block-b"
    assert "b" not in tier.cold_hashes()
    assert tier.cold_hashes() == ["c"]
    try:
        tier.remote_read("c")
        raised = False
    except RemoteReadRefused:
        raised = True
    assert raised
    for group in ("gdn", "ple", "recurrent", "mamba", "qsa_main_kv"):
        try:
            tier.evict("x", "payload", group)
            refused = False
        except RefusedStateGroup:
            refused = True
        assert refused, group
    assert tier.cold_hashes() == ["c"]


def test_qsa_heap_moves_together_or_not_at_all():
    small = HeteroKVTier(capacity_blocks=1)
    assert small.evict_qsa_heap("main", "m", "comp", "c") is False
    assert small.cold_hashes() == []
    tier = HeteroKVTier(capacity_blocks=2)
    assert tier.evict_qsa_heap("main", "m", "comp", "c")
    assert tier.cold_hashes() == [("qsa_heap", "main", "comp")]
    assert tier.on_cold("main") is False
    restored = tier.prepare_for_attention([("qsa_heap", "main", "comp")])
    assert restored[("qsa_heap", "main", "comp")] == ("m", "c")
    assert tier.cold_hashes() == []
    assert tier.evict_qsa_heap("main", "m", "comp", "c")
    assert tier.evict("a", "block-a")
    assert tier.evict("b", "block-b")
    assert ("qsa_heap", "main", "comp") not in tier.cold_hashes()
    assert "main" not in tier.cold_hashes()
    assert "comp" not in tier.cold_hashes()


def test_probe_script_does_not_invent_a_bandwidth():
    script = ROOT / "tools/rdna2/probe_hetero_peer.py"
    text = script.read_text(encoding="utf-8")
    assert "GB/s" not in text
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)
    if report["measured"]:
        assert report["nbytes"] > 0
        assert report["seconds"] > 0
        assert report["bandwidth_bytes_per_sec"] == (
            report["nbytes"] / report["seconds"]
        )
    else:
        assert report["bandwidth_bytes_per_sec"] is None
        assert report["nbytes"] is None
        assert "not measured" in report["note"]
