# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavior tests for benchmarks/kernels/check_rdna2_offload_tp2_ar.py.

The qualified TP2 pair used to be hardcoded everywhere (world 2, ranks
[0, 1], probe sum 3, 256 local experts, two spawned workers). These tests
run the real main() with the spawn call and the device probe replaced, so
they observe the resolved topology and the CLI refusals rather than source
text. The GPU collective itself is exercised only on hardware.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmarks" / "kernels" / "check_rdna2_offload_tp2_ar.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ar_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ar = _load_module()


def _fake_spawn(calls):
    def spawn(fn, args, nprocs, join):
        assert fn is ar._worker and join is True
        calls.append({"args": args, "nprocs": nprocs})

    return spawn


def _run_main(monkeypatch, argv, devices):
    calls = []
    monkeypatch.setattr(ar.mp, "spawn", _fake_spawn(calls))
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    monkeypatch.setattr(sys, "argv", ["check_rdna2_offload_tp2_ar.py", *argv])
    ar.main()
    assert len(calls) == 1
    return calls[0]


def test_topology_tp2_matches_maintainer_geometry():
    # The merged TP2 qualification: 256 local experts, shared width 320,
    # probe 1.0+2.0 = 3.0.
    assert ar._topology(2) == (256, 320, 3)


def test_topology_tp4_owns_quarter_of_the_model():
    # EP4: 512/4 = 128 local experts (the resident-K2 oracle width), 640/4
    # shared width, probe sum 1+2+3+4 = 10.
    assert ar._topology(4) == (128, 160, 10)


@pytest.mark.parametrize("tp", [1, 3, 5, 8])
def test_topology_rejects_unqualified_world_sizes(tp):
    with pytest.raises(ValueError):
        ar._topology(tp)


def test_default_invocation_is_the_tp2_pair(monkeypatch):
    call = _run_main(monkeypatch, [], devices=2)
    options, port = call["args"]
    assert call["nprocs"] == 2 and options.tp == 2
    assert options.rows == [24, 12, 6, 3, 1] and options.slots == 3
    assert isinstance(port, int)


def test_tp4_spawns_four_workers_with_quarter_geometry(monkeypatch):
    call = _run_main(monkeypatch, ["--tp", "4", "--slots", "6"], devices=4)
    options, _ = call["args"]
    assert call["nprocs"] == 4 and options.tp == 4 and options.slots == 6


def test_tp4_slots_may_reach_the_local_expert_count(monkeypatch):
    call = _run_main(monkeypatch, ["--tp", "4", "--slots", "128"], devices=4)
    assert call["args"][0].slots == 128


@pytest.mark.parametrize(
    "argv, devices",
    [
        (["--tp", "3"], 3),  # not a qualified topology
        (["--tp", "8"], 8),  # rdna_ar supports world 8; qualification does not
        (["--tp", "4"], 2),  # four-card plan on a two-card view
        (["--tp", "2"], 4),  # stale pair while four cards are visible
        (["--tp", "4", "--slots", "129"], 4),  # slot overrun past 128 local
        (["--tp", "2", "--slots", "257"], 2),  # old TP2 bound still enforced
        (["--tp", "4", "--rows", "25"], 4),  # row width gate unchanged
        (["--tp", "4", "--rows", "8,0"], 4),
    ],
)
def test_main_refuses_invalid_topologies(monkeypatch, argv, devices):
    monkeypatch.setattr(ar.mp, "spawn", _fake_spawn([]))
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: True)
    monkeypatch.setattr(sys, "argv", ["check_rdna2_offload_tp2_ar.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        ar.main()
    assert exit_info.value.code == 2


def test_main_refuses_peer_incapped_host_with_reason(monkeypatch, capsys):
    """P2P-incapable hosts (e.g. four V620s behind separate PCIe switches)
    must be refused with the actionable reason, not left to fail as an
    opaque hipDeviceEnablePeerAccess error inside rdna_ar_init."""
    monkeypatch.setattr(ar.mp, "spawn", _fake_spawn([]))
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda a, b: False)
    monkeypatch.setattr(
        sys, "argv", ["check_rdna2_offload_tp2_ar.py", "--tp", "4", "--slots", "48"]
    )
    with pytest.raises(SystemExit) as exit_info:
        ar.main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "peer access" in err and "RCCL" in err
