# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned-host mlock GPU proof driver and its serve_validate.sh hooks, run
against stub launchers: the dry run, the gating, a full run of four arms with
the verdict, the retries, and the quiet-window wait. No GPU, no server."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PORT_V031 = ROOT / "tools/rdna/port_v031"
LINUX = sys.platform.startswith("linux")

STUB_LAUNCHER = """#!/usr/bin/env bash
echo "# ple:       bf16"
echo "export GPU_MAX_HW_QUEUES='2'"
echo "'--speculative-config' EXTRA_ARGS=${EXTRA_ARGS:-}"
"""

# What serve_validate.sh leaves behind for one arm, derived from the env it is
# called with (VLLM_ROCM_MLOCK_PINNED_HOST): the lock lines of the fix arm and
# the evictions of the baseline, so the driver's gating, retries and verdict run
# for real without a server or a GPU.
STUB_SERVE_VALIDATE = r"""#!/usr/bin/env bash
out=$OUT
mkdir -p "$out/proof"
{
    echo "$TAG mlock=${VLLM_ROCM_MLOCK_PINNED_HOST:-unset} quiet=${QUIET_DIR:-}" \
        "own=${QUIET_OWN:-} hold=${QUIET_HOLD:-} grace=${STOP_GRACE:-}" \
        "lock=${BOOT_LOCK:-} hook=${LOAD_HOOK:-} cells=${CELLS:-}" \
        "hip=${HIP_VISIBLE_DEVICES:-} neighbour=${NEIGHBOUR:-}" \
        "refused=${REFUSED_GPUS-unset}" \
        "lease=$(ls "$STUB_LEASE" 2> /dev/null | tr '\n' ',')"
} >> "$STUB_LOG"
n=$(grep -c "^$TAG " "$STUB_LOG")
: > "$out/serve.log"
for r in 0 1 2 3; do
    echo "rdna_ar: one-shot all-reduce active (handle 0, rank $r/4)" >> "$out/serve.log"
    if [[ ${VLLM_ROCM_MLOCK_PINNED_HOST} == 1 && -z ${STUB_NO_LOCK_LINES:-} ]]; then
        echo "(Worker_TP$r pid=$((100 + r))) INFO 11-05 13:00:00 [pinned_host.py:200]" \
            "Locked 329 of 329 pinned host mappings (1.06 GiB);" \
            "VmLck 23.84 -> 24.90 GiB." >> "$out/serve.log"
    fi
done
echo "${STUB_SERVE_PID:-1999999}" > "$out/serve.pid"
sleep "${STUB_HOLD:-0}"
vm=23.84
[[ ${VLLM_ROCM_MLOCK_PINNED_HOST} == 1 ]] && vm=24.9
ev=${STUB_BASE_MS:-7100}
[[ ${VLLM_ROCM_MLOCK_PINNED_HOST} == 1 ]] && ev=${STUB_FIX_MS:-12}
"$REAL_PY" - "$out" "$ev" "$vm" <<'PY'
import json, sys
out, ev, vm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
workers = [f"Worker_TP{i}" for i in range(4)]
win = lambda name, ms: {"name": name, "evicted_ms": {w: ms for w in workers},
                        "vmlck": {w: vm for w in workers}}
json.dump({"windows": [win("idle", 0), win("load1", ev), win("load2", 0)],
           "vmlck_end": {w: vm for w in workers}},
          open(f"{out}/proof/windows.json", "w"))
for cell, ttft in (("bench-1024x512-c1", 300.0), ("bench-1024x512-c8", 900.0)):
    json.dump({"median_ttft_ms": ttft, "p99_ttft_ms": ttft * 1.2,
               "median_itl_ms": 22.0, "p99_itl_ms": 30.0,
               "output_throughput": 100.0}, open(f"{out}/{cell}.json", "w"))
PY
hook=PASS
if [[ -n ${STUB_FAIL_ONCE:-} && $TAG == *"-$STUB_FAIL_ONCE" && $n == 1 ]]; then
    hook=FAIL
fi
{
    echo "[0] READY in 10s served=flash-next"
    echo "[0] bench bench-1024x512-c1 (n=4)"
    echo "    evicted_ms Worker_TP0=+0 -> ok"
    if [[ $hook == PASS ]]; then
        echo "[0] LOAD HOOK rc=0 PASS"
    else
        echo "[0] LOAD HOOK rc=1 FAIL"
    fi
} > "$out/summary.txt"
[[ $hook == PASS ]]
"""

# The HIP index of each PCI bus given, as the lease README lists them
# (A: ce d3 d8 db = 1-4, B: 0a 0f 14 17 = 6-9). STUB_HIP_SHIFT moves them, from
# the STUB_HIP_SHIFT_FROM-th call on: the topology changing under the driver.
RESOLVER = r"""#!/usr/bin/env bash
n=$(( $(cat "$STUB_LOG.resolves" 2> /dev/null || echo 0) + 1 ))
echo "$n" > "$STUB_LOG.resolves"
shift_by=${STUB_HIP_SHIFT:-0}
(( n < ${STUB_HIP_SHIFT_FROM:-1} )) && shift_by=0
out=
for b in "$@"; do
    case $b in
        ce) i=1 ;; d3) i=2 ;; d8) i=3 ;; db) i=4 ;;
        0a) i=6 ;; 0f) i=7 ;; 14) i=8 ;; 17) i=9 ;;
        cb) i=${STUB_CB:-0} ;; 07) i=${STUB_07:-5} ;;
        *) exit 1 ;;
    esac
    out+=${out:+,}$(( i + shift_by ))
done
echo "$out"
"""


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    tree = tmp_path / "tree"
    tools = tree / "tools/rdna"
    (tools / "port_v031").mkdir(parents=True)
    for name in (
        "mlock_pinned_proof.sh",
        "mlock_pinned_hook.sh",
        "pinned_proof.py",
        "pinned_report.py",
        "kfd_evictions.py",
        "spec_acceptance.py",
    ):
        shutil.copy(PORT_V031 / name, tools / "port_v031" / name)
    (tools / "serve_rdna.sh").write_text(STUB_LAUNCHER)
    (tools / "port_v031/serve_validate.sh").write_text(STUB_SERVE_VALIDATE)
    (tree / ".sha").write_text("0123456789abcdef\n")
    (tree / "vllm/utils").mkdir(parents=True)
    (tree / "vllm/utils/pinned_host.py").write_text("")
    for so in ("_C", "_rocm_C"):
        (tree / f"vllm/{so}.abi3.so").write_text("")
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    # a python whose -c is the free-memory query (30 GiB on each of four cards)
    (venv / "python").write_text(
        "#!/usr/bin/env bash\n"
        '[[ ${1:-} == -c ]] && { echo "30.0 30.0 30.0 30.0"; exit 0; }\n'
        'exec "$REAL_PY" "$@"\n'
    )
    (venv / "python").chmod(0o755)
    (tmp_path / "model").mkdir()
    (tmp_path / "home").mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "sudo").write_text(
        "#!/usr/bin/env bash\n[[ $1 == -n && $2 == true ]] && exit 0\nexit 1\n"
    )
    (bindir / "sysctl").write_text(
        "#!/usr/bin/env bash\n"
        "case $2 in\n"
        '    vm.compact_unevictable_allowed) echo "${STUB_COMPACT:-0}" ;;\n'
        "    *) echo 60 ;;\n"
        "esac\n"
    )
    for name in ("sudo", "sysctl"):
        (bindir / name).chmod(0o755)
    for d in ("lease", "lanes/quiet", "run"):
        (tmp_path / d).mkdir(parents=True)
    (bindir / "busresolver").write_text(RESOLVER)
    (bindir / "busresolver").chmod(0o755)
    for bus in ("ce", "d3", "d8", "db", "0a", "0f", "14", "17"):
        dev = tmp_path / f"pci/0000:{bus}:00.0"
        dev.mkdir(parents=True)
        (dev / "device").write_text("0x73a1\n")  # Navi 21, the V620
    return tree


def run_group(
    cmd: list[str], env: dict[str, str], timeout: float
) -> "subprocess.CompletedProcess[str]":
    """``subprocess.run`` in a session of its own, killed as a group on a timeout."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


class Proof:
    def __init__(self, tree: Path) -> None:
        self.tree = tree
        self.base = tree.parent

    def env(self, **env: str) -> dict[str, str]:
        base = self.base
        return {
            "PATH": f"{base / 'bin'}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(base / "home"),
            "RUN": str(base / "run"),
            "LEASE": str(base / "lease"),
            "LANES": str(base / "lanes"),
            "BOOT_LOCK": str(base / "boot.lock"),
            "VENV": str(base / "venv"),
            "MODEL": str(base / "model"),
            "EXPECT_SHA": "0123456789abcdef",
            "REAL_PY": sys.executable,
            "STUB_LOG": str(base / "stub.log"),
            "STUB_LEASE": str(base / "lease"),
            "BUS_RESOLVER": str(base / "bin/busresolver"),
            "PCI_SYSFS": str(base / "pci"),
            "MEMLOCK_MIN_KB": "0",
            "TEMP_GUARD": "0",  # the guard has its own tests
            **env,
        }

    def __call__(
        self, *, ulimit_l: int | None = None, timeout: float = 120, **env: str
    ) -> "subprocess.CompletedProcess[str]":
        script = str(self.tree / "tools/rdna/port_v031/mlock_pinned_proof.sh")
        cmd = ["bash", script]
        if ulimit_l is not None:
            cmd = ["bash", "-c", f'ulimit -S -l {ulimit_l} && exec bash "$0"', script]
        return run_group(cmd, self.env(**env), timeout)

    def stub_log(self) -> list[str]:
        p = self.base / "stub.log"
        return p.read_text().splitlines() if p.exists() else []


@pytest.fixture()
def proof(tree: Path) -> Proof:
    p = Proof(tree)
    (p.base / "run/GO").write_text("")
    return p


# ---- the dry run and what it refuses ---------------------------------------------


def test_dry_run_hands_each_arm_its_lock_setting(proof: Proof):
    res = proof(DRY="1", ARMS="base fix fix2")
    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout
    assert "arm base: hook off: only the PLE table is locked; " in out
    assert "arm fix: every pinned host mapping locked after warm-up; " in out
    assert out.count("VLLM_ROCM_MLOCK_PINNED_HOST=0") == 1
    assert "arm fix2:" in out and out.count("VLLM_ROCM_MLOCK_PINNED_HOST=1") == 2
    assert "slot A: HIP 1,2,3,4 = V620 buses ce d3 d8 db" in out
    assert (
        "neighbour: replay x 3; one quiet window per load: idle 30s settle 30s "
        "gap 120s (at least 120 s)" in out
    )
    assert not proof.stub_log()  # nothing booted, no lease taken


def test_dry_run_prints_the_ple_override_of_the_arms(proof: Proof):
    res = proof(DRY="1", ARMS="fix", PLE_MODE="int4")
    assert res.returncode == 0 and "overrides=PLE=int4" in res.stdout


@pytest.mark.parametrize(
    "env, needle",
    [
        ({"EXPECT_SHA": ""}, "EXPECT_SHA is required"),
        ({"EXPECT_SHA": "deadbeef"}, "is not deadbeef"),
        ({"GPUS": "0,1,2,3"}, "GPUS=0,1,2,3 is not slot A (HIP 1,2,3,4"),
        ({"GPUS": "2,3,4,5"}, "GPUS=2,3,4,5 is not slot A"),
        ({"SLOT": "C"}, "unknown SLOT 'C'"),
        ({"SLOT": "B", "GPUS": "1,2,3,4"}, "GPUS=1,2,3,4 is not slot B (HIP 6,7,8,9"),
        ({"ARMS": "base quux"}, "unknown arm quux"),
        ({"NEIGHBOUR": "cmd"}, "NEIGHBOUR=cmd needs NEIGHBOUR_CMD"),
    ],
)
def test_dry_run_refuses(proof: Proof, env: dict[str, str], needle: str):
    res = proof(DRY="1", **env)
    assert res.returncode == 2 and needle in res.stdout, res.stdout + res.stderr


def test_it_needs_the_fix_tree(proof: Proof):
    (proof.tree / "vllm/utils/pinned_host.py").unlink()
    res = proof(DRY="1")
    assert res.returncode == 2 and "not the fix tree" in res.stdout


def test_it_needs_the_built_extensions(proof: Proof):
    (proof.tree / "vllm/_rocm_C.abi3.so").unlink()
    res = proof(DRY="1")
    assert res.returncode == 2 and "_rocm_C.abi3.so is missing" in res.stdout


@pytest.mark.skipif(not LINUX, reason="ulimit -l")
def test_it_refuses_a_low_memlock_limit(proof: Proof):
    res = proof(ulimit_l=16, MEMLOCK_MIN_KB="33554432")  # below any hard limit
    assert res.returncode == 2 and "RLIMIT_MEMLOCK is 16 kB" in res.stdout
    assert not proof.stub_log()


def test_it_refuses_when_compaction_may_move_locked_pages(proof: Proof):
    res = proof(STUB_COMPACT="1")
    assert res.returncode == 2
    assert "vm.compact_unevictable_allowed is 1" in res.stdout
    assert not proof.stub_log()


# ---- a full run against the stub serve_validate.sh --------------------------------


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_four_arms_boot_in_abba_order_and_the_proof_holds(proof: Proof):
    res = proof(ARMS="base fix fix2 base2")
    assert res.returncode == 0, res.stdout + res.stderr
    base = proof.base
    for arm in ("base", "fix", "fix2", "base2"):
        assert (base / f"run/arms/{arm}/DONE").exists(), arm
    assert (base / "run/DONE").exists()
    verdict = (base / "run/results/verdict.md").read_text()
    assert verdict.splitlines()[0] == "# Pinned-host mlock proof: PROVEN", verdict
    assert json.loads((base / "run/results/verdict.json").read_text())[
        "reproduced"
    ] == [
        "base",
        "base2",
    ]
    log = proof.stub_log()
    assert [ln.split()[0] for ln in log] == [
        "mlock-proof-base",
        "mlock-proof-fix",
        "mlock-proof-fix2",
        "mlock-proof-base2",
    ]
    assert [ln.split()[1] for ln in log] == ["mlock=0", "mlock=1", "mlock=1", "mlock=0"]
    first = log[0]
    quiet = base / "lanes/quiet"
    assert f"quiet={quiet} own=EvictFix hold=1 grace=300" in first
    assert f"lock={base / 'boot.lock'}" in first
    assert "hook=" in first and first.count("mlock_pinned_hook.sh") == 1
    assert "cells=1024:512:1 1024:512:8" in first and "hip=1,2,3,4" in first
    # serve_validate.sh's default refusal is the old topology's HIP 0,1: the W7800s
    # are handed over by bus (HIP 0 = cb, 5 = 07)
    assert all("refused=0,5" in ln for ln in log)
    # the lease while the arm boots: our request and holder of the slot; all gone after
    assert "lease=EvictFix.slotA,slotA.request," in first
    assert list((base / "lease").iterdir()) == []
    assert not (base / "lanes/quiet/EvictFix").exists()
    assert "neighbour=replay" in first
    arm_txt = (base / "run/arms/fix/arm.txt").read_text()
    assert (
        "VLLM_ROCM_MLOCK_PINNED_HOST=1" in arm_txt
        and "compact_unevictable=0" in arm_txt
    )


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_the_verdict_is_inconclusive_when_the_baseline_does_not_evict(proof: Proof):
    res = proof(ARMS="base fix", STUB_BASE_MS="30")
    assert res.returncode == 0, res.stdout + res.stderr
    verdict = (proof.base / "run/results/verdict.md").read_text()
    assert verdict.splitlines()[0] == "# Pinned-host mlock proof: INCONCLUSIVE"


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_the_verdict_fails_when_the_fix_arm_is_still_evicted(proof: Proof):
    res = proof(ARMS="base fix", STUB_FIX_MS="1800")
    assert res.returncode == 0, res.stdout + res.stderr
    verdict = (proof.base / "run/results/verdict.md").read_text()
    assert "FAILED" in verdict.splitlines()[0] and "evicted 1800 ms" in verdict


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_a_fix_arm_without_lock_lines_is_a_mismatch_not_a_retry(proof: Proof):
    res = proof(ARMS="fix", STUB_NO_LOCK_LINES="1")
    assert res.returncode == 1, res.stdout + res.stderr
    assert "0 'Locked N of M' lines, expected 4" in res.stdout
    assert len(proof.stub_log()) == 1  # not booted again
    assert not (proof.base / "run/arms/fix/DONE").exists()
    assert list((proof.base / "lease").iterdir()) == []


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_an_arm_whose_hook_failed_is_booted_again(proof: Proof):
    res = proof(ARMS="fix", STUB_FAIL_ONCE="fix")
    assert res.returncode == 0, res.stdout + res.stderr
    assert len(proof.stub_log()) == 2
    assert (proof.base / "run/arms/fix/attempt1/serve.log").exists()
    assert (proof.base / "run/arms/fix/DONE").exists()
    assert "boot or hook failed" in res.stdout


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_stop_file_ends_the_run_between_arms(proof: Proof):
    (proof.base / "run/STOP").write_text("")
    res = proof(ARMS="base fix")
    assert res.returncode == 3, res.stdout + res.stderr
    assert not proof.stub_log() and list((proof.base / "lease").iterdir()) == []


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_it_waits_for_the_other_holders_of_its_slot(proof: Proof):
    lease = proof.base / "lease"
    (lease / "TwoLaneFeat.slotA").write_text("TwoLaneFeat\n")
    (lease / "TwoLaneFeat.slotB").write_text("TwoLaneFeat\n")  # not our slot
    started = time.time()
    p = subprocess.Popen(
        ["bash", str(proof.tree / "tools/rdna/port_v031/mlock_pinned_proof.sh")],
        env=proof.env(ARMS="base"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        time.sleep(4)
        assert p.poll() is None and not proof.stub_log()
        assert (lease / "slotA.request").read_text().startswith("EvictFix")
        (proof.base / "run/STOP").write_text("")
        p.communicate(timeout=90)  # the 60 s poll sees STOP
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert p.returncode == 3 and time.time() - started < 100
    assert not (lease / "slotA.request").exists()  # its own request is withdrawn
    assert (lease / "TwoLaneFeat.slotA").exists()  # the other holder is untouched


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_it_queues_behind_a_request_of_another_job(proof: Proof):
    lease = proof.base / "lease"
    (lease / "slotA.request").write_text("TopoSmoke 2026-11-05T15:58 slot A\n")
    p = subprocess.Popen(
        ["bash", str(proof.tree / "tools/rdna/port_v031/mlock_pinned_proof.sh")],
        env=proof.env(ARMS="base"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        time.sleep(4)
        assert p.poll() is None and not proof.stub_log()
        assert (lease / "slotA.request").read_text().startswith("TopoSmoke")
        (proof.base / "run/STOP").write_text("")
        p.communicate(timeout=90)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert p.returncode == 3
    # someone else's request is never removed
    assert (lease / "slotA.request").read_text().startswith("TopoSmoke")


# ---- the slot and its buses ---------------------------------------------------------


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_slot_b_runs_on_its_own_cards_and_lease_files(proof: Proof):
    res = proof(ARMS="base", SLOT="B")
    assert res.returncode == 0, res.stdout + res.stderr
    first = proof.stub_log()[0]
    assert "hip=6,7,8,9" in first and "refused=0,5" in first
    assert "slot B verified: HIP 6,7,8,9 = V620 buses 0a 0f 14 17" in res.stdout
    # holder and request of slot B, none of slot A, while the arm booted
    assert "EvictFix.slotB,slotB.request," in first
    assert list((proof.base / "lease").iterdir()) == []


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_it_refuses_when_the_hip_order_moved(proof: Proof):
    res = proof(ARMS="base", STUB_HIP_SHIFT="1")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "slot A (buses ce d3 d8 db) is HIP 2,3,4,5 now, not 1,2,3,4" in res.stdout
    assert not proof.stub_log() and list((proof.base / "lease").iterdir()) == []


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_it_refuses_a_slot_that_holds_a_w7800(proof: Proof):
    # the W7800 on bus cb resolves to HIP 3, which slot A has: refuse before any lease
    res = proof(ARMS="base", STUB_CB="3")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "HIP 3 of slot A is a W7800 (buses cb 07 = HIP 3,5)" in res.stdout
    assert not proof.stub_log() and list((proof.base / "lease").iterdir()) == []


def make_hwmon(base: Path, temps: dict[str, int]) -> Path:
    """A fake /sys/class/hwmon with one amdgpu per PCI bus (millidegrees)."""
    root = base / "hwmon"
    for i, (bus, mc) in enumerate(temps.items()):
        h = root / f"hwmon{i}"
        h.mkdir(parents=True)
        (h / "name").write_text("amdgpu\n")
        (h / "device").symlink_to(base / f"pci/0000:{bus}:00.0")
        (h / "temp2_input").write_text(f"{mc}\n")
    return root


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_the_temperature_guard_logs_the_slot_during_each_arm(proof: Proof):
    root = make_hwmon(proof.base, {"ce": 61000, "d3": 62000, "d8": 63000, "db": 64000})
    res = proof(ARMS="base fix", TEMP_GUARD="1", HWMON=str(root))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "no temperature guard" not in res.stdout
    for arm in ("base", "fix"):
        log = (proof.base / f"run/arms/{arm}/temps.log").read_text().splitlines()
        assert log and all(
            ln.split(" ", 1)[1] == "ce=61 d3=62 d8=63 db=64" for ln in log
        )


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_a_thermal_breach_stops_the_server_with_sigterm_and_the_proof(proof: Proof):
    root = make_hwmon(proof.base, {"ce": 61000, "d3": 96000, "d8": 63000, "db": 64000})
    server = subprocess.Popen(["sleep", "120"], start_new_session=True)
    try:
        res = proof(
            ARMS="base fix",
            TEMP_GUARD="1",
            HWMON=str(root),
            STUB_SERVE_PID=str(server.pid),
            STUB_HOLD="4",  # the guard looks for the pid file once a second
        )
        assert res.returncode == 3, res.stdout + res.stderr
        assert "THERMAL ABORT:" in res.stdout and "d3 sample 96 C >= 95" in res.stdout
        assert [ln.split()[0] for ln in proof.stub_log()] == ["mlock-proof-base"]
        assert server.wait(timeout=10) == -signal.SIGTERM  # not SIGKILL
        assert list((proof.base / "lease").iterdir()) == []
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_a_missing_temperature_sensor_is_a_warning_not_a_stop(proof: Proof):
    root = make_hwmon(proof.base, {"ce": 61000})  # d3, d8, db have none
    res = proof(ARMS="base", TEMP_GUARD="1", HWMON=str(root))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARNING: no temperature guard: no amdgpu temperature sensor" in res.stdout


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_it_refuses_a_bus_that_is_not_a_v620(proof: Proof):
    (proof.base / "pci/0000:d8:00.0/device").write_text("0x7449\n")  # a W7800
    res = proof(ARMS="base")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "PCI bus d8 of slot A is device 0x7449, not a V620" in res.stdout
    assert not proof.stub_log()


@pytest.mark.skipif(not LINUX, reason="GNU find/stat of the lease code")
def test_it_checks_the_buses_again_before_every_arm(proof: Proof):
    # each check asks twice (the slot's buses, then the W7800s): calls 1-2 are after
    # GO, 3-4 before the base arm; the order moves for call 5, before the fix arm
    res = proof(ARMS="base fix", STUB_HIP_SHIFT="1", STUB_HIP_SHIFT_FROM="5")
    assert res.returncode == 2, res.stdout + res.stderr
    assert "is HIP 2,3,4,5 now, not 1,2,3,4" in res.stdout
    assert [ln.split()[0] for ln in proof.stub_log()] == ["mlock-proof-base"]
    assert (proof.base / "run/arms/base/DONE").exists()
    assert not (proof.base / "run/arms/fix/DONE").exists()
    assert list((proof.base / "lease").iterdir()) == []  # released on the way out


# ---- serve_validate.sh: the quiet-window protocol ---------------------------------


def _sv_setup(tmp_path: Path, **env: str) -> tuple[list[str], dict[str, str]]:
    tree = tmp_path / "svtree"
    (tree / "tools/rdna/port_v031").mkdir(parents=True, exist_ok=True)
    shutil.copy(PORT_V031 / "serve_validate.sh", tree / "tools/rdna/port_v031/")
    stub = tmp_path / "serve_stub.sh"
    stub.write_text("#!/usr/bin/env bash\nexit 1\n")  # the boot fails at once
    full = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "RECIPE": "r",
        "MODEL": str(tmp_path),
        "GPUS": "2,3,4,5",
        "OUT": str(tmp_path / "out"),
        "CACHE": str(tmp_path / "cache"),
        "VENV": str(tmp_path / "venv"),
        "SERVE_SCRIPT": str(stub),
        **env,
    }
    return ["bash", str(tree / "tools/rdna/port_v031/serve_validate.sh")], full


def _serve_validate(tmp_path: Path, **env: str) -> "subprocess.CompletedProcess[str]":
    cmd, full = _sv_setup(tmp_path, **env)
    return run_group(cmd, full, timeout=120)


@pytest.mark.skipif(not LINUX, reason="GNU find -mmin")
def test_serve_validate_waits_for_a_foreign_quiet_file_and_not_for_its_own(
    tmp_path: Path,
):
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    (quiet / "EvictFix").write_text("")  # ours: ignored
    (quiet / "EvictFix.mlock-proof-base").write_text("")  # ours: ignored
    (quiet / "Old.proof").write_text("")
    old = time.time() - 3600
    os.utime(quiet / "Old.proof", (old, old))
    res = _serve_validate(tmp_path, QUIET_DIR=str(quiet), QUIET_OWN="EvictFix")
    assert "BOOT FAILED" in res.stdout and "quiet window" not in res.stdout
    # a measuring neighbour: serve_validate polls, then goes on when it is gone
    (tmp_path / "out").rename(tmp_path / "out1")
    (quiet / "TwoLaneFeat.proof").write_text("")
    p = subprocess.Popen(
        ["bash", "-c", f'sleep 3; rm -f "{quiet}/TwoLaneFeat.proof"'],
        start_new_session=True,
    )
    try:
        res = _serve_validate(
            tmp_path, QUIET_DIR=str(quiet), QUIET_OWN="EvictFix", QUIET_POLL="1"
        )
    finally:
        p.wait()
    assert "quiet window (TwoLaneFeat.proof): waiting" in res.stdout
    assert "BOOT FAILED" in res.stdout


@pytest.mark.skipif(not LINUX, reason="flock and GNU find")
def test_serve_validate_does_not_hold_the_boot_lock_while_it_waits(tmp_path: Path):
    quiet, lock = tmp_path / "quiet", tmp_path / "boot.lock"
    quiet.mkdir()
    (quiet / "TwoLaneFeat.proof").write_text("")
    cmd, env = _sv_setup(
        tmp_path,
        QUIET_DIR=str(quiet),
        QUIET_OWN="EvictFix",
        QUIET_POLL="1",
        BOOT_LOCK=str(lock),
    )
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        time.sleep(2.5)  # polling for the foreign window
        # the README's booter waits for the windows first, then takes the lock: a
        # booter holding it here would stall the measurer that waits for it
        assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
        (quiet / "TwoLaneFeat.proof").unlink()
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert "quiet window (TwoLaneFeat.proof): waiting" in out
    assert "waiting for boot lock" in out and "BOOT FAILED" in out


@pytest.mark.skipif(not LINUX, reason="flock and GNU find")
def test_serve_validate_lets_go_of_the_lock_when_a_window_opens_under_it(
    tmp_path: Path,
):
    fake = tmp_path / "fakebin"
    fake.mkdir()
    # the third look at the quiet directory (the one under the lock) finds a window
    (fake / "find").write_text(
        "#!/usr/bin/env bash\n"
        'n=$(( $(cat "$FIND_COUNT" 2> /dev/null || echo 0) + 1 ))\n'
        'echo "$n" > "$FIND_COUNT"\n'
        '[[ $n == 2 ]] && echo "$1/TwoLaneFeat.proof"\n'
        "exit 0\n"
    )
    (fake / "find").chmod(0o755)
    res = _serve_validate(
        tmp_path,
        PATH=f"{fake}:/usr/local/bin:/usr/bin:/bin",
        FIND_COUNT=str(tmp_path / "find.count"),
        QUIET_DIR=str(tmp_path / "quiet"),
        QUIET_OWN="EvictFix",
        QUIET_POLL="1",
        BOOT_LOCK=str(tmp_path / "boot.lock"),
    )
    out = res.stdout
    assert "quiet window under the boot lock (TwoLaneFeat.proof): lock released" in out
    assert out.count("waiting for boot lock") == 2 and "BOOT FAILED" in out


def test_serve_validate_without_quiet_dir_is_unchanged(tmp_path: Path):
    res = _serve_validate(tmp_path)
    assert "BOOT FAILED" in res.stdout and "quiet" not in res.stdout


def _quiet_functions() -> str:
    """quiet_end, quiet_lock_wait and quiet_begin, cut out of serve_validate.sh."""
    text = (PORT_V031 / "serve_validate.sh").read_text()
    out = []
    for name in ("quiet_end", "quiet_lock_wait", "quiet_begin"):
        start = text.index(f"\n{name}() {{") + 1
        out.append(text[start : text.index("\n}\n", start) + 3])
    return "".join(out)


def _run_quiet(tmp_path: Path, body: str, **env: str) -> list[str]:
    script = tmp_path / "quiet_fns.sh"
    script.write_text(
        'log() { echo "log: $*"; }\nQUIET_FILE=""\nQUIET_PID=""\n'
        + _quiet_functions()
        + body
    )
    full = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "QUIET_DIR": str(tmp_path / "quiet"),
        "QUIET_OWN": "EvictFix",
        "QUIET_HOLD": "1",
        "QUIET_POLL": "0.3",
        "QUIET_SETTLE": "0.2",
        "QUIET_TOUCH": "0.2",
        **env,
    }
    (tmp_path / "quiet").mkdir(exist_ok=True)
    res = run_group(["bash", str(script)], full, timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout.splitlines()


@pytest.mark.skipif(not LINUX, reason="flock and GNU stat")
def test_quiet_begin_waits_for_a_boot_keeps_the_file_fresh_and_end_removes_it(
    tmp_path: Path,
):
    q = tmp_path / "quiet" / "EvictFix"
    lines = _run_quiet(
        tmp_path,
        f"""
( flock 8; sleep 1.2 ) 8> "{tmp_path}/boot.lock" &
sleep 0.3
BOOT_LOCK="{tmp_path}/boot.lock" quiet_begin
echo "begun: file=$([[ -e {q} ]] && echo yes || echo no)"
m1=$(stat -c %Y.%y "{q}")
sleep 0.8
m2=$(stat -c %Y.%y "{q}")
echo "refreshed: $([[ $m1 != "$m2" ]] && echo yes || echo no)"
quiet_end
echo "ended: file=$([[ -e {q} ]] && echo yes || echo no) pid=${{QUIET_PID:-none}}"
sleep 0.6
echo "later: file=$([[ -e {q} ]] && echo yes || echo no)"
""",
        QUIET_CAP="30",
    )
    assert any("is held (a boot is in flight): waiting" in ln for ln in lines)
    assert "begun: file=yes" in lines and "refreshed: yes" in lines
    assert "ended: file=no pid=none" in lines and "later: file=no" in lines


@pytest.mark.skipif(not LINUX, reason="GNU stat")
def test_quiet_begin_stops_touching_at_the_cap_and_waits_for_a_build(tmp_path: Path):
    fake = tmp_path / "fakebin"
    fake.mkdir()
    # pgrep sees a docker buildx build twice, then none
    (fake / "pgrep").write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "$PGREP_ARGS"\n'
        'n=$(( $(cat "$PGREP_COUNT" 2> /dev/null || echo 0) + 1 ))\n'
        'echo "$n" > "$PGREP_COUNT"\n'
        "(( n <= 2 ))\n"
    )
    (fake / "pgrep").chmod(0o755)
    q = tmp_path / "quiet" / "EvictFix"
    lines = _run_quiet(
        tmp_path,
        f"""
quiet_begin
sleep 1.4
m1=$(stat -c %Y.%y "{q}")
sleep 0.8
m2=$(stat -c %Y.%y "{q}")
echo "frozen: $([[ $m1 == "$m2" ]] && echo yes || echo no)"
quiet_end
""",
        PATH=f"{fake}:/usr/local/bin:/usr/bin:/bin",
        PGREP_COUNT=str(tmp_path / "pgrep.count"),
        PGREP_ARGS=str(tmp_path / "pgrep.args"),
        QUIET_CAP="1",
    )
    assert sum("docker buildx build is running" in ln for ln in lines) == 2
    assert "frozen: yes" in lines
    # anchored, so that another agent's waiter naming the command does not match
    args = (tmp_path / "pgrep.args").read_text().splitlines()
    assert args and all(
        a == "-f ^(/usr/sbin/|/usr/bin/)?docker buildx build" for a in args
    )


def test_quiet_begin_is_a_noop_without_hold_or_dir(tmp_path: Path):
    lines = _run_quiet(
        tmp_path,
        'quiet_begin; echo "file=$(ls "$QUIET_DIR" | wc -l | tr -d " ")"; quiet_end',
        QUIET_HOLD="0",
    )
    assert lines == ["file=0"]


def _serve_validate_cells(
    tmp_path: Path, bench: str = ":", **env: str
) -> "tuple[subprocess.CompletedProcess[str], list[list[str]], Path]":
    """serve_validate.sh with a stub server, two cells and a hook.

    The stub bench client and the hook log when they run and which quiet files
    exist; ``bench`` is shell run by the client at every cell.
    """
    quiet, obs = tmp_path / "quiet", tmp_path / "obs.log"
    quiet.mkdir()
    (tmp_path / "venv/bin").mkdir(parents=True)
    (tmp_path / "venv/bin/python").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ " $* " == *" bench serve "* ]]; then\n'
        '    echo "cell $(date +%s.%N) quiet=$(ls "$QUIET_DIR" | tr "\\n" ,)"'
        ' >> "$STUB_OBS"\n'
        f"    {bench}\n"
        "fi\n"
        "exit 0\n"
    )
    (tmp_path / "venv/bin/python").chmod(0o755)
    (tmp_path / "hook.sh").write_text(
        "#!/usr/bin/env bash\n"
        'echo "hook $(date +%s.%N) quiet=$(ls "$QUIET_DIR" | tr "\\n" ,)"'
        ' >> "$STUB_OBS"\n'
        "echo PASS\n"
    )
    (tmp_path / "server.py").write_text(
        "import http.server, json, sys\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        '        body = json.dumps({"data": [{"id": "stub"}]}).encode()\n'
        "        self.send_response(200)\n"
        '        self.send_header("Content-Length", str(len(body)))\n'
        "        self.end_headers()\n"
        "        self.wfile.write(body)\n"
        "    def log_message(self, *a):\n"
        "        pass\n"
        'http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()\n'
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    cmd, full = _sv_setup(
        tmp_path,
        PORT=str(port),
        QUIET_DIR=str(quiet),
        QUIET_OWN="EvictFix",
        QUIET_HOLD="1",
        QUIET_GAP="2",
        QUIET_TOUCH="1",
        QUIET_POLL="1",
        QUIET_SETTLE="1",
        BOOT_LOCK=str(tmp_path / "boot.lock"),
        LOAD_HOOK=str(tmp_path / "hook.sh"),
        CELLS="1024:512:1 1024:512:8",
        CONC_PROBE="0",
        PPL_PROBE="0",
        PREFIX_PROBE="0",
        WARMUP="0",
        STUB_OBS=str(obs),
        REAL_PY=sys.executable,
        **env,
    )
    (tmp_path / "serve_stub.sh").write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do [[ $a == PORT=* ]] && PORT=${a#PORT=}; done\n'
        f'exec "$REAL_PY" "{tmp_path}/server.py" "$PORT"\n'
    )
    res = run_group(cmd, full, timeout=120)
    assert "READY" in res.stdout, res.stdout + res.stderr
    rows = [ln.split() for ln in obs.read_text().splitlines()]
    return res, rows, quiet


@pytest.mark.skipif(
    not (LINUX and shutil.which("curl") and shutil.which("flock")),
    reason="curl and flock",
)
def test_serve_validate_measures_the_cells_in_one_window_and_the_hook_in_none(
    tmp_path: Path,
):
    res, rows, quiet = _serve_validate_cells(tmp_path)
    assert [r[0] for r in rows] == ["cell", "cell", "hook"]
    assert [r[2] for r in rows] == ["quiet=EvictFix,", "quiet=EvictFix,", "quiet="]
    assert float(rows[2][1]) - float(rows[1][1]) >= 2  # QUIET_GAP before the hook
    assert res.stdout.count("quiet window open") == 1
    assert "quiet window closed" in res.stdout and list(quiet.iterdir()) == []
    assert "a boot is in flight" not in res.stdout


@pytest.mark.skipif(
    not (LINUX and shutil.which("curl") and shutil.which("flock")),
    reason="curl and flock",
)
def test_serve_validate_rechecks_the_boot_lock_before_every_cell(tmp_path: Path):
    # a booter takes boot.lock while the first cell runs (it had it before our
    # file appeared, or ignores it): the second cell must wait for it and settle
    hold = (
        'if [[ ! -e "$STUB_FLAG" ]]; then touch "$STUB_FLAG"; '
        '( flock 8; sleep 2 ) 8> "$BOOT_LOCK" & sleep 0.3; fi'
    )
    res, rows, quiet = _serve_validate_cells(
        tmp_path, hold, STUB_FLAG=str(tmp_path / "flag")
    )
    assert [r[0] for r in rows] == ["cell", "cell", "hook"]
    assert float(rows[1][1]) - float(rows[0][1]) >= 2.5  # the hold, then the settle
    waits = [ln for ln in res.stdout.splitlines() if "a boot is in flight" in ln]
    assert waits and "still held" not in res.stdout
    assert res.stdout.index("a boot is in flight") < res.stdout.index(
        "bench-1024x512-c8"
    )
    assert res.stdout.count("quiet window open") == 1
    assert list(quiet.iterdir()) == []
