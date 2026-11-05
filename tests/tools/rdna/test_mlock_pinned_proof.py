# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned-host mlock GPU proof driver and its serve_validate.sh hooks, run
against stub launchers: the dry run, the gating, a full run of four arms with
the verdict, the retries, and the quiet-window wait. No GPU, no server."""

import json
import os
import shutil
import signal
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
        "hip=${HIP_VISIBLE_DEVICES:-} neighbour=${NEIGHBOUR:-}"
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
echo 1999999 > "$out/serve.pid"
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
    echo "[0] bench 1024x512-c1 (n=4)"
    echo "    evicted_ms Worker_TP0=+0 -> ok"
    if [[ $hook == PASS ]]; then
        echo "[0] LOAD HOOK rc=0 PASS"
    else
        echo "[0] LOAD HOOK rc=1 FAIL"
    fi
} > "$out/summary.txt"
[[ $hook == PASS ]]
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
            "MEMLOCK_MIN_KB": "0",
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
    assert "neighbour: replay x 3; idle 60s gap 90s settle 45s" in out
    assert not proof.stub_log()  # nothing booted, no lease taken


def test_dry_run_prints_the_ple_override_of_the_arms(proof: Proof):
    res = proof(DRY="1", ARMS="fix", PLE_MODE="int4")
    assert res.returncode == 0 and "overrides=PLE=int4" in res.stdout


@pytest.mark.parametrize(
    "env, needle",
    [
        ({"EXPECT_SHA": ""}, "EXPECT_SHA is required"),
        ({"EXPECT_SHA": "deadbeef"}, "is not deadbeef"),
        ({"GPUS": "0,1,2,3"}, "GPU 0 is not in 2-9"),
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
    assert "cells=1024:512:1 1024:512:8" in first and "hip=2,3,4,5" in first
    # the lease: holder files and the request are gone, the lane marker is set
    assert list((base / "lease").iterdir()) == []
    assert (base / "lanes/hip25.free").exists()
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
def test_it_waits_for_the_other_holders_of_its_cards(proof: Proof):
    lease = proof.base / "lease"
    (lease / "TwoLaneFeat.hip4").write_text("TwoLaneFeat\n")
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
        assert (lease / "hip2-5.request").read_text().startswith("EvictFix")
        (proof.base / "run/STOP").write_text("")
        p.communicate(timeout=90)  # the 60 s poll sees STOP
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert p.returncode == 3 and time.time() - started < 100
    assert not (lease / "hip2-5.request").exists()  # its own request is withdrawn
    assert (lease / "TwoLaneFeat.hip4").exists()  # the other holder is untouched


# ---- serve_validate.sh: the opt-in quiet-window wait ------------------------------


def _serve_validate(tmp_path: Path, **env: str) -> "subprocess.CompletedProcess[str]":
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
    return run_group(
        ["bash", str(tree / "tools/rdna/port_v031/serve_validate.sh")],
        full,
        timeout=120,
    )


@pytest.mark.skipif(not LINUX, reason="GNU find -mmin")
def test_serve_validate_waits_for_a_foreign_quiet_file_and_not_for_its_own(
    tmp_path: Path,
):
    quiet = tmp_path / "quiet"
    quiet.mkdir()
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


def test_serve_validate_without_quiet_dir_is_unchanged(tmp_path: Path):
    res = _serve_validate(tmp_path)
    assert "BOOT FAILED" in res.stdout and "quiet" not in res.stdout
