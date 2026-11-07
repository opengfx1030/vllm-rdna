# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run guards of the pciescope probe on fake /proc, quiet, lock and sysfs trees.

The fake trees copy the real formats seen on par1-cs25 on 2026-11-07: hwmon
``device`` symlinks ``../../../0000:0a:00.0``, KFD node ``gpu_id`` files with a
``location_id <n>`` line in ``properties`` (``51968`` = bus ``cb``), per-process
``stats_<gpu_id>/evicted_ms`` and empty quiet files.
"""

import fcntl
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools" / "rdna"))

from pciescope import model as m  # noqa: E402
from pciescope.probe import guard  # noqa: E402
from pciescope.sysfs import Sysfs  # noqa: E402

NOW = 1_800_000_000.0  # fixed clock of the pure age tests
OLD = 1_000_000_000.0  # a real mtime far in the past (the keeper must beat it)


def wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def write(path: Path, text: str | bytes = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    return path


def make_gpu(
    bus: str, *, hwmon: str | None = None, kfd_gpu_id: int | None = None
) -> m.Gpu:
    return m.Gpu(
        id=m.gpu_entity(bus),
        bus=bus,
        bdf=f"0000:{bus}:00.0",
        name="Radeon Pro V620",
        arch="gfx1030",
        device_id=0,
        root="00",
        switch="B",
        slot="B",
        port=None,
        link=None,
        hwmon=hwmon,
        kfd_gpu_id=kfd_gpu_id,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCIESCOPE_QUIET_DIR", raising=False)


# --- buildx scan ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("docker buildx build --push .", True),
        ("/usr/bin/docker buildx build", True),
        ("/usr/sbin/docker buildx build -t x", True),
        ("docker buildx builder ls", True),  # no word boundary, like the pattern
        ("/opt/docker buildx build", False),
        ("sudo docker buildx build", False),
        ("bash -c docker buildx build .", False),
        ("python -m pciescope docker buildx build", False),
        ("pgrep -f docker buildx build", False),
        ("docker buildx ls", False),
        ("docker  buildx build", False),
        ("", False),
    ],
)
def test_buildx_pattern_is_anchored(cmd: str, expected: bool) -> None:
    assert guard.is_buildx_cmdline(cmd) is expected


def test_buildx_pattern_constant_documents_the_box_rule() -> None:
    assert guard.BUILDX_PATTERN == r"^(/usr/sbin/|/usr/bin/)?docker buildx build"


def test_buildx_running_on_fake_proc(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    write(proc / "100" / "cmdline", b"docker\0buildx\0build\0--push\0.\0")
    write(proc / "101" / "cmdline", b"/usr/bin/docker\0buildx\0build\0")
    write(proc / "102" / "cmdline", b"/usr/sbin/docker\0buildx\0build\0")
    # callers and parents that merely contain the text never match
    write(proc / "200" / "cmdline", b"bash\0-c\0docker buildx build .\0")
    write(proc / "201" / "cmdline", b"python3\0-m\0pciescope\0docker buildx build\0")
    write(proc / "202" / "cmdline", b"sudo\0docker\0buildx\0build\0")
    write(proc / "203" / "cmdline", b"/home/x/docker\0buildx\0build\0")
    write(proc / "204" / "cmdline", b"")  # kernel thread
    (proc / "205").mkdir()  # process without a readable cmdline
    write(proc / "self" / "cmdline", b"docker\0buildx\0build\0")  # not a pid
    write(proc / "version", "Linux")
    assert guard.buildx_running(proc) == [100, 101, 102]
    assert guard.buildx_running(proc, exclude_pid=101) == [100, 102]


def test_buildx_running_missing_proc(tmp_path: Path) -> None:
    assert guard.buildx_running(tmp_path / "absent") == []


# --- quiet files ---------------------------------------------------------------


def quiet_tree(tmp_path: Path) -> Path:
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    for name, age in (
        ("Alpha", 10.0),
        ("Bravo", 359.0),
        ("Stale", 361.0),
        ("Mine", 1.0),
        ("Future", -50.0),
    ):
        path = write(quiet / name)
        os.utime(path, (NOW - age, NOW - age))
    (quiet / "SubDir").mkdir()
    return quiet


def test_fresh_quiet_files_other_agents_only(tmp_path: Path) -> None:
    quiet = quiet_tree(tmp_path)
    found = guard.fresh_quiet_files("Mine", quiet_dir=quiet, now=NOW)
    assert [(q.name, round(q.age_s)) for q in found] == [
        ("Alpha", 10),
        ("Bravo", 359),
        ("Future", 0),  # a future mtime is fresh, age clamped
    ]
    names = [q.name for q in guard.fresh_quiet_files(None, quiet_dir=quiet, now=NOW)]
    assert names == ["Alpha", "Bravo", "Future", "Mine"]


def test_fresh_quiet_files_max_age(tmp_path: Path) -> None:
    quiet = quiet_tree(tmp_path)
    found = guard.fresh_quiet_files("Mine", max_age_s=60.0, quiet_dir=quiet, now=NOW)
    assert [q.name for q in found] == ["Alpha", "Future"]
    found = guard.fresh_quiet_files("Mine", max_age_s=400.0, quiet_dir=quiet, now=NOW)
    assert "Stale" in [q.name for q in found]


def test_fresh_quiet_files_env_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quiet = quiet_tree(tmp_path)
    monkeypatch.setenv("PCIESCOPE_QUIET_DIR", str(quiet))
    assert guard.default_quiet_dir() == quiet
    assert [q.name for q in guard.fresh_quiet_files("Mine", now=NOW)][0] == "Alpha"


def test_fresh_quiet_files_default_dir_is_the_box_path() -> None:
    assert guard.default_quiet_dir().parts[-3:] == ("w4a8_runs", "lanes", "quiet")
    assert guard.DEFAULT_BOOT_LOCK.parts[-2:] == ("w4a8_runs", "boot.lock")


def test_fresh_quiet_files_missing_dir(tmp_path: Path) -> None:
    assert guard.fresh_quiet_files("Me", quiet_dir=tmp_path / "absent") == []


# --- boot lock -----------------------------------------------------------------


def test_boot_lock_missing_is_not_held(tmp_path: Path) -> None:
    assert guard.boot_lock_held(tmp_path / "absent") is False


def test_boot_lock_held_by_another_open_file(tmp_path: Path) -> None:
    lock = write(tmp_path / "boot.lock")
    assert guard.boot_lock_held(lock) is False
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert guard.boot_lock_held(lock) is True
        # the probe released its own attempt: nothing is left locked afterwards
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert guard.boot_lock_held(lock) is False
        assert guard.boot_lock_held(lock) is False
    finally:
        os.close(fd)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
def test_boot_lock_unreadable_counts_as_held(tmp_path: Path) -> None:
    lock = write(tmp_path / "boot.lock")
    lock.chmod(0)
    assert guard.boot_lock_held(lock) is True


# --- noisy_reasons -------------------------------------------------------------


def test_noisy_reasons_quiet_box(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    write(proc / "1" / "cmdline", b"/sbin/init\0")
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    assert (
        guard.noisy_reasons(
            "Me",
            proc_root=proc,
            quiet_dir=quiet,
            boot_lock=tmp_path / "absent.lock",
            now=NOW,
        )
        == []
    )


def test_noisy_reasons_lists_every_cause(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    write(proc / "100" / "cmdline", b"docker\0buildx\0build\0")
    quiet = quiet_tree(tmp_path)
    lock = write(tmp_path / "boot.lock")
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        reasons = guard.noisy_reasons(
            "Mine", proc_root=proc, quiet_dir=quiet, boot_lock=lock, now=NOW
        )
    finally:
        os.close(fd)
    assert len(reasons) == 5
    assert "buildx" in reasons[0]
    assert "100" in reasons[0]
    assert "Alpha" in reasons[1]
    assert all("Mine" not in r for r in reasons)
    assert "boot.lock" in reasons[-1]


def test_noisy_reasons_ignores_the_callers_own_process(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    write(proc / str(os.getpid()) / "cmdline", b"docker\0buildx\0build\0")
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    assert (
        guard.noisy_reasons(
            None,
            proc_root=proc,
            quiet_dir=quiet,
            boot_lock=tmp_path / "absent.lock",
        )
        == []
    )


# --- QuietWindow ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["", " ", "a/b", "../x", "..", ".", ".hid", "a\\b", " a"]
)
def test_quiet_window_refuses_bad_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid quiet file name"):
        guard.QuietWindow(name, quiet_dir=tmp_path)


def test_quiet_window_refuses_non_positive_times(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        guard.QuietWindow("Me", quiet_dir=tmp_path, touch_every_s=0)
    with pytest.raises(ValueError, match="positive"):
        guard.QuietWindow("Me", quiet_dir=tmp_path, cap_s=-1)


def test_quiet_window_creates_touches_and_removes(tmp_path: Path) -> None:
    mine = tmp_path / "Me"
    other = write(tmp_path / "Other")
    os.utime(other, (OLD, OLD))
    with guard.QuietWindow("Me", quiet_dir=tmp_path, touch_every_s=0.02) as window:
        assert mine.is_file()
        assert not window.expired
        os.utime(mine, (OLD, OLD))  # age it; the keeper must refresh it
        assert wait_for(lambda: mine.stat().st_mtime > OLD + 1000)
        assert window.age_s > 0
        assert window._thread is not None
        assert window._thread.daemon
    assert not mine.exists()
    assert other.stat().st_mtime == OLD  # never touched
    assert not window.expired
    assert window.error is None


def test_quiet_window_keeper_freshens_for_other_agents(tmp_path: Path) -> None:
    with guard.QuietWindow("Me", quiet_dir=tmp_path, touch_every_s=0.02):
        os.utime(tmp_path / "Me", (OLD, OLD))
        assert wait_for(
            lambda: guard.fresh_quiet_files("You", quiet_dir=tmp_path) != []
        )


def test_quiet_window_stops_touching_after_its_cap(tmp_path: Path) -> None:
    mine = tmp_path / "Me"
    other = write(tmp_path / "Other")
    os.utime(other, (OLD, OLD))
    with guard.QuietWindow(
        "Me", quiet_dir=tmp_path, touch_every_s=0.02, cap_s=0.2
    ) as window:
        assert wait_for(lambda: window.expired)
        assert not mine.exists()  # removed at the cap
        thread = window._thread
        assert thread is not None
        assert wait_for(lambda: not thread.is_alive())  # and no longer touching
        time.sleep(0.1)
        assert not mine.exists()
    assert window.expired
    assert other.stat().st_mtime == OLD


def test_quiet_window_exit_is_prompt_with_a_long_interval(tmp_path: Path) -> None:
    start = time.monotonic()
    with guard.QuietWindow("Me", quiet_dir=tmp_path, touch_every_s=600) as window:
        assert (tmp_path / "Me").is_file()
    assert time.monotonic() - start < 2.0
    assert window._thread is not None
    assert not window._thread.is_alive()
    assert not (tmp_path / "Me").exists()


def test_quiet_window_leaves_a_file_someone_else_re_touched(tmp_path: Path) -> None:
    mine = tmp_path / "Me"
    with guard.QuietWindow("Me", quiet_dir=tmp_path, touch_every_s=600):
        # another process of the same name re-touched the file after our last touch
        st = mine.stat()
        other_ns = st.st_mtime_ns + 5_000_000_000
        os.utime(mine, ns=(other_ns, other_ns))
    assert mine.exists()


def test_quiet_window_does_not_remove_an_unrelated_replacement(tmp_path: Path) -> None:
    mine = tmp_path / "Me"
    with guard.QuietWindow("Me", quiet_dir=tmp_path, touch_every_s=600):
        mine.unlink()
        write(mine, "someone else's marker")
        os.utime(mine, (OLD, OLD))
    assert mine.read_text() == "someone else's marker"


def test_quiet_window_needs_an_existing_directory(tmp_path: Path) -> None:
    window = guard.QuietWindow("Me", quiet_dir=tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        window.__enter__()
    assert window._thread is None


def test_quiet_window_default_dir_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PCIESCOPE_QUIET_DIR", str(tmp_path))
    with guard.QuietWindow("Me", touch_every_s=600):
        assert (tmp_path / "Me").is_file()
    assert not (tmp_path / "Me").exists()


# --- temperature rule ----------------------------------------------------------


def test_evaluate_temps_rules() -> None:
    assert guard.evaluate_temps([]) is None
    assert guard.evaluate_temps([80.0, 94.9]) is None  # single sample below 95
    reason = guard.evaluate_temps([60.0, 95.0])
    assert reason is not None
    assert "95.0" in reason
    assert guard.evaluate_temps([96.0, 50.0]) is not None  # any sample in history
    # the 5-sample mean needs five samples
    assert guard.evaluate_temps([94.0, 94.0, 94.0, 94.0]) is None
    mean_reason = guard.evaluate_temps([94.0] * 5)
    assert mean_reason is not None
    assert "mean" in mean_reason
    assert guard.evaluate_temps([89.9] * 5) is None
    assert guard.evaluate_temps([90.0] * 5) is not None  # >= is a breach
    # only the last five count
    assert guard.evaluate_temps([79.0] + [85.0] * 5) is None
    assert guard.evaluate_temps([50.0] + [91.0] * 5) is not None
    # custom limits
    assert guard.evaluate_temps([70.0] * 3, mean_n=3, mean_limit_c=70.0) is not None
    assert guard.evaluate_temps([70.0], sample_limit_c=70.0) is not None


# --- TempWatcher ---------------------------------------------------------------


class FakeTemps:
    """Per-bus scripted temperatures; None entries mean an unreadable sensor."""

    def __init__(self, script: dict[str, list[float | None]]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: dict[str, int] = {}

    def __call__(self, gpu: m.Gpu) -> float | None:
        n = self.calls.get(gpu.bus, 0)
        self.calls[gpu.bus] = n + 1
        values = self.script.get(gpu.bus)
        if not values:
            return None
        return values[min(n, len(values) - 1)]


def test_temp_watcher_default_read_uses_the_hwmon_of_the_gpu(tmp_path: Path) -> None:
    hw = tmp_path / "class/hwmon"
    write(hw / "hwmon3" / "temp2_input", "60000\n")  # real format: milli-degrees
    write(hw / "hwmon9" / "temp2_input", "49000\n")
    write(hw / "hwmon12" / "temp2_input", "n/a\n")
    gpus = [
        make_gpu("cb", hwmon="class/hwmon/hwmon3"),
        make_gpu("0a", hwmon="class/hwmon/hwmon9"),
        make_gpu("17", hwmon="class/hwmon/hwmon12"),  # unreadable content
        make_gpu("14", hwmon="class/hwmon/absent"),  # no such attribute
        make_gpu("ce"),  # no hwmon at all
    ]
    watcher = guard.TempWatcher(gpus, sysfs=Sysfs(tmp_path))
    watcher.check_once()
    assert watcher.last_c == {"cb": 60.0, "0a": 49.0}
    assert watcher.max_c == {"cb": 60.0, "0a": 49.0}
    assert watcher.n_samples == 2
    assert watcher.missing == ["17", "14", "ce"]  # never treated as cool
    assert watcher.breach is None


def test_temp_watcher_records_max_and_last() -> None:
    fake = FakeTemps({"0a": [50.0, 70.0, 60.0], "0f": [40.0]})
    watcher = guard.TempWatcher([make_gpu("0a"), make_gpu("0f")], read=fake)
    for _ in range(3):
        watcher.check_once()
    assert watcher.max_c == {"0a": 70.0, "0f": 40.0}
    assert watcher.last_c == {"0a": 60.0, "0f": 40.0}
    assert watcher.n_samples == 6
    assert watcher.breach is None
    assert watcher.missing == []


def test_temp_watcher_single_sample_breach_calls_back_once() -> None:
    fake = FakeTemps({"0a": [80.0, 95.0, 96.0]})
    seen: list[str] = []
    watcher = guard.TempWatcher([make_gpu("0a")], read=fake, on_breach=seen.append)
    watcher.check_once()
    assert watcher.breach is None
    assert seen == []
    watcher.check_once()
    assert watcher.breach is not None
    assert "gpu:0a" in watcher.breach
    assert "95.0" in watcher.breach
    watcher.check_once()
    assert len(seen) == 1  # only the first breach is reported
    assert seen[0] == watcher.breach
    assert watcher.max_c["0a"] == 96.0


def test_temp_watcher_five_sample_mean_breach() -> None:
    fake = FakeTemps({"0a": [91.0, 89.0, 92.0, 90.0, 93.0]})  # mean 91.0
    watcher = guard.TempWatcher([make_gpu("0a")], read=fake)
    for _ in range(4):
        watcher.check_once()
        assert watcher.breach is None  # needs five samples
    watcher.check_once()
    assert watcher.breach is not None
    assert "mean" in watcher.breach


def test_temp_watcher_mean_window_slides() -> None:
    cool: list[float | None] = [89.0] * 5  # below both limits
    watcher = guard.TempWatcher([make_gpu("0a")], read=FakeTemps({"0a": cool}))
    for _ in range(12):
        watcher.check_once()
    assert watcher.breach is None


def test_temp_watcher_unreadable_sensor_is_reported_and_stays_reported() -> None:
    fake = FakeTemps({"0a": [50.0, None, 55.0]})
    watcher = guard.TempWatcher([make_gpu("0a"), make_gpu("ce")], read=fake)
    watcher.check_once()
    assert watcher.missing == ["ce"]  # no reading at all
    assert watcher.n_samples == 1
    watcher.check_once()  # the sensor stops answering: reported, not cool
    watcher.check_once()
    assert watcher.missing == ["ce", "0a"]  # a blind spot stays visible
    assert watcher.n_samples == 2
    assert watcher.last_c == {"0a": 55.0}


def test_temp_watcher_keys_are_buses_and_duplicates_collapse() -> None:
    fake = FakeTemps({"0a": [50.0]})
    watcher = guard.TempWatcher([make_gpu("0a"), make_gpu("0a")], read=fake)
    watcher.check_once()
    assert watcher.n_samples == 1
    assert list(watcher.max_c) == ["0a"]


def test_temp_watcher_thread_samples_and_stops_promptly() -> None:
    fake = FakeTemps({"0a": [50.0]})
    watcher = guard.TempWatcher([make_gpu("0a")], read=fake, period_s=0.01)
    watcher.start()
    watcher.start()  # idempotent
    thread = watcher._thread
    assert thread is not None
    assert thread.daemon
    assert wait_for(lambda: watcher.n_samples >= 3)
    watcher.stop()
    assert not thread.is_alive()
    done = watcher.n_samples
    time.sleep(0.05)
    assert watcher.n_samples == done


def test_temp_watcher_stop_is_prompt_with_a_long_period() -> None:
    watcher = guard.TempWatcher(
        [make_gpu("0a")], read=FakeTemps({"0a": [50.0]}), period_s=600.0
    )
    watcher.start()
    assert wait_for(lambda: watcher.n_samples >= 1)
    start = time.monotonic()
    watcher.stop()
    assert time.monotonic() - start < 2.0
    assert watcher._thread is not None
    assert not watcher._thread.is_alive()


def test_temp_watcher_thread_triggers_breach_callback() -> None:
    fired = threading.Event()
    watcher = guard.TempWatcher(
        [make_gpu("0a")],
        read=FakeTemps({"0a": [97.0]}),
        period_s=0.01,
        on_breach=lambda why: fired.set(),
    )
    watcher.start()
    try:
        assert fired.wait(5.0)
    finally:
        watcher.stop()
    assert watcher.breach is not None


def test_temp_watcher_stop_without_start() -> None:
    guard.TempWatcher([make_gpu("0a")]).stop()


# --- VRAM ----------------------------------------------------------------------


def test_vram_info(tmp_path: Path) -> None:
    dev = tmp_path / "bus/pci/devices/0000:0a:00.0"
    write(dev / "mem_info_vram_used", "28418838528\n")
    write(dev / "mem_info_vram_total", "32195477504\n")
    sysfs = Sysfs(tmp_path)
    assert guard.vram_info("0a", sysfs) == (3776638976, 32195477504)
    assert guard.vram_info("gpu:0A", sysfs) == (3776638976, 32195477504)
    assert guard.vram_info("0f", sysfs) is None  # no such device
    assert guard.vram_info("nonsense", sysfs) is None


def test_vram_info_unreadable_and_odd_values(tmp_path: Path) -> None:
    dev = tmp_path / "bus/pci/devices/0000:0a:00.0"
    write(dev / "mem_info_vram_used", "5\n")
    assert guard.vram_info("0a", Sysfs(tmp_path)) is None  # total missing
    write(dev / "mem_info_vram_total", "garbage\n")
    assert guard.vram_info("0a", Sysfs(tmp_path)) is None
    write(dev / "mem_info_vram_total", "4\n")  # used > total: no negative free
    assert guard.vram_info("0a", Sysfs(tmp_path)) == (0, 4)


# --- KFD evictions -------------------------------------------------------------

# KFD gpu_id of the GPUs as read on the box (cb 9708, 0a 18860, 14 57637, 07 28857)
CB = make_gpu("cb", kfd_gpu_id=9708)
G14 = make_gpu("14", kfd_gpu_id=57637)
NO_KFD = make_gpu("ce")  # no KFD gpu_id known


def add_proc(root: Path, pid: int, evicted: dict[int, int]) -> None:
    proc = root / "class/kfd/kfd/proc" / str(pid)
    for gpu_id, ms in evicted.items():
        write(proc / f"stats_{gpu_id}" / "evicted_ms", f"{ms}\n")
        write(proc / f"counters_{gpu_id}" / "x", "1\n")
    write(proc / "pasid", "32771\n")


def fake_proc(root: Path, pid: int, comm: str, cmdline: bytes) -> None:
    write(root / str(pid) / "comm", comm + "\n")
    write(root / str(pid) / "cmdline", cmdline)


def test_unknown_buses(tmp_path: Path) -> None:
    watcher = guard.EvictionWatcher([CB, NO_KFD], sysfs=Sysfs(tmp_path))
    assert watcher.unknown_buses() == ["ce"]
    assert guard.EvictionWatcher([CB, G14]).unknown_buses() == []


def test_snapshot_only_watched_gpus(tmp_path: Path) -> None:
    add_proc(tmp_path, 627550, {9708: 0, 57637: 120, 28857: 9})
    add_proc(tmp_path, 680562, {28857: 7})  # only an unwatched GPU
    add_proc(tmp_path, 681904, {9708: 3, 18860: 4})
    (tmp_path / "class/kfd/kfd/proc/not_a_pid").mkdir()
    procfs = tmp_path / "procfs"
    fake_proc(procfs, 627550, "python3", b"python3\0-m\0vllm\0serve\0")
    fake_proc(procfs, 681904, "bench", b"bench\0--x\0")
    watcher = guard.EvictionWatcher(
        [CB, G14], sysfs=Sysfs(tmp_path), procfs=Sysfs(procfs)
    )
    snap = watcher.snapshot()
    assert sorted(snap) == [627550, 681904]
    first = snap[627550]
    assert first.pid == 627550
    assert first.comm == "python3"
    assert first.cmdline == "python3 -m vllm serve"
    assert first.evicted_ms == {"cb": 0, "14": 120}
    assert snap[681904].evicted_ms == {"cb": 3}


def test_snapshot_gpu_without_kfd_id_contributes_nothing(tmp_path: Path) -> None:
    add_proc(tmp_path, 10, {9708: 1})
    watcher = guard.EvictionWatcher(
        [CB, NO_KFD], sysfs=Sysfs(tmp_path), procfs=Sysfs(tmp_path / "procfs")
    )
    assert watcher.snapshot()[10].evicted_ms == {"cb": 1}
    only = guard.EvictionWatcher([NO_KFD], sysfs=Sysfs(tmp_path))
    assert only.snapshot() == {}


def test_snapshot_exclude_pids_and_gone_process(tmp_path: Path) -> None:
    add_proc(tmp_path, 10, {9708: 1})
    add_proc(tmp_path, 11, {9708: 2})
    procfs = tmp_path / "procfs"
    procfs.mkdir()  # /proc entries gone: comm and cmdline empty, counters kept
    watcher = guard.EvictionWatcher(
        [CB], sysfs=Sysfs(tmp_path), procfs=Sysfs(procfs), exclude_pids=[10]
    )
    snap = watcher.snapshot()
    assert sorted(snap) == [11]
    assert (snap[11].comm, snap[11].cmdline) == ("", "")


def test_snapshot_ignores_unparsable_counters(tmp_path: Path) -> None:
    write(tmp_path / "class/kfd/kfd/proc/5/stats_9708/evicted_ms", "oops\n")
    assert guard.EvictionWatcher([CB], sysfs=Sysfs(tmp_path)).snapshot() == {}


def test_snapshot_without_kfd_is_empty(tmp_path: Path) -> None:
    assert guard.EvictionWatcher([CB], sysfs=Sysfs(tmp_path)).snapshot() == {}


def test_eviction_diff(tmp_path: Path) -> None:
    add_proc(tmp_path, 1, {9708: 100, 57637: 5})
    add_proc(tmp_path, 2, {9708: 0})
    add_proc(tmp_path, 3, {9708: 50})
    procfs = Sysfs(tmp_path / "procfs")
    watcher = guard.EvictionWatcher([CB, G14], sysfs=Sysfs(tmp_path), procfs=procfs)
    before = watcher.snapshot()
    write(tmp_path / "class/kfd/kfd/proc/1/stats_9708/evicted_ms", "7100\n")
    write(tmp_path / "class/kfd/kfd/proc/3/stats_9708/evicted_ms", "10\n")  # reset
    (tmp_path / "class/kfd/kfd/proc/2/stats_9708/evicted_ms").unlink()
    add_proc(tmp_path, 4, {9708: 999})  # appeared during the window
    after = watcher.snapshot()
    assert guard.EvictionWatcher.diff(before, after) == {
        1: {"cb": 7000, "14": 0},  # zero entries are kept
        3: {"cb": -40},  # a negative value flags a counter reset
    }
    assert guard.EvictionWatcher.diff(before, before) == {
        1: {"cb": 0, "14": 0},
        2: {"cb": 0},
        3: {"cb": 0},
    }
    assert guard.EvictionWatcher.diff({}, after) == {}
