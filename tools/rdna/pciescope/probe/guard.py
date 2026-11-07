# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run guards of the active probe: quiet box, temperature, VRAM, evictions.

Real formats on par1-cs25 (observed read-only on 2026-11-07):

* ``~/w4a8_runs/lanes/quiet/<Agent>``: empty files, the mtime is the signal.
* ``~/w4a8_runs/boot.lock``: empty file, taken with ``flock``.
* ``/sys/class/hwmon/hwmon*/``: ``name`` = ``amdgpu`` and ``device`` is a
  symlink to ``../../../0000:0a:00.0``; ``temp2_label`` = ``junction`` and
  ``temp2_input`` holds milli-degrees (``49000``).
* ``/sys/bus/pci/devices/0000:0a:00.0/mem_info_vram_{used,total}``: bytes in
  decimal (``28418838528`` / ``32195477504``).
* ``/sys/class/kfd/kfd/topology/nodes/<n>/gpu_id`` holds the KFD gpu id
  (``9708``; node 0 is the CPU with ``0``); ``properties`` has one
  ``<name> <value>`` per line, ``location_id 51968`` = ``0xcb00`` so the bus
  is ``(location_id >> 8) & 0xff`` = ``cb``.
* ``/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms``: one decimal
  integer.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from pciescope.collectors.amdgpu import junction_temp_c
from pciescope.collectors.kfd import evicted_ms, process_name
from pciescope.model import Gpu
from pciescope.sysfs import Sysfs

from .ids import normalize_bus

BUILDX_PATTERN = r"^(/usr/sbin/|/usr/bin/)?docker buildx build"
_BUILDX_PREFIXES = ("/usr/sbin/", "/usr/bin/", "")  # the optional group
_BUILDX_WORDS = "docker buildx build"

DEFAULT_QUIET_DIR: Path = Path("~/w4a8_runs/lanes/quiet").expanduser()
DEFAULT_BOOT_LOCK: Path = Path("~/w4a8_runs/boot.lock").expanduser()
ENV_QUIET_DIR = "PCIESCOPE_QUIET_DIR"
_JOIN_TIMEOUT_S = 5.0


def _decode_cmdline(raw: bytes) -> str:
    """Turn the NUL-separated ``/proc/<pid>/cmdline`` bytes into one string."""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").rstrip()


def _cmdline(proc_root: Path, pid: str) -> str | None:
    """Return ``/proc/<pid>/cmdline`` as text, None when unreadable."""
    try:
        return _decode_cmdline((proc_root / pid / "cmdline").read_bytes())
    except OSError:
        return None


def is_buildx_cmdline(cmd: str) -> bool:
    """Return True when ``cmd`` matches ``BUILDX_PATTERN`` (anchored at the start).

    Written as plain string code (no regex module on the box venv); no word
    boundary after ``build``, exactly like the pattern.
    """
    return any(cmd.startswith(prefix + _BUILDX_WORDS) for prefix in _BUILDX_PREFIXES)


def buildx_running(
    proc_root: Path = Path("/proc"), exclude_pid: int | None = None
) -> list[int]:
    """Return the pids whose command line starts with ``docker buildx build``.

    Same anchored pattern as the box rules (no ``pgrep``): a shell, an editor or
    this process that merely mentions the text does not match.
    """
    found: list[int] = []
    try:
        names = [p.name for p in proc_root.iterdir()]
    except OSError:
        return found
    for name in names:
        if not name.isdigit() or int(name) == exclude_pid:
            continue
        cmd = _cmdline(proc_root, name)
        if cmd is not None and is_buildx_cmdline(cmd):
            found.append(int(name))
    return sorted(found)


@dataclass(frozen=True)
class QuietFile:
    """A fresh quiet file of another agent."""

    name: str
    age_s: float


def default_quiet_dir() -> Path:
    """Return the quiet directory (env ``PCIESCOPE_QUIET_DIR`` overrides)."""
    env = os.environ.get(ENV_QUIET_DIR)
    return Path(env).expanduser() if env else DEFAULT_QUIET_DIR


def fresh_quiet_files(
    me: str | None = None,
    *,
    max_age_s: float = 360.0,
    quiet_dir: Path | None = None,
    now: float | None = None,
) -> list[QuietFile]:
    """Return the quiet files of other agents younger than ``max_age_s``.

    Files of ``me`` are skipped; an unreadable directory gives ``[]`` (nothing
    can be seen, the box rule only asks to respect files that exist).
    """
    directory = quiet_dir if quiet_dir is not None else default_quiet_dir()
    stamp = time.time() if now is None else now
    out: list[QuietFile] = []
    try:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except OSError:
        return out
    for path in entries:
        if path.name == me:
            continue
        try:
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
        except OSError:
            continue
        age = max(stamp - mtime, 0.0)  # a future mtime counts as fresh
        if age < max_age_s:
            out.append(QuietFile(path.name, age))
    return out


def boot_lock_held(path: Path | None = None) -> bool:
    """Return True when another open file description holds the boot lock.

    Takes ``flock(LOCK_EX | LOCK_NB)`` and releases it at once. A missing file
    is not held. A file that cannot be opened cannot be proven free: True.
    """
    target = path if path is not None else DEFAULT_BOOT_LOCK
    try:
        fd = os.open(target, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:  # includes BlockingIOError: held, or cannot be proven free
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def noisy_reasons(
    me: str | None,
    *,
    proc_root: Path = Path("/proc"),
    quiet_dir: Path | None = None,
    boot_lock: Path | None = None,
    now: float | None = None,
) -> list[str]:
    """Return why a measured window must not start now (empty = quiet)."""
    reasons: list[str] = []
    pids = buildx_running(proc_root, exclude_pid=os.getpid())
    if pids:
        reasons.append(
            "docker buildx build running (pid " + ", ".join(map(str, pids)) + ")"
        )
    reasons.extend(
        f"quiet file of {q.name} is {q.age_s:.0f} s old (window of another agent)"
        for q in fresh_quiet_files(me, quiet_dir=quiet_dir, now=now)
    )
    if boot_lock_held(boot_lock):
        reasons.append("boot.lock is held (a boot or weight load is in progress)")
    return reasons


class QuietWindow:
    """Context manager that keeps ``quiet/<me>`` fresh for one measured window.

    The file is created on entry, re-touched every ``touch_every_s`` by a daemon
    thread and removed on exit. After ``cap_s`` the thread stops touching,
    removes the file and sets ``expired``. Only the own file is ever touched or
    removed, and it is removed only while it still carries the modification
    time this window wrote last (a concurrent re-touch of the same name by
    another process is left alone).
    """

    def __init__(
        self,
        me: str,
        *,
        quiet_dir: Path | None = None,
        touch_every_s: float = 60.0,
        cap_s: float = 900.0,
    ) -> None:
        if (
            not me.strip()
            or me != me.strip()
            or any(c in me for c in "/\\\0")
            or me in {".", ".."}
            or me.startswith(".")
        ):
            raise ValueError(f"invalid quiet file name: {me!r}")
        if touch_every_s <= 0 or cap_s <= 0:
            raise ValueError("touch_every_s and cap_s must be positive")
        directory = quiet_dir if quiet_dir is not None else default_quiet_dir()
        self.me = me
        self.path = directory / me
        self.touch_every_s = touch_every_s
        self.cap_s = cap_s
        self.expired = False
        self.error: OSError | None = None
        self._start = 0.0
        self._entered = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stamp: tuple[int, int] | None = None  # (st_ino, st_mtime_ns) we wrote

    @property
    def age_s(self) -> float:
        """Seconds since the window was entered (0 before)."""
        return time.monotonic() - self._start if self._entered else 0.0

    def _touch(self) -> None:
        with self._lock:
            if self._stop.is_set() and self._thread is not None:
                return
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                os.utime(fd)
                st = os.fstat(fd)
            finally:
                os.close(fd)
            self._stamp = (st.st_ino, st.st_mtime_ns)

    def _remove(self) -> None:
        with self._lock:
            if self._stamp is None:
                return
            try:
                st = self.path.stat()
                if (st.st_ino, st.st_mtime_ns) == self._stamp:
                    self.path.unlink()
            except FileNotFoundError:
                pass
            self._stamp = None

    def _run(self) -> None:
        while True:
            remaining = self.cap_s - self.age_s
            if self._stop.wait(max(min(self.touch_every_s, remaining), 0.0)):
                return
            if self.age_s >= self.cap_s:
                self._remove()
                self.expired = True
                return
            try:
                self._touch()
            except OSError as exc:
                self.error = exc  # the file ages out: the caller must see it

    def __enter__(self) -> QuietWindow:
        self._touch()
        self._start = time.monotonic()
        self._entered = True
        self._thread = threading.Thread(
            target=self._run, name=f"quiet-window-{self.me}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_JOIN_TIMEOUT_S)
        self._remove()


def evaluate_temps(
    history: Sequence[float],
    *,
    mean_n: int = 5,
    mean_limit_c: float = 90.0,
    sample_limit_c: float = 95.0,
) -> str | None:
    """Apply the box abort rule to ``history`` (Celsius, oldest first).

    Returns the reason when any sample is >= ``sample_limit_c`` or the mean of
    the last ``mean_n`` samples (needs that many) is >= ``mean_limit_c``.
    """
    if not history:
        return None
    peak = max(history)
    if peak >= sample_limit_c:
        return f"junction sample {peak:.1f} C >= {sample_limit_c:g} C"
    if len(history) >= mean_n:
        tail = history[-mean_n:]
        mean = sum(tail) / len(tail)
        if mean >= mean_limit_c:
            return (
                f"junction mean of last {mean_n} samples {mean:.1f} C "
                f">= {mean_limit_c:g} C"
            )
    return None


class TempWatcher:
    """Daemon thread sampling the junction temperature of the probed GPUs.

    ``max_c``/``last_c`` per bus (``gpu.bus``), ``n_samples`` counts successful
    readings, ``missing`` lists the buses that ever returned None (no hwmon or
    unreadable sensor; they are NOT silently treated as cool: the caller
    decides). ``breach`` holds the first violated-rule text; ``on_breach`` is
    called once with it.
    """

    def __init__(
        self,
        gpus: Sequence[Gpu],
        *,
        sysfs: Sysfs | None = None,
        period_s: float = 1.0,
        mean_n: int = 5,
        mean_limit_c: float = 90.0,
        sample_limit_c: float = 95.0,
        on_breach: Callable[[str], None] | None = None,
        read: Callable[[Gpu], float | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gpus = tuple({g.bus: g for g in gpus}.values())
        self.sysfs = sysfs if sysfs is not None else Sysfs()
        self.period_s = period_s
        self.mean_n = mean_n
        self.mean_limit_c = mean_limit_c
        self.sample_limit_c = sample_limit_c
        self.on_breach = on_breach
        self._read = read if read is not None else self._read_junction
        self._clock = clock
        self.max_c: dict[str, float] = {}
        self.last_c: dict[str, float] = {}
        self.breach: str | None = None
        self.n_samples = 0
        self.missing: list[str] = []
        self._history: dict[str, deque[float]] = {
            g.bus: deque(maxlen=max(mean_n, 1)) for g in self.gpus
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_junction(self, gpu: Gpu) -> float | None:
        return junction_temp_c(self.sysfs, gpu)

    def check_once(self) -> None:
        """Sample every GPU once and apply the abort rule."""
        reasons: list[str] = []
        with self._lock:
            for gpu in self.gpus:
                bus = gpu.bus
                value = self._read(gpu)
                if value is None:
                    if bus not in self.missing:
                        self.missing.append(bus)
                    continue
                self.n_samples += 1
                self.last_c[bus] = value
                self.max_c[bus] = max(value, self.max_c.get(bus, value))
                hist = self._history[bus]
                hist.append(value)
                why = evaluate_temps(
                    list(hist),
                    mean_n=self.mean_n,
                    mean_limit_c=self.mean_limit_c,
                    sample_limit_c=self.sample_limit_c,
                )
                if why is not None:
                    reasons.append(f"gpu:{bus}: {why}")
            first = self.breach is None and bool(reasons)
            if first:
                self.breach = "; ".join(reasons)
        if first and self.on_breach is not None:
            self.on_breach(self.breach or "")

    def _run(self) -> None:
        while not self._stop.is_set():
            began = self._clock()
            self.check_once()
            wait = self.period_s - (self._clock() - began)
            if self._stop.wait(max(wait, 0.0)):
                return

    def start(self) -> None:
        """Start the sampling thread (no-op when it already runs)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="pciescope-temp-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_JOIN_TIMEOUT_S)


def vram_info(bus: str, sysfs: Sysfs | None = None) -> tuple[int, int] | None:
    """Return ``(free_bytes, total_bytes)`` of the GPU on ``bus``.

    None when unreadable; ``sysfs`` None means the real ``/sys``.
    """
    view = sysfs if sysfs is not None else Sysfs()
    try:
        base = f"bus/pci/devices/0000:{normalize_bus(bus)}:00.0"
    except ValueError:
        return None
    used = view.read_int(f"{base}/mem_info_vram_used")
    total = view.read_int(f"{base}/mem_info_vram_total")
    if used is None or total is None:
        return None
    return max(total - used, 0), total


def gpu_suspended(bus: str, sysfs: Sysfs | None = None) -> bool:
    """True when the GPU on ``bus`` is runtime-suspended (D3).

    Reading most attributes of a suspended GPU (VRAM counters, link state)
    resumes it and its bridge (10-35 ms and a kernel log line each time);
    ``power/runtime_status`` does not. Callers that only plan or check, for
    example a dry run, use this to leave a sleeping GPU alone. An unreadable
    status counts as awake.
    """
    view = sysfs if sysfs is not None else Sysfs()
    try:
        base = f"bus/pci/devices/0000:{normalize_bus(bus)}:00.0"
    except ValueError:
        return False
    status = view.read(f"{base}/power/runtime_status")
    return status is not None and status.strip() == "suspended"


@dataclass(frozen=True)
class KfdProc:
    """A KFD process with its eviction counters of the watched GPUs."""

    pid: int
    comm: str
    cmdline: str
    evicted_ms: dict[str, int]  # bus -> ms


class EvictionWatcher:
    """Reads KFD ``evicted_ms`` of every process for the probed GPUs.

    A measured window with an ``evicted_ms`` jump of a process on the probed
    GPUs is invalid (box rules): ``snapshot()`` before and after, then
    ``diff()``.
    """

    def __init__(
        self,
        gpus: Sequence[Gpu],
        *,
        sysfs: Sysfs | None = None,
        procfs: Sysfs | None = None,
        exclude_pids: Iterable[int] = (),
    ) -> None:
        self.gpus = tuple({g.bus: g for g in gpus}.values())
        self.sysfs = sysfs if sysfs is not None else Sysfs()
        self.procfs = procfs if procfs is not None else Sysfs("/proc")
        self.exclude_pids = frozenset(exclude_pids)

    def unknown_buses(self) -> list[str]:
        """Return the watched buses without a KFD gpu_id (no eviction data)."""
        return [g.bus for g in self.gpus if g.kfd_gpu_id is None]

    def snapshot(self) -> dict[int, KfdProc]:
        """Return every KFD process that has counters for a watched GPU."""
        by_pid: dict[int, dict[str, int]] = {}
        for gpu in self.gpus:
            for pid, ms in evicted_ms(self.sysfs, gpu).items():
                if pid not in self.exclude_pids:
                    by_pid.setdefault(pid, {})[gpu.bus] = ms
        procs: dict[int, KfdProc] = {}
        for pid, per_bus in sorted(by_pid.items()):
            raw = self.procfs.read_bytes(f"{pid}/cmdline")
            cmdline = _decode_cmdline(raw) if raw is not None else ""
            procs[pid] = KfdProc(pid, process_name(pid, self.procfs), cmdline, per_bus)
        return procs

    @staticmethod
    def diff(
        before: dict[int, KfdProc], after: dict[int, KfdProc]
    ) -> dict[int, dict[str, int]]:
        """Return per pid per bus the ms gained between two snapshots.

        Only pids and buses present in both snapshots appear; zero entries are
        kept. A negative value means the counter was reset (process restart).
        """
        out: dict[int, dict[str, int]] = {}
        for pid, old in before.items():
            new = after.get(pid)
            if new is None:
                continue
            out[pid] = {
                bus: new.evicted_ms[bus] - ms
                for bus, ms in old.evicted_ms.items()
                if bus in new.evicted_ms
            }
        return out
