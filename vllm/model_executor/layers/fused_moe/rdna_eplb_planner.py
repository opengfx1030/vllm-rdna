# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online EPLB planner: a numpy-only helper process that turns counters into swaps.

Rank 0 of the worker owns a :class:`PlannerClient`. The client allocates three
shared-memory blocks (the counter state of every layer, the call ring, the owner
table), spawns this file as a helper process (``python -I -S rdna_eplb_planner.py
--serve ...``, niced to 19) and talks to it over two pipes with one short text line
per message:

* ``S <step> <gseq>``: the snapshot in shared memory is complete (``gseq`` = records
  ever written to the ring). The helper copies the three blocks to private memory
  before it does anything else, so the client may overwrite them with the next
  snapshot as soon as the answer has been polled (see ``PlannerClient.pending``);
* answer ``P <step> <l:a:b,...|-> |<json report>``: swaps ``(layer_pos, a, b)``;
* ``E <step> <message>``: the plan failed, keep the placement; ``M``: debug, which
  of ``torch`` / ``vllm`` the helper has imported (a test checks that it has not).

Nothing here blocks the caller: ``submit`` and ``poll`` are one non-blocking pipe
write and read; a dead helper turns into ``alive() == False`` and ``poll() == []``.
The helper exits when its stdin closes (a dead parent leaves no orphan) and the
client unlinks the shared memory in ``close()`` and via ``atexit``.

The helper must not import torch or ``vllm``: it loads ``rdna_eplb_policy.py`` by
file path (that module needs only numpy), and this file imports only numpy and the
standard library.

Planning (:func:`plan`)
-----------------------
Per layer and window the snapshot is decoded as

* decode (``objective == "touch"``): ``weight[e]`` = touches of expert ``e`` over the
  decode bands divided by the decode calls of the window (call-weighted touch
  frequency), and ``samples[c, r]`` = the ``T`` words (touched experts of rank ``r``)
  of the layer's decode records;
* prefill (``objective == "pairs"``): ``weight[e]`` = prefill pairs per call and
  ``samples`` = the ``P`` words of the layer's prefill records.

Two windows are needed, the previous one (``prev.before`` -> ``prev``) and the current
one (``prev`` -> ``cur``): the policy's holdout judges the candidate fitted on the
pooled windows on each window with the other window's weights, so a one-window
transient is rejected. The ring records of a window are those of the layer with a
per-layer ordinal inside the window's ordinal range (counter word ``ordinal`` of the
two snapshots); ``gseq`` says how many ring slots hold data and where the oldest is
(``slot = record % ring_records``), so a ring that wrapped, or is shorter than the
window, yields the newest records it still holds. A layer whose owner row changed
between the three snapshots (a commit) is skipped until two windows of the new
placement exist: the ring words are per rank of the placement in force.

Per layer ``policy.decide_swaps`` (realised predictor, holdout, ``distinct``) picks the
best prefix of the swap descent; across layers the prefixes with the best predicted
net gain (``gain * horizon - safety * move``) are kept first until ``max_swaps``.
The answer is a deterministic function of the input.
"""

import argparse
import atexit
import contextlib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from types import ModuleType
from typing import Any

if __name__ == "__main__" and "--site" in sys.argv:  # the helper process, run with -S
    sys.path.insert(0, sys.argv[sys.argv.index("--site") + 1])

import numpy as np  # noqa: E402

logger = logging.getLogger("vllm.rdna_eplb_planner")

# ring record words (see tools/rdna/eplb/summary.py and rdna_fused_glue.cu)
W_LAYER, W_PHASE, W_CALL, W_T0, W_P0 = 0, 6, 7, 8, 16
PHASE_DECODE, PHASE_PREFILL = 0, 1


def _load_policy() -> ModuleType:
    """The pure-numpy policy: normal import in the worker, by path in the helper."""
    if __package__:
        return importlib.import_module(f"{__package__}.rdna_eplb_policy")
    name = "rdna_eplb_policy"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


policy: Any = _load_policy()


@dataclass(frozen=True)
class PlannerSpec:
    layers: int  # MoE layers registered, in decoder order
    experts: int  # global experts per layer
    ranks: int  # EP ranks
    nb: int  # touch bands
    edges: tuple[int, ...]  # band edges (valid rows), len nb
    state_words: int  # int64 words of one layer's counter state
    off_touch: int  # rdna_eplb_stats.offsets(): touch[nb][E]
    off_calls: int  # calls[nb + 1]
    off_rows: int  # rows[2]
    off_ordinal: int  # calls of the layer so far, sampled or not
    ring_records: int  # records of the call ring (0 = none)
    ring_words: int  # 32
    max_swaps: int  # cap of one plan
    objective: str  # "touch" | "pairs"
    expert_bytes: int  # bytes moved per expert (for the move cost)
    # policy knobs (defaults: tools/rdna/eplb/policy.PolicyConfig; the transfer and
    # horizon numbers are [INFERENCE] until measured on the island)
    horizon: int = 10000  # layer calls a placement is expected to stay useful
    min_samples: int = 32  # ring records per window below which a layer is skipped
    min_rel_gain: float = 0.01
    safety: float = 2.0
    bw_gbps: float = 14.0
    stall: float = 0.25
    layer_fixed_us: float = 300.0

    def __post_init__(self) -> None:
        if self.objective not in ("touch", "pairs"):
            raise ValueError(f"objective must be touch or pairs: {self.objective!r}")
        if len(self.edges) != self.nb:
            raise ValueError("edges must hold one edge per band")
        if not 2 <= self.ranks <= 8 or self.experts % self.ranks:
            raise ValueError("ranks must be 2..8 and divide the experts")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "PlannerSpec":
        d = json.loads(text)
        d["edges"] = tuple(d["edges"])
        return cls(**d)


@dataclass
class Snapshot:
    """Private copy of the shared blocks at one submit.

    ``before`` links the snapshot of the previous submit (state and owner only; the
    ring of the newest snapshot holds the records of both windows).
    """

    state: np.ndarray  # int64 [layers, state_words]
    owner: np.ndarray  # uint8 [layers, experts]
    gseq: int
    ring: np.ndarray | None = None  # int32 [ring_records, ring_words]
    before: "Snapshot | None" = None


class _Ring:
    """The ring in record order, indexed once per plan (no copy of the records)."""

    def __init__(self, ring: np.ndarray | None, gseq: int) -> None:
        records = 0 if ring is None else ring.shape[0]
        self.ring = ring
        if ring is None or records == 0 or gseq <= 0:
            self.slots = np.empty(0, dtype=np.int64)
        else:
            n = min(gseq, records)
            self.slots = np.arange(gseq - n, gseq, dtype=np.int64) % records
        if ring is None or not self.slots.size:
            self.layer = self.phase = self.call = np.empty(0, dtype=np.int32)
        else:
            self.layer = ring[self.slots, W_LAYER]
            self.phase = ring[self.slots, W_PHASE]
            self.call = ring[self.slots, W_CALL]

    def samples(
        self, layer: int, phase: int, lo: int, hi: int, base: int, ranks: int
    ) -> np.ndarray:
        """``[n, ranks]`` loads of the layer's records with ordinal in [lo, hi)."""
        if not self.slots.size:
            return np.empty((0, ranks), dtype=np.float64)
        assert self.ring is not None
        m = (
            (self.layer == layer)
            & (self.phase == phase)
            & (self.call >= lo)
            & (self.call < hi)
        )
        return self.ring[self.slots[m], base : base + ranks].astype(np.float64)


def ring_samples(
    spec: PlannerSpec,
    ring: np.ndarray,
    gseq: int,
    layer: int,
    lo: int,
    hi: int,
) -> np.ndarray:
    """Per-call rank loads of one layer in the ordinal window ``[lo, hi)``.

    ``T`` words (decode) for ``objective == "touch"``, ``P`` words (prefill) for
    ``"pairs"``; oldest record first.
    """
    touch = spec.objective == "touch"
    return _Ring(ring, gseq).samples(
        layer,
        PHASE_DECODE if touch else PHASE_PREFILL,
        lo,
        hi,
        W_T0 if touch else W_P0,
        spec.ranks,
    )


def _window(spec: PlannerSpec, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """``(sum of the objective's per-call weight, calls)`` between two state rows."""
    e, nb = spec.experts, spec.nb
    if spec.objective == "touch":
        ta = a[spec.off_touch : spec.off_touch + nb * e]
        tb = b[spec.off_touch : spec.off_touch + nb * e]
        total = (tb - ta).reshape(nb, e).sum(axis=0)
        lo, hi = spec.off_calls, spec.off_calls + nb
        calls = int((b[lo:hi] - a[lo:hi]).sum())
    else:
        total = b[e : 2 * e] - a[e : 2 * e]  # pairs[1]: prefill
        calls = int(b[spec.off_calls + nb] - a[spec.off_calls + nb])
    return total.astype(np.float64), calls


def _config(spec: PlannerSpec) -> Any:
    return policy.PolicyConfig(
        interval=spec.horizon,
        objective=spec.objective,
        min_rel_gain=spec.min_rel_gain,
        safety=spec.safety,
        holdout=True,
        bw_gbps=spec.bw_gbps,
        stall=spec.stall,
        layer_fixed_us=spec.layer_fixed_us,
        max_swaps=spec.max_swaps,
    )


def _plan_layer(
    spec: PlannerSpec,
    cfg: Any,
    geometry: Any,
    ring: _Ring,
    layer: int,
    owner: np.ndarray,
    pp: Snapshot,
    p: Snapshot,
    c: Snapshot,
) -> tuple[Any | None, dict[str, Any]]:
    """``(decision or None, report entry)`` of one layer."""
    entry: dict[str, Any] = {"reason": "", "swaps": 0}
    if not (
        np.array_equal(pp.owner[layer], owner) and np.array_equal(p.owner[layer], owner)
    ):
        entry["reason"] = "placement changed in the last two windows"
        return None, entry
    rows = [s.state[layer] for s in (pp, p, c)]
    old_sum, old_calls = _window(spec, rows[0], rows[1])
    new_sum, new_calls = _window(spec, rows[1], rows[2])
    entry["calls"] = [old_calls, new_calls]
    if old_calls <= 0 or new_calls <= 0:
        entry["reason"] = "no calls of the objective's phase in a window"
        return None, entry
    touch = spec.objective == "touch"
    phase, base = (PHASE_DECODE, W_T0) if touch else (PHASE_PREFILL, W_P0)
    ords = [int(r[spec.off_ordinal]) for r in rows]
    old = ring.samples(layer, phase, ords[0], ords[1], base, spec.ranks)
    new = ring.samples(layer, phase, ords[1], ords[2], base, spec.ranks)
    m = min(old.shape[0], new.shape[0])
    entry["samples"] = [int(old.shape[0]), int(new.shape[0])]
    if m < spec.min_samples:
        entry["reason"] = "too few ring records in a window"
        return None, entry
    # equal halves, so the policy's holdout split is exactly previous/current window
    samples = np.concatenate([old[-m:], new[-m:]])
    w_old, w_new = old_sum / old_calls, new_sum / new_calls
    pooled = (old_sum + new_sum) / (old_calls + new_calls)
    cur = owner.astype(np.int16)
    d = policy.decide_swaps(
        pooled,
        cur,
        spec.ranks,
        cfg,
        samples,
        older=w_old,
        newer=w_new,
        geometry=geometry,
        distinct=True,
    )
    unit = policy.unit_us(cfg, policy.DEFAULT_COST)
    now = float(samples.max(axis=1).mean())
    entry.update(
        reason=d.reason,
        cur_max_load=now,
        pred_max_load=now - d.gain_us_per_step / unit,
        gain_us=d.gain_us_per_step,
        move_us=d.move_us,
        experts_moved=d.experts_moved,
        swaps=len(d.swaps) if d.owner is not None else 0,
    )
    return (d if d.owner is not None else None), entry


def plan(
    prev: Snapshot | None, cur: Snapshot, owner: np.ndarray, spec: PlannerSpec
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Swaps ``(layer_pos, a, b)`` for the window ``prev -> cur`` and a report.

    ``owner`` is the placement in force, ``[layers, experts]`` (rank of each expert).
    The holdout needs the window before ``prev``, which is ``prev.before``; without
    two windows the answer is ``[]``. Swaps of one layer are applied in order (``a``
    of the layer's heaviest rank goes to ``b``'s rank and vice versa); no expert
    appears twice in a plan and ``owner[a] != owner[b]`` at the placement in force.
    """
    report: dict[str, Any] = {"objective": spec.objective, "layers": {}}
    if prev is None:
        report["reason"] = "first snapshot: baseline primed"
        return [], report
    if prev.before is None:
        report["reason"] = "second snapshot: two windows are needed for the holdout"
        return [], report
    cfg = _config(spec)
    geometry = policy.Geometry(expert_bytes=spec.expert_bytes, resident_extra_bytes=0)
    ring = _Ring(cur.ring, cur.gseq)
    chosen: list[tuple[float, int, Any]] = []
    for layer in range(spec.layers):
        d, entry = _plan_layer(
            spec, cfg, geometry, ring, layer, owner[layer], prev.before, prev, cur
        )
        report["layers"][layer] = entry
        if d is not None:
            net = d.gain_us_per_step * spec.horizon - spec.safety * d.move_us
            chosen.append((net, layer, d))
    chosen.sort(key=lambda t: (-t[0], t[1]))
    swaps: list[tuple[int, int, int]] = []
    for net, layer, d in chosen:
        room = spec.max_swaps - len(swaps)
        if room <= 0:
            report["layers"][layer]["reason"] = "plan is full"
            report["layers"][layer]["swaps"] = 0
            continue
        take = d.swaps[:room]
        swaps.extend((layer, int(a), int(b)) for a, b in take)
        report["layers"][layer]["swaps"] = len(take)
        report["layers"][layer]["net_us"] = net
    report["reason"] = "rebalance" if swaps else "keep the placement"
    report["swaps"] = len(swaps)
    return swaps, report


# ---- helper process ---------------------------------------------------------------


class _Planner:
    """State of the helper: the snapshot chain and the plan of each submit."""

    def __init__(self, spec: PlannerSpec) -> None:
        self.spec = spec
        self.last: Snapshot | None = None

    def step(self, snap: Snapshot) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
        swaps, report = plan(self.last, snap, snap.owner, self.spec)
        if self.last is not None:
            self.last.ring = None  # the newest ring holds both windows
            self.last.before = None
        snap.before = self.last
        self.last = snap
        return swaps, report


def _attach(name: str) -> shared_memory.SharedMemory:
    """Attach to a block of the client without registering it for unlink at exit."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # type: ignore[call-arg]
    except TypeError:  # Python < 3.13 registers every attach with the resource tracker
        from multiprocessing import resource_tracker

        register = resource_tracker.register
        resource_tracker.register = lambda *a, **k: None  # type: ignore[assignment]
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = register  # type: ignore[assignment]


def _answer(step: int, swaps: list[tuple[int, int, int]], report: dict) -> bytes:
    text = ",".join(f"{lay}:{a}:{b}" for lay, a, b in swaps) or "-"
    body = json.dumps(report, separators=(",", ":"))
    return f"P {step} {text}|{body}\n".encode()


def _serve(spec_json: str, names: list[str]) -> int:
    with contextlib.suppress(OSError):
        os.nice(19)
    spec = PlannerSpec.from_json(spec_json)
    blocks = [_attach(n) for n in names]
    layers, e = spec.layers, spec.experts
    state = np.ndarray((layers, spec.state_words), np.int64, buffer=blocks[0].buf)
    ring = np.ndarray(
        (spec.ring_records, spec.ring_words), np.int32, buffer=blocks[1].buf
    )
    owner = np.ndarray((layers, e), np.uint8, buffer=blocks[2].buf)
    planner = _Planner(spec)
    out = sys.stdout.buffer
    for raw in sys.stdin.buffer:
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == b"Q":
            break
        if cmd == b"M":
            bad = sorted(m for m in ("torch", "vllm") if m in sys.modules)
            out.write(b"M " + (",".join(bad) or "-").encode() + b"\n")
            out.flush()
            continue
        if cmd != b"S":
            continue
        step, gseq = int(parts[1]), int(parts[2])
        try:
            snap = Snapshot(state.copy(), owner.copy(), gseq, ring.copy())
            swaps, report = planner.step(snap)
            line = _answer(step, swaps, report)
        except Exception as exc:  # keep serving: the answer is "keep the placement"
            print(f"planner: step {step} failed: {exc!r}", file=sys.stderr, flush=True)
            line = f"E {step} {type(exc).__name__}: {exc}\n".replace("\r", " ").encode()
        try:
            out.write(line)
            out.flush()
        except OSError:
            break  # the parent is gone
    return 0


# ---- client -----------------------------------------------------------------------


class PlannerClient:
    """Rank 0's handle on the helper; see the module docstring for the protocol."""

    def __init__(self, spec: PlannerSpec) -> None:
        self.spec = spec
        self._blocks: list[shared_memory.SharedMemory] = []
        self.state = self._alloc((spec.layers, spec.state_words), np.int64)
        self.ring = self._alloc((spec.ring_records, spec.ring_words), np.int32)
        self.owner = self._alloc((spec.layers, spec.experts), np.uint8)
        self._proc: subprocess.Popen[bytes] | None = None
        self._wfd = self._rfd = -1
        self._buf = b""
        self._pending = False
        self._dead = False
        self._closed = False
        self.answered_step: int | None = None
        self.last_error: str | None = None
        self._report_json: bytes | None = None
        atexit.register(self.close)

    def _alloc(self, shape: tuple[int, int], dtype: Any) -> np.ndarray:
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        block = shared_memory.SharedMemory(create=True, size=max(size, 1))
        self._blocks.append(block)
        return np.ndarray(shape, dtype, buffer=block.buf)

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Spawn the helper (niced to 19, no torch, no vllm)."""
        if self._proc is not None or self._closed:
            return
        env = dict(os.environ)
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[var] = "1"
        cmd = [
            sys.executable,
            # -I: no PYTHONPATH, no script directory on sys.path. -S: no site, whose
            # .pth files may import torch/vllm (an editable install does); numpy is
            # reached through --site instead.
            "-I",
            "-S",
            os.path.abspath(__file__),
            "--serve",
            "--site",
            os.path.dirname(os.path.dirname(os.path.abspath(np.__file__))),
            "--spec",
            self.spec.to_json(),
            "--shm",
            *(b.name for b in self._blocks),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
                close_fds=True,
            )
        except OSError as exc:
            self._fail(f"cannot start the planner helper: {exc}")
            return
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._wfd = self._proc.stdin.fileno()
        self._rfd = self._proc.stdout.fileno()
        os.set_blocking(self._wfd, False)
        os.set_blocking(self._rfd, False)

    def alive(self) -> bool:
        if self._dead or self._proc is None:
            return False
        if self._proc.poll() is not None:
            self._fail("the planner helper exited")
            return False
        return True

    def _fail(self, why: str) -> None:
        if not self._dead:
            self._dead = True
            self._pending = False
            self.last_error = why
            logger.warning("EPLB planner: %s; the placement stays as it is", why)

    @property
    def pending(self) -> bool:
        """A submit is unanswered: the helper may still read the shared blocks."""
        return self._pending and not self._dead

    def close(self) -> None:
        """Stop the helper and unlink the shared memory (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._dead = True
        self._pending = False
        atexit.unregister(self.close)
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(OSError):
                if proc.stdin is not None:
                    proc.stdin.close()  # EOF: the helper's own exit signal
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
        for block in self._blocks:
            with contextlib.suppress(FileNotFoundError):
                block.unlink()
            with contextlib.suppress(BufferError):  # numpy views are still alive
                block.close()
        self._blocks = []

    # -- the two hot-path calls ------------------------------------------------

    def submit(self, step: int, gseq: int) -> bool:
        """Tell the helper the snapshot in shared memory is complete (non-blocking).

        ``False`` when nothing was sent: an answer is still pending (the helper has
        not copied the blocks yet, so they must not be rewritten), the pipe is full
        or the helper is gone.
        """
        if self._dead or self._proc is None or self._pending:
            return False
        try:
            os.write(self._wfd, b"S %d %d\n" % (step, gseq))
        except BlockingIOError:
            return False
        except OSError:
            self._fail("the planner helper closed its pipe")
            return False
        self._pending = True
        return True

    def poll(self) -> list[tuple[int, int, int]] | None:
        """``None`` = no answer yet, ``[]`` = keep the placement, else the swaps."""
        if self._dead or self._proc is None:
            return []
        try:
            data = os.read(self._rfd, 1 << 16)
        except BlockingIOError:
            data = None
        except OSError:
            self._fail("the planner helper closed its pipe")
            return []
        if data == b"":
            self._fail("the planner helper exited")
            return []
        if data:
            self._buf += data
        end = self._buf.find(b"\n")
        if end < 0:
            if data is None and self._proc.poll() is not None:
                self._fail("the planner helper exited")
                return []
            return None
        line, self._buf = self._buf[:end], self._buf[end + 1 :]
        self._pending = False
        return self._parse(line)

    def _parse(self, line: bytes) -> list[tuple[int, int, int]]:
        kind, _, rest = line.partition(b" ")
        if kind == b"P":
            step, _, rest = rest.partition(b" ")
            text, _, report = rest.partition(b"|")
            self.answered_step = int(step)
            self._report_json = report
            if text == b"-":
                return []
            return [
                (int(x), int(y), int(z))
                for x, y, z in (item.split(b":") for item in text.split(b","))
            ]
        self._report_json = None
        self.last_error = rest.decode(errors="replace")
        logger.warning("EPLB planner: %s", self.last_error)
        return []

    @property
    def last_report(self) -> dict[str, Any] | None:
        """The report of the last answer (parsed on demand, not on ``poll``)."""
        return None if self._report_json is None else json.loads(self._report_json)

    def helper_modules(self, timeout: float = 5.0) -> list[str]:
        """Which of ``torch`` / ``vllm`` the helper imported (blocking; for tests)."""
        import select

        assert self._proc is not None
        os.set_blocking(self._wfd, True)
        try:
            os.write(self._wfd, b"M\n")
        finally:
            os.set_blocking(self._wfd, False)
        end = self._buf.find(b"\n")
        while end < 0:
            if not select.select([self._rfd], [], [], timeout)[0]:
                raise TimeoutError("no answer from the planner helper")
            self._buf += os.read(self._rfd, 1 << 16)
            end = self._buf.find(b"\n")
        line, self._buf = self._buf[:end], self._buf[end + 1 :]
        names = line.split()[1].decode()
        return [] if names == "-" else names.split(",")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--serve", action="store_true", required=True)
    ap.add_argument("--site", help="directory that holds numpy (put on sys.path early)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--shm", nargs=3, required=True)
    a = ap.parse_args(argv)
    return _serve(a.spec, a.shm)


if __name__ == "__main__":
    sys.exit(main())
