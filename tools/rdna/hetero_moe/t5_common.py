# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared pieces of the T5 harness (hub replay against 4 real expert servers).

numpy only, importable without torch or the vLLM stack:

* the deterministic input family (``make_inputs``) that the golden run on the
  V620 and the hub replay both generate from ``(seed, width, pattern)``;
* the host statement of the wire rules (``expected_split``, ``server_ids``,
  ``combine_reference``) and the per-layer comparison (``check_layer``);
* the golden file (``golden_<server>.npz``): the digest of what each server's
  local chain computes for every (width, pattern, layer), plus the raw fp16 bytes
  for widths up to ``RAW_MAX_WIDTH``;
* percentile tables and the ``T5 <tag> PASS|FAIL`` line format.

Torch and the hetero_moe package are imported lazily by the two helpers that
need them (``build_tables``, ``expected_split``).
"""

import hashlib
import inspect
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GOLDEN_FORMAT = "hx-t5-golden/1"
RAW_MAX_WIDTH = 48  # widths up to this keep the raw y bytes in the golden file
DEFAULT_WIDTHS = (3, 24, 48, 2048)
DEFAULT_PATTERNS = 4
ZIPF_ALPHA = 1.1
# Gates of spec 11.2 (T5 / S3 rows): the constants and functions are below, at
# ``EXPERT_US`` (the former fixed 47 / 161 / 256 us table was the H2 remote path).

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_ABORT = 3
EXIT_SETUP = 4


# ---- digests ------------------------------------------------------------------------


def digest(data) -> str:
    """blake2b-16 hex of bytes or of the raw bytes of an array (C order)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(_raw(data))
    return h.hexdigest()


def _raw(data) -> memoryview | bytes:
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).view(np.uint8).reshape(-1).data
    return bytes(data)


# ---- inputs -------------------------------------------------------------------------


@dataclass(frozen=True)
class StepInputs:
    """One step of layer calls: ``x[l % x_pool]`` is the input of layer ``l``."""

    x: np.ndarray  # fp16 [x_pool, T, H]
    ids: np.ndarray  # int32 [L, T, K], distinct per row, -1 on pad rows
    w: np.ndarray  # fp32 [L, T, K], rows sum to 1

    @property
    def width(self) -> int:
        return int(self.x.shape[1])

    @property
    def nlayers(self) -> int:
        return int(self.ids.shape[0])

    def x_of(self, layer: int) -> np.ndarray:
        return self.x[layer % self.x.shape[0]]


def make_inputs(
    seed: int,
    width: int,
    pattern: int,
    *,
    nlayers: int = 48,
    num_experts: int = 512,
    top_k: int = 10,
    hidden: int = 2560,
    x_pool: int = 4,
    pad_rows: int = 0,
) -> StepInputs:
    """Deterministic step inputs, a function of ``(seed, width, pattern)`` and the
    shape arguments only (and of the numpy version: the golden file records it).

    ``x ~ N(0, 0.5^2)`` as fp16. ``ids``: per layer a Zipf(1.1) popularity over a
    random permutation of the experts (the hot experts differ by layer) and a
    Gumbel top-k per row, so a row never repeats an expert. ``w``: positive,
    normalised to sum 1 per row in fp32. The last ``pad_rows`` rows get ids -1.
    """
    if not 0 <= pad_rows < width:
        raise ValueError(f"pad_rows={pad_rows} must be in [0, width={width})")
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")
    rng = np.random.Generator(np.random.PCG64([seed, width, pattern]))
    x = (rng.standard_normal((x_pool, width, hidden), dtype=np.float32) * 0.5).astype(
        np.float16
    )
    log_pop = -ZIPF_ALPHA * np.log(np.arange(1, num_experts + 1, dtype=np.float64))
    ids = np.empty((nlayers, width, top_k), np.int32)
    for layer in range(nlayers):
        score = np.empty(num_experts, np.float64)
        score[rng.permutation(num_experts)] = log_pop
        score = score[None, :] + rng.gumbel(size=(width, num_experts))
        ids[layer] = np.argsort(-score, axis=1, kind="stable")[:, :top_k]
    w = rng.random((nlayers, width, top_k), dtype=np.float32) + np.float32(0.05)
    w /= w.sum(axis=2, keepdims=True)
    if pad_rows:
        ids[:, width - pad_rows :, :] = -1
    return StepInputs(x, ids, w)


def inputs_digest(inp: StepInputs) -> str:
    """blake2b-16 over the bytes of x, ids and w."""
    h = hashlib.blake2b(digest_size=16)
    for arr in (inp.x, inp.ids, inp.w):
        h.update(_raw(arr))
    return h.hexdigest()


def synthetic_y_hub(
    seed: int, width: int, pattern: int, layer: int, T: int, H: int
) -> np.ndarray:
    """The hub's own partial of a layer: small deterministic fp16 ``[T, H]``."""
    rng = np.random.Generator(np.random.PCG64([seed, width, pattern, layer, 0x59]))
    return (rng.standard_normal((T, H), dtype=np.float32) * 0.05).astype(np.float16)


# ---- the wire rules on the host -----------------------------------------------------


def expected_split(
    tables, layer: int, ids: np.ndarray, k_keep: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(hub_ids, dest)`` of one layer: the plain-loop statement of the dispatch
    rule (``tables.split_pairs_loop``) on ``ids`` [T, K]. ``tables`` is an
    ``IslandTables``; the rule is taken from the module that defines it."""
    module = inspect.getmodule(type(tables))
    assert module is not None
    return module.split_pairs_loop(ids, tables.layers[layer], layer, k_keep)


def server_ids(ids: np.ndarray, dest: np.ndarray, s: int) -> np.ndarray:
    """Ids that server ``s`` receives: the expert id of a pair that goes to it,
    else -1 (int32 [T, K])."""
    return np.where(dest == s, ids, -1).astype(np.int32)


def combine_reference(y_hub: np.ndarray, ys: Iterable[np.ndarray]) -> np.ndarray:
    """The wire combine: fp32 sum left to right starting from the hub partial,
    rounded once to fp16."""
    acc = y_hub.astype(np.float32)
    for y in ys:
        acc = acc + y.astype(np.float32)
    return acc.astype(np.float16)


# ---- comparison ---------------------------------------------------------------------


def first_diff(got: np.ndarray, want: np.ndarray) -> dict | None:
    """Bytewise comparison of two equally shaped arrays; ``None`` when equal,
    else the count of differing elements and the first one."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return {
            "count": -1,
            "note": f"shape/dtype {got.shape} {got.dtype} vs {want.shape} {want.dtype}",
        }
    a, b = (np.ascontiguousarray(v) for v in (got, want))
    bad = a.view(f"u{a.itemsize}") != b.view(f"u{b.itemsize}")
    count = int(bad.sum())
    if not count:
        return None
    first = tuple(int(i) for i in np.argwhere(bad)[0])
    return {
        "count": count,
        "index": first,
        "got": float(got[first]),
        "want": float(want[first]),
    }


def describe_diff(diff: dict) -> str:
    if "note" in diff:
        return diff["note"]
    return (
        f"{diff['count']} element(s) differ, first at {list(diff['index'])}: "
        f"got {diff['got']!r}, want {diff['want']!r}"
    )


def check_layer(
    layer: int,
    tap: np.ndarray,
    out: np.ndarray,
    hub_ids: np.ndarray,
    y_hub: np.ndarray,
    want_hub_ids: np.ndarray,
    golden_digests: Sequence[str],
    golden_y: Sequence[np.ndarray | None] | None = None,
) -> list[str]:
    """Everything T5 checks for one layer; returns one line per violation.

    ``tap`` fp16 [S, T, H]: copies of the servers' replies; ``out`` fp16 [T, H]:
    the combined output; ``hub_ids`` int32 [T, K]: the hub's ``hub_ids_out``;
    ``golden_digests`` / ``golden_y``: what each server's local chain computed.
    """
    problems: list[str] = []
    for s in range(tap.shape[0]):
        got = digest(tap[s])
        if got != golden_digests[s]:
            problems.append(
                f"layer {layer} server {s}: reply digest {got} != golden "
                f"{golden_digests[s]}"
            )
        raw = None if golden_y is None else golden_y[s]
        if raw is not None:
            diff = first_diff(tap[s], raw)
            if diff is not None:
                problems.append(
                    f"layer {layer} server {s}: reply bytes differ from golden: "
                    + describe_diff(diff)
                )
    diff = first_diff(out, combine_reference(y_hub, tap))
    if diff is not None:
        problems.append(
            f"layer {layer}: combine output differs from the fp32 reference: "
            + describe_diff(diff)
        )
    if not np.array_equal(hub_ids, want_hub_ids):
        bad = np.argwhere(hub_ids != want_hub_ids)
        r, k = (int(i) for i in bad[0])
        problems.append(
            f"layer {layer}: hub_ids_out differs from the split rule at {len(bad)} "
            f"pair(s), first row {r} slot {k}: got {int(hub_ids[r, k])}, want "
            f"{int(want_hub_ids[r, k])}"
        )
    return problems


# ---- golden file --------------------------------------------------------------------


class GoldenError(RuntimeError):
    """A golden file that cannot be used for this run."""


def golden_path(directory: str | Path, server_id: str) -> Path:
    return Path(directory) / f"golden_{server_id}.npz"


def digest_key(width: int, pattern: int, layer: int) -> str:
    return f"{width}/{pattern}/{layer}"


def raw_key(width: int, pattern: int) -> str:
    return f"y_{width}_{pattern}"


class Golden:
    """Digests (and raw replies for narrow widths) of one server's local chain."""

    def __init__(self, meta: dict, digests: dict[str, str], raw: dict[str, np.ndarray]):
        self.meta = meta
        self.digests = digests
        self.raw = raw

    @classmethod
    def new(
        cls,
        plan_id: str,
        server_id: str,
        seed: int,
        layers: Sequence[int],
    ) -> "Golden":
        meta = {
            "format": GOLDEN_FORMAT,
            "plan_id": plan_id,
            "server_id": server_id,
            "seed": int(seed),
            "numpy_version": np.__version__,
            "layers": [int(layer) for layer in layers],
            "widths": {},
        }
        return cls(meta, {}, {})

    def add_width(
        self, width: int, patterns: int, k_keep: int, inputs_digests: dict[int, str]
    ) -> None:
        self.meta["widths"][str(width)] = {
            "patterns": int(patterns),
            "nlayers": len(self.meta["layers"]),
            "k_keep": int(k_keep),
            "inputs_digest": {str(p): d for p, d in inputs_digests.items()},
        }

    def add_pattern(self, width: int, pattern: int, ys: Sequence[np.ndarray]) -> None:
        """``ys[i]``: fp16 [T, H] reply of ``meta['layers'][i]``."""
        for layer, y in zip(self.meta["layers"], ys):
            self.digests[digest_key(width, pattern, layer)] = digest(y)
        if width <= RAW_MAX_WIDTH:
            self.raw[raw_key(width, pattern)] = np.stack(ys).astype(np.float16)

    # ---- queries ----

    @property
    def layers(self) -> list[int]:
        return list(self.meta["layers"])

    def widths(self) -> list[int]:
        return sorted(int(w) for w in self.meta["widths"])

    def patterns(self, width: int) -> int:
        return int(self._width(width)["patterns"])

    def k_keep(self, width: int) -> int:
        return int(self._width(width)["k_keep"])

    def inputs_digest(self, width: int, pattern: int) -> str:
        return self._width(width)["inputs_digest"][str(pattern)]

    def layer_digest(self, width: int, pattern: int, layer: int) -> str:
        return self.digests[digest_key(width, pattern, layer)]

    def layer_y(self, width: int, pattern: int, index: int) -> np.ndarray | None:
        """Raw reply of the ``index``-th layer, ``None`` when not stored."""
        arr = self.raw.get(raw_key(width, pattern))
        return None if arr is None else arr[index]

    def _width(self, width: int) -> dict:
        try:
            return self.meta["widths"][str(width)]
        except KeyError:
            raise GoldenError(
                f"golden of {self.meta['server_id']} has no width {width} "
                f"(has {self.widths()})"
            ) from None

    # ---- file ----

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            meta=_encode(self.meta),
            digests=_encode(self.digests),
            **self.raw,
        )
        tmp.replace(path)
        return path


def _encode(obj) -> np.ndarray:
    return np.frombuffer(json.dumps(obj, sort_keys=True).encode(), dtype=np.uint8)


def _decode(arr: np.ndarray):
    return json.loads(arr.tobytes().decode())


def load_golden(
    path: str | Path,
    *,
    plan_id: str | None = None,
    server_id: str | None = None,
) -> Golden:
    """Read and validate a golden file: format, plan, server and the numpy version
    (the input generation is only reproducible under the same numpy)."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = _decode(z["meta"])
            digests = _decode(z["digests"])
            raw = {k: z[k] for k in z.files if k.startswith("y_")}
    except (OSError, KeyError, ValueError) as exc:
        raise GoldenError(f"cannot read golden {path}: {exc}") from exc
    if meta.get("format") != GOLDEN_FORMAT:
        raise GoldenError(
            f"{path}: format {meta.get('format')!r}, expected {GOLDEN_FORMAT!r}"
        )
    if plan_id is not None and meta.get("plan_id") != plan_id:
        raise GoldenError(
            f"{path}: golden is for plan {meta.get('plan_id')}, this run uses {plan_id}"
        )
    if server_id is not None and meta.get("server_id") != server_id:
        raise GoldenError(
            f"{path}: golden is of server {meta.get('server_id')}, expected {server_id}"
        )
    if meta.get("numpy_version") != np.__version__:
        raise GoldenError(
            f"{path}: input generation differs: golden made with numpy "
            f"{meta.get('numpy_version')}, this process runs numpy {np.__version__}"
        )
    return Golden(meta, digests, raw)


# ---- plan / tables helpers (lazy imports) -----------------------------------


def hx_layers(tables) -> list[int]:
    """Layers that talk to the servers: present and not hub-only, ascending."""
    return sorted(li for li, lt in tables.layers.items() if lt.uses_hx)


def build_tables(plan, hub_id: str, spill_k_text: str | None = None):
    """``IslandTables`` of the hub: spill K from ``--spill-k`` (``"3:10,24:20"``),
    else from the plan (``runtime.spill_k``), else the package default. Golden and
    hub replay both call this so they agree on K."""
    from vllm.distributed.hetero_moe import tables as tables_mod

    spill = tables_mod.parse_spill_k(spill_k_text) if spill_k_text else None
    return tables_mod.build_island_tables(
        plan, plan.device_index(hub_id), spill_k=spill
    )


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.replace(" ", "").split(",") if part]


# ---- reports ----------------------------------------------------------------


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50 / p99 / mean / min / max and the count of a sample."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no samples")
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def format_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    """Fixed-width table; floats print with one decimal."""
    cells = [
        [f"{v:.1f}" if isinstance(v, float) else str(v) for v in row] for row in rows
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
        for i, h in enumerate(headers)
    ]
    lines = [
        "  ".join(h.rjust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in cells]
    return "\n".join(lines)


def latency_row(
    width: int,
    mode: str,
    per_layer_us: Sequence[float],
    n_max: float | None = None,
    limit_us: float | None = None,
) -> tuple[list, dict]:
    """Table row and JSON record of a per-layer latency sample (us). ``n_max``: the
    experts of the busiest server per layer, ``limit_us``: the plan-specific
    1.3 x model (``layer_limit_us``); both only for widths with decode routing."""
    stat = percentiles(per_layer_us)
    record = {"width": width, "mode": mode, **stat}
    ratio = None
    if n_max is not None:
        record["n_max"] = n_max
    if limit_us is not None:
        ratio = stat["mean"] / limit_us
        record["limit_us"] = limit_us
        record["ratio_limit"] = ratio
    row = [
        width,
        mode,
        stat["p50"],
        stat["p99"],
        stat["mean"],
        n_max if n_max is not None else "-",
        limit_us if limit_us is not None else "-",
        f"{ratio:.2f}" if ratio is not None else "-",
    ]
    return row, record


LATENCY_HEADERS = (
    "W",
    "mode",
    "p50_us",
    "p99_us",
    "mean_us",
    "n_max",
    "limit_us",
    "mean/limit",
)


# ---- latency gates (spec 11.2, T5 / S3 rows) ----------------------------------
#
# (b) server chain: real-server layer time minus synthetic-server layer time
#     <= 1.15 x n_max x EXPERT_US, n_max = experts of the busiest server per layer,
#     counted from the plan and the replay inputs (judged only with a baseline).
# (c) exchange / layer time against the plan-specific 1.3 x model: recorded.
# The exchange model is perf_model.transport_us of origin/feat/hetero-moe (hub
# dispatch 3 us, push, flag detection, server fixed part 5 us, pull).

EXPERT_US = 6.67  # V620 streaming time of one W4A16 expert (perf_model.expert_us)
CHAIN_TOLERANCE = 1.15
LAYER_TOLERANCE = 1.3
PCIE_BW = 25.0e9
PCIE_LAT_US = 6.3
ROW_B = 2560 * 2
ROW_REQ_B = ROW_B + 10 * 6
DETECT_US = 3.0


def transport_model_us(width: int, nsrv: int) -> float:
    """Model of the exchange of one layer with no expert compute (us)."""
    push = PCIE_LAT_US + nsrv * width * ROW_REQ_B / PCIE_BW * 1e6
    pull = PCIE_LAT_US + nsrv * width * ROW_B / PCIE_BW * 1e6
    return 3.0 + push + DETECT_US + 5.0 + pull


def layer_limit_us(width: int, nsrv: int, n_max: float) -> float:
    """Gate (c): 1.3 x (exchange + the busiest server's experts) per layer."""
    return LAYER_TOLERANCE * (transport_model_us(width, nsrv) + n_max * EXPERT_US)


def chain_limit_us(n_max: float) -> float:
    """Gate (b): 1.15 x the model's expert time of the busiest server per layer."""
    return CHAIN_TOLERANCE * n_max * EXPERT_US


def busiest_server_experts(
    tables, layers: Sequence[int], inp: StepInputs, k_keep: int, nsrv: int
) -> list[int]:
    """Per layer, the distinct experts the busiest server runs for ``inp``."""
    counts = []
    for i, layer in enumerate(layers):
        _, dest = expected_split(tables, layer, inp.ids[i], k_keep)
        best = 0
        for s in range(nsrv):
            mine = server_ids(inp.ids[i], dest, s)
            best = max(best, int(np.unique(mine[mine >= 0]).size))
        counts.append(best)
    return counts


def mean_busiest_experts(
    tables,
    layers: Sequence[int],
    inputs: Iterable[StepInputs],
    k_keep: int,
    nsrv: int,
) -> float:
    """``n_max``: the busiest-server count averaged over layers and patterns."""
    counts = [
        c
        for inp in inputs
        for c in busiest_server_experts(tables, layers, inp, k_keep, nsrv)
    ]
    if not counts:
        raise ValueError("no inputs to count")
    return float(np.mean(counts))


def judge_chain(real_us: float, synth_us: float, n_max: float) -> dict:
    """Gate (b) of one width from the mean layer times (us)."""
    chain = real_us - synth_us
    limit = chain_limit_us(n_max)
    return {
        "real_us": real_us,
        "synthetic_us": synth_us,
        "chain_us": chain,
        "n_max": n_max,
        "limit_us": limit,
        "ratio": chain / (n_max * EXPERT_US),  # 1.0 = exactly the model
        "ok": chain <= limit,
    }


def chain_text(width: int, j: dict) -> str:
    return (
        f"W={width} server chain {j['real_us']:.1f} - {j['synthetic_us']:.1f} "
        f"(synthetic) = {j['chain_us']:.1f} us/layer; limit {CHAIN_TOLERANCE} x "
        f"{j['n_max']:.1f} experts x {EXPERT_US} us = {j['limit_us']:.1f} us "
        f"(ratio to the model {j['ratio']:.2f})"
    )


class BaselineError(RuntimeError):
    """A baseline file that cannot be used for gate (b)."""


def load_baseline(path: str | Path, plan_id: str | None = None) -> dict[int, float]:
    """Graph-mode mean layer time (us) per width of an earlier synthetic-server
    ``bench`` run: the ``t5_run.json`` of ``run --synthetic --modes bench`` or the
    hub's result JSON. Refuses another plan and a run marked as real servers."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise BaselineError(f"cannot read baseline {path}: {exc}") from exc
    hub = doc.get("hub") if isinstance(doc.get("hub"), dict) else doc
    if plan_id is not None and doc.get("plan_id", plan_id) != plan_id:
        raise BaselineError(
            f"{path}: baseline is for plan {doc.get('plan_id')}, this run uses "
            f"{plan_id}"
        )
    marks = [
        m for m in (doc.get("synthetic"), hub.get("servers_synthetic")) if m is not None
    ]
    if marks and not all(marks):
        raise BaselineError(f"{path}: the baseline was run with real servers")
    try:
        records = hub["modes"]["bench"]["records"]
    except (KeyError, TypeError) as exc:
        raise BaselineError(f"{path}: no bench records (modes.bench.records)") from exc
    out = {
        int(r["width"]): float(r["mean"]) for r in records if r.get("mode") == "graph"
    }
    if not out:
        raise BaselineError(f"{path}: no graph-mode bench record")
    return out


# ---- output lines ---------------------------------------------------------------


def status_line(tag: str, status: str, details: str = "") -> str:
    """``T5 <tag> <status> <details>``; status PASS, FAIL, INFO or NOT JUDGED."""
    return f"T5 {tag} {status}" + (f" {details}" if details else "")


def tag_line(tag: str, ok: bool, details: str = "") -> str:
    """``T5 <tag> PASS|FAIL <details>``."""
    return status_line(tag, "PASS" if ok else "FAIL", details)


def emit(tag: str, ok: bool, details: str = "") -> bool:
    print(tag_line(tag, ok, details), flush=True)
    return ok


def note(tag: str, status: str, details: str = "") -> None:
    print(status_line(tag, status, details), flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---- child process environments -----------------------------------------------

# HIP 5 is the second W7800 (bus 07): never used without the lead's approval.
ALLOWED_HIPS = frozenset({0, 1, 2, 3, 4, 6, 7, 8, 9})
_GPU_SELECTORS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


class SetupError(RuntimeError):
    """A process that would start on a GPU it must not use, or without a pin."""


def gpu_child_env(base: Mapping[str, str], hip: int) -> dict[str, str]:
    """Environment of a GPU child: pinned to ``hip`` through HIP_VISIBLE_DEVICES,
    no CPU-only vllm switch, no other device selector. Raises ``SetupError`` for a
    HIP index that is not an int of ``ALLOWED_HIPS`` (HIP 5 is refused)."""
    if not isinstance(hip, int) or isinstance(hip, bool) or hip not in ALLOWED_HIPS:
        raise SetupError(
            f"HIP index {hip!r} refused: allowed {sorted(ALLOWED_HIPS)} (HIP 5 never)"
        )
    env = dict(base)
    for key in ("VLLM_TARGET_DEVICE", *_GPU_SELECTORS):
        env.pop(key, None)
    env["HIP_VISIBLE_DEVICES"] = str(hip)
    return env


def cpu_child_env(base: Mapping[str, str]) -> dict[str, str]:
    """Environment of a CPU-only process: no GPU visible and vllm's platform probe
    (amdsmi enumerates every GPU, whatever HIP_VISIBLE_DEVICES says) short-circuited
    by ``VLLM_TARGET_DEVICE=cpu``."""
    env = dict(base)
    for key in _GPU_SELECTORS:
        env.pop(key, None)
    env["HIP_VISIBLE_DEVICES"] = ""
    env["VLLM_TARGET_DEVICE"] = "cpu"
    return env
