# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension that captures RidgeFill activations inside each worker.

A TP>1 serve runs the model in worker processes, so hooks registered in the
driver never see a forward. ``worker_extension_cls`` injects this class into
every worker; the driver then drives it with ``collective_rpc``.

What is captured
----------------
The split state ``x_S`` is the materialized four-stream state entering layer
``split``. The model forward delays each layer's MoE combine into the next
layer (``Qwen4ExpDecoderLayer.forward`` takes ``prev_block_output`` and
``prev_injection``), so the split layer's ``hidden_states`` argument still
lacks layer ``split - 1``'s MoE output. ``split_state`` applies the split
layer's own ``attn_hyper_connection.combine`` to it, as the runtime does.
The targets are the inputs of each late layer's attention module.

Rank 0 holds every row. Both quantities are replicated on every TP rank of
the Flash-Next serve: the hyperconnection weights are ``ReplicatedLinear``
and ``MergedColumnParallelLinear(disable_tp=True)``
(``models/qwen4_exp/amd/hyperconnection.py``), attention, GDN and the MoE
all-reduce their outputs (``reduce_results=not use_hc_sequence_parallel``),
and HC sequence parallelism (``is_hc_sequence_parallel_enabled``: TP > 1 and
``enable_hc_sp`` or MoE SP, which needs DP > 1) is off. The statistics
capture refuses a split layer with ``use_hc_sequence_parallel`` set, and
every rank keeps an fp64 ``sum x`` per split so the driver can compare them.

Raw capture (8 x 4096 rows)::

    llm.collective_rpc("ridgefill_capture", args=(split, end))
    llm.generate(prompts, params)
    llm.collective_rpc("ridgefill_save", args=(dump_dir,))

writes ``rank<rank>.pt`` with every fp32 row, one entry per forward.

Statistics capture (64 x 16K rows; the 24 targets alone would be ~250 GB
fp32)::

    llm.collective_rpc("ridgefill_stats_begin", args=(splits, end, ...))
    llm.collective_rpc("ridgefill_stats_mode", args=("stats", prompt))
    llm.generate([prompt], params)  # one prompt per mode switch
    llm.collective_rpc("ridgefill_stats_snapshot", args=(path,))
    llm.collective_rpc("ridgefill_stats_mode", args=("raw", prompt))
    llm.generate([held_prompt], params)
    llm.collective_rpc("ridgefill_stats_save_rows", args=("held", path))

``stats`` accumulates every prefill forward into fp64 host sums on rank 0
(``StatsAccumulator``). The GPU adds each forward's fp32 partial sums into
fp32 device accumulators (``addmm_``, no temporaries), and every mode switch,
so once per prompt, folds them into fp64 and zeroes them. Folding per
forward instead would cost the 24,36 sums over a second of host time per
2048-row forward (36 cross terms at about 35 ms each on the box CPU), so the
GPUs would idle for over ten minutes of a 64 x 16K capture. If the device
accumulators (4.6 GB at splits 24,36) do not fit, it falls back to folding
every forward. ``raw`` keeps fp16 copies of every ``raw_stride``-th row and
of the first ``raw_head`` positions, with their positions (``RowStore``).
``stats`` also keeps the first ``stats_head`` positions of every fit prompt
as fp16 rows, so a fit can drop the attention-sink rows exactly. Forwards of
one row (decode) are skipped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

STATS_FORMAT = "ridgefill-stats-v1"
ROWS_FORMAT = "ridgefill-rows-v1"
MODES = ("off", "stats", "raw")


def split_state(module: Any, args: tuple, kwargs: dict) -> Any:
    """The materialized four-stream state entering ``module``, or None.

    ``module`` is the split decoder layer and the arguments are its forward
    pre-hook's. A pending combine (``prev_block_output`` and
    ``prev_injection`` both set) is applied with the layer's
    ``attn_hyper_connection.combine``; without one, ``hidden_states`` is
    already materialized.
    """
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    if hidden is None:
        return None
    block_output = kwargs.get("prev_block_output")
    injection = kwargs.get("prev_injection")
    if block_output is not None and injection is not None:
        hidden = module.attn_hyper_connection.combine(hidden, block_output, injection)
    return hidden


def _block_input(args: tuple, kwargs: dict) -> Any:
    return kwargs.get("hidden_states", args[0] if args else None)


def _attention(layer: Any, index: int) -> Any:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        attn = getattr(layer, "linear_attn", None)
    if attn is None:
        raise RuntimeError(f"layer {index} has no attention module")
    return attn


def _row_positions(positions: Any) -> Any:
    """Absolute row positions; M-RoPE passes ``[3, T]`` whose text rows repeat."""
    if positions is None:
        return None
    return positions[0] if positions.dim() > 1 else positions


def _save_atomic(payload: Any, path: str) -> int:
    """``torch.save`` through a temporary name; returns the file size."""
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    return out.stat().st_size


class StatsAccumulator:
    """fp64 host sums over every row of every prefill forward.

    Rows are shifted by a fixed vector, the row mean of the first forward
    each tensor appears in, so the fp32 partial sums do not cancel against a
    large mean. Per split ``s`` with ``u = x_s - shift_s`` and per late layer
    ``L`` with ``v = y_L - shift_L``::

        x[s]:        shift, sum = sum(u), gram = u^T u
        y[L]:        shift, sum = sum(v), sumsq = sum(v * v)  (per column)
        cross[s][L]: u^T v                                     (L >= s)

    With a zero shift these are the raw sums ``sum x``, ``X^T X``,
    ``sum y``, ``X^T Y`` and ``sum y^2``. The fit centers them, which the
    shift does not change.

    With ``device_sums`` each forward adds into fp32 accumulators on the
    rows' device and ``flush`` (once per prompt) folds them into fp64;
    without it, or once an accumulator does not fit, every forward folds.
    """

    def __init__(
        self,
        splits: list[int] | tuple[int, ...],
        end: int,
        *,
        device_sums: bool = True,
    ) -> None:
        if not splits:
            raise ValueError("need at least one split")
        self.splits = tuple(sorted(int(s) for s in splits))
        self.end = int(end)
        if self.splits[0] < 0 or self.splits[-1] >= self.end:
            raise ValueError(f"splits {self.splits} must lie in [0, {self.end})")
        self.device_sums = bool(device_sums)
        self.rows = 0
        self.forwards = 0
        self.folds = 0
        self.din = 0
        self.dout = 0
        self.x: dict[int, dict[str, Any]] = {}
        self.y: dict[int, dict[str, Any]] = {}
        self.cross: dict[int, dict[int, Any]] = {s: {} for s in self.splits}
        self._shift: dict[tuple[str, int], Any] = {}
        self._xs: dict[int, Any] = {}
        # Device fp32 accumulators and the host fp64 sums they fold into.
        self._dev: dict[tuple, tuple[Any, Any]] = {}
        self._dirty = False
        self._stage: dict[tuple[int, ...], Any] = {}
        self._wide: dict[tuple[int, ...], Any] = {}

    def _fold(self, acc: Any, part: Any) -> None:
        """Add an fp32 partial to an fp64 host accumulator.

        Through a pinned fp32 stage and a cached fp64 buffer: on the box CPU
        the mixed-dtype ``acc.add_(part)`` runs at a third of the speed.
        """
        import torch

        shape = tuple(part.shape)
        if part.device.type != "cpu":
            stage = self._stage.get(shape)
            if stage is None:
                stage = torch.empty(shape, dtype=torch.float32, pin_memory=True)
                self._stage[shape] = stage
            stage.copy_(part)
            part = stage
        wide = self._wide.get(shape)
        if wide is None:
            wide = torch.empty(shape, dtype=torch.float64)
            self._wide[shape] = wide
        wide.copy_(part)
        acc.add_(wide)

    def _device_acc(self, key: tuple, host: Any, like: Any) -> Any:
        """The fp32 device accumulator for ``host``, or None to fold now."""
        import torch

        if not self.device_sums:
            return None
        entry = self._dev.get(key)
        if entry is None:
            try:
                dev = torch.zeros(host.shape, dtype=torch.float32, device=like.device)
            except torch.OutOfMemoryError:
                # Keep what the accumulators hold, then fold every forward.
                self.flush()
                self._dev.clear()
                self.device_sums = False
                return None
            entry = (dev, host)
            self._dev[key] = entry
        return entry[0]

    def _add_sum(self, key: tuple, host: Any, part: Any) -> None:
        dev = self._device_acc(key, host, part)
        if dev is None:
            self._fold(host, part)
        else:
            dev.add_(part)
            self._dirty = True

    def _add_product(self, key: tuple, host: Any, a: Any, b: Any) -> None:
        """``host += a^T b``."""
        dev = self._device_acc(key, host, a)
        if dev is None:
            self._fold(host, a.T @ b)
        else:
            dev.addmm_(a.T, b)
            self._dirty = True

    def flush(self) -> None:
        """Fold the device accumulators into the fp64 sums and zero them."""
        if not self._dirty:
            return
        for dev, host in self._dev.values():
            self._fold(host, dev)
            dev.zero_()
        self._dirty = False
        self.folds += 1

    def _shifted(self, kind: str, index: int, rows: Any) -> tuple[Any, bool]:
        """``rows - shift`` in fp32; the first call fixes the shift."""
        rows32 = rows.detach().float()
        shift = self._shift.get((kind, index))
        first = shift is None
        if shift is None:
            shift = rows32.mean(dim=0)
            self._shift[(kind, index)] = shift
        return rows32 - shift, first

    def add_split(self, split: int, x: Any) -> None:
        """Add one forward's split state ``x`` [T, din]."""
        import torch

        if split not in self.splits:
            raise ValueError(f"{split} is not a split of {self.splits}")
        u, first = self._shifted("x", split, x)
        din = int(u.shape[1])
        if first:
            if self.din and self.din != din:
                raise RuntimeError(f"split {split} is {din} wide, not {self.din}")
            self.din = din
            self.x[split] = {
                "shift": self._shift[("x", split)].double().cpu(),
                "sum": torch.zeros(din, dtype=torch.float64),
                "gram": torch.zeros(din, din, dtype=torch.float64),
            }
        state = self.x[split]
        self._add_sum(("x", split, "sum"), state["sum"], u.sum(dim=0))
        self._add_product(("x", split, "gram"), state["gram"], u, u)
        self._xs[split] = u
        if split == self.splits[0]:
            self.rows += int(u.shape[0])
            self.forwards += 1

    def add_target(self, layer: int, y: Any) -> None:
        """Add one forward's block input ``y`` [T, dout] of late layer ``layer``."""
        import torch

        v, first = self._shifted("y", layer, y)
        dout = int(v.shape[1])
        if first:
            if self.dout and self.dout != dout:
                raise RuntimeError(f"layer {layer} is {dout} wide, not {self.dout}")
            self.dout = dout
            self.y[layer] = {
                "shift": self._shift[("y", layer)].double().cpu(),
                "sum": torch.zeros(dout, dtype=torch.float64),
                "sumsq": torch.zeros(dout, dtype=torch.float64),
            }
        state = self.y[layer]
        self._add_sum(("y", layer, "sum"), state["sum"], v.sum(dim=0))
        self._add_sum(("y", layer, "sumsq"), state["sumsq"], (v * v).sum(dim=0))
        for split in self.splits:
            if split > layer:
                continue
            u = self._xs.get(split)
            if u is None or int(u.shape[0]) != int(v.shape[0]):
                raise RuntimeError(
                    f"layer {layer} has {int(v.shape[0])} rows but split {split} "
                    "has no state of that length in this forward"
                )
            cross = self.cross[split].get(layer)
            if cross is None:
                cross = torch.zeros(self.din, dout, dtype=torch.float64)
                self.cross[split][layer] = cross
            self._add_product(("cross", split, layer), cross, u, v)
        if layer == self.end - 1:
            # The last target of the forward: drop the fp32 split states.
            self._xs.clear()

    def payload(self) -> dict[str, Any]:
        """The cumulative sums (flushed first); the tensors are shared."""
        self.flush()
        return {
            "format": STATS_FORMAT,
            "rows": self.rows,
            "forwards": self.forwards,
            "device_sums": self.device_sums,
            "splits": list(self.splits),
            "end": self.end,
            "din": self.din,
            "dout": self.dout,
            "x": dict(self.x),
            "y": dict(self.y),
            "cross": {s: dict(c) for s, c in self.cross.items()},
        }


class RowStore:
    """fp16 copies of selected rows, with their positions and prompt index.

    A row is kept when its position is below ``head`` or, with ``stride``,
    a multiple of ``stride``. The model's activations are fp16, so the copy
    is exact.
    """

    def __init__(
        self,
        splits: list[int] | tuple[int, ...],
        layers: list[int] | range,
        *,
        stride: int,
        head: int,
    ) -> None:
        self.stride = int(stride)
        self.head = int(head)
        self.x: dict[int, list] = {int(s): [] for s in splits}
        self.y: dict[int, list] = {int(layer): [] for layer in layers}
        self.positions: list = []
        self.prompts: list = []
        self._idx: Any = None

    def begin(self, positions: Any, prompt: int) -> None:
        """Pick this forward's rows from their absolute positions."""
        import torch

        if positions is None:
            raise RuntimeError("row capture needs the split layer's positions")
        keep = positions < self.head
        if self.stride > 0:
            keep = keep | (positions % self.stride == 0)
        idx = keep.nonzero().flatten()
        if int(idx.numel()) == 0:
            self._idx = None
            return
        self._idx = idx
        self.positions.append(positions.index_select(0, idx).to(torch.int32).cpu())
        self.prompts.append(torch.full((int(idx.numel()),), prompt, dtype=torch.int32))

    def _take(self, rows: Any) -> Any:
        import torch

        return rows.detach().index_select(0, self._idx).to(torch.float16).cpu()

    def add_split(self, split: int, x: Any) -> None:
        if self._idx is not None:
            self.x[split].append(self._take(x))

    def add_target(self, layer: int, y: Any) -> None:
        if self._idx is not None:
            self.y[layer].append(self._take(y))

    def count(self) -> int:
        return sum(int(p.numel()) for p in self.positions)

    def payload(self) -> dict[str, Any]:
        import torch

        rows = self.count()

        def cat(chunks: list, name: str) -> Any:
            if not chunks:
                raise RuntimeError(f"no rows captured for {name}")
            out = torch.cat(chunks, dim=0)
            if int(out.shape[0]) != rows:
                raise RuntimeError(f"{name} has {int(out.shape[0])} rows, want {rows}")
            return out

        return {
            "format": ROWS_FORMAT,
            "stride": self.stride,
            "head": self.head,
            "positions": cat(self.positions, "positions"),
            "prompts": cat(self.prompts, "prompts"),
            "x": {s: cat(c, f"split {s}") for s, c in self.x.items()},
            "y": {layer: cat(c, f"layer {layer}") for layer, c in self.y.items()},
        }


class _StatsRun:
    """One worker's statistics-capture state, driven by the probe's hooks."""

    def __init__(
        self,
        rank: int,
        splits: list[int] | tuple[int, ...],
        end: int,
        *,
        raw_stride: int,
        raw_head: int,
        stats_head: int,
        device_sums: bool = True,
    ) -> None:
        self.rank = rank
        self.splits = tuple(sorted(int(s) for s in splits))
        self.end = int(end)
        self.mode = "off"
        self.prompt = -1
        self.skip = True
        self.chunks: list[tuple[int, int]] = []
        self.check: dict[int, Any] = {}
        self.check_rows = 0
        self.seconds = 0.0
        layers = range(self.splits[0], self.end)
        lead = rank == 0
        self.acc = (
            StatsAccumulator(self.splits, self.end, device_sums=device_sums)
            if lead
            else None
        )
        self.held = (
            RowStore(self.splits, layers, stride=raw_stride, head=raw_head)
            if lead
            else None
        )
        self.head = (
            RowStore(self.splits, layers, stride=0, head=stats_head)
            if lead and stats_head > 0
            else None
        )

    def store(self) -> RowStore | None:
        if self.mode == "raw":
            return self.held
        if self.mode == "stats":
            return self.head
        return None

    def on_split(self, split: int, x: Any, positions: Any) -> None:
        if self.mode == "off" or x is None:
            return
        start = time.perf_counter()
        if split == self.splits[0]:
            rows = int(x.shape[0])
            # A one-row forward is a decode step: none of its hooks count.
            self.skip = rows <= 1
            if self.skip:
                return
            pos = _row_positions(positions)
            first = -1 if pos is None else int(pos[0])
            self.chunks.append((first, rows))
            store = self.store()
            if store is not None:
                store.begin(pos, self.prompt)
        if self.skip:
            return
        if self.mode == "stats":
            self._checksum(split, x)
            if self.acc is not None:
                self.acc.add_split(split, x)
        store = self.store()
        if store is not None:
            store.add_split(split, x)
        self.seconds += time.perf_counter() - start

    def on_target(self, layer: int, y: Any) -> None:
        if self.mode == "off" or self.skip or y is None:
            return
        start = time.perf_counter()
        if self.mode == "stats" and self.acc is not None:
            self.acc.add_target(layer, y)
        store = self.store()
        if store is not None:
            store.add_target(layer, y)
        self.seconds += time.perf_counter() - start

    def _checksum(self, split: int, x: Any) -> None:
        import torch

        part = x.detach().float().sum(dim=0).cpu()
        acc = self.check.get(split)
        if acc is None:
            acc = torch.zeros(int(part.shape[0]), dtype=torch.float64)
            self.check[split] = acc
        acc.add_(part)
        if split == self.splits[0]:
            self.check_rows += int(x.shape[0])


class RidgeFillProbe:
    """Extension with the capture RPC methods; it overrides no worker method."""

    def _ridgefill_layers(self):
        # The runner holds Qwen4ExpForConditionalGeneration, whose text stack
        # is ``language_model.model.layers``; a text-only runner holds the
        # causal LM, whose stack is ``model.layers``.
        model = self.model_runner.model  # type: ignore[attr-defined]
        inner = getattr(model, "language_model", model)
        inner = getattr(inner, "model", inner)
        return inner.layers

    def ridgefill_capture(self, split: int, end: int) -> None:
        """Hook the split layer and the late attention modules."""
        layers = self._ridgefill_layers()
        if len(layers) < end:
            raise RuntimeError(f"model has {len(layers)} layers, capture wants {end}")
        self._ridgefill_split: list = []
        self._ridgefill_inputs: dict[int, list] = {
            layer: [] for layer in range(split, end)
        }

        def split_hook(module, args, kwargs):
            hidden = split_state(module, args, kwargs)
            if hidden is not None:
                self._ridgefill_split.append(hidden.detach().float().cpu())

        def input_hook(layer: int):
            def hook(module, args, kwargs):
                hidden = _block_input(args, kwargs)
                if hidden is not None:
                    self._ridgefill_inputs[layer].append(hidden.detach().float().cpu())

            return hook

        for layer in range(split, end):
            module = layers[layer]
            if layer == split:
                module.register_forward_pre_hook(split_hook, with_kwargs=True)
            _attention(module, layer).register_forward_pre_hook(
                input_hook(layer), with_kwargs=True
            )

    def ridgefill_save(self, path: str) -> None:
        """Write this rank's captures; the driver reads rank 0."""
        import torch

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        rank = int(self.rank)  # type: ignore[attr-defined]
        torch.save(
            {
                "split": self._ridgefill_split,
                "inputs": self._ridgefill_inputs,
            },
            out / f"rank{rank}.pt",
        )

    def ridgefill_stats_begin(
        self,
        splits: list[int],
        end: int,
        raw_stride: int = 4,
        raw_head: int = 64,
        stats_head: int = 64,
        fold_threads: int = 0,
        device_sums: bool = True,
    ) -> dict[str, Any]:
        """Install the statistics hooks (mode ``off``); returns this rank's facts."""
        if getattr(self, "_ridgefill_stats", None) is not None:
            raise RuntimeError("ridgefill_stats_begin already ran in this worker")
        layers = self._ridgefill_layers()
        if len(layers) < end:
            raise RuntimeError(f"model has {len(layers)} layers, capture wants {end}")
        rank = int(self.rank)  # type: ignore[attr-defined]
        run = _StatsRun(
            rank,
            splits,
            end,
            raw_stride=raw_stride,
            raw_head=raw_head,
            stats_head=stats_head,
            device_sums=device_sums,
        )
        for split in run.splits:
            if getattr(layers[split], "use_hc_sequence_parallel", False):
                # Each rank would hold a slice of the rows, not all of them.
                raise RuntimeError(
                    f"layer {split} shards its rows (HC sequence parallel); "
                    "rank 0 would not hold the whole split state"
                )
        if rank == 0 and fold_threads > 0:
            import torch

            torch.set_num_threads(int(fold_threads))

        def split_hook(split: int):
            def hook(module, args, kwargs):
                run.on_split(
                    split, split_state(module, args, kwargs), kwargs.get("positions")
                )

            return hook

        def target_hook(layer: int):
            def hook(module, args, kwargs):
                run.on_target(layer, _block_input(args, kwargs))

            return hook

        for split in run.splits:
            layers[split].register_forward_pre_hook(split_hook(split), with_kwargs=True)
        for layer in range(run.splits[0], end):
            _attention(layers[layer], layer).register_forward_pre_hook(
                target_hook(layer), with_kwargs=True
            )
        self._ridgefill_stats = run
        return {"rank": rank, "layers": len(layers), "splits": list(run.splits)}

    def ridgefill_stats_mode(self, mode: str, prompt: int) -> dict[str, Any]:
        """Set the mode for the next prompt; returns the previous prompt's chunks.

        Rank 0 first folds the previous prompt's device sums into fp64.
        """
        import torch

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        run = self._ridgefill_stats
        fold = 0.0
        if run.acc is not None:
            start = time.perf_counter()
            # The hooks create the sums during the worker's forward, inside
            # its inference mode, so they are inference tensors: torch only
            # updates those in place inside inference mode.
            with torch.inference_mode():
                run.acc.flush()
            fold = time.perf_counter() - start
            run.seconds += fold
        done = {
            "rank": run.rank,
            "mode": run.mode,
            "prompt": run.prompt,
            "chunks": list(run.chunks),
            "fold_seconds": fold,
        }
        run.mode = mode
        run.prompt = int(prompt)
        run.chunks = []
        run.skip = True
        return done

    def ridgefill_stats_snapshot(self, path: str) -> dict[str, Any]:
        """Rank 0 writes the cumulative fp64 sums; every rank returns its check."""
        run = self._ridgefill_stats
        out: dict[str, Any] = {
            "rank": run.rank,
            "rows": run.check_rows,
            "check": {str(s): v.tolist() for s, v in run.check.items()},
            "seconds": run.seconds,
        }
        if run.acc is not None:
            import torch

            # ``payload`` flushes the inference-tensor sums first (see above).
            with torch.inference_mode():
                out["bytes"] = _save_atomic(run.acc.payload(), path)
            out["path"] = path
            out["forwards"] = run.acc.forwards
            out["folds"] = run.acc.folds
            out["device_sums"] = run.acc.device_sums
        return out

    def ridgefill_stats_save_rows(self, kind: str, path: str) -> dict[str, Any]:
        """Rank 0 writes the ``held`` (raw mode) or ``head`` (stats mode) rows."""
        if kind not in ("held", "head"):
            raise ValueError(f"kind must be held or head, got {kind!r}")
        run = self._ridgefill_stats
        store = run.held if kind == "held" else run.head
        out: dict[str, Any] = {"rank": run.rank, "kind": kind}
        if store is not None:
            out["rows"] = store.count()
            out["bytes"] = _save_atomic(store.payload(), path)
            out["path"] = path
        return out
