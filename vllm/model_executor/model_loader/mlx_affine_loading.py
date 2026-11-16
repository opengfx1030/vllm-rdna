# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-scoped source admission for immutable native MLX affine views.

The ordinary callback is the model's existing logical loader. It receives one
CPU permutation view at a time and must return the actual loaded names. The
public default/RunAI loader remains responsible for destination validation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from vllm.model_executor.layers.quantization.mlx_affine_format import (
    MlxAffineCoverageReport,
    MlxAffineDestination,
    MlxAffineInterval,
    MlxAffineLoadCoverage,
    MlxAffineSourcePlan,
)


@dataclass(frozen=True)
class MlxAffineLoadBinding:
    original_name: str
    plan: MlxAffineSourcePlan
    destination_sizes: Mapping[str, int]
    load: Callable[[Any], None]
    loaded_parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.original_name, str) or not self.original_name:
            raise ValueError("binding requires an original tensor name")
        if not isinstance(self.plan, MlxAffineSourcePlan) or not callable(self.load):
            raise ValueError("binding requires an explicit plan and synchronous copy")
        sizes = dict(self.destination_sizes)
        if any(
            not isinstance(n, str) or not n or type(s) is not int or s <= 0
            for n, s in sizes.items()
        ):
            raise ValueError("invalid binding destination sizes")
        names = tuple(self.loaded_parameters)
        if len(set(names)) != len(names) or any(
            not isinstance(n, str) or not n for n in names
        ):
            raise ValueError("invalid actual loaded parameter names")
        if bool(self.plan.local) != bool(names):
            raise ValueError("only local bindings may declare actual loaded parameters")
        object.__setattr__(self, "destination_sizes", MappingProxyType(sizes))
        object.__setattr__(self, "loaded_parameters", names)


@dataclass(frozen=True)
class MlxAffineLoadReceipt:
    manifest_sha256: str
    view: str
    source_revision: str
    module_specs_sha256: str
    coverage: MlxAffineCoverageReport
    compact_components: int
    ordinary_components: int
    source_bytes: int
    copy_bytes: int
    copy_tiles: int
    max_copy_tile_bytes: int
    gpu_copy_scratch_bytes: int = 0
    shared_aliases: tuple[tuple[str, str], ...] = ()


@dataclass
class _CompletedLoad:
    coverage: MlxAffineLoadCoverage
    receipt: MlxAffineLoadReceipt
    aliases: Mapping[str, str]


_active_load: ContextVar[Any | None] = ContextVar("mlx_affine_load", default=None)


@dataclass
class _ActiveLoad:
    named_modules_values: tuple[Any, ...]
    copy_bytes: int = 0
    copy_tiles: int = 0
    max_copy_tile_bytes: int = 0


@dataclass(frozen=True)
class MlxAffineCopyStats:
    copy_bytes: int
    copy_tiles: int
    max_copy_tile_bytes: int


def is_mlx_affine_loading() -> bool:
    return _active_load.get() is not None


def copy_mlx_affine_tensor(destination, source) -> MlxAffineCopyStats:
    """Copy CPU views to final storage, casting only <=1MiB CPU tiles.

    Equal-dtype tiles transfer directly into their final destination; no
    full-source device allocation or device-side dtype conversion is used.
    """
    if source.device.type != "cpu":
        raise ValueError("bounded affine loading requires a CPU source view")
    if destination.is_meta:
        raise ValueError(
            "native affine loading requires materialized destination storage"
        )
    destination = destination.detach()
    source = source.detach()
    if str(source.dtype) == "torch.uint32":
        if str(destination.dtype) != "torch.int32":
            raise ValueError("packed U32 codes require canonical int32 bit storage")
        source = source.view(destination.dtype)
    if source.shape != destination.shape:
        if source.numel() == destination.numel() == 1:
            source = source.reshape(destination.shape)
        else:
            raise ValueError("bounded affine copy source/destination shapes differ")
    byte_count = 0
    tiles = 0
    maximum = 0

    def copy_tile(dst, src):
        nonlocal byte_count, tiles, maximum
        if (
            src.numel() * max(src.element_size(), dst.element_size()) > 1024 * 1024
            or not dst.is_contiguous()
        ):
            # Preserve contiguous storage spans even for permuted destinations.
            axis = max(
                (i for i, size in enumerate(src.shape) if size > 1),
                key=lambda i: dst.stride(i),
            )
            middle = src.shape[axis] // 2
            for start, size in ((0, middle), (middle, src.shape[axis] - middle)):
                copy_tile(dst.narrow(axis, start, size), src.narrow(axis, start, size))
            return
        tile = src.to(dtype=dst.dtype).contiguous()
        dst.copy_(tile, non_blocking=False)
        size = tile.numel() * tile.element_size()
        byte_count += size
        tiles += 1
        maximum = max(maximum, size)

    copy_tile(destination, source)
    active = _active_load.get()
    if active is not None:
        active.copy_bytes += byte_count
        active.copy_tiles += tiles
        active.max_copy_tile_bytes = max(active.max_copy_tile_bytes, maximum)
    return MlxAffineCopyStats(byte_count, tiles, maximum)


def _qualify(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _config(model):
    configs = []
    for _, module in model.named_modules():
        config = getattr(module, "quant_config", None)
        if config is not None and config.get_name() == "mlx_affine":
            configs.append(config)
    if not configs or any(getattr(c, "checkpoint", None) is None for c in configs):
        raise ValueError(
            "native affine loading requires a construction-pinned checkpoint"
        )
    first = configs[0]
    identity = (
        first.checkpoint.root,
        first.checkpoint.manifest_sha256,
        first.specs.sha256,
    )
    if any(
        (c.checkpoint.root, c.checkpoint.manifest_sha256, c.specs.sha256) != identity
        for c in configs
    ):
        raise ValueError("one affine load cannot mix checkpoint views/specifications")
    return first


def _bindings(model):
    result = {}
    sizes = {}
    actual = {}
    for prefix, module in model.named_modules():
        for binding in getattr(module, "_mlx_affine_load_bindings", ()):
            if not isinstance(binding, MlxAffineLoadBinding):
                raise ValueError("invalid native affine module binding")
            name = binding.original_name
            if name in result:
                raise ValueError(f"duplicate affine source binding: {name}")
            destinations = tuple(
                MlxAffineDestination(_qualify(prefix, d.name), d.interval)
                for d in binding.plan.destinations
            )
            plan = MlxAffineSourcePlan(
                binding.plan.local,
                binding.plan.nonlocal_,
                destinations,
                binding.plan.omission_reason,
            )
            for local_name, size in binding.destination_sizes.items():
                full_name = _qualify(prefix, local_name)
                if full_name in sizes and sizes[full_name] != size:
                    raise ValueError(
                        f"conflicting affine destination extent: {full_name}"
                    )
                sizes[full_name] = size
            result[name] = (binding, plan)
            actual[name] = tuple(_qualify(prefix, n) for n in binding.loaded_parameters)
    return result, sizes, actual


def _ordinary_destinations(model, checkpoint, compact_names, available, specs):
    declared = getattr(model, "_mlx_affine_unquantized_destinations", {})
    if not isinstance(declared, Mapping) or set(declared) - checkpoint.tensor_names:
        raise ValueError("ordinary destination hook contains undeclared sources")
    if set(declared) & compact_names:
        raise ValueError("compact sources cannot have ordinary destination mappings")
    result = {}
    for name, entry in checkpoint.manifest["tensors"].items():
        if name in compact_names:
            continue
        prefix, _, suffix = name.rpartition(".")
        if (
            suffix in ("weight", "scales", "biases")
            and specs.resolve_checkpoint_quantization(prefix) is not None
        ):
            raise ValueError(f"affine source has no compact load binding: {name}")
        names = declared.get(name, (entry["runtime_name"],))
        if isinstance(names, str) or not isinstance(names, (tuple, list)) or not names:
            raise ValueError(
                f"ordinary source requires explicit destination names: {name}"
            )
        if len(set(names)) != len(names) or any(n not in available for n in names):
            raise ValueError(
                f"declare actual ordinary loader destinations for {name}: {names}"
            )
        result[name] = tuple(names)
    return result


def _dtype_name(tensor) -> str:
    # Safetensors spellings, independent of the source streamer's implementation.
    names = {
        "torch.bool": "BOOL",
        "torch.uint8": "U8",
        "torch.int8": "I8",
        "torch.int16": "I16",
        "torch.uint16": "U16",
        "torch.float16": "F16",
        "torch.bfloat16": "BF16",
        "torch.int32": "I32",
        "torch.uint32": "U32",
        "torch.float32": "F32",
        "torch.int64": "I64",
        "torch.uint64": "U64",
        "torch.float64": "F64",
    }
    try:
        return names[str(tensor.dtype)]
    except KeyError as error:
        raise ValueError(f"unsupported affine source dtype: {tensor.dtype}") from error


def _actual_loaded(value) -> set[str]:
    if not isinstance(value, set) or any(not isinstance(n, str) for n in value):
        raise ValueError(
            "native affine logical loader must return its actual loaded set"
        )
    return value


def load_mlx_affine_weights(
    model,
    weights: Iterable[tuple[str, Any]],
    load_unquantized: Callable[[Iterable[tuple[str, Any]]], set[str]],
) -> set[str]:
    """Admit originals once, copy owned compact slices and run logical loaders.

    Model hooks are module-local ``_mlx_affine_load_bindings``; root-relative
    ``_mlx_affine_unquantized_destinations`` maps original ordinary source names
    to actual parameter/buffer names when the manifest name is not sufficient.
    Draft ``_mlx_affine_shared_aliases`` maps draft module paths to target module
    paths. Aliases are deliberately finalized only after proposer sharing.
    Nested public wrappers delegate to the active session without re-admission.
    """
    active = _active_load.get()
    if active is not None:
        if not any(module is model for module in active.named_modules_values):
            raise ValueError("unrelated native affine loads cannot nest")
        return _actual_loaded(load_unquantized(weights))

    # A failed replacement load must never leave an earlier admission usable.
    for attribute in ("_mlx_affine_completed_load", "_mlx_affine_load_receipt"):
        if hasattr(model, attribute):
            delattr(model, attribute)

    config = _config(model)
    checkpoint = config.checkpoint
    bindings, sizes, compact_actual = _bindings(model)
    if set(bindings) - checkpoint.tensor_names:
        raise ValueError("native affine binding is outside selected checkpoint view")
    available = dict(model.named_parameters(remove_duplicate=False))
    available.update(dict(model.named_buffers(remove_duplicate=False)))
    for names in compact_actual.values():
        if any(name not in available for name in names):
            raise ValueError(f"binding names nonexistent actual destinations: {names}")
    ordinary = _ordinary_destinations(
        model, checkpoint, set(bindings), available, config.specs
    )
    for destinations in (compact_actual, ordinary):
        if any(
            available[name].is_meta for names in destinations.values() for name in names
        ):
            raise ValueError(
                "native affine loading requires materialized destination storage"
            )
    plans = {name: pair[1] for name, pair in bindings.items()}
    for name in ordinary:
        rows = checkpoint.manifest["tensors"][name]["shape"]
        rows = rows[0] if rows else 1
        # One source-component completion token, not flattened tensor geometry.
        destination = f"__ordinary_source__.{name}"
        sizes[destination] = 1
        plans[name] = MlxAffineSourcePlan(
            (MlxAffineInterval(0, rows),),
            (),
            (MlxAffineDestination(destination, MlxAffineInterval(0, 1)),),
        )
    aliases = dict(getattr(model, "_mlx_affine_shared_aliases", {}))
    if any(
        not isinstance(path, str) or not path
        for pair in aliases.items()
        for path in pair
    ):
        raise ValueError("shared aliases require explicit draft/target module paths")
    if aliases and checkpoint.manifest["view"] != "mtp":
        raise ValueError("shared target owners must be declared on the draft view")
    qualified_aliases = {f"draft.{a}": f"target.{o}" for a, o in aliases.items()}
    coverage = MlxAffineLoadCoverage(
        checkpoint.manifest,
        config.specs,
        source_plans=plans,
        destination_sizes=sizes,
        shared_aliases=qualified_aliases,
    )

    active = _ActiveLoad(tuple(m for _, m in model.named_modules()))
    token = _active_load.set(active)
    loaded = set()
    source_bytes = 0
    compact_count = 0
    ordinary_count = 0
    try:
        for name, tensor in weights:
            if tensor.device.type != "cpu":
                raise ValueError("native affine requires protected CPU staging")
            if name not in plans:
                raise ValueError(f"undeclared source tensor: {name}")
            plan = plans[name]
            coverage.record(
                name,
                shape=tuple(tensor.shape),
                dtype=_dtype_name(tensor),
                nonlocal_=plan.nonlocal_,
            )
            source_bytes += tensor.numel() * tensor.element_size()
            if name in bindings:
                binding, _ = bindings[name]
                binding.load(tensor)
                if plan.destinations:
                    coverage.record_destinations(name, plan.destinations)
                loaded.update(compact_actual[name])
                compact_count += 1
            else:
                descriptor = coverage.tensors[name]
                transformed = tensor.permute(descriptor.layout.permutation)
                actual = _actual_loaded(
                    load_unquantized(iter(((descriptor.runtime_name, transformed),)))
                )
                missing = set(ordinary[name]) - actual
                if missing:
                    raise ValueError(
                        f"ordinary source {name} did not initialize "
                        f"destinations: {missing}"
                    )
                loaded.update(actual)
                coverage.record_destinations(name, plan.destinations)
                ordinary_count += 1
                del transformed
            del tensor
        report = coverage.finish(defer_aliases=bool(aliases))
        receipt = MlxAffineLoadReceipt(
            checkpoint.manifest_sha256,
            checkpoint.manifest["view"],
            checkpoint.manifest["source_revision"],
            checkpoint.manifest["module_specs_sha256"],
            report,
            compact_count,
            ordinary_count,
            source_bytes,
            active.copy_bytes,
            active.copy_tiles,
            active.max_copy_tile_bytes,
            shared_aliases=tuple(sorted(aliases.items())),
        )
        model._mlx_affine_load_receipt = receipt
        model._mlx_affine_completed_load = _CompletedLoad(coverage, receipt, aliases)
        return loaded
    finally:
        _active_load.reset(token)


def finish_mlx_affine_sharing(target_model, draft_model) -> MlxAffineLoadReceipt:
    """Finalize actual whole-module aliases after either proposer share path."""
    from dataclasses import replace

    completed = getattr(draft_model, "_mlx_affine_completed_load", None)
    target = getattr(target_model, "_mlx_affine_completed_load", None)
    if completed is None or target is None:
        raise ValueError("both native target and draft must complete strict loading")
    if target.receipt.view != "target" or completed.receipt.view != "mtp":
        raise ValueError("sharing requires distinct target and MTP checkpoint views")
    current_aliases = dict(getattr(draft_model, "_mlx_affine_shared_aliases", {}))
    if (
        set(current_aliases) != set(completed.aliases)
        or len(current_aliases) != 2
        or len(set(current_aliases.values())) != 2
    ):
        raise ValueError(
            "native draft requires the declared two distinct embedding/head owners"
        )
    if (
        target.receipt.source_revision != completed.receipt.source_revision
        or target.receipt.module_specs_sha256 != completed.receipt.module_specs_sha256
    ):
        raise ValueError(
            "target/draft sharing cannot mix source or precision identities"
        )
    modules = {}
    for alias, owner in completed.aliases.items():
        try:
            modules[f"draft.{alias}"] = draft_model.get_submodule(alias)
            modules[f"target.{owner}"] = target_model.get_submodule(
                current_aliases[alias]
            )
        except AttributeError as error:
            raise ValueError(
                "shared aliases are not installed as whole modules"
            ) from error
    report = completed.coverage.finish(shared_modules=modules)
    receipt = replace(
        completed.receipt,
        coverage=report,
        shared_aliases=tuple(sorted(current_aliases.items())),
    )
    completed.receipt = receipt
    draft_model._mlx_affine_load_receipt = receipt
    return receipt
