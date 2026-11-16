# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checkpoint descriptors and load admission for native MLX affine.

No tensor framework is imported here. Coordinates are logical, half-open ranges;
packed-word coordinates are never substituted for logical input columns.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _name(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ValueError("shape must be a sequence of positive dimensions")
    return tuple(_integer(n, "shape dimension", 1) for n in shape)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_mlx_affine_json(text: str) -> dict[str, Any]:
    """Decode a config/manifest without silently losing duplicate components."""
    result = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


@dataclass(frozen=True)
class MlxAffineSpec:
    bits: int
    group_size: int
    mode: str
    original_prefix: str

    def __post_init__(self) -> None:
        _integer(self.bits, "bits", 1)
        _integer(self.group_size, "group_size", 1)
        if self.bits not in (6, 8) or self.group_size not in (32, 64, 128):
            raise ValueError("MLX affine supports bits 6/8 and groups 32/64/128")
        if self.mode != "affine":
            raise ValueError(f"unsupported quantization mode: {self.mode!r}")
        _name(self.original_prefix, "original prefix")


@dataclass(frozen=True)
class MlxAffineSpecMap:
    module_specs: Mapping[str, MlxAffineSpec]

    def __post_init__(self) -> None:
        copied = dict(self.module_specs)
        for prefix, spec in copied.items():
            if not isinstance(spec, MlxAffineSpec) or prefix != spec.original_prefix:
                raise ValueError(f"specification identity mismatch: {prefix}")
        object.__setattr__(self, "module_specs", MappingProxyType(copied))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MlxAffineSpecMap":
        if set(config) != {"quant_method", "format_version", "module_specs"}:
            raise ValueError("expected normalized mlx_affine configuration")
        if config["quant_method"] != "mlx_affine":
            raise ValueError("quant_method must be mlx_affine")
        if type(config["format_version"]) is not int or config["format_version"] != 1:
            raise ValueError("unsupported affine format version")
        raw = config["module_specs"]
        if not isinstance(raw, Mapping):
            raise ValueError("module_specs must be an explicit map")
        specs = {}
        for prefix, entry in raw.items():
            if not isinstance(entry, Mapping) or set(entry) != {
                "bits",
                "group_size",
                "mode",
            }:
                raise ValueError(f"invalid module specification: {prefix}")
            specs[prefix] = MlxAffineSpec(original_prefix=prefix, **entry)
        return cls(specs)

    def resolve_checkpoint_quantization(
        self, original_prefix: str
    ) -> MlxAffineSpec | None:
        """Exact original identity only; no prefix/default/fused inference."""
        _name(original_prefix, "original prefix")
        return self.module_specs.get(original_prefix)

    @property
    def sha256(self) -> str:
        raw = {
            prefix: {"bits": s.bits, "group_size": s.group_size, "mode": s.mode}
            for prefix, s in self.module_specs.items()
        }
        encoded = json.dumps(
            raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MlxAffineShard:
    logical_k_start: int
    logical_k_size: int
    copied_word_start: int
    copied_word_count: int
    code_bit_offset: int
    metadata_group_start: int
    metadata_group_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), name)
        if not self.logical_k_size or not self.copied_word_count:
            raise ValueError("empty affine shards must be explicit nonlocal omissions")
        if not self.metadata_group_count or self.code_bit_offset >= 32:
            raise ValueError("invalid affine shard offsets")

    @classmethod
    def from_interval(
        cls, spec: MlxAffineSpec, k0: int, k1: int, *, logical_k: int
    ) -> "MlxAffineShard":
        _integer(k0, "k0")
        _integer(k1, "k1", 1)
        _integer(logical_k, "logical_k", 1)
        if not k0 < k1 <= logical_k:
            raise ValueError("input interval is empty or outside logical K")
        word_start, bit_offset = divmod(k0 * spec.bits, 32)
        return cls(
            k0,
            k1 - k0,
            word_start,
            (k1 * spec.bits + 31) // 32 - word_start,
            bit_offset,
            k0 // spec.group_size,
            (k1 + spec.group_size - 1) // spec.group_size - k0 // spec.group_size,
        )

    def validate(self, spec: MlxAffineSpec) -> None:
        end = self.logical_k_start + self.logical_k_size
        expected = self.from_interval(spec, self.logical_k_start, end, logical_k=end)
        if self != expected:
            raise ValueError("shard does not cover its declared logical interval")

    def metadata_index(self, local_k: int, spec: MlxAffineSpec) -> int:
        self.validate(spec)
        _integer(local_k, "local_k")
        if local_k >= self.logical_k_size:
            raise ValueError("local input index outside shard")
        return (self.logical_k_start + local_k) // spec.group_size - (
            self.metadata_group_start
        )


def extract_mlx_affine_code(
    words: Sequence[int], local_k: int, spec: MlxAffineSpec, shard: MlxAffineShard
) -> int:
    """Reference extraction from copied words, accepting U32 or bit-view I32.

    This scalar CPU utility is for validation, not request-time execution.
    Bounds-check the crossing word rather than reading a padded word implicitly.
    """
    shard.validate(spec)
    _integer(local_k, "local_k")
    if local_k >= shard.logical_k_size or len(words) != shard.copied_word_count:
        raise ValueError("input index or copied-word extent does not match shard")
    index, shift = divmod(shard.code_bit_offset + local_k * spec.bits, 32)

    def word(index: int) -> int:
        if index >= len(words):
            raise ValueError("truncated packed word stream")
        value = words[index]
        if type(value) is not int or not -(1 << 31) <= value < (1 << 32):
            raise ValueError("packed word must be a U32 or an I32 bit view")
        return value & 0xFFFFFFFF

    value = word(index) >> shift
    if shift + spec.bits > 32:
        value |= word(index + 1) << (32 - shift)
    return value & ((1 << spec.bits) - 1)


@dataclass(frozen=True)
class MlxAffineIdentity:
    original_name: str
    runtime_name: str
    construction_prefix: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _name(getattr(self, field), field)

    @classmethod
    def mtp(cls, original_name: str, runtime_name: str, *, backbone_layers: int):
        _integer(backbone_layers, "backbone_layers", 1)
        original = "mtp.layers.0."
        runtime = "model.layers.0."
        if not original_name.startswith(original) or runtime_name != (
            runtime + original_name[len(original) :]
        ):
            raise ValueError("MTP decoder identity must retain registered layer 0")
        module = original_name.rsplit(".", 1)[0]
        construction = f"mtp.layers.{backbone_layers}." + module[len(original) :]
        return cls(original_name, runtime_name, construction)


@dataclass(frozen=True)
class MlxAffineSegment:
    source: MlxAffineIdentity
    spec: MlxAffineSpec
    shard: MlxAffineShard
    output_offset: int
    output_size: int

    def __post_init__(self) -> None:
        _integer(self.output_offset, "output_offset")
        _integer(self.output_size, "output_size", 1)
        if self.source.original_name != self.spec.original_prefix + ".weight":
            raise ValueError("segment source does not match original specification")
        self.shard.validate(self.spec)


@dataclass(frozen=True)
class MlxAffineLayout:
    layout: str
    source_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_shape", _shape(self.source_shape))
        shapes = {
            "gdn_conv_mlx": (10240, 4, 1),
            "vision_patch_mlx": (1152, 2, 16, 16, 3),
        }
        if self.layout != "identity" and (
            self.layout not in shapes or self.source_shape != shapes[self.layout]
        ):
            raise ValueError(f"invalid declared {self.layout} layout/shape")

    @property
    def permutation(self) -> tuple[int, ...]:
        if self.layout == "gdn_conv_mlx":
            return (0, 2, 1)
        if self.layout == "vision_patch_mlx":
            return (0, 4, 1, 2, 3)
        return tuple(range(len(self.source_shape)))

    @property
    def runtime_shape(self) -> tuple[int, ...]:
        return tuple(self.source_shape[i] for i in self.permutation)

    def gdn_channel_intervals(self, rank: int, tp_size: int = 8):
        """Q/K/V logical channel slices, concatenated in this order."""
        _integer(rank, "rank")
        _integer(tp_size, "tp_size", 1)
        if self.layout != "gdn_conv_mlx" or rank >= tp_size:
            raise ValueError("invalid GDN channel partition")
        if 2048 % tp_size or 6144 % tp_size:
            raise ValueError("TP must divide GDN Q/K/V channel counts")
        q, v = 2048 // tp_size, 6144 // tp_size
        return (
            (rank * q, (rank + 1) * q),
            (2048 + rank * q, 2048 + (rank + 1) * q),
            (4096 + rank * v, 4096 + (rank + 1) * v),
        )


@dataclass(frozen=True)
class MlxAffineInterval:
    start: int
    stop: int

    def __post_init__(self) -> None:
        _integer(self.start, "interval start")
        _integer(self.stop, "interval stop", 1)
        if self.start >= self.stop:
            raise ValueError("empty or reversed interval")


def _cover(intervals: Sequence[MlxAffineInterval], size: int, name: str) -> None:
    cursor = 0
    for interval in sorted(intervals, key=lambda item: item.start):
        if interval.start != cursor:
            raise ValueError(f"{name}: coverage hole or overlap at {cursor}")
        cursor = interval.stop
    if cursor != size:
        raise ValueError(f"{name}: incomplete or out-of-bounds coverage")


@dataclass(frozen=True)
class MlxAffineDestination:
    name: str
    interval: MlxAffineInterval

    def __post_init__(self) -> None:
        _name(self.name, "destination name")
        if not isinstance(self.interval, MlxAffineInterval):
            raise ValueError("destination requires a declared logical interval")


@dataclass(frozen=True)
class MlxAffineSourcePlan:
    """Partition source rows; destination units are explicitly caller-declared.

    Each destination names one native component/axis (e.g. expert codes, PLE
    scale rows, or a fused output segment), never an inferred flattened shape.
    The runtime's existing destination parameter validator remains authoritative.
    """

    local: tuple[MlxAffineInterval, ...]
    nonlocal_: tuple[MlxAffineInterval, ...]
    destinations: tuple[MlxAffineDestination, ...]
    omission_reason: str | None = None

    def __post_init__(self) -> None:
        for field in ("local", "nonlocal_", "destinations"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if any(
            not isinstance(i, MlxAffineInterval) for i in self.local + self.nonlocal_
        ):
            raise ValueError("source ownership requires logical intervals")
        if any(not isinstance(d, MlxAffineDestination) for d in self.destinations):
            raise ValueError("source plan requires declared destinations")
        if self.nonlocal_:
            _name(self.omission_reason, "nonlocal omission reason")
        elif self.omission_reason is not None:
            raise ValueError("omission reason without nonlocal rows")
        if bool(self.local) != bool(self.destinations):
            raise ValueError(
                "local rows require destinations; nonlocal rows forbid writes"
            )


@dataclass(frozen=True)
class MlxAffineTensor:
    original_name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    runtime_name: str
    layout: MlxAffineLayout


@dataclass(frozen=True)
class MlxAffineCoverageReport:
    source_components: int
    local_components: int
    nonlocal_components: int
    destinations: int
    pending_aliases: tuple[str, ...]


class MlxAffineLoadCoverage:
    """Load-scoped component admission; no tensor storage is retained.

    Construct once from the selected manifest and rank-local plans. Filter the
    streamer by ``tensor_names`` *before* allocating/copying tensors. Call record
    before transforming each source, then record_destinations after its copies.
    A wholly nonlocal tensor still needs record(..., nonlocal_=plan.nonlocal_).
    Call finish(defer_aliases=True) at draft load end, then finish(shared_modules=)
    after proposer sharing. Values in shared_modules must be the actual modules;
    alias identity is checked with ``is``, not matching shapes or storage sizes.
    Distinct declared owners (such as untied embedding/head) must remain distinct.
    """

    def __init__(
        self,
        manifest: Mapping[str, Any],
        specs: MlxAffineSpecMap,
        *,
        source_plans: Mapping[str, MlxAffineSourcePlan],
        destination_sizes: Mapping[str, int],
        shared_aliases: Mapping[str, str] | None = None,
        synthetic_padding: Sequence[MlxAffineDestination] = (),
    ) -> None:
        if set(manifest) != {
            "format_version",
            "view",
            "source_revision",
            "module_specs_sha256",
            "tensors",
        }:
            raise ValueError("invalid affine view manifest fields")
        if (
            type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
        ):
            raise ValueError("unsupported manifest version")
        if manifest["view"] not in ("target", "mtp"):
            raise ValueError("invalid affine manifest view")
        revision = manifest["source_revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        ):
            raise ValueError("source_revision must be an immutable commit hash")
        if manifest["module_specs_sha256"] != specs.sha256:
            raise ValueError("manifest quantization map digest mismatch")
        raw_tensors = manifest["tensors"]
        if not isinstance(raw_tensors, Mapping) or not raw_tensors:
            raise ValueError("manifest must select tensors")
        tensors = {}
        file_intervals: dict[str, list[MlxAffineInterval]] = {}
        runtime_names = set()
        dtype_bytes = {
            "BOOL": 1,
            "U8": 1,
            "I8": 1,
            "I16": 2,
            "U16": 2,
            "F16": 2,
            "BF16": 2,
            "I32": 4,
            "U32": 4,
            "F32": 4,
            "I64": 8,
            "U64": 8,
            "F64": 8,
        }
        for name, entry in raw_tensors.items():
            _name(name, "original tensor name")
            if (manifest["view"] == "mtp") != name.startswith("mtp."):
                raise ValueError(f"tensor is outside selected view: {name}")
            if not isinstance(entry, Mapping) or set(entry) != {
                "file",
                "dtype",
                "shape",
                "data_offsets",
                "runtime_name",
                "layout",
            }:
                raise ValueError(f"invalid manifest tensor: {name}")
            file = _name(entry["file"], "shard basename")
            if PurePosixPath(file).name != file or "\\" in file or file in (".", ".."):
                raise ValueError("manifest shard must be a basename")
            shape = _shape(entry["shape"])
            dtype = entry["dtype"]
            if dtype not in dtype_bytes:
                raise ValueError(f"unsupported manifest dtype: {dtype}")
            offsets = tuple(entry["data_offsets"])
            if len(offsets) != 2:
                raise ValueError("data_offsets must be a start/end pair")
            interval = MlxAffineInterval(*offsets)
            if interval.stop - interval.start != prod(shape) * dtype_bytes[dtype]:
                raise ValueError(f"manifest byte extent mismatch: {name}")
            file_intervals.setdefault(file, []).append(interval)
            runtime_name = _name(entry["runtime_name"], "runtime tensor name")
            if runtime_name in runtime_names:
                raise ValueError(
                    f"duplicate runtime destination identity: {runtime_name}"
                )
            runtime_names.add(runtime_name)
            layout = MlxAffineLayout(entry["layout"], shape)
            if layout.layout != "identity" and dtype != "BF16":
                raise ValueError("published nonmatrix layouts require BF16 sources")
            tensors[name] = MlxAffineTensor(
                name, file, dtype, shape, offsets, runtime_name, layout
            )
        for file, intervals in file_intervals.items():
            ordered = sorted(intervals, key=lambda i: i.start)
            if any(a.stop > b.start for a, b in zip(ordered, ordered[1:])):
                raise ValueError(f"overlapping source payloads in {file}")
        self._validate_affine_components(tensors, specs)
        if set(source_plans) != set(tensors):
            raise ValueError("source plans must exactly cover selected tensor names")
        self._plans = dict(source_plans)
        self._sizes = dict(destination_sizes)
        for name, size in self._sizes.items():
            _name(name, "destination name")
            _integer(size, "destination size", 1)
        self._padding = tuple(synthetic_padding)
        planned: dict[str, list[MlxAffineInterval]] = {n: [] for n in self._sizes}
        if any(not isinstance(p, MlxAffineSourcePlan) for p in self._plans.values()):
            raise ValueError("source plans must be explicit ownership descriptors")
        for name, plan in self._plans.items():
            rows = tensors[name].shape[0] if tensors[name].shape else 1
            _cover(plan.local + plan.nonlocal_, rows, name)
            prefix, _, suffix = name.rpartition(".")
            if suffix in ("scales", "biases"):
                weight_plan = self._plans[prefix + ".weight"]
                if (plan.local, plan.nonlocal_) != (
                    weight_plan.local,
                    weight_plan.nonlocal_,
                ):
                    raise ValueError("affine metadata ownership differs from codes")
            for destination in plan.destinations:
                self._add_destination(planned, destination)
        for padding in self._padding:
            self._add_destination(planned, padding)
        for name, intervals in planned.items():
            _cover(intervals, self._sizes[name], name)
        self._aliases = dict(shared_aliases or {})
        for alias, owner in self._aliases.items():
            _name(alias, "shared alias")
            _name(owner, "shared owner")
            if alias == owner or owner in self._aliases or alias in self._sizes:
                raise ValueError("aliases must name distinct direct shared owners")
        self.tensors = MappingProxyType(tensors)
        self.tensor_names = frozenset(tensors)
        self._seen: set[str] = set()
        self._destinations_seen: set[str] = set()
        self._written: dict[str, list[MlxAffineInterval]] = {n: [] for n in self._sizes}
        self._padding_seen: set[MlxAffineDestination] = set()
        self._sealed = False

    @staticmethod
    def _validate_affine_components(
        tensors: Mapping[str, MlxAffineTensor], specs: MlxAffineSpecMap
    ) -> None:
        for prefix, spec in specs.module_specs.items():
            names = tuple(
                prefix + suffix for suffix in (".weight", ".scales", ".biases")
            )
            present = tuple(name in tensors for name in names)
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f"missing affine weight/scales/biases: {prefix}")
            weight, scales, biases = (tensors[name] for name in names)
            if (
                weight.dtype != "U32"
                or scales.dtype != "BF16"
                or biases.dtype != "BF16"
            ):
                raise ValueError(f"invalid affine component dtype: {prefix}")
            if len(weight.shape) < 2 or scales.shape != biases.shape:
                raise ValueError(f"invalid affine metadata shapes: {prefix}")
            max_k = weight.shape[-1] * 32 // spec.bits
            groups = (max_k + spec.group_size - 1) // spec.group_size
            if scales.shape != weight.shape[:-1] + (groups,):
                raise ValueError(f"affine metadata group extent mismatch: {prefix}")
            if any(t.layout.layout != "identity" for t in (weight, scales, biases)):
                raise ValueError("packed affine components cannot use dense layouts")
        for name, tensor in tensors.items():
            prefix, _, suffix = name.rpartition(".")
            if (
                suffix in ("scales", "biases")
                or (suffix == "weight" and tensor.dtype == "U32")
            ) and specs.resolve_checkpoint_quantization(prefix) is None:
                raise ValueError(f"undeclared affine component: {name}")

    @staticmethod
    def _add_destination(
        target: dict[str, list[MlxAffineInterval]], destination: MlxAffineDestination
    ) -> None:
        if (
            not isinstance(destination, MlxAffineDestination)
            or destination.name not in target
        ):
            raise ValueError(f"undeclared destination: {destination}")
        target[destination.name].append(destination.interval)

    def record(
        self,
        original_name: str,
        *,
        shape: Sequence[int],
        dtype: str,
        destinations: Sequence[MlxAffineDestination] | None = None,
        nonlocal_: Sequence[MlxAffineInterval] = (),
    ) -> None:
        """Admit the original source before layout transforms or native slicing.

        Destinations may be supplied after synchronous copy, or admitted through
        record_destinations() after existing runtime destination validation.
        """
        if self._sealed:
            raise ValueError("load coverage is sealed")
        if original_name not in self.tensors:
            raise ValueError(f"undeclared source tensor: {original_name}")
        if original_name in self._seen:
            raise ValueError(f"duplicate source component: {original_name}")
        tensor, plan = self.tensors[original_name], self._plans[original_name]
        if _shape(shape) != tensor.shape or dtype != tensor.dtype:
            raise ValueError(f"source shape/dtype mismatch: {original_name}")
        if (destinations is not None and tuple(destinations) != plan.destinations) or (
            tuple(nonlocal_) != plan.nonlocal_
        ):
            raise ValueError(f"source ownership/destination mismatch: {original_name}")
        if destinations is not None:
            for destination in destinations:
                self._add_destination(self._written, destination)
            self._destinations_seen.add(original_name)
        elif not plan.destinations:
            self._destinations_seen.add(original_name)
        self._seen.add(original_name)

    def record_destinations(
        self, original_name: str, destinations: Sequence[MlxAffineDestination]
    ) -> None:
        """Admit a source's exact declared local writes after copy/validation."""
        if self._sealed or original_name not in self._seen:
            raise ValueError("source must be admitted before destination writes")
        if original_name in self._destinations_seen:
            raise ValueError(f"duplicate component destinations: {original_name}")
        if tuple(destinations) != self._plans[original_name].destinations:
            raise ValueError(f"component destination mismatch: {original_name}")
        for destination in destinations:
            self._add_destination(self._written, destination)
        self._destinations_seen.add(original_name)

    def record_padding(self, destination: MlxAffineDestination) -> None:
        """Admit a declared synthetic interval only after zero initialization."""
        if self._sealed or destination not in self._padding:
            raise ValueError("undeclared synthetic padding or sealed load")
        if destination in self._padding_seen:
            raise ValueError("duplicate synthetic padding")
        self._add_destination(self._written, destination)
        self._padding_seen.add(destination)

    def finish(
        self,
        *,
        shared_modules: Mapping[str, object] | None = None,
        defer_aliases: bool = False,
    ) -> MlxAffineCoverageReport:
        missing = self.tensor_names - self._seen
        if missing:
            raise ValueError(f"missing source components: {sorted(missing)}")
        if self.tensor_names != self._destinations_seen:
            raise ValueError("missing component destination writes")
        if set(self._padding) != self._padding_seen:
            raise ValueError("missing declared synthetic zero padding")
        for name, intervals in self._written.items():
            _cover(intervals, self._sizes[name], name)
        pending = []
        modules = shared_modules or {}
        for alias, owner in self._aliases.items():
            if alias not in modules or owner not in modules:
                pending.append(alias)
            elif modules[alias] is None or modules[alias] is not modules[owner]:
                raise ValueError(f"shared module identity mismatch: {alias} -> {owner}")
        owner_ids: dict[int, str] = {}
        for owner in set(self._aliases.values()):
            if owner not in modules:
                continue
            identity = id(modules[owner])
            if identity in owner_ids:
                raise ValueError(
                    f"distinct shared owners alias: {owner_ids[identity]}, {owner}"
                )
            owner_ids[identity] = owner
        if pending and not defer_aliases:
            raise ValueError(f"shared aliases not finalized: {pending}")
        self._sealed = True
        return MlxAffineCoverageReport(
            len(self._seen),
            sum(bool(p.local) for p in self._plans.values()),
            sum(bool(p.nonlocal_) for p in self._plans.values()),
            len(self._sizes),
            tuple(sorted(pending)),
        )
