# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure stdlib tests; direct execution avoids vLLM's tensor-importing conftest.

Run: .venv/bin/python -B tests/quantization/test_mlx_affine_format.py
The real format module is loaded directly, without mocking its dependencies.
"""

import hashlib
import importlib.util
import json
import sys
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from math import prod
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / (
    "vllm/model_executor/layers/quantization/mlx_affine_format.py"
)
_MODULE_NAME = "_vllm_mlx_affine_format_cpu"
_import_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PATH)
assert _import_spec is not None and _import_spec.loader is not None
fmt = importlib.util.module_from_spec(_import_spec)
sys.modules[_MODULE_NAME] = fmt
_import_spec.loader.exec_module(fmt)

Interval = fmt.MlxAffineInterval
D = fmt.MlxAffineDestination
P = fmt.MlxAffineSourcePlan


def _config(entries=None):
    return {
        "quant_method": "mlx_affine",
        "format_version": 1,
        "module_specs": entries
        if entries is not None
        else {"model.down": {"bits": 6, "group_size": 128, "mode": "affine"}},
    }


def _manifest(specs, records=None, view="target"):
    if records is None:
        records = {
            "model.down.weight": ("U32", (4, 120)),
            "model.down.scales": ("BF16", (4, 5)),
            "model.down.biases": ("BF16", (4, 5)),
        }
    tensors = {}
    offset = 0
    for name, (dtype, shape) in records.items():
        size = prod(shape) * {"U32": 4, "BF16": 2, "F16": 2, "I64": 8}[dtype]
        tensors[name] = {
            "file": "model-00001-of-00028.safetensors",
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
            "runtime_name": name,
            "layout": "identity",
        }
        offset += size
    return {
        "format_version": 1,
        "view": view,
        "source_revision": "e171af86f499f1855b0fb71d781105e8dd609610",
        "module_specs_sha256": specs.sha256,
        "tensors": tensors,
    }


def _plans(manifest):
    plans, sizes = {}, {}
    for name, entry in manifest["tensors"].items():
        rows = entry["shape"][0] if entry["shape"] else 1
        destination = D(entry["runtime_name"], Interval(0, rows))
        plans[name] = P((Interval(0, rows),), (), (destination,))
        sizes[destination.name] = rows
    return plans, sizes


def _tracker(manifest, specs, **kwargs):
    plans, sizes = _plans(manifest)
    return fmt.MlxAffineLoadCoverage(
        manifest,
        specs,
        source_plans=kwargs.pop("source_plans", plans),
        destination_sizes=kwargs.pop("destination_sizes", sizes),
        **kwargs,
    )


def _admit(tracker, manifest, plans=None, separate=False):
    if plans is None:
        plans, _ = _plans(manifest)
    for name, entry in manifest["tensors"].items():
        plan = plans[name]
        kwargs = {
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "nonlocal_": plan.nonlocal_,
        }
        if not separate:
            kwargs["destinations"] = plan.destinations
        tracker.record(name, **kwargs)
        if separate and plan.destinations:
            tracker.record_destinations(name, plan.destinations)


def _pack_independently(codes, bits):
    """Write individual little-endian bits into bytes, not extraction formulas."""
    raw = bytearray((len(codes) * bits + 31) // 32 * 4)
    cursor = 0
    for code in codes:
        for digit in reversed(f"{code:0{bits}b}"):
            if digit == "1":
                raw[cursor // 8] |= 1 << (cursor % 8)
            cursor += 1
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]


class TestSpecifications(unittest.TestCase):
    def test_exact_mixed_resolution_never_quantizes_unlisted_modules(self):
        raw = _config(
            {
                "original.gate": {"bits": 6, "group_size": 64, "mode": "affine"},
                "original.up": {"bits": 8, "group_size": 128, "mode": "affine"},
                "ple.shards.0": {"bits": 6, "group_size": 32, "mode": "affine"},
            }
        )
        specs = fmt.MlxAffineSpecMap.from_config(raw)
        self.assertEqual(specs.resolve_checkpoint_quantization("original.gate").bits, 6)
        self.assertEqual(specs.resolve_checkpoint_quantization("original.up").bits, 8)
        for name in ("original", "original.gate.weight", "visual.proj", "norm"):
            self.assertIsNone(specs.resolve_checkpoint_quantization(name))
        expected = hashlib.sha256(
            json.dumps(
                raw["module_specs"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(specs.sha256, expected)
        raw["module_specs"]["original.gate"]["bits"] = 8
        self.assertEqual(specs.resolve_checkpoint_quantization("original.gate").bits, 6)
        with self.assertRaises(TypeError):
            specs.module_specs["new"] = specs.module_specs["original.gate"]
        with self.assertRaises(FrozenInstanceError):
            specs.module_specs["original.gate"].bits = 8

    def test_unsupported_or_ambiguous_declarations_fail(self):
        for field, values in {
            "bits": (0, 4, True, "6"),
            "group_size": (0, 80, True),
            "mode": ("symmetric", None),
        }.items():
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    raw = _config()
                    raw["module_specs"]["model.down"][field] = value
                    fmt.MlxAffineSpecMap.from_config(raw)
        raw = _config()
        raw["group_size"] = 64
        with self.assertRaises(ValueError):
            fmt.MlxAffineSpecMap.from_config(raw)
        for text in ('{"x":1,"x":1}', '{"x":{"scales":1,"scales":2}}'):
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                fmt.load_mlx_affine_json(text)


class TestBitstream(unittest.TestCase):
    def test_all_codes_word_crossings_and_tails(self):
        for bits in (6, 8):
            spec = fmt.MlxAffineSpec(bits, 32, "affine", "packed")
            codes = list(range(1 << bits)) + [1, 0, (1 << bits) - 1]
            words = _pack_independently(codes, bits)
            for k0, k1 in (
                (0, len(codes)),
                (1, 33),
                (5, 21),
                (5, 22),
                (5, 37),
                (31, 65),
                (32, 64),
                (len(codes) - 2, len(codes)),
            ):
                with self.subTest(bits=bits, interval=(k0, k1)):
                    shard = fmt.MlxAffineShard.from_interval(
                        spec, k0, k1, logical_k=len(codes)
                    )
                    copied = words[
                        shard.copied_word_start : shard.copied_word_start
                        + shard.copied_word_count
                    ]
                    signed = [w if w < 2**31 else w - 2**32 for w in copied]
                    actual = [
                        fmt.extract_mlx_affine_code(copied, i, spec, shard)
                        for i in range(k1 - k0)
                    ]
                    self.assertEqual(actual, codes[k0:k1])
                    self.assertEqual(
                        [
                            fmt.extract_mlx_affine_code(signed, i, spec, shard)
                            for i in range(k1 - k0)
                        ],
                        actual,
                    )
                    self.assertEqual(
                        [shard.metadata_index(i, spec) for i in range(k1 - k0)],
                        [k // 32 - k0 // 32 for k in range(k0, k1)],
                    )

    def test_tp80_uses_global_k640_group128(self):
        spec = fmt.MlxAffineSpec(6, 128, "affine", "shared.down")
        codes = [(i * 17 + 11) % 64 for i in range(640)]
        words = _pack_independently(codes, 6)
        for rank in range(8):
            k0, k1 = rank * 80, (rank + 1) * 80
            shard = fmt.MlxAffineShard.from_interval(spec, k0, k1, logical_k=640)
            self.assertEqual(shard.copied_word_start, rank * 15)
            self.assertEqual(shard.copied_word_count, 15)
            self.assertEqual(shard.metadata_group_start, k0 // 128)
            copied = words[rank * 15 : (rank + 1) * 15]
            self.assertEqual(
                [
                    fmt.extract_mlx_affine_code(copied, i, spec, shard)
                    for i in range(80)
                ],
                codes[k0:k1],
            )
            self.assertEqual(
                [shard.metadata_index(i, spec) for i in range(80)],
                [(k0 + i) // 128 - k0 // 128 for i in range(80)],
            )
        rank1 = fmt.MlxAffineShard.from_interval(spec, 80, 160, logical_k=640)
        self.assertEqual(rank1.metadata_group_count, 2)
        self.assertEqual(rank1.metadata_index(47, spec), 0)
        self.assertEqual(rank1.metadata_index(48, spec), 1)
        rank4 = fmt.MlxAffineShard.from_interval(spec, 320, 400, logical_k=640)
        self.assertEqual(
            (rank4.metadata_group_start, rank4.metadata_group_count), (2, 2)
        )
        self.assertEqual(rank4.metadata_index(63, spec), 0)
        self.assertEqual(rank4.metadata_index(64, spec), 1)

    def test_misaligned_group_tail_does_not_require_divisible_local_k(self):
        spec = fmt.MlxAffineSpec(6, 64, "affine", "tail")
        shard = fmt.MlxAffineShard.from_interval(spec, 63, 131, logical_k=131)
        self.assertEqual((shard.copied_word_start, shard.code_bit_offset), (11, 26))
        self.assertEqual(
            (shard.metadata_group_start, shard.metadata_group_count), (0, 3)
        )
        words = _pack_independently([i % 64 for i in range(131)], 6)
        copied = words[11:25]
        self.assertEqual(shard.copied_word_count, 14)
        self.assertEqual(fmt.extract_mlx_affine_code(copied, 67, spec, shard), 2)
        self.assertEqual(shard.metadata_index(67, spec), 2)

    def test_bad_shards_extents_and_word_values_fail(self):
        spec = fmt.MlxAffineSpec(6, 64, "affine", "packed")
        shard = fmt.MlxAffineShard.from_interval(spec, 5, 17, logical_k=17)
        for bad in (
            replace(shard, copied_word_count=shard.copied_word_count + 1),
            replace(shard, metadata_group_start=1),
            replace(shard, code_bit_offset=0),
        ):
            with self.assertRaises(ValueError):
                bad.validate(spec)
        for k0, k1 in ((0, 0), (17, 18), (-1, 3), (True, 3)):
            with self.assertRaises(ValueError):
                fmt.MlxAffineShard.from_interval(spec, k0, k1, logical_k=17)
        words = [0] * shard.copied_word_count
        for bad_words, index in (
            (words[:-1], 0),
            (words, 12),
            ([2**32] * len(words), 0),
            ([-(2**31) - 1] * len(words), 0),
        ):
            with self.assertRaises(ValueError):
                fmt.extract_mlx_affine_code(bad_words, index, spec, shard)


class TestIdentitiesAndLayouts(unittest.TestCase):
    def test_mtp_parameter_layer_zero_has_separate_construction_layer48(self):
        identity = fmt.MlxAffineIdentity.mtp(
            "mtp.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            backbone_layers=48,
        )
        self.assertEqual(identity.construction_prefix, "mtp.layers.48.self_attn.q_proj")
        spec = fmt.MlxAffineSpec(6, 64, "affine", "mtp.layers.0.self_attn.q_proj")
        shard = fmt.MlxAffineShard.from_interval(spec, 0, 80, logical_k=640)
        segment = fmt.MlxAffineSegment(identity, spec, shard, 128, 64)
        self.assertEqual((segment.output_offset, segment.output_size), (128, 64))
        with self.assertRaises(ValueError):
            fmt.MlxAffineIdentity.mtp(
                identity.original_name,
                "model.layers.48.self_attn.q_proj.weight",
                backbone_layers=48,
            )
        with self.assertRaises(ValueError):
            fmt.MlxAffineSegment(
                identity, replace(spec, original_prefix="other"), shard, 128, 64
            )
        with self.assertRaises(ValueError):
            fmt.MlxAffineSegment(identity, spec, shard, 0, 0)

    def test_header_derived_gdn_and_ple_convolution_share_transpose_not_slicing(self):
        # Original e171af86 headers, shards 1 and 10 respectively; no payload read.
        headers = {
            "language_model.model.layers.0.linear_attn.conv1d.weight": {
                "shape": [10240, 4, 1],
                "dtype": "BF16",
                "data_offsets": [4423534912, 4423616832],
            },
            "language_model.model.layers.1.ple.conv1d.weight": {
                "shape": [10240, 4, 1],
                "dtype": "BF16",
                "data_offsets": [4341871010, 4341952930],
            },
        }
        for header in headers.values():
            layout = fmt.MlxAffineLayout("gdn_conv_mlx", header["shape"])
            self.assertEqual(layout.permutation, (0, 2, 1))
            self.assertEqual(layout.runtime_shape, (10240, 1, 4))
            self.assertEqual(
                header["data_offsets"][1] - header["data_offsets"][0],
                prod(layout.source_shape) * 2,
            )
        gdn = fmt.MlxAffineLayout("gdn_conv_mlx", (10240, 4, 1))
        covered = []
        for rank in range(8):
            intervals = gdn.gdn_channel_intervals(rank)
            self.assertEqual(
                intervals,
                (
                    (256 * rank, 256 * (rank + 1)),
                    (2048 + 256 * rank, 2048 + 256 * (rank + 1)),
                    (4096 + 768 * rank, 4096 + 768 * (rank + 1)),
                ),
            )
            self.assertEqual(sum(end - start for start, end in intervals), 1280)
            covered.extend(i for start, end in intervals for i in range(start, end))
        self.assertEqual(sorted(covered), list(range(10240)))
        for rank, tp in ((8, 8), (-1, 8), (0, 3)):
            with self.assertRaises(ValueError):
                gdn.gdn_channel_intervals(rank, tp)

    def test_header_derived_vision_permutation_preserves_explicit_axes(self):
        # vision_tower.patch_embed.proj.weight, original shard 1 header.
        layout = fmt.MlxAffineLayout("vision_patch_mlx", [1152, 2, 16, 16, 3])
        self.assertEqual(layout.permutation, (0, 4, 1, 2, 3))
        self.assertEqual(layout.runtime_shape, (1152, 3, 2, 16, 16))
        self.assertEqual(prod(layout.source_shape) * 2, 3697571072 - 3694032128)
        for tag, shape in (
            ("vision_patch_mlx", (1152, 3, 2, 16, 16)),
            ("gdn_conv_mlx", (2560, 4, 4)),
            ("gdn_conv_mlx", (10240, 1, 4)),
            ("unknown", (1,)),
            ("identity", (0,)),
            ("identity", (True,)),
        ):
            with self.assertRaises(ValueError):
                fmt.MlxAffineLayout(tag, shape)
        self.assertEqual(fmt.MlxAffineLayout("identity", (4, 8)).runtime_shape, (4, 8))


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.specs = fmt.MlxAffineSpecMap.from_config(_config())
        self.manifest = _manifest(self.specs)

    def test_pretransform_admission_then_destination_completion(self):
        tracker = _tracker(self.manifest, self.specs)
        plans, _ = _plans(self.manifest)
        name = "model.down.weight"
        entry = self.manifest["tensors"][name]
        with self.assertRaises(ValueError):
            tracker.record_destinations(name, plans[name].destinations)
        tracker.record(name, shape=entry["shape"], dtype=entry["dtype"])
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            tracker.record(name, shape=entry["shape"], dtype=entry["dtype"])
        tracker.record_destinations(name, plans[name].destinations)
        with self.assertRaisesRegex(ValueError, "duplicate component destinations"):
            tracker.record_destinations(name, plans[name].destinations)
        for other in ("model.down.scales", "model.down.biases"):
            entry = self.manifest["tensors"][other]
            tracker.record(other, shape=entry["shape"], dtype=entry["dtype"])
        with self.assertRaisesRegex(ValueError, "missing component destination"):
            tracker.finish()
        for other in ("model.down.scales", "model.down.biases"):
            tracker.record_destinations(other, plans[other].destinations)
        report = tracker.finish()
        self.assertEqual(
            (
                report.source_components,
                report.local_components,
                report.nonlocal_components,
                report.destinations,
            ),
            (3, 3, 0, 3),
        )
        with self.assertRaisesRegex(ValueError, "sealed"):
            tracker.record(name, shape=(4, 120), dtype="U32")

    def test_missing_duplicate_undeclared_and_bad_observed_source_fail(self):
        tracker = _tracker(self.manifest, self.specs)
        with self.assertRaisesRegex(ValueError, "missing source"):
            tracker.finish()
        for name, shape, dtype in (
            ("other", (4, 120), "U32"),
            ("model.down.weight", (4, 119), "U32"),
            ("model.down.weight", (4, 120), "I32"),
            ("model.down.weight", (4, 120), "BF16"),
        ):
            with self.assertRaises(ValueError):
                tracker.record(name, shape=shape, dtype=dtype)
        _admit(tracker, self.manifest)
        self.assertEqual(tracker.finish().source_components, 3)

    def test_manifest_rejects_missing_empty_mismatched_or_undeclared_metadata(self):
        for suffix in ("weight", "scales", "biases"):
            manifest = deepcopy(self.manifest)
            del manifest["tensors"]["model.down." + suffix]
            with self.assertRaisesRegex(ValueError, "missing affine"):
                _tracker(manifest, self.specs)
        for shape in ([4, 0], [4, 4], [20], [True, 5]):
            manifest = deepcopy(self.manifest)
            manifest["tensors"]["model.down.scales"]["shape"] = shape
            with self.assertRaises(ValueError):
                _tracker(manifest, self.specs)
        manifest = deepcopy(self.manifest)
        manifest["tensors"]["model.down.scales"]["dtype"] = "F16"
        with self.assertRaisesRegex(ValueError, "dtype"):
            _tracker(manifest, self.specs)
        empty_specs = fmt.MlxAffineSpecMap.from_config(_config({}))
        manifest = deepcopy(self.manifest)
        manifest["module_specs_sha256"] = empty_specs.sha256
        with self.assertRaisesRegex(ValueError, "undeclared affine"):
            _tracker(manifest, empty_specs)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            fmt.load_mlx_affine_json('{"tensors":{"x.scales":{},"x.scales":{}}}')

    def test_metadata_shapes_are_checked_even_when_byte_extent_is_consistent(self):
        records = {
            "model.down.weight": ("U32", (4, 120)),
            "model.down.scales": ("BF16", (4, 4)),
            "model.down.biases": ("BF16", (4, 4)),
        }
        with self.assertRaisesRegex(ValueError, "group extent"):
            _tracker(_manifest(self.specs, records), self.specs)
        records["model.down.scales"] = ("BF16", (2, 10))
        records["model.down.biases"] = ("BF16", (2, 10))
        with self.assertRaisesRegex(ValueError, "group extent"):
            _tracker(_manifest(self.specs, records), self.specs)

    def test_architectural_bias_is_not_affine_biases(self):
        records = {
            name: (entry["dtype"], tuple(entry["shape"]))
            for name, entry in self.manifest["tensors"].items()
        }
        records["model.down.bias"] = ("BF16", (4,))
        manifest = _manifest(self.specs, records)
        tracker = _tracker(manifest, self.specs)
        _admit(tracker, manifest)
        self.assertEqual(tracker.finish().source_components, 4)
        del manifest["tensors"]["model.down.biases"]
        with self.assertRaisesRegex(ValueError, "missing affine"):
            _tracker(manifest, self.specs)

    def test_real_expert_and_ple_geometry_need_no_tensor_allocation(self):
        for prefix, group, weight, metadata in (
            ("experts.gate", 64, (512, 640, 480), (512, 640, 40)),
            ("experts.down", 64, (512, 2560, 120), (512, 2560, 10)),
            ("ple.shards.0", 32, (2500012, 30), (2500012, 5)),
        ):
            specs = fmt.MlxAffineSpecMap.from_config(
                _config({prefix: {"bits": 6, "group_size": group, "mode": "affine"}})
            )
            manifest = _manifest(
                specs,
                {
                    prefix + ".weight": ("U32", weight),
                    prefix + ".scales": ("BF16", metadata),
                    prefix + ".biases": ("BF16", metadata),
                },
            )
            tracker = _tracker(manifest, specs)
            _admit(tracker, manifest)
            self.assertEqual(tracker.finish().source_components, 3)

    def test_partition_holes_overlaps_and_destination_holes_fail(self):
        for local, nonlocal_ in (
            ((Interval(0, 2),), (Interval(3, 4),)),
            ((Interval(0, 3),), (Interval(2, 4),)),
            ((Interval(0, 5),), ()),
        ):
            plans, sizes = _plans(self.manifest)
            for name in plans:
                plans[name] = P(
                    local,
                    nonlocal_,
                    plans[name].destinations,
                    "EP nonowner" if nonlocal_ else None,
                )
            with self.assertRaisesRegex(ValueError, "coverage"):
                _tracker(
                    self.manifest,
                    self.specs,
                    source_plans=plans,
                    destination_sizes=sizes,
                )
        for interval in (Interval(1, 4), Interval(0, 3), Interval(0, 5)):
            plans, sizes = _plans(self.manifest)
            name = "model.down.weight"
            plans[name] = replace(plans[name], destinations=(D(name, interval),))
            with self.assertRaisesRegex(ValueError, "coverage"):
                _tracker(
                    self.manifest,
                    self.specs,
                    source_plans=plans,
                    destination_sizes=sizes,
                )
        plans, sizes = _plans(self.manifest)
        name = "model.down.weight"
        plans[name] = replace(
            plans[name], destinations=(D(name, Interval(0, 3)), D(name, Interval(2, 4)))
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            _tracker(
                self.manifest, self.specs, source_plans=plans, destination_sizes=sizes
            )

    def test_nonlocal_ownership_is_explicit_and_metadata_matches_codes(self):
        plans, sizes = _plans(self.manifest)
        for name in plans:
            plans[name] = P(
                (Interval(1, 3),),
                (Interval(0, 1), Interval(3, 4)),
                (D(name, Interval(0, 2)),),
                "not owned on EP rank",
            )
            sizes[name] = 2
        tracker = _tracker(
            self.manifest, self.specs, source_plans=plans, destination_sizes=sizes
        )
        with self.assertRaisesRegex(ValueError, "ownership"):
            tracker.record("model.down.weight", shape=(4, 120), dtype="U32")
        _admit(tracker, self.manifest, plans, separate=True)
        self.assertEqual(tracker.finish().nonlocal_components, 3)
        for name in plans:
            plans[name] = P((), (Interval(0, 4),), (), "entire tensor nonlocal")
        tracker = _tracker(
            self.manifest, self.specs, source_plans=plans, destination_sizes={}
        )
        _admit(tracker, self.manifest, plans, separate=True)
        self.assertEqual(tracker.finish().local_components, 0)
        with self.assertRaises(ValueError):
            P((), (Interval(0, 4),), ())
        plans, sizes = _plans(self.manifest)
        plans["model.down.scales"] = P((), (Interval(0, 4),), (), "nonlocal scales")
        del sizes["model.down.scales"]
        with self.assertRaisesRegex(ValueError, "metadata ownership"):
            _tracker(
                self.manifest, self.specs, source_plans=plans, destination_sizes=sizes
            )

    def test_declared_padding_requires_explicit_zero_completion(self):
        plans, sizes = _plans(self.manifest)
        sizes["hc.block_inject"] = 16
        padding = D("hc.block_inject", Interval(4, 16))
        name = "model.down.weight"
        plans[name] = replace(
            plans[name],
            destinations=plans[name].destinations
            + (D("hc.block_inject", Interval(0, 4)),),
        )
        tracker = _tracker(
            self.manifest,
            self.specs,
            source_plans=plans,
            destination_sizes=sizes,
            synthetic_padding=(padding,),
        )
        _admit(tracker, self.manifest, plans)
        with self.assertRaisesRegex(ValueError, "synthetic zero padding"):
            tracker.finish()
        tracker.record_padding(padding)
        with self.assertRaisesRegex(ValueError, "duplicate synthetic"):
            tracker.record_padding(padding)
        self.assertEqual(tracker.finish().destinations, 4)

    def test_shared_aliases_finalize_only_after_real_identity_sharing(self):
        tracker = _tracker(
            self.manifest,
            self.specs,
            shared_aliases={"draft.embed": "target.embed", "draft.head": "target.head"},
        )
        _admit(tracker, self.manifest)
        report = tracker.finish(defer_aliases=True)
        self.assertEqual(report.pending_aliases, ("draft.embed", "draft.head"))
        with self.assertRaisesRegex(ValueError, "not finalized"):
            tracker.finish()
        embed, head = object(), object()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            tracker.finish(
                shared_modules={"draft.embed": object(), "target.embed": embed}
            )
        with self.assertRaisesRegex(ValueError, "distinct shared owners"):
            tracker.finish(
                shared_modules={
                    "draft.embed": embed,
                    "target.embed": embed,
                    "draft.head": embed,
                    "target.head": embed,
                }
            )
        report = tracker.finish(
            shared_modules={
                "draft.embed": embed,
                "target.embed": embed,
                "draft.head": head,
                "target.head": head,
            }
        )
        self.assertEqual(report.pending_aliases, ())
        with self.assertRaisesRegex(ValueError, "distinct direct"):
            _tracker(self.manifest, self.specs, shared_aliases={"x": "x"})

    def test_manifest_identity_paths_and_payloads_are_not_inferred(self):
        for mutate in (
            lambda m: m.update(module_specs_sha256="0" * 64),
            lambda m: m.update(view="mtp"),
            lambda m: m.update(format_version=True),
            lambda m: m.update(source_revision="main"),
            lambda m: m["tensors"]["model.down.weight"].update(file="../raw/model"),
            lambda m: m["tensors"]["model.down.weight"].update(data_offsets=[0, 1]),
            lambda m: m["tensors"]["model.down.scales"].update(
                runtime_name="model.down.weight"
            ),
        ):
            manifest = deepcopy(self.manifest)
            mutate(manifest)
            with self.assertRaises(ValueError):
                _tracker(manifest, self.specs)
        plans, sizes = _plans(self.manifest)
        del plans["model.down.biases"]
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            _tracker(
                self.manifest, self.specs, source_plans=plans, destination_sizes=sizes
            )


if __name__ == "__main__":
    unittest.main()
