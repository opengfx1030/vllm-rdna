# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU publication admission/consumer tests; no model runtime or GPU import."""

import copy
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_EXPORTER = Path(__file__).resolve().parents[3] / "tools/rdna/prepare_oq6e.py"
_SPEC = importlib.util.spec_from_file_location("prepare_oq6e", _EXPORTER)
publication = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = publication
_SPEC.loader.exec_module(publication)


INDEX = publication.INDEX
TARGET = "language_model.model.layers.0.linear_attn.in_proj_qkv"
DRAFT = "mtp.layers.0.self_attn.q_proj"
HEAD = "language_model.lm_head"
PLE = publication.PLE_PREFIX


def dump(path, value):
    path.write_bytes(publication.canonical_bytes(value, newline=True))


def tensor(dtype, shape, payload=None):
    size = math.prod(shape) * publication.DTYPE_BYTES[dtype]
    return (
        dtype,
        shape,
        payload if payload is not None else bytes(i % 251 for i in range(size)),
    )


def write_tensors(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    raw_header = publication.canonical_bytes(header)
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    return header, len(payload)


def affine(prefix, bits=6, group=64, k=64, rows=2):
    return {
        prefix + ".weight": tensor("U32", [rows, k * bits // 32]),
        prefix + ".scales": tensor("BF16", [rows, math.ceil(k / group)]),
        prefix + ".biases": tensor("BF16", [rows, math.ceil(k / group)]),
    }


def file_record(path, *, lfs=False):
    raw = path.read_bytes()
    record = {"rfilename": path.name, "size": len(raw)}
    if lfs:
        record["lfs"] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    else:
        record["blobId"] = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    return record


class PublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source, self.assets = self.root / "raw", self.root / "official"
        self.source.mkdir()
        self.assets.mkdir()
        self.destination = self.root / "native"
        self.acquisition_manifest = self.root / "download.json"
        self.acquisition_result = self.root / "result.json"
        self.assets_manifest = self.root / "assets.json"
        self.shard_a = "model-00001-of-00002.safetensors"
        self.shard_b = "model-00002-of-00002.safetensors"
        self.tensors = affine(TARGET)
        self.tensors.update(affine(DRAFT))
        self.tensors.update(affine(HEAD, 8, 128, 128))
        self.tensors["language_model.model.layers.0.norm.weight"] = tensor("BF16", [2])
        self.tensors["vision_tower.blocks.0.norm.weight"] = tensor("BF16", [2])
        self.tensors["language_model.model.layers.0.linear_attn.conv1d.weight"] = (
            tensor("BF16", [10240, 4, 1])
        )
        self.tensors["language_model.model.layers.1.ple.conv1d.weight"] = tensor(
            "BF16", [10240, 4, 1]
        )
        self.quantization = {
            "bits": 6,
            "group_size": 64,
            "mode": "affine",
            HEAD: {"bits": 8, "group_size": 128, "mode": "affine"},
        }
        for i in range(2):
            prefix = PLE + f"ngram_embedding.shards.{i}"
            self.tensors.update(affine(prefix, 6, 32, 160, 1))
            self.quantization[prefix] = {"bits": 6, "group_size": 32, "mode": "affine"}
        for suffix, values in publication.PLE_HASHES.items():
            self.tensors[PLE + suffix] = tensor(
                "I64", [len(values)], struct.pack("<" + "q" * len(values), *values)
            )
        self.tensors[PLE + "ngram_embedding.weight_scale"] = tensor("BF16", [1])
        self.config = {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "tie_word_embeddings": False,
            "text_config": {
                "num_hidden_layers": 48,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ]
                * 12,
                "hc_count": 4,
                "mtp_num_hidden_layers": 1,
                "mtp": {"num_hidden_layers": 1},
                "tie_word_embeddings": False,
                "eos_token_id": 248044,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
            },
            "vision_config": {"depth": 27},
            "quantization": self.quantization,
            "quantization_config": copy.deepcopy(self.quantization),
        }
        self.official_tokenizer = {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": ["a b"],
            },
            "added_tokens": [
                {
                    "id": ident,
                    "content": content,
                    "special": True,
                    "normalized": False,
                    "lstrip": False,
                    "rstrip": False,
                    "single_word": False,
                }
                for content, ident in publication.CONTROL_IDS.items()
            ],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "ByteLevel", "use_regex": False},
        }
        raw_tokenizer = copy.deepcopy(self.official_tokenizer)
        raw_tokenizer["model"]["merges"] = [["a", "b"]]
        raw_tokenizer["added_tokens"].reverse()
        raw_tokenizer["pre_tokenizer"]["use_regex"] = True
        dump(self.source / "tokenizer.json", raw_tokenizer)
        for name in publication.ASSETS:
            if name == "tokenizer.json":
                dump(self.assets / name, self.official_tokenizer)
            elif name == "merges.txt":
                (self.assets / name).write_text(
                    "#version: 0.2\na b\n", encoding="utf-8"
                )
            elif name == "tokenizer_config.json":
                dump(
                    self.assets / name,
                    {
                        "chat_template": "do not replace the live template",
                        "eos_token": "<|im_end|>",
                    },
                )
            else:
                dump(self.assets / name, {})
        # Synthetic acquisition fixtures use their own cryptographic pins, never
        # weakening the real entry point's pinned production identities.
        self.patchers = [
            patch.object(
                publication,
                "TOKENIZER_SHA256",
                hashlib.sha256(
                    (self.assets / "tokenizer.json").read_bytes()
                ).hexdigest(),
            ),
            patch.object(
                publication,
                "SEMANTIC_SHA256",
                hashlib.sha256(
                    publication.canonical_bytes(
                        publication.tokenizer_semantics(self.official_tokenizer),
                        newline=True,
                    )
                ).hexdigest(),
            ),
            patch.object(publication, "PLE_ROWS", 1),
            patch.object(publication, "PLE_SHARDS", 2),
            patch.object(publication, "EXPECTED_TENSORS", len(self.tensors)),
            patch.object(publication, "EXPECTED_MTP_TENSORS", 3),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.reseal()

    def reseal(self):
        dump(self.source / "config.json", self.config)
        first = {k: v for k, v in self.tensors.items() if not k.endswith(".biases")}
        second = {k: v for k, v in self.tensors.items() if k.endswith(".biases")}
        header_a, payload_a = write_tensors(self.source / self.shard_a, first)
        header_b, payload_b = write_tensors(self.source / self.shard_b, second)
        file_bytes = sum(
            (self.source / name).stat().st_size for name in (self.shard_a, self.shard_b)
        )
        dump(
            self.source / INDEX,
            {
                "metadata": {"total_size": file_bytes},
                "weight_map": {name: self.shard_a for name in header_a}
                | {name: self.shard_b for name in header_b},
            },
        )
        (self.source / ".gitattributes").write_text(
            "*.safetensors filter=lfs\n", encoding="utf-8"
        )
        self.reseal_manifests()

    def reseal_manifests(self):
        files = [
            file_record(path, lfs=path.suffix == ".safetensors")
            for path in sorted(self.source.iterdir())
        ]
        dump(
            self.acquisition_manifest,
            {
                "repo_id": publication.REPO,
                "revision": publication.REVISION,
                "destination": str(self.source),
                "total_bytes": sum(f["size"] for f in files),
                "files": files,
            },
        )
        verified = [
            {
                "file": f["rfilename"],
                "bytes": f["size"],
                "verified": True,
                "digest": f["lfs"]["sha256"] if "lfs" in f else f["blobId"],
                "hash_kind": "sha256" if "lfs" in f else "git_blob_sha1",
            }
            for f in files
        ]
        dump(
            self.acquisition_result,
            {
                "ok": True,
                "revision": publication.REVISION,
                "destination": str(self.source),
                "total_bytes": sum(f["size"] for f in files),
                "files": verified,
            },
        )
        dump(
            self.assets_manifest,
            {
                "repo_id": publication.ASSET_REPO,
                "revision": publication.ASSET_REVISION,
                "destination": str(self.assets),
                "files": [file_record(p) for p in sorted(self.assets.iterdir())],
            },
        )

    def publish(self):
        return publication.prepare_checkpoint(
            self.source,
            self.destination,
            self.acquisition_manifest,
            self.acquisition_result,
            self.assets,
            self.assets_manifest,
            self.root,
        )

    def test_publication_partitions_mixed_shards_without_changing_payload(self):
        source_bytes = {p.name: p.read_bytes() for p in self.source.iterdir()}
        receipt = self.publish()
        target = json.loads((self.destination / publication.MANIFEST).read_bytes())
        draft = json.loads(
            (self.destination / "mtp" / publication.MANIFEST).read_bytes()
        )
        self.assertFalse(target["tensors"].keys() & draft["tensors"].keys())
        self.assertEqual(
            target["tensors"].keys() | draft["tensors"].keys(), self.tensors.keys()
        )
        self.assertEqual(
            draft["tensors"][DRAFT + ".weight"]["runtime_name"],
            "model.layers.0.self_attn.q_proj.weight",
        )
        for view in (self.destination, self.destination / "mtp"):
            for shard in (self.shard_a, self.shard_b):
                self.assertTrue((view / shard).is_symlink())
                self.assertFalse(os.path.isabs(os.readlink(view / shard)))
                self.assertEqual((view / shard).resolve(), self.source / shard)
                self.assertEqual((view / shard).read_bytes(), source_bytes[shard])
            config = json.loads((view / "config.json").read_bytes())
            self.assertEqual(
                config["mlx_affine"]["manifest_sha256"],
                hashlib.sha256((view / publication.MANIFEST).read_bytes()).hexdigest(),
            )
            self.assertNotIn("quantization", config)
            specs = config["quantization_config"]["module_specs"]
            self.assertEqual(
                specs[HEAD], {"bits": 8, "group_size": 128, "mode": "affine"}
            )
            self.assertEqual(specs[TARGET]["bits"], 6)
            self.assertNotIn("vision_tower.blocks.0.norm", specs)
            self.assertNotIn("language_model.model.layers.0.norm", specs)
            self.assertEqual(config["text_config"]["indexer_budget"], 4096)
            self.assertEqual(config["text_config"]["indexer_compress_ratio"], 4)
            self.assertFalse((view / "chat_template.jinja").exists())
            self.assertNotIn(
                "chat_template",
                json.loads((view / "tokenizer_config.json").read_bytes()),
            )
            self.assertEqual(
                (view / "tokenizer.json").read_bytes(),
                (self.assets / "tokenizer.json").read_bytes(),
            )
        self.assertEqual(
            source_bytes, {p.name: p.read_bytes() for p in self.source.iterdir()}
        )
        draft_config = json.loads(
            (self.destination / "mtp" / "config.json").read_bytes()
        )
        self.assertEqual(draft_config["architectures"], ["Qwen4ExpMTP"])
        self.assertEqual(draft_config["model_type"], "qwen4_exp_mtp")
        self.assertEqual(draft_config["hc_mult"], 4)
        self.assertEqual(draft_config["n_predict"], 1)
        self.assertEqual(draft_config["text_config"]["num_hidden_layers"], 48)
        self.assertEqual(receipt["mtp"]["construction_layer_prefix"], "mtp.layers.48")
        self.assertFalse(receipt["tokenizer_processor"]["pipelines_equal"])
        self.assertEqual(
            hashlib.sha256(
                (self.destination / "tokenizer-semantics.json").read_bytes()
            ).hexdigest(),
            publication.SEMANTIC_SHA256,
        )
        specs = json.loads((self.destination / "config.json").read_bytes())[
            "quantization_config"
        ]["module_specs"]
        self.assertEqual(
            hashlib.sha256(publication.canonical_bytes(specs)).hexdigest(),
            receipt["module_specs_sha256"],
        )
        self.assertEqual(receipt["ple"]["padded_rows"], 2)
        for family in ("layers.0.linear_attn", "layers.1.ple"):
            self.assertEqual(
                target["tensors"]["language_model.model." + family + ".conv1d.weight"][
                    "layout"
                ],
                "gdn_conv_mlx",
            )

    def test_configs_do_not_share_quantization_dictionaries(self):
        specs = {TARGET: {"bits": 6, "group_size": 64, "mode": "affine"}}
        configs = publication.normalize_configs(self.config, specs)
        configs["mtp"]["quantization_config"]["module_specs"][TARGET]["bits"] = 8
        self.assertEqual(
            configs["target"]["quantization_config"]["module_specs"][TARGET]["bits"], 6
        )
        self.assertEqual(specs[TARGET]["bits"], 6)

    def test_existing_destination_and_dangling_symlink_are_never_replaced(self):
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            self.publish()
        self.destination.rmdir()
        self.destination.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertTrue(self.destination.is_symlink())

    def test_publication_race_keeps_concurrently_created_directory(self):
        original = publication._publish

        def race(staging, destination):
            destination.mkdir()
            (destination / "owner").write_text("concurrent owner")
            original(staging, destination)

        with (
            patch.object(publication, "_publish", side_effect=race),
            self.assertRaises(OSError),
        ):
            self.publish()
        self.assertEqual((self.destination / "owner").read_text(), "concurrent owner")
        self.assertFalse(list(self.root.glob(".native.prepare-*")))

    def test_protected_root_and_symlink_containment(self):
        outside = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: outside.rmdir())
        with self.assertRaisesRegex(ValueError, "protected model root"):
            publication.prepare_checkpoint(
                self.source,
                outside / "native",
                self.acquisition_manifest,
                self.acquisition_result,
                self.assets,
                self.assets_manifest,
                self.root,
            )
        shard = self.source / self.shard_a
        moved = self.root / "moved.safetensors"
        shard.rename(moved)
        shard.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.publish()
        self.assertFalse(self.destination.exists())

    def test_shard_corruption_and_forged_result_digest_fail_closed(self):
        path = self.source / self.shard_a
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.publish()
        self.reseal()
        result = json.loads(self.acquisition_result.read_bytes())
        result["files"][0]["digest"] = "0" * 40
        dump(self.acquisition_result, result)
        with self.assertRaisesRegex(ValueError, "verification disagrees"):
            self.publish()

    def test_missing_and_orphan_metadata_are_rejected(self):
        del self.tensors[TARGET + ".biases"]
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Missing affine metadata"):
            self.publish()
        self.tensors[TARGET + ".biases"] = tensor("BF16", [2, 1])
        self.tensors["language_model.model.orphan.scales"] = tensor("BF16", [2, 1])
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Orphan affine metadata"):
            self.publish()

    def test_duplicate_metadata_header_keys_are_rejected(self):
        path = self.source / self.shard_b
        raw = path.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + n])
        key = TARGET + ".biases"
        pairs = list(header.items()) + [(key, header[key])]
        duplicate = (
            "{" + ",".join(json.dumps(k) + ":" + json.dumps(v) for k, v in pairs) + "}"
        ).encode()
        path.write_bytes(struct.pack("<Q", len(duplicate)) + duplicate + raw[8 + n :])
        self.reseal_manifests()
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.publish()

    def test_duplicate_tensor_across_shards_is_rejected(self):
        other = {TARGET + ".weight": self.tensors[TARGET + ".weight"]}
        other.update({k: v for k, v in self.tensors.items() if k.endswith(".biases")})
        write_tensors(self.source / self.shard_b, other)
        self.reseal_manifests()
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            self.publish()

    def test_raw_index_binds_complete_shard_bytes_not_only_payload(self):
        index = json.loads((self.source / INDEX).read_bytes())
        index["metadata"]["total_size"] = sum(
            math.prod(shape) * publication.DTYPE_BYTES[dtype]
            for dtype, shape, _ in self.tensors.values()
        )
        dump(self.source / INDEX, index)
        self.reseal_manifests()
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(self.destination.exists())

    def test_missing_index_entry_and_wrong_metadata_groups_fail(self):
        index = json.loads((self.source / INDEX).read_bytes())
        del index["weight_map"][TARGET + ".scales"]
        dump(self.source / INDEX, index)
        self.reseal_manifests()
        with self.assertRaisesRegex(ValueError, "Unindexed"):
            self.publish()
        self.tensors[TARGET + ".scales"] = tensor("BF16", [2, 2])
        self.tensors[TARGET + ".biases"] = tensor("BF16", [2, 2])
        self.reseal()
        with self.assertRaisesRegex(ValueError, "group count"):
            self.publish()

    def test_conflicting_duplicate_configs_and_unpacked_override_fail(self):
        self.config["quantization_config"]["bits"] = 8
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.publish()
        self.config["quantization_config"] = copy.deepcopy(self.quantization)
        self.quantization["vision_tower.blocks.0.norm"] = {
            "bits": 6,
            "group_size": 64,
            "mode": "affine",
        }
        self.config["quantization_config"] = copy.deepcopy(self.quantization)
        self.reseal()
        with self.assertRaisesRegex(ValueError, "lacks a packed source"):
            self.publish()

    def test_incomplete_ple_rows_and_bad_hash_buffers_fail(self):
        name = PLE + "ngram_embedding.shards.1.weight"
        self.tensors[name] = tensor("U32", [2, 30])
        self.tensors[name.removesuffix("weight") + "scales"] = tensor("BF16", [2, 5])
        self.tensors[name.removesuffix("weight") + "biases"] = tensor("BF16", [2, 5])
        self.reseal()
        with self.assertRaisesRegex(ValueError, "complete-row geometry"):
            self.publish()
        self.tensors.update(affine(PLE + "ngram_embedding.shards.1", 6, 32, 160, 1))
        self.tensors[PLE + "layer_multipliers"] = tensor(
            "I64", [3], struct.pack("<qqq", 1, 2, 3)
        )
        self.reseal()
        with self.assertRaisesRegex(ValueError, "hash identity"):
            self.publish()

    def test_tokenizer_semantics_check_full_records_not_only_vocab(self):
        raw = json.loads((self.source / "tokenizer.json").read_bytes())
        raw["added_tokens"][0]["lstrip"] = True
        dump(self.source / "tokenizer.json", raw)
        self.reseal_manifests()
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            self.publish()

    def test_declared_layouts_require_exact_dimensions(self):
        runtime, layout = publication.runtime_mapping(
            "vision_tower.patch_embed.proj.weight",
            {"dtype": "BF16", "shape": [1152, 2, 16, 16, 3]},
        )
        self.assertEqual(
            (runtime, layout), ("visual.patch_embed.proj.weight", "vision_patch_mlx")
        )
        for name, shape in (
            ("vision_tower.patch_embed.proj.weight", [1152, 3, 2, 16, 16]),
            ("language_model.model.layers.1.ple.conv1d.weight", [10240, 1, 4]),
        ):
            with self.assertRaisesRegex(ValueError, "layout"):
                publication.runtime_mapping(name, {"dtype": "BF16", "shape": shape})
        with self.assertRaisesRegex(ValueError, "layer0"):
            publication.runtime_mapping(
                "mtp.layers.48.self_attn.q_proj.weight",
                {"dtype": "U32", "shape": [2, 12]},
            )

    def test_unsupported_precision_zero_metadata_and_manifest_traversal_fail(self):
        self.config["quantization"]["bits"] = 4
        self.config["quantization_config"]["bits"] = 4
        self.reseal()
        with self.assertRaisesRegex(ValueError, "Unsupported affine"):
            self.publish()
        self.config["quantization"]["bits"] = 6
        self.config["quantization_config"]["bits"] = 6
        self.tensors[TARGET + ".scales"] = tensor("BF16", [2, 0])
        self.tensors[TARGET + ".biases"] = tensor("BF16", [2, 0])
        self.reseal()
        with self.assertRaisesRegex(ValueError, "matrix geometry"):
            self.publish()
        manifest = json.loads(self.acquisition_manifest.read_bytes())
        manifest["files"][0]["rfilename"] = "../outside"
        dump(self.acquisition_manifest, manifest)
        with self.assertRaisesRegex(ValueError, "Unsafe flat filename"):
            self.publish()


class CanonicalTokenizerBehaviorTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("OQ6E_CANONICAL_ASSETS")
        and importlib.util.find_spec("tokenizers"),
        "Set OQ6E_CANONICAL_ASSETS with CPU tokenizers available",
    )
    def test_canonical_combining_marks_whitespace_and_control_tokens(self):
        from tokenizers import Tokenizer

        source = Path(os.environ["OQ6E_CANONICAL_ASSETS"])
        raw = (source / "tokenizer.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), publication.TOKENIZER_SHA256)
        document = json.loads(raw)
        normalized = {
            "vocab": document["model"]["vocab"],
            "merges": [
                item.split(" ") if isinstance(item, str) else item
                for item in document["model"]["merges"]
            ],
            "added_tokens": sorted(
                document["added_tokens"], key=lambda item: item["id"]
            ),
        }
        independent = (
            json.dumps(
                normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(independent).hexdigest(),
            "327de7f97ba22799dee302f7e48263b664816b4e1ef717641491872a0a62bd3b",
        )
        tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        for text in ("e\u0301", "a\u0308\u0301", "   a\t\nb  ", "\n\n "):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            import unicodedata

            self.assertEqual(
                tokenizer.decode(encoded.ids, skip_special_tokens=False),
                unicodedata.normalize("NFC", text),
            )
        for content, ident in publication.CONTROL_IDS.items():
            self.assertEqual(
                tokenizer.encode(content, add_special_tokens=False).ids, [ident]
            )


if __name__ == "__main__":
    unittest.main()
