# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manifest admission must reject a changed view before opening tensor payloads."""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "mlx_affine_checkpoint_cpu_test",
    _SOURCE / "vllm/model_executor/model_loader/mlx_affine_checkpoint.py",
)
checkpoint = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checkpoint
_SPEC.loader.exec_module(checkpoint)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


class NativeViewAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = self.root / "models"
        self.raw = self.models / "raw"
        self.target = self.models / "target"
        self.draft = self.target / "mtp"
        self.raw.mkdir(parents=True)
        self.draft.mkdir(parents=True)
        self.shard = self.raw / "model-00001.safetensors"
        self.shard.write_bytes(b"payload must remain unopened by manifest admission")
        self.specs = {"target.proj": {"bits": 6, "group_size": 64, "mode": "affine"}}
        self.config = {
            "quantization_config": {
                "quant_method": "mlx_affine",
                "format_version": 1,
                "module_specs": self.specs,
            },
        }

    def publish(self, view):
        directory = self.target if view == "target" else self.draft
        (directory / self.shard.name).symlink_to(os.path.relpath(self.shard, directory))
        manifest = {
            "format_version": 1,
            "view": view,
            "source_revision": "e171af86f499f1855b0fb71d781105e8dd609610",
            "module_specs_sha256": _digest(self.specs),
            "tensors": {
                f"{view}.proj.weight": {
                    "file": self.shard.name,
                    "dtype": "U32",
                    "shape": [1, 12],
                    "data_offsets": [0, 48],
                    "runtime_name": "proj.weight",
                    "layout": "identity",
                },
            },
        }
        raw = json.dumps(manifest).encode()
        (directory / "mlx-affine-manifest.json").write_bytes(raw)
        config = dict(
            self.config,
            mlx_affine={
                "manifest": "mlx-affine-manifest.json",
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "view": view,
            },
        )
        return directory, config

    def test_target_and_nested_draft_bind_only_their_selected_components(self):
        for view in ("target", "mtp"):
            directory, config = self.publish(view)
            admitted = checkpoint.load_mlx_affine_checkpoint(
                directory, config, protected_model_root=self.models
            )
            self.assertEqual(admitted.tensor_names, frozenset({f"{view}.proj.weight"}))
            self.assertEqual(admitted.shard_paths, (str(directory / self.shard.name),))
            self.assertEqual(
                admitted.manifest_sha256, config["mlx_affine"]["manifest_sha256"]
            )

    def test_changed_manifest_or_precision_map_fails_before_loading(self):
        directory, config = self.publish("target")
        path = directory / "mlx-affine-manifest.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "config pin"):
            checkpoint.load_mlx_affine_checkpoint(directory, config)
        path.write_bytes(original)
        config["quantization_config"]["module_specs"]["target.proj"]["bits"] = 8
        with self.assertRaisesRegex(ValueError, "quantization map disagree"):
            checkpoint.load_mlx_affine_checkpoint(directory, config)

    def test_relative_shard_cannot_escape_protected_storage(self):
        directory, config = self.publish("target")
        outside = self.root / "outside.safetensors"
        outside.write_bytes(b"not admitted")
        link = directory / self.shard.name
        link.unlink()
        link.symlink_to(os.path.relpath(outside, directory))
        with self.assertRaisesRegex(ValueError, "escapes protected"):
            checkpoint.load_mlx_affine_checkpoint(
                directory, config, protected_model_root=self.models
            )

    def test_even_hash_pinned_duplicate_names_are_rejected(self):
        directory, config = self.publish("target")
        path = directory / "mlx-affine-manifest.json"
        raw = path.read_bytes().replace(
            b'"view": "target"', b'"view": "mtp", "view": "target"'
        )
        path.write_bytes(raw)
        config["mlx_affine"]["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "Duplicate affine manifest key"):
            checkpoint.load_mlx_affine_checkpoint(directory, config)


if __name__ == "__main__":
    unittest.main()
