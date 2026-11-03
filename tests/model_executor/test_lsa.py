# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the LSA reference and the default-off guard.

Loads the modules by path so the test does not import torch. Run:

    python3 tests/model_executor/test_lsa.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REF = _load("lsa_reference", ROOT / "docs/explore/lsa/reference_lsa.py")
GUARD = _load(
    "lsa_guard",
    ROOT / "vllm/model_executor/layers/lsa/guard.py",
)


def _pattern(n_layers: int) -> list[str]:
    return [
        "full_attention" if i % 4 == 3 else "linear_attention"
        for i in range(n_layers)
    ]


class ReferenceTest(unittest.TestCase):
    def test_pool_sums_consecutive_keys(self) -> None:
        keys = [[1.0, 0.0], [3.0, 4.0], [0.0, 1.0], [0.0, 2.0]]
        pooled = REF.pool_keys(keys, [0, 1, 2, 3], 2)
        self.assertEqual(len(pooled), 2)
        self.assertEqual(pooled[0].total, [4.0, 4.0])
        self.assertEqual(pooled[0].count, 2)
        self.assertEqual(pooled[0].positions, [0, 1])
        self.assertEqual(pooled[1].positions, [2, 3])

    def test_ward_separates_two_directions(self) -> None:
        keys = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        pooled = REF.pool_keys(keys, [0, 1, 2, 3], 1)
        clustered = REF.ward_cluster(pooled, m=2, n_b_max=2)
        groups = [set(b.positions) for b in clustered]
        self.assertIn({0, 1}, groups)
        self.assertIn({2, 3}, groups)

    def test_ward_skips_merges_past_max_keys(self) -> None:
        keys = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        pooled = REF.pool_keys(keys, [0, 1, 2], 1)
        clustered = REF.ward_cluster(pooled, m=1, n_b_max=2)
        self.assertGreater(len(clustered), 1)
        self.assertTrue(all(b.count <= 2 for b in clustered))

    def test_flat_selects_the_matching_bucket(self) -> None:
        keys = [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4
        pooled = REF.pool_keys(keys, list(range(8)), 1)
        clustered = REF.ward_cluster(pooled, m=2, n_b_max=4)
        chosen = REF.expand_selection(
            [1.0, 0.0],
            clustered,
            k=2,
            p=1,
            n_sink=1,
            n_window=0,
            query_pos=8,
        )
        self.assertEqual(chosen[:1], [0])
        self.assertEqual(set(chosen[1:]), {1, 2})

    def test_tree_beam_reaches_the_matching_leaf(self) -> None:
        keys = [[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2
        pooled = REF.pool_keys(keys, list(range(4)), 1)
        chosen = REF.tree_select(
            [0.0, 1.0],
            pooled,
            arity=2,
            beam=1,
            k=2,
            p=2,
            n_sink=0,
            n_window=0,
            query_pos=4,
        )
        self.assertEqual(set(chosen), {2, 3})

    def test_hub_pattern_counts(self) -> None:
        # Qwen/Qwen3.8-Flash-Next text_config: 48 layers, softmax every 4th.
        flash = REF.softmax_layer_ids(_pattern(48))
        self.assertEqual(len(flash), 12)
        self.assertEqual(flash[0], 3)
        self.assertEqual(flash[-1], 47)
        # Qwen/Qwen3.8-27B and Qwen/Qwen3.6-27B: 64 layers, same rhythm.
        dense = REF.softmax_layer_ids(_pattern(64))
        self.assertEqual(len(dense), 16)
        self.assertEqual(dense[-1], 63)
        self.assertNotIn(0, dense)

    def test_kv_and_centroid_bytes(self) -> None:
        flash_heads = REF.heads_per_rank(2, 4)
        self.assertEqual(flash_heads, 1)
        flash = REF.kv_bytes_per_token(12, flash_heads, 256)
        self.assertEqual(flash, 12288)
        self.assertEqual(flash * 262144, 3 * 2**30)
        qwen_heads = REF.heads_per_rank(4, 4)
        qwen = REF.kv_bytes_per_token(16, qwen_heads, 256)
        self.assertEqual(qwen, 16384)
        self.assertEqual(qwen * 1048576, 16 * 2**30)
        # Paper Appendix F: 160 MiB of centroids at 10 * 2**20 tokens.
        self.assertEqual(REF.centroid_bytes(10 * 2**20), 160 * 2**20)

    def test_toy_kl_falls(self) -> None:
        history = REF.train_toy_indexer()
        self.assertGreater(len(history), 2)
        self.assertLess(history[-1], history[0])


class GuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get("VLLM_LSA")
        os.environ.pop("VLLM_LSA", None)
        GUARD.reset_lsa_state_for_tests()

    def test_default_is_off_and_exact_flag_does_not_select(self) -> None:
        self.assertEqual(GUARD.lsa_mode(), "off")
        self.assertEqual(
            GUARD.check_layer_types(["linear_attention", "full_attention"]),
            [1],
        )
        GUARD.reset_lsa_state_for_tests()
        os.environ["VLLM_LSA"] = "1"
        self.assertEqual(GUARD.lsa_mode(), "exact")
        self.assertEqual(
            GUARD.check_layer_types(
                [
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "qwen_sparse_attention",
                ]
            ),
            [1, 3],
        )

    def test_selector_value_fails_closed(self) -> None:
        os.environ["VLLM_LSA"] = "select"
        with self.assertRaises(GUARD.LSAConfigError):
            GUARD.lsa_mode()

    def test_armed_gdn_only_model_raises(self) -> None:
        os.environ["VLLM_LSA"] = "exact"
        with self.assertRaises(GUARD.LSAConfigError):
            GUARD.check_layer_types(["linear_attention", "linear_attention"])

    def test_armed_unknown_layer_raises(self) -> None:
        os.environ["VLLM_LSA"] = "1"
        with self.assertRaises(GUARD.LSAConfigError):
            GUARD.check_layer_types(["full_attention", "not_a_layer"])

    def test_gfx1030_capability_collides_with_their_b300(self) -> None:
        # dispatch.py at 906be195: B300 is (10, 3), and runs_cutedsl
        # requires device.type == "cuda". rocm.py parses gfx1030 the
        # same way: four digits, major = digits[:2], minor = digits[2].
        self.assertTrue(
            GUARD.upstream_would_pick_cute("cutedsl", "cuda", (10, 3))
        )
        self.assertFalse(
            GUARD.upstream_would_pick_cute("triton", "cuda", (10, 3))
        )

    def test_exact_boot_refuses_cute_tf32_and_bf16_split(self) -> None:
        os.environ["VLLM_LSA"] = "1"
        os.environ["VLLM_LSA_KERNELS"] = "cutedsl"
        with self.assertRaises(GUARD.LSAConfigError):
            GUARD.lsa_mode()
        GUARD.reset_lsa_state_for_tests()
        os.environ["VLLM_LSA_KERNELS"] = "triton"
        os.environ["VLLM_LSA_WARD_PRECISION"] = "tf32x3"
        with self.assertRaises(GUARD.LSAConfigError):
            GUARD.lsa_mode()
        GUARD.reset_lsa_state_for_tests()
        os.environ["VLLM_LSA_WARD_PRECISION"] = "ieee"
        os.environ["VLLM_LSA_SCORE"] = "bf16-split"
        with self.assertRaises(GUARD.LSAConfigError):
            GUARD.lsa_mode()

    def test_flag_off_ignores_a_stray_cute_request(self) -> None:
        os.environ["VLLM_LSA_KERNELS"] = "cutedsl"
        self.assertEqual(GUARD.lsa_mode(), "off")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("VLLM_LSA", None)
        else:
            os.environ["VLLM_LSA"] = self._old
        for key in (
            "VLLM_LSA_KERNELS",
            "VLLM_LSA_WARD_PRECISION",
            "VLLM_LSA_SCORE",
        ):
            os.environ.pop(key, None)
        GUARD.reset_lsa_state_for_tests()


if __name__ == "__main__":
    unittest.main()
