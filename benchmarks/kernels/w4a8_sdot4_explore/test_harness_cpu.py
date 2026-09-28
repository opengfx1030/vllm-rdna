# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU smoke test of the V620 harness plumbing (needs torch, no GPU).

``bench.py`` is exercised end to end with the HIP kernels and the vLLM ops
replaced by the NumPy reference, reading the same tiled device buffers the
real kernels would. This catches buffer-size, layout and bound mistakes in
the harness before GPU time is spent; it says nothing about the kernels.

    .venv/bin/python -m pytest benchmarks/kernels/w4a8_sdot4_explore -q
"""

import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.kernels.w4a8_sdot4_explore import bench  # noqa: E402
from benchmarks.kernels.w4a8_sdot4_explore import reference as ref  # noqa: E402
from benchmarks.kernels.w4a8_sdot4_explore.lib import Config  # noqa: E402

INV_A_PERM = np.argsort(ref.A_PERM)


def _unpack_zeros(qzeros: np.ndarray) -> np.ndarray:
    g, n8 = qzeros.shape
    z = np.stack([(qzeros >> np.uint32(4 * j)) & 0xF for j in range(8)], -1)
    return z.reshape(g, n8 * 8).astype(np.int64)


class FakeOps:
    """The three vLLM ops the harness calls, via the reference model."""

    @staticmethod
    def gptq_shuffle(w, g_idx, bits):
        assert bits == 4 and g_idx.numel() == 0
        w.copy_(
            torch.from_numpy(
                ref.exllama_shuffle(w.numpy().view(np.uint32)).view(np.int32)
            )
        )

    @staticmethod
    def gptq_gemm_rdna2_prefill(x, w, qzeros, scales, g_idx, use_v2_format):
        q = ref.unpack_shuffled(w.numpy().view(np.uint32))
        z = _unpack_zeros(qzeros.numpy().view(np.uint32)) + (0 if use_v2_format else 1)
        gs = q.shape[0] // z.shape[0]
        c = ref.w4a16_reference(x.numpy(), q, z, scales.numpy(), gs)
        return torch.from_numpy(c.astype(np.float16))

    @staticmethod
    def gptq_gemm(x, w, qzeros, scales, g_idx, use_exllama, use_v2_format, bits):
        assert use_exllama and bits == 4
        return FakeOps.gptq_gemm_rdna2_prefill(
            x, w, qzeros, scales, g_idx, use_v2_format
        )

    @staticmethod
    def scaled_int8_quant(x):
        q, s = ref.quantize_per_token(x.numpy())
        return torch.from_numpy(q), torch.from_numpy(s[:, None]), None


class FakeLib:
    """W4A8Lib stand-in that decodes the tiled buffers like the kernels."""

    probe_chains = 8

    def __init__(self):
        self.configs = [
            Config(0, "a16_lds_k32", 16, 1024),
            Config(3, "a8_smem_k32", 8, 1024),
        ]

    def config(self, key):
        return next(c for c in self.configs if key in (c.id, c.name))

    def pick_split_k(self, m, n, k, g, cfg):
        c = self.config(cfg)
        return ref.pick_split_k(m, n, k, g, c.m_tile, c.n_tile)

    def act_quant(self, x, a, a_scale, asum, group_size, m_tile):
        act = ref.quantize_act(x.numpy(), group_size)
        assert a.numel() == -(-x.shape[0] // m_tile) * m_tile * x.shape[1]
        a.copy_(torch.from_numpy(ref.tile_a(act.a_perm, m_tile)))
        a_scale.copy_(torch.from_numpy(act.scale))
        asum.copy_(torch.from_numpy(ref.tile_asum(act.asum, m_tile)))

    def gemm(
        self,
        a,
        w,
        qzeros,
        scales,
        a_scale,
        asum,
        out,
        k,
        group_size,
        zero_offset,
        cfg,
        split_k=0,
    ):
        m, n = out.shape
        mt = self.config(cfg).m_tile
        a_perm = ref.untile_a(a.numpy(), m, k, mt)
        a_nat = a_perm.reshape(m, k // 8, 8)[:, :, INV_A_PERM].reshape(m, k)
        groups = k // group_size
        tiles = -(-m // mt)
        sums = asum.numpy().reshape(tiles, groups, mt).transpose(0, 2, 1)
        act = ref.ActQuant(
            a_nat, a_perm, a_scale.numpy(), sums.reshape(tiles * mt, groups)[:m]
        )
        split = split_k or self.pick_split_k(m, n, k, group_size, cfg)
        z = _unpack_zeros(qzeros.numpy().view(np.uint32)) + zero_offset
        c, _ = ref.emulate_kernel(
            act,
            w.numpy().view(np.uint32),
            z,
            scales.numpy(),
            group_size,
            split_k=split,
            out_f16=out.dtype == torch.float16,
        )
        out.copy_(torch.from_numpy(c))


@pytest.fixture
def cpu_bench(monkeypatch):
    monkeypatch.setattr(bench, "DEVICE", "cpu")
    monkeypatch.setattr(bench, "_ops", lambda: FakeOps)
    monkeypatch.setattr(bench, "time_us", lambda fn, *a, **k: (fn(), 1.0)[1])
    small = [(17, 1032, 256, "tail"), (40, 64, 384, "G=128 x3")]
    monkeypatch.setattr(bench, "EDGE_CELLS", small)
    monkeypatch.setattr(bench, "QUICK_PROD", [(8, 16, 128, "tiny")])
    monkeypatch.setattr(bench, "CELL_SETS", {"prefill": small})
    return bench


def _args(**kw):
    base = dict(
        configs=None,
        seed=0,
        quick=True,
        cells="prefill",
        group_size=32,
        weight_type="uint4",
        cold=False,
        warmup=1,
        iters=1,
        baseline="prefill",
    )
    return argparse.Namespace(**{**base, **kw})


def test_check_passes_on_emulated_kernels(cpu_bench):
    records = cpu_bench.cmd_check(FakeLib(), _args())
    w4a8 = [r for r in records if r.get("check") == "w4a8"]
    assert w4a8 and all(r["pass"] and r["act_quant_exact"] for r in w4a8)
    assert records[0]["shuffle_ok"] and not records[0]["split_mismatches"]
    base = [r for r in records if r.get("check") == "w4a16_baseline"]
    assert all(r["vllm_int8_quant_match"] is True for r in base)
    assert all(r["w4a16_rel_l2"] < 1e-3 for r in base)


def test_check_flags_a_broken_kernel(cpu_bench, monkeypatch):
    lib = FakeLib()
    real = lib.gemm

    def off_by_one_group(*args, **kwargs):  # consume the wrong zero point
        real(*args[:9], args[9] + 1, *args[10:], **kwargs)

    monkeypatch.setattr(lib, "gemm", off_by_one_group)
    records = cpu_bench.cmd_check(lib, _args())
    assert not any(r["pass"] for r in records if r.get("check") == "w4a8")


@pytest.mark.parametrize("baseline", ["prefill", "exllama"])
def test_bench_reports_every_config(cpu_bench, baseline):
    records = cpu_bench.cmd_bench(FakeLib(), _args(baseline=baseline))
    assert len(records) == 2 * 2
    assert {r["config"] for r in records} == {"a16_lds_k32", "a8_smem_k32"}
    assert {r["w4a16_kernel"] for r in records} == {baseline}
    assert all(r["speedup_total"] == 0.5 for r in records)  # 1 / (1 + 1)


def test_device_problem_matches_reference_layouts(cpu_bench):
    p = ref.make_problem(8, 16, 64, 32, "uint4b8", seed=1)
    dp = cpu_bench.DeviceProblem(p, FakeOps)
    np.testing.assert_array_equal(dp.w.numpy().view(np.uint32), p.w_shuf)
    assert dp.use_v2_format is False
    a, a_scale, asum = dp.act_buffers(16)
    assert (a.numel(), a_scale.numel(), asum.numel()) == (16 * 64, 8, 16 * 2)


def test_fakequant_hooks_on_toy_mlp():
    """G1 hooks: only FFN inputs, only when enabled, counted, vLLM rounding."""
    from benchmarks.kernels.w4a8_sdot4_explore import fakequant_eval as fq

    torch.manual_seed(0)
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.gate_up_proj = torch.nn.Linear(64, 32, bias=False)
    model.mlp.down_proj = torch.nn.Linear(16, 64, bias=False)
    model.attn_proj = torch.nn.Linear(64, 64, bias=False)
    assert fq.install_hooks(model, fq.LAYERS, min_rows=4) == 2

    x = torch.randn(8, 64)
    w = model.mlp.gate_up_proj.weight
    plain = model.mlp.gate_up_proj(x)
    torch.testing.assert_close(plain, x @ w.T)  # disabled: untouched

    fq.set_fake_quant(model, True)
    got = model.mlp.gate_up_proj(x)
    torch.testing.assert_close(got, fq.fake_quant_per_token(x) @ w.T)
    model.mlp.gate_up_proj(x[:3])  # below min_rows: not quantized
    torch.testing.assert_close(model.attn_proj(x), x @ model.attn_proj.weight.T)
    assert fq.set_fake_quant(model, False) == 1

    q, s = ref.quantize_per_token(x.numpy())
    want = torch.from_numpy(q.astype(np.float32) * s[:, None])
    torch.testing.assert_close(fq.fake_quant_per_token(x), want, rtol=0, atol=0)
