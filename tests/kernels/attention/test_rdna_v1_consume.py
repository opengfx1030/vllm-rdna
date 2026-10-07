# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hippihx V1 consume wiring. Host-only: no GPU, no kernel launch.

Needs the hippihx Python package and a built libhippihx_v1.so in
VLLM_HIPPIHX_LIB (a host-stub build is enough). Every hippihx op is
not ready yet, so these tests check loading, planning and the torch ->
V1 descriptor mapping, and that the extras kernel stays in charge.
"""

import os
import struct

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers import hippihx_v1

v1 = pytest.importorskip("hippihx.v1")
cv = pytest.importorskip("hippihx.v1_ctypes")

LIB = os.environ.get("VLLM_HIPPIHX_LIB")
pytestmark = pytest.mark.skipif(not LIB, reason="VLLM_HIPPIHX_LIB not set")

DECODE = dict(
    mode=0,
    head_dim=128,
    num_q_heads=32,
    num_kv_heads=8,
    block_size=16,
    kv_splits=16,
    sliding_window=0,
    causal=1,
    max_tokens=8,
    scale=128**-0.5,
)


@pytest.fixture
def rt():
    return hippihx_v1.HippihxRuntime(cv.load(LIB), "gfx1030")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(hippihx_v1, "_RUNTIME", None)
    monkeypatch.setattr(hippihx_v1, "_TRIED", False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)


def fa_tensors(tokens=8):
    return {
        "q": torch.zeros(tokens, 32, 128, dtype=torch.float16),
        "k_cache": torch.zeros(4, 8, 16, 16, 8, dtype=torch.float16),
        "v_cache": torch.zeros(4, 8, 16, 16, 8, dtype=torch.float16),
        # padded rows: a [:, :64] view of an [8, 80] table
        "block_table": torch.zeros(tokens, 80, dtype=torch.int32)[:, :64],
        "seq_lens": torch.ones(tokens, dtype=torch.int32),
        "out": torch.zeros(tokens, 32, 128, dtype=torch.float16),
    }


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_HIPPIHX", False)
    assert hippihx_v1.get_runtime() is None
    assert not hippihx_v1.enabled()
    t = fa_tensors()
    assert not hippihx_v1.fa_fdot2_decode(
        t["q"],
        t["k_cache"],
        t["v_cache"],
        t["block_table"],
        t["seq_lens"],
        t["out"],
        block_size=16,
        kv_splits=16,
        sliding_window=0,
        scale=128**-0.5,
    )


def test_load_needs_a_matching_code_object(monkeypatch, tmp_path):
    # EF_AMDGPU_MACH 0x36 = gfx1030. The host-stub library validates and
    # reports NO_HIP; a HIP build would hipModuleLoadData it.
    ident = b"\x7fELF" + bytes([2, 1, 1, 64, 3]) + bytes(7)
    elf = ident + struct.pack(
        "<HHIQQQIHHHHHH", 3, 224, 1, 0, 64, 0, 0x36, 64, 56, 0, 64, 0, 0
    )
    (tmp_path / "hippihx_gfx1030.hsaco").write_bytes(elf)
    monkeypatch.setattr(envs, "VLLM_HIPPIHX", True)
    monkeypatch.setattr(envs, "VLLM_HIPPIHX_LIB", LIB)
    monkeypatch.setattr(envs, "VLLM_HIPPIHX_CODE_OBJECT", str(tmp_path))
    monkeypatch.setattr(hippihx_v1, "_device_arch", lambda: "gfx1030")
    lib = cv.load(LIB)
    path = hippihx_v1._code_object_path("gfx1030").encode()
    assert v1.V1Status(lib.hippihx_v1_load(b"gfx1030", path)) in (
        v1.V1Status.OK,
        v1.V1Status.ERR_NO_HIP,
    )
    assert (
        v1.V1Status(lib.hippihx_v1_load(b"gfx1100", path))
        is v1.V1Status.ERR_FOREIGN_ISA
    )
    if v1.V1Status(lib.hippihx_v1_load(b"gfx1030", path)) is v1.V1Status.ERR_NO_HIP:
        assert hippihx_v1.get_runtime() is None


def test_fa_decode_plans_but_is_not_ready(rt):
    status, plan = rt.plan_raw("attention.fa_fdot2", torch.float16, **DECODE)
    assert status is v1.V1Status.OK
    assert plan.ready == 0
    assert plan.scratch_nbytes > 0  # fp32 split-K partials
    assert (
        rt.plan("attention.fa_fdot2", torch.float16, torch.device("cpu"), **DECODE)
        is None
    )
    status, _ = rt.plan_raw("attention.fa_fdot2", torch.bfloat16, **DECODE)
    assert status is v1.V1Status.ERR_UNSUPPORTED_DTYPE  # never fdot2.bf16


def test_torch_views_map_to_valid_descriptors(rt):
    spec = v1.find_spec("attention.fa_fdot2")
    status, plan = rt.plan_raw("attention.fa_fdot2", torch.float16, **DECODE)
    assert status is v1.V1Status.OK
    scratch = hippihx_v1.aligned_zeros(plan.scratch_nbytes, v1.SCRATCH_ALIGN, "cpu")
    assert scratch.data_ptr() % v1.SCRATCH_ALIGN == 0
    good = fa_tensors()
    # All checks pass; no body is linked, so run says NOT_READY.
    assert rt.run_raw(plan, spec, good, scratch) is v1.V1Status.ERR_NOT_READY
    bad = dict(good, q=torch.zeros(32, 8, 128, dtype=torch.float16).transpose(0, 1))
    assert rt.run_raw(plan, spec, bad, scratch) is v1.V1Status.ERR_TENSOR
    wide = dict(good, q=torch.zeros(9, 32, 128, dtype=torch.float16))
    assert (
        rt.run_raw(plan, spec, wide, scratch) is v1.V1Status.ERR_TENSOR
    )  # > max_tokens
    assert rt.run_raw(plan, spec, good, None) is v1.V1Status.ERR_SCRATCH
    assert rt.run_raw(plan, spec, good, scratch[8:]) is v1.V1Status.ERR_SCRATCH


def test_bucket():
    assert [hippihx_v1._bucket(n) for n in (1, 2, 3, 8, 9)] == [1, 2, 4, 8, 16]
