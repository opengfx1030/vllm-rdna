# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The metadata reader of isa_check, on a captured clang -S tail."""

from benchmarks.kernels.glm5next_rdna.isa_check import (
    markdown,
    parse_kernels,
    row,
    short_name,
    waves_per_simd,
)

SAMPLE = """
        .end_amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
    .group_segment_fixed_size: 516
    .kernarg_segment_align: 8
    .max_flat_workgroup_size: 1024
    .name: _ZN12_GLOBAL__N_131glm5_kda_recurrent_rdna2_kernelILNS_7ActKindE0EEEvPKv
    .private_segment_fixed_size: 0
    .sgpr_count:     68
    .sgpr_spill_count: 0
    .vgpr_count:     76
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .address_space:  global
    .group_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .name:           _ZN12glm5_dsa_idx25glm5_dsa_first_key_kernelEPK6__halfiPi
    .private_segment_fixed_size: 656
    .sgpr_count:     49
    .sgpr_spill_count: 0
    .vgpr_count:     256
    .vgpr_spill_count: 160
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
"""


def test_parse_kernels_reads_every_entry():
    kernels = parse_kernels(SAMPLE)
    assert [k["vgpr_count"] for k in kernels] == ["76", "256"]
    assert kernels[0]["group_segment_fixed_size"] == "516"
    assert kernels[1]["vgpr_spill_count"] == "160"


def test_short_name_strips_the_mangling():
    assert (
        short_name(
            "_ZN12_GLOBAL__N_131glm5_kda_recurrent_rdna2_kernelILNS_7ActKindE0EEEvPKv"
        )
        == "glm5_kda_recurrent_rdna2_kernel"
    )
    assert (
        short_name("_ZN12glm5_dsa_idx25glm5_dsa_first_key_kernelEPK6__halfiPi")
        == "glm5_dsa_first_key_kernel"
    )


def test_waves_per_simd_follows_the_gfx1030_vgpr_file():
    assert waves_per_simd(76) == 12  # 80 allocated -> 12 waves
    assert waves_per_simd(128) == 8
    assert waves_per_simd(256) == 4
    assert waves_per_simd(24) == 16  # capped


def test_markdown_table_has_one_row_per_kernel():
    rows = [row("tu", k) for k in parse_kernels(SAMPLE)]
    table = markdown(rows)
    assert table.count("\n") == 3
    assert (
        "| `tu` | `glm5_dsa_first_key_kernel` | 256 | 49 | 0 | 656 | 160/0 | 256 | 4 |"
        in table
    )
