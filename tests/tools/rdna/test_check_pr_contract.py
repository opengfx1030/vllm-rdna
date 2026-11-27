# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MOD_PATH = ROOT / "tools" / "rdna" / "check_pr_contract.py"
SPEC = importlib.util.spec_from_file_location("check_pr_contract", MOD_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules["check_pr_contract"] = MOD
SPEC.loader.exec_module(MOD)


VALID_RUNTIME_PR = """\
## Overview

Pure-HIP `gfx1030` prefill optimization (+36% cold prefill).

- **Duplicate-work check**: Checked open PRs; no conflict.
- **AI assistance**: Reviewed and validated on 4x V620 hardware.

## Claims

- Opt-in W8A8 prefill (`VLLM_RDNA_DENSE_W8A8_PREFILL=0` default).

## Validation

### Kernel-by-Kernel Before vs After

| Kernel | Before | After | Delta | ISA |
|---|---|---|---|---|
| `gemm` | 1.8ms | 0.6ms | 2.7x | 229 VGPR, 31 SGPR, 32K LDS, 0 spills, 0 scratch |

### How to Reproduce Locally

1. Incremental build: `cmake --build build --target _rocm_C`
2. Launch command: `bash tools/rdna/serve_rdna.sh flashnext` (`active`)
3. Run receipt: `QUALITY=1 FAIL_FAST=0 bash tools/rdna/pr_receipt.sh`

## Details

Gated behind `on_gfx10x()` and `#ifdef VLLM_ROCM_GFX1030`. Current stream.

## RDNA receipt: head123 against base456

| gate | result |
|---|---|
| token-exact | FAIL (lossy INT8: 12/16 equal base vs head; concurrency head PASS) |
| quality | PASS (KL within floor, PPL +0.48%) |
| overlap | PASS |
| stall, hard gate | PASS (0 starved decoders) |
| stall, relative metrics | ABBA: no measurable difference |
| speed | win (4k +36.4%) |
| added syncs and launches | none |
| boots | all exited 0 |

### Environment

4x Radeon PRO V620 (`gfx1030`), pinned 200 W / 2449 MHz.

### Change class and reach

- Change class: Lossy feature + kernel.
- `tools/rdna/archdiff.sh`: PASS (0 diffs outside `gfx1030`).

### Exactness

Greedy hashes differed on 4/16 prompts; head concurrency probe 0 FAIL.

### Quality

KL within noise floor; PPL +0.48%; passkey needle 3/3 -> 3/3 PASS (32k, 64k, 128k).

### Speed

| cell | TTFT |
|---|---|
| 1k/512 c=1 | -29% |
| 64k/1k c=1 | -24% |

### Concurrency and stalls

#### Overlap

c=8 scaling x5.1 PASS.

#### Stall probe

0 starved decoders across 3 repeats.

#### Added host syncs and launches

0 host syncs, 0 null-stream launches.

### Boots

All 4 ABBA boots + XA/XB exited 0.

### Raw results

`/home/chenco_adm/receipts/run1`
"""


def test_valid_runtime_csrc_pr_passes():
    errors, scope = MOD.validate_pr_contract(
        VALID_RUNTIME_PR, changed_files=["csrc/rocm/foo.cu", "vllm/bar.py"]
    )
    assert scope == "csrc + runtime"
    assert errors == []


def test_lossy_not_run_token_exact_fails_when_ready_and_advisory_in_draft(tmp_path):
    bad = VALID_RUNTIME_PR.replace(
        "FAIL (lossy INT8: 12/16 equal base vs head; concurrency head PASS)",
        "NOT RUN (lossy by design)",
    )
    body = tmp_path / "body.md"
    files = tmp_path / "files.txt"
    body.write_text(bad, encoding="utf-8")
    files.write_text("csrc/rocm/foo.cu\n", encoding="utf-8")

    assert MOD.main(["--body-file", str(body), "--files-file", str(files)]) == 1
    assert (
        MOD.main(["--body-file", str(body), "--files-file", str(files), "--draft"]) == 0
    )


def test_stall_not_run_fails_for_runtime_pr():
    bad = VALID_RUNTIME_PR.replace(
        "| stall, hard gate | PASS (0 starved decoders) |",
        "| stall, hard gate | NOT RUN |",
    )
    errors, _ = MOD.validate_pr_contract(bad, changed_files=["vllm/bar.py"])
    assert any("stall, hard gate" in e for e in errors)


def test_csrc_missing_isa_and_archdiff_fails():
    no_isa = VALID_RUNTIME_PR.replace(
        "229 VGPR, 31 SGPR, 32K LDS, 0 spills, 0 scratch", "faster"
    ).replace("archdiff.sh", "diff check")
    errors, _ = MOD.validate_pr_contract(no_isa, changed_files=["csrc/rocm/foo.cu"])
    assert any("ISA" in e for e in errors)
    assert any("archdiff" in e for e in errors)


def test_missing_duplicate_work_and_repro_fails():
    bad = VALID_RUNTIME_PR.replace("Duplicate-work check", "Summary note").replace(
        "How to Reproduce Locally", "Notes"
    )
    errors, _ = MOD.validate_pr_contract(bad, changed_files=["vllm/bar.py"])
    assert any("duplicate-work" in e for e in errors)
    assert any("How to Reproduce Locally" in e for e in errors)


def test_docs_tools_pr_allows_not_run_gpu_gates():
    docs_pr = """\
## Overview

Update contributing contract docs.

- **Duplicate-work check**: No overlapping open PR.
- **AI assistance**: Authored with Codex and reviewed locally.

## Claims

- Require token-exact measurement even for lossy PRs.

## Validation

`pre-commit` passed; `pytest` 73 passed.

## Details

Docs and tooling comments only.

## RDNA receipt: head against base

| gate | result |
|---|---|
| token-exact | NOT RUN (docs/tools only) |
| quality | NOT RUN |
| overlap | NOT RUN |
| stall, hard gate | NOT RUN |
| stall, relative metrics | NOT RUN |
| speed | NOT RUN |
| added syncs and launches | none |
| boots | NOT RUN |

### Change class and reach

Docs, tools, CI. No runtime code touched.
"""
    errors, scope = MOD.validate_pr_contract(
        docs_pr,
        changed_files=[".github/CONTRIBUTING.md", "tools/rdna/check_pr_contract.py"],
    )
    assert scope == "docs/tools/ci"
    assert errors == []


def test_unfilled_pr_template_fails():
    template = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
        encoding="utf-8"
    )
    errors, _ = MOD.validate_pr_contract(template, changed_files=["csrc/rocm/foo.cu"])
    assert len(errors) >= 6


def test_runtime_pr_missing_long_context_cell_fails():
    no_long = VALID_RUNTIME_PR.replace("| 64k/1k c=1 | -24% |\n", "")
    errors, _ = MOD.validate_pr_contract(no_long, changed_files=["vllm/bar.py"])
    assert any("long-context cell" in e for e in errors)


def test_runtime_pr_short_context_exemption_passes():
    short_ctx = VALID_RUNTIME_PR.replace(
        "| 64k/1k c=1 | -24% |",
        "long-context N/A (max-model-len <= 32k; PP=3 BF16 drafter headroom)",
    )
    errors, _ = MOD.validate_pr_contract(short_ctx, changed_files=["vllm/bar.py"])
    assert errors == []


def test_quality_gate_missing_passkey_needle_fails():
    no_needle = VALID_RUNTIME_PR.replace(
        "KL within noise floor; PPL +0.48%; "
        "passkey needle 3/3 -> 3/3 PASS (32k, 64k, 128k).",
        "KL within noise floor; PPL +0.48%.",
    )
    errors, _ = MOD.validate_pr_contract(no_needle, changed_files=["vllm/bar.py"])
    assert any("passkey needle" in e for e in errors)


def test_quality_not_run_does_not_require_passkey_needle():
    q_not_run = VALID_RUNTIME_PR.replace(
        "| quality | PASS (KL within floor, PPL +0.48%) |",
        "| quality | NOT RUN (pure scheduling fix; token-exact 26/26) |",
    ).replace(
        "KL within noise floor; PPL +0.48%; "
        "passkey needle 3/3 -> 3/3 PASS (32k, 64k, 128k).",
        "Not run (token-exact 26/26).",
    )
    errors, _ = MOD.validate_pr_contract(q_not_run, changed_files=["vllm/bar.py"])
    assert errors == []
