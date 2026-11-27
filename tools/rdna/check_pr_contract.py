#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate RDNA fork pull request bodies against .github/CONTRIBUTING.md.

Checks standardized section presence, duplicate-work and AI-assistance notes,
local reproduction steps, kernel ISA tables and archdiff results for csrc
changes, and the 8-row RDNA receipt gate table (requiring measured token-exact
and hard stall gates on Ready runtime PRs, including known-lossy techniques).

    python3 tools/rdna/check_pr_contract.py --body-file pr.md \
        [--files-file files.txt] [--draft]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re

REQUIRED_GATES = (
    "token-exact",
    "quality",
    "overlap",
    "stall, hard gate",
    "stall, relative metrics",
    "speed",
    "added syncs and launches",
    "boots",
)

REQUIRED_RUNTIME_RECEIPT_SUBSECTIONS = (
    "environment",
    "exactness",
    "quality",
    "speed",
    "concurrency and stalls",
    "overlap",
    "stall probe",
    "added host syncs and launches",
    "boots",
    "raw results",
)

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(r"^(#{2,4})\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
NOT_RUN_RE = re.compile(r"^not\s+run\b", re.IGNORECASE)
CSRC_PATH_RE = re.compile(r"^(csrc/|CMakeLists\.txt$)")
RUNTIME_PATH_RE = re.compile(r"^(csrc/|CMakeLists\.txt$|vllm/)")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")
LONG_CTX_CELL_RE = re.compile(r"\b(64k|65536|128k|131072|256k|262144)\b", re.IGNORECASE)
LONG_CTX_NA_RE = re.compile(
    r"\b(long[- ]context\s*[:\-]?\s*n/a|short[- ]context\s+only|"
    r"max[-_ ]model[-_ ]len\s*(?:<|<=|=)\s*(?:32768|32k|49152|48k))\b",
    re.IGNORECASE,
)
NEEDLE_RE = re.compile(r"\b(needle|passkey)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    level: int
    title: str
    raw_body: str
    clean_body: str


def strip_comments(text: str) -> str:
    """Remove HTML comments and outer whitespace."""
    return COMMENT_RE.sub("", text).strip()


def parse_sections(markdown: str) -> list[Section]:
    """Parse ##, ###, #### markdown sections while ignoring fenced code blocks."""
    lines = markdown.splitlines()
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    fence_marker = ""

    for idx, line in enumerate(lines):
        m_fence = FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif line.lstrip().startswith(fence_marker * 3):
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        m_head = HEADING_RE.match(line)
        if m_head:
            headings.append((idx, len(m_head.group(1)), m_head.group(2).strip()))

    sections: list[Section] = []
    for pos, (line_idx, level, title) in enumerate(headings):
        end_idx = len(lines)
        for next_line_idx, next_level, _ in headings[pos + 1 :]:
            if next_level <= level:
                end_idx = next_line_idx
                break
        raw_body = "\n".join(lines[line_idx + 1 : end_idx])
        sections.append(
            Section(
                level=level,
                title=title,
                raw_body=raw_body,
                clean_body=strip_comments(raw_body),
            )
        )
    return sections


def find_section(
    sections: Sequence[Section],
    *,
    level: int | None = None,
    exact_title: str | None = None,
    pattern: str | None = None,
) -> Section | None:
    """Return the first matching section by level and exact title or regex."""
    regex = re.compile(pattern, re.IGNORECASE) if pattern else None
    target = exact_title.lower() if exact_title else None
    for sec in sections:
        if level is not None and sec.level != level:
            continue
        if target is not None and sec.title.strip().lower() != target:
            continue
        if regex is not None and not regex.search(sec.title.strip()):
            continue
        return sec
    return None


def parse_gate_table(receipt_raw_body: str) -> dict[str, str]:
    """Extract `{gate_name_lower: cell_value}` from the RDNA receipt gate table."""
    gates: dict[str, str] = {}
    for line in receipt_raw_body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or TABLE_SEP_RE.match(stripped):
            continue
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        if len(parts) < 2:
            continue
        key = strip_comments(parts[0]).lower()
        val = strip_comments(parts[1])
        if key == "gate":
            continue
        if key in REQUIRED_GATES:
            gates[key] = val
    return gates


def classify_files(
    paths: Sequence[str] | None, *, docs_only: bool = False
) -> tuple[bool, bool]:
    """Return `(touches_runtime, touches_csrc)` from changed file paths."""
    if docs_only:
        return False, False
    if paths is None:
        return True, True
    cleaned = [p.strip().lstrip("./") for p in paths if p.strip()]
    touches_csrc = any(CSRC_PATH_RE.match(p) for p in cleaned)
    touches_runtime = touches_csrc or any(RUNTIME_PATH_RE.match(p) for p in cleaned)
    return touches_runtime, touches_csrc


def _has_kernel_isa_evidence(sections: Sequence[Section], validation_text: str) -> bool:
    """Check Validation for kernel before/after + ISA table (or explicit N/A)."""
    kernel_sec = find_section(sections, pattern=r"kernel")
    if kernel_sec and kernel_sec.clean_body:
        text = kernel_sec.clean_body
        if re.search(r"\bn/a\b", text, re.IGNORECASE):
            return True
        if "|" in text and re.search(r"(vgpr|sgpr|lds|spill|scratch)", text, re.I):
            return True
    if re.search(r"n/a\s*\(\s*no kernel", validation_text, re.IGNORECASE):
        return True
    return "|" in validation_text and bool(
        re.search(r"(vgpr|sgpr|lds|spill|scratch)", validation_text, re.I)
    )


def validate_pr_contract(
    body: str,
    *,
    changed_files: Sequence[str] | None = None,
    docs_only: bool = False,
) -> tuple[list[str], str]:
    """Validate PR body against the RDNA contribution contract.

    Returns `(errors, scope_label)`.
    """
    touches_runtime, touches_csrc = classify_files(changed_files, docs_only=docs_only)
    if touches_csrc:
        scope = "csrc + runtime"
    elif touches_runtime:
        scope = "runtime (vllm/)"
    else:
        scope = "docs/tools/ci"

    sections = parse_sections(body)
    errors: list[str] = []

    overview = find_section(sections, level=2, exact_title="overview")
    if overview is None or not overview.clean_body:
        errors.append("missing or empty `## Overview` section")
    else:
        low_ov = overview.clean_body.lower()
        if "duplicate" not in low_ov:
            errors.append(
                "`## Overview` must include a duplicate-work check against open PRs"
            )
        if "ai assistance" not in low_ov:
            errors.append("`## Overview` must include an AI assistance disclosure")

    for name in ("claims", "validation", "details"):
        sec = find_section(sections, level=2, exact_title=name)
        if sec is None or not sec.clean_body:
            errors.append(f"missing or empty `## {name.title()}` section")

    validation = find_section(sections, level=2, exact_title="validation")
    receipt = find_section(sections, level=2, pattern=r"^rdna receipt\b")
    if receipt is None or not receipt.clean_body:
        errors.append("missing or empty `## RDNA receipt` section")
        return errors, scope

    gates = parse_gate_table(receipt.raw_body)
    for gate in REQUIRED_GATES:
        val = gates.get(gate, "")
        if not val or val == "-":
            errors.append(f"RDNA receipt gate table missing non-empty `{gate}` row")

    change_class = find_section(sections, exact_title="change class and reach")
    if change_class is None or not change_class.clean_body:
        errors.append("missing or empty `### Change class and reach` subsection")

    if touches_runtime:
        repro = find_section(sections, pattern=r"reproduce")
        if repro is None or not repro.clean_body:
            errors.append(
                "missing or empty `### How to Reproduce Locally` under `## Validation`"
            )

        for sub in REQUIRED_RUNTIME_RECEIPT_SUBSECTIONS:
            sec = find_section(sections, exact_title=sub)
            if sec is None or not sec.clean_body:
                errors.append(f"missing or empty RDNA receipt subsection `### {sub}`")

        token_exact = gates.get("token-exact", "")
        if token_exact and NOT_RUN_RE.match(token_exact):
            errors.append(
                "`token-exact` gate cannot be `NOT RUN` on runtime changes "
                "(even for known-lossy techniques, run with `EXACT=1 FAIL_FAST=0`)"
            )

        stall_hard = gates.get("stall, hard gate", "")
        if stall_hard and NOT_RUN_RE.match(stall_hard):
            errors.append(
                "`stall, hard gate` cannot be `NOT RUN` on runtime changes "
                "(run `stall_probe.py` / `pr_receipt.sh` with `STALL=1`)"
            )

        speed_sec = find_section(sections, exact_title="speed")
        if (
            speed_sec is not None
            and speed_sec.clean_body
            and not (
                LONG_CTX_CELL_RE.search(speed_sec.clean_body)
                or LONG_CTX_NA_RE.search(speed_sec.clean_body)
            )
        ):
            errors.append(
                "`### Speed` must include at least one long-context cell "
                "(`>= 64k`, e.g. `64k/1k` or `128k/512` via `LONG_CTX=auto`) "
                "or an explicit `long-context N/A` reason (`max-model-len <= 32k`)"
            )

        quality_gate = gates.get("quality", "")
        quality_sec = find_section(sections, exact_title="quality")
        if (
            quality_gate
            and not NOT_RUN_RE.match(quality_gate)
            and quality_sec is not None
            and quality_sec.clean_body
            and not (
                NEEDLE_RE.search(quality_sec.clean_body)
                or LONG_CTX_NA_RE.search(quality_sec.clean_body)
            )
        ):
            errors.append(
                "`### Quality` must report long-context passkey needle results "
                "(`quality_ab.py --needle-lens`) when the quality gate is run "
                "(or an explicit `long-context N/A` reason)"
            )

    if touches_csrc:
        val_text = validation.clean_body if validation else ""
        if not _has_kernel_isa_evidence(sections, val_text):
            errors.append(
                "`csrc/` changes must include a kernel before/after table with "
                "compiled ISA counts (VGPR, SGPR, LDS, spills, scratch) or `N/A`"
            )
        reach_text = change_class.clean_body if change_class else ""
        has_archdiff = (
            "archdiff" in reach_text.lower() or "archdiff" in val_text.lower()
        )
        if not has_archdiff:
            errors.append(
                "`csrc/` changes must report `tools/rdna/archdiff.sh` results in "
                "`### Change class and reach` or `## Validation`"
            )

    return errors, scope


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate PR body against the RDNA fork contributing contract."
    )
    parser.add_argument(
        "--body-file", required=True, type=Path, help="PR markdown body"
    )
    parser.add_argument(
        "--files-file",
        type=Path,
        default=None,
        help="Optional newline-separated list of changed repo-relative paths",
    )
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Treat PR as docs/tools/CI only when --files-file is omitted",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Advisory mode for Draft PRs (print findings but exit 0)",
    )
    args = parser.parse_args(argv)

    body = args.body_file.read_text(encoding="utf-8", errors="replace")
    changed_files: list[str] | None = None
    if args.files_file is not None:
        changed_files = args.files_file.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()

    errors, scope = validate_pr_contract(
        body, changed_files=changed_files, docs_only=args.docs_only
    )
    print(f"pr-contract: scope={scope}, draft={args.draft}")
    if not errors:
        print("pr-contract: PASS")
        return 0

    for err in errors:
        print(f"pr-contract: error: {err}", file=sys.stderr)
    if args.draft:
        print("pr-contract: ADVISORY (draft PR; ready-for-review would fail)")
        return 0
    print(f"pr-contract: FAIL ({len(errors)} violation(s))", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
