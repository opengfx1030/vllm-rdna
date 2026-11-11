# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Classify kernel-log PCIe/AER events into corrected-link noise and fatal faults.

The raw qualification gate in qualify_serving.py matches every AER / "Hardware
Error" line, so a *corrected* link-layer event fails a run even though the
kernel states the error "has been corrected by h/w and requires no further
action". This module separates the two classes using the kernel's own severity
classification, so corrected-link noise can be reported separately from real
device faults without weakening the raw gate.

Raw evidence is never rewritten: qualify_serving.py keeps its raw status
semantics and writes the classification to a separate artifact.
"""

from typing import Any

import regex as re

# The kernel states the recovery outcome explicitly. Trust it.
CORRECTED_MARKERS = re.compile(
    r"It has been corrected by h/w|"
    r"\[Hardware Error\]:\s*event severity:\s*corrected|"
    r"Error\s+\d+,\s*type:\s*corrected",
    re.I,
)

# Genuinely fatal device-level signatures. Note "uncor_status" (a register dump
# that also appears inside corrected groups) is deliberately NOT matched.
FATAL_MARKERS = re.compile(
    r"event severity:\s*(?:fatal|uncorrected)|"
    r"Error\s+\d+,\s*type:\s*(?:fatal|uncorrected)|"
    r"\[Hardware Error\]:\s*Uncorrected|"
    r"amdgpu.*(?:fault|timeout|reset)|"
    r"UTCL2|"
    r"data fabric sync flood|"
    r"Previous system reset reason|"
    r"HSA_STATUS_ERROR|"
    r"page fault|Segmentation fault|core dumped|"
    r"ring timeout|GPU reset",
    re.I,
)

# Mirrors the raw gate in qualify_serving.py: anything the raw gate matches
# must be classified, never silently reported as clean.
RAW_MARKERS = re.compile(
    r"amdgpu.*(?:fault|timeout|reset)|UTCL2|AER|Hardware Error", re.I
)

GROUP = re.compile(
    r"\[\s*(?P<ts>[\d.]+)\]\s*\{(?P<index>\d+)\}\[Hardware Error\]:\s*"
    r"event severity:\s*(?P<severity>corrected|fatal|uncorrected|recoverable)",
    re.I,
)
ERROR_LINE = re.compile(
    r"\[\s*(?P<ts>[\d.]+)\].*\[Hardware Error\]:\s*Error\s+(?P<no>\d+),"
    r"\s*type:\s*(?P<type>\w+)"
)
ENDPOINT = re.compile(
    r"(?P<kind>pcieport|amdgpu)\s+(?P<bdf>[0-9a-f:.]+):\s*"
    r"aer_status:\s*(?P<status>0x[0-9a-f]+)"
)
STATUS_LABEL = re.compile(r"\[\s*(?P<bit>\d+)\]\s*(?P<label>[A-Za-z ]+?)\s*$")
RECOVERY_OVERFLOW = re.compile(
    r"AER: buffer overflow in recovery for\s+(?P<bdf>[0-9a-f:.]+)"
)

SEVERITY_FATAL = {"fatal", "uncorrected"}


def classify(text: str) -> dict[str, Any]:
    """Classify a kernel-log delta into corrected-link events and fatal faults."""
    groups: list[dict[str, Any]] = []
    by_key: dict[Any, dict[str, Any]] = {}
    for match in GROUP.finditer(text):
        key = (match.group("index"), match.group("ts"))
        if key in by_key:
            continue
        group = {
            "index": int(match.group("index")),
            "timestamp": float(match.group("ts")),
            "severity": match.group("severity").lower(),
            "error_count": 0,
            "error_types": [],
            "endpoints": [],
        }
        by_key[key] = group
        groups.append(group)

    # The severity banner and the per-error lines carry different microsecond
    # timestamps, so attribution is positional: an error belongs to the most
    # recent severity banner that precedes it.
    ordered = sorted(
        [(m.start(), "group", m) for m in GROUP.finditer(text)]
        + [(m.start(), "error", m) for m in ERROR_LINE.finditer(text)]
    )
    current: dict[str, Any] | None = None
    for _, kind, match in ordered:
        if kind == "group":
            key = (match.group("index"), match.group("ts"))
            current = by_key.get(key)
            continue
        if current is None:
            continue
        current["error_count"] += 1
        current["error_types"].append(match.group("type").lower())

    endpoints: list[dict[str, Any]] = []
    for match in ENDPOINT.finditer(text):
        endpoint = {
            "kind": match.group("kind"),
            "bdf": match.group("bdf"),
            "aer_status": match.group("status"),
            "labels": [],
        }
        endpoints.append(endpoint)
    for endpoint in endpoints:
        # Attach the per-endpoint status labels ("[ 6] BadTLP", "[13] NonFatalErr").
        pattern = re.compile(
            re.escape(endpoint["kind"])
            + r"\s+"
            + re.escape(endpoint["bdf"])
            + r":\s+(?P<body>.*)"
        )
        for line in text.splitlines():
            found = pattern.search(line)
            if not found:
                continue
            for label in STATUS_LABEL.finditer(found.group("body")):
                endpoint["labels"].append(label.group("label").strip())

    fatal_lines = [line for line in text.splitlines() if FATAL_MARKERS.search(line)]
    corrected_groups = [g for g in groups if g["severity"] == "corrected"]
    fatal_groups = [g for g in groups if g["severity"] in SEVERITY_FATAL]
    # Safety net: the raw gate matches more than the severity banners. If the
    # raw gate fired but no severity group was recognised, the event is
    # unclassified - never report "clean" in that case.
    raw_lines = [line for line in text.splitlines() if RAW_MARKERS.search(line)]

    if fatal_lines or fatal_groups:
        verdict = "fatal"
    elif corrected_groups:
        verdict = "corrected-only"
    elif raw_lines:
        verdict = "unclassified"
    else:
        verdict = "clean"

    nonfatal = sum(
        1 for e in endpoints if any("NonFatalErr" in label for label in e["labels"])
    )
    bad_tlp = sum(
        1 for e in endpoints if any("BadTLP" in label for label in e["labels"])
    )
    return {
        "verdict": verdict,
        "event_groups": groups,
        "corrected_event_groups": len(corrected_groups),
        "corrected_error_count": sum(g["error_count"] for g in corrected_groups),
        "fatal_event_groups": len(fatal_groups),
        "fatal_events": fatal_lines,
        "endpoints": endpoints,
        "endpoints_with_nonfatal_err": nonfatal,
        "endpoints_with_bad_tlp": bad_tlp,
        "aer_recovery_buffer_overflows": [
            m.group("bdf") for m in RECOVERY_OVERFLOW.finditer(text)
        ],
        "corrected_marker_count": len(CORRECTED_MARKERS.findall(text)),
        "raw_gate_matched_lines": raw_lines,
        "interpretation": (
            "All PCIe/AER activity was classified by the kernel as corrected "
            "('It has been corrected by h/w'); no device fault, timeout or reset "
            "was reported."
            if verdict == "corrected-only"
            else "Fatal device events present; hardware attribution requires "
            "investigation."
            if verdict == "fatal"
            else "No PCIe/AER activity observed in this delta."
        ),
    }


def summarize(report: dict[str, Any]) -> str:
    return (
        f"{report['verdict']}: {report['corrected_event_groups']} corrected group(s), "
        f"{report['corrected_error_count']} corrected error(s), "
        f"{report['fatal_event_groups']} fatal group(s), "
        f"{len(report['fatal_events'])} fatal line(s)"
    )


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        raise SystemExit("usage: kernel_events.py <kernel-delta.log> [...]")
    out = []
    for name in sys.argv[1:]:
        path = Path(name)
        report = classify(path.read_text(errors="replace"))
        report["source"] = str(path)
        out.append(report)
    print(json.dumps(out, indent=2))
