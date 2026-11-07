# pciescope: PCIe fabric monitor and probe for par1-cs25 (2026-11-07)

Status: in progress on branch `feat/pciescope`. Core owns this document; the
Switch, Probe and TUI sections are written by their owners.

## Goal

Read and monitor the two Broadcom PEX88096 switches and the GPU links of
par1-cs25 while we work, and measure latency, bandwidth and effectiveness
(host to device, slot to slot inside a switch, between the switches). Surfaces:
a CLI with `--json` plus JSONL and Perfetto traces for agents, a TUI for
humans, and an MCP server.

## Fabric as discovered from sysfs

Filled from the live box by `python -m pciescope topo`.

## Usage

Filled when the CLI lands.

## Data model and metric definitions

Filled from `pciescope/model.py` (`METRICS`).

## Storage and traces

Filled when the store lands.

## MCP server

Filled when the MCP server lands.

## Switch registers (PEX88096)

Section owner: PlxSwitch.

## Active probe

Section owner: PcieProbe.

## TUI

Section owner: PcieTui.

## Evidence

Filled with real output from the box.
