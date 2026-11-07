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

`pciescope tui` is a live terminal dashboard (Textual) over the passive
collectors, the recorded sessions and the active probe. It is built for an ssh
pty at 80x24 and uses the extra room of a larger terminal for wider
sparklines. Only one view is shown at a time (keys `1`-`5`).

```bash
python -m pciescope tui                       # live, records a session
python -m pciescope tui --no-record           # live, nothing written
python -m pciescope tui --replay latest       # replay (dir, name or latest)
python -m pciescope tui --replay <dir> --speed 8
ssh -tt par1-cs25 'cd ~/Apps/pciescope-src && PYTHONPATH=tools/rdna \
  ~/Apps/pciescope-venv/bin/python -m pciescope tui'
```

Flags: `--replay SESSION`, `--interval S` (sampling period, default 1.0),
`--collectors a,b` (default: the passive set), `--session-dir DIR`,
`--sysfs-root DIR` (a sysfs tree for dry runs), `--owner NAME` (lease owner for
the measure panel, default `$PCIESCOPE_OWNER`), `--no-record`, `--dry-run`
(measure panel plans and validates but launches nothing), `--speed X` (replay).
The TUI is passive: it reads sysfs/procfs only and writes nothing to the
switches or devices; the only thing that can touch a GPU is the measure panel,
and only under a lease held by `--owner`.

### Views

| Key | View | Content |
| --- | --- | --- |
| `1` | Topology | root complexes, switches (slot A/B, link), ports, GPUs (HIP, KFD node, arch, link speed x width, state colour), management endpoints, unused ports |
| `2` | Live | per GPU RX/TX GB/s, sparkline, busy, temperature, VRAM; per switch uplink h2d/d2h and utilisation; the collectors that run, their per-tick cost and those that are off |
| `3` | Matrix | latest probe results as a heatmap with efficiency, intra-switch versus cross-root summaries |
| `4` | Errors | AER counter deltas, PLX error registers, DPC and link events with timestamps, and the number of standing non-zero counters |
| `5` | Measure | starts a probe suite on leased GPUs and streams its progress |

The header shows the mode (`LIVE` or `REPLAY`), the recording indicator
(`● REC <session dir>`, or `not recording`), the clock, and `frozen` while the
display is paused. The last line lists the keys of the current view.

### Keys

| Key | Action |
| --- | --- |
| `1`-`5`, `tab`, `shift+tab` | switch view |
| `space` | live: freeze the display (recording continues); replay: play/pause |
| `r` | live: stop/resume recording (a new session is created when started without one) |
| `e` | export the session as a Perfetto trace, `<session>/trace.perfetto.json` |
| `left`, `right` | replay: seek -/+ 10 s (`shift`: 60 s); `home`, `end`: start/end |
| `+`, `-` | replay: speed x2 / x0.5 |
| `up`, `down` | scroll; on the measure view, select the suite |
| `d` | topology: detail chain (root, switch, port, GPU) |
| `m`, `[`, `]`, `v` | matrix: metric (bandwidth, efficiency, latency), transfer size, direction |
| `g` | measure: choose the GPU set |
| `s`, `x` | measure: start / cancel the suite |
| `c` | errors: clear the list (counters keep their baseline) |
| `?` | help; `q` quit |

### What "n/a" means

* RX/TX GB/s is `n/a` when no throughput counter exists. The amdgpu `pcie_bw`
  sysfs file is not exposed on RDNA2/RDNA3 (neither the V620 nor the W7800 on
  this box has it), and the PLX per-port counters are off until the switch
  register writes are approved (collector `plx`, see the switch section). The
  Live view says so and falls back to the GPU busy percentage for the history.
* The Matrix is empty until a session holds probe measurements: run the
  measure panel or load a session that has them.
* Errors are changes, not levels: a counter that is non-zero but constant is
  counted as "standing" and listed, a counter that moves becomes a timestamped row.

### Replay

Sessions are deduplicated (a series is stored when it changes, plus a 30 s
heartbeat); the TUI holds each series at its last value between records, so a
replayed sparkline looks like the live one. Seeking back rebuilds the view from
the start of the session. `--replay` needs `topology.json` in the session.

### Measure panel

The suites and the lease check come from the probe package
(`list_suites`, `check_leases`, `run_suite`). The panel checks the leases for
the chosen GPUs first, shows who holds them when they are missing, and refuses
to start (nothing is launched); HIP 5 (bus 07) is never offered. Progress
events (plan, step, guard, abort) stream into the panel and every measurement
is written to the recording and lands in the Matrix view as it completes. `x`
cancels between steps.

### Tests

`tests/tools/rdna/pciescope/test_pciescope_tui_*.py` need no GPU. The render
functions are pure and checked against golden screens in `golden_tui/`
(80x24 screens, plus 80/120/160 column renders), the app with Textual's pilot, and the replay against a real
34 s passive session recorded on the box (`fixtures/tui_real_session_*`).
`PCIESCOPE_UPDATE_GOLDEN=1` rewrites the goldens.

## Evidence

Filled with real output from the box.
