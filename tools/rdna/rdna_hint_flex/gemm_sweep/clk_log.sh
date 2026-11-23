#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Sample the graphics clock, socket power, throttle status and hotspot
# temperature of one V620 while a benchmark runs (amd-smi numbers GPUs by PCI
# order, so on <build-host> amd-smi index 5 is bus e2 = HIP 5; check
# `amd-smi list` before using another index).
#
#   bash clk_log.sh <amd-smi gpu index> <samples> <interval_s> > clk.log
#
# One line per sample: time power_W gfx_MHz throttle hotspot_C.
set -euo pipefail
gpu=${1:?amd-smi gpu index}
n=${2:-10}
dt=${3:-1}
for _ in $(seq 1 "$n"); do
  pw=$(amd-smi metric -g "$gpu" --clock --power --csv 2>/dev/null | python3 -c '
import csv, sys
rows = list(csv.DictReader(sys.stdin))
r = rows[-1] if rows else {}
print(r.get("socket_power", ""), r.get("gfx_0_clk", ""), r.get("throttle_status", ""))
')
  tp=$(amd-smi metric -g "$gpu" --temperature --csv 2>/dev/null | python3 -c '
import csv, sys
rows = list(csv.DictReader(sys.stdin))
print(rows[-1].get("hotspot", "") if rows else "")
')
  read -r p c t <<<"$pw"
  echo "$(date +%T) power_W $p gfx_MHz $c throttle $t hotspot_C $tp"
  sleep "$dt"
done
