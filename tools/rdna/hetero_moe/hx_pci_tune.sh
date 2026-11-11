#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Reversible runtime diagnostics of the hx ordering hunt (box rules: approved by the lead,
# only on the GPU endpoint functions of the lease the caller holds, never a switch port):
#
#   hx_pci_tune.sh [--dpm-high] [--ro-clear] [--dry-run] --record DIR --buses "d4 d9 ..."
#                  -- CMD [ARGS...]
#
#   --dpm-high  power_dpm_force_performance_level = high (pins SCLK/MCLK/FCLK/PCIe DPM)
#   --ro-clear  clear bit 4 (Enable Relaxed Ordering) of the PCIe Device Control word
#               (setpci CAP_EXP+8.w, mask write)
#   --dry-run   only record and print the plan (reads, no writes)
#
# Per GPU the script records the DPM level, the Device Control word, the lspci DevCtl lines
# and the AER counters under DIR/before.<bus>, applies the change, runs CMD, and in an EXIT
# trap restores the recorded values (the exact DevCtl word), records DIR/after.<bus> and
# diffs it with before. Any difference after the restore is reported as RESTORE MISMATCH
# and the script exits 7 (stop and report). Otherwise it exits with the status of CMD.
# Buses must be GPU endpoint functions (PCI class 0x03xxxx) of the six V620s:
# d1 d4 d9 dc df e2 (function 0). If the script is killed with SIGKILL nothing is restored:
# run it again with the same --record DIR and no CMD ("--restore") to restore from the record.
set -uo pipefail
export LC_ALL=C

dpm=0
ro=0
dry=0
restore_only=0
record=""
buses=""
cmd=()

die() {
    echo "hx_pci_tune: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dpm-high) dpm=1; shift ;;
        --ro-clear) ro=1; shift ;;
        --dry-run) dry=1; shift ;;
        --restore) restore_only=1; shift ;;
        --record) record=${2:?}; shift 2 ;;
        --buses) buses=${2:?}; shift 2 ;;
        --) shift; cmd=("$@"); break ;;
        -h | --help) sed -n '4,/^set -uo/p' "${BASH_SOURCE[0]}" | grep '^#' | cut -c3-; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done
[[ -n $record && -n $buses ]] || die "--record and --buses are required"
if ((restore_only)); then
    ((dpm == 0 && ro == 0 && ${#cmd[@]} == 0)) || die "--restore takes no other action"
else
    ((dpm || ro)) || die "nothing to do (--dpm-high and/or --ro-clear)"
    ((${#cmd[@]} || dry)) || die "no command after --"
fi

for b in $buses; do
    case $b in d1 | d4 | d9 | dc | df | e2) ;; *) die "bus $b is not one of the six V620s (d1 d4 d9 dc df e2)" ;; esac
    d=/sys/bus/pci/devices/0000:$b:00.0
    [[ -d $d ]] || die "no such device $d"
    [[ $(<"$d/class") == 0x03* ]] || die "$b:00.0 is not a display-class endpoint function"
done
mkdir -p "$record"

# snapshot TAG: DPM level, DevCtl word, lspci DevCtl lines and AER counters per bus
snapshot() {
    local tag=$1 b d
    for b in $buses; do
        d=/sys/bus/pci/devices/0000:$b:00.0
        {
            echo "dpm $(<"$d/power_dpm_force_performance_level")"
            echo "devctl $(sudo -n setpci -s "0000:$b:00.0" CAP_EXP+8.w)"
            sudo -n lspci -vv -s "0000:$b:00.0" | grep -E "RlxdOrd|MaxPayload.*MaxReadReq"
            cat "$d/aer_dev_correctable" "$d/aer_dev_nonfatal" "$d/aer_dev_fatal" 2>/dev/null
        } >"$record/$tag.$b"
    done
}

restored=0
rc=0
restore() {
    ((restored)) && return
    restored=1
    local b d dpm_old ctl_old
    for b in $buses; do
        d=/sys/bus/pci/devices/0000:$b:00.0
        dpm_old=$(awk '$1 == "dpm" {print $2}' "$record/before.$b")
        ctl_old=$(awk '$1 == "devctl" {print $2}' "$record/before.$b")
        if ((dpm || restore_only)); then
            echo "$dpm_old" | sudo -n tee "$d/power_dpm_force_performance_level" >/dev/null
        fi
        if ((ro || restore_only)); then
            sudo -n setpci -s "0000:$b:00.0" "CAP_EXP+8.w=$ctl_old"
        fi
    done
    snapshot after
    for b in $buses; do
        if ! diff -u "$record/before.$b" "$record/after.$b" >"$record/restore.$b.diff"; then
            echo "RESTORE MISMATCH bus $b (see $record/restore.$b.diff)"
            cat "$record/restore.$b.diff"
            rc=7
        fi
    done
    ((rc == 7)) || echo "hx_pci_tune: restored, before/after identical for: $buses"
}

if ((restore_only)); then
    for b in $buses; do [[ -s $record/before.$b ]] || die "no record for bus $b in $record"; done
    restore
    exit "$rc"
fi

snapshot before
for b in $buses; do
    echo "plan bus $b: $(tr '\n' ' ' <"$record/before.$b" | cut -c1-200)"
done
((dry)) && exit 0

trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for b in $buses; do
    d=/sys/bus/pci/devices/0000:$b:00.0
    if ((dpm)); then
        echo high | sudo -n tee "$d/power_dpm_force_performance_level" >/dev/null
        [[ $(<"$d/power_dpm_force_performance_level") == high ]] || die "bus $b: DPM level did not change"
    fi
    if ((ro)); then
        sudo -n setpci -s "0000:$b:00.0" CAP_EXP+8.w=0:10
        now=$(sudo -n setpci -s "0000:$b:00.0" CAP_EXP+8.w)
        ((16#$now & 16#10)) && die "bus $b: RlxdOrd still set (DevCtl $now)"
    fi
    echo "applied on bus $b: dpm=$(<"$d/power_dpm_force_performance_level") devctl=$(sudo -n setpci -s "0000:$b:00.0" CAP_EXP+8.w)"
done

"${cmd[@]}"
cmd_rc=$?
restore
((rc == 7)) && exit 7
exit "$cmd_rc"
