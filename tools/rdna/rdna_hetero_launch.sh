#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Hetero-MoE launcher stanza, sourced by serve_rdna.sh when VLLM_HETERO_MOE=1
# (docs/rdna2/hetero-moe-design-2026-11-07.md sections 6.10, 6.11 and C.8).
#
# One island = a hub engine (the recipe's vLLM command, bound to its own
# PCI_BUSES) plus one expert-server process per expert device of the plan, each
# on exactly one V620. The launcher starts the servers FIRST, waits until every
# server has written its ready file, runs the hub as a child, supervises the
# group and tears everything down together. The shell stays the supervisor
# (there is no exec in hetero mode).
#
#   rdna_hetero_prepare HUB_HIPS HUB_BUSES   validate the plan, resolve the
#                                            server buses, build the commands
#   rdna_hetero_print                        the PRINT=1 stanza
#   rdna_hetero_launch                       start / wait / supervise / tear down
#
# Everything is driven by the plan and by islands: the number of hubs, islands
# and servers comes from `cli.py hubs|servers PLAN`, never from a constant.
#
# Knobs (shell variables, not exported):
#   HETERO_PYTHON          python that runs the plan CLI (numpy only; default
#                          $VENV/bin/python)
#   HETERO_LOG_DIR         server logs and ready files
#                          (default ${RUN_CWD:-<tree>/cache/run}/hetero)
#   HETERO_READY_TIMEOUT   seconds to wait for every ready file (default 900)
#   HETERO_TERM_TIMEOUT    seconds a SIGTERM may take, per stage (default 30)
#   HETERO_SERVER_ARGS     extra server CLI arguments, word-split
#   HETERO_POLL_INTERVAL   supervisor poll period in seconds (default 1)
#
# The supervisor is injectable (tests): launch uses
#   rdna_hetero_sids=(id ...)              server names
#   rdna_hetero_slogs=(file ...)           server log files
#   rdna_hetero_sready=(file ...)          server ready files
#   rdna_hetero_server_cmd_<i>=(word ...)  server i command
#   rdna_hetero_hub_cmd=(word ...)         hub command (LAUNCH_WRAPPER applies)

# Server CLI contract (vllm.distributed.hetero_moe.server).
RDNA_HETERO_SERVER_MODULE=vllm.distributed.hetero_moe.server
RDNA_HETERO_MAX_ROWS=2048

rdna_hetero_sids=()
rdna_hetero_slogs=()
rdna_hetero_sready=()
rdna_hetero_hub_cmd=()
rdna_hetero_spids=()
rdna_hetero_hub_pid=""
rdna_hetero_torn_down=0
rdna_hetero_unresolved='(not resolvable on this host; resolved at launch)'

rdna_hetero_log() { printf '[hetero] %s\n' "$*" >&2; }

# Shell-quoted words, safe tokens left bare so the command reads as typed.
rdna_hetero_words() {
    local w out=""
    for w in "$@"; do
        if [[ $w =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then
            out+="$w "
        else
            out+="'${w//\'/\'\\\'\'}' "
        fi
    done
    printf '%s' "${out% }"
}

# --- prepare ----------------------------------------------------------------

# rdna_hetero_plan_cli ARGS...: the plan CLI (numpy + regex only, no vLLM
# import, so it cannot wake a GPU).
rdna_hetero_plan_cli() {
    "$rdna_hetero_py" "$rdna_hetero_cli" "$@"
}

# rdna_hetero_prepare [HUB_HIPS [HUB_BUSES]]
#   HUB_HIPS   the hub engine's HIP indices when known (resolved or explicit)
#   HUB_BUSES  the hub engine's PCI buses when it is bound by PCI_BUSES
# Needs from serve_rdna.sh: source_dir VENV MODEL TP PP (after rdna_init).
# shellcheck disable=SC2154 # source_dir, TP, MODEL come from serve_rdna.sh / rdna_init
rdna_hetero_prepare() {
    local hub_hips=${1:-} hub_buses=${2:-}
    local hint='generate one with: python vllm/distributed/hetero_moe/cli.py stripe --out <plan> (flags: --hub-hips --expert-hips --stripe --hub-arch)'
    local plan=${VLLM_HETERO_MOE_PLAN:-}
    [[ -n $plan ]] || rdna_die "VLLM_HETERO_MOE=1 needs VLLM_HETERO_MOE_PLAN=<plan.json>; $hint"
    [[ -f $plan && -r $plan ]] || rdna_die "VLLM_HETERO_MOE_PLAN='$plan' is not a readable file; $hint"
    [[ $plan == /* ]] || plan=$PWD/$plan
    export VLLM_HETERO_MOE_PLAN=$plan

    rdna_hetero_py=${HETERO_PYTHON:-$VENV/bin/python}
    rdna_hetero_cli=$source_dir/vllm/distributed/hetero_moe/cli.py
    [[ -f $rdna_hetero_cli ]] || rdna_die "plan CLI missing: $rdna_hetero_cli"

    local out
    if ! out=$(rdna_hetero_plan_cli validate "$plan" 2>&1); then
        rdna_die "plan '$plan' is not usable (checked with $rdna_hetero_py; set HETERO_PYTHON=<python with numpy and regex> if that interpreter cannot run the plan CLI):
$out"
    fi
    rdna_hetero_plan_desc=$out

    # Roles: one line per device, tab separated: island id hip switch arch.
    local -a lines=()
    local island id hip arch
    rdna_hetero_hub_ids=()
    rdna_hetero_hub_isl=()
    rdna_hetero_hub_arch=()
    if ! out=$(rdna_hetero_plan_cli hubs "$plan" 2>&1); then
        rdna_die "plan CLI 'hubs' failed: $out"
    fi
    mapfile -t lines <<<"$out"
    for out in "${lines[@]}"; do
        [[ -n $out ]] || continue
        IFS=$'\t' read -r island id hip _ arch <<<"$out"
        rdna_hetero_hub_ids+=("$id")
        rdna_hetero_hub_isl+=("$island")
        rdna_hetero_hub_arch+=("$arch")
    done
    rdna_hetero_sids=()
    rdna_hetero_sisl=()
    rdna_hetero_splan_hip=()
    rdna_hetero_sgfx=()
    if ! out=$(rdna_hetero_plan_cli servers "$plan" 2>&1); then
        rdna_die "plan CLI 'servers' failed: $out"
    fi
    mapfile -t lines <<<"$out"
    for out in "${lines[@]}"; do
        [[ -n $out ]] || continue
        IFS=$'\t' read -r island id hip _ arch <<<"$out"
        rdna_hetero_sids+=("$id")
        rdna_hetero_sisl+=("$island")
        rdna_hetero_splan_hip+=("$hip")
        rdna_hetero_sgfx+=("$arch")
    done
    ((${#rdna_hetero_hub_ids[@]})) || rdna_die "plan '$plan' has no hub device."
    ((${#rdna_hetero_sids[@]})) || rdna_die "plan '$plan' has no expert device."

    # The engine is TP (hub index = TP rank); PCI_BUSES (already checked
    # against TP*PP by serve_rdna.sh) must therefore list one bus per hub.
    ((${PP:-1} == 1)) || rdna_die "hetero-MoE needs PP=1 (got PP=${PP})."
    ((TP == ${#rdna_hetero_hub_ids[@]})) ||
        rdna_die "TP=$TP but the plan has ${#rdna_hetero_hub_ids[@]} hub(s) (${rdna_hetero_hub_ids[*]}); TP and the number of PCI_BUSES of the hub group must equal the plan's hub count."
    rdna_hetero_hub_list=$(IFS=,; printf '%s' "${rdna_hetero_hub_ids[*]}")
    export VLLM_HETERO_MOE_ROLE=hub
    export VLLM_HETERO_MOE_DEVICE_ID=$rdna_hetero_hub_list
    export VLLM_HETERO_MOE_WAIT_MS=${VLLM_HETERO_MOE_WAIT_MS:-2000}
    [[ $VLLM_HETERO_MOE_WAIT_MS =~ ^[0-9]+$ ]] ||
        rdna_die "VLLM_HETERO_MOE_WAIT_MS must be a number of milliseconds (got '$VLLM_HETERO_MOE_WAIT_MS')."

    local knob
    for knob in HETERO_READY_TIMEOUT HETERO_TERM_TIMEOUT HETERO_POLL_INTERVAL; do
        [[ -z ${!knob:-} || ${!knob} =~ ^[0-9]+(\.[0-9]+)?$ ]] ||
            rdna_die "$knob must be a number of seconds (got '${!knob}')."
    done

    # Servers per island. VLLM_HETERO_MOE_SERVERS = buses per island (';'
    # between islands, ',' between buses), mapped in plan order onto the
    # island's expert devices.
    local -A per_island=()
    local i k n_islands=0
    for i in "${!rdna_hetero_sids[@]}"; do
        k=${rdna_hetero_sisl[i]}
        if [[ -z ${per_island[$k]:-} ]]; then
            n_islands=$((n_islands + 1))
        fi
        per_island[$k]=$((${per_island[$k]:-0} + 1))
    done
    rdna_hetero_sbus=()
    rdna_hetero_ship=()
    rdna_hetero_buses_note=""
    local spec=${VLLM_HETERO_MOE_SERVERS:-} b
    local -a isl=() bl=() all_buses=()
    local -A taken=()
    if [[ -n $spec ]]; then
        spec=${spec// /}
        IFS=';' read -r -a isl <<<"$spec"
        ((${#isl[@]} == n_islands)) ||
            rdna_die "VLLM_HETERO_MOE_SERVERS='$spec' lists ${#isl[@]} island(s) but the plan has $n_islands."
        for i in "${!rdna_hetero_sids[@]}"; do
            k=${rdna_hetero_sisl[i]}
            if ((k >= ${#isl[@]})); then
                rdna_die "plan island index $k is outside VLLM_HETERO_MOE_SERVERS='$spec' (islands are numbered 0..$((${#isl[@]} - 1)))."
            fi
            IFS=',' read -r -a bl <<<"${isl[k]}"
            if ((${#bl[@]} != ${per_island[$k]})); then
                rdna_die "VLLM_HETERO_MOE_SERVERS island $k lists ${#bl[@]} bus(es) (${isl[k]}) but the plan has ${per_island[$k]} expert device(s) there."
            fi
            b=${bl[${taken[$k]:-0}]:-}
            taken[$k]=$((${taken[$k]:-0} + 1))
            [[ -n $b ]] || rdna_die "VLLM_HETERO_MOE_SERVERS island $k has an empty bus."
            b=${b,,}
            rdna_hetero_sbus+=("$b")
            all_buses+=("$b")
        done
        rdna_hetero_buses_note="VLLM_HETERO_MOE_SERVERS=$spec"
    else
        printf 'WARNING: VLLM_HETERO_MOE_SERVERS is unset; the servers use the plan'\''s own HIP indices (%s). HIP indices move when cards, risers or switches change: set VLLM_HETERO_MOE_SERVERS=<buses> (e.g. ce,d3,d8,db).\n' \
            "${rdna_hetero_splan_hip[*]}" >&2
        for i in "${!rdna_hetero_sids[@]}"; do
            rdna_hetero_sbus+=("-")
        done
        rdna_hetero_buses_note="VLLM_HETERO_MOE_SERVERS unset: plan hip column ${rdna_hetero_splan_hip[*]}"
    fi

    # Hub and servers never share a card.
    local -a hb=()
    local hbus sbus hb_norm
    if [[ -n $hub_buses ]]; then
        read -r -a hb <<<"${hub_buses//,/ }"
        for hbus in "${hb[@]}"; do
            for sbus in "${rdna_hetero_sbus[@]}"; do
                hb_norm=${hbus,,}
                [[ ${hb_norm##*:} != "${sbus##*:}" ]] ||
                    rdna_die "PCI bus '$hbus' is both a hub bus (PCI_BUSES) and a server bus (VLLM_HETERO_MOE_SERVERS)."
            done
        done
    fi

    # Server HIP indices: resolved from the buses once, all in one call.
    rdna_hetero_resolved=0
    if ((${#all_buses[@]})); then
        local hips
        if [[ -x $VENV/bin/python ]] && hips=$(rdna_resolve_pci_buses "${all_buses[@]}"); then
            local -a hl=()
            IFS=',' read -r -a hl <<<"$hips"
            ((${#hl[@]} == ${#all_buses[@]})) ||
                rdna_die "resolving the server buses '${all_buses[*]}' returned '$hips'."
            rdna_hetero_ship=("${hl[@]}")
            rdna_hetero_resolved=1
        elif [[ ${RDNA_DRY_RUN:-0} == 1 ]]; then
            for i in "${!rdna_hetero_sids[@]}"; do
                rdna_hetero_ship+=("")
            done
        else
            rdna_die "server buses '${all_buses[*]}' did not resolve to GPUs (see above)."
        fi
    else
        rdna_hetero_ship=("${rdna_hetero_splan_hip[@]}")
        rdna_hetero_resolved=1
    fi
    if ((rdna_hetero_resolved)) && [[ -n $hub_hips ]]; then
        local -a hubl=()
        local h s
        IFS=',' read -r -a hubl <<<"$hub_hips"
        for s in "${!rdna_hetero_ship[@]}"; do
            for h in "${hubl[@]}"; do
                [[ $h != "${rdna_hetero_ship[s]}" ]] ||
                    rdna_die "HIP $h is both the hub engine's device and the device of server ${rdna_hetero_sids[s]}; hub and servers must see disjoint GPUs."
            done
        done
    fi

    # Logs, ready files, commands.
    local dir=${HETERO_LOG_DIR:-${RUN_CWD:-$source_dir/cache/run}/hetero}
    [[ $dir == /* ]] || dir=$PWD/$dir
    rdna_hetero_slogs=()
    rdna_hetero_sready=()
    rdna_hetero_buckets=""
    local -a spill=()
    if [[ -n ${VLLM_HETERO_MOE_SPILL_K:-} ]]; then
        spill=(--spill-k "$VLLM_HETERO_MOE_SPILL_K")
    fi
    if ! rdna_hetero_buckets=$(rdna_hetero_plan_cli buckets "$plan" "${spill[@]}" 2>&1); then
        rdna_die "plan CLI 'buckets' failed (VLLM_HETERO_MOE_SPILL_K='${VLLM_HETERO_MOE_SPILL_K:-}'): $rdna_hetero_buckets"
    fi
    local -a extra=()
    read -r -a extra <<<"${HETERO_SERVER_ARGS:-}"
    local -a senv
    for i in "${!rdna_hetero_sids[@]}"; do
        id=${rdna_hetero_sids[i]}
        rdna_hetero_slogs+=("$dir/$id.log")
        rdna_hetero_sready+=("$dir/$id.ready")
        hip=${rdna_hetero_ship[i]}
        if [[ -z $hip ]]; then
            hip="@bus=${rdna_hetero_sbus[i]}"
        fi
        # The servers' kernels must not follow the hub recipe's env, so the
        # kernel variables are set explicitly; PYTHONPATH, LD_LIBRARY_PATH,
        # the HF offline flags and VLLM_CACHE_ROOT are inherited from rdna_init.
        senv=(
            env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES
            "HIP_VISIBLE_DEVICES=$hip"
            VLLM_RDNA2_W4A8_SDOT4=1
            VLLM_RDNA2_MOE_DETERMINISTIC=1
            GPU_MAX_HW_QUEUES=4
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
            PYTORCH_TUNABLEOP_ENABLED=0
            VLLM_HETERO_MOE_ROLE=expert
            "VLLM_HETERO_MOE_DEVICE_ID=$id"
            "VLLM_HETERO_MOE_PLAN=$plan"
            "VLLM_HETERO_MOE_WAIT_MS=$VLLM_HETERO_MOE_WAIT_MS"
        )
        declare -ga "rdna_hetero_server_cmd_$i"
        local -n scmd="rdna_hetero_server_cmd_$i"
        scmd=(
            "${senv[@]}"
            "$VENV/bin/python" -m "$RDNA_HETERO_SERVER_MODULE"
            --plan "$plan" --device-id "$id" --model "$MODEL"
            --ready-file "$dir/$id.ready"
            --buckets "$rdna_hetero_buckets"
            --max-rows "$RDNA_HETERO_MAX_ROWS"
            --wait-ms "$VLLM_HETERO_MOE_WAIT_MS"
        )
        if ((${#extra[@]})); then
            scmd+=("${extra[@]}")
        fi
        unset -n scmd
    done
}

# rdna_hetero_print: the PRINT=1 stanza (needs rdna_hetero_prepare first).
rdna_hetero_print() {
    local i hipdesc
    printf '# hetero:    plan %s\n' "$VLLM_HETERO_MOE_PLAN"
    printf '#            %s\n' "$rdna_hetero_plan_desc"
    for i in "${!rdna_hetero_hub_ids[@]}"; do
        printf '# hetero:    hub %s (island %s, %s) -> VLLM_HETERO_MOE_ROLE=hub VLLM_HETERO_MOE_DEVICE_ID=%s (TP rank %s)\n' \
            "${rdna_hetero_hub_ids[i]}" "${rdna_hetero_hub_isl[i]}" "${rdna_hetero_hub_arch[i]}" \
            "$rdna_hetero_hub_list" "$i"
    done
    printf '# hetero:    servers: %s\n' "$rdna_hetero_buses_note"
    for i in "${!rdna_hetero_sids[@]}"; do
        if ((rdna_hetero_resolved)); then
            hipdesc="HIP ${rdna_hetero_ship[i]}"
        else
            hipdesc=$rdna_hetero_unresolved
        fi
        printf '# hetero:    server %s island %s bus %s -> %s (plan hip %s, %s) log %s ready %s\n' \
            "${rdna_hetero_sids[i]}" "${rdna_hetero_sisl[i]}" "${rdna_hetero_sbus[i]}" "$hipdesc" \
            "${rdna_hetero_splan_hip[i]}" "${rdna_hetero_sgfx[i]}" \
            "${rdna_hetero_slogs[i]}" "${rdna_hetero_sready[i]}"
    done
    printf '# hetero:    buckets %s  max-rows %s  wait-ms %s  ready-timeout %ss  term-timeout %ss\n' \
        "$rdna_hetero_buckets" "$RDNA_HETERO_MAX_ROWS" "$VLLM_HETERO_MOE_WAIT_MS" \
        "${HETERO_READY_TIMEOUT:-900}" "${HETERO_TERM_TIMEOUT:-30}"
    printf '# hetero:    servers start first, each in its own process group; the hub starts after every ready file exists\n'
    printf '\n# --- hetero server commands (environment printed below is inherited) ---\n'
    for i in "${!rdna_hetero_sids[@]}"; do
        # shellcheck disable=SC2178 # a nameref to an array
        local -n scmd="rdna_hetero_server_cmd_$i"
        printf '# %s > %s 2>&1\n' "${rdna_hetero_sids[i]}" "${rdna_hetero_slogs[i]}"
        printf '%s\n' "$(rdna_hetero_words "${scmd[@]}")"
        unset -n scmd
    done
}

# --- supervision ------------------------------------------------------------

# rdna_hetero_spawn LOG CMD...: run CMD in its own session/process group in the
# background (LOG empty: inherit stdout/stderr); the pid lands in
# rdna_hetero_pid. setsid where available, job control otherwise (macOS).
rdna_hetero_spawn() {
    local log=$1
    shift
    if command -v setsid >/dev/null 2>&1; then
        if [[ -n $log ]]; then
            setsid "$@" >"$log" 2>&1 </dev/null &
        else
            setsid "$@" &
        fi
    else
        set -m
        if [[ -n $log ]]; then
            "$@" >"$log" 2>&1 </dev/null &
        else
            "$@" &
        fi
        set +m
    fi
    rdna_hetero_pid=$!
}

rdna_hetero_alive() { kill -0 "$1" 2>/dev/null; }

# A group is alive while any member is; fall back to the pid when the child is
# not a group leader.
rdna_hetero_group_alive() {
    kill -0 -- "-$1" 2>/dev/null || kill -0 "$1" 2>/dev/null
}

rdna_hetero_signal_group() {
    local sig=$1 pid=$2
    kill -"$sig" -- "-$pid" 2>/dev/null || kill -"$sig" "$pid" 2>/dev/null || true
}

# rdna_hetero_ticks SECONDS STEP: ceil(SECONDS / STEP).
rdna_hetero_ticks() {
    awk -v s="$1" -v d="$2" 'BEGIN { n = int(s / d); if (n * d < s) n++; print n }'
}

# rdna_hetero_wait_gone SECONDS PID...: 0 when every group is gone in time.
rdna_hetero_wait_gone() {
    local secs=$1 t ticks pid alive
    shift
    ticks=$(rdna_hetero_ticks "$secs" 0.1)
    for ((t = 0; t <= ticks; t++)); do
        alive=0
        for pid in "$@"; do
            if [[ -n $pid ]] && rdna_hetero_group_alive "$pid"; then
                alive=1
            fi
        done
        ((alive)) || return 0
        sleep 0.1
    done
    return 1
}

# Idempotent: SIGTERM the hub group, wait HETERO_TERM_TIMEOUT, SIGTERM the
# servers, wait again, SIGKILL whatever is left.
rdna_hetero_teardown() {
    if ((rdna_hetero_torn_down)); then
        return 0
    fi
    rdna_hetero_torn_down=1
    trap '' INT TERM HUP
    local term=${HETERO_TERM_TIMEOUT:-30} pid
    if [[ -n $rdna_hetero_hub_pid ]]; then
        rdna_hetero_signal_group TERM "$rdna_hetero_hub_pid"
        rdna_hetero_wait_gone "$term" "$rdna_hetero_hub_pid" || true
    fi
    for pid in "${rdna_hetero_spids[@]}"; do
        rdna_hetero_signal_group TERM "$pid"
    done
    if ((${#rdna_hetero_spids[@]})); then
        rdna_hetero_wait_gone "$term" "${rdna_hetero_spids[@]}" || true
    fi
    for pid in ${rdna_hetero_hub_pid:+"$rdna_hetero_hub_pid"} "${rdna_hetero_spids[@]}"; do
        if rdna_hetero_group_alive "$pid"; then
            rdna_hetero_log "SIGKILL process group $pid (still alive after SIGTERM)"
            rdna_hetero_signal_group KILL "$pid"
        fi
        wait "$pid" 2>/dev/null || true
    done
}

# rdna_hetero_on_signal CODE: tear down and leave with the conventional status.
rdna_hetero_on_signal() {
    rdna_hetero_log "signal received (exit $1); tearing the hub and the expert servers down"
    rdna_hetero_teardown
    exit "$1"
}

# rdna_hetero_server_died I: report server I (exited) with its log tail.
rdna_hetero_server_died() {
    local i=$1 pid=${rdna_hetero_spids[$1]} rc=0
    wait "$pid" 2>/dev/null || rc=$?
    printf 'ERROR: expert server %s (pid %s) exited with code %s %s; log %s\n' \
        "${rdna_hetero_sids[i]}" "$pid" "$rc" "$2" "${rdna_hetero_slogs[i]}" >&2
    if [[ -f ${rdna_hetero_slogs[i]} ]]; then
        printf -- '--- tail of %s ---\n' "${rdna_hetero_slogs[i]}" >&2
        tail -n 40 "${rdna_hetero_slogs[i]}" >&2 || true
        printf -- '--- end of log ---\n' >&2
    fi
}

rdna_hetero_start_servers() {
    local i
    rdna_hetero_spids=()
    for i in "${!rdna_hetero_sids[@]}"; do
        # shellcheck disable=SC2178 # a nameref to an array
        local -n scmd="rdna_hetero_server_cmd_$i"
        mkdir -p "$(dirname "${rdna_hetero_slogs[i]}")" "$(dirname "${rdna_hetero_sready[i]}")"
        # A ready file of a previous run would release the hub too early.
        rm -f "${rdna_hetero_sready[i]}"
        rdna_hetero_spawn "${rdna_hetero_slogs[i]}" "${scmd[@]}"
        rdna_hetero_spids+=("$rdna_hetero_pid")
        rdna_hetero_log "started expert server ${rdna_hetero_sids[i]} (pid $rdna_hetero_pid), log ${rdna_hetero_slogs[i]}"
        unset -n scmd
    done
}

# Wait until every ready file exists. 1 on a dead server or on timeout.
rdna_hetero_wait_ready() {
    local timeout=${HETERO_READY_TIMEOUT:-900} interval=${HETERO_POLL_INTERVAL:-1}
    local ticks t i pending last=$SECONDS
    local -a ready=()
    ticks=$(rdna_hetero_ticks "$timeout" "$interval")
    rdna_hetero_log "waiting up to ${timeout}s for ${#rdna_hetero_sids[@]} expert server(s) to become ready"
    for ((t = 0; t <= ticks; t++)); do
        pending=0
        for i in "${!rdna_hetero_sids[@]}"; do
            [[ -z ${ready[i]:-} ]] || continue
            if [[ -e ${rdna_hetero_sready[i]} ]]; then
                ready[i]=1
                rdna_hetero_log "expert server ${rdna_hetero_sids[i]} is ready"
            elif ! rdna_hetero_alive "${rdna_hetero_spids[i]}"; then
                rdna_hetero_server_died "$i" "before it was ready; the hub was not started"
                return 1
            else
                pending=$((pending + 1))
            fi
        done
        ((pending)) || return 0
        if ((SECONDS - last >= 30)); then
            last=$SECONDS
            rdna_hetero_log "still waiting for $pending expert server(s)"
        fi
        ((t < ticks)) && sleep "$interval"
    done
    for i in "${!rdna_hetero_sids[@]}"; do
        [[ -n ${ready[i]:-} ]] && continue
        printf 'ERROR: expert server %s (pid %s) not ready after %ss; log %s\n' \
            "${rdna_hetero_sids[i]}" "${rdna_hetero_spids[i]}" "$timeout" "${rdna_hetero_slogs[i]}" >&2
        if [[ -f ${rdna_hetero_slogs[i]} ]]; then
            printf -- '--- tail of %s ---\n' "${rdna_hetero_slogs[i]}" >&2
            tail -n 40 "${rdna_hetero_slogs[i]}" >&2 || true
            printf -- '--- end of log ---\n' >&2
        fi
    done
    return 1
}

rdna_hetero_start_hub() {
    local -a wrap=()
    read -r -a wrap <<<"${LAUNCH_WRAPPER:-}"
    if ((${#wrap[@]})); then
        rdna_hetero_spawn "" "${wrap[@]}" "${rdna_hetero_hub_cmd[@]}"
    else
        rdna_hetero_spawn "" "${rdna_hetero_hub_cmd[@]}"
    fi
    rdna_hetero_hub_pid=$rdna_hetero_pid
    rdna_hetero_log "started the hub (pid $rdna_hetero_hub_pid)"
}

# Poll until the hub exits (its code is returned) or a server dies (3).
rdna_hetero_supervise() {
    local interval=${HETERO_POLL_INTERVAL:-1} i rc
    while :; do
        if ! rdna_hetero_alive "$rdna_hetero_hub_pid"; then
            rc=0
            wait "$rdna_hetero_hub_pid" 2>/dev/null || rc=$?
            rdna_hetero_log "the hub exited with code $rc; stopping the expert servers"
            rdna_hetero_teardown
            return "$rc"
        fi
        for i in "${!rdna_hetero_sids[@]}"; do
            if ! rdna_hetero_alive "${rdna_hetero_spids[i]}"; then
                rdna_hetero_server_died "$i" "while the hub was running; stopping the hub"
                rdna_hetero_teardown
                return 3
            fi
        done
        sleep "$interval"
    done
}

# Servers first, then the hub, then supervise. Returns the hub's exit code, 3
# when a server died under a running hub, 1 when the servers never became ready.
rdna_hetero_launch() {
    ((${#rdna_hetero_sids[@]})) || rdna_die "no expert servers to launch."
    ((${#rdna_hetero_hub_cmd[@]})) || rdna_die "no hub command to launch."
    local rc=0
    rdna_hetero_torn_down=0
    rdna_hetero_hub_pid=""
    rdna_hetero_spids=()
    trap rdna_hetero_teardown EXIT
    trap 'rdna_hetero_on_signal 130' INT
    trap 'rdna_hetero_on_signal 143' TERM
    trap 'rdna_hetero_on_signal 129' HUP
    rdna_hetero_start_servers
    if rdna_hetero_wait_ready; then
        rdna_hetero_start_hub
        rdna_hetero_supervise || rc=$?
    else
        rc=1
    fi
    rdna_hetero_teardown
    trap - EXIT INT TERM HUP
    return "$rc"
}
