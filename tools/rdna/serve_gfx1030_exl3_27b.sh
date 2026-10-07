#!/usr/bin/env bash
# Alias: canonical launcher is tools/rdna/serve_rdna.sh RECIPE=27b-exl3.
# Env vars and trailing KEY=value args are forwarded unchanged.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_rdna.sh" RECIPE=27b-exl3 "$@"
