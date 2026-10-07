#!/usr/bin/env bash
# Alias: canonical launcher is scripts/serve_rdna.sh RECIPE=flashnext.
# Env vars and trailing KEY=value args are forwarded unchanged.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_rdna.sh" RECIPE=flashnext "$@"
