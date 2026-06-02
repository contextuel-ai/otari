#!/usr/bin/env bash
# Tear the guardrails demo stack down.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

cd "$GATEWAY_ROOT"
# Both guardrail profiles are included so opt-in containers (anyguardrails +
# encoderfile) are torn down too, regardless of how the stack was started.
# --env-file is conditional — tear-down needs no env vars, so a missing .env
# (e.g. fresh checkout that never ran start.sh) isn't a blocker.
PROFILES=(--profile guardrails --profile guardrails-encoderfile)
if [[ -f "$HERE/.env" ]]; then
  exec docker compose --env-file "$HERE/.env" "${PROFILES[@]}" down "$@"
else
  exec docker compose "${PROFILES[@]}" down "$@"
fi
