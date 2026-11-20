#!/usr/bin/env bash
# Stage 4: three audits, then the replays on their saved gates.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
"$PY" campaign.py run --stage 4 --deadline 15 "$@"
