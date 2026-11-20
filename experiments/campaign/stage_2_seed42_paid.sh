#!/usr/bin/env bash
# Stage 2: seed 42, the agent arms and the HPO winners.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
"$PY" campaign.py run --stage 2 --deadline 6.5 "$@"
