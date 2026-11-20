#!/usr/bin/env bash
# Stage 3: seeds 43 and 44, all arms.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
"$PY" campaign.py run --stage 3 --deadline 11.5 "$@"
