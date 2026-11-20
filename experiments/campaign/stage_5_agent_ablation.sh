#!/usr/bin/env bash
# Stage 5: Sonnet at high effort for both agents.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
"$PY" campaign.py run --stage 5 --deadline 15 "$@"
