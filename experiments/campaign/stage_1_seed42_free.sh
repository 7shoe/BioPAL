#!/usr/bin/env bash
# Stage 1: seed 42, the runs without a Claude call (plain, matched, HPO studies).
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
"$PY" campaign.py run --stage 1 --deadline 6.5 "$@"
