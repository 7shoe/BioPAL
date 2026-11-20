#!/usr/bin/env bash
# Stage 1b: the pilot, three Claude settings on one or two pairs.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
"$PY" campaign.py run --stage 1b --deadline 2.5 "$@"
