#!/usr/bin/env bash
# Stage 0 (README.md, section 3): checks, then one tiny run of every kind. Pass --no-live to skip the Claude check.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
./qwen_server.sh start || echo "the local expert is not up: the runs that need it will be skipped"
"$PY" campaign.py check "$@"
"$PY" campaign.py run --stage 0 --deadline 1
"$PY" campaign.py verdict
