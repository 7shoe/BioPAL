#!/usr/bin/env bash
# Stage 6: tables, plots, spend and the list of missing runs. Trains nothing.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
PY="${CAMPAIGN_PYTHON:-$(command -v python || command -v python3)}"   # plots need matplotlib; without it the tables are still written
"$PY" report.py "$@"
