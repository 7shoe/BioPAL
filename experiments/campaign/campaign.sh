#!/usr/bin/env bash
# The whole campaign, stage after stage (README.md, section 3). Start it in a tmux session of its own:
#   tmux new-session -d -s campaign 'experiments/campaign/campaign.sh'
# Everything it prints is also in status/campaign.log. Starting it again continues where it stopped: runs
# with a result are skipped, and the clock of the first start (status/t0) stays. Delete status/t0 for a new clock.
set -u
cd "$(dirname "$0")"
PY="${CAMPAIGN_PYTHON:-python3}"
mkdir -p status
exec > >(tee -a status/campaign.log) 2>&1
if [ -f status/t0 ]; then
  echo "continuing the campaign started $(date -d "@$(cut -d. -f1 status/t0)")"
else
  date +%s > status/t0
  echo "campaign starts $(date)"
  "$PY" campaign.py fresh-ready
fi
./stage_0_readiness.sh
./stage_1_seed42_free.sh &        # the free runs go on while the pilot and stage 2 work
./stage_1b_pilot.sh
./stage_2_seed42_paid.sh
wait
./stage_6_report.sh               # the seed-42 table and the trajectories: a result on its own
./stage_3_seeds43_44.sh
./stage_4_expert_ablation.sh &
./stage_5_agent_ablation.sh &
wait
"$PY" campaign.py run --catchup --deadline 15     # what was deferred or failed once, most important first
./stage_6_report.sh
echo "campaign ended $(date)"
