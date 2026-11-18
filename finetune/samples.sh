#!/usr/bin/env bash
# Quick "feel" runs: plain AdamW fine-tuning of a few models, few epochs.
#   finetune/samples.sh <gpu>
# Outputs: finetune/runs/feel_<name>_<timestamp>/ (result.json has val/test metrics).
set -u
cd "$(dirname "$0")"
[ $# -eq 1 ] || { echo "usage: $0 <gpu>"; exit 2; }
export CUDA_VISIBLE_DEVICES=$1
stamp=$(date +%Y%m%d_%H%M%S)

# name | env | model | task | epochs
runs=(
  "ntv2_donors|biopal-v2|ntv2_50m|nt_tasks_official_splice_sites_donors|2"
  "ntv3_100m_donors|biopal-v2|ntv3_100m_pre|nt_tasks_official_splice_sites_donors|2"
  "dnabert2_donors|biopal-v2|dnabert2_117m|nt_tasks_official_splice_sites_donors|2"
  "esm2_35m_gfp|biopal-v2|esm2_t12_35m|proteingym_gfp_fluorescence|1"
  "splicebert_mrl|bio-rna|splicebert|rna_dst_mean_ribosome_load|2"
)

summary=""
for r in "${runs[@]}"; do
  IFS='|' read -r name env model task epochs <<< "$r"
  out=runs/feel_${name}_${stamp}
  echo "=== $name ($model on $task, $epochs epochs, env $env, GPU $1) ==="
  if conda run -n "$env" --no-capture-output python finetune.py \
       --model "$model" --task "$task" --mode plain --epochs "$epochs" --out "$out"; then
    summary+="DONE  $name -> $out/result.json"$'\n'
  else
    summary+="FAIL  $name"$'\n'
  fi
done

echo; echo "=== summary ==="; printf "%s" "$summary"
