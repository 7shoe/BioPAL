#!/usr/bin/env bash
# Smoke-test every runnable backbone on one GPU and record the results.
#   models/backbones/smoke_sweep.sh <gpu> [model ...]
# Models come from models/backbones/models.yaml (audit registry) joined with
# experiments/backbone_flags.py (env + quirks). Models whose conda env does not
# exist (e.g. bio-mamba, deferred) are skipped. Writes
# models/backbones/model_smoke.yaml; exit code 1 if any model fails.
set -u
cd "$(dirname "$0")/../.."
[ $# -ge 1 ] || { echo "usage: $0 <gpu> [model ...]"; exit 2; }
export CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1
shift
out=models/backbones/model_smoke.yaml

# one row per model: name env repo revision modality cache trust block_triton pad_multiple
rows=$(conda run -n biopal-v2 python -c '
import platform, sys, yaml
sys.path.insert(0, "experiments")
from backbone_flags import FLAGS
reg = yaml.safe_load(open("models/backbones/models.yaml"))["models"]
paths = yaml.safe_load(open("paths/paths.yaml"))
base = paths["machines"][platform.node().split(".")[0]]["models_base"]
for name, f in FLAGS.items():
    if name not in reg or (sys.argv[1:] and name not in sys.argv[1:]):
        continue
    m = reg[name]
    print(name, f["env"], m["repo_id"], m["revision"], m["modality"],
          base + "/" + paths["caches"][m["caches"][0]],
          int(f.get("trust", False)), int(f.get("block_triton", False)),
          f.get("pad_multiple", 0))
' "$@") || exit 2
[ -n "$rows" ] || { echo "no matching models"; exit 2; }

envs=$(conda env list | awk '{print $1}')
{ echo "# Written by models/backbones/smoke_sweep.sh; do not edit."
  echo "# $(date -Is) on $(hostname -s), GPU $CUDA_VISIBLE_DEVICES"
  echo "models:"; } > "$out"
failed=0
while read -r name env repo rev mod cache trust triton pad; do
  if ! grep -qx "$env" <<< "$envs"; then
    echo "  $name: {status: SKIP, env: $env}" >> "$out"; continue
  fi
  args=(--repo "$repo" --revision "$rev" --modality "$mod" --cache "$cache")
  [ "$trust" = 1 ] && args+=(--trust-remote-code)
  [ "$pad" != 0 ] && args+=(--lengths "$pad" "$((2 * pad))")
  pp=""; [ "$triton" = 1 ] && pp="envs/shims/no_triton"
  echo "=== $name ($env) ==="
  log=$(PYTHONPATH=$pp conda run -n "$env" --no-capture-output \
          python models/backbones/smoke_backbone.py "${args[@]}" 2>&1 < /dev/null)
  echo "$log"
  facts=$(grep -m1 '^PASS ' <<< "$log" | cut -c6-)
  if [ -n "$facts" ]; then   # JSON is valid YAML flow syntax
    echo "  $name: {status: PASS, env: $env, facts: $facts}" >> "$out"
  else
    echo "  $name: {status: FAIL, env: $env}" >> "$out"; failed=1
  fi
done <<< "$rows"

echo; echo "=== summary -> $out ==="
grep -o '^  [^:]*: {status: [A-Z]*' "$out" | sed 's/: {status: / /'
exit $failed
