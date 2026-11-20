#!/usr/bin/env bash
# The local expert of the campaign: Qwen3-8B on vLLM, in its own detached tmux session "qwen".
#   qwen_server.sh start | stop | status
# The snapshot folder of the weights comes from $QWEN_SNAPSHOT or from status/qwen_snapshot.path.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${QWEN_PORT:-8000}"
REV=b968826d9c46dd6066d109eabc6255188de91218
ENVS="${CONDA_ENVS:-$(dirname "$(dirname "${CONDA_EXE:-$HOME/miniconda3/bin/conda}")")/envs}"
up() { curl -s --noproxy '*' --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q "Qwen/Qwen3-8B"; }

case "${1:-status}" in
  status)
    if up; then echo "up on port $PORT"; else echo "down"; exit 1; fi ;;
  stop)
    tmux kill-session -t qwen 2>/dev/null; echo "stopped" ;;
  start)
    if up; then echo "already up on port $PORT"; exit 0; fi
    SNAP="${QWEN_SNAPSHOT:-$(cat "$HERE/status/qwen_snapshot.path" 2>/dev/null)}"
    [ -n "$SNAP" ] && [ -d "$SNAP" ] || { echo "set QWEN_SNAPSHOT to the snapshot folder of Qwen3-8B, or write it to status/qwen_snapshot.path"; exit 1; }
    case "$SNAP" in *"$REV"*) ;; *) echo "the snapshot folder does not name revision $REV"; exit 1 ;; esac
    tmux kill-session -t qwen 2>/dev/null
    # the card with the most free memory, if it has 26 GB (the server takes about 24)
    CARD=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
           awk -F', *' '{f = $2 - $3; if (f > best) {best = f; c = $1}} END {if (best >= 26000) print c}')
    [ -n "$CARD" ] || { echo "no card has 26 GB free for the server"; exit 1; }
    mkdir -p "$HERE/status"
    tmux new-session -d -s qwen "CUDA_VISIBLE_DEVICES=$CARD HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PATH=$ENVS/agent/bin:\$PATH $ENVS/agent/bin/vllm serve '$SNAP' --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port $PORT --max-model-len 8192 --dtype bfloat16 --seed 0 --gpu-memory-utilization 0.25 --kv-cache-memory-bytes 2G --max-num-seqs 4 > '$HERE/status/qwen_server.log' 2>&1"
    echo "starting on card $CARD ..."
    for _ in $(seq 1 120); do
      if up; then echo "$CARD" > "$HERE/status/qwen_card"; echo "up on port $PORT (card $CARD)"; exit 0; fi
      tmux has-session -t qwen 2>/dev/null || { echo "the server ended; see status/qwen_server.log"; exit 1; }
      sleep 10
    done
    echo "no answer after 20 minutes; see status/qwen_server.log"; exit 1 ;;
  *)
    echo "usage: qwen_server.sh start | stop | status"; exit 2 ;;
esac
