#!/usr/bin/env bash
set -euo pipefail
HERE=/workspace/SpecCon-LoRA/reproduction/gora_paper_20261115
mkdir -p "$HERE/seed42"
echo "$(date -Is) waiting for GPUs 0 and 2" > "$HERE/seed42/status.txt"
ready=0
while (( ready < 3 )); do
  if nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F, '$1 == 0 || $1 == 2 {gsub(/ /,"",$2); if ($2 < 1000) n++} END {exit n == 2 ? 0 : 1}'; then
    ready=$((ready + 1))
  else
    ready=0
  fi
  if (( ready < 3 )); then sleep 30; fi
done
echo "$(date -Is) training started" > "$HERE/seed42/status.txt"
if bash "$HERE/run.sh"; then
  echo "$(date -Is) training and evaluation completed" > "$HERE/seed42/status.txt"
else
  code=$?
  echo "$(date -Is) run failed with exit code $code" > "$HERE/seed42/status.txt"
  exit "$code"
fi
