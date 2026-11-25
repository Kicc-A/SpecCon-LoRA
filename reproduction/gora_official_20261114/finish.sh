#!/usr/bin/env bash
set -euo pipefail
HERE=/workspace/SpecCon-LoRA/reproduction/gora_official_20261114
while pgrep -f '[p]ython3.10 -u train/u_train.py' >/dev/null; do
  sleep 30
done
if test -f "$HERE/seed42/gora-gsm8k-seed42/final.ckpt"; then
  bash "$HERE/eval.sh" > "$HERE/seed42/finish.log" 2>&1
else
  echo 'Training ended without final.ckpt' > "$HERE/seed42/finish.log"
  exit 1
fi
