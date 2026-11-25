#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROFILE=${1:-author}
case "$PROFILE" in author|paper) ;; *) echo 'Expected author or paper' >&2; exit 2;; esac
for SEED in 42 777 2026; do
  bash "$HERE/train_01.sh" --profile "$PROFILE" --seed "$SEED"
  bash "$HERE/eval_01.sh" --profile "$PROFILE" --seed "$SEED"
done
/root/anaconda3/envs/gora/bin/python "$HERE/summarize.py" --profile "$PROFILE"
