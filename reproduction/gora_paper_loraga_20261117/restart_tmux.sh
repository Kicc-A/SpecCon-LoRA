#!/usr/bin/env bash
set -uo pipefail
ROOT=/workspace/SpecCon-LoRA/reproduction/gora_paper_loraga_20261117
printf 'started_at=%s\n' "$(date -Is)" > "$ROOT/restart_status.txt"
bash "$ROOT/pipeline.sh" > "$ROOT/pipeline_tmux.log" 2>&1
rc=$?
printf 'finished_at=%s\nexit_code=%s\n' "$(date -Is)" "$rc" >> "$ROOT/restart_status.txt"
exit "$rc"
