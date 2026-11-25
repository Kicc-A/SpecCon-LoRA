#!/usr/bin/env bash
set -euo pipefail

HERE=/workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112
FIRST_PID=${FIRST_PID:-57003}
while kill -0 "$FIRST_PID" 2>/dev/null; do sleep 30; done
test -f "$HERE/seed42/output/last_adapter/adapter_config.json"
SEED=42 bash "$HERE/eval.sh"
for seed in 777 2026; do
  SEED=$seed bash "$HERE/train.sh"
  SEED=$seed bash "$HERE/eval.sh"
done
python - "$HERE" <<'PY'
import json
import statistics
import sys
from pathlib import Path
root = Path(sys.argv[1])
scores = {str(seed): json.loads((root / f'seed{seed}/eval/metrics.json').read_text()) for seed in (42, 777, 2026)}
values = [v['accuracy'] * 100 for v in scores.values()]
summary = {'seeds': scores, 'mean_percent': statistics.mean(values),
           'sample_std_percent': statistics.stdev(values)}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
