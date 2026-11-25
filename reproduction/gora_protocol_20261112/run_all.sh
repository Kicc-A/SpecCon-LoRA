#!/usr/bin/env bash
set -euo pipefail

HERE=/workspace/SpecCon-LoRA/reproduction/gora_protocol_20261112
for seed in 42 777 2026; do
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
