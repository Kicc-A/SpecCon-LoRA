#!/bin/bash
set -euo pipefail
cd /workspace/SpecCon-LoRA/reproduction/gora_llama2_metamathqa/paper_v1
export PYTHONPATH="$PWD/source" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
py=/root/anaconda3/envs/gora/bin/python
exec 9>run.lock
flock -n 9
test ! -e smoke.log
"$py" ../verify_base.py /root/.cache/modelscope/models/shakechen--Llama-2-7b-hf/snapshots/master
bash launch.sh smoke > smoke.log 2>&1
CUDA_VISIBLE_DEVICES='' "$py" validate_smoke.py > smoke_validation.log 2>&1
for SEED in 42 43 44; do
  export SEED
  run="paper-seed$SEED"
  test ! -e "$run"
  bash launch.sh full > "train-$SEED.log" 2>&1
  test -s "$run/final.ckpt"
  "$py" export_merge.py --run "$run" > "export-$SEED.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 "$py" eval_hf.py "$run" 0 > "eval-$SEED.log" 2>&1
done
"$py" - <<'PY'
import json,statistics,pathlib
rows=[json.loads(pathlib.Path(f'paper-seed{s}/result.json').read_text()) for s in (42,43,44)]
scores=[r['accuracy']*100 for r in rows]
pathlib.Path('summary.json').write_text(json.dumps(dict(runs=rows,mean_pct=statistics.mean(scores),sample_sd_pct=statistics.stdev(scores)),indent=2))
PY
