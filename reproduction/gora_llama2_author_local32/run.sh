set -euo pipefail
export SEED=42 PYTHONPATH="$PWD/source" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
py=/root/anaconda3/envs/gora/bin/python
bash launch.sh full > train.log 2>&1
test -s paper-seed42/final.ckpt
"$py" export_merge.py --run paper-seed42 > export.log 2>&1
CUDA_VISIBLE_DEVICES=1 "$py" -u eval_hf.py paper-seed42 0 > eval.log 2>&1
