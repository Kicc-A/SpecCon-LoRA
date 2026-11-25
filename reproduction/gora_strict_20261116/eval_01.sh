#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=0,1
export PATH=/workspace/.venvs/gora-vllm/bin:$PATH
export PYTHONPATH=
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
exec /workspace/.venvs/gora-vllm/bin/python "$HERE/evaluate.py" "$@"
