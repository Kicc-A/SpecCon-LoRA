#!/bin/bash
set -eu
cd /workspace/SpecCon-LoRA/reproduction/gora_llama2_metamathqa
test ! -e full.log
test ! -e smoke-seed42
bash launch.sh smoke > smoke.log 2>&1
CUDA_VISIBLE_DEVICES='' /root/anaconda3/envs/gora/bin/python validate_smoke.py > smoke_validation.log 2>&1
