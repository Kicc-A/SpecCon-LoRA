set -euo pipefail
cd /workspace/SpecCon-LoRA/reproduction/gora_llama2_public_native_20261124
while tmux has-session -t gora-public-native-probe 2>/dev/null; do sleep 5; done
test -s smoke-seed42/final.ckpt
if rg -q 'Traceback|OutOfMemoryError' probe.log; then exit 1; fi
export SEED=42 CALIBRATION_STEPS=256
bash launch.sh full > train.log 2>&1
