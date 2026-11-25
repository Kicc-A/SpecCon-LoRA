set -euo pipefail
cd /workspace/SpecCon-LoRA/reproduction/gora_llama2_public_native_20261124
while ! rg -q 'round-trip verified' prepare.log; do sleep 5; done
CALIBRATION_STEPS=2 bash launch.sh smoke > probe.log 2>&1
