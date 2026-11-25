#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec /root/anaconda3/envs/gora/bin/python "$HERE/launch.py" "$@"
