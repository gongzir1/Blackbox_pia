#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /home/test/anaconda3/envs/llama/bin/python -u run_black_tuning.py "$@"
