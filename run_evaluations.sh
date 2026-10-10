#!/usr/bin/env bash
# Run from any directory. The full uncertainty run fits 1536 conditional trials.
set -euo pipefail
cd -- "$(dirname -- "$0")"
PYTHON="${PYTHON:-python}"
"$PYTHON" main_run.py --no-show
"$PYTHON" revision_baselines.py
"$PYTHON" revision_ablations.py
"$PYTHON" revision_uncertainty.py
"$PYTHON" -m unittest -v test_reproduction
