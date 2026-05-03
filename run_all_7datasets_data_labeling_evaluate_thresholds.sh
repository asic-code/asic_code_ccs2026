#!/usr/bin/env bash
# Per dataset: Data_Labeling_Evaluate_Thresholds_best_clustering.py --generate_cache then --evaluate_thresholds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -f "$ROOT/.venv/Scripts/python.exe" ]]; then
  PY="$ROOT/.venv/Scripts/python.exe"
else
  PY="${PYTHON:-python3}"
fi

FILE_TYPES=(MiraiBotnet netML CICIDS2017 NSL-KDD DARPA98 CICIoT2023 Kitsune)

for ft in "${FILE_TYPES[@]}"; do
  echo "=== Data_Labeling generate_cache: $ft ==="
  "$PY" Data_Labeling_Evaluate_Thresholds_best_clustering.py \
    --file_type "$ft" \
    --max_score \
    --generate_cache
  echo "=== Data_Labeling evaluate_thresholds: $ft ==="
  "$PY" Data_Labeling_Evaluate_Thresholds_best_clustering.py \
    --file_type "$ft" \
    --max_score \
    --evaluate_thresholds
done

echo "Done: run_all_7datasets_data_labeling_evaluate_thresholds.sh"
