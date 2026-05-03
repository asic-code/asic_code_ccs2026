#!/usr/bin/env bash
# Run Evaluate_Mapping_Conditions.py for all seven file_type values.
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
  echo "=== Evaluate_Mapping_Conditions: $ft ==="
  "$PY" Evaluate_Mapping_Conditions.py \
    --file_type "$ft" \
    --min_support 0.1 \
    --min_confidence 0.9 \
    --negative_filtering
done

echo "Done: run_all_7datasets_evaluate_mapping_conditions.sh"
