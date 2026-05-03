#!/usr/bin/env bash
# Run best_clustering_refine_uwkn.py for all seven file_type values.
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
  echo "=== best_clustering_refine_uwkn: $ft ==="
  "$PY" best_clustering_refine_uwkn.py \
    --file_type "$ft" \
    --max_score \
    --refine_clusters \
    --refine_second_gate
done

echo "Done: run_all_7datasets_best_clustering_refine_uwkn.sh"
