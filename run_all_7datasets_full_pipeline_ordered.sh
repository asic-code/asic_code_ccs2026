#!/usr/bin/env bash
# Full 7-dataset pipeline: best_clustering -> Evaluate_Mapping -> Data_Labeling -> variance/ISV/annotate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo ">>> Stage 1/4: best_clustering_refine_uwkn (all datasets)"
bash "$ROOT/run_all_7datasets_best_clustering_refine_uwkn.sh"

echo ">>> Stage 2/4: Evaluate_Mapping_Conditions (all datasets)"
bash "$ROOT/run_all_7datasets_evaluate_mapping_conditions.sh"

echo ">>> Stage 3/4: Data_Labeling_Evaluate_Thresholds (cache + thresholds, all datasets)"
bash "$ROOT/run_all_7datasets_data_labeling_evaluate_thresholds.sh"

echo ">>> Stage 4/4: variance, ISV, annotate (all datasets)"
bash "$ROOT/run_all_7datasets_isv_eex_turneval.sh"

echo "Done: run_all_7datasets_full_pipeline_ordered.sh"
