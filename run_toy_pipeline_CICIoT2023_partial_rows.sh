#!/usr/bin/env bash
#
# CICIoT2023 toy run: truncate training-flow.csv to header + N data rows, then run the same
# stage order as run_all_7datasets_full_pipeline_ordered.sh (single file_type only).
#
# Optional env:
#   TOY_MAX_DATA_ROWS     Max data rows after header (default 8000).
#   TOY_ISV_N_SPLITS      ISV --n_splits (default 80; full pipeline uses 200).
#   TOY_ISV_RUN_TURN_END  ISV --run_turn_end when set; unset defaults to 15. Export empty
#                         (TOY_ISV_RUN_TURN_END=) to omit --run_turn_end.
#   TOY_SKIP_RESTORE      If 1, do not copy the snapshot back onto training-flow.csv on exit.
#
# Files under ../Dataset/load_dataset/CICIoT2023/:
#   - First run copies the current training-flow.csv to training-flow.asic_present_toy_full_snapshot.csv.
#   - Each run overwrites training-flow.csv with the first N+1 lines of that snapshot.
#   - On exit (unless TOY_SKIP_RESTORE=1), training-flow.csv is restored from the snapshot.
#
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

FT="CICIoT2023"
TOY_MAX_DATA_ROWS="${TOY_MAX_DATA_ROWS:-8000}"
TOY_ISV_N_SPLITS="${TOY_ISV_N_SPLITS:-80}"
TOY_SKIP_RESTORE="${TOY_SKIP_RESTORE:-0}"

CICIOT_DIR="$(cd "$ROOT/../Dataset/load_dataset/CICIoT2023" && pwd)"
SRC="$CICIOT_DIR/training-flow.csv"
SNAP="$CICIOT_DIR/training-flow.asic_present_toy_full_snapshot.csv"

if [[ ! -f "$SRC" ]]; then
  echo "Error: CICIoT2023 train CSV not found: $SRC"
  exit 1
fi

if ! [[ "$TOY_MAX_DATA_ROWS" =~ ^[0-9]+$ ]] || [[ "$TOY_MAX_DATA_ROWS" -lt 1 ]]; then
  echo "Error: TOY_MAX_DATA_ROWS must be a positive integer, got: $TOY_MAX_DATA_ROWS"
  exit 1
fi

restore_training_flow() {
  if [[ "$TOY_SKIP_RESTORE" == "1" ]]; then
    return 0
  fi
  if [[ -f "$SNAP" && -f "$SRC" ]]; then
    cp -f "$SNAP" "$SRC" || true
    echo "[toy] Restored training-flow.csv from snapshot."
  fi
}

trap 'restore_training_flow' EXIT

if [[ ! -f "$SNAP" ]]; then
  echo "[toy] Creating snapshot: $SNAP"
  cp -f "$SRC" "$SNAP"
fi

echo "[toy] Writing header + ${TOY_MAX_DATA_ROWS} data rows from snapshot -> $SRC"
head -n $((TOY_MAX_DATA_ROWS + 1)) "$SNAP" >"${SRC}.asic_toy.$$" && mv "${SRC}.asic_toy.$$" "$SRC"

if [[ ! -v TOY_ISV_RUN_TURN_END ]]; then
  TOY_ISV_RUN_TURN_END=15
fi
ISV_EXTRA_END=()
if [[ -n "$TOY_ISV_RUN_TURN_END" ]]; then
  ISV_EXTRA_END=(--run_turn_end "$TOY_ISV_RUN_TURN_END")
fi

echo ">>> Toy 1/4: best_clustering_refine_uwkn ($FT)"
"$PY" best_clustering_refine_uwkn.py \
  --file_type "$FT" \
  --max_score \
  --refine_clusters \
  --refine_second_gate

echo ">>> Toy 2/4: Evaluate_Mapping_Conditions ($FT)"
"$PY" Evaluate_Mapping_Conditions.py \
  --file_type "$FT" \
  --min_support 0.1 \
  --min_confidence 0.9 \
  --negative_filtering

echo ">>> Toy 3/4: Data_Labeling ($FT)"
"$PY" Data_Labeling_Evaluate_Thresholds_best_clustering.py \
  --file_type "$FT" \
  --max_score \
  --generate_cache
"$PY" Data_Labeling_Evaluate_Thresholds_best_clustering.py \
  --file_type "$FT" \
  --max_score \
  --evaluate_thresholds

echo ">>> Toy 4/4: variance, ISV, annotate ($FT)"
"$PY" compute_feature_variance_benign_attack.py --file_type "$FT"

"$PY" ISV_eex_turneval_nonredinac_finalsig_turnmap.py \
  --file_type "$FT" \
  --min_support 0.01 \
  --min_confidence 0.1 \
  --normal_min_support 0.06 \
  --negative_filtering \
  --cstemporal \
  --itemset_limit 200000 \
  --dominant_freq_threshold 0.95 \
  --n_splits "$TOY_ISV_N_SPLITS" \
  --use_fp_metrics \
  --association_method rarm \
  --precision_underlimit 0.7 \
  --dominant_attach_filter \
  --fp_replace_by_coverage \
  --fp_reduce_supersets \
  --separability \
  --dominant_attach_threshold 0.9 \
  --use_separation_feature_filter \
  --use_separation_stability_filter \
  --separation_turn_min_ks 0.3 \
  --separation_turn_max_overlap 0.8 \
  --separation_turn_min_abs_cohens_d 0.25 \
  --separation_top_k_features_turn 15 \
  --max_level 13 \
  --fp_belief_threshold 0.9 \
  --adaptive_support_retry \
  --save_turn_fp_contribution \
  --adaptive_support_retry_candidate_l1_threshold 12000 \
  --precision_underlimit_keep_no_alert 0 \
  --adaptive_support_retry_bidirectional \
  --adaptive_support_retry_low_candidate_l1_threshold 6000 \
  --adaptive_support_retry_target_level 100 \
  --save_attackwise_turn_recall \
  "${ISV_EXTRA_END[@]}"

"$PY" annotate_performance_history_signature_plot_counts.py --file_type "$FT"

echo "Done: run_toy_pipeline_CICIoT2023_partial_rows.sh"
