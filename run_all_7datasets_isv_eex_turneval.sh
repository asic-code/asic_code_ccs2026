#!/usr/bin/env bash
# Per dataset: compute_feature_variance_benign_attack, ISV_eex_turneval, annotate_performance_history.
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
  echo "=== compute_feature_variance_benign_attack: $ft ==="
  "$PY" compute_feature_variance_benign_attack.py --file_type "$ft"

  echo "=== ISV_eex_turneval_nonredinac_finalsig_turnmap: $ft ==="
  "$PY" ISV_eex_turneval_nonredinac_finalsig_turnmap.py \
    --file_type "$ft" \
    --min_support 0.01 \
    --min_confidence 0.1 \
    --normal_min_support 0.06 \
    --negative_filtering \
    --cstemporal \
    --itemset_limit 200000 \
    --dominant_freq_threshold 0.95 \
    --n_splits 200 \
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
    --save_attackwise_turn_recall

  echo "=== annotate_performance_history_signature_plot_counts: $ft ==="
  "$PY" annotate_performance_history_signature_plot_counts.py --file_type "$ft"
done

echo "Done: run_all_7datasets_isv_eex_turneval.sh"
