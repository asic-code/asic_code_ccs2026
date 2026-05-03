#!/usr/bin/env python3
"""
Compute feature-level variance difference between benign and attack:
1) whole dataset
2) per turn (cstemporal chunk)

Outputs (default):
  ../Seperate_Attack_Ex/CICIDS2017/var_benign_ack/feature_variance_diff_full.csv
  ../Seperate_Attack_Ex/CICIDS2017/var_benign_ack/feature_variance_diff_turn.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path
from Dataset_Choose_Rule.dataset_window_preset import get_temporal_chunk_size
from utils.class_row import get_label_columns_to_exclude
from utils.time_transfer import time_scalar_transfer


EPS = 1e-12
SIGN_EPS = 1e-12


def _default_out_dir(file_type: str) -> str:
    return os.path.join("..", "Seperate_Attack_Ex", str(file_type), "var_benign_ack")


def _resolve_binary_label_col(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (column_name, mode):
      mode='label'   => benign: 0, attack: 1
      mode='cluster' => benign: 0, attack: !=0
    """
    if "label" in df.columns:
        return "label", "label"
    if "cluster" in df.columns:
        return "cluster", "cluster"
    raise ValueError("No 'label' or 'cluster' column found for benign/attack split.")


def _split_masks(df: pd.DataFrame, label_col: str, mode: str):
    if mode == "label":
        benign_mask = df[label_col] == 0
        attack_mask = df[label_col] == 1
    else:
        benign_mask = df[label_col] == 0
        attack_mask = df[label_col] != 0
    return benign_mask, attack_mask


def _numeric_feature_cols(df: pd.DataFrame, file_type: str) -> list[str]:
    exclude = set(get_label_columns_to_exclude(file_type))
    # Also exclude common attack-type string columns from analysis list.
    exclude.update(["Label", "attack_name", "class", "Class", "Attack"])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _var_from_stats(n: int, s: float, ss: float) -> float:
    if n <= 0:
        return np.nan
    mean = s / n
    var = (ss / n) - (mean * mean)
    # guard tiny negative from fp error
    return float(max(var, 0.0))


def _mean_from_stats(n: int, s: float) -> float:
    if n <= 0:
        return np.nan
    return float(s / n)


def _cohens_d(mean_a: float, var_a: float, n_a: int, mean_b: float, var_b: float, n_b: int) -> float:
    """Attack vs benign effect size by pooled std."""
    if n_a <= 1 or n_b <= 1:
        return np.nan
    pooled_num = (n_a - 1) * var_a + (n_b - 1) * var_b
    pooled_den = (n_a + n_b - 2)
    if pooled_den <= 0:
        return np.nan
    pooled_var = pooled_num / pooled_den
    pooled_std = np.sqrt(max(pooled_var, 0.0))
    if pooled_std <= 0:
        return np.nan
    return float((mean_a - mean_b) / pooled_std)


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample KS statistic D (no p-value)."""
    if x.size == 0 or y.size == 0:
        return np.nan
    x = np.sort(x)
    y = np.sort(y)
    vals = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, vals, side="right") / x.size
    cdf_y = np.searchsorted(y, vals, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _quantile_overlap_ratio(x: np.ndarray, y: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> float:
    """
    Overlap ratio of [q_low, q_high] intervals between benign and attack:
    overlap_len / union_len in [0,1], bigger means more overlap.
    """
    if x.size == 0 or y.size == 0:
        return np.nan
    x_lo, x_hi = np.quantile(x, [q_low, q_high])
    y_lo, y_hi = np.quantile(y, [q_low, q_high])
    lo = max(x_lo, y_lo)
    hi = min(x_hi, y_hi)
    overlap = max(0.0, hi - lo)
    union = max(x_hi, y_hi) - min(x_lo, y_lo)
    if union <= 0:
        return 1.0
    return float(overlap / union)


def _sample_and_merge(existing: np.ndarray, values: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if values.size == 0:
        return existing
    if existing.size == 0:
        combined = values
    else:
        combined = np.concatenate([existing, values])
    if combined.size <= max_n:
        return combined
    idx = rng.choice(combined.size, size=max_n, replace=False)
    return combined[idx]


def _build_result_rows(feature_cols: list[str], b_stats: dict, a_stats: dict, turn: int | None = None):
    rows = []
    for c in feature_cols:
        bn, bs, bss = b_stats[c]
        an, a_s, a_ss = a_stats[c]
        mean_b = _mean_from_stats(bn, bs)
        mean_a = _mean_from_stats(an, a_s)
        var_b = _var_from_stats(bn, bs, bss)
        var_a = _var_from_stats(an, a_s, a_ss)
        diff = var_a - var_b if not (np.isnan(var_a) or np.isnan(var_b)) else np.nan
        ratio = ((var_a + EPS) / (var_b + EPS)) if not (np.isnan(var_a) or np.isnan(var_b)) else np.nan
        rec = {
            "feature": c,
            "benign_count": int(bn),
            "attack_count": int(an),
            "benign_mean": mean_b,
            "attack_mean": mean_a,
            "mean_diff_attack_minus_benign": (mean_a - mean_b) if not (np.isnan(mean_a) or np.isnan(mean_b)) else np.nan,
            "benign_variance": var_b,
            "attack_variance": var_a,
            "variance_diff_attack_minus_benign": diff,
            "variance_ratio_attack_over_benign": ratio,
            "cohens_d_attack_vs_benign": _cohens_d(mean_a, var_a, an, mean_b, var_b, bn),
            "abs_variance_diff": abs(diff) if not np.isnan(diff) else np.nan,
        }
        if turn is not None:
            rec["turn"] = int(turn)
        rows.append(rec)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Save benign/attack feature variance differences (full + per-turn).")
    parser.add_argument("--file_type", default="CICIDS2017")
    parser.add_argument("--file_number", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=None, help="Rows per turn. Default: cstemporal preset.")
    parser.add_argument("--max_turns", type=int, default=None, help="Optional cap on turns to process.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Default: ../Seperate_Attack_Ex/<file_type>/var_benign_ack/")
    parser.add_argument("--max_samples_per_feature", type=int, default=50000, help="Max sampled rows per feature/class for KS/overlap.")
    parser.add_argument("--sample_per_chunk_per_feature", type=int, default=2000, help="Per chunk sample cap per feature/class.")
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = _default_out_dir(args.file_type)

    file_path, _ = get_clustered_data_path(args.file_type, args.file_number)
    file_path = os.path.normpath(file_path)
    if not os.path.isfile(file_path):
        print(f"Data file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_size = get_temporal_chunk_size(args.file_type) or 100_000

    os.makedirs(args.out_dir, exist_ok=True)
    out_full = os.path.join(args.out_dir, "feature_variance_diff_full.csv")
    out_turn = os.path.join(args.out_dir, "feature_variance_diff_turn.csv")
    out_stability = os.path.join(args.out_dir, "feature_variance_turn_stability.csv")
    out_advanced = os.path.join(args.out_dir, "feature_distribution_separation_full.csv")
    out_turn_advanced = os.path.join(args.out_dir, "feature_distribution_separation_turn.csv")

    # global accumulators: feature -> [n, sum, sumsq]
    b_global = defaultdict(lambda: [0, 0.0, 0.0])
    a_global = defaultdict(lambda: [0, 0.0, 0.0])
    # global sampled values for distribution-based metrics
    b_samples = defaultdict(lambda: np.array([], dtype=float))
    a_samples = defaultdict(lambda: np.array([], dtype=float))
    rng = np.random.default_rng(20260312)
    turn_rows = []
    feature_cols: list[str] | None = None
    label_col = None
    split_mode = None

    turn = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        turn += 1
        if args.max_turns is not None and turn > args.max_turns:
            break

        df = time_scalar_transfer(chunk.copy(), args.file_type)
        if label_col is None:
            label_col, split_mode = _resolve_binary_label_col(df)
            feature_cols = _numeric_feature_cols(df, args.file_type)
            print(f"Using split column '{label_col}' (mode={split_mode}), numeric features={len(feature_cols)}")
        assert feature_cols is not None
        benign_mask, attack_mask = _split_masks(df, label_col, split_mode)

        # turn-local stats
        b_local = defaultdict(lambda: [0, 0.0, 0.0])
        a_local = defaultdict(lambda: [0, 0.0, 0.0])

        turn_sep_metrics = {}
        for c in feature_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            bvals = s[benign_mask & s.notna()]
            avals = s[attack_mask & s.notna()]

            bn = int(len(bvals))
            an = int(len(avals))
            bs = float(bvals.sum()) if bn > 0 else 0.0
            a_s = float(avals.sum()) if an > 0 else 0.0
            bss = float((bvals * bvals).sum()) if bn > 0 else 0.0
            a_ss = float((avals * avals).sum()) if an > 0 else 0.0

            b_local[c] = [bn, bs, bss]
            a_local[c] = [an, a_s, a_ss]

            # accumulate global
            gb = b_global[c]
            ga = a_global[c]
            gb[0] += bn
            gb[1] += bs
            gb[2] += bss
            ga[0] += an
            ga[1] += a_s
            ga[2] += a_ss

            # sample for global distribution-based metrics
            if bn > 0:
                b_arr = bvals.to_numpy(dtype=float, copy=False)
                if b_arr.size > args.sample_per_chunk_per_feature:
                    b_arr = rng.choice(b_arr, size=args.sample_per_chunk_per_feature, replace=False)
                b_samples[c] = _sample_and_merge(b_samples[c], b_arr, args.max_samples_per_feature, rng)
            if an > 0:
                a_arr = avals.to_numpy(dtype=float, copy=False)
                if a_arr.size > args.sample_per_chunk_per_feature:
                    a_arr = rng.choice(a_arr, size=args.sample_per_chunk_per_feature, replace=False)
                a_samples[c] = _sample_and_merge(a_samples[c], a_arr, args.max_samples_per_feature, rng)

            # turn-local distribution metrics (sampled for speed/stability)
            b_turn = bvals.to_numpy(dtype=float, copy=False) if bn > 0 else np.array([], dtype=float)
            a_turn = avals.to_numpy(dtype=float, copy=False) if an > 0 else np.array([], dtype=float)
            if b_turn.size > args.sample_per_chunk_per_feature:
                b_turn = rng.choice(b_turn, size=args.sample_per_chunk_per_feature, replace=False)
            if a_turn.size > args.sample_per_chunk_per_feature:
                a_turn = rng.choice(a_turn, size=args.sample_per_chunk_per_feature, replace=False)
            ks_t = _ks_statistic(b_turn, a_turn)
            ov_t = _quantile_overlap_ratio(b_turn, a_turn, q_low=0.05, q_high=0.95)
            sep_t = (1.0 - ov_t) if not np.isnan(ov_t) else np.nan
            turn_sep_metrics[c] = {
                "ks_statistic": ks_t,
                "q05_q95_overlap_ratio": ov_t,
                "separation_score_1_minus_overlap": sep_t,
            }

        turn_base_rows = _build_result_rows(feature_cols, b_local, a_local, turn=turn)
        for rec in turn_base_rows:
            m = turn_sep_metrics.get(rec["feature"], {})
            rec["ks_statistic"] = m.get("ks_statistic", np.nan)
            rec["q05_q95_overlap_ratio"] = m.get("q05_q95_overlap_ratio", np.nan)
            rec["separation_score_1_minus_overlap"] = m.get("separation_score_1_minus_overlap", np.nan)
        turn_rows.extend(turn_base_rows)
        print(f"Processed turn {turn}")

    if feature_cols is None:
        print("No data processed.", file=sys.stderr)
        sys.exit(1)

    full_rows = _build_result_rows(feature_cols, b_global, a_global, turn=None)
    # add distribution-based metrics (sample approximation)
    for rec in full_rows:
        c = rec["feature"]
        xb = b_samples[c]
        xa = a_samples[c]
        rec["ks_statistic"] = _ks_statistic(xb, xa)
        rec["q05_q95_overlap_ratio"] = _quantile_overlap_ratio(xb, xa, q_low=0.05, q_high=0.95)
        # lower overlap is better separation; add a separation score for convenience
        if rec["q05_q95_overlap_ratio"] is not np.nan:
            rec["separation_score_1_minus_overlap"] = (1.0 - rec["q05_q95_overlap_ratio"]) if not np.isnan(rec["q05_q95_overlap_ratio"]) else np.nan
        else:
            rec["separation_score_1_minus_overlap"] = np.nan

    full_df_all = pd.DataFrame(full_rows).sort_values("abs_variance_diff", ascending=False)
    turn_df_all = pd.DataFrame(turn_rows).sort_values(["turn", "abs_variance_diff"], ascending=[True, False])

    # Keep the original CSVs easy to read (minimal + intuitive additions only).
    full_cols_main = [
        "feature",
        "benign_count", "attack_count",
        "benign_mean", "attack_mean", "mean_diff_attack_minus_benign",
        "benign_variance", "attack_variance",
        "variance_diff_attack_minus_benign", "variance_ratio_attack_over_benign",
        "abs_variance_diff",
    ]
    turn_cols_main = [
        "feature",
        "benign_count", "attack_count",
        "benign_mean", "attack_mean", "mean_diff_attack_minus_benign",
        "benign_variance", "attack_variance",
        "variance_diff_attack_minus_benign", "variance_ratio_attack_over_benign",
        "abs_variance_diff",
        "turn",
    ]
    full_df = full_df_all[[c for c in full_cols_main if c in full_df_all.columns]].copy()
    turn_df = turn_df_all[[c for c in turn_cols_main if c in turn_df_all.columns]].copy()

    # Advanced separation metrics in a separate CSV.
    advanced_cols = [
        "feature",
        "benign_count", "attack_count",
        "cohens_d_attack_vs_benign",
        "ks_statistic",
        "q05_q95_overlap_ratio",
        "separation_score_1_minus_overlap",
        "benign_mean", "attack_mean", "mean_diff_attack_minus_benign",
        "benign_variance", "attack_variance",
        "variance_diff_attack_minus_benign", "variance_ratio_attack_over_benign",
        "abs_variance_diff",
    ]
    advanced_df = full_df_all[[c for c in advanced_cols if c in full_df_all.columns]].copy()
    advanced_df = advanced_df.sort_values(
        ["ks_statistic", "separation_score_1_minus_overlap", "abs_variance_diff"],
        ascending=[False, False, False],
    )

    turn_advanced_cols = [
        "turn",
        "feature",
        "benign_count", "attack_count",
        "cohens_d_attack_vs_benign",
        "ks_statistic",
        "q05_q95_overlap_ratio",
        "separation_score_1_minus_overlap",
        "benign_mean", "attack_mean", "mean_diff_attack_minus_benign",
        "benign_variance", "attack_variance",
        "variance_diff_attack_minus_benign", "variance_ratio_attack_over_benign",
        "abs_variance_diff",
    ]
    turn_advanced_df = turn_df_all[[c for c in turn_advanced_cols if c in turn_df_all.columns]].copy()
    turn_advanced_df = turn_advanced_df.sort_values(
        ["turn", "ks_statistic", "separation_score_1_minus_overlap", "abs_variance_diff"],
        ascending=[True, False, False, False],
    )

    # turn stability summary: consistency and sign-flip instability
    stab_rows = []
    for c in feature_cols:
        sub = turn_df_all[turn_df_all["feature"] == c]
        diff = sub["variance_diff_attack_minus_benign"].dropna().to_numpy(dtype=float)
        valid_turns = diff.size
        if valid_turns == 0:
            continue
        pos_ratio = float(np.mean(diff > SIGN_EPS))
        neg_ratio = float(np.mean(diff < -SIGN_EPS))
        signs = np.sign(diff)
        signs = signs[np.abs(signs) > 0]
        sign_changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size >= 2 else 0
        flip_rate = float(sign_changes / (signs.size - 1)) if signs.size >= 2 else 0.0
        stab_rows.append({
            "feature": c,
            "valid_turns": int(valid_turns),
            "attack_gt_benign_ratio": pos_ratio,
            "benign_gt_attack_ratio": neg_ratio,
            "sign_changes": sign_changes,
            "sign_flip_rate": flip_rate,
            "mean_abs_variance_diff": float(np.mean(np.abs(diff))),
        })
    stab_df = pd.DataFrame(stab_rows).sort_values(["attack_gt_benign_ratio", "mean_abs_variance_diff"], ascending=[False, False])

    full_df.to_csv(out_full, index=False, encoding="utf-8-sig")
    turn_df.to_csv(out_turn, index=False, encoding="utf-8-sig")
    stab_df.to_csv(out_stability, index=False, encoding="utf-8-sig")
    advanced_df.to_csv(out_advanced, index=False, encoding="utf-8-sig")
    turn_advanced_df.to_csv(out_turn_advanced, index=False, encoding="utf-8-sig")

    print(f"Wrote {out_full}")
    print(f"Wrote {out_turn}")
    print(f"Wrote {out_stability}")
    print(f"Wrote {out_advanced}")
    print(f"Wrote {out_turn_advanced}")
    if not stab_df.empty:
        top_consistent = stab_df.sort_values(["attack_gt_benign_ratio", "mean_abs_variance_diff"], ascending=[False, False]).head(10)
        top_unstable = stab_df.sort_values(["sign_flip_rate", "mean_abs_variance_diff"], ascending=[False, False]).head(10)
        print("\nTop consistent features (attack variance > benign variance across turns):")
        for _, r in top_consistent.iterrows():
            print(f"  {r['feature']}: attack_gt_ratio={r['attack_gt_benign_ratio']:.3f}, valid_turns={int(r['valid_turns'])}, mean_abs_diff={r['mean_abs_variance_diff']:.3e}")
        print("\nTop unstable features (frequent sign flips across turns):")
        for _, r in top_unstable.iterrows():
            print(f"  {r['feature']}: flip_rate={r['sign_flip_rate']:.3f}, sign_changes={int(r['sign_changes'])}, valid_turns={int(r['valid_turns'])}")


if __name__ == "__main__":
    main()
