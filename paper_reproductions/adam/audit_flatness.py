"""Audit per-class metrics across training fractions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, confusion_matrix

from data_loader import default_splits, CATEGORY_INT, CATEGORY_NAME
import pipeline as dp
from dataset_lost import _subsample

OUT = Path(__file__).parent / "out"
LABELS = [0, 1, 2, 3, 4]
LABEL_NAMES = [CATEGORY_NAME[i] for i in LABELS]


def uniqueness(train, test) -> dict:
    a = train.items_int.astype(np.int64)
    train_keys = (a[:, 0] << 50) | (a[:, 1] << 38) | (a[:, 2] << 26) | \
                 (a[:, 3] << 14) | (a[:, 4] << 2) | (a[:, 5] & 3)
    a = test.items_int.astype(np.int64)
    test_keys = (a[:, 0] << 50) | (a[:, 1] << 38) | (a[:, 2] << 26) | \
                (a[:, 3] << 14) | (a[:, 4] << 2) | (a[:, 5] & 3)

    train_distinct_items = int(pd.Series(train_keys).nunique())
    test_distinct_items = int(pd.Series(test_keys).nunique())
    train_distinct_pairs = int(
        pd.DataFrame({"k": train_keys, "y": train.y}).drop_duplicates().shape[0]
    )
    return {
        "train_rows": int(len(train.df)),
        "train_distinct_item_tuples": train_distinct_items,
        "train_distinct_(item,label)_pairs": train_distinct_pairs,
        "train_compression_ratio": int(len(train.df)) / train_distinct_items,
        "test_rows": int(len(test.df)),
        "test_distinct_item_tuples": test_distinct_items,
    }


def constant_baseline(test) -> dict:
    """Always-predict-DoS baseline metrics."""
    pred = np.full(len(test.y), CATEGORY_INT["DoS"], dtype=test.y.dtype)
    return per_class_metrics("always_DoS", pred, test.y)


def per_class_metrics(name, pred, y_true) -> dict:
    rec = recall_score(y_true, pred, labels=LABELS,
                        average=None, zero_division=0)
    f1 = f1_score(y_true, pred, labels=LABELS,
                   average=None, zero_division=0)
    macro_f1 = float(f1.mean())
    macro_recall = float(rec.mean())
    is_atk = y_true != CATEGORY_INT["normal"]
    is_pred_atk = pred != CATEGORY_INT["normal"]
    tp = int((is_pred_atk & is_atk).sum())
    fp = int((is_pred_atk & ~is_atk).sum())
    tn = int((~is_pred_atk & ~is_atk).sum())
    fn = int((~is_pred_atk & is_atk).sum())
    out = {
        "name": name,
        "macro_F1": macro_f1,
        "macro_recall": macro_recall,
        "binary_recall(TPR)": tp / max(1, tp + fn),
        "binary_FPR": fp / max(1, fp + tn),
        "binary_F1": (2 * tp) / max(1, 2 * tp + fp + fn),
    }
    for i, lbl in zip(LABELS, LABEL_NAMES):
        out[f"recall_{lbl}"] = float(rec[i])
        out[f"F1_{lbl}"] = float(f1[i])
    return out


def evaluate_subsample(sub, test, *, seed: int) -> dict:
    """Plain supervised tree on subsample, predict full test."""
    nov_tr = np.zeros(len(sub.df), dtype=np.float32)
    win_tr = np.zeros(len(sub.df), dtype=np.float32)
    nov_te = np.zeros(len(test.df), dtype=np.float32)
    win_te = np.zeros(len(test.df), dtype=np.float32)
    tc = dp.train_clf(sub.df, sub.y, nov_tr, win_tr, random_state=seed)
    pred, _ = dp.predict_clf(tc, test.df, nov_te, win_te)
    return per_class_metrics("trained", pred, test.y)


def main() -> None:
    train, test = default_splits()

    print("=== 1. Uniqueness audit ===")
    u = uniqueness(train, test)
    for k, v in u.items():
        if isinstance(v, float):
            print(f"  {k}: {v:>14.2f}")
        else:
            print(f"  {k}: {v:>14,}")

    print()
    print("=== 2. Constant-baseline (always-predict-DoS) on full test ===")
    base = constant_baseline(test)
    for k, v in base.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print()
    print("=== 3. Trained tree, sub-1% fractions, full test set ===")
    print()
    extra_fracs = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    seeds = (0, 1, 2)
    rows = []
    for frac in extra_fracs:
        for seed in seeds:
            sub, _ = _subsample(train, frac, seed=seed)
            cls_count = {c: int((sub.category == c).sum())
                         for c in ("normal", "DoS", "Probe", "R2L", "U2R")}
            r = evaluate_subsample(sub, test, seed=seed)
            r.update({"frac": frac, "seed": seed,
                      "n_sub": len(sub.df), "n_DoS_train": cls_count["DoS"],
                      "n_R2L_train": cls_count["R2L"],
                      "n_U2R_train": cls_count["U2R"]})
            rows.append(r)
            print(f"  frac={frac:>7.4f} seed={seed} n={len(sub.df):>7,} "
                  f"|R2L|={cls_count['R2L']:>4} |U2R|={cls_count['U2R']:>3}  "
                  f"macro_F1={r['macro_F1']:.4f}  "
                  f"recall_DoS={r['recall_DoS']:.3f}  "
                  f"recall_Probe={r['recall_Probe']:.3f}  "
                  f"recall_R2L={r['recall_R2L']:.3f}  "
                  f"recall_U2R={r['recall_U2R']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "audit_flatness.csv", index=False)

    print()
    print("=== 4. Mean (over 3 seeds) per fraction ===")
    summary = df.groupby("frac").agg({
        "macro_F1": "mean",
        "macro_recall": "mean",
        "binary_F1": "mean",
        "binary_FPR": "mean",
        "recall_DoS": "mean",
        "recall_Probe": "mean",
        "recall_R2L": "mean",
        "recall_U2R": "mean",
        "F1_DoS": "mean",
        "F1_Probe": "mean",
        "F1_R2L": "mean",
        "F1_U2R": "mean",
    })
    summary.to_csv(OUT / "audit_flatness_summary.csv")
    with pd.option_context("display.width", 220, "display.precision", 4):
        print(summary.to_string())

    print()
    print("=== Reference: constant-baseline metrics (re-print) ===")
    print(f"  always-DoS: macro_F1={base['macro_F1']:.4f}  "
          f"macro_recall={base['macro_recall']:.4f}  "
          f"recall_DoS={base['recall_DoS']:.3f}  "
          f"recall_R2L={base['recall_R2L']:.3f}  "
          f"recall_U2R={base['recall_U2R']:.3f}")


if __name__ == "__main__":
    main()
