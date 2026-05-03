"""Dataset-Lost analysis on DARPA'98."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from data_loader import Split, default_splits, CATEGORY_INT
import pipeline as dp

LABELS = [0, 1, 2, 3, 4]
ALWAYS_DOS_PRED = None

OUT = Path(__file__).parent / "out"
FRACTIONS = [1.00, 0.50, 0.25, 0.10, 0.05, 0.01,
              0.005, 0.001, 0.0005, 0.0001]
SEEDS = [0, 1, 2]
TAU_NOVELTY = 0.60
TAU_CONF = 0.60
S_MIN = 0.005


def _subsample(train: Split, frac: float, seed: int) -> tuple[Split, np.ndarray]:
    n = len(train.df)
    if frac >= 1.0:
        idx_sorted = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        k = max(1, int(round(frac * n)))
        idx = rng.choice(n, size=k, replace=False)
        idx_sorted = np.sort(idx)
    df_sub = train.df.iloc[idx_sorted].reset_index(drop=True)
    cat_sub = train.category[idx_sorted]
    y_sub = train.y[idx_sorted]
    items_int_sub = train.items_int[idx_sorted]
    inst_sub = train.instance[idx_sorted]
    sub = Split(df=df_sub, category=cat_sub, y=y_sub,
                items_int=items_int_sub, vocab=train.vocab,
                instance=inst_sub)
    return sub, idx_sorted


def _subsample_dedup(train: Split, frac: float, seed: int
                      ) -> tuple[Split, np.ndarray]:
    """Sample frac× of distinct (items_int, y) pairs, one row per pair."""
    a = train.items_int.astype(np.int64)
    keys = (a[:, 0] << 50) | (a[:, 1] << 38) | (a[:, 2] << 26) | \
           (a[:, 3] << 14) | (a[:, 4] << 2) | (a[:, 5] & 3)
    df_keys = pd.DataFrame({"key": keys, "y": train.y,
                              "row": np.arange(len(train.df))})
    rep = df_keys.drop_duplicates(subset=["key", "y"], keep="first")
    n_unique = len(rep)
    rng = np.random.default_rng(seed)
    if frac >= 1.0:
        keep = np.array(rep["row"].values, copy=True)
    else:
        k = max(1, int(round(frac * n_unique)))
        keep = rng.choice(rep["row"].values, size=k, replace=False)
    keep = np.sort(keep)
    df_sub = train.df.iloc[keep].reset_index(drop=True)
    cat_sub = train.category[keep]
    y_sub = train.y[keep]
    items_int_sub = train.items_int[keep]
    inst_sub = train.instance[keep]
    sub = Split(df=df_sub, category=cat_sub, y=y_sub,
                items_int=items_int_sub, vocab=train.vocab,
                instance=inst_sub)
    return sub, keep


def _hash_indices(idx: np.ndarray) -> str:
    return hashlib.md5(idx.tobytes()).hexdigest()[:16]


def _hash_labels(y: np.ndarray) -> str:
    return hashlib.md5(y.tobytes()).hexdigest()[:16]


def _run_trial(
    train_full: Split,
    test: Split,
    *,
    frac: float,
    seed: int,
    test_label_hash: str,
    dedup: bool = False,
) -> dict:
    sampler = _subsample_dedup if dedup else _subsample
    sub, idx = sampler(train_full, frac, seed=seed)
    n_sub = len(sub.df)
    pos = int((sub.category != "normal").sum())
    neg = n_sub - pos
    sub_idx_hash = _hash_indices(idx)
    class_counts = {c: int((sub.category == c).sum())
                    for c in ("normal", "DoS", "Probe", "R2L", "U2R")}
    n_classes_present = sum(1 for c, v in class_counts.items() if v > 0)
    degenerate = n_classes_present < 2 or pos == 0

    t0 = time.time()
    if degenerate:
        return {
            "frac": frac, "seed": seed, "status": "degenerate_monoclass",
            "n_subsample": n_sub, "pos_count": pos, "neg_count": neg,
            "pos_rate": pos / max(1, n_sub),
            "class_counts": class_counts,
            "sub_index_hash": sub_idx_hash,
            "test_label_hash": test_label_hash,
            "runtime_s": 0.0,
        }

    prof = dp.mine_profile(sub, s_min=S_MIN)
    m_train = dp.run_miner(sub, prof, tau_novelty=TAU_NOVELTY)
    tc = dp.train_clf(sub.df, sub.y, m_train.row_novelty, m_train.win_novelty,
                       random_state=seed)
    m_test = dp.run_miner(test, prof, tau_novelty=TAU_NOVELTY)
    pred, conf = dp.predict_clf(tc, test.df,
                                 m_test.row_novelty, m_test.win_novelty)
    r = dp.evaluate(test, m_test.flagged, pred, conf, tau_conf=TAU_CONF)

    cls_recall = recall_score(test.y, pred, labels=LABELS,
                                average=None, zero_division=0)
    cls_f1 = f1_score(test.y, pred, labels=LABELS,
                        average=None, zero_division=0)
    macro_f1 = float(cls_f1.mean())
    macro_recall = float(cls_recall.mean())
    runtime = time.time() - t0

    return {
        "frac": frac, "seed": seed, "status": "ok",
        "n_subsample": n_sub, "pos_count": pos, "neg_count": neg,
        "pos_rate": pos / max(1, n_sub),
        "class_counts": class_counts,
        "sub_index_hash": sub_idx_hash,
        "test_label_hash": test_label_hash,
        "profile_size1": len(prof.size1),
        "profile_size2": len(prof.size2),
        "accuracy": r.accuracy, "precision": r.precision,
        "recall": r.recall, "f1": r.f1, "FAR": r.false_alarm_rate,
        "conn_det_DoS": r.detection_rate["DoS"],
        "conn_det_Probe": r.detection_rate["Probe"],
        "conn_det_R2L": r.detection_rate["R2L"],
        "conn_det_U2R": r.detection_rate["U2R"],
        "conn_det_overall": r.detected_count / max(1, r.total_attacks),
        "conn_id_acc": r.identification_accuracy,
        "inst_det_overall": r.instances_detected / max(1, r.instances_total),
        "inst_det_DoS": r.instance_detection_rate["DoS"],
        "inst_det_Probe": r.instance_detection_rate["Probe"],
        "inst_det_R2L": r.instance_detection_rate["R2L"],
        "inst_det_U2R": r.instance_detection_rate["U2R"],
        "inst_id_acc": r.instance_identification_accuracy,
        "macro_F1": macro_f1,
        "macro_recall": macro_recall,
        "recall_normal": float(cls_recall[0]),
        "recall_DoS": float(cls_recall[1]),
        "recall_Probe": float(cls_recall[2]),
        "recall_R2L": float(cls_recall[3]),
        "recall_U2R": float(cls_recall[4]),
        "F1_DoS": float(cls_f1[1]),
        "F1_Probe": float(cls_f1[2]),
        "F1_R2L": float(cls_f1[3]),
        "F1_U2R": float(cls_f1[4]),
        "runtime_s": runtime,
    }


def _run_one(train, test, *, dedup: bool, jsonl_path: Path,
              test_label_hash: str) -> pd.DataFrame:
    with open(jsonl_path, "w") as f:
        for frac in FRACTIONS:
            for seed in SEEDS:
                trial = _run_trial(train, test, frac=frac, seed=seed,
                                    test_label_hash=test_label_hash,
                                    dedup=dedup)
                trial["sampler"] = "dedup" if dedup else "row"
                f.write(json.dumps(trial) + "\n")
                f.flush()
                print(f"[{'dedup' if dedup else 'row  '}  "
                      f"frac={frac:>7.4f} seed={seed}] "
                      f"n={trial['n_subsample']:>8,} "
                      f"R2L={trial['class_counts']['R2L']:>4} "
                      f"U2R={trial['class_counts']['U2R']:>3}  "
                      f"macro_F1={trial.get('macro_F1', float('nan')):.3f} "
                      f"r_DoS={trial.get('recall_DoS', float('nan')):.3f} "
                      f"r_R2L={trial.get('recall_R2L', float('nan')):.3f} "
                      f"r_U2R={trial.get('recall_U2R', float('nan')):.3f} "
                      f"FAR={trial.get('FAR', float('nan')):.4f}")
    return pd.read_json(jsonl_path, lines=True)


def main() -> None:
    train, test = default_splits()
    OUT.mkdir(parents=True, exist_ok=True)
    test_label_hash = _hash_labels(test.y)
    print(f"train n={len(train.df):,}  test n={len(test.df):,}  "
          f"test_label_hash={test_label_hash}")

    print("\n--- ROW-uniform sampling ---")
    df_row = _run_one(train, test, dedup=False,
                       jsonl_path=OUT / "dataset_lost_trials.jsonl",
                       test_label_hash=test_label_hash)
    print("\n--- DEDUP sampling ---")
    df_dedup = _run_one(train, test, dedup=True,
                         jsonl_path=OUT / "dataset_lost_trials_dedup.jsonl",
                         test_label_hash=test_label_hash)

    agg_cols = ["macro_F1", "macro_recall", "f1", "FAR",
                 "recall_DoS", "recall_Probe", "recall_R2L", "recall_U2R",
                 "F1_DoS", "F1_Probe", "F1_R2L", "F1_U2R",
                 "inst_det_overall", "inst_id_acc", "conn_id_acc"]
    for tag, df in (("row", df_row), ("dedup", df_dedup)):
        ok = df[df["status"] == "ok"]
        agg = ok.groupby("frac")[agg_cols].agg(["mean", "std"])
        agg.to_csv(OUT / f"dataset_lost_summary_{tag}.csv")
        with pd.option_context("display.width", 220, "display.precision", 3):
            print(f"\n=== {tag.upper()} sampler — mean ± std over seeds ===")
            print(agg.to_string())


if __name__ == "__main__":
    main()
