"""End-to-end ADAM pipeline for DARPA'98 connection records."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_loader import Split, CATEGORY_INT


@dataclass
class Profile:
    size1: set[tuple]
    size2: set[tuple]
    n_rows_mined: int
    vocab: dict

    def has(self, s: tuple) -> bool:
        if len(s) == 1:
            return s in self.size1
        return s in self.size2


def _remap_to_vocab(items_int: np.ndarray, src_vocab: dict,
                    dst_vocab: dict) -> np.ndarray:
    inv_src = {v: k for k, v in src_vocab.items()}
    n_src = len(src_vocab)
    table = np.full(n_src, -1, dtype=np.int32)
    for i, key in inv_src.items():
        table[i] = dst_vocab.get(key, -1)
    return table[items_int]


def mine_profile(split: Split, s_min: float = 0.005) -> Profile:
    """Frequent 1- and 2-itemsets mined on attack-free rows."""
    idx = np.where(split.category == "normal")[0]
    items = split.items_int[idx]
    n = len(items)
    min_count = max(1, int(s_min * n))
    c1: Counter = Counter()
    c2: Counter = Counter()
    for j in range(items.shape[1]):
        col = items[:, j]
        uniq, counts = np.unique(col, return_counts=True)
        for u, c in zip(uniq, counts):
            c1[(int(u),)] += int(c)
    for ja, jb in combinations(range(items.shape[1]), 2):
        a = items[:, ja]
        b = items[:, jb]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        key = (lo.astype(np.int64) << 32) | hi.astype(np.int64)
        uniq, counts = np.unique(key, return_counts=True)
        for u, c in zip(uniq, counts):
            lo_v = int(u >> 32)
            hi_v = int(u & ((1 << 32) - 1))
            c2[(lo_v, hi_v)] += int(c)
    size1 = {k for k, v in c1.items() if v >= min_count}
    size2 = {k for k, v in c2.items() if v >= min_count}
    return Profile(size1=size1, size2=size2, n_rows_mined=n,
                   vocab=split.vocab)


def _row_items_for(split: Split, profile: Profile) -> np.ndarray:
    if split.vocab is profile.vocab:
        return split.items_int
    return _remap_to_vocab(split.items_int, split.vocab, profile.vocab)


@dataclass
class MinerOut:
    row_novelty: np.ndarray
    win_novelty: np.ndarray
    flagged: np.ndarray
    tau_novelty: float


def run_miner(
    split: Split,
    profile: Profile,
    *,
    window_size: int = 2000,
    s_win: float = 0.05,
    tau_novelty: float = 0.2,
) -> MinerOut:
    items = _row_items_for(split, profile)
    n, K = items.shape
    row_novelty = np.zeros(n, dtype=np.float32)
    win_novelty = np.zeros(n, dtype=np.float32)

    sz1 = profile.size1
    sz2 = profile.size2
    s1_known = np.zeros((n, K), dtype=bool)
    for j in range(K):
        col = items[:, j]
        uniq = np.unique(col)
        known_uniq = {int(u) for u in uniq if (int(u),) in sz1}
        s1_known[:, j] = np.isin(col, list(known_uniq)) if known_uniq \
                         else np.zeros(n, dtype=bool)

    n_pairs = (K * (K - 1)) // 2
    s2_known = np.zeros((n, n_pairs), dtype=bool)
    pair_idx = 0
    for ja, jb in combinations(range(K), 2):
        a = items[:, ja]
        b = items[:, jb]
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b).astype(np.int64)
        key = (lo << 32) | hi
        uniq_keys = np.unique(key)
        known_keys = []
        for u in uniq_keys:
            lo_v = int(u >> 32)
            hi_v = int(u & ((1 << 32) - 1))
            if (lo_v, hi_v) in sz2:
                known_keys.append(int(u))
        s2_known[:, pair_idx] = np.isin(key, known_keys) if known_keys \
                                 else np.zeros(n, dtype=bool)
        pair_idx += 1

    total = K + n_pairs
    unknown = (K - s1_known.sum(axis=1)) + (n_pairs - s2_known.sum(axis=1))
    row_novelty = (unknown / total).astype(np.float32)

    w = min(window_size, n)
    if w > 0:
        csum = np.concatenate(([0.0], np.cumsum(row_novelty, dtype=np.float64)))
        idx = np.arange(n)
        lo = np.maximum(0, idx - w + 1)
        win_novelty = ((csum[idx + 1] - csum[lo]) / (idx - lo + 1)).astype(np.float32)

    flagged = row_novelty > tau_novelty
    return MinerOut(row_novelty=row_novelty, win_novelty=win_novelty,
                    flagged=flagged, tau_novelty=tau_novelty)


CAT_COLS = ["proto", "service"]
NUM_COLS = ["duration_s"]


@dataclass
class TrainedClf:
    clf: DecisionTreeClassifier
    ohe: OneHotEncoder
    scaler: StandardScaler


def _featurize(df: pd.DataFrame, nov: np.ndarray, win: np.ndarray,
                ohe=None, scaler=None, fit=False):
    cats = df[CAT_COLS].values
    nums = df[NUM_COLS].values.astype(np.float32)
    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             max_categories=100)
        cats_enc = ohe.fit_transform(cats)
        scaler = StandardScaler()
        nums_enc = scaler.fit_transform(nums)
    else:
        cats_enc = ohe.transform(cats)
        nums_enc = scaler.transform(nums)
    return (np.hstack([cats_enc, nums_enc,
                        nov.reshape(-1, 1), win.reshape(-1, 1)]).astype(np.float32),
            ohe, scaler)


def train_clf(df, y, nov, win, *, max_depth=15, min_samples_leaf=20,
               random_state=0) -> TrainedClf:
    X, ohe, scaler = _featurize(df, nov, win, fit=True)
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  class_weight="balanced",
                                  random_state=random_state)
    clf.fit(X, y)
    return TrainedClf(clf=clf, ohe=ohe, scaler=scaler)


def predict_clf(tc: TrainedClf, df, nov, win):
    X, _, _ = _featurize(df, nov, win, ohe=tc.ohe, scaler=tc.scaler)
    proba = tc.clf.predict_proba(X)
    pred = tc.clf.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    return pred, conf


UNKNOWN = -1


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    detection_rate: dict
    identification_accuracy: float
    detected_count: int
    total_attacks: int
    unknown_rate: float
    instance_detection_rate: dict
    instances_detected: int
    instances_total: int
    instance_identification_accuracy: float


def evaluate(test: Split, flagged, pred, conf, tau_conf=0.6) -> EvalResult:
    is_unknown = flagged & (conf < tau_conf)
    alarm = flagged & ((pred != CATEGORY_INT["normal"]) | is_unknown)
    label = pred.copy()
    label[is_unknown] = UNKNOWN
    y = test.y
    is_atk = y != CATEGORY_INT["normal"]
    is_norm = ~is_atk
    tp = int((alarm & is_atk).sum())
    fp = int((alarm & is_norm).sum())
    fn = int((~alarm & is_atk).sum())
    tn = int((~alarm & is_norm).sum())
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    far = fp / max(1, fp + tn)
    det = {}
    for cat in ("DoS", "Probe", "R2L", "U2R"):
        mask = test.category == cat
        det[cat] = float(alarm[mask].mean()) if mask.any() else float("nan")
    detected = alarm & is_atk
    detected_known = detected & (~is_unknown)
    correct = int((detected_known & (label == y)).sum())
    id_acc = correct / max(1, int(detected_known.sum()))
    unknown_rate = float((is_unknown & is_atk).sum() / max(1, int(detected.sum())))

    inst_det = {c: [0, 0] for c in ("DoS", "Probe", "R2L", "U2R")}
    inst_detected_total = 0
    inst_total = 0
    inst_id_correct = 0
    inst_id_known_total = 0
    if test.instance is not None:
        insts = test.instance
        keys = [k for k in set(insts.tolist()) if k]
        for key in keys:
            mask = insts == key
            cats_here = test.category[mask]
            if len(cats_here) == 0:
                continue
            true_cat = cats_here[0]
            if true_cat == "normal":
                continue
            inst_total += 1
            inst_det[true_cat][1] += 1
            is_det = bool(alarm[mask].any())
            if is_det:
                inst_detected_total += 1
                inst_det[true_cat][0] += 1
                known = alarm[mask] & (~is_unknown[mask])
                if known.any():
                    inst_id_known_total += 1
                    vals = label[mask][known]
                    uniq, counts = np.unique(vals, return_counts=True)
                    mode_pred = int(uniq[counts.argmax()])
                    if mode_pred == CATEGORY_INT[true_cat]:
                        inst_id_correct += 1
    inst_det_rate = {c: (v[0] / max(1, v[1])) for c, v in inst_det.items()}
    inst_id_acc = inst_id_correct / max(1, inst_id_known_total)

    return EvalResult(accuracy=acc, precision=prec, recall=rec, f1=f1,
                      false_alarm_rate=far, detection_rate=det,
                      identification_accuracy=id_acc,
                      detected_count=int(detected.sum()),
                      total_attacks=int(is_atk.sum()),
                      unknown_rate=unknown_rate,
                      instance_detection_rate=inst_det_rate,
                      instances_detected=inst_detected_total,
                      instances_total=inst_total,
                      instance_identification_accuracy=inst_id_acc)


def format_result(r: EvalResult) -> str:
    lines = [
        "=== Connection-level ===",
        f"Accuracy:             {r.accuracy:.4f}",
        f"Precision:            {r.precision:.4f}",
        f"Recall:               {r.recall:.4f}",
        f"F1:                   {r.f1:.4f}",
        f"False-alarm rate:     {r.false_alarm_rate:.4f}",
        "Per-category connection detection:",
    ]
    for cat, v in r.detection_rate.items():
        lines.append(f"  {cat:>5}: {v:.4f}")
    lines += [
        f"Overall connections detected: {r.detected_count:,} / {r.total_attacks:,}"
        f" ({r.detected_count / max(1, r.total_attacks):.4f})",
        f"Connection identification accuracy:  {r.identification_accuracy:.4f}",
        "",
        "=== Instance-level ===",
        "Per-category instance detection:",
    ]
    for cat, v in r.instance_detection_rate.items():
        lines.append(f"  {cat:>5}: {v:.4f}")
    lines += [
        f"Overall instances detected: {r.instances_detected} / {r.instances_total}"
        f" ({r.instances_detected / max(1, r.instances_total):.4f})",
        f"Instance identification accuracy: {r.instance_identification_accuracy:.4f}",
    ]
    return "\n".join(lines)


def run_end_to_end(train: Split, test: Split, *, s_min=0.005,
                    window_size=2000, s_win=0.05, tau_novelty=0.60,
                    tau_conf=0.60, random_state=0) -> EvalResult:
    prof = mine_profile(train, s_min=s_min)
    print(f"  profile: size1={len(prof.size1)} size2={len(prof.size2)} "
          f"(mined on {prof.n_rows_mined:,} attack-free rows)")
    m_train = run_miner(train, prof, window_size=window_size, s_win=s_win,
                         tau_novelty=tau_novelty)
    tc = train_clf(train.df, train.y, m_train.row_novelty, m_train.win_novelty,
                    random_state=random_state)
    m_test = run_miner(test, prof, window_size=window_size, s_win=s_win,
                        tau_novelty=tau_novelty)
    pred, conf = predict_clf(tc, test.df, m_test.row_novelty, m_test.win_novelty)
    return evaluate(test, m_test.flagged, pred, conf, tau_conf=tau_conf)


if __name__ == "__main__":
    from data_loader import default_splits
    train, test = default_splits()
    print(f"train n={len(train.df):,}  test n={len(test.df):,}")
    r = run_end_to_end(train, test)
    print(format_result(r))
