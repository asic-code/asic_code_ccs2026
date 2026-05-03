"""Wraps the official KitNET-py implementation."""
from __future__ import annotations
import os
import sys
from collections import Counter
from typing import Optional

import numpy as np


def _ensure_kitnet_path() -> None:
    p = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "ref-src", "KitNET-py")
    )
    if p not in sys.path:
        sys.path.insert(0, p)
    # Restore numpy<2 aliases so unmodified KitNET runs on numpy 2.x.
    import numpy as _np
    if not hasattr(_np, "Inf"):
        _np.Inf = _np.inf
    if not hasattr(_np, "NaN"):
        _np.NaN = _np.nan


def run_kitnet(
    X: np.ndarray,
    fm_grace: int,
    ad_grace: int,
    seed: int = 0,
    max_ae: int = 10,
    learning_rate: float = 0.1,
    hidden_ratio: float = 0.75,
) -> np.ndarray:
    """Run KitNET over X, return per-row RMSE.

    Rows 0..(fm_grace + ad_grace - 1) are training; KitNET returns 0.0
    for those. From `fm_grace + ad_grace` onwards the values are real
    RMSEs.
    """
    _ensure_kitnet_path()
    np.random.seed(seed)
    import KitNET as kit  # noqa
    model = kit.KitNET(
        X.shape[1], max_ae, fm_grace, ad_grace,
        learning_rate=learning_rate, hidden_ratio=hidden_ratio,
    )
    rmse = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        rmse[i] = model.process(X[i, :])
    return rmse


def headline_metrics(
    y_true: np.ndarray,
    rmse: np.ndarray,
    eval_start: int,
) -> dict:
    """Paper-style and attack-side metrics on the eval slice."""
    from sklearn.metrics import (
        roc_auc_score,
        roc_curve,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        balanced_accuracy_score,
    )

    y = y_true[eval_start:]
    s = rmse[eval_start:]
    finite = np.isfinite(s)
    if not finite.all():
        s = np.where(finite, s, np.where(s > 0, 1e300, -1e300))

    auc = float(roc_auc_score(y, s))

    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)

    def tpr_at_fpr(target_fpr: float) -> float:
        ok = fpr <= target_fpr
        if not ok.any():
            return float("nan")
        return float(tpr[ok].max())
    tpr_at_0 = tpr_at_fpr(0.0)
    tpr_at_001 = tpr_at_fpr(0.001)

    benign_scores = s[y == 0]
    if benign_scores.size > 0:
        thr_val = float(np.percentile(benign_scores, 99.9))
    else:
        thr_val = float(np.median(s))
    pred = (s >= thr_val).astype(np.int64)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    n_b = tn + fp
    n_a = tp + fn
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    attack_recall = tp / max(1, n_a)
    attack_precision = tp / max(1, tp + fp)
    attack_f1 = 2 * attack_precision * attack_recall / max(
        1e-12, attack_precision + attack_recall
    )
    fpr_at_op = fp / max(1, n_b)

    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y, pred)

    return dict(
        auc=auc,
        eer=eer,
        tpr_at_fpr0=tpr_at_0,
        tpr_at_fpr_001=tpr_at_001,
        threshold=float(thr_val),
        accuracy=float(accuracy),
        attack_recall=float(attack_recall),
        attack_precision=float(attack_precision),
        attack_f1=float(attack_f1),
        fpr=float(fpr_at_op),
        macro_f1=float(macro_f1),
        balanced_accuracy=float(bal_acc),
        tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
        n_eval_benign=int(n_b), n_eval_attack=int(n_a),
        pred_dist=dict(Counter(pred.tolist())),
    )
