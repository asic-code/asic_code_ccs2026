"""Runs the official Mateen pipeline given pre-loaded train/test data.

Wraps `MateenUtils.main.adaptive_ensemble` so we can:
  - inject device (MPS) before any model construction
  - re-seed deterministically per trial
  - skip the disk-load step (pass data as in-memory arrays)

Returns predictions + per-window probabilities + headline metrics.
"""
from __future__ import annotations
import argparse
import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import device as device_mod  # patches the official `device` constants
from data_loader import Split, partition_test


# ----------- args object that matches the official CLI ------------- #
@dataclass
class MateenArgs:
    # Per Appendix B.4 (Kitsune): ρ=1500, σ=50%, λ₀=0.1, ensemble=3.
    dataset_name: str = "Kitsune"
    window_size: int = 50_000
    performance_thres: float = 0.99
    max_ensemble_length: int = 3
    selection_budget: float = 0.01
    mini_batch_size: int = 1500
    retention_rate: float = 0.50
    lambda_0: float = 0.1
    shift_threshold: float = 0.05


# ----------- headline metrics (paper convention + attack-side) ----- #
def headline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute BOTH the paper's metrics (positive=benign) and proper
    attack-detection metrics (positive=attack).

    The paper's `getResult` swaps the confusion matrix so the
    "positive" class is benign. That is what its F1/Acc/mF1 numbers in
    Table 2 are. We reproduce those for paper-vs-ours comparison.

    But on a 65/35 benign-skewed test set, "high benign-F1" can be hit
    by a near-trivial always-benign classifier — so we ALSO compute
    attack-as-positive Recall/Precision/F1 + the FPR / TPR pair, plus
    the prediction-distribution counts, so we can spot collapse to
    one-class.
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        balanced_accuracy_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm is [[tn_attack, fp_attack], [fn_attack, tp_attack]] with the
    # convention that label=1 is the positive class. Equivalently:
    #   tn (true benign) = cm[0,0]
    #   fp (false attack flag on benign) = cm[0,1]
    #   fn (missed attack)               = cm[1,0]
    #   tp (caught attack)               = cm[1,1]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    n_benign = tn + fp
    n_attack = tp + fn
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    # Attack-as-positive (the metrics we actually care about for IDS):
    attack_recall = tp / max(1, n_attack)            # = TPR
    attack_precision = tp / max(1, tp + fp)
    attack_f1 = 2 * attack_precision * attack_recall / max(
        1e-12, attack_precision + attack_recall
    )
    fpr = fp / max(1, n_benign)                      # benign mis-flagged

    # Benign-as-positive (paper convention: maps to its F1 / mF1 / Acc):
    benign_recall = tn / max(1, n_benign)            # = 1 - FPR
    benign_precision = tn / max(1, tn + fn)
    benign_f1 = 2 * benign_precision * benign_recall / max(
        1e-12, benign_precision + benign_recall
    )

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return dict(
        # Paper's headline (benign as positive):
        accuracy=float(accuracy),
        f1_paper=float(benign_f1),                # paper "F1"
        macro_f1=float(macro_f1),                  # paper "mF1"
        # Attack-detection (the IDS-meaningful ones):
        attack_recall=float(attack_recall),       # TPR
        attack_precision=float(attack_precision),
        attack_f1=float(attack_f1),
        fpr=float(fpr),
        tpr=float(attack_recall),
        # Benign-side (companion to attack-side):
        benign_recall=float(benign_recall),
        benign_precision=float(benign_precision),
        benign_f1=float(benign_f1),
        # Macro:
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        balanced_accuracy=float(bal_acc),
        # Raw counts (so the reader can verify):
        tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
        n_benign=int(n_benign), n_attack=int(n_attack),
    )


def auc_roc_per_chunk(y_true: np.ndarray, probs: np.ndarray, chunk: int = 50_000) -> float:
    """Reproduce `utils.auc_roc_in_chunks` (paper's AUC-ROC).

    On small-fraction training, the DAE can produce inf RMSE for some
    test samples (MinMax extrapolation × overflow). roc_auc_score
    rejects inf — we clip to a large finite value, which preserves the
    rank order that AUC depends on.
    """
    from sklearn.metrics import roc_auc_score
    finite_max = 1e30
    probs = np.where(np.isfinite(probs), probs,
                     np.where(probs > 0, finite_max, -finite_max))
    n = len(y_true)
    n_chunks = n // chunk + (1 if n % chunk else 0)
    scores = []
    for i in range(n_chunks):
        s = i * chunk
        e = s + chunk
        yt = y_true[s:e]
        pp = probs[s:e]
        if set(np.unique(yt)) == {0, 1}:
            scores.append(roc_auc_score(yt, pp))
    return float(np.mean(scores)) if scores else float("nan")


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ----------- run modes --------------------------------------------- #
def run_no_update(split: Split, args: Optional[MateenArgs] = None,
                  init_epochs: int = 100, seed: int = 0) -> dict:
    """Train DAE once on benign-train, evaluate on full test (no
    adaptation). This is the paper's `No-Update` baseline."""
    args = args or MateenArgs()
    _set_seeds(seed)
    device_mod.patch_official_device()
    import AE as model_base
    import data_processing as dp
    import utils as mu

    benign_train = split.x_train[split.y_train == 0]
    train_loader, _ = dp.loading_datasets(benign_train)
    model = model_base.autoencoder(split.feature_dim)
    model = model_base.train_autoencoder(model, train_loader,
                                         num_epochs=init_epochs,
                                         learning_rate=1e-4)
    threshold = mu.threshold_calulation(model, benign_train)

    xs, ys = partition_test(split.x_test, split.y_test, args.window_size)
    preds, probs = [], []
    for xi in xs:
        yp, pp = mu.preds_and_probs(model, threshold, xi)
        preds.extend(yp.tolist())
        probs.extend(pp.tolist())
    preds = np.array(preds)
    probs = np.array(probs, dtype=np.float64)
    y_true = split.y_test[:len(preds)]
    out = headline_metrics(y_true, preds)
    out["auc_roc"] = auc_roc_per_chunk(y_true, probs, args.window_size)
    out["mode"] = "no_update"
    out["init_epochs"] = init_epochs
    out["pred_dist"] = dict(Counter(preds.tolist()))
    return out


def run_mateen(split: Split, args: Optional[MateenArgs] = None,
               init_epochs: int = 100, seed: int = 0) -> dict:
    """Run the full Mateen adaptive ensemble (paper's headline)."""
    args = args or MateenArgs()
    _set_seeds(seed)
    device_mod.patch_official_device()
    # Bump QoS so the macOS scheduler doesn't deprioritize us when we
    # run as a backgrounded subprocess. Without this, MPS work can stall
    # for minutes at a time on long-running training.
    try:
        import os, ctypes
        # taskpolicy-like QoS bump
        libc = ctypes.CDLL(None)
        libc.pthread_set_qos_class_self_np(0x21, 0)  # QOS_CLASS_USER_INITIATED
    except Exception:
        pass
    import main as mateen_main

    xs, ys = partition_test(split.x_test, split.y_test, args.window_size)

    # Monkeypatch `ensemble_training` to honor our init_epochs without
    # rewriting the official code.
    orig_init_epochs = mateen_main.adaptive_ensemble.__defaults__
    # Actually the official code hardcodes 100 in adaptive_ensemble's call
    # to ensemble_training(... num_epochs=100). Override by inlining.
    preds, probs = _adaptive_ensemble_with_epochs(
        mateen_main, split.x_train, split.y_train, xs, ys, args, init_epochs
    )

    preds = np.array(preds)
    probs = np.array(probs, dtype=np.float64)
    y_true = split.y_test[:len(preds)]
    out = headline_metrics(y_true, preds)
    out["auc_roc"] = auc_roc_per_chunk(y_true, probs, args.window_size)
    out["mode"] = "mateen"
    out["init_epochs"] = init_epochs
    out["pred_dist"] = dict(Counter(preds.tolist()))
    return out


def _adaptive_ensemble_with_epochs(mateen_main, x_train, y_train, x_slice,
                                    y_slice, args, init_epochs: int):
    """Mirror of `mateen_main.adaptive_ensemble` but with parameterized
    init_epochs (the official code hardcodes 100).
    """
    import utils as mu
    model = mateen_main.ensemble_training(
        x_train, y_train=y_train, num_epochs=init_epochs,
        mode="init", scenario=args.dataset_name, load_mode="new",
    )
    benign_train = x_train[y_train == 0]
    selected_threshold = mu.threshold_calulation(model, benign_train)
    predicitons = []
    probs_list = []
    print("Updating Models Process Started!")
    models_list = [model]
    threshold_list = [selected_threshold]
    selected_model = model
    for i in range(len(x_slice)):
        print(f"Step {i+1}/{len(x_slice)}")
        y_pred, probs = mu.preds_and_probs(selected_model, selected_threshold,
                                            x_slice[i])
        _, old_probs = mu.preds_and_probs(
            selected_model,
            selected_threshold,
            benign_train[-len(x_slice[i]):],
        )
        predicitons.extend(y_pred)
        probs_list.extend(probs)
        data_slice = x_slice[i]
        label_slice = y_slice[i]
        if i + 1 == len(x_slice):
            return predicitons, probs_list
        if mateen_main.isit_shift(old_probs, probs, args.shift_threshold):
            probs_vector = mu.get_features_error(selected_model, x_slice[i])
            (models_list, threshold_list, selected_model,
             selected_threshold, benign_train, x_train,
             y_train) = mateen_main.select_and_adapt(
                probs, probs_vector, data_slice, label_slice, models_list,
                threshold_list, benign_train, selected_model, y_pred,
                selected_threshold, x_train, y_train, args,
            )
    return predicitons, probs_list
