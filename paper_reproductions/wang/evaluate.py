"""End-to-end MANDA evaluation on the full NSL-KDD dataset.

Pipeline:
  1. Load + preprocess full NSL-KDD (DoS vs Normal).
  2. Train IDS MLP (or load from checkpoint).
  3. Fit manifold model (LabelSpreading on stratified subsample; eval on all).
  4. For each attack in {FGSM, BIM, CW}:
       a. Generate AEs from all correctly-classified test samples (p=5%).
       b. Keep only successful AEs (prediction flipped).
       c. Build mixed set: AEs (label=1) + equal number of random clean samples (label=0).
       d. Split 50/50 into train/test for MANDA's logistic regression.
       e. Compute score1 (manifold) and score2 (DB uncertainty) on all samples.
       f. Train LR on train split, evaluate Manifold/DB/MANDA on test split.
       g. Report AUC-ROC, TPR@5%FPR, TPR@15%FPR.
  5. Save ROC curve plot + results table.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from data_loader import load_nsl_kdd, preprocess
from ids_model import IDSModel, train_ids
from attacks import generate_aes
from manifold_model import ManifoldModel
from manda import (
    compute_score1, compute_score2, _model_proba,
    train_manda_lr, manda_score,
)
from metrics import metrics_at_fpr


def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Largest TPR at fpr <= target_fpr
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def main(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading + preprocessing NSL-KDD (FULL dataset)")
    tr, te = load_nsl_kdd(args.data_dir)
    d = preprocess(tr, te)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    diff_idx = d["diff_idx"]
    non_diff_idx = d["non_diff_idx"]
    feature_ranges = d["feature_ranges"]
    print(f"      X_train {X_train.shape}, X_test {X_test.shape}, n_features {d['n_features']}")

    print("[2/4] training IDS MLP")
    ckpt_path = Path(args.ids_ckpt)
    if ckpt_path.exists() and not args.force_retrain:
        ckpt = torch.load(ckpt_path, map_location=device)
        model = IDSModel(ckpt["in_dim"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"      loaded checkpoint from {ckpt_path}")
    else:
        model = train_ids(X_train, y_train, X_test, y_test,
                          epochs=args.ids_epochs, device=device, seed=args.seed)
        torch.save({"state_dict": model.state_dict(), "in_dim": d["n_features"]}, ckpt_path)

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).argmax(1).cpu().numpy()
    acc_clean = (preds == y_test).mean()
    print(f"      IDS clean test acc: {acc_clean:.4f}  (paper: 0.9064)")
    correct_mask = preds == y_test
    Xc = X_test[correct_mask]
    yc = y_test[correct_mask]
    print(f"      correctly classified test samples for AE generation: {Xc.shape[0]}")

    print("[3/4] fitting manifold model (LabelSpreading, stratified subsample)")
    manifold = ManifoldModel(
        n_fit=args.manifold_fit,
        kernel="knn",
        n_neighbors=args.manifold_k,
        alpha=0.2,
        random_state=args.seed,
    )
    manifold.fit(X_train, y_train)
    # Sanity: manifold accuracy on a held-out subsample of test
    sample_idx = np.random.default_rng(args.seed).choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    m_probs = manifold.predict_proba(X_test[sample_idx])
    m_acc = (m_probs.argmax(1) == y_test[sample_idx]).mean()
    print(f"      manifold clean acc on 2k test sample: {m_acc:.4f}")

    results = {}

    for attack in args.attacks:
        print(f"\n[4/4] === attack: {attack.upper()} (p={args.p}) ===")

        print(f"      generating AEs on {Xc.shape[0]} correctly-classified test samples")
        x_adv, success = generate_aes(
            model, Xc, yc, attack, diff_idx, non_diff_idx, feature_ranges,
            p=args.p, device=device,
        )
        sr = success.mean()
        post_acc = 1 - sr
        print(f"      attack success rate: {sr:.4f}  |  IDS acc on ALL AEs: {post_acc:.4f}")

        # Keep only successful AEs (as per paper: 'combine the successful AEs ...')
        x_ae = x_adv[success]
        y_ae_orig = yc[success]  # original true labels of the AEs
        n_ae = x_ae.shape[0]
        if n_ae == 0:
            print("      no successful AEs — skipping")
            continue
        print(f"      successful AEs: {n_ae}")

        # Build mixed detection dataset: AEs (1) + equal number of random clean test samples (0).
        # Clean samples drawn from the correctly-classified test set.
        rng = np.random.default_rng(args.seed + hash(attack) % 1000)
        clean_pool_idx = np.arange(Xc.shape[0])
        n_clean = min(n_ae, len(clean_pool_idx))
        clean_idx = rng.choice(clean_pool_idx, size=n_clean, replace=False)
        x_clean = Xc[clean_idx]

        X_det = np.concatenate([x_ae, x_clean], axis=0)
        y_det = np.concatenate([np.ones(n_ae, dtype=np.int64), np.zeros(n_clean, dtype=np.int64)])

        # Shuffle
        perm = rng.permutation(len(X_det))
        X_det = X_det[perm]
        y_det = y_det[perm]

        # Compute scores on the full detection set
        print(f"      computing score1 (manifold) + score2 (DB) on {len(X_det)} samples")
        ids_probs = _model_proba(model, X_det, device)
        manifold_probs = manifold.predict_proba(X_det)
        s1 = compute_score1(manifold_probs, ids_probs)
        s2 = compute_score2(model, X_det, sigma=args.sigma, N=args.noise_n, device=device)

        # Train/test split 50/50
        n = len(X_det)
        n_tr = n // 2
        tr_sl = slice(0, n_tr)
        te_sl = slice(n_tr, n)
        s1_tr, s1_te = s1[tr_sl], s1[te_sl]
        s2_tr, s2_te = s2[tr_sl], s2[te_sl]
        y_tr, y_te = y_det[tr_sl], y_det[te_sl]

        lr = train_manda_lr(s1_tr, s2_tr, y_tr)
        manda_sc_te = manda_score(lr, s1_te, s2_te)

        def metrics(name, scores):
            auc = roc_auc_score(y_te, scores)
            t5 = tpr_at_fpr(y_te, scores, 0.05)
            t15 = tpr_at_fpr(y_te, scores, 0.15)
            cls5 = metrics_at_fpr(y_te, scores, 0.05)
            print(f"      {name:10s}  AUC={auc:.4f}  TPR@5%={t5*100:.2f}%  "
                  f"TPR@15%={t15*100:.2f}%  | @FPR=5%: Acc={cls5['accuracy']*100:.2f}%  "
                  f"Prec={cls5['precision']*100:.2f}%  Rec={cls5['recall']*100:.2f}%  "
                  f"F1={cls5['f1']*100:.2f}%")
            return {"auc": auc, "tpr_at_5": t5, "tpr_at_15": t15,
                    "at_fpr_5": cls5}

        attack_results = {
            "manifold": metrics("Manifold", s1_te),
            "db":       metrics("DB", s2_te),
            "manda":    metrics("MANDA", manda_sc_te),
            "attack_success_rate": float(sr),
            "clean_ids_acc": float(acc_clean),
            "n_ae": int(n_ae),
            "n_detection_test": int(n - n_tr),
        }
        results[attack] = attack_results

        # ROC plot
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        for label, scores in [("Manifold", s1_te), ("DB", s2_te), ("MANDA", manda_sc_te)]:
            fpr, tpr, _ = roc_curve(y_te, scores)
            auc = roc_auc_score(y_te, scores)
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{attack.upper()} (NSL-KDD)")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_{attack}.png", dpi=150)
        plt.close(fig)

    # Write results JSON
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 72)
    print("SUMMARY (NSL-KDD, full dataset, DoS vs Normal, p=5%)")
    print("=" * 72)
    print(f"{'Attack':<8}{'Method':<10}{'AUC':>10}{'TPR@5%':>10}{'TPR@15%':>10}  Paper")
    paper_table = {
        "fgsm": {"manifold": (0.9792, 94.04, 100.00),
                 "db":       (0.8471, 17.27, 53.57),
                 "manda":    (0.9765, 92.88, 99.89)},
        "bim":  {"manifold": (0.9714, 98.38, 100.00),
                 "db":       (0.9340, 71.00, 97.91),
                 "manda":    (0.9726, 95.93, 100.00)},
        "cw":   {"manifold": (0.9805, 98.41, 99.98),
                 "db":       (0.9439, 27.91, 98.62),
                 "manda":    (0.9851, 98.04, 100.00)},
    }
    for attack in args.attacks:
        if attack not in results:
            continue
        for method in ["manifold", "db", "manda"]:
            r = results[attack][method]
            p = paper_table.get(attack, {}).get(method)
            paper_str = f"(paper AUC={p[0]:.4f}, T5={p[1]:.2f}, T15={p[2]:.2f})" if p else ""
            print(f"{attack:<8}{method:<10}{r['auc']:>10.4f}{r['tpr_at_5']*100:>9.2f}%{r['tpr_at_15']*100:>9.2f}%  {paper_str}")

    # Classification metrics at FPR=5%
    print("\n" + "=" * 72)
    print("Classification metrics @ FPR=5% (balanced detection set)")
    print("=" * 72)
    print(f"{'Attack':<8}{'Method':<10}{'Acc':>10}{'Prec':>10}{'Recall':>10}{'F1':>10}")
    for attack in args.attacks:
        if attack not in results:
            continue
        for method in ["manifold", "db", "manda"]:
            m = results[attack][method].get("at_fpr_5", {})
            print(f"{attack:<8}{method:<10}"
                  f"{m.get('accuracy',0)*100:>9.2f}%"
                  f"{m.get('precision',0)*100:>9.2f}%"
                  f"{m.get('recall',0)*100:>9.2f}%"
                  f"{m.get('f1',0)*100:>9.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--ids-ckpt", default="ids_model.pt")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--ids-epochs", type=int, default=30)
    parser.add_argument("--manifold-fit", type=int, default=10000,
                        help="subsample size for LabelSpreading fit (eval is over full test set)")
    parser.add_argument("--manifold-k", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--noise-n", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.05, help="per-feature perturbation budget")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "bim", "cw"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
