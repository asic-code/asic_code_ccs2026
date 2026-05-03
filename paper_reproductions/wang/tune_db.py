"""Sweep DB hyperparameters (sigma, N) and BIM variants on full NSL-KDD."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from data_loader import load_nsl_kdd, preprocess
from ids_model import IDSModel
from attacks import generate_aes
from manifold_model import ManifoldModel
from manda import compute_score1, compute_score2, _model_proba, train_manda_lr, manda_score


def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def main(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")
    tr, te = load_nsl_kdd(args.data_dir)
    d = preprocess(tr, te)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    diff_idx = d["diff_idx"]
    non_diff_idx = d["non_diff_idx"]
    feature_ranges = d["feature_ranges"]

    ckpt = torch.load(args.ids_ckpt, map_location=device)
    model = IDSModel(ckpt["in_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).argmax(1).cpu().numpy()
    correct_mask = preds == y_test
    Xc = X_test[correct_mask]
    yc = y_test[correct_mask]

    manifold = ManifoldModel(n_fit=args.manifold_fit, kernel="knn",
                             n_neighbors=args.manifold_k, alpha=0.2,
                             random_state=args.seed)
    manifold.fit(X_train, y_train)

    attack_data = {}
    for attack in args.attacks:
        print(f"[gen] {attack} ...")
        kwargs = {}
        if attack == "bim":
            kwargs = {"n_steps": args.bim_steps, "alpha_frac": args.bim_alpha_frac}
        if attack == "bim_early":
            kwargs = {"n_steps": args.bim_steps, "alpha_frac": args.bim_alpha_frac,
                      "early_stop": True}
            x_adv, success = bim_early_stop(model, Xc, yc, diff_idx, non_diff_idx,
                                            feature_ranges, p=args.p,
                                            n_steps=kwargs["n_steps"],
                                            alpha_frac=kwargs["alpha_frac"],
                                            device=device)
        else:
            x_adv, success = generate_aes(model, Xc, yc, attack, diff_idx,
                                          non_diff_idx, feature_ranges,
                                          p=args.p, device=device, **kwargs)
        n_succ = int(success.sum())
        if n_succ == 0:
            print(f"      no successful AEs for {attack}")
            continue
        x_ae = x_adv[success]
        print(f"      {attack}: success={success.mean():.4f}  n_successful={n_succ}")

        rng = np.random.default_rng(args.seed + hash(attack) % 1000)
        clean_idx = rng.choice(Xc.shape[0], size=min(n_succ, Xc.shape[0]), replace=False)
        X_det = np.concatenate([x_ae, Xc[clean_idx]], axis=0)
        y_det = np.concatenate([np.ones(n_succ), np.zeros(len(clean_idx))]).astype(np.int64)
        perm = rng.permutation(len(X_det))
        X_det = X_det[perm]
        y_det = y_det[perm]

        ids_probs = _model_proba(model, X_det, device)
        manifold_probs = manifold.predict_proba(X_det)
        s1 = compute_score1(manifold_probs, ids_probs)

        attack_data[attack] = {"X_det": X_det, "y_det": y_det, "s1": s1}

    results = []
    for attack, d_ in attack_data.items():
        X_det = d_["X_det"]
        y_det = d_["y_det"]
        s1 = d_["s1"]
        n = len(X_det)
        n_tr = n // 2
        for sigma in args.sigmas:
            for N in args.noise_ns:
                s2 = compute_score2(model, X_det, sigma=sigma, N=N, device=device)
                s1_tr, s1_te = s1[:n_tr], s1[n_tr:]
                s2_tr, s2_te = s2[:n_tr], s2[n_tr:]
                y_tr, y_te = y_det[:n_tr], y_det[n_tr:]
                lr = train_manda_lr(s1_tr, s2_tr, y_tr)
                manda_sc = manda_score(lr, s1_te, s2_te)

                row = {
                    "attack": attack, "sigma": sigma, "N": N,
                    "manifold_auc": float(roc_auc_score(y_te, s1_te)),
                    "manifold_tpr5": tpr_at_fpr(y_te, s1_te, 0.05),
                    "manifold_tpr15": tpr_at_fpr(y_te, s1_te, 0.15),
                    "db_auc": float(roc_auc_score(y_te, s2_te)),
                    "db_tpr5": tpr_at_fpr(y_te, s2_te, 0.05),
                    "db_tpr15": tpr_at_fpr(y_te, s2_te, 0.15),
                    "manda_auc": float(roc_auc_score(y_te, manda_sc)),
                    "manda_tpr5": tpr_at_fpr(y_te, manda_sc, 0.05),
                    "manda_tpr15": tpr_at_fpr(y_te, manda_sc, 0.15),
                }
                print(f"{attack} sigma={sigma:<6} N={N:<4} "
                      f"Manif AUC={row['manifold_auc']:.4f} "
                      f"DB AUC={row['db_auc']:.4f} (T5={row['db_tpr5']*100:.1f}%) "
                      f"MANDA AUC={row['manda_auc']:.4f} (T5={row['manda_tpr5']*100:.1f}%)")
                results.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out}")


def bim_early_stop(model, x_np, y_np, diff_idx, non_diff_idx, feature_ranges,
                   p=0.05, n_steps=20, alpha_frac=0.1, device="cpu"):
    """BIM that stops perturbing each sample once its prediction flips."""
    model.eval()
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    diff_idx_t = torch.from_numpy(diff_idx).to(device) if isinstance(diff_idx, np.ndarray) else diff_idx
    non_diff_idx_t = torch.from_numpy(non_diff_idx).to(device) if isinstance(non_diff_idx, np.ndarray) else non_diff_idx
    feature_ranges_t = torch.from_numpy(feature_ranges).to(device) if isinstance(feature_ranges, np.ndarray) else feature_ranges

    alpha = alpha_frac * p * feature_ranges_t
    x_adv = x.clone()
    flipped = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="none")
        grad = torch.autograd.grad(loss.sum(), x_adv)[0]
        step = alpha * grad.sign()
        step[flipped] = 0.0
        x_new = x_adv.detach() + step
        x_new[:, non_diff_idx_t] = x[:, non_diff_idx_t]
        delta = x_new[:, diff_idx_t] - x[:, diff_idx_t]
        per_feat_bound = p * feature_ranges_t[diff_idx_t]
        delta = torch.clamp(delta, -per_feat_bound, per_feat_bound)
        x_new[:, diff_idx_t] = x[:, diff_idx_t] + delta
        x_new = torch.clamp(x_new, 0.0, 1.0)
        x_adv = x_new
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
            flipped = flipped | (preds != y)
        if flipped.all():
            break
    with torch.no_grad():
        preds = model(x_adv).argmax(1).cpu().numpy()
    success = preds != y_np
    return x_adv.detach().cpu().numpy(), success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ids-ckpt", default="ids_model.pt")
    parser.add_argument("--out", default="out/tune_results.json")
    parser.add_argument("--manifold-fit", type=int, default=10000)
    parser.add_argument("--manifold-k", type=int, default=10)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--bim-steps", type=int, default=10)
    parser.add_argument("--bim-alpha-frac", type=float, default=0.2)
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--noise-ns", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--attacks", nargs="+",
                        default=["fgsm", "bim", "cw", "bim_early"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
