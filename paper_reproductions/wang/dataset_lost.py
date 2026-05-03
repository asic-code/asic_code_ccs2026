"""Dataset-Lost experiment for MANDA / NSL-KDD."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from data_loader import load_nsl_kdd, preprocess, DOS_ATTACKS
from ids_model import IDSModel, train_ids
from attacks import generate_aes
from manifold_model import ManifoldModel
from manda import (
    compute_score1, compute_score2, _model_proba,
    train_manda_lr, manda_score,
)
from metrics import metrics_at_fpr


def _filter_dos_normal_raw(df):
    mask = df["label"].apply(lambda x: x == "normal" or x in DOS_ATTACKS)
    return df[mask].reset_index(drop=True)


def tpr_at_fpr(y_true, y_score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def run_trial(train_raw, test_raw, frac, seed, config, device,
              ref_test_hash=None):
    """Run one (fraction, seed) trial. Each trial subsamples the
    raw train rows, fits its own scaler/encoder, applies to full test."""
    rng = np.random.default_rng(int(seed * 1_000_003 + int(frac * 1_000_000)))
    n_full = len(train_raw)
    k = max(1, int(round(n_full * frac)))
    idx = rng.choice(n_full, size=k, replace=False)
    train_sub = train_raw.iloc[idx].reset_index(drop=True)

    d_run = preprocess(train_sub, test_raw)
    X_tr, y_tr = d_run["X_train"], d_run["y_train"]
    X_test, y_test = d_run["X_test"], d_run["y_test"]
    d_meta = d_run

    if ref_test_hash is not None:
        test_identity = hashlib.sha1(y_test.tobytes()).hexdigest()
        assert test_identity == ref_test_hash, (
            f"Test row identity drifted between runs! "
            f"got {test_identity[:12]}, expected {ref_test_hash[:12]}"
        )

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    result = {
        "frac": frac, "seed": seed, "n_train": int(k),
        "n_pos_train": n_pos, "n_neg_train": n_neg,
        "pos_rate_train": (n_pos / k) if k else 0.0,
        "n_features_this_run": int(X_tr.shape[1]),
    }

    if n_pos == 0 or n_neg == 0:
        result["status"] = "degenerate_monoclass"
        return result

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = IDSModel(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    n = Xt.shape[0]
    batch = min(256, max(4, n // 4))
    for epoch in range(config["ids_epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            sl = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xt[sl]), yt[sl])
            loss.backward()
            opt.step()
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).argmax(1).cpu().numpy()
    acc_clean = float((preds == y_test).mean())
    result["ids_clean_acc"] = acc_clean
    correct_mask = preds == y_test
    Xc = X_test[correct_mask]
    yc = y_test[correct_mask]
    result["n_correct_test"] = int(correct_mask.sum())

    attack_mask = (y_test == 1)
    benign_mask = (y_test == 0)
    n_attack = int(attack_mask.sum())
    n_benign = int(benign_mask.sum())
    ids_tpr = float((preds[attack_mask] == 1).mean()) if n_attack else 0.0
    ids_fpr = float((preds[benign_mask] == 1).mean()) if n_benign else 0.0
    result["ids_n_attack"] = n_attack
    result["ids_n_benign"] = n_benign
    result["ids_attack_recall"] = ids_tpr
    result["ids_fpr"] = ids_fpr
    result["ids_pred_dist"] = {
        "0": int((preds == 0).sum()),
        "1": int((preds == 1).sum()),
    }
    t_ids = time.time() - t0

    manifold_fit = min(k, config["manifold_cap"])
    k_neighbors = max(1, min(config["manifold_k"], manifold_fit - 1))
    try:
        manifold = ManifoldModel(n_fit=manifold_fit, kernel="knn",
                                 n_neighbors=k_neighbors, alpha=0.2,
                                 random_state=seed)
        manifold.fit(X_tr, y_tr)
    except Exception as e:
        result["status"] = f"manifold_fit_failed: {e}"
        return result
    t_manifold = time.time() - t0 - t_ids

    result["attacks"] = {}
    diff_idx = d_meta["diff_idx"]
    non_diff_idx = d_meta["non_diff_idx"]
    feature_ranges = d_meta["feature_ranges"]

    for attack in config["attacks"]:
        a_start = time.time()
        kwargs = {}
        if attack == "bim":
            kwargs = {"early_stop": False, "n_steps": 10, "alpha_frac": 0.2}
        x_adv, success = generate_aes(model, Xc, yc, attack,
                                      diff_idx, non_diff_idx, feature_ranges,
                                      p=config["p"], device=device, **kwargs)
        n_succ = int(success.sum())
        sr = float(success.mean())
        if n_succ > 0:
            x_ae_for_norm = x_adv[success]
            xc_for_norm = Xc[success]
            delta = x_ae_for_norm - xc_for_norm
            perturb_l_inf = float(np.abs(delta).max(axis=1).mean())
            perturb_l2 = float(np.linalg.norm(delta, axis=1).mean())
        else:
            perturb_l_inf = float("nan")
            perturb_l2 = float("nan")
        a_out = {
            "attack_success_rate": sr,
            "n_ae": n_succ,
            "perturb_l_inf_mean": perturb_l_inf,
            "perturb_l2_mean": perturb_l2,
        }
        if n_succ < 20:
            a_out["status"] = f"too_few_ae ({n_succ})"
            result["attacks"][attack] = a_out
            continue

        x_ae = x_adv[success]
        rng2 = np.random.default_rng(int(seed * 7919 + (hash(attack) % 10_000_019)))
        n_clean = min(n_succ, Xc.shape[0])
        clean_idx = rng2.choice(Xc.shape[0], size=n_clean, replace=False)
        X_det = np.concatenate([x_ae, Xc[clean_idx]], axis=0)
        y_det = np.concatenate([np.ones(n_succ), np.zeros(n_clean)]).astype(np.int64)
        perm = rng2.permutation(len(X_det))
        X_det = X_det[perm]
        y_det = y_det[perm]

        ids_probs = _model_proba(model, X_det, device)
        manifold_probs = manifold.predict_proba(X_det)
        s1 = compute_score1(manifold_probs, ids_probs)
        s2 = compute_score2(model, X_det, sigma=config["sigma"],
                            N=config["noise_n"], device=device)

        m = len(X_det); m_tr = m // 2
        s1_tr, s1_te = s1[:m_tr], s1[m_tr:]
        s2_tr, s2_te = s2[:m_tr], s2[m_tr:]
        y_trL, y_teL = y_det[:m_tr], y_det[m_tr:]
        if len(np.unique(y_trL)) < 2 or len(np.unique(y_teL)) < 2:
            a_out["status"] = "single_class_in_split"
            result["attacks"][attack] = a_out
            continue
        lr = train_manda_lr(s1_tr, s2_tr, y_trL)
        manda_sc = manda_score(lr, s1_te, s2_te)

        for method, scores in [("manifold", s1_te), ("db", s2_te), ("manda", manda_sc)]:
            auc = float(roc_auc_score(y_teL, scores))
            t5 = tpr_at_fpr(y_teL, scores, 0.05)
            cls5 = metrics_at_fpr(y_teL, scores, 0.05)
            a_out[method] = {
                "auc": auc, "tpr_at_5": t5,
                "accuracy": cls5["accuracy"], "precision": cls5["precision"],
                "recall": cls5["recall"], "f1": cls5["f1"],
            }
        a_out["elapsed_sec"] = time.time() - a_start
        result["attacks"][attack] = a_out

    result["status"] = "ok"
    result["elapsed_sec"] = {"ids": t_ids, "manifold": t_manifold,
                             "total": time.time() - t0}
    return result


def aggregate(trials, fractions, attacks, methods):
    agg = defaultdict(list)
    ids_acc = defaultdict(list)
    ids_tpr = defaultdict(list)
    ids_fpr = defaultdict(list)
    attack_sr = defaultdict(list)
    perturb = defaultdict(list)
    n_valid = defaultdict(int)
    n_degenerate = defaultdict(int)
    pos_rate = defaultdict(list)
    for t in trials:
        f = t["frac"]
        pos_rate[f].append(t["pos_rate_train"])
        if t.get("status") != "ok":
            n_degenerate[f] += 1
            continue
        n_valid[f] += 1
        ids_acc[f].append(t["ids_clean_acc"])
        if "ids_attack_recall" in t:
            ids_tpr[f].append(t["ids_attack_recall"])
        if "ids_fpr" in t:
            ids_fpr[f].append(t["ids_fpr"])
        for a in attacks:
            ar = t.get("attacks", {}).get(a, {})
            attack_sr[(f, a)].append(ar.get("attack_success_rate", float("nan")))
            if ar.get("perturb_l_inf_mean") is not None:
                perturb[(f, a, "l_inf")].append(ar["perturb_l_inf_mean"])
                perturb[(f, a, "l2")].append(ar.get("perturb_l2_mean",
                                                       float("nan")))
            for m in methods:
                if m not in ar:
                    continue
                for metric in ("auc", "tpr_at_5", "accuracy",
                               "precision", "recall", "f1"):
                    agg[(f, a, m, metric)].append(ar[m][metric])
    return agg, ids_acc, attack_sr, n_valid, n_degenerate, pos_rate, \
        ids_tpr, ids_fpr, perturb


def main(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}")

    tr_raw, te_raw = load_nsl_kdd(args.data_dir)
    tr_raw = _filter_dos_normal_raw(tr_raw)
    te_raw = _filter_dos_normal_raw(te_raw)

    y_test_ref = te_raw["label"].apply(
        lambda x: 1 if x in DOS_ATTACKS else 0
    ).values.astype(np.int64)
    ref_test_hash = hashlib.sha1(y_test_ref.tobytes()).hexdigest()
    print(f"[isolation] raw train rows (DoS+Normal): {len(tr_raw)}")
    print(f"[isolation] raw test  rows (DoS+Normal): {len(te_raw)}  "
          f"label-hash={ref_test_hash[:16]}")
    print(f"[isolation] train class counts: pos={int((tr_raw['label'].isin(DOS_ATTACKS)).sum())} "
          f"neg={int((tr_raw['label']=='normal').sum())}")
    print(f"[isolation] test  class counts: pos={int(y_test_ref.sum())} "
          f"neg={int((y_test_ref==0).sum())}")

    config = {
        "ids_epochs": args.ids_epochs,
        "manifold_cap": args.manifold_cap,
        "manifold_k": args.manifold_k,
        "sigma": args.sigma,
        "noise_n": args.noise_n,
        "p": args.p,
        "attacks": args.attacks,
    }
    print(f"[config] {config}")
    print(f"[config] fractions={args.fractions} seeds={args.seeds}")

    trials = []
    total_start = time.time()
    for frac in args.fractions:
        for seed in args.seeds:
            print(f"\n--- frac={frac*100:.2f}% seed={seed} ---")
            trial = run_trial(tr_raw, te_raw, frac, seed, config, device,
                              ref_test_hash=ref_test_hash)
            trials.append(trial)
            if trial.get("status") == "ok":
                print(f"  n_train={trial['n_train']} "
                      f"(pos={trial['n_pos_train']}, neg={trial['n_neg_train']}, "
                      f"pos_rate={trial['pos_rate_train']*100:.1f}%)  "
                      f"IDS_acc={trial['ids_clean_acc']*100:.2f}%")
                for a in args.attacks:
                    ar = trial["attacks"].get(a, {})
                    if "manda" in ar:
                        print(f"    {a}: sr={ar['attack_success_rate']:.3f} "
                              f"MANDA F1={ar['manda']['f1']*100:.2f}% "
                              f"Rec={ar['manda']['recall']*100:.2f}% "
                              f"Acc={ar['manda']['accuracy']*100:.2f}%")
                    else:
                        print(f"    {a}: {ar.get('status', 'n/a')}")
            else:
                print(f"  status={trial['status']} "
                      f"pos={trial['n_pos_train']} neg={trial['n_neg_train']}")

    print(f"\n[time] total elapsed: {(time.time()-total_start)/60:.1f} min")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset_lost_raw.json", "w") as f:
        json.dump(trials, f, indent=2)

    methods = ["manifold", "db", "manda"]
    (agg, ids_acc, attack_sr, n_valid, n_degenerate, pos_rate,
     ids_tpr, ids_fpr, perturb) = aggregate(
        trials, args.fractions, args.attacks, methods)

    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("\n" + "=" * 90)
    P("IDS CLEAN TEST ACCURACY (mean ± std across seeds)")
    P("=" * 90)
    P(f"{'frac':<8}{'n_train':<10}{'pos_rate':<12}{'valid':<8}{'degen':<8}"
      f"{'IDS_acc_mean':<14}{'IDS_acc_std':<14}{'range':<22}")
    for frac in args.fractions:
        vals = np.array(ids_acc[frac]) if len(ids_acc[frac]) else np.array([])
        nt = int(round(len(tr_raw) * frac))
        pr = np.mean(pos_rate[frac]) * 100 if pos_rate[frac] else 0.0
        if len(vals) > 0:
            P(f"{frac*100:<7.2f}%{nt:<10}{pr:<11.2f}%{n_valid[frac]:<8}{n_degenerate[frac]:<8}"
              f"{vals.mean()*100:<14.2f}{vals.std()*100:<14.2f}"
              f"[{vals.min()*100:.2f}, {vals.max()*100:.2f}]")
        else:
            P(f"{frac*100:<7.2f}%{nt:<10}{pr:<11.2f}%{n_valid[frac]:<8}{n_degenerate[frac]:<8} all-degenerate")

    P("\nATTACK SUCCESS RATE (mean across seeds)")
    P(f"{'frac':<8}{'FGSM':<10}{'BIM':<10}{'CW':<10}")
    for frac in args.fractions:
        row = f"{frac*100:<7.2f}%"
        for a in ["fgsm", "bim", "cw"]:
            if a not in args.attacks:
                row += f"{'-':<10}"; continue
            vals = [v for v in attack_sr[(frac, a)] if not np.isnan(v)]
            row += f"{np.mean(vals):<9.3f} " if vals else f"{'-':<10}"
        P(row)

    for metric_key, metric_label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("auc", "AUC-ROC"),
    ]:
        P(f"\n{metric_label.upper()} @ FPR=5% (mean ± std across seeds)")
        header = f"{'frac':<8}{'valid':<7}"
        for a in args.attacks:
            for m in methods:
                header += f"{(a+'_'+m):<18}"
        P(header)
        for frac in args.fractions:
            row = f"{frac*100:<7.2f}%{n_valid[frac]:<7}"
            for a in args.attacks:
                for m in methods:
                    vals = np.array(agg[(frac, a, m, metric_key)])
                    if len(vals) == 0:
                        row += f"{'-':<18}"
                        continue
                    if metric_key == "auc":
                        row += f"{vals.mean():.4f}±{vals.std():.4f}  "
                    else:
                        row += f"{vals.mean()*100:5.2f}±{vals.std()*100:4.2f}      "
            P(row)

    with open(out_dir / "dataset_lost_report.txt", "w") as f:
        f.write("\n".join(lines))

    def keystr(k):
        return "|".join(map(str, k))
    agg_save = {keystr(k): {"values": [float(x) for x in v],
                            "mean": float(np.mean(v)), "std": float(np.std(v)),
                            "n": len(v)} for k, v in agg.items()}
    with open(out_dir / "dataset_lost_agg.json", "w") as f:
        json.dump({
            "ids_acc": {str(k): list(map(float, v)) for k, v in ids_acc.items()},
            "ids_attack_recall": {str(k): list(map(float, v))
                                    for k, v in ids_tpr.items()},
            "ids_fpr": {str(k): list(map(float, v)) for k, v in ids_fpr.items()},
            "n_valid": {str(k): v for k, v in n_valid.items()},
            "n_degenerate": {str(k): v for k, v in n_degenerate.items()},
            "attack_success_rate": {f"{k[0]}|{k[1]}": list(map(float, v))
                                     for k, v in attack_sr.items()},
            "perturb": {f"{k[0]}|{k[1]}|{k[2]}": list(map(float, v))
                         for k, v in perturb.items()},
            "agg": agg_save,
            "isolation": {"ref_test_hash": ref_test_hash,
                           "fractions": args.fractions, "seeds": args.seeds,
                           "config": config,
                           "preprocessing": "per-subsample"},
        }, f, indent=2)

    fracs_sorted = sorted(args.fractions)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs, means, stds = [], [], []
    for f in fracs_sorted:
        if len(ids_acc[f]) == 0:
            continue
        xs.append(f * 100)
        means.append(np.mean(ids_acc[f]) * 100)
        stds.append(np.std(ids_acc[f]) * 100)
        for v in ids_acc[f]:
            ax.scatter([f * 100], [v * 100], alpha=0.25, color='tab:blue', s=18)
    ax.errorbar(xs, means, yerr=stds, marker='o', capsize=4, color='tab:blue', label='mean ± std')
    ax.set_xscale("log")
    ax.set_xlabel("Training fraction (%)  [log scale]")
    ax.set_ylabel("IDS clean test accuracy (%)")
    ax.set_title(f"IDS accuracy vs training fraction  (random subsample, {len(args.seeds)} seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ids_acc_vs_frac.png", dpi=150)
    plt.close(fig)

    n_atk = len(args.attacks)
    fig, axes = plt.subplots(1, n_atk, figsize=(4.6 * n_atk, 4.2), sharey=True)
    if n_atk == 1:
        axes = [axes]
    for ax_i, a in zip(axes, args.attacks):
        for m in methods:
            xs, ys, yerr = [], [], []
            for f in fracs_sorted:
                vals = agg[(f, a, m, "f1")]
                if len(vals) == 0:
                    continue
                xs.append(f * 100)
                ys.append(np.mean(vals) * 100)
                yerr.append(np.std(vals) * 100)
            if xs:
                ax_i.errorbar(xs, ys, yerr=yerr, marker='o', capsize=3, label=m.upper())
        ax_i.set_xscale("log")
        ax_i.set_xlabel("Training fraction (%)  [log]")
        ax_i.set_title(a.upper())
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(fontsize=8)
    axes[0].set_ylabel("F1 @ FPR=5% (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "f1_vs_frac.png", dpi=150)
    plt.close(fig)

    print(f"\n[saved] {out_dir}/dataset_lost_raw.json")
    print(f"[saved] {out_dir}/dataset_lost_agg.json")
    print(f"[saved] {out_dir}/dataset_lost_report.txt")
    print(f"[saved] {out_dir}/ids_acc_vs_frac.png")
    print(f"[saved] {out_dir}/f1_vs_frac.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="out/dataset_lost")
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[1.00, 0.50, 0.25, 0.10, 0.05, 0.01])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ids-epochs", type=int, default=30)
    parser.add_argument("--manifold-cap", type=int, default=200_000)
    parser.add_argument("--manifold-k", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--noise-n", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "bim", "cw"])
    args = parser.parse_args()
    main(args)
