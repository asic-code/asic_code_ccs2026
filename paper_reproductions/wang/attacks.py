"""Adversarial-example generation for NSL-KDD IDS.

Implements FGSM, BIM, and CW (L2). Constraints: L-inf per-feature
budget p*R_i on differentiable (numeric) features, freeze categoricals,
clip to [0, 1].
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _apply_constraints(x_adv, x, diff_idx, non_diff_idx, feature_ranges, p):
    x_adv = x_adv.clone()
    x_adv[:, non_diff_idx] = x[:, non_diff_idx]
    delta = x_adv[:, diff_idx] - x[:, diff_idx]
    per_feat_bound = p * feature_ranges[diff_idx]
    delta = torch.clamp(delta, -per_feat_bound, per_feat_bound)
    x_adv[:, diff_idx] = x[:, diff_idx] + delta
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def fgsm(model, x, y, diff_idx, non_diff_idx, feature_ranges, p=0.05, device="cpu"):
    model.eval()
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    x.requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    step = p * feature_ranges
    x_adv = x.detach() + step * grad.sign()
    x_adv = _apply_constraints(x_adv, x.detach(), diff_idx, non_diff_idx, feature_ranges, p)
    return x_adv.detach()


def bim(model, x, y, diff_idx, non_diff_idx, feature_ranges, p=0.05, n_steps=20,
        alpha_frac=0.1, early_stop=True, device="cpu"):
    """BIM with optional early-stop on prediction flip (minimum-perturbation AEs)."""
    model.eval()
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    x_adv = x.clone()
    alpha = alpha_frac * p * feature_ranges
    flipped = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="none")
        grad = torch.autograd.grad(loss.sum(), x_adv)[0]
        step = alpha * grad.sign()
        if early_stop:
            step = step.clone()
            step[flipped] = 0.0
        x_adv = x_adv.detach() + step
        x_adv = _apply_constraints(x_adv, x, diff_idx, non_diff_idx, feature_ranges, p)
        if early_stop:
            with torch.no_grad():
                preds = model(x_adv).argmax(1)
                flipped = flipped | (preds != y)
            if flipped.all():
                break
    return x_adv.detach()


def cw_l2(model, x, y, diff_idx, non_diff_idx, feature_ranges, p=0.05,
          n_steps=200, lr=0.01, c=1.0, kappa=0.0, device="cpu"):
    """Carlini-Wagner L2 with per-feature budget via tanh parameterization."""
    model.eval()
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    n_samples = x.shape[0]
    num_classes = int(model(x[:1]).shape[-1])
    if num_classes == 2:
        target = 1 - y
    else:
        target = (y + 1) % num_classes

    budget = (p * feature_ranges[diff_idx]).to(device)
    w = torch.zeros(n_samples, len(diff_idx), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for _ in range(n_steps):
        delta_diff = budget * torch.tanh(w)
        x_adv = x.clone()
        x_adv[:, diff_idx] = x_adv[:, diff_idx] + delta_diff
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        logits = model(x_adv)
        one_hot_y = F.one_hot(y, num_classes).float()
        one_hot_t = F.one_hot(target, num_classes).float()
        real = (logits * one_hot_y).sum(1)
        other = (logits * one_hot_t).sum(1)
        f = torch.clamp(real - other, min=-kappa)
        l2 = (delta_diff ** 2).sum(1)
        loss = (l2 + c * f).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        delta_diff = budget * torch.tanh(w)
        x_adv = x.clone()
        x_adv[:, diff_idx] = x_adv[:, diff_idx] + delta_diff
        x_adv = _apply_constraints(x_adv, x, diff_idx, non_diff_idx, feature_ranges, p)
    return x_adv.detach()


def generate_aes(model, x, y, attack, diff_idx, non_diff_idx, feature_ranges, p=0.05,
                 device="cpu", batch_size=1024, **kwargs):
    """Batched AE generation. Returns (x_adv, success_mask)."""
    model.eval()
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    x = x.to(device)
    y = y.to(device)
    diff_idx_t = torch.from_numpy(diff_idx).to(device) if isinstance(diff_idx, np.ndarray) else diff_idx.to(device)
    non_diff_idx_t = torch.from_numpy(non_diff_idx).to(device) if isinstance(non_diff_idx, np.ndarray) else non_diff_idx.to(device)
    feature_ranges_t = torch.from_numpy(feature_ranges).to(device) if isinstance(feature_ranges, np.ndarray) else feature_ranges.to(device)

    out = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        if attack == "fgsm":
            xa = fgsm(model, xb, yb, diff_idx_t, non_diff_idx_t, feature_ranges_t, p=p, device=device)
        elif attack == "bim":
            xa = bim(model, xb, yb, diff_idx_t, non_diff_idx_t, feature_ranges_t, p=p, device=device, **kwargs)
        elif attack == "cw":
            xa = cw_l2(model, xb, yb, diff_idx_t, non_diff_idx_t, feature_ranges_t, p=p, device=device, **kwargs)
        else:
            raise ValueError(f"unknown attack: {attack}")
        out.append(xa.cpu())
    x_adv = torch.cat(out, dim=0)

    model.eval()
    with torch.no_grad():
        preds = model(x_adv.to(device)).argmax(1).cpu().numpy()
    y_np = y.cpu().numpy()
    success = preds != y_np
    return x_adv.numpy(), success


if __name__ == "__main__":
    import torch
    from data_loader import load_nsl_kdd, preprocess
    from ids_model import IDSModel
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tr, te = load_nsl_kdd()
    d = preprocess(tr, te)
    ckpt = torch.load("ids_model.pt", map_location=device)
    model = IDSModel(ckpt["in_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    Xte = torch.from_numpy(d["X_test"]).to(device)
    yte = torch.from_numpy(d["y_test"]).to(device)
    with torch.no_grad():
        preds = model(Xte).argmax(1)
    correct_mask = (preds == yte).cpu().numpy()
    Xc = d["X_test"][correct_mask]
    yc = d["y_test"][correct_mask]
    print(f"correctly classified test samples: {Xc.shape[0]} / {d['X_test'].shape[0]}")

    for attack in ["fgsm", "bim", "cw"]:
        x_adv, succ = generate_aes(
            model, Xc, yc, attack, d["diff_idx"], d["non_diff_idx"], d["feature_ranges"],
            p=0.05, device=device,
        )
        with torch.no_grad():
            post = model(torch.from_numpy(x_adv).to(device)).argmax(1).cpu().numpy()
        acc = (post == yc).mean()
        print(f"{attack}: success_rate={succ.mean():.4f} | model_acc_on_all_AE={acc:.4f}")
