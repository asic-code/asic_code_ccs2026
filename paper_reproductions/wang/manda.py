"""MANDA detector: Manifold + DB + logistic-regression combiner."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def _model_proba(model, x, device, batch_size=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(device)
            out.append(F.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_score1(manifold_probs: np.ndarray, ids_probs: np.ndarray) -> np.ndarray:
    """score1 = ||p|| + ||q|| - ||p+q||"""
    p = manifold_probs
    q = ids_probs
    np_ = np.linalg.norm(p, axis=1)
    nq_ = np.linalg.norm(q, axis=1)
    npq = np.linalg.norm(p + q, axis=1)
    return np_ + nq_ - npq


def compute_score2(model, x: np.ndarray, sigma: float = 0.05, N: int = 50,
                   device: str = "cpu", batch_size: int = 1024) -> np.ndarray:
    """score2 = mean_i ||F(x_i)|| - ||mean_i F(x_i)||, x_i = x + N(0, sigma^2)."""
    model.eval()
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x[start:end]).to(device)
            b = xb.shape[0]
            noise = torch.randn(N, b, xb.shape[1], device=device) * sigma
            noisy = (xb.unsqueeze(0) + noise).reshape(N * b, -1)
            noisy = torch.clamp(noisy, 0.0, 1.0)
            probs = F.softmax(model(noisy), dim=-1)
            probs = probs.reshape(N, b, -1)
            mean_norm = probs.norm(dim=2).mean(dim=0)
            norm_mean = probs.mean(dim=0).norm(dim=1)
            out[start:end] = (mean_norm - norm_mean).cpu().numpy()
    return out


def compute_scores(model, manifold, x: np.ndarray, sigma: float = 0.05, N: int = 50,
                   device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    ids_probs = _model_proba(model, x, device)
    manifold_probs = manifold.predict_proba(x)
    s1 = compute_score1(manifold_probs, ids_probs)
    s2 = compute_score2(model, x, sigma=sigma, N=N, device=device)
    return s1, s2


def train_manda_lr(score1: np.ndarray, score2: np.ndarray, y_adv: np.ndarray) -> LogisticRegression:
    X = np.stack([score1, score2], axis=1).astype(np.float64)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y_adv)
    return lr


def manda_score(lr: LogisticRegression, score1: np.ndarray, score2: np.ndarray) -> np.ndarray:
    X = np.stack([score1, score2], axis=1).astype(np.float64)
    return lr.predict_proba(X)[:, 1]
