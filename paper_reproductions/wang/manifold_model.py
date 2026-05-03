"""Manifold model via sklearn LabelSpreading on a stratified subsample."""
import numpy as np
from sklearn.semi_supervised import LabelSpreading


class ManifoldModel:
    def __init__(self, n_fit=5000, kernel="knn", n_neighbors=10, alpha=0.2, max_iter=30, random_state=0):
        self.n_fit = n_fit
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.ls: LabelSpreading | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]
        k = self.n_fit // 2
        sel_pos = rng.choice(idx_pos, size=min(k, len(idx_pos)), replace=False)
        sel_neg = rng.choice(idx_neg, size=min(k, len(idx_neg)), replace=False)
        sel = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(sel)
        X = X_train[sel]
        y = y_train[sel]

        self.ls = LabelSpreading(
            kernel=self.kernel,
            n_neighbors=self.n_neighbors,
            alpha=self.alpha,
            max_iter=self.max_iter,
        )
        self.ls.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.ls is not None
        return self.ls.predict_proba(X)


if __name__ == "__main__":
    from data_loader import load_nsl_kdd, preprocess
    tr, te = load_nsl_kdd()
    d = preprocess(tr, te)
    m = ManifoldModel(n_fit=5000, kernel="knn", n_neighbors=10)
    m.fit(d["X_train"], d["y_train"])
    probs = m.predict_proba(d["X_test"][:1000])
    preds = probs.argmax(1)
    acc = (preds == d["y_test"][:1000]).mean()
    print(f"manifold eval acc on 1000 test samples: {acc:.4f}")
    print(f"first 5 probs:\n{probs[:5]}")
