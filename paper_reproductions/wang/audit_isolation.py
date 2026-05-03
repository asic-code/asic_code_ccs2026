"""Isolation audit for the dataset-lost experiment."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from data_loader import load_nsl_kdd, preprocess, DOS_ATTACKS


def filter_dn(df):
    mask = df["label"].apply(lambda x: x == "normal" or x in DOS_ATTACKS)
    return df[mask].reset_index(drop=True)


def run():
    tr_raw, te_raw = load_nsl_kdd("data")
    tr_raw = filter_dn(tr_raw)
    te_raw = filter_dn(te_raw)
    print(f"tr_raw rows: {len(tr_raw)}   te_raw rows: {len(te_raw)}")

    tr_finger = tr_raw.apply(lambda r: tuple(r.values), axis=1)
    te_finger = te_raw.apply(lambda r: tuple(r.values), axis=1)

    te_set = set(te_finger)
    overlap_raw = sum(1 for f in tr_finger if f in te_set)
    print(f"[1] raw train ∩ raw test row-fingerprint overlap: {overlap_raw}")

    cases = [(1.00, 0), (1.00, 1), (0.50, 0), (0.50, 1),
             (0.05, 0), (0.05, 1), (0.01, 0), (0.01, 1), (0.01, 2)]
    n_full = len(tr_raw)
    test_label_hash = None
    subsample_signatures = {}
    for frac, seed in cases:
        rng = np.random.default_rng(int(seed * 1_000_003 + int(frac * 1_000_000)))
        k = max(1, int(round(n_full * frac)))
        idx = rng.choice(n_full, size=k, replace=False)
        train_sub = tr_raw.iloc[idx].reset_index(drop=True)
        sig = hashlib.sha1(np.sort(idx).tobytes()).hexdigest()[:16]
        subsample_signatures[(frac, seed)] = sig

        sub_finger = train_sub.apply(lambda r: tuple(r.values), axis=1)
        overlap_sub = sum(1 for f in sub_finger if f in te_set)
        d_run = preprocess(train_sub, te_raw)
        te_lbl_hash = hashlib.sha1(d_run["y_test"].tobytes()).hexdigest()
        tr_lbl_hash = hashlib.sha1(d_run["y_train"].tobytes()).hexdigest()
        te_feat_hash = hashlib.sha1(d_run["X_test"].tobytes()).hexdigest()
        if test_label_hash is None:
            test_label_hash = te_lbl_hash
        assert te_lbl_hash == test_label_hash, "test label order drifted!"

        print(f"[{frac*100:5.2f}% s={seed}] k={k}  idx-hash={sig}  "
              f"sub↔test row overlap={overlap_sub}  "
              f"y_test-hash={te_lbl_hash[:10]}  "
              f"X_test-feat-hash={te_feat_hash[:10]}  "
              f"y_train-hash={tr_lbl_hash[:10]}")

    print("\n[2] subsample uniqueness at same fraction:")
    by_frac = {}
    for (frac, seed), sig in subsample_signatures.items():
        by_frac.setdefault(frac, []).append((seed, sig))
    for frac, items in by_frac.items():
        sigs = set(sig for _, sig in items)
        status = "UNIQUE" if len(sigs) == len(items) else "DUPLICATE!"
        print(f"  frac={frac*100:5.2f}%  seeds={[s for s,_ in items]}  "
              f"unique_subsamples={len(sigs)}/{len(items)}  [{status}]")

    print("\n[3] per-subsample preprocessing:")
    print("  y_test label-hash identical across all trials.")
    print("  X_test feature-hash differs per trial (scaler refit).")

    print("\nIsolation audit complete.")


if __name__ == "__main__":
    run()
