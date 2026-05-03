"""Isolation audit for the dataset-lost ablation."""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from data_loader import (
    TRAIN_ROWS_IDS17,
    find_clean_csv,
    _hash_array,
    _hash_int_set,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[1.0, 0.5, 0.25, 0.10, 0.05, 0.01])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--csv", type=str, default=None)
    args_cli = ap.parse_args()

    csv_path = args_cli.csv or find_clean_csv()
    print(f"reading {csv_path}")
    df = pd.read_csv(csv_path, usecols=["Label"])
    n_total = len(df)
    n_train_full = TRAIN_ROWS_IDS17
    n_test = n_total - n_train_full
    y_train_full = df.iloc[:n_train_full]["Label"].to_numpy()
    y_test_full = df.iloc[n_train_full:]["Label"].to_numpy()
    print(f"n_total={n_total}, n_train_full={n_train_full}, n_test={n_test}")
    print(f"train_pos_rate_full = {np.mean(y_train_full==1):.4f}")
    print(f"test_pos_rate       = {np.mean(y_test_full==1):.4f}")
    test_lbl_hash = _hash_array(y_test_full.astype(np.int64))
    print(f"test_label_hash     = {test_lbl_hash}")

    print("\n{:>6} {:>4} {:>10} {:>10} {:>8} {:>10} {:>20}".format(
        "frac", "seed", "n_sub", "expected", "n_pos", "pos_rate", "subsample_hash"
    ))
    seen_hashes: dict = {}
    for frac in args_cli.fractions:
        expected = max(1, int(round(n_train_full * frac)))
        for seed in args_cli.seeds:
            rng = np.random.default_rng(seed)
            if frac < 1.0:
                idx = rng.choice(n_train_full, size=expected, replace=False)
            else:
                idx = np.arange(n_train_full)
            n_sub = len(idx)
            n_pos = int(np.sum(y_train_full[idx] == 1))
            pos_rate = n_pos / n_sub
            h = _hash_int_set(idx)
            print("{:>5.2f}% {:>4} {:>10} {:>10} {:>8} {:>10.4f} {:>20}".format(
                frac * 100, seed, n_sub, expected, n_pos, pos_rate, h
            ))
            seen_hashes.setdefault((frac,), set()).add(h)

    print("\n=== isolation checks ===")
    for (frac,), hs in seen_hashes.items():
        if frac == 1.0:
            ok = (len(hs) == 1)
            note = ("100% across seeds yields the same rows (only model "
                    "init noise differs)") if ok else (
                "WARN: 100% gave different hashes across seeds")
            print(f"  fraction=100% : {note}")
        else:
            ok = (len(hs) == len(args_cli.seeds))
            print(f"  fraction={frac*100:.1f}% : {len(hs)} unique subsample "
                  f"hash(es) across {len(args_cli.seeds)} seeds "
                  f"({'distinct' if ok else 'WARN: collisions'})")

    print(f"\n  test_label_hash constant across all trials: {test_lbl_hash}")
    print("\nNote: CICIDS2017 train/test come from disjoint contiguous row "
          "ranges in a time-sorted CSV; row overlap is 0 by construction.")


if __name__ == "__main__":
    main()
