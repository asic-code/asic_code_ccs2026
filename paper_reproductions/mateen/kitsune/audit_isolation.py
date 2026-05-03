"""Isolation audit for the Kitsune dataset-lost ablation."""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from data_loader import (
    find_kitsune_csv,
    _hash_array,
    _hash_int_set,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[1.0, 0.5, 0.25, 0.10, 0.05, 0.01])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-csv", type=str, default=None)
    ap.add_argument("--test-csv", type=str, default=None)
    ap.add_argument("--test-variant", type=str, default="TestData.csv")
    args_cli = ap.parse_args()

    train_csv = args_cli.train_csv or find_kitsune_csv("TrainData.csv")
    test_csv = args_cli.test_csv or find_kitsune_csv(args_cli.test_variant)
    print(f"train: {train_csv}\ntest : {test_csv}")
    y_train_full = pd.read_csv(train_csv, usecols=["Label"])["Label"].to_numpy()
    y_test_full = pd.read_csv(test_csv, usecols=["Label"])["Label"].to_numpy()
    n_train_full = len(y_train_full)
    n_test = len(y_test_full)
    print(f"n_train_full={n_train_full}, n_test={n_test}")
    print(f"train_pos_rate_full = {np.mean(y_train_full==1):.4f}")
    print(f"test_pos_rate       = {np.mean(y_test_full==1):.4f}")
    test_lbl_hash = _hash_array(y_test_full.astype(np.int64))
    print(f"test_label_hash     = {test_lbl_hash}")

    print("\n{:>6} {:>4} {:>10} {:>10} {:>8} {:>10} {:>20}".format(
        "frac", "seed", "n_sub", "expected", "n_pos", "pos_rate",
        "subsample_hash"
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
                "WARN: 100% should give one hash")
            print(f"  fraction=100% : {note}")
        else:
            ok = (len(hs) == len(args_cli.seeds))
            print(f"  fraction={frac*100:.1f}% : {len(hs)} unique subsample "
                  f"hash(es) across {len(args_cli.seeds)} seeds "
                  f"({'distinct' if ok else 'WARN: collisions'})")

    print(f"\n  test_label_hash constant across all trials: {test_lbl_hash}")


if __name__ == "__main__":
    main()
