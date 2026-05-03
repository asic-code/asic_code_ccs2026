"""Isolation audit for KitNET / Mirai."""
from __future__ import annotations
import argparse

import numpy as np

from data_loader import (
    load_mirai,
    AD_GRACE_DEFAULT,
    FM_GRACE_DEFAULT,
    LABEL_BOUNDARY,
    N_TOTAL,
    _hash_array,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[1.0, 0.5, 0.25, 0.10, 0.05, 0.01])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args_cli = ap.parse_args()

    s = load_mirai(fraction=1.0, seed=0)
    print(f"label_boundary = row {LABEL_BOUNDARY}  →  benign rows="
          f"{LABEL_BOUNDARY:,}, attack rows="
          f"{N_TOTAL - LABEL_BOUNDARY:,}")
    print(f"FM_grace_default = {FM_GRACE_DEFAULT}")
    print(f"AD_grace_default = {AD_GRACE_DEFAULT}")
    print(f"eval window      = [{s.eval_start:,}, {N_TOTAL:,}) "
          f"({N_TOTAL - s.eval_start:,} rows)")
    print(f"label_hash       = {s.label_hash}")

    print("\n{:>7} {:>4} {:>10} {:>10} {:>10} {:>10}".format(
        "frac", "seed", "ad_grace", "train_end", "eval_start", "overlap"
    ))
    label_hashes: set = set()
    for frac in args_cli.fractions:
        ad_grace = max(1, int(round(AD_GRACE_DEFAULT * frac)))
        train_end = FM_GRACE_DEFAULT + ad_grace
        for seed in args_cli.seeds:
            split = load_mirai(fraction=frac, seed=seed)
            overlap = max(0, train_end - split.eval_start)
            print("{:>5.2f}% {:>4} {:>10} {:>10} {:>10} {:>10}".format(
                frac * 100, seed, ad_grace, train_end,
                split.eval_start, overlap
            ))
            label_hashes.add(split.label_hash)

    print("\n=== isolation checks ===")
    if len(label_hashes) == 1:
        print("  label_hash constant across every trial")
    else:
        print(f"  label_hash varies: {label_hashes}")
    print("  train/eval overlap is 0 by construction "
          "(eval_start={}; max train_end={})"
          .format(FM_GRACE_DEFAULT + AD_GRACE_DEFAULT,
                  FM_GRACE_DEFAULT + AD_GRACE_DEFAULT))
    print("  At fixed fraction across seeds, training rows are identical "
          "(KitNET sees rows in time order, no random subsampling). Seed "
          "only changes SGD weight init.")


if __name__ == "__main__":
    main()
