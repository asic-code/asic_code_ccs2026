"""Isolation audit for the Dataset-Lost analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import default_splits
from dataset_lost import FRACTIONS, SEEDS, _subsample

OUT = Path(__file__).parent / "out"
_KEY_COLS = ["svc_proto", "src_port", "dst_port", "src_ip", "dst_ip",
             "duration_s", "date", "time"]


def _row_keys(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df[_KEY_COLS], index=False).values.astype(
        np.uint64
    )


def main() -> None:
    train, test = default_splits()
    n_train = len(train.df)

    test_label_hash = hashlib.md5(test.y.tobytes()).hexdigest()[:16]
    test_keys_arr = _row_keys(test.df)
    test_keys = set(test_keys_arr.tolist())
    print(f"train n={n_train:,}  test n={len(test.df):,}")
    print(f"test_label_hash = {test_label_hash}")
    print(f"test distinct row-keys = {len(test_keys):,}")

    audit_rows = []
    for frac in FRACTIONS:
        sub_hashes = []
        for seed in SEEDS:
            sub, idx = _subsample(train, frac, seed=seed)
            k = len(sub.df)
            expected = round(n_train * frac)
            sub_idx_hash = hashlib.md5(idx.tobytes()).hexdigest()[:16]
            sub_hashes.append(sub_idx_hash)
            sub_keys = _row_keys(sub.df)
            overlap = int(np.isin(sub_keys, test_keys_arr).sum())
            class_counts = {c: int((sub.category == c).sum())
                            for c in ("normal", "DoS", "Probe",
                                       "R2L", "U2R")}
            n_classes = sum(1 for v in class_counts.values() if v > 0)
            pos = int((sub.category != "normal").sum())
            status = "ok"
            if n_classes < 2:
                status = "degenerate_monoclass"
            elif pos == 0:
                status = "no_positives"
            elif class_counts["U2R"] == 0 and frac <= 0.01:
                status = "ok_but_no_u2r"
            audit_rows.append({
                "frac": frac, "seed": seed,
                "k": k, "expected_k": expected,
                "k_matches_expected": (abs(k - expected) <= 1),
                "sub_index_hash": sub_idx_hash,
                "test_label_hash": test_label_hash,
                "overlap_with_test": overlap,
                "pos_count": pos,
                "class_counts": class_counts,
                "n_classes_present": n_classes,
                "status": status,
            })
        if frac >= 1.0:
            assert len(set(sub_hashes)) == 1, \
                f"100% seeds yielded different hashes: {sub_hashes}"
        else:
            assert len(set(sub_hashes)) == len(SEEDS), \
                f"seeds collapsed at frac={frac}: {sub_hashes}"

    df = pd.DataFrame(audit_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "isolation_audit.csv", index=False)
    (OUT / "isolation_audit.jsonl").write_text(
        "\n".join(json.dumps(r) for r in audit_rows) + "\n"
    )

    print()
    print("=== isolation audit ===")
    with pd.option_context("display.width", 180, "display.max_colwidth", 80):
        print(df.drop(columns=["class_counts"]).to_string(index=False))
    print()
    print("class_counts per row:")
    for r in audit_rows:
        print(f"  frac={r['frac']:>4.2f} seed={r['seed']}  {r['class_counts']}")

    leaks = df["overlap_with_test"].sum()
    degen = (df["status"] != "ok").sum()
    print()
    print(f"total test-row-key overlap across all trials: {leaks}")
    print(f"non-OK trials: {degen} / {len(df)}")


if __name__ == "__main__":
    main()
