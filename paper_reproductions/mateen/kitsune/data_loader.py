"""Kitsune data loader for the Mateen reproduction.

The dataset is split across two CSVs (mirroring the official
`MateenUtils/data_processing.prepare_data("Kitsune")`):

  - data/Kitsune/TrainData.csv  (≈ 1.5 GB, 751,280 rows)
  - data/Kitsune/TestData.csv   (≈ 10.9 GB, 5,324,759 rows)

Each row has 115 statistical features + a `Label` column (0=benign,
1=attack).

For the dataset-lost ablation we expose a fraction/seed knob on the
TRAIN file only; test is untouched. MinMaxScaler refits per
subsample.

Memory: TestData.csv is large; we load with `low_memory=False` to
avoid type mixing, but the resulting DataFrame is ≈ 5 GB in RAM. Mac
Studio has plenty.
"""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_paths(name: str) -> list[str]:
    h = _here()
    return [
        os.path.join(h, "data", "Kitsune", name),
        os.path.join(h, "Datasets", "Kitsune", name),
        os.path.join(h, "data", name),
    ]


def find_kitsune_csv(name: str) -> str:
    for p in _candidate_paths(name):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found. Looked in:\n  "
                            + "\n  ".join(_candidate_paths(name)))


# In-process cache for the giant TestData.csv (10.9 GB on disk, ~5 GB
# in pandas RAM). Phase 4 calls load_kitsune 18 times; without this we
# spend ~90 min just on test-CSV I/O.
_TRAIN_CACHE: dict = {}
_TEST_CACHE: dict = {}


def _load_cached(path: str, cache: dict) -> "pd.DataFrame":
    if path in cache:
        return cache[path]
    df = pd.read_csv(path, low_memory=False)
    cache[path] = df
    return df


def _hash_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _hash_int_set(idx: np.ndarray) -> str:
    s = np.sort(idx.astype(np.int64))
    return _hash_array(s)


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_dim: int
    train_size: int
    test_size: int
    train_pos_rate: float
    test_pos_rate: float
    test_label_hash: str
    train_index_hash: str
    fraction: float
    seed: int


def load_kitsune(
    fraction: float = 1.0,
    seed: int = 0,
    train_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    test_variant: str = "TestData.csv",
    cache_full: bool = True,
) -> Split:
    """Load Kitsune. `test_variant` selects between Kitsune (default),
    mKitsune (`NewTestData.csv`), or rKitsune (`Recurring.csv`)."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    train_csv = train_csv or find_kitsune_csv("TrainData.csv")
    test_csv = test_csv or find_kitsune_csv(test_variant)

    train_full = _load_cached(train_csv, _TRAIN_CACHE)
    test_full = _load_cached(test_csv, _TEST_CACHE)

    # Uniform unstratified subsample on train rows.
    rng = np.random.default_rng(seed)
    if fraction < 1.0:
        n_keep = max(1, int(round(len(train_full) * fraction)))
        keep_idx = rng.choice(len(train_full), size=n_keep, replace=False)
        train = train_full.iloc[keep_idx].reset_index(drop=True)
        train_index_hash = _hash_int_set(keep_idx)
    else:
        train = train_full
        train_index_hash = _hash_int_set(np.arange(len(train_full)))

    test = test_full

    y_train = train["Label"].to_numpy().astype(np.int64)
    y_test = test["Label"].to_numpy().astype(np.int64)
    x_train_raw = np.nan_to_num(
        train.drop(columns=["Label"]).to_numpy(dtype=np.float64)
    )
    x_test_raw = np.nan_to_num(
        test.drop(columns=["Label"]).to_numpy(dtype=np.float64)
    )

    scaler = MinMaxScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    return Split(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_dim=x_train.shape[1],
        train_size=len(train),
        test_size=len(test),
        train_pos_rate=float(np.mean(y_train == 1)),
        test_pos_rate=float(np.mean(y_test == 1)),
        test_label_hash=_hash_array(y_test),
        train_index_hash=train_index_hash,
        fraction=fraction,
        seed=seed,
    )


def partition_test(
    x_test: np.ndarray,
    y_test: np.ndarray,
    window: int = 50_000,
) -> Tuple[list, list]:
    n = len(x_test)
    n_slices = n // window + 1
    xs, ys = [], []
    for i in range(n_slices):
        s = i * window
        e = min((i + 1) * window, n)
        if s >= e:
            continue
        xs.append(x_test[s:e])
        ys.append(y_test[s:e])
    return xs, ys


def sanity_print(split: Split) -> None:
    print(f"=== Kitsune split (fraction={split.fraction}, seed={split.seed}) ===")
    print(f"  feature_dim   : {split.feature_dim}")
    print(f"  train_size    : {split.train_size:,}")
    print(f"  test_size     : {split.test_size:,}")
    print(f"  train_pos_rate: {split.train_pos_rate:.4f}")
    print(f"  test_pos_rate : {split.test_pos_rate:.4f}")
    print(f"  test_lbl_hash : {split.test_label_hash}")
    print(f"  train_idx_hash: {split.train_index_hash}")
