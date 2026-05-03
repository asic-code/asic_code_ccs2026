"""CICIDS2017 data loader for the Mateen reproduction."""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TRAIN_ROWS_IDS17 = 693_702


def _candidate_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, "data", "CICIDS2017", "clean_data.csv"),
        os.path.join(here, "Datasets", "CICIDS2017", "clean_data.csv"),
        os.path.join(here, "data", "clean_data.csv"),
    ]


def find_clean_csv() -> str:
    for p in _candidate_paths():
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "clean_data.csv not found. Looked in:\n  "
        + "\n  ".join(_candidate_paths())
    )


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


def load_ids17(
    fraction: float = 1.0,
    seed: int = 0,
    csv_path: Optional[str] = None,
    cache_full: bool = True,
) -> Split:
    """Load CICIDS2017 with optional uniform unstratified subsampling
    of the training rows. Test set is always the full second half.
    MinMaxScaler fits on the (possibly subsampled) training rows."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    csv_path = csv_path or find_clean_csv()
    df = pd.read_csv(csv_path)

    train_full = df.iloc[:TRAIN_ROWS_IDS17].copy()
    test_full = df.iloc[TRAIN_ROWS_IDS17:].copy()

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
    print(f"=== CICIDS2017 split (fraction={split.fraction}, seed={split.seed}) ===")
    print(f"  feature_dim   : {split.feature_dim}")
    print(f"  train_size    : {split.train_size:,}")
    print(f"  test_size     : {split.test_size:,}")
    print(f"  train_pos_rate: {split.train_pos_rate:.4f}")
    print(f"  test_pos_rate : {split.test_pos_rate:.4f}")
    print(f"  test_lbl_hash : {split.test_label_hash}")
    print(f"  train_idx_hash: {split.train_index_hash}")
