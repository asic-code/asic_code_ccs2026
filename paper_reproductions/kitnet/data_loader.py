"""KitNET / Mirai data loader."""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


LABEL_BOUNDARY = 71_662
N_TOTAL = 100_000

FM_GRACE_DEFAULT = 5_000
AD_GRACE_DEFAULT = 50_000


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def find_mirai_csv(name: str = "mirai3.csv") -> str:
    candidates = [
        os.path.join(_here(), "data", name),
        os.path.join(_here(), name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found. Looked in:\n  "
                             + "\n  ".join(candidates))


def _hash_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    fm_grace: int
    ad_grace: int
    eval_start: int
    feature_dim: int
    n_total: int
    label_hash: str
    seed: int
    fraction: float


def load_mirai(
    fraction: float = 1.0,
    seed: int = 0,
    csv_path: Optional[str] = None,
    ts_path: Optional[str] = None,
    eval_start: Optional[int] = None,
) -> Split:
    """Load Mirai. `fraction` scales ADgrace; eval window stays fixed
    at row 55_000 onward regardless of fraction."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    csv_path = csv_path or find_mirai_csv("mirai3.csv")
    ts_path = ts_path or find_mirai_csv("mirai3_ts.csv")

    X = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float64)
    timestamps = pd.read_csv(ts_path, header=None).to_numpy(
        dtype=np.float64
    ).flatten()

    if X.shape[0] != N_TOTAL:
        raise ValueError(
            f"expected {N_TOTAL} rows in mirai3.csv, got {X.shape[0]}")

    y = np.zeros(N_TOTAL, dtype=np.int64)
    y[LABEL_BOUNDARY:] = 1

    ad_grace = max(1, int(round(AD_GRACE_DEFAULT * fraction)))
    eval_start = (
        eval_start
        if eval_start is not None
        else FM_GRACE_DEFAULT + AD_GRACE_DEFAULT
    )

    return Split(
        X=X, y=y, timestamps=timestamps,
        fm_grace=FM_GRACE_DEFAULT,
        ad_grace=ad_grace,
        eval_start=eval_start,
        feature_dim=X.shape[1],
        n_total=N_TOTAL,
        label_hash=_hash_array(y),
        seed=seed,
        fraction=fraction,
    )


def sanity_print(s: Split) -> None:
    print(f"=== Mirai split (fraction={s.fraction}, seed={s.seed}) ===")
    print(f"  feature_dim    : {s.feature_dim}")
    print(f"  n_total        : {s.n_total:,}")
    print(f"  fm_grace       : {s.fm_grace:,}")
    print(f"  ad_grace       : {s.ad_grace:,}")
    print(f"  eval_start     : {s.eval_start:,}")
    print(f"  eval_size      : {s.n_total - s.eval_start:,}")
    n_eval_benign = int(np.sum(s.y[s.eval_start:] == 0))
    n_eval_attack = int(np.sum(s.y[s.eval_start:] == 1))
    print(f"  eval_benign    : {n_eval_benign:,}")
    print(f"  eval_attack    : {n_eval_attack:,}")
    print(f"  label_boundary : row {LABEL_BOUNDARY}")
    print(f"  label_hash     : {s.label_hash}")
    print(f"  ts range       : {s.timestamps[0]:.2f} → "
          f"{s.timestamps[-1]:.2f} minutes")
