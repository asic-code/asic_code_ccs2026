"""NSL-KDD data loader for MANDA."""
import numpy as np
import pandas as pd
from pathlib import Path


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm", "mailbomb",
}


def load_nsl_kdd(data_dir: str = "data"):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "KDDTrain+.txt", header=None, names=NSL_KDD_COLUMNS)
    test = pd.read_csv(data_dir / "KDDTest+.txt", header=None, names=NSL_KDD_COLUMNS)
    return train, test


def _binary_label(label: str) -> int:
    return 1 if label in DOS_ATTACKS else 0


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    """Filter to {DoS, normal}; one-hot categoricals over train+test union;
    min-max scale numerics by train. Returns dict with X/y/feature indices."""
    train = train.drop(columns=["difficulty"])
    test = test.drop(columns=["difficulty"])

    def filter_dos_normal(df):
        mask = df["label"].apply(lambda x: x == "normal" or x in DOS_ATTACKS)
        return df[mask].reset_index(drop=True)

    train = filter_dos_normal(train)
    test = filter_dos_normal(test)

    y_train = train["label"].apply(_binary_label).values.astype(np.int64)
    y_test = test["label"].apply(_binary_label).values.astype(np.int64)

    train_X = train.drop(columns=["label"])
    test_X = test.drop(columns=["label"])

    categories = {}
    for col in CATEGORICAL:
        vals = sorted(set(train_X[col].unique()) | set(test_X[col].unique()))
        categories[col] = vals

    numeric_cols = [c for c in train_X.columns if c not in CATEGORICAL]

    numeric_train = train_X[numeric_cols].astype(np.float32).values
    numeric_test = test_X[numeric_cols].astype(np.float32).values

    mins = numeric_train.min(axis=0)
    maxs = numeric_train.max(axis=0)
    ranges = maxs - mins
    ranges_safe = np.where(ranges == 0, 1.0, ranges)
    numeric_train_n = (numeric_train - mins) / ranges_safe
    numeric_test_n = (numeric_test - mins) / ranges_safe
    numeric_train_n = np.clip(numeric_train_n, 0.0, 1.0)
    numeric_test_n = np.clip(numeric_test_n, 0.0, 1.0)

    cat_train_parts = []
    cat_test_parts = []
    cat_col_ranges = []
    for col in CATEGORICAL:
        vals = categories[col]
        idx_map = {v: i for i, v in enumerate(vals)}
        tr_idx = train_X[col].map(idx_map).values
        te_idx = test_X[col].map(idx_map).values
        tr_oh = np.eye(len(vals), dtype=np.float32)[tr_idx]
        te_oh = np.eye(len(vals), dtype=np.float32)[te_idx]
        cat_train_parts.append(tr_oh)
        cat_test_parts.append(te_oh)
        cat_col_ranges.append(len(vals))

    X_train = np.concatenate([numeric_train_n] + cat_train_parts, axis=1).astype(np.float32)
    X_test = np.concatenate([numeric_test_n] + cat_test_parts, axis=1).astype(np.float32)

    n_numeric = numeric_train_n.shape[1]
    diff_idx = np.arange(n_numeric, dtype=np.int64)
    non_diff_idx = np.arange(n_numeric, X_train.shape[1], dtype=np.int64)

    feature_ranges = np.ones(X_train.shape[1], dtype=np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "diff_idx": diff_idx,
        "non_diff_idx": non_diff_idx,
        "feature_ranges": feature_ranges,
        "n_features": X_train.shape[1],
        "n_numeric": n_numeric,
        "cat_col_ranges": cat_col_ranges,
    }


if __name__ == "__main__":
    train, test = load_nsl_kdd()
    data = preprocess(train, test)
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"y_train positive rate (DoS): {data['y_train'].mean():.4f}")
    print(f"y_test positive rate (DoS): {data['y_test'].mean():.4f}")
    print(f"n_numeric (differentiable): {data['n_numeric']}")
    print(f"n_categorical dims (non-diff): {len(data['non_diff_idx'])}")
    print(f"Total features: {data['n_features']}")
