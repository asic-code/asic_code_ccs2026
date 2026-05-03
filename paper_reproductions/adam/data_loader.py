"""DARPA'98 connection-record loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DARPA = Path(__file__).parent / "darpa"

CATEGORY_CANON = {"dos": "DoS", "probe": "Probe", "r2l": "R2L", "u2r": "U2R"}
CATEGORY_INT = {"normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4}
CATEGORY_NAME = {v: k for k, v in CATEGORY_INT.items()}

NAMED_CATEGORY = {
    "back": "DoS", "land": "DoS", "neptune": "DoS", "neptunettl": "DoS",
    "pod": "DoS", "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "syslogd": "DoS", "arppoison": "DoS", "crashiis": "DoS", "selfping": "DoS",
    "dosnuke": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe", "illegal-sniffer": "Probe",
    "lsdomain": "Probe", "ls_domain": "Probe", "ntinfoscan": "Probe",
    "queso": "Probe",
    "guess": "R2L", "guest": "R2L", "guess_passwd": "R2L", "ftp_write": "R2L",
    "imap": "R2L", "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L", "dict": "R2L", "dictsimple": "R2L",
    "framespoof": "R2L", "ncftp": "R2L", "ppmacro": "R2L", "netcat": "R2L",
    "netbus": "R2L", "sshtrojan": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
    "eject": "U2R", "ffb": "U2R", "ffbconfig": "U2R", "fdformat": "U2R",
    "sechole": "U2R", "casesen": "U2R", "ntfsdos": "U2R", "yaga": "U2R",
    "anypw": "U2R", "nukepw": "U2R", "secret": "U2R",
}


ITEM_COLS = ("proto", "service", "src_port", "dst_port", "src_ip", "dst_ip")


@dataclass
class Split:
    df: pd.DataFrame
    category: np.ndarray
    y: np.ndarray
    items_int: np.ndarray
    vocab: dict
    instance: np.ndarray

    @property
    def n(self) -> int:
        return len(self.df)

    def vocab_inverse(self) -> dict:
        return {v: k for k, v in self.vocab.items()}


def _parse_label(token: str) -> tuple[str, str]:
    if token == "-":
        return "normal", ""
    if "," in token:
        parts = token.split(",")
        name = parts[0].lower().strip()
        inst_id = parts[1].strip() if len(parts) >= 2 else ""
        if len(parts) >= 5:
            cat = parts[4].lower().strip()
            if cat in CATEGORY_CANON:
                return CATEGORY_CANON[cat], f"{name}:{inst_id}"
        return NAMED_CATEGORY.get(name, "R2L"), f"{name}:{inst_id}"
    name = token.lower().strip()
    return NAMED_CATEGORY.get(name, "R2L"), f"{name}:"


_WELL_KNOWN_PORTS = {21, 22, 23, 25, 53, 80, 110, 111, 113, 143,
                      161, 443, 514, 515}


def _port_bucket_vec(s: pd.Series) -> np.ndarray:
    n = pd.to_numeric(s, errors="coerce")
    out = np.full(len(s), "other", dtype=object)
    out[s.values == "-"] = "none"
    is_wk = (n < 1024)
    out[is_wk.values] = "wk_other"
    for p in _WELL_KNOWN_PORTS:
        out[(n == p).values] = f"wk_{p}"
    out[((n >= 1024) & (n < 5000)).values] = "reg_lo"
    out[((n >= 5000) & (n < 32768)).values] = "reg_hi"
    out[(n >= 32768).values] = "ephem"
    return out


def _ip_bucket_vec(s: pd.Series) -> np.ndarray:
    parts = s.str.split(".", expand=True)
    a = pd.to_numeric(parts.get(0), errors="coerce")
    b = pd.to_numeric(parts.get(1), errors="coerce")
    out = np.full(len(s), "other", dtype=object)
    a_arr = a.values
    b_arr = b.values
    in172 = (a_arr == 172) & (b_arr >= 16) & (b_arr <= 31)
    in192 = (a_arr == 192) & (b_arr == 168)
    loop = a_arr == 127
    ext = ~(in172 | in192 | loop) & ~np.isnan(a_arr)
    if in172.any():
        b_int = b_arr[in172].astype(int).astype(str)
        out[in172] = np.array([f"in_172.{x}" for x in b_int], dtype=object)
    out[in192] = "in_192.168"
    out[loop] = "loopback"
    if ext.any():
        a_int = a_arr[ext].astype(int).astype(str)
        out[ext] = np.array([f"ext_{x}" for x in a_int], dtype=object)
    return out


def _service_split_vec(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    parts = s.str.split("/", n=1, expand=True)
    svc = parts[0].values
    proto = parts[1].fillna("t").values if 1 in parts.columns else \
            np.full(len(s), "t", dtype=object)
    return svc, proto


def _parse_file(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue
            rows.append({
                "idx": parts[0], "date": parts[1], "time": parts[2],
                "duration": parts[3], "svc_proto": parts[4],
                "src_port": parts[5], "dst_port": parts[6],
                "src_ip": parts[7], "dst_ip": parts[8],
                "score": parts[9], "attack_label": parts[10].strip(),
            })
    return pd.DataFrame(rows)


def _intern_columns(cols: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    n = len(cols[0])
    vocab: dict = {}
    out = np.empty((n, len(cols)), dtype=np.int32)
    for j, (col_name, col_vals) in enumerate(zip(ITEM_COLS, cols)):
        codes, uniques = pd.factorize(col_vals, sort=False)
        col_ids = np.empty(len(uniques), dtype=np.int32)
        for k, val in enumerate(uniques):
            key = (col_name, str(val))
            i = vocab.get(key)
            if i is None:
                i = len(vocab)
                vocab[key] = i
            col_ids[k] = i
        out[:, j] = col_ids[codes]
    return out, vocab


def load_listfiles(paths: list[Path]) -> Split:
    df = pd.concat([_parse_file(p) for p in paths], ignore_index=True)
    df["duration_s"] = df["duration"].map(_dur_to_secs).astype(np.float32)

    svc, proto = _service_split_vec(df["svc_proto"])
    df["service"] = svc
    df["proto"] = proto
    src_pb = _port_bucket_vec(df["src_port"])
    dst_pb = _port_bucket_vec(df["dst_port"])
    src_ib = _ip_bucket_vec(df["src_ip"])
    dst_ib = _ip_bucket_vec(df["dst_ip"])

    items_int, vocab = _intern_columns([proto, svc, src_pb, dst_pb,
                                         src_ib, dst_ib])

    parsed = [_parse_label(lbl) for lbl in df["attack_label"].values]
    cat = np.array([p[0] for p in parsed])
    instance = np.array([p[1] for p in parsed])
    y = np.array([CATEGORY_INT[c] for c in cat], dtype=np.int64)
    return Split(df=df, category=cat, y=y,
                 items_int=items_int, vocab=vocab, instance=instance)


def _dur_to_secs(s: str) -> float:
    if s == "-" or ":" not in s:
        return 0.0
    try:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)
    except ValueError:
        return 0.0


CACHE_DIR = Path(__file__).parent / "out"


def default_splits(*, use_cache: bool = True) -> tuple[Split, Split]:
    """Train weeks 3-7; test from truth weeks 1-2."""
    import pickle
    train_pkl = CACHE_DIR / "darpa_train.pkl"
    test_pkl = CACHE_DIR / "darpa_test.pkl"
    if use_cache and train_pkl.exists() and test_pkl.exists():
        with open(train_pkl, "rb") as f:
            train = pickle.load(f)
        with open(test_pkl, "rb") as f:
            test = pickle.load(f)
        return train, test

    train_paths = sorted((DARPA).glob("train_w*_*/tcpdump.list"))
    test_paths = sorted((DARPA / "truth").glob("*week/*/tcpdump.lllist"))
    train = load_listfiles(train_paths)
    test = load_listfiles(test_paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(train_pkl, "wb") as f:
        pickle.dump(train, f)
    with open(test_pkl, "wb") as f:
        pickle.dump(test, f)
    return train, test


if __name__ == "__main__":
    train, test = default_splits()
    print(f"train: n={len(train.df):,}  "
          f"categories={pd.Series(train.category).value_counts().to_dict()}")
    print(f"test:  n={len(test.df):,}  "
          f"categories={pd.Series(test.category).value_counts().to_dict()}")
    inv = train.vocab_inverse()
    print(f"vocab size: train={len(train.vocab)} test={len(test.vocab)}")
    print(f"first row item ids: {train.items_int[0].tolist()}")
    print(f"first row items:    {[inv[i] for i in train.items_int[0]]}")
    for cat in ("DoS", "Probe", "R2L", "U2R"):
        mask = test.category == cat
        n = int(mask.sum())
        print(f"  test {cat}: {n:,} connections")
