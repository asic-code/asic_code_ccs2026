"""Dataset-lost ablation for Mateen / Kitsune."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import time
from typing import Iterable

import numpy as np

# Bump QoS so the macOS scheduler doesn't deprioritize MPS work when run
# as a backgrounded subprocess.
try:
    ctypes.CDLL(None).pthread_set_qos_class_self_np(0x21, 0)
except Exception:
    pass

import device as device_mod
from data_loader import load_kitsune
from mateen_runner import MateenArgs, run_no_update, run_mateen


DEFAULT_FRACTIONS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.01)


def derived_seed(base_seed: int, fraction: float) -> int:
    return int((base_seed * 9973) ^ int(fraction * 1_000_000)) & 0x7fffffff


def trial_record(mode: str, frac: float, seed: int, metrics: dict, split,
                 wall: float) -> dict:
    m = {k: v for k, v in metrics.items() if k not in ("mode",)}
    return dict(
        mode=mode,
        fraction=frac,
        seed=seed,
        train_size=split.train_size,
        train_pos_rate=split.train_pos_rate,
        train_index_hash=split.train_index_hash,
        test_label_hash=split.test_label_hash,
        wall_seconds=wall,
        **m,
    )


def aggregate(rows: list[dict]) -> dict:
    keys = (
        "accuracy", "f1_paper", "macro_f1", "auc_roc",
        "attack_recall", "attack_precision", "attack_f1", "fpr",
    )
    agg: dict = {}
    by_cell: dict = {}
    for r in rows:
        cell = (r["mode"], r["fraction"])
        by_cell.setdefault(cell, []).append(r)
    for (mode, frac), group in by_cell.items():
        cell_agg = dict(mode=mode, fraction=frac, n_seeds=len(group))
        for k in keys:
            vals = [g.get(k) for g in group if g.get(k) is not None
                    and not (isinstance(g.get(k), float) and np.isnan(g.get(k)))]
            if vals:
                cell_agg[f"{k}_mean"] = float(np.mean(vals))
                cell_agg[f"{k}_std"] = float(np.std(vals))
                cell_agg[f"{k}_per_seed"] = [float(v) for v in vals]
        agg[f"{mode}@{frac}"] = cell_agg
    return agg


def fmt_table(agg: dict, mode: str) -> str:
    lines = [
        f"\n=== {mode} — attack-side metrics (mean ± std) ===",
        f"{'frac':>6} {'n':>3} | {'attack_F1':>14} {'recall(TPR)':>14} "
        f"{'fpr':>14} {'accuracy':>14} {'auc_roc':>14}",
        "-" * 92,
    ]
    rows = sorted(
        [(k, v) for k, v in agg.items() if v["mode"] == mode],
        key=lambda kv: -kv[1]["fraction"],
    )
    for _, c in rows:
        f1m, f1s = c.get("attack_f1_mean"), c.get("attack_f1_std")
        rm, rs = c.get("attack_recall_mean"), c.get("attack_recall_std")
        pm, ps = c.get("fpr_mean"), c.get("fpr_std")
        am, as_ = c.get("accuracy_mean"), c.get("accuracy_std")
        um, us = c.get("auc_roc_mean"), c.get("auc_roc_std")
        def fmt(m, s):
            if m is None: return f"{'-':>14}"
            return f"{m*100:6.2f}±{(s or 0)*100:5.2f}".rjust(14)
        lines.append(f"{c['fraction']*100:5.1f}% {c['n_seeds']:>3} | "
                     f"{fmt(f1m,f1s)} {fmt(rm,rs)} {fmt(pm,ps)} "
                     f"{fmt(am,as_)} {fmt(um,us)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", choices=["both", "no_update", "mateen"], default="both")
    ap.add_argument("--init-epochs", type=int, default=100)
    ap.add_argument("--train-csv", type=str, default=None)
    ap.add_argument("--test-variant", type=str, default="TestData.csv",
                    help="TestData.csv | NewTestData.csv (mKitsune) | Recurring.csv (rKitsune)")
    ap.add_argument("--out-dir", type=str, default="out/dataset_lost")
    args_cli = ap.parse_args()

    print(f"DEVICE = {device_mod.DEVICE_STR}")
    print(f"fractions = {args_cli.fractions}")
    print(f"seeds = {args_cli.seeds}")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    raw_path = os.path.join(args_cli.out_dir, "raw.jsonl")
    agg_path = os.path.join(args_cli.out_dir, "agg.json")

    rows: list[dict] = []
    args = MateenArgs()

    done: set = set()
    if os.path.exists(raw_path):
        with open(raw_path) as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                    rows.append(r)
                    done.add((r["mode"], round(float(r["fraction"]), 6),
                              int(r["seed"])))
                except Exception:
                    pass
        print(f"resumed: {len(done)} trials already complete")

    with open(raw_path, "a") as f:
        for frac in args_cli.fractions:
            for seed in args_cli.seeds:
                split = load_kitsune(
                    fraction=frac, seed=seed,
                    train_csv=args_cli.train_csv,
                    test_variant=getattr(args_cli, "test_variant",
                                          "TestData.csv"),
                )
                print(f"\n--- fraction={frac:.4f}, seed={seed}, "
                      f"train_size={split.train_size}, "
                      f"train_pos_rate={split.train_pos_rate:.4f} ---")
                modes: Iterable[str]
                if args_cli.modes == "both":
                    modes = ("no_update", "mateen")
                else:
                    modes = (args_cli.modes,)
                trial_seed = derived_seed(seed, frac)
                for mode in modes:
                    if (mode, round(float(frac), 6), int(seed)) in done:
                        print(f"  [{mode}] SKIP (already in raw.jsonl)")
                        continue
                    fn = run_no_update if mode == "no_update" else run_mateen
                    t0 = time.time()
                    m = fn(split, args, init_epochs=args_cli.init_epochs,
                           seed=trial_seed)
                    wall = time.time() - t0
                    rec = trial_record(mode, frac, seed, m, split, wall)
                    rows.append(rec)
                    f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
                    print(f"  [{mode}] f1_paper={m['f1_paper']*100:.2f}% "
                          f"attack_f1={m['attack_f1']*100:.2f}% "
                          f"tpr={m['attack_recall']*100:.2f}% "
                          f"fpr={m['fpr']*100:.2f}% "
                          f"acc={m['accuracy']*100:.2f}% "
                          f"pred_dist={m.get('pred_dist')}")

    agg = aggregate(rows)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\nWrote {raw_path} ({len(rows)} trials)")
    print(f"Wrote {agg_path}")

    if any(r["mode"] == "no_update" for r in rows):
        print(fmt_table(agg, "no_update"))
    if any(r["mode"] == "mateen" for r in rows):
        print(fmt_table(agg, "mateen"))


if __name__ == "__main__":
    main()
