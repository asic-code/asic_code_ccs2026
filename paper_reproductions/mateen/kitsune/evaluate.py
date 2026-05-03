"""Single end-to-end Mateen run on Kitsune."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import time

import numpy as np

try:
    ctypes.CDLL(None).pthread_set_qos_class_self_np(0x21, 0)
except Exception:
    pass

import device as device_mod
from data_loader import load_kitsune, sanity_print
from mateen_runner import MateenArgs, run_no_update, run_mateen


def fmt_metrics(d: dict) -> str:
    return (
        f"  ── paper-side ─────────\n"
        f"    accuracy    : {d['accuracy']*100:7.4f}%\n"
        f"    f1_paper    : {d['f1_paper']*100:7.4f}%   (benign-as-positive)\n"
        f"    macro_f1    : {d['macro_f1']*100:7.4f}%   (paper 'mF1')\n"
        f"    auc_roc     : {d.get('auc_roc',float('nan'))*100:7.4f}%   (paper window-avg)\n"
        f"  ── attack-side ────────\n"
        f"    attack_recall: {d['attack_recall']*100:7.4f}%   (TPR)\n"
        f"    attack_prec  : {d['attack_precision']*100:7.4f}%\n"
        f"    attack_f1    : {d['attack_f1']*100:7.4f}%\n"
        f"    fpr          : {d['fpr']*100:7.4f}%\n"
        f"  ── confusion ──────────\n"
        f"    tp={d['tp']:>8} fn={d['fn']:>8} fp={d['fp']:>8} tn={d['tn']:>8}\n"
        f"    pred_dist={d.get('pred_dist','-')}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-epochs", type=int, default=100)
    ap.add_argument("--mode", choices=["both", "no_update", "mateen"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-csv", type=str, default=None,
                    help="path to TrainData.csv; auto-detected if omitted")
    ap.add_argument("--test-variant", type=str, default="TestData.csv",
                    help="TestData.csv (Kitsune), NewTestData.csv (mKitsune), Recurring.csv (rKitsune)")
    ap.add_argument("--out", type=str, default="out/results.json")
    args_cli = ap.parse_args()

    print(f"DEVICE = {device_mod.DEVICE_STR}")
    t0 = time.time()
    split = load_kitsune(
        fraction=1.0, seed=args_cli.seed,
        train_csv=args_cli.train_csv, test_variant=args_cli.test_variant,
    )
    sanity_print(split)
    print(f"loaded in {time.time()-t0:.1f}s")

    args = MateenArgs()
    out: dict = {"args": vars(args), "device": device_mod.DEVICE_STR,
                 "split": dict(
                     train_size=split.train_size,
                     test_size=split.test_size,
                     feature_dim=split.feature_dim,
                     train_pos_rate=split.train_pos_rate,
                     test_pos_rate=split.test_pos_rate,
                     test_label_hash=split.test_label_hash,
                     train_index_hash=split.train_index_hash,
                 )}

    if args_cli.mode in ("both", "no_update"):
        print("\n=== No-Update baseline ===")
        t0 = time.time()
        nu = run_no_update(split, args, init_epochs=args_cli.init_epochs,
                           seed=args_cli.seed)
        nu["wall_seconds"] = time.time() - t0
        print(fmt_metrics(nu))
        out["no_update"] = nu

    if args_cli.mode in ("both", "mateen"):
        print("\n=== Mateen (full adaptive ensemble) ===")
        t0 = time.time()
        mt = run_mateen(split, args, init_epochs=args_cli.init_epochs,
                        seed=args_cli.seed)
        mt["wall_seconds"] = time.time() - t0
        print(fmt_metrics(mt))
        out["mateen"] = mt

    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {args_cli.out}")


if __name__ == "__main__":
    main()
