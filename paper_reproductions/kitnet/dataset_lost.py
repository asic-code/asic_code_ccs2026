"""Dataset-lost ablation for KitNET / Mirai."""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

from data_loader import (
    load_mirai,
    AD_GRACE_DEFAULT,
)
from kitnet_runner import run_kitnet, headline_metrics


DEFAULT_FRACTIONS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.01)


def trial_record(frac: float, seed: int, split, metrics: dict,
                  wall: float) -> dict:
    return dict(
        fraction=frac,
        seed=seed,
        ad_grace=split.ad_grace,
        fm_grace=split.fm_grace,
        eval_start=split.eval_start,
        eval_size=int(split.n_total - split.eval_start),
        feature_dim=split.feature_dim,
        label_hash=split.label_hash,
        wall_seconds=wall,
        **metrics,
    )


def aggregate(rows: list[dict]) -> dict:
    keys = (
        "auc", "eer", "tpr_at_fpr0", "tpr_at_fpr_001",
        "accuracy", "attack_recall", "attack_precision", "attack_f1",
        "fpr",
    )
    by_cell: dict = {}
    for r in rows:
        by_cell.setdefault(r["fraction"], []).append(r)
    agg: dict = {}
    for frac, group in by_cell.items():
        cell = dict(fraction=frac, n_seeds=len(group))
        for k in keys:
            vals = [g.get(k) for g in group if g.get(k) is not None
                    and not (isinstance(g.get(k), float) and np.isnan(g.get(k)))]
            if vals:
                cell[f"{k}_mean"] = float(np.mean(vals))
                cell[f"{k}_std"] = float(np.std(vals))
                cell[f"{k}_per_seed"] = [float(v) for v in vals]
        agg[f"frac@{frac}"] = cell
    return agg


def fmt_table(agg: dict) -> str:
    lines = [
        "\n=== KitNET — attack-side metrics (mean ± std, %) ===",
        f"{'frac':>7} {'n':>3} | {'attack_F1':>14} {'TPR':>14} "
        f"{'fpr':>14} {'auc':>14} {'eer':>14} {'tpr@fpr=.001':>14}",
        "-" * 105,
    ]
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["fraction"])
    for _, c in rows:
        def fmt(m, s):
            if m is None: return f"{'-':>14}"
            return f"{m*100:6.2f}±{(s or 0)*100:5.2f}".rjust(14)
        lines.append(
            f"{c['fraction']*100:5.1f}% {c['n_seeds']:>3} | "
            f"{fmt(c.get('attack_f1_mean'), c.get('attack_f1_std'))} "
            f"{fmt(c.get('attack_recall_mean'), c.get('attack_recall_std'))} "
            f"{fmt(c.get('fpr_mean'), c.get('fpr_std'))} "
            f"{fmt(c.get('auc_mean'), c.get('auc_std'))} "
            f"{fmt(c.get('eer_mean'), c.get('eer_std'))} "
            f"{fmt(c.get('tpr_at_fpr_001_mean'), c.get('tpr_at_fpr_001_std'))}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=list(DEFAULT_FRACTIONS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-ae", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="out/dataset_lost")
    args_cli = ap.parse_args()

    print(f"fractions = {args_cli.fractions}")
    print(f"seeds = {args_cli.seeds}")
    os.makedirs(args_cli.out_dir, exist_ok=True)
    raw_path = os.path.join(args_cli.out_dir, "raw.jsonl")
    agg_path = os.path.join(args_cli.out_dir, "agg.json")

    rows: list[dict] = []
    done: set = set()
    if os.path.exists(raw_path):
        with open(raw_path) as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                    rows.append(r)
                    done.add((round(float(r["fraction"]), 6),
                              int(r["seed"])))
                except Exception:
                    pass
        print(f"resumed: {len(done)} trials already complete")

    with open(raw_path, "a") as f:
        for frac in args_cli.fractions:
            for seed in args_cli.seeds:
                if (round(float(frac), 6), int(seed)) in done:
                    print(f"frac={frac} seed={seed} SKIP (already in raw.jsonl)")
                    continue
                split = load_mirai(fraction=frac, seed=seed)
                print(f"\nfrac={frac} seed={seed} ad_grace={split.ad_grace} ...")
                t0 = time.time()
                rmse = run_kitnet(
                    split.X,
                    fm_grace=split.fm_grace,
                    ad_grace=split.ad_grace,
                    seed=seed,
                    max_ae=args_cli.max_ae,
                )
                wall = time.time() - t0
                m = headline_metrics(split.y, rmse, eval_start=split.eval_start)
                rec = trial_record(frac, seed, split, m, wall)
                rows.append(rec)
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                print(f"  AUC={m['auc']*100:.2f}% EER={m['eer']*100:.2f}% "
                      f"attack_f1={m['attack_f1']*100:.2f}% "
                      f"tpr={m['attack_recall']*100:.2f}% "
                      f"fpr={m['fpr']*100:.2f}% "
                      f"wall={wall:.1f}s "
                      f"pred={m.get('pred_dist')}")

    agg = aggregate(rows)
    with open(agg_path, "w") as fh:
        json.dump(agg, fh, indent=2, default=str)
    print(f"\nWrote {raw_path} ({len(rows)} trials)")
    print(f"Wrote {agg_path}")
    print(fmt_table(agg))


if __name__ == "__main__":
    main()
