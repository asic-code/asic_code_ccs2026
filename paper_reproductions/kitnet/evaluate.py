"""Single end-to-end KitNET run on the released Mirai sample."""
from __future__ import annotations
import argparse
import json
import os
import time

from data_loader import load_mirai, sanity_print
from kitnet_runner import run_kitnet, headline_metrics


def fmt(d: dict) -> str:
    return (
        f"  ── paper-side ─────────\n"
        f"    AUC          : {d['auc']*100:7.4f}%\n"
        f"    EER          : {d['eer']*100:7.4f}%\n"
        f"    TPR @ FPR=0    : {d['tpr_at_fpr0']*100:7.4f}%\n"
        f"    TPR @ FPR=.001 : {d['tpr_at_fpr_001']*100:7.4f}%\n"
        f"  ── operating point (99.9th-pct benign threshold) ─\n"
        f"    accuracy     : {d['accuracy']*100:7.4f}%\n"
        f"    macro_f1     : {d['macro_f1']*100:7.4f}%\n"
        f"    threshold    : {d['threshold']:.6e}\n"
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
    ap.add_argument("--max-ae", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="out/results.json")
    args_cli = ap.parse_args()

    print("DEVICE = cpu (KitNET is pure numpy)")
    t0 = time.time()
    split = load_mirai(fraction=1.0, seed=args_cli.seed)
    sanity_print(split)
    print(f"loaded in {time.time()-t0:.1f}s")

    print(f"\n=== KitNET (m={args_cli.max_ae}, FMgrace={split.fm_grace}, "
          f"ADgrace={split.ad_grace}) ===")
    t0 = time.time()
    rmse = run_kitnet(
        split.X,
        fm_grace=split.fm_grace,
        ad_grace=split.ad_grace,
        seed=args_cli.seed,
        max_ae=args_cli.max_ae,
    )
    wall = time.time() - t0
    print(f"KitNET wall: {wall:.1f}s")

    metrics = headline_metrics(split.y, rmse, eval_start=split.eval_start)
    print(fmt(metrics))

    out = dict(
        seed=args_cli.seed,
        max_ae=args_cli.max_ae,
        fm_grace=split.fm_grace,
        ad_grace=split.ad_grace,
        eval_start=split.eval_start,
        eval_size=int(split.n_total - split.eval_start),
        feature_dim=split.feature_dim,
        label_hash=split.label_hash,
        wall_seconds=wall,
        **metrics,
    )
    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {args_cli.out}")


if __name__ == "__main__":
    main()
