"""Export Phase-4 attack-F1 + TPR for the four reproductions covered
in this session (KitNET, Mateen-CICIDS, Mateen-Kitsune, Wang/MANDA)
into two CSV layouts:

  cross_paper_summary_long.csv  (one row per work × mode × fraction × metric)
  cross_paper_summary_wide.csv  (one row per work × mode × fraction)
"""
from __future__ import annotations
import csv
import json
import os
from typing import Iterable

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- KitNET
def load_kitnet():
    p = os.path.join(HERE, "kitnet", "out", "dataset_lost", "raw.jsonl")
    rows = []
    for line in open(p):
        r = json.loads(line)
        rows.append({
            "work": "KitNET / Mirai",
            "mode": "kitnet",
            "fraction": r["fraction"],
            "seed": r["seed"],
            "attack_f1": r["attack_f1"],
            "tpr": r["attack_recall"],
            "fpr": r["fpr"],
        })
    return rows


# ---------------------------------------------------------------- Mateen-CICIDS
def load_mateen_cicids():
    p = os.path.join(HERE, "mateen", "cicids", "out", "dataset_lost",
                     "raw.jsonl")
    rows = []
    for line in open(p):
        r = json.loads(line)
        rows.append({
            "work": "Mateen / CICIDS2017",
            "mode": r["mode"],   # 'mateen' or 'no_update'
            "fraction": r["fraction"],
            "seed": r["seed"],
            "attack_f1": r["attack_f1"],
            "tpr": r["attack_recall"],
            "fpr": r["fpr"],
        })
    return rows


# ---------------------------------------------------------------- Mateen-Kitsune
def load_mateen_kitsune():
    p = os.path.join(HERE, "mateen", "kitsune", "out", "dataset_lost",
                     "raw.jsonl")
    rows = []
    for line in open(p):
        r = json.loads(line)
        rows.append({
            "work": "Mateen / Kitsune",
            "mode": r["mode"],
            "fraction": r["fraction"],
            "seed": r["seed"],
            "attack_f1": r["attack_f1"],
            "tpr": r["attack_recall"],
            "fpr": r["fpr"],
        })
    return rows


# ---------------------------------------------------------------- Wang / MANDA
def load_wang_manda():
    """Wang's raw is a per-trial dict with `attacks: {fgsm: {manifold: {...},
    db: {...}, manda: {...}}, bim: ..., cw: ...}`. We export the MANDA
    combined detector for each attack as separate 'modes', and the
    IDS-side TPR/FPR (newly added) as a fourth 'mode'.
    """
    p = os.path.join(HERE, "wang", "out", "dataset_lost",
                     "dataset_lost_raw.json")
    raw = json.load(open(p))
    rows = []
    for r in raw:
        if r.get("status") != "ok":
            continue
        # IDS-side row (positive = attack flow). Compute attack-F1 from
        # IDS confusion-matrix counts so MANDA-IDS is comparable to the
        # other works on the same axis.
        if "ids_attack_recall" in r:
            tpr = r["ids_attack_recall"]
            fpr = r["ids_fpr"]
            n_a = r["ids_n_attack"]
            n_b = r["ids_n_benign"]
            tp = tpr * n_a
            fp = fpr * n_b
            prec = tp / max(1e-9, tp + fp)
            f1 = 2 * prec * tpr / max(1e-9, prec + tpr) if (prec + tpr) > 0 else 0.0
            rows.append({
                "work": "Wang / MANDA-IDS",
                "mode": "ids",
                "fraction": r["frac"],
                "seed": r["seed"],
                "attack_f1": float(f1),
                "tpr": float(tpr),
                "fpr": float(fpr),
            })
        # Per-attack MANDA combined (positive = AE)
        for atk in ("fgsm", "bim", "cw"):
            ar = r.get("attacks", {}).get(atk, {})
            if "manda" not in ar:
                continue
            m = ar["manda"]
            rows.append({
                "work": f"Wang / MANDA-{atk.upper()}",
                "mode": "manda",
                "fraction": r["frac"],
                "seed": r["seed"],
                "attack_f1": m["f1"],
                "tpr": m["recall"],
                "fpr": float("nan"),  # MANDA evaluated at FPR=5% by op-point
            })
    return rows


# ---------------------------------------------------------------- combine
def collect_long():
    rows = []
    rows.extend(load_kitnet())
    rows.extend(load_mateen_cicids())
    rows.extend(load_mateen_kitsune())
    rows.extend(load_wang_manda())
    return rows


def aggregate_wide(rows: list[dict]) -> list[dict]:
    """Per (work, mode, fraction): compute mean ± std across seeds for
    attack_f1, tpr, fpr."""
    by_cell: dict = {}
    for r in rows:
        key = (r["work"], r["mode"], r["fraction"])
        by_cell.setdefault(key, []).append(r)
    wide = []
    for (work, mode, frac), group in by_cell.items():
        cell = dict(work=work, mode=mode, fraction=frac, n_seeds=len(group))
        for k in ("attack_f1", "tpr", "fpr"):
            vals = [g[k] for g in group
                    if g.get(k) is not None
                    and not (isinstance(g[k], float) and np.isnan(g[k]))]
            if vals:
                cell[f"{k}_mean"] = float(np.mean(vals))
                cell[f"{k}_std"] = float(np.std(vals))
            else:
                cell[f"{k}_mean"] = float("nan")
                cell[f"{k}_std"] = float("nan")
        wide.append(cell)
    return wide


def write_csv(rows: list[dict], path: str, fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    long_rows = collect_long()
    wide_rows = aggregate_wide(long_rows)

    # Sort wide for readability
    work_order = [
        "KitNET / Mirai",
        "Mateen / CICIDS2017",
        "Mateen / Kitsune",
        "Wang / MANDA-IDS",
        "Wang / MANDA-FGSM",
        "Wang / MANDA-BIM",
        "Wang / MANDA-CW",
    ]
    work_idx = {w: i for i, w in enumerate(work_order)}
    wide_rows.sort(key=lambda r: (work_idx.get(r["work"], 99),
                                    r["mode"],
                                    -r["fraction"]))
    long_rows.sort(key=lambda r: (work_idx.get(r["work"], 99),
                                    r["mode"],
                                    -r["fraction"],
                                    r["seed"]))

    long_path = os.path.join(HERE, "cross_paper_summary_long.csv")
    wide_path = os.path.join(HERE, "cross_paper_summary_wide.csv")

    write_csv(long_rows, long_path,
              ["work", "mode", "fraction", "seed",
               "attack_f1", "tpr", "fpr"])
    write_csv(wide_rows, wide_path,
              ["work", "mode", "fraction", "n_seeds",
               "attack_f1_mean", "attack_f1_std",
               "tpr_mean", "tpr_std",
               "fpr_mean", "fpr_std"])

    print(f"long: {long_path}  ({len(long_rows)} rows)")
    print(f"wide: {wide_path}  ({len(wide_rows)} rows)")

    # Pretty preview: wide
    print("\n=== preview cross_paper_summary_wide.csv ===")
    print(f"{'work':<22} {'mode':<10} {'frac':>7} {'n':>3} | "
          f"{'attack_F1':>14} {'TPR':>14} {'FPR':>14}")
    print("-" * 100)
    for r in wide_rows:
        def fmt(k):
            m, s = r.get(f"{k}_mean"), r.get(f"{k}_std")
            if m is None or (isinstance(m, float) and np.isnan(m)):
                return f"{'-':>14}"
            return f"{m*100:6.2f}±{s*100:5.2f}".rjust(14)
        print(f"{r['work']:<22} {r['mode']:<10} {r['fraction']*100:>6.1f}% "
              f"{r['n_seeds']:>3} | {fmt('attack_f1')} {fmt('tpr')} {fmt('fpr')}")


if __name__ == "__main__":
    main()
