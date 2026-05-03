"""Dump the minimal CSV used to draw cross_paper_summary.{pdf,png}.

Output: cross_paper_plot_data.csv with one row per (work, fraction).
Columns: work, fraction, attack_f1_mean, attack_f1_std, tpr_mean, tpr_std
"""
from __future__ import annotations
import csv
import os


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cross_paper_summary_wide.csv")
OUT = os.path.join(HERE, "cross_paper_plot_data.csv")

# (work, mode) pairs that the plot actually draws
KEEP = [
    ("KitNET / Mirai",      "kitnet"),
    ("Mateen / CICIDS2017", "mateen"),
    ("Mateen / Kitsune",    "mateen"),
    ("Wang / MANDA-IDS",    "ids"),
    ("Wang / MANDA-CW",     "manda"),
]


def main():
    out_rows = []
    with open(SRC) as f:
        for r in csv.DictReader(f):
            if (r["work"], r["mode"]) not in KEEP:
                continue
            out_rows.append({
                "work": r["work"],
                "fraction": float(r["fraction"]),
                "attack_f1_mean": r["attack_f1_mean"],
                "attack_f1_std": r["attack_f1_std"],
                "tpr_mean": r["tpr_mean"],
                "tpr_std": r["tpr_std"],
            })

    # Sort: by work order in KEEP, then fraction descending
    work_idx = {w: i for i, (w, _) in enumerate(KEEP)}
    out_rows.sort(key=lambda r: (work_idx[r["work"]], -r["fraction"]))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "work", "fraction",
            "attack_f1_mean", "attack_f1_std",
            "tpr_mean", "tpr_std",
        ])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"saved {OUT}  ({len(out_rows)} rows)")
    print(f"\n{'work':<22} {'frac':>7} {'F1mean':>8} {'F1std':>7} "
          f"{'TPRmean':>8} {'TPRstd':>7}")
    print("-" * 65)
    for r in out_rows:
        print(f"{r['work']:<22} {r['fraction']:>7.4f} "
              f"{float(r['attack_f1_mean'])*100:>7.2f}% "
              f"{float(r['attack_f1_std'])*100:>6.2f}% "
              f"{float(r['tpr_mean'])*100:>7.2f}% "
              f"{float(r['tpr_std'])*100:>6.2f}%")


if __name__ == "__main__":
    main()
