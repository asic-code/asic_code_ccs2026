"""Render the cross-paper attack-F1 + TPR vs training-fraction figure."""
from __future__ import annotations
import os
import csv

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "cross_paper_summary_wide.csv")


LINES = [
    ("KitNET / Mirai",       "kitnet",  "KitNET / Mirai",            "#1f77b4", "o"),
    ("Mateen / CICIDS2017",  "mateen",  "Alotaibi et al. / CICIDS2017", "#d62728", "s"),
    ("Mateen / Kitsune",     "mateen",  "Alotaibi et al. / Kitsune", "#9467bd", "^"),
    ("Wang / MANDA-CW",      "manda",   "Wang et al. / NSL-KDD",     "#2ca02c", "D"),
]


def load_wide():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def filter_line(rows, work, mode):
    out = [r for r in rows if r["work"] == work and r["mode"] == mode]
    out.sort(key=lambda r: -float(r["fraction"]))
    return out


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    rows = load_wide()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    work_handles = []
    for (work, mode, label, color, marker) in LINES:
        cells = filter_line(rows, work, mode)
        if not cells:
            continue
        xs = np.array([float(r["fraction"]) for r in cells])

        f1_mean = np.array([float(r["attack_f1_mean"])
                              if r["attack_f1_mean"] not in ("", "nan") else np.nan
                              for r in cells]) * 100
        f1_std = np.array([float(r["attack_f1_std"])
                             if r["attack_f1_std"] not in ("", "nan") else 0
                             for r in cells]) * 100
        tpr_mean = np.array([float(r["tpr_mean"])
                              if r["tpr_mean"] not in ("", "nan") else np.nan
                              for r in cells]) * 100
        tpr_std = np.array([float(r["tpr_std"])
                              if r["tpr_std"] not in ("", "nan") else 0
                              for r in cells]) * 100

        ax.errorbar(xs, f1_mean, yerr=f1_std,
                     marker=marker, ls="-", color=color,
                     capsize=2.5, capthick=0.6, lw=1.3,
                     markersize=4.5)
        ax.errorbar(xs, tpr_mean, yerr=tpr_std,
                     marker=marker, ls="--", color=color,
                     capsize=2.5, capthick=0.6, lw=1.3,
                     markersize=4.5,
                     markerfacecolor="white")

        work_handles.append(mlines.Line2D(
            [], [], color=color, marker=marker, lw=1.3, ls="-",
            markersize=5.5, label=label,
        ))

    ax.set_xscale("log")
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Detection performance (%)")
    ax.set_xticks([1.0, 0.50, 0.25, 0.10, 0.05, 0.01])
    ax.set_xticklabels(["100%", "50%", "25%", "10%", "5%", "1%"])
    ax.set_xlim(1.18, 0.0085)
    ax.set_ylim(-3, 103)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.grid(False, which="minor")
    ax.tick_params(axis="both", which="major", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    metric_handles = [
        mlines.Line2D([], [], color="black", lw=1.3, ls="-",
                       label="Attack-F1"),
        mlines.Line2D([], [], color="black", lw=1.3, ls="--",
                       label="TPR"),
    ]
    leg = ax.legend(
        handles=work_handles + metric_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=3, borderpad=0.5, handlelength=2.4,
        fontsize=9, columnspacing=1.6, handletextpad=0.7,
    )

    fig.tight_layout()
    out_pdf = os.path.join(HERE, "cross_paper_summary.pdf")
    out_png = os.path.join(HERE, "cross_paper_summary.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
