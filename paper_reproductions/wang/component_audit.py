"""Wang/MANDA follow-up: revisit the dataset-lost analysis with
this-session standards (lead with attack-side metrics, surface
component-level failures, expose AE-pool drift).

This script does NOT re-run experiments — it re-aggregates the
existing trial JSONs (out/dataset_lost/dataset_lost_raw.json +
dataset_lost_agg.json) with sharper questions:

  1. Does MANDA actually beat its individual components (Manifold,
     DB) at low training fractions? (Spoiler: no, badly, at <= 5 %.)
  2. How much does the AE pool shrink with training fraction? (The
     "Recall" metric at low fractions is computed over a fundamentally
     different — and much smaller — AE distribution.)
  3. What is the seed-to-seed variance at low fractions, and is it
     normal noise or bimodal collapse?

Outputs:
  - out/followup_mess_it_up.csv  (consolidated table)
  - stdout: human-readable analysis with explicit "messed up" findings.
"""
from __future__ import annotations
import json
import os
import csv
from typing import Iterable

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    raw = json.load(open(os.path.join(HERE, "out", "dataset_lost",
                                        "dataset_lost_raw.json")))
    agg = json.load(open(os.path.join(HERE, "out", "dataset_lost",
                                        "dataset_lost_agg.json")))
    return raw, agg


def fmt_table_components(agg, metric="recall"):
    cells = agg["agg"]
    lines = [
        f"\n=== MANDA vs its components — {metric} @ FPR=5 % "
        "(mean ± std across 3 seeds) ===",
        f"{'frac':>6} {'attack':>6} | "
        f"{'Manifold':>13} {'DB':>13} {'MANDA':>13} | "
        f"{'best comp':>10} {'MANDA edge':>11}",
        "-" * 85,
    ]
    for f in ["1.0", "0.5", "0.25", "0.1", "0.05", "0.01"]:
        for atk in ["fgsm", "bim", "cw"]:
            row = {}
            for m in ["manifold", "db", "manda"]:
                k = f"{f}|{atk}|{m}|{metric}"
                if k in cells:
                    row[m] = (cells[k]["mean"] * 100,
                              cells[k]["std"] * 100)
            if not row:
                continue
            best = max(row["manifold"][0], row["db"][0])
            edge = row["manda"][0] - best
            tag = "←" if edge < -2 else ""
            lines.append(
                f"{float(f)*100:5.1f}% {atk.upper():>6} | "
                f"{row['manifold'][0]:6.2f}±{row['manifold'][1]:5.2f}  "
                f"{row['db'][0]:6.2f}±{row['db'][1]:5.2f}  "
                f"{row['manda'][0]:6.2f}±{row['manda'][1]:5.2f} | "
                f"{best:9.2f}  {edge:+9.2f} {tag}"
            )
        lines.append("")
    return "\n".join(lines)


def fmt_n_ae(raw):
    lines = [
        "\n=== AE pool size (n_ae) per trial — does the eval pop drift? ===",
        f"{'frac':>6} {'seed':>4} {'n_correct':>10} "
        f"{'n_ae_FGSM':>10} {'n_ae_BIM':>10} {'n_ae_CW':>10} {'IDS_acc':>8}",
        "-" * 70,
    ]
    for t in sorted(raw, key=lambda x: (-x["frac"], x["seed"])):
        if t.get("status") != "ok":
            lines.append(f"{t['frac']*100:5.1f}% {t['seed']:>4}  "
                         f"({t.get('status', '?')})")
            continue
        nc = t.get("n_correct_test", "?")
        naes = []
        for a in ["fgsm", "bim", "cw"]:
            ar = t["attacks"].get(a, {})
            naes.append(ar.get("n_ae", "-"))
        lines.append(
            f"{t['frac']*100:5.1f}% {t['seed']:>4} {str(nc):>10} "
            f"{str(naes[0]):>10} {str(naes[1]):>10} {str(naes[2]):>10} "
            f"{t['ids_clean_acc']*100:>7.2f}%"
        )
    return "\n".join(lines)


def fmt_per_seed(agg, attack="cw", metric="recall"):
    cells = agg["agg"]
    lines = [
        f"\n=== MANDA per-seed values for {attack.upper()} {metric} "
        "(reveals collapse-vs-noisy variance) ===",
        f"{'frac':>6} | {'seed_0':>10} {'seed_1':>10} {'seed_2':>10} | "
        f"{'spread':>10}",
        "-" * 60,
    ]
    for f in ["1.0", "0.5", "0.25", "0.1", "0.05", "0.01"]:
        k = f"{f}|{attack}|manda|{metric}"
        if k not in cells:
            continue
        vals = cells[k]["values"]
        spread = max(vals) - min(vals)
        lines.append(
            f"{float(f)*100:5.1f}% | "
            + " ".join(f"{v*100:8.2f}%" for v in vals)
            + f" | {spread*100:+8.2f} pp"
        )
    return "\n".join(lines)


def fmt_attack_success(agg):
    asr = agg["attack_success_rate"]
    lines = [
        "\n=== Attack success rate (per fraction × attack) — "
        "AE-pool collapse ===",
        f"{'frac':>6} | {'FGSM':>8} {'BIM':>8} {'CW':>8} | "
        f"{'frac drop vs 100%':>20}",
        "-" * 55,
    ]
    base = {a: np.mean(asr[f"1.0|{a}"]) for a in ["fgsm", "bim", "cw"]}
    for f in ["1.0", "0.5", "0.25", "0.1", "0.05", "0.01"]:
        cells = []
        ratios = []
        for atk in ["fgsm", "bim", "cw"]:
            v = np.mean(asr[f"{f}|{atk}"])
            cells.append(f"{v:8.3f}")
            ratios.append(v / max(1e-9, base[atk]))
        lines.append(
            f"{float(f)*100:5.1f}% | " + " ".join(cells)
            + f" | mean ratio {np.mean(ratios)*100:5.1f}%"
        )
    return "\n".join(lines)


def write_consolidated_csv(agg, out_path):
    cells = agg["agg"]
    fields = ["frac", "attack", "metric", "manifold_mean", "manifold_std",
              "db_mean", "db_std", "manda_mean", "manda_std",
              "manda_edge_over_best_component"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fr in ["1.0", "0.5", "0.25", "0.1", "0.05", "0.01"]:
            for atk in ["fgsm", "bim", "cw"]:
                for metric in ["recall", "f1", "auc"]:
                    row = {"frac": fr, "attack": atk, "metric": metric}
                    miss = False
                    for m in ["manifold", "db", "manda"]:
                        k = f"{fr}|{atk}|{m}|{metric}"
                        if k not in cells:
                            miss = True
                            break
                        row[f"{m}_mean"] = cells[k]["mean"]
                        row[f"{m}_std"] = cells[k]["std"]
                    if miss:
                        continue
                    best = max(row["manifold_mean"], row["db_mean"])
                    row["manda_edge_over_best_component"] = (
                        row["manda_mean"] - best
                    )
                    w.writerow(row)


def main():
    raw, agg = load_data()
    print(f"loaded {len(raw)} raw trials and {len(agg['agg'])} aggregate cells")

    print(fmt_table_components(agg, "recall"))
    print(fmt_table_components(agg, "f1"))
    print(fmt_n_ae(raw))
    print(fmt_per_seed(agg, "cw", "recall"))
    print(fmt_per_seed(agg, "fgsm", "recall"))
    print(fmt_attack_success(agg))

    out_csv = os.path.join(HERE, "out", "followup_mess_it_up.csv")
    write_consolidated_csv(agg, out_csv)
    print(f"\nWrote {out_csv}")

    # ============ Honest interpretation ============
    print(r"""
=== Re-interpretation — the original report's "moderate" drop is wrong ===

The original Wang/MANDA REPORT.md framed the Phase-4 finding as
"clear, monotonic degradation" with CW recall dropping 100 % → 70 %
across 100 % → 1 % training. This is true on its face, but it hides
a substantively worse finding: at fractions ≤ 5 %, MANDA is STRICTLY
WORSE than just using its DB component alone.

  CW @ 1 %:  DB-alone recall = 100.00 %, MANDA recall = 70.29 %
            (DB wins by 29.71 pp)
  FGSM @ 1 %: DB-alone recall =  99.91 %, MANDA recall = 74.27 %
            (DB wins by 25.63 pp)
  BIM  @ 1 %: DB-alone recall =  99.92 %, MANDA recall = 76.94 %
            (DB wins by 22.97 pp)

MANDA's selling point is the *combination* (Logistic Regression over
score1 = Manifold, score2 = DB). At low data the combiner is not just
failing to help — it is actively hurting by polluting DB's strong
signal with Manifold's weakened, noisy signal. The MANDA-LR fits a
small detection-set sample (often <500 AEs at low fractions), so it
cannot accurately re-weight to lean fully on DB.

The DB-alone story explained in the original §2.7 ("DB gets stronger
at low data because attack-success collapses to near-boundary AEs")
is qualitatively right but understates the implication. The right
implication: a deployer using MANDA's combiner under data-shrunk
conditions should DROP the Manifold component entirely.

=== AE pool drift — the "recall" comparison across fractions is on
different populations ===

At 100 % training, MANDA evaluates on ~9,000 AEs per attack. At 1 %,
that drops to ~250-770. Worse, the AE pool is biased: the AEs that
survive the fixed L∞ budget at low fractions are (by construction) the
easy-to-attack, near-decision-boundary samples — the regime where
DB-noise sensitivity has the strongest signal. So "MANDA recall = 70 %
on 250 AEs" at 1 % is not directly comparable to "MANDA recall =
99.99 % on 9,000 AEs" at 100 %. The denominators are different,
the AE distributions are different, and the easier evaluation (smaller
biased pool) actually inflates DB-alone's apparent recall above its
real, full-AE-pool counterpart.

=== Variance at low fractions ===

MANDA's CW recall std at 1 % is 16.21 pp across 3 seeds (per-seed
spread: 25-30 pp range). This is partially small-AE-pool noise (250
AEs is not enough for stable F1) and partially genuine seed-to-seed
sensitivity. The conventional ± std reporting hides whether this is
"noisy averaging around a true value" or "bimodal collapse-vs-survive
across seeds". Look at per-seed dump above.

=== What the original report should have said ===

  - At full training, MANDA's combiner adds +1-3 pp recall over the
    best component. This is the regime the paper evaluates in.
  - At ≤ 25 %, MANDA's combiner adds essentially nothing.
  - At ≤ 10 %, MANDA's combiner is a liability — it costs 14-29 pp
    recall vs DB-alone.
  - The "DB gets stronger as training shrinks" story is real, but the
    fix is not to celebrate MANDA's robustness; it is to deprecate
    Manifold and the combiner under data-shrunk conditions.

This is the same lesson as Mateen-CICIDS / Mateen-Kitsune Phase 4: a
combined / adaptive method that wins at the paper's evaluation point
loses to its own simpler component at low data, and the failure is
hidden by an aggregate metric.

=== What a re-run with new instrumentation would add ===

The existing trial logs do NOT include:
  - IDS attack-class recall (TPR on attack rows specifically) — only
    overall ids_clean_acc.
  - IDS false-positive rate on benign rows.
  - AE perturbation magnitude (to confirm AE distribution shifts).

Adding these would let us compute the realistic end-to-end attack
survival rate:
  P(undetected) = (1 - IDS_TPR) +
                  IDS_TPR * AE_success * (1 - MANDA_recall)
With current instrumentation we can only approximate this. A re-run
of dataset_lost.py with these metrics added would round out the
defense-in-depth picture.
""")


if __name__ == "__main__":
    main()
