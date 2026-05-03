"""Re-aggregate Wang/MANDA dataset-lost trial JSONs with component breakdown."""
from __future__ import annotations
import json
import os
import csv

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
        f"\n=== MANDA vs components — {metric} @ FPR=5 % (mean ± std) ===",
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
            tag = "<--" if edge < -2 else ""
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
        "\n=== AE pool size (n_ae) per trial ===",
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
        f"\n=== Per-seed values for {attack.upper()} {metric} ===",
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
        "\n=== Attack success rate per (fraction, attack) ===",
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

    out_csv = os.path.join(HERE, "out", "component_audit.csv")
    write_consolidated_csv(agg, out_csv)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
