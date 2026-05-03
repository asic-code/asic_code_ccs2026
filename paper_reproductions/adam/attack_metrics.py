"""Re-aggregate trial data with attack-side metrics."""
from __future__ import annotations
import csv
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def load_main_trials() -> list[dict]:
    path = os.path.join(HERE, "out", "dataset_lost_trials.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_audit_rows() -> list[dict]:
    path = os.path.join(HERE, "out", "audit_flatness.csv")
    out = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append({k: (float(v) if v not in ("", None) else None)
                        if k not in ("name",) else v
                        for k, v in r.items()})
    return out


def fmt_main(trials: list[dict]) -> str:
    by_frac: dict = {}
    for t in trials:
        by_frac.setdefault(t["frac"], []).append(t)
    lines = [
        "\n=== Main results — attack-side ===",
        f"{'frac':>6} {'n':>2} | "
        f"{'attack_F1':>14} {'TPR':>14} {'FPR':>14} "
        f"{'rec_DoS':>14} {'rec_R2L':>14} {'rec_U2R':>14}",
        "-" * 110,
    ]
    for frac in sorted(by_frac.keys(), reverse=True):
        group = by_frac[frac]
        def stat(key: str) -> str:
            vals = [g.get(key) for g in group if g.get(key) is not None]
            if not vals: return f"{'-':>14}"
            m = float(np.mean(vals))
            s = float(np.std(vals))
            return f"{m*100:6.2f}±{s*100:5.2f}".rjust(14)
        lines.append(
            f"{frac*100:5.1f}% {len(group):>2} | "
            f"{stat('f1')} {stat('recall')} {stat('FAR')} "
            f"{stat('recall_DoS')} {stat('recall_R2L')} {stat('recall_U2R')}"
        )
    return "\n".join(lines)


def fmt_audit(rows: list[dict]) -> str:
    rows_sorted = sorted(rows, key=lambda r: -r["frac"])
    lines = [
        "\n=== Extended audit — attack-side ===",
        f"{'frac':>8} | "
        f"{'binary_F1':>10} {'FPR':>10} {'rec_DoS':>10} "
        f"{'rec_Probe':>10} {'rec_R2L':>10} {'rec_U2R':>10} {'macro_F1':>10}",
        "-" * 95,
    ]
    for r in rows_sorted:
        def fmt(k):
            v = r.get(k)
            return f"{v*100:7.2f}%".rjust(10) if v is not None else f"{'-':>10}"
        lines.append(
            f"{r['frac']*100:7.4f}% | "
            f"{fmt('binary_F1')} {fmt('binary_FPR')} {fmt('recall_DoS')} "
            f"{fmt('recall_Probe')} {fmt('recall_R2L')} {fmt('recall_U2R')} "
            f"{fmt('macro_F1')}"
        )
    return "\n".join(lines)


def write_consolidated_csv(main_trials: list[dict],
                            audit_rows: list[dict],
                            out_path: str) -> None:
    fields = [
        "source", "frac", "seed",
        "binary_F1", "binary_FPR",
        "recall_DoS", "recall_Probe", "recall_R2L", "recall_U2R",
        "macro_F1",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in main_trials:
            w.writerow({
                "source": "main",
                "frac": t.get("frac"),
                "seed": t.get("seed"),
                "binary_F1": t.get("f1"),
                "binary_FPR": t.get("FAR"),
                "recall_DoS": t.get("recall_DoS"),
                "recall_Probe": t.get("recall_Probe"),
                "recall_R2L": t.get("recall_R2L"),
                "recall_U2R": t.get("recall_U2R"),
                "macro_F1": t.get("macro_F1"),
            })
        for r in audit_rows:
            w.writerow({
                "source": "audit",
                "frac": r.get("frac"),
                "seed": "-",
                "binary_F1": r.get("binary_F1"),
                "binary_FPR": r.get("binary_FPR"),
                "recall_DoS": r.get("recall_DoS"),
                "recall_Probe": r.get("recall_Probe"),
                "recall_R2L": r.get("recall_R2L"),
                "recall_U2R": r.get("recall_U2R"),
                "macro_F1": r.get("macro_F1"),
            })


def main() -> None:
    main_trials = load_main_trials()
    audit_rows = load_audit_rows()
    print(f"loaded {len(main_trials)} main trials, {len(audit_rows)} audit rows")

    print(fmt_main(main_trials))
    print(fmt_audit(audit_rows))

    out_csv = os.path.join(HERE, "out", "attack_metrics.csv")
    write_consolidated_csv(main_trials, audit_rows, out_csv)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
