#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys


@dataclass
class EvidenceCheck:
    name: str
    rel_root: str
    pattern: str


CHECKS = [
    EvidenceCheck(
        name="Figure 2 mapping evidence",
        rel_root="../Dataset_Paral/Evaluate_Mapping",
        pattern="**/*mapping_evaluation*.csv",
    ),
    EvidenceCheck(
        name="Figure 3 threshold evidence",
        rel_root="../Dataset/threshold_evaluation",
        pattern="**/Kmeans_thresholds.csv",
    ),
    EvidenceCheck(
        name="Figure 4/5 history evidence",
        rel_root="../Dataset_ISV_turnmap",
        pattern="**/*performance_history_eex.csv",
    ),
    EvidenceCheck(
        name="Table 2 summary metrics evidence",
        rel_root="../Dataset_ISV_turnmap",
        pattern="**/*summary_metrics.csv",
    ),
    EvidenceCheck(
        name="Table 4 incremental signatures evidence",
        rel_root="../Dataset_ISV_turnmap",
        pattern="**/*incremental_signatures_eex.csv",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check existence of paper-evidence CSV outputs."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to repository root (default: current directory)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    all_ok = True

    for check in CHECKS:
        root = (repo_root / check.rel_root).resolve()
        if not root.exists():
            print(f"[MISSING] {check.name}: root not found: {root}")
            all_ok = False
            continue

        matches = list(root.glob(check.pattern))
        if not matches:
            print(f"[MISSING] {check.name}: no match for {check.pattern} under {root}")
            all_ok = False
            continue

        print(f"[OK] {check.name}: {len(matches)} file(s)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
