#!/usr/bin/env python3
"""
ISV가 저장한 *_performance_history_eex.csv에 대해,
create_attackwise_history_graph.py 와 동일한 방식으로 턴별 막대 높이(bar_gen, bar_rem)를 계산해
CSV에 열만 추가한다(기존 열 값은 다른 열에 대해 변경하지 않음).

추가 열(1개):
  - signature_plot_total_displayed
    create_attackwise_history_graph 의 막대 높이(bar_gen + bar_rem) 합과 동일(턴 정렬·carry_in/out 반영).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils.signature_bar_plot_series import compute_signature_bar_plot_series, pick_signature_bar_columns


def resolve_existing_csv(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    if os.name == "nt":
        abs_path = os.path.abspath(path_str)
        if not abs_path.startswith("\\\\?\\"):
            extended = "\\\\?\\" + abs_path
            if os.path.isfile(extended):
                return Path(extended)
    return p


def find_latest_performance_history_csv(
    file_type: str,
    file_number: int,
    isv_root: Path,
) -> Path | None:
    base = isv_root / file_type
    if not base.is_dir():
        return None
    candidates: list[Path] = []
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        for p in sub.glob(f"{file_type}_{file_number}_*_performance_history_eex.csv"):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def annotate_csv(csv_path: Path, keep_no_alert: int, dry_run: bool) -> int:
    csv_path = resolve_existing_csv(str(csv_path))
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "turn" not in df.columns:
        print(f"Error: missing 'turn' column in {csv_path}", file=sys.stderr)
        return 1

    gen_col, rem_col = pick_signature_bar_columns(df, keep_no_alert)
    if not gen_col or not rem_col:
        print(
            f"Error: could not pick Generated/Removed bar columns in {csv_path}. "
            f"Columns: {list(df.columns)[:40]}...",
            file=sys.stderr,
        )
        return 1

    bar_gen, bar_rem, _used_eff = compute_signature_bar_plot_series(df, gen_col, rem_col)
    total = np.asarray(bar_gen, dtype=float) + np.asarray(bar_rem, dtype=float)
    col_name = "signature_plot_total_displayed"
    df[col_name] = np.rint(total).astype(np.int64)

    if dry_run:
        print(f"[dry-run] would write {len(df)} rows to {csv_path}")
        print(df[["turn", col_name]].head(10).to_string(index=False))
        return 0

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Updated {csv_path} with column: {col_name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add per-turn displayed signature bar counts to ISV performance_history_eex.csv"
    )
    parser.add_argument("--file_type", type=str, default=None, help="Dataset name (used with --isv_root to find CSV)")
    parser.add_argument("--file_number", type=int, default=1)
    parser.add_argument("--csv_path", type=str, default=None, help="Explicit path to *_performance_history_eex.csv")
    parser.add_argument(
        "--isv_root",
        type=str,
        default="../Dataset_ISV_turnmap",
        help="Root folder containing <file_type>/<run_params>/...csv (default: ../Dataset_ISV_turnmap)",
    )
    parser.add_argument(
        "--keep_no_alert",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, use *_no_alert_excluded actual_only columns when present (same as create_attackwise_history_graph).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print sample only; do not write CSV")
    args = parser.parse_args()

    if args.csv_path:
        target = Path(args.csv_path)
    elif args.file_type:
        isv_root = Path(args.isv_root)
        found = find_latest_performance_history_csv(args.file_type, args.file_number, isv_root)
        if found is None:
            print(
                f"Error: no *performance_history_eex.csv under {isv_root / args.file_type}",
                file=sys.stderr,
            )
            return 1
        target = found
        print(f"Using latest performance history: {target}")
    else:
        print("Error: provide --csv_path or --file_type", file=sys.stderr)
        return 1

    return annotate_csv(target, args.keep_no_alert, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
