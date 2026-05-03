#!/usr/bin/env bash
# Driver for the four cross-paper reproductions plus the dataset-removal
# (Phase-4) experiment that produces the values in the paper figure.
#
# Each subproject is a self-contained `uv` project. This script runs them
# in order: KitNET → Mateen-CICIDS → Mateen-Kitsune → Wang-MANDA → ADAM,
# then aggregates the per-trial outputs into `cross_paper_summary_*.csv`
# and regenerates `cross_paper_summary.{pdf,png}`.
#
# Datasets are NOT bundled. See the per-subproject `data/` paths printed
# below; each loader raises a FileNotFoundError with the expected location
# if the data is missing.
#
# Env overrides:
#   SKIP_KITNET=1          skip KitNET / Mirai
#   SKIP_MATEEN_CICIDS=1   skip Mateen / CICIDS2017
#   SKIP_MATEEN_KITSUNE=1  skip Mateen / Kitsune
#   SKIP_WANG=1            skip Wang / MANDA (NSL-KDD)
#   SKIP_ADAM=1            skip ADAM / DARPA'98
#   SKIP_AGGREGATE=1       skip the cross-paper aggregation + plot
#   PYTHON=python3.11      override Python interpreter for `uv` projects

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

run_subproject() {
  local label="$1" subdir="$2"; shift 2
  echo
  echo "================================================================"
  echo "[$label] $subdir"
  echo "================================================================"
  ( cd "$subdir" && uv sync && uv run python "$@" )
}

[[ "${SKIP_KITNET:-0}"        = "1" ]] || run_subproject "KitNET / Mirai"        kitnet         main.py phase4 --seeds 0 1 2
[[ "${SKIP_MATEEN_CICIDS:-0}" = "1" ]] || run_subproject "Mateen / CICIDS2017"   mateen/cicids  main.py phase4 --seeds 0 1 2
[[ "${SKIP_MATEEN_KITSUNE:-0}" = "1" ]] || run_subproject "Mateen / Kitsune"     mateen/kitsune main.py phase4 --seeds 0 1 2
[[ "${SKIP_WANG:-0}"          = "1" ]] || run_subproject "Wang / MANDA"          wang           dataset_lost.py --seeds 0 1 2
[[ "${SKIP_ADAM:-0}"          = "1" ]] || run_subproject "ADAM / DARPA'98"      adam           main.py removal

if [[ "${SKIP_AGGREGATE:-0}" != "1" ]]; then
  echo
  echo "================================================================"
  echo "[aggregate] cross-paper summary + figure"
  echo "================================================================"
  ( cd kitnet && uv run python "$HERE/cross_paper_summary.py" )
  ( cd kitnet && uv run python "$HERE/cross_paper_plot_data.py" )
  ( cd wang   && uv run python "$HERE/cross_paper_plot.py" )
fi

echo
echo "Done. Outputs:"
echo "  paper_reproductions/cross_paper_summary_long.csv"
echo "  paper_reproductions/cross_paper_summary_wide.csv"
echo "  paper_reproductions/cross_paper_plot_data.csv"
echo "  paper_reproductions/cross_paper_summary.{pdf,png}"
