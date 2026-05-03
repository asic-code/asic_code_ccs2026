#!/usr/bin/env bash
# Repo-root bootstrap: CCS26 dataset prep, .venv + requirements.txt, rule_eval_c in-place build.
#
# Env (optional): SKIP_DATASETS SKIP_VENV SKIP_RULE_EVAL_C PREPARE_ARGS PYTHON
# Mirai: copies mirai_train.csv -> mirai_train_sample.csv when the sample file is missing.
# Windows: rule_eval_c may need MSVC; failed build logs a warning only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
SKIP_DATASETS="${SKIP_DATASETS:-0}"
SKIP_VENV="${SKIP_VENV:-0}"
SKIP_RULE_EVAL_C="${SKIP_RULE_EVAL_C:-0}"
PREPARE_ARGS="${PREPARE_ARGS:-netml mirai}"

ARTIFACT_ROOT="$ROOT/ccs26_dataset_artifact_files"
PREPARE_SH="$ARTIFACT_ROOT/scripts/prepare_ccs26_datasets.sh"
REQ_FILE="$ROOT/requirements.txt"

log() { printf '[bootstrap] %s\n' "$*" >&2; }
warn() { printf '[bootstrap][WARN] %s\n' "$*" >&2; }

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Missing required command: $1"
    exit 1
  fi
}

# Datasets
if [[ "$SKIP_DATASETS" != "1" ]]; then
  need_cmd bash
  if [[ ! -f "$PREPARE_SH" ]]; then
    log "Dataset prepare script not found: $PREPARE_SH"
    exit 1
  fi
  log "Preparing datasets (artifact_data -> ../Dataset/load_dataset) ..."
  log "  ARTIFACT_DATA_ROOT=$ARTIFACT_ROOT/artifact_data"
  log "  args: $PREPARE_ARGS"
  # shellcheck disable=SC2086
  bash "$PREPARE_SH" \
    --artifact-data-root "$ARTIFACT_ROOT/artifact_data" \
    $PREPARE_ARGS

  MIRAI_DIR="$ROOT/../Dataset/load_dataset/MiraiBotnet"
  if [[ -f "$MIRAI_DIR/mirai_train.csv" && ! -f "$MIRAI_DIR/mirai_train_sample.csv" ]]; then
    log "Creating mirai_train_sample.csv from mirai_train.csv (loader default path)."
    cp -f "$MIRAI_DIR/mirai_train.csv" "$MIRAI_DIR/mirai_train_sample.csv"
  elif [[ ! -f "$MIRAI_DIR/mirai_train_sample.csv" ]]; then
    warn "Neither mirai_train.csv nor mirai_train_sample.csv found under $MIRAI_DIR"
  fi

  CHECK_HDR="$ARTIFACT_ROOT/scripts/check_loader_csv_headers.py"
  if [[ -f "$CHECK_HDR" ]]; then
    need_cmd "$PYTHON"
    log "Optional: validating loader CSV headers..."
    (cd "$ARTIFACT_ROOT" && "$PYTHON" scripts/check_loader_csv_headers.py --root ../Dataset/load_dataset) || warn "Header check failed (non-fatal)."
  fi
else
  log "Skipping dataset preparation (SKIP_DATASETS=1)."
fi

# venv
if [[ "$SKIP_VENV" != "1" ]]; then
  need_cmd "$PYTHON"
  if [[ ! -f "$REQ_FILE" ]]; then
    log "requirements.txt not found: $REQ_FILE"
    exit 1
  fi
  VENV_DIR="$ROOT/.venv"
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv: $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
  else
    log "Using existing venv: $VENV_DIR"
  fi
  PY="$VENV_DIR/bin/python"
  PIP="$VENV_DIR/bin/pip"
  if [[ ! -f "$PY" ]]; then
    PY="$VENV_DIR/Scripts/python.exe"
    PIP="$VENV_DIR/Scripts/pip.exe"
  fi
  if [[ ! -f "$PY" ]]; then
    log "Could not find venv python under $VENV_DIR"
    exit 1
  fi
  log "Upgrading pip and installing requirements.txt ..."
  "$PY" -m pip install --upgrade pip
  "$PIP" install -r "$REQ_FILE"
  export PATH="$VENV_DIR/bin:$VENV_DIR/Scripts:${PATH:-}"
else
  log "Skipping venv (SKIP_VENV=1)."
  PY="${PY:-$PYTHON}"
fi

if [[ -f "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -f "$ROOT/.venv/Scripts/python.exe" ]]; then
  PY="$ROOT/.venv/Scripts/python.exe"
else
  PY="${PY:-$PYTHON}"
fi

# rule_eval_c
if [[ "$SKIP_RULE_EVAL_C" != "1" ]]; then
  RULE_DIR="$ROOT/rule_eval_c"
  if [[ ! -f "$RULE_DIR/setup.py" ]]; then
    warn "rule_eval_c/setup.py missing; skip C build."
  else
    log "Building rule_eval_c (in-place) with: $PY"
    (cd "$RULE_DIR" && "$PY" setup.py build_ext --inplace) || warn "rule_eval_c build failed (install MSVC Build Tools on Windows, or set SKIP_RULE_EVAL_C=1)."
  fi
else
  log "Skipping rule_eval_c (SKIP_RULE_EVAL_C=1)."
fi

log "Done."
log "  Activate venv: source .venv/bin/activate   (Windows Git Bash: source .venv/Scripts/activate)"
log "  Or run tools with: $PY <script.py>"
