# Mateen / CICIDS2017 Reproduction

Reproduces *Mateen: Adaptive Ensemble Learning for Network Anomaly
Detection* (Alotaibi & Maffeis, RAID 2024) on CICIDS2017, on Apple
Silicon (MPS).

The official author code lives at `../ref-src/Mateen/` (cloned from
github.com/ICL-ml4csec/Mateen). We import it directly and only patch
the device constant for MPS — the methodology is unmodified.

## Layout

- `PLAN.md`             — Phase 1 plan: paper extract + reproduction strategy.
- `device.py`           — MPS-aware device picker; patches the official
                          modules so they run on Apple Silicon.
- `data_loader.py`      — loads `clean_data.csv`, supports
                          fraction/seed subsampling for the ablation.
- `mateen_runner.py`    — wraps the official code; exposes
                          `run_no_update` and `run_mateen`; computes
                          BOTH paper-side (benign-positive) and
                          attack-side (TPR/FPR/attack-F1) metrics.
- `evaluate.py`         — Phase 3: single end-to-end run on the full
                          dataset.
- `dataset_lost.py`     — Phase 4: ablation across fractions {1, 5,
                          10, 25, 50, 100} % × multiple seeds.
- `audit_isolation.py`  — Phase 4 isolation audit: hashes, sizes,
                          positive counts.
- `main.py`             — CLI entry: `sanity | audit | phase3 | phase4`.

## Data

Mateen uses Engelen et al.'s revised CICIDS2017 (a single
`clean_data.csv` with first 693,702 rows = train, rest = test).

Downloaded fresh (this repro) from the authors' Google Drive
(referenced in `../ref-src/Mateen/README.md`):

```
data/CICIDS2017/clean_data.csv
```

Sanity criteria (verified by `python main.py sanity`):
- Train rows = 693,702
- Test rows  = 1,406,274
- Test windows at 50K = 29 (matches Table 1 of paper)
- Train benign-rate ~ 0.99 (initial training is benign-only by design)
- Test  benign-rate ~ 0.6455

## Run

```bash
uv run python main.py sanity
uv run python main.py audit
uv run python main.py phase3 --init-epochs 100
uv run python main.py phase4 --init-epochs 100 --seeds 0 1 2
```

## Metric note (important)

The paper's `getResult` swaps the confusion matrix so its reported
F1/Acc/Precision/Recall are **benign-as-positive**. On a benign-skewed
test set (~65% benign) those numbers can stay high even when attack
detection collapses.

We always report:
- `f1_paper`, `accuracy`, `macro_f1`, `auc_roc`  (paper-comparable)
- `attack_recall (TPR)`, `attack_precision`, `attack_f1`, `fpr`
  (the IDS-meaningful ones)

Phase 4 tables lead with the attack-side metrics. Per-trial
prediction distributions are printed so model collapse to one class is
visible in the raw log.
