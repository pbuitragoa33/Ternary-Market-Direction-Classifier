# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ternary Market Direction Classifier (TMDC) — a BTC directional prediction system that classifies each day (or 5-day window) into Bearish / Neutral / Bullish based on OHLCV data, technical indicators, macro indicators (FRED), and Hidden Markov Model regime features. The pipeline ends with walk-forward temporal validation and cost-aware financial backtesting.

Two parallel pipelines exist: **1-day prediction horizon (H=1)** and **5-day prediction horizon (H=5)**.

## Environment Setup

```bash
# Activate the virtual environment (Windows)
env_TMDC\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the configs package in editable mode (needed for imports from configs/)
pip install -e .
```

The project reads from a `.env` file in the root. Required variables:
- `S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — S3 data lake
- `FRED_API_KEY` — macroeconomic data from FRED
- `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_PROJECT_5D` — experiment tracking
- `EDA_CONTENT_PATH`, `FT_ENGINEERING_CONTENT_PATH`, `ARTIFACTS_LOCAL_PATH`, `TUNING_PATH`, `WALKFORWARD_PATH`, `STRATEGIES_PATH` — local output paths

## Running the Pipeline

Scripts must be executed in numbered order. Run from the repo root with the venv active.

### Stage 1 — Data Operations (Raw → Enriched)

**H=1 pipeline:**
```bash
python src/data-operations/01_data_supply.py
python src/data-operations/02_data_target_supply.py
python src/data-operations/03_derived_indicators.py
python src/data-operations/04_external_data_consolidation.py
python src/data-operations/05_target_var_treatment.py
python src/data-operations/06_financial_assets_processing.py
python src/data-operations/07_fred_data_completion.py
python src/data-operations/08_data_unification.py
python src/data-operations/09_feature_HMM_regime.py
```

**H=5 additions (run after 08):**
```bash
python src/data-operations/10_target_var_treatment_5d_pred.py
python src/data-operations/11_data_unification_5d_pred.py
python src/data-operations/12_feature_HMM_regime_5d_pred.py
```

Alternatively, trigger via Airflow DAG `tmdc_ml_preprocessing_dag`:
```bash
cd airflow && docker-compose up
```

### Stage 2 — Data Processing (Enriched → ML-ready splits)

```bash
python src/data-processing/processing_pipeline.py          # H=1
python src/data-processing/processing_pipeline_5d_pred.py  # H=5
```

Outputs: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` (Parquet/CSV on S3 or local), plus fitted scaler/encoder objects in `outputs/objects/`.

### Stage 3 — Hyperparameter Tuning

```bash
python src/tuning_training_evaluation/temporal_split_tuning.py          # H=1
python src/tuning_training_evaluation/temporal_split_tuning_5d_pred.py  # H=5
```

Uses Optuna to search across 9 model families (Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, Stacking). Results written to `outputs/tuning_folder/` and logged to W&B.

### Stage 4 — Walk-Forward Validation

```bash
python src/tuning_training_evaluation/walk_forward_train_eval.py          # H=1
python src/tuning_training_evaluation/walk_forward_train_eval_5d_pred.py  # H=5
```

Rolling window: 3-year train → 6-month test, slide by 6 months → 16 folds. Per-fold metrics (F1-macro, per-class precision/recall, AUC-ROC, AUC-PR, calibration) uploaded to W&B.

### Stage 5 — Backtesting

```bash
python src/backtesting/models_backtesting.py
```

Maps predictions to positions (Bullish → +1 long, Neutral → 0 flat, Bearish → -1 short), applies 5 bps transaction costs per side, and compares vs Buy & Hold.

## Architecture

```
Yahoo Finance / FRED API
        ↓
src/data-operations/  (01-09 for H=1, 10-12 extend to H=5)
        ↓  data_enriched.csv  (stored on S3)
src/data-processing/  (temporal split 80/10/10 by date, RobustScaler, encoding)
        ↓  X/y splits  (stored on S3)
src/tuning_training_evaluation/temporal_split_tuning.py  (Optuna + W&B)
        ↓  best hyperparameters JSON
src/tuning_training_evaluation/walk_forward_train_eval.py  (16 folds, W&B)
        ↓  OOS predictions per fold
src/backtesting/models_backtesting.py  (P&L, Sharpe vs Buy & Hold)
        ↓
outputs/  +  W&B dashboard  +  reports/
```

**configs/** is a Python package (installed via `pip install -e .`) that holds data-loading utilities reused across stages. `configs/load_enriched_data.py` handles H=1 and `configs/load_enriched_data_5d_pred.py` handles H=5.

## Key Design Decisions

- **Ternary target thresholds**: ±1% log-return for H=1; ±2% for H=5.
- **Temporal split**: train ends 2023-12-31, val ends 2024-12-31, test is everything after. No look-ahead leakage — scaler and encoder are fit only on train.
- **Class imbalance**: ~39.5% Neutral, ~33% Bullish, ~27.5% Bearish. Addressed via `class_weight='balanced'` during training (not resampling).
- **No test suite**: validation is done through walk-forward fold metrics and backtesting P&L, not unit tests.
- **Experiment tracking**: All tuning and walk-forward runs are logged to W&B. W&B project names are `TMDC_temporal_tuning` (H=1) and `TMDC_temporal_tuning_5d_pred` (H=5).

## Outputs Layout

```
outputs/
├── objects/          # Pickled scalers, encoders, best models
├── tuning_folder/    # Optuna study results, best params JSON
├── walkforward/      # Per-fold dashboards, confusion matrices, calibration curves
├── strategies/       # Backtesting strategy results
├── EDA_content/      # Figures from exploratory analysis
└── feature_engineering/  # Feature importance plots
```
