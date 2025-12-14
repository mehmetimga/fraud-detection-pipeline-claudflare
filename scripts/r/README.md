# Credit Card Default Prediction - R Pipeline

R implementation of the fraud detection model training pipeline. Uses the same 4 models as the Rust inference pipeline.

## Models

| Model | R Package | Description |
|-------|-----------|-------------|
| XGBoost | `xgboost` | Gradient boosting with regularization |
| LightGBM | `lightgbm` | Fast gradient boosting |
| CatBoost | `catboost` | Gradient boosting with categorical support |
| Random Forest | `randomForest` | Ensemble of decision trees |

## Quick Start

```bash
# Install R packages
make install

# Run full pipeline
make all
```

## Structure

```
scripts/r/
├── Makefile              # Build commands
├── README.md             # This file
├── install_packages.R    # Package installation script
├── data/
│   └── UCI_Credit_Card.csv   # Dataset
├── src/
│   ├── preprocess.R      # Data preprocessing
│   ├── train.R           # Model training
│   └── evaluate.R        # Model evaluation
└── results/              # Output (generated)
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_test.csv
    ├── models.rds
    ├── features.rds
    ├── model_comparison.csv
    └── optimal_thresholds.csv
```

## Pipeline Steps

### 1. Preprocess (`make preprocess`)

- Load UCI Credit Card dataset
- Clean and encode categorical variables
- Engineer features:
  - `times_late`: Number of late payments
  - `max_months_late`: Maximum months delayed
  - `avg_bill`: Average bill amount
  - `avg_payment`: Average payment amount
  - `payment_ratio`: Payment to bill ratio
  - `credit_util`: Credit utilization ratio
- Split data (80/20 stratified)
- Save processed data

### 2. Train (`make train`)

- Train 4 models with class balancing:
  - **XGBoost**: `scale_pos_weight` for imbalance
  - **LightGBM**: `is_unbalance = TRUE`
  - **CatBoost**: `auto_class_weights = "Balanced"`
  - **Random Forest**: `classwt` parameter
- Early stopping on validation set
- Save models to `results/models.rds`

### 3. Evaluate (`make evaluate`)

- Load test data and trained models
- Get probability predictions
- Find optimal threshold for each model
- Calculate metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Calculate ensemble prediction (average)
- Save results

## Requirements

- R >= 4.0
- Conda (recommended for CatBoost)
- Packages:
  - `tidyverse`
  - `caret`
  - `xgboost`
  - `lightgbm`
  - `catboost`
  - `randomForest`
  - `pROC`
  - `MLmetrics`

## Install Packages

### Option 1: Conda (Recommended)

```bash
# Install all packages including CatBoost
make install-conda

# Or manually:
conda install -y -c conda-forge r-tidyverse r-caret r-xgboost r-lightgbm r-catboost r-randomforest r-proc
```

### Option 2: CRAN (CatBoost will be skipped)

```r
# Run in R console
source("install_packages.R")

# Or via command line
make install
```

**Note:** CatBoost for R is only available via conda. If using CRAN packages, CatBoost will be skipped.

## Output

### Model Comparison (results/model_comparison.csv)

| model | accuracy | precision | recall | f1 | roc_auc |
|-------|----------|-----------|--------|-----|---------|
| xgboost | 0.78 | 0.51 | 0.59 | 0.55 | 0.78 |
| lightgbm | 0.78 | 0.51 | 0.58 | 0.54 | 0.78 |
| catboost | 0.78 | 0.51 | 0.59 | 0.55 | 0.78 |
| random_forest | 0.78 | 0.50 | 0.59 | 0.54 | 0.78 |
| ensemble | 0.79 | 0.52 | 0.58 | 0.55 | 0.79 |

### Optimal Thresholds (results/optimal_thresholds.csv)

| model | optimal_threshold |
|-------|-------------------|
| xgboost | 0.54 |
| lightgbm | 0.51 |
| catboost | 0.57 |
| random_forest | 0.48 |
| ensemble | 0.52 |

## Comparison with Python Pipeline

This R pipeline produces similar results to the Python pipeline:

| Metric | Python (sklearn) | R (4 models) |
|--------|------------------|--------------|
| Best F1 | 0.47 | 0.55 |
| Best ROC-AUC | 0.78 | 0.79 |

The R pipeline uses the same 4 models as the Rust inference pipeline, making it easier to compare training and inference performance.

