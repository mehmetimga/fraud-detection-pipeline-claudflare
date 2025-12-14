# Python Model Training Pipeline

Credit card default prediction using the same models as the Rust fraud detection pipeline.

## Models

This pipeline trains the same 4 models used in the Rust project:

| Model | Description |
|-------|-------------|
| **XGBoost** | Gradient boosting with regularization |
| **CatBoost** | Gradient boosting with categorical feature handling |
| **LightGBM** | Light gradient boosting machine |
| **Random Forest** | Ensemble of decision trees |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
make all

# Or run steps individually
make preprocess  # Preprocess data and engineer features
make train       # Train all 4 models
make evaluate    # Evaluate models with ensemble comparison
```

## Data

Place `UCI_Credit_Card.csv` in the `data/` folder. Download from:
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

## Pipeline Steps

### 1. Preprocessing (`src/preprocess.py`)

- Loads UCI Credit Card dataset
- Engineers features matching Rust project:
  - Payment behavior: `avg_pay_delay`, `max_pay_delay`, `months_delayed`, `pay_trend`
  - Bill features: `avg_bill_amt`, `bill_trend`, `bill_volatility`
  - Payment features: `avg_pay_amt`, `total_pay_amt`
  - Utilization: `utilization_ratio`, `over_limit`, `pay_bill_ratio`
  - Demographics: `age_group`
- Splits into 80/20 train/test

### 2. Training (`src/train.py`)

- Trains all 4 models with optimized hyperparameters
- Uses early stopping for gradient boosting models
- Handles class imbalance with:
  - XGBoost: `scale_pos_weight`
  - CatBoost: `auto_class_weights='Balanced'`
  - LightGBM: `is_unbalance=True`
  - Random Forest: `class_weight='balanced'`

### 3. Evaluation (`src/evaluate.py`)

- Evaluates all models on test set
- Creates weighted ensemble predictions
- Uses optimal thresholds from R analysis:
  - XGBoost: 0.61
  - CatBoost: 0.57
  - LightGBM: 0.60
  - Random Forest: 0.33
  - Ensemble: 0.56
- Compares XGBoost vs CatBoost
- Generates comparison reports

## Output Files

| File | Description |
|------|-------------|
| `results/X_train.csv` | Preprocessed training features |
| `results/X_test.csv` | Preprocessed test features |
| `results/y_train.csv` | Training labels |
| `results/y_test.csv` | Test labels |
| `results/xgboost_model.pkl` | Trained XGBoost model |
| `results/catboost_model.pkl` | Trained CatBoost model |
| `results/lightgbm_model.pkl` | Trained LightGBM model |
| `results/random_forest_model.pkl` | Trained Random Forest model |
| `results/model_comparison.csv` | Model performance comparison |
| `results/optimal_thresholds.csv` | Optimal classification thresholds |
| `results/test_predictions.csv` | All model predictions on test set |

## Ensemble Weights

Based on R analysis performance:

```python
MODEL_WEIGHTS = {
    'xgboost': 0.30,      # Best ROC-AUC
    'catboost': 0.28,     # Second best
    'lightgbm': 0.25,     # Third
    'random_forest': 0.17  # Fourth
}
```

## Expected Results

| Model | ROC-AUC | F1 Score |
|-------|---------|----------|
| XGBoost | ~0.787 | ~0.560 |
| CatBoost | ~0.786 | ~0.560 |
| LightGBM | ~0.785 | ~0.561 |
| Random Forest | ~0.768 | ~0.539 |
| **Ensemble** | **~0.788** | **~0.561** |

## Compatibility with Rust Project

These models can be exported to ONNX format for use with the Rust fraud detection pipeline. See the `training/` directory for ONNX export functionality.
