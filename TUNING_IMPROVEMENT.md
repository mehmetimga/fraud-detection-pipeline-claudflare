# Model Tuning & Improvement Results

This document details the hyperparameter tuning and threshold optimization applied to improve the fraud detection models.

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Best F1 Score | 0.543 | **0.547** | +0.7% |
| Best ROC-AUC | 0.781 | **0.782** | +0.1% |
| XGBoost F1 | 0.526 | **0.547** | **+4.0%** |

## Improvements Applied

### 1. Hyperparameter Tuning

Added `RandomizedSearchCV` to find optimal parameters for each model.

**Search spaces:**

```python
# CatBoost
CATBOOST_PARAM_DIST = {
    'iterations': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 5, 6, 7, 8],
    'l2_leaf_reg': [1, 3, 5, 7],
}

# XGBoost
XGBOOST_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

# LightGBM
LIGHTGBM_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.7, 0.8, 0.9, 1.0],
}

# Random Forest
RANDOM_FOREST_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 8, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}
```

### 2. Class Weights (Already Implemented)

The original models already had class balancing:

| Model | Method |
|-------|--------|
| CatBoost | `auto_class_weights='Balanced'` |
| XGBoost | `scale_pos_weight = n_neg / n_pos` |
| LightGBM | `is_unbalance=True` |
| Random Forest | `class_weight='balanced'` |

### 3. Threshold Optimization

Instead of using the default 0.5 threshold, we find the optimal threshold for each model by maximizing F1 score on the validation set.

```python
def find_optimal_threshold(y_true, y_proba, metric='f1'):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]
```

## Results Comparison

### Original Models (Default Parameters)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CatBoost | 75.5% | 0.460 | 62.7% | 0.531 | 0.781 |
| LightGBM | 75.9% | 0.466 | 62.5% | 0.534 | 0.780 |
| Random Forest | 78.7% | 0.517 | 57.1% | 0.543 | 0.778 |
| XGBoost | 77.1% | 0.485 | 57.4% | 0.526 | 0.766 |

### Tuned Models (Hyperparameter Tuning + Threshold Optimization)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Optimal Threshold |
|-------|----------|-----------|--------|----------|---------|-------------------|
| CatBoost | 78.5% | 0.512 | 58.6% | **0.546** | 0.782 | 0.566 |
| XGBoost | 78.4% | 0.510 | 59.0% | **0.547** | 0.782 | 0.540 |
| LightGBM | 78.5% | 0.512 | 58.0% | 0.544 | 0.778 | 0.508 |
| Random Forest | 78.0% | 0.502 | 59.1% | 0.543 | 0.779 | 0.483 |

### Improvement by Model

| Model | Original F1 | Tuned F1 | Improvement |
|-------|-------------|----------|-------------|
| **XGBoost** | 0.526 | 0.547 | **+4.0%** |
| **CatBoost** | 0.531 | 0.546 | **+2.8%** |
| LightGBM | 0.534 | 0.544 | +1.9% |
| Random Forest | 0.543 | 0.543 | +0.0% |

## Best Hyperparameters Found

### CatBoost
```python
{
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 4,
    'l2_leaf_reg': 5
}
# CV F1: 0.5399
```

### XGBoost
```python
{
    'n_estimators': 300,
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.8
}
# CV F1: 0.5434
```

### LightGBM
```python
{
    'n_estimators': 200,
    'learning_rate': 0.01,
    'max_depth': 6,
    'num_leaves': 127,
    'min_child_samples': 20,
    'subsample': 0.7
}
# CV F1: 0.5452
```

### Random Forest
```python
{
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}
# CV F1: 0.5428
```

## Optimal Thresholds

The default classification threshold of 0.5 is not always optimal. Our optimization found:

| Model | Default (0.5) F1 | Optimal Threshold | Optimized F1 |
|-------|------------------|-------------------|--------------|
| CatBoost | 0.531 | 0.566 | 0.546 |
| XGBoost | 0.530 | 0.540 | 0.547 |
| LightGBM | 0.540 | 0.508 | 0.544 |
| Random Forest | 0.539 | 0.483 | 0.543 |

## Trade-offs Observed

### Precision vs Recall

| Configuration | Precision | Recall | F1 |
|---------------|-----------|--------|-----|
| High threshold (0.6) | Higher | Lower | Similar |
| Default (0.5) | Medium | Medium | Baseline |
| Low threshold (0.4) | Lower | Higher | Similar |

The optimal threshold balances precision and recall for maximum F1.

### Accuracy vs Fraud Detection

| Model | Accuracy | Fraud Detection Rate |
|-------|----------|---------------------|
| Original CatBoost | 75.5% | **62.7%** |
| Tuned CatBoost | 78.5% | 58.6% |

Higher accuracy sometimes means catching fewer fraud cases. We optimized for F1 which balances both.

## How to Run Tuned Training

```bash
# Quick tuning (10 iterations per model, ~5 minutes)
make train-tuned

# Full tuning (50 iterations per model, ~20 minutes)
make train-tuned-full

# Or directly:
cd training
python train_tuned.py --n-iter 20 --cv 3
```

## Files Added

| File | Description |
|------|-------------|
| `training/src/tuning.py` | Hyperparameter tuning functions |
| `training/train_tuned.py` | Enhanced training script |
| `models/tuned_config.json` | Best parameters and thresholds |
| `models/model_comparison_tuned.csv` | Tuned model results |

## Conclusions

1. **XGBoost benefited most** from tuning (+4.0% F1)
2. **Threshold optimization** provides easy wins without retraining
3. **Class weights were already implemented** - good baseline
4. **Random Forest was already near-optimal** - robust defaults
5. **Lower learning rates** (0.01-0.05) worked better than defaults (0.1)

## Future Improvements

1. **Bayesian Optimization**: More efficient than random search
2. **Feature Selection**: Remove low-importance features
3. **Ensemble Stacking**: Combine model predictions
4. **SMOTE/ADASYN**: Synthetic oversampling for minority class
5. **Cost-Sensitive Learning**: Weight fraud cases higher

