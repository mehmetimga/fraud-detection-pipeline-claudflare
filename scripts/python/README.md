# Credit Card Default Prediction

Predict whether a credit card customer will default on their next payment.

## Dataset

- **Source:** [Kaggle - Default of Credit Card Clients](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)
- **Size:** 30,000 samples, 24 features
- **Target:** `default.payment.next.month` (1 = default, 0 = no default)

## Results

| Metric | Score |
|--------|-------|
| F1 Score | 0.47 |
| ROC-AUC | 0.78 |

Best model: **Gradient Boosting** with tuned hyperparameters.

## Project Structure

```
├── data/                 # Raw data (not in git)
├── src/
│   ├── preprocess.py     # Data loading, feature engineering
│   ├── train.py          # Model training
│   └── evaluate.py       # Test evaluation
├── results/              # Processed data, model, results (not in git)
├── lab4.ipynb            # Full analysis notebook
├── Makefile              # Pipeline automation
└── requirements.txt      # Dependencies
```

## How to Run

### 1. Setup
```bash
conda activate fraud-ml
pip install -r requirements.txt
```

### 2. Download Data
Download from [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) and place in `data/` folder.

### 3. Run Pipeline
```bash
make all          # Run full pipeline
make preprocess   # Step 1: Process data
make train        # Step 2: Train model
make evaluate     # Step 3: Evaluate
make clean        # Remove results
```

## Key Findings

1. **Most important features:** Payment history (PAY_1, PAY_2, etc.)
2. **Class imbalance:** 78% no default, 22% default
3. **Best model:** Gradient Boosting beats Logistic Regression and Random Forest

## Author

Mehmet Imga
