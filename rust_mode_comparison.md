# Rust Pipeline Mode Comparison

**Date:** $(date)

## Test Configuration

- **Transactions Sent:** 500 per mode
- **Dataset:** Credit Card Default (23.8% default rate in test set)

---

## 1. XGBoost Primary Mode

| Setting | Value |
|---------|-------|
| Strategy | primary |
| Model | xgboost |
| Threshold | 0.61 |

**Performance:**
- Throughput: ~194 tx/s (at 500 processed)
- Avg Latency: 98μs

---

## 2. CatBoost Primary Mode

| Setting | Value |
|---------|-------|
| Strategy | primary |
| Model | catboost |
| Threshold | 0.57 |

**Performance:**
- Throughput: ~213 tx/s (at 500 processed)
- Avg Latency: 173μs

---

## 3. Ensemble Mode

| Setting | Value |
|---------|-------|
| Strategy | ensemble |
| Models | All 4 (weighted average) |
| Threshold | 0.56 |

**Performance:**
- Throughput: ~209 tx/s (at 500 processed)
- Avg Latency: 218μs

---

## Performance Comparison

| Mode | Throughput | Latency | Notes |
|------|------------|---------|-------|
| **XGBoost Primary** | 194 tx/s | 98μs | Fastest latency |
| **CatBoost Primary** | 213 tx/s | 173μs | Highest throughput |
| **Ensemble** | 209 tx/s | 218μs | Maximum accuracy |

---

## Key Observations

1. **CatBoost Primary** has highest throughput (~213 tx/s)
2. **XGBoost Primary** has lowest latency (98μs)
3. **Ensemble** mode is ~2.2x slower but uses all models for maximum accuracy
4. All modes successfully processed 500 transactions

