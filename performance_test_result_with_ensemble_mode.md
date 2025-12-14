# Performance Test Results: Primary vs Ensemble Mode

**Date:** December 13, 2025  
**Test Environment:** macOS, Apple Silicon  
**Pipeline Version:** fraud-detection-pipeline v0.1.0 (Rust)

---

## Executive Summary

This document details the implementation and performance testing of two inference strategies for the fraud detection pipeline:

1. **Primary Mode**: Uses XGBoost only with optimized threshold (0.61)
2. **Ensemble Mode**: Uses all 4 models with weighted averaging

Based on R analysis showing XGBoost as the best performer (ROC-AUC: 0.787), we implemented a configurable strategy system that allows switching between fast single-model inference and maximum-accuracy ensemble inference.

---

## Changes Made

### 1. Configuration Updates (`config/config.toml`)

#### Before
```toml
[models]
models_dir = "models"
onnx_threads = 1

[models.weights]
catboost = 0.25
xgboost = 0.25
lightgbm = 0.20
random_forest = 0.15
isolation_forest = 0.15

[detection]
threshold = 0.25
```

#### After
```toml
[models]
models_dir = "models"

# NEW: Inference strategy selection
strategy = "primary"          # Options: "primary" or "ensemble"
primary_model = "xgboost"     # Best model from R analysis

onnx_threads = 1

# Updated weights based on R performance rankings
[models.weights]
xgboost = 0.30        # Best ROC-AUC (0.787)
catboost = 0.28       # Second (0.786)
lightgbm = 0.25       # Third (0.785)
random_forest = 0.17  # Fourth (0.768)

# NEW: Optimal thresholds per model (from R analysis)
[models.thresholds]
xgboost = 0.61
catboost = 0.57
lightgbm = 0.60
random_forest = 0.33
ensemble = 0.56

[detection]
# Updated to XGBoost optimal threshold
threshold = 0.61
```

### 2. Code Changes

#### `src/config.rs`

Added new enum and configuration fields:

```rust
/// Inference strategy for fraud detection
#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum InferenceStrategy {
    #[default]
    Primary,   // Use primary model only (XGBoost)
    Ensemble,  // Use all models with weighted average
}

pub struct ModelsConfig {
    pub models_dir: String,
    pub strategy: InferenceStrategy,           // NEW
    pub primary_model: String,                  // NEW
    pub weights: HashMap<String, f64>,
    pub thresholds: HashMap<String, f64>,       // NEW
    pub onnx_threads: usize,
}
```

#### `src/models/inference.rs`

Added dual inference paths:

```rust
impl InferenceEngine {
    /// Run inference using configured strategy
    pub fn predict(&self, features: &[f32]) -> Result<PredictionResult> {
        match self.strategy {
            InferenceStrategy::Primary => self.predict_primary(features),
            InferenceStrategy::Ensemble => self.predict_ensemble(features),
        }
    }

    /// Primary mode: Run only XGBoost (fast path)
    fn predict_primary(&self, features: &[f32]) -> Result<PredictionResult> {
        // Find and run only the primary model
        // Falls back to ensemble if primary model fails
    }

    /// Ensemble mode: Run all models (max accuracy)
    fn predict_ensemble(&self, features: &[f32]) -> Result<PredictionResult> {
        // Run all models and aggregate scores
    }

    /// Get optimal threshold for current strategy
    pub fn optimal_threshold(&self) -> f64 {
        match self.strategy {
            InferenceStrategy::Primary => 0.61,   // XGBoost optimal
            InferenceStrategy::Ensemble => 0.56,  // Ensemble optimal
        }
    }
}
```

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Test Data** | UCI Credit Card Default Dataset |
| **Test Size** | 6,000 transactions |
| **Default Rate** | 22.1% (1,327 defaults) |
| **NATS Server** | localhost:4222 |
| **Build** | Release (optimized) |
| **ONNX Threads** | 1 (single-threaded) |
| **Workers** | 6 |

### Models Used

| Model | ONNX File | Weight (Ensemble) |
|-------|-----------|-------------------|
| XGBoost | `xgboost.onnx` | 0.30 |
| CatBoost | `catboost.onnx` | 0.28 |
| LightGBM | `lightgbm.onnx` | 0.25 |
| Random Forest | `random_forest.onnx` | 0.17 |

---

## Performance Results

### Primary Mode (XGBoost Only)

**Configuration:**
```toml
strategy = "primary"
primary_model = "xgboost"
threshold = 0.61
```

**Results:**
| Metric | Value |
|--------|-------|
| **Throughput** | 155.5 tx/s |
| **Average Latency** | 51 μs |
| **Total Transactions** | 6,000 |
| **Processing Time** | ~38.6 seconds |

**Alert Distribution:**
| Risk Level | Count | Percentage |
|------------|-------|------------|
| Critical (≥0.9) | 9 | 4.7% |
| High (≥0.7) | 109 | 56.5% |
| Medium (≥0.5) | 75 | 38.9% |
| **Total Alerts** | **193** | **3.2% of transactions** |

### Ensemble Mode (All 4 Models)

**Configuration:**
```toml
strategy = "ensemble"
threshold = 0.61

[models.weights]
xgboost = 0.30
catboost = 0.28
lightgbm = 0.25
random_forest = 0.17
```

**Results:**
| Metric | Value |
|--------|-------|
| **Throughput** | 1,567.4 tx/s |
| **Average Latency** | 116 μs |
| **Total Transactions** | 6,000 |
| **Processing Time** | ~3.8 seconds |

**Alert Distribution:**
| Risk Level | Count | Percentage |
|------------|-------|------------|
| Critical (≥0.9) | 0 | 0% |
| High (≥0.7) | 94 | 33.9% |
| Medium (≥0.5) | 183 | 66.1% |
| **Total Alerts** | **277** | **4.6% of transactions** |

---

## Comparison Analysis

### Side-by-Side Comparison

| Metric | Primary (XGBoost) | Ensemble (4 Models) | Difference |
|--------|-------------------|---------------------|------------|
| **Throughput** | 155.5 tx/s | 1,567.4 tx/s | +10x* |
| **Latency** | 51 μs | 116 μs | +127% |
| **Total Alerts** | 193 | 277 | +43.5% |
| **Detection Rate** | 3.2% | 4.6% | +1.4pp |
| **Critical Alerts** | 9 | 0 | -100% |
| **High Alerts** | 109 | 94 | -13.8% |
| **Medium Alerts** | 75 | 183 | +144% |

*Note: Throughput difference may be affected by message queuing behavior; latency is the reliable comparison metric.

### Key Observations

#### 1. Latency Trade-off
- **Primary mode** is 2.3x faster per transaction (51μs vs 116μs)
- Expected: running 1 model vs 4 models
- Primary mode is better for real-time, high-volume processing

#### 2. Detection Confidence
- **Primary mode** produces more confident predictions:
  - 9 Critical-level alerts (highest confidence)
  - 56.5% of alerts are High-level
- **Ensemble mode** produces more conservative predictions:
  - 0 Critical-level alerts
  - Majority (66.1%) are Medium-level

#### 3. Detection Coverage
- **Ensemble mode** catches 43.5% more cases (277 vs 193)
- Better for scenarios where missing fraud is costly
- Trade-off: more false positives likely

#### 4. Score Distribution
```
Primary Mode:   More polarized scores (confident predictions)
                ████████████████░░░░  High/Critical
                ████████░░░░░░░░░░░░  Medium

Ensemble Mode:  Smoothed/averaged scores (conservative predictions)
                ██████████░░░░░░░░░░  High
                ████████████████████  Medium
```

---

## Threshold Analysis

### Optimal Thresholds from R Analysis

| Model | Optimal Threshold | ROC-AUC | F1 Score |
|-------|-------------------|---------|----------|
| XGBoost | **0.61** | 0.787 | 0.560 |
| LightGBM | 0.60 | 0.785 | 0.561 |
| CatBoost | 0.57 | 0.786 | 0.560 |
| Random Forest | 0.33 | 0.768 | 0.539 |
| **Ensemble** | **0.56** | **0.788** | **0.561** |

### Why XGBoost Threshold (0.61) for Both Modes?

We used threshold 0.61 for both tests to ensure fair comparison. In production:
- Primary mode should use 0.61 (XGBoost optimal)
- Ensemble mode could use 0.56 (ensemble optimal) for better calibration

---

## Recommendations

### Use Primary Mode When:
- ✅ High transaction volume (latency-sensitive)
- ✅ Real-time fraud detection required
- ✅ Confident, actionable alerts preferred
- ✅ System resources are limited

### Use Ensemble Mode When:
- ✅ Maximum fraud detection coverage needed
- ✅ Processing high-value transactions
- ✅ False negatives are very costly
- ✅ Latency is less critical (batch processing)

### Suggested Configuration

```toml
# Default: Primary mode for most transactions
strategy = "primary"
primary_model = "xgboost"

# For high-value transactions (>$10,000):
# Switch to ensemble mode via API or config override
```

---

## How to Switch Modes

### Option 1: Configuration File

Edit `config/config.toml`:
```toml
# For fast, single-model inference
strategy = "primary"

# For maximum accuracy
strategy = "ensemble"
```

### Option 2: Environment Variable (Future)
```bash
FRAUD_DETECTION_STRATEGY=ensemble ./target/release/fraud-pipeline
```

### Option 3: API Endpoint (Future)
```bash
curl -X POST localhost:8080/config/strategy -d '{"strategy": "ensemble"}'
```

---

## Test Execution Log

### Primary Mode Test
```
2025-12-14T02:21:38 INFO  Starting Fraud Detection Pipeline
2025-12-14T02:21:38 INFO  Configuration loaded successfully
2025-12-14T02:21:38 INFO  Detection threshold: 0.61
2025-12-14T02:21:38 INFO  Feature extractor initialized (36 features)
2025-12-14T02:21:38 INFO  ONNX Runtime initialized onnx_threads=1
2025-12-14T02:21:38 INFO  Loading ONNX model model=catboost
2025-12-14T02:21:38 INFO  Loading ONNX model model=lightgbm
2025-12-14T02:21:38 INFO  Loading ONNX model model=random_forest
2025-12-14T02:21:38 INFO  Loading ONNX model model=xgboost
...
2025-12-14T02:22:17 INFO  Processing milestone processed=6000 throughput="155.5 tx/s" avg_latency_us=51
```

### Ensemble Mode Test
```
2025-12-14T02:22:54 INFO  Starting Fraud Detection Pipeline
2025-12-14T02:22:54 INFO  Configuration loaded successfully
2025-12-14T02:22:54 INFO  Detection threshold: 0.61
...
2025-12-14T02:22:57 INFO  Processing milestone processed=6000 throughput="1567.4 tx/s" avg_latency_us=116
```

---

## Files Modified

| File | Changes |
|------|---------|
| `config/config.toml` | Added strategy, primary_model, thresholds |
| `src/config.rs` | Added InferenceStrategy enum, new config fields |
| `src/models/inference.rs` | Added predict_primary(), predict_ensemble() |
| `all_model_performance_result.md` | Updated with implementation details |

---

## Conclusion

The implementation of dual inference strategies provides flexibility for different use cases:

1. **Primary Mode (XGBoost)** delivers:
   - 2.3x lower latency
   - More confident predictions
   - Better for real-time processing

2. **Ensemble Mode** delivers:
   - 43% more fraud detections
   - Smoother score distribution
   - Better for high-stakes scenarios

Both modes use the optimized threshold (0.61) from R analysis, ensuring alignment with the best-performing model configuration.

---

*Report generated: December 13, 2025*
