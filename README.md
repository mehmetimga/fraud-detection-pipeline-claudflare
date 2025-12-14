# Fraud Detection Pipeline

A high-performance, real-time payment fraud detection pipeline built in Rust, inspired by Cloudflare's BLISS architecture. Uses NATS as the message broker and ensemble ML models for fraud scoring.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────────────────┐
│ Payment Gateway │────▶│  NATS Broker     │────▶│     Rust Fraud Pipeline             │
│ Test Producer   │     │  (transactions)  │     │  ┌─────────────────────────────┐    │
└─────────────────┘     └──────────────────┘     │  │ Feature Extractor           │    │
                                                 │  └─────────────────────────────┘    │
                                                 │              ▼                      │
                                                 │  ┌─────────────────────────────┐    │
                                                 │  │ Multi-Model Inference       │    │
                                                 │  │ (CatBoost, XGBoost,         │    │
                                                 │  │  LightGBM, RF, IsoForest)   │    │
                                                 │  └─────────────────────────────┘    │
                                                 │              ▼                      │
                                                 │  ┌─────────────────────────────┐    │
                                                 │  │ Score Aggregator            │    │
                                                 │  └─────────────────────────────┘    │
                                                 │              ▼                      │
                                                 │  ┌─────────────────────────────┐    │
                                                 │  │ Alert Generator             │    │
                                                 └──┴─────────────────────────────┴────┘
                                                                ▼
                                                 ┌──────────────────┐
                                                 │  NATS Broker     │
                                                 │  (fraud.alerts)  │
                                                 └──────────────────┘
```

## ML Models

| Model | Type | Strength |
|-------|------|----------|
| CatBoost | Gradient Boosting | Handles categorical features natively |
| XGBoost | Gradient Boosting | Industry standard, highly optimized |
| LightGBM | Gradient Boosting | Fast training, memory efficient |
| Random Forest | Ensemble | Robust baseline, interpretable |
| Isolation Forest | Anomaly Detection | Unsupervised, catches novel fraud patterns |

## Project Structure

```
fraud-detection-pipeline/
├── Cargo.toml              # Rust dependencies
├── config/
│   └── config.toml         # Pipeline configuration
├── models/                 # Trained ONNX models
├── src/                    # Rust source code
│   ├── main.rs
│   ├── lib.rs
│   ├── config.rs
│   ├── consumer.rs
│   ├── producer.rs
│   ├── feature_extractor.rs
│   ├── models/
│   └── types/
├── training/               # Python training pipeline
│   ├── requirements.txt
│   ├── train.py
│   └── src/
└── tools/
    └── test_producer.rs
```

## Quick Start

### Prerequisites

- Rust 1.70+
- Python 3.9+
- NATS Server
- ONNX Runtime

### 1. Train the Models (Python)

```bash
cd training
pip install -r requirements.txt
python train.py
```

This will train all 5 models and export them to ONNX format in the `models/` directory.

### 2. Start NATS Server

```bash
# Using Docker
docker run -p 4222:4222 nats:latest

# Or install locally: https://docs.nats.io/running-a-nats-service/introduction/installation
```

### 3. Run the Fraud Pipeline

```bash
cargo run --release --bin fraud-pipeline
```

### 4. Send Test Transactions

```bash
cargo run --release --bin test-producer
```

## Configuration

Edit `config/config.toml` to customize:

- NATS connection settings
- Model weights for ensemble scoring
- Risk thresholds
- Logging settings

## Performance

### Throughput Benchmarks

Tested on MacBook Pro (10-core CPU):

| Workers | Throughput | Latency | Daily Capacity |
|---------|-----------|---------|----------------|
| 1 (sequential) | 190 tx/s | 25 μs | 16.4 million |
| 4 (parallel) | 891 tx/s | 89 μs | 77 million |
| **6 (optimal)** | **1,069 tx/s** | 121 μs | **92.4 million** |
| 10 | 1,020 tx/s | 241 μs | 88 million |

**Key metrics:**
- Processing time per transaction: **121 microseconds**
- Daily capacity: **92 million transactions** (single machine)
- Parallel processing with configurable workers

### Real-World Comparison

| Company | Daily Transactions | Our Pipeline |
|---------|-------------------|--------------|
| Stripe | ~50 million | ✅ 1.8x capacity |
| PayPal | ~40 million | ✅ 2.3x capacity |
| Square | ~20 million | ✅ 4.6x capacity |

### Architecture Highlights

Inspired by Cloudflare's BLISS architecture:
- Zero-copy message processing where possible
- Pre-allocated buffers for feature vectors
- ONNX Runtime for efficient model inference
- Async I/O with Tokio
- Parallel transaction processing

## Model Performance

### Original vs Tuned Models

We implemented hyperparameter tuning and threshold optimization to improve model accuracy.

**Original Models (default parameters):**

| Model | Accuracy | F1 Score | ROC-AUC | Fraud Detection |
|-------|----------|----------|---------|-----------------|
| CatBoost | 75.5% | 0.531 | 0.781 | 62.7% |
| XGBoost | 77.1% | 0.526 | 0.766 | 57.4% |
| LightGBM | 75.9% | 0.534 | 0.780 | 62.5% |
| Random Forest | 78.7% | 0.543 | 0.778 | 57.1% |

**Tuned Models (with hyperparameter tuning + threshold optimization):**

| Model | Accuracy | F1 Score | ROC-AUC | Fraud Detection | Improvement |
|-------|----------|----------|---------|-----------------|-------------|
| CatBoost | 78.5% | **0.546** | 0.782 | 58.6% | +2.8% F1 |
| XGBoost | 78.4% | **0.547** | 0.782 | 59.0% | **+4.0% F1** |
| LightGBM | 78.5% | 0.544 | 0.778 | 58.0% | +1.9% F1 |
| Random Forest | 78.0% | 0.543 | 0.779 | 59.1% | +0.0% F1 |

### Optimal Classification Thresholds

Instead of using the default 0.5 threshold, we found optimal thresholds for each model:

| Model | Optimal Threshold |
|-------|-------------------|
| CatBoost | 0.566 |
| XGBoost | 0.540 |
| LightGBM | 0.508 |
| Random Forest | 0.483 |

### Key Improvements Applied

1. **Hyperparameter Tuning**: RandomizedSearchCV with cross-validation
2. **Class Weights**: Automatic balancing for imbalanced fraud data
3. **Threshold Optimization**: Per-model optimal thresholds
4. **Feature Engineering**: 36 features including payment patterns, credit utilization, and trends

### Training Commands

```bash
# Standard training
make train

# Training with hyperparameter tuning (recommended)
make train-tuned

# Full tuning (slower but best results)
make train-tuned-full
```

## Configuration

Edit `config/config.toml` to customize:

```toml
[pipeline]
workers = 6  # Parallel workers (optimal for 10-core CPU)

[models]
onnx_threads = 1  # Threads per model

[detection]
threshold = 0.25  # Risk score threshold for alerts

[detection.risk_levels]
low = 0.3
medium = 0.5
high = 0.7
```

## Make Commands

```bash
make help           # Show all commands
make setup          # Full setup from scratch
make nats-start     # Start NATS broker
make run-release    # Run pipeline
make benchmark      # Run performance test
make train-tuned    # Train with hyperparameter tuning
```

## License

MIT

