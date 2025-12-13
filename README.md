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

Inspired by Cloudflare's BLISS architecture:
- Zero-copy message processing where possible
- Pre-allocated buffers for feature vectors
- ONNX Runtime for efficient model inference
- Async I/O with Tokio

## License

MIT

