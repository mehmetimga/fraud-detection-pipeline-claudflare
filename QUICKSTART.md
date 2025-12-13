# Quick Start Guide

This guide will help you get the fraud detection pipeline running quickly.

## Prerequisites

- Docker (for NATS message broker)
- Rust 1.70+
- Python 3.9+

## Quick Start with Make

```bash
# 1. Setup everything (install deps, download data, train models, build)
make setup

# 2. Start NATS message broker
make nats-start

# 3. Run the pipeline
make run-release

# 4. In another terminal, send test transactions
make send-test

# 5. View results
make logs
```

## Available Make Commands

Run `make help` to see all commands:

```
BUILD:
  make build          - Build debug version
  make build-release  - Build optimized release version
  make clean          - Clean build artifacts

RUN:
  make run-release    - Run pipeline (foreground)
  make run-bg         - Run pipeline (background)
  make stop           - Stop pipeline
  make logs           - Show pipeline logs

NATS:
  make nats-start     - Start NATS server (Docker)
  make nats-stop      - Stop NATS server
  make nats-status    - Check NATS status

TRAINING:
  make train          - Train all ML models
  make inspect-models - Inspect ONNX model outputs

DATA:
  make data-setup     - Download and prepare dataset

TESTING:
  make send-test      - Send 100 test transactions
  make send-all       - Send all 6000 test transactions
  make receive-alerts - Receive fraud alerts
  make benchmark      - Run full benchmark
```

## Step-by-Step Setup

### Step 1: Install Dependencies

```bash
# Install Python dependencies
make install-python-deps

# Or manually:
pip install -r training/requirements.txt
```

### Step 2: Download and Prepare Data

```bash
# Download UCI Credit Card Default dataset and prepare train/test splits
make data-setup

# Or manually:
make data-download
make data-prepare
```

### Step 3: Train Models

```bash
make train

# Or manually:
cd training && python train.py
```

This will:
- Load the UCI Credit Card Default dataset
- Train 4 ML models (CatBoost, XGBoost, LightGBM, Random Forest)
- Export models to ONNX format in `models/` directory

### Step 4: Build the Pipeline

```bash
make build-release
```

### Step 5: Start NATS

```bash
make nats-start

# Check status
make nats-status
```

### Step 6: Run the Pipeline

```bash
# Foreground (see logs directly)
make run-release

# Or background (logs to file)
make run-bg
make logs  # View logs
```

### Step 7: Send Test Transactions

```bash
# Send 100 test transactions
make send-test

# Send all 6000 test transactions
make send-all

# Run full benchmark
make benchmark
```

### Step 8: Receive Alerts

```bash
make receive-alerts
```

## Configuration

Edit `config/config.toml` to customize:

```toml
[nats]
url = "nats://localhost:4222"
transaction_subject = "transactions"
alert_subject = "fraud.alerts"

[models]
models_dir = "models"
onnx_threads = 1  # Threads per ONNX model (1 is optimal for small models)

[models.weights]
catboost = 0.25
xgboost = 0.25
lightgbm = 0.20
random_forest = 0.15
isolation_forest = 0.15

[detection]
threshold = 0.25  # Risk score threshold for alerts

[detection.risk_levels]
low = 0.3
medium = 0.5
high = 0.7
critical = 0.9

[pipeline]
workers = 6  # Parallel transaction processing workers
batch_size = 32
timeout_ms = 1000
```

## Performance

Current benchmarks on 10-core CPU:

| Workers | Throughput | Latency | Daily Capacity |
|---------|-----------|---------|----------------|
| 1 | 190 tx/s | 25 μs | 16.4 million |
| 4 | 891 tx/s | 89 μs | 77 million |
| **6** | **1069 tx/s** | 121 μs | **92.4 million** |
| 10 | 1020 tx/s | 241 μs | 88 million |

**Recommended: `workers = 6`** for optimal throughput.

## Testing with Python Scripts

### Send Test Data

```bash
# Send specific number of transactions
python tools/send_test_data.py --num_transactions 100

# Send all test data
python tools/benchmark_sender.py
```

### Receive Alerts

```bash
python tools/receive_alerts.py
```

### Analyze Results

```bash
python tools/analyze_results.py
```

## Transaction Format

The pipeline expects transactions in this JSON format (matching UCI Credit Card dataset):

```json
{
  "ID": 12345,
  "LIMIT_BAL": 50000.0,
  "SEX": 1,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 30,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 10000.0,
  "BILL_AMT2": 9500.0,
  "BILL_AMT3": 9000.0,
  "BILL_AMT4": 8500.0,
  "BILL_AMT5": 8000.0,
  "BILL_AMT6": 7500.0,
  "PAY_AMT1": 1000.0,
  "PAY_AMT2": 1000.0,
  "PAY_AMT3": 1000.0,
  "PAY_AMT4": 1000.0,
  "PAY_AMT5": 1000.0,
  "PAY_AMT6": 1000.0
}
```

## Alert Format

When a fraud alert is generated:

```json
{
  "alert_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "transaction_id": "12345",
  "risk_score": 0.45,
  "risk_level": "medium",
  "model_scores": {
    "catboost": 0.48,
    "xgboost": 0.45,
    "lightgbm": 0.42,
    "random_forest": 0.44
  },
  "triggered_features": ["catboost:0.48"],
  "timestamp": "2024-01-15T10:30:05Z",
  "credit_limit": 50000.0,
  "customer_info": "age_30_sex_1",
  "context": "edu_2_marriage_1"
}
```

## Troubleshooting

### NATS Connection Failed

```bash
# Check if NATS is running
make nats-status

# Restart NATS
make nats-restart
```

### Models Not Found

```bash
# Check models directory
ls -la models/

# Retrain models
make train
```

### Pipeline Won't Start

```bash
# Check for existing processes
pgrep -l fraud

# Stop and restart
make stop
make run-release
```

### High Latency

1. Ensure release build: `make build-release`
2. Adjust workers in `config/config.toml`: `workers = 6`
3. Check NATS connection: `make nats-status`

## Docker Alternative

```bash
# Start all services
make docker-up

# View logs
make docker-logs

# Stop all
make docker-down
```

## Integration Test

Run a quick end-to-end test:

```bash
make integration-test
```

This will:
1. Start NATS
2. Build and run the pipeline
3. Send 50 test transactions
4. Report alerts generated
5. Clean up

## Next Steps

1. **Train with your own data**: Replace `training/data/` with your dataset
2. **Tune model hyperparameters**: Edit `training/train.py`
3. **Adjust ensemble weights**: Edit `config/config.toml`
4. **Configure alerting thresholds**: Set `detection.threshold`
5. **Scale horizontally**: Run multiple pipeline instances
6. **Integrate with your system**: Connect to your payment gateway
