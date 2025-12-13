# Fraud Detection Pipeline - Makefile
# =====================================

.PHONY: all build build-release clean test run run-release \
        train nats-start nats-stop \
        benchmark send-test receive-alerts \
        install-deps install-python-deps \
        data-download data-prepare \
        docker-up docker-down \
        help

# Default target
all: build

# =====================================
# Build Targets
# =====================================

## Build debug version
build:
	cargo build

## Build release version (optimized)
build-release:
	cargo build --release

## Clean build artifacts
clean:
	cargo clean
	rm -rf training/__pycache__ training/src/__pycache__
	rm -f /tmp/fraud*.log /tmp/*workers*.log /tmp/parallel*.log /tmp/benchmark*.log

## Run tests
test:
	cargo test

## Check code without building
check:
	cargo check
	cargo clippy

# =====================================
# Run Pipeline
# =====================================

## Run pipeline (debug mode)
run:
	RUST_LOG=info cargo run

## Run pipeline (release mode, optimized)
run-release:
	RUST_LOG=warn,fraud_pipeline=info ./target/release/fraud-pipeline

## Run pipeline with debug logging
run-debug:
	RUST_LOG=debug ./target/release/fraud-pipeline

## Run pipeline in background with logging
run-bg:
	@echo "Starting pipeline in background..."
	@RUST_LOG=warn,fraud_pipeline=info ./target/release/fraud-pipeline > /tmp/fraud_pipeline.log 2>&1 &
	@sleep 2
	@pgrep -f fraud-pipeline && echo "Pipeline running. Logs: /tmp/fraud_pipeline.log"

## Stop pipeline
stop:
	@pkill -f fraud-pipeline 2>/dev/null && echo "Pipeline stopped" || echo "Pipeline not running"

## Show pipeline logs
logs:
	@tail -f /tmp/fraud_pipeline.log 2>/dev/null || echo "No log file found. Run 'make run-bg' first."

# =====================================
# NATS Message Broker
# =====================================

## Start NATS server (Docker)
nats-start:
	@docker ps -q -f name=nats-fraud 2>/dev/null | grep -q . && echo "NATS already running" || \
	(docker run -d --name nats-fraud -p 4222:4222 -p 8222:8222 nats:latest && echo "NATS started on port 4222")

## Stop NATS server
nats-stop:
	@docker stop nats-fraud 2>/dev/null && docker rm nats-fraud 2>/dev/null && echo "NATS stopped" || echo "NATS not running"

## Restart NATS server
nats-restart: nats-stop nats-start

## Check NATS status
nats-status:
	@docker ps -f name=nats-fraud --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "NATS not running"
	@curl -s http://localhost:8222/varz 2>/dev/null | head -5 || true

# =====================================
# Model Training (Python)
# =====================================

## Install Python dependencies
install-python-deps:
	pip install -r training/requirements.txt

## Train all models
train:
	cd training && python train.py

## Train models with verbose output
train-verbose:
	cd training && python train.py --verbose

## Export models to ONNX format
export-models:
	cd training && python -c "from src.export_onnx import export_all_models; export_all_models()"

## Inspect ONNX models
inspect-models:
	python tools/inspect_onnx_models.py

# =====================================
# Data Preparation
# =====================================

## Download dataset (UCI Credit Card Default)
data-download:
	@echo "Downloading UCI Credit Card Default dataset..."
	@mkdir -p training/data
	@curl -L -o training/data/credit_card_default.xls \
		"https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
	@echo "Download complete: training/data/credit_card_default.xls"

## Convert and prepare data (XLS -> CSV -> Parquet, train/test split)
data-prepare:
	@echo "Converting XLS to CSV and Parquet, splitting into train/test..."
	@python3 -c "\
import pandas as pd; \
from sklearn.model_selection import train_test_split; \
df = pd.read_excel('training/data/credit_card_default.xls', header=1); \
df = df.rename(columns={'default payment next month': 'is_default'}); \
df.to_csv('training/data/credit_card_default.csv', index=False); \
df.to_parquet('training/data/credit_card_default.parquet', index=False); \
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_default']); \
train.to_parquet('training/data/train.parquet', index=False); \
test.to_parquet('training/data/test.parquet', index=False); \
print(f'Train: {len(train)} rows, Test: {len(test)} rows'); \
print(f'Default rate - Train: {train[\"is_default\"].mean():.1%}, Test: {test[\"is_default\"].mean():.1%}')"

## Full data pipeline (download + prepare)
data-setup: data-download data-prepare

# =====================================
# Testing & Benchmarking
# =====================================

## Send test transactions (100 records)
send-test:
	python tools/send_test_data.py --num_transactions 100

## Send all test data (6000 records)
send-all:
	python tools/benchmark_sender.py

## Receive and display alerts
receive-alerts:
	python tools/receive_alerts.py

## Run full benchmark (6000 transactions)
benchmark: build-release
	@echo "=== FULL BENCHMARK ==="
	@make stop 2>/dev/null || true
	@make run-bg
	@sleep 3
	@python tools/benchmark_sender.py
	@sleep 5
	@grep "processed=6000" /tmp/fraud_pipeline.log || echo "Check /tmp/fraud_pipeline.log for results"
	@make stop

## Quick benchmark (1000 transactions)
benchmark-quick:
	@make stop 2>/dev/null || true
	@make run-bg
	@sleep 3
	@python tools/benchmark_sender.py --num_transactions 1000
	@sleep 3
	@grep "processed=1000" /tmp/fraud_pipeline.log || tail -5 /tmp/fraud_pipeline.log
	@make stop

## Analyze results
analyze:
	python tools/analyze_results.py

# =====================================
# Docker
# =====================================

## Start all services with Docker Compose
docker-up:
	docker-compose up -d

## Stop all Docker services
docker-down:
	docker-compose down

## Build Docker images
docker-build:
	docker-compose build

## View Docker logs
docker-logs:
	docker-compose logs -f

# =====================================
# Development
# =====================================

## Format code
fmt:
	cargo fmt
	@cd training && black . 2>/dev/null || true

## Lint code
lint:
	cargo clippy -- -W warnings
	@cd training && ruff check . 2>/dev/null || true

## Watch for changes and rebuild
watch:
	cargo watch -x build

# =====================================
# Full Workflows
# =====================================

## Setup everything from scratch
setup: install-python-deps data-setup train build-release
	@echo ""
	@echo "=== SETUP COMPLETE ==="
	@echo "1. Start NATS:    make nats-start"
	@echo "2. Run pipeline:  make run-release"
	@echo "3. Send test:     make send-test"

## Full integration test
integration-test: nats-start build-release
	@echo "Running integration test..."
	@make run-bg
	@sleep 3
	@python tools/send_test_data.py --num_transactions 50
	@sleep 3
	@grep "Fraud alert published" /tmp/fraud_pipeline.log | wc -l | xargs echo "Alerts generated:"
	@make stop
	@echo "Integration test complete!"

# =====================================
# Help
# =====================================

## Show this help message
help:
	@echo ""
	@echo "Fraud Detection Pipeline - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "BUILD:"
	@echo "  make build          - Build debug version"
	@echo "  make build-release  - Build optimized release version"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "RUN:"
	@echo "  make run-release    - Run pipeline (foreground)"
	@echo "  make run-bg         - Run pipeline (background)"
	@echo "  make stop           - Stop pipeline"
	@echo "  make logs           - Show pipeline logs"
	@echo ""
	@echo "NATS:"
	@echo "  make nats-start     - Start NATS server (Docker)"
	@echo "  make nats-stop      - Stop NATS server"
	@echo "  make nats-status    - Check NATS status"
	@echo ""
	@echo "TRAINING:"
	@echo "  make train          - Train all ML models"
	@echo "  make inspect-models - Inspect ONNX model outputs"
	@echo ""
	@echo "DATA:"
	@echo "  make data-setup     - Download and prepare dataset"
	@echo ""
	@echo "TESTING:"
	@echo "  make send-test      - Send 100 test transactions"
	@echo "  make send-all       - Send all 6000 test transactions"
	@echo "  make receive-alerts - Receive fraud alerts"
	@echo "  make benchmark      - Run full benchmark"
	@echo ""
	@echo "SETUP:"
	@echo "  make setup          - Full setup from scratch"
	@echo "  make integration-test - Run integration test"
	@echo ""
	@echo "Current config:"
	@grep -E "workers|onnx_threads|threshold" config/config.toml 2>/dev/null || true
	@echo ""

