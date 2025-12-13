# Fraud Detection Pipeline - Rust Build
FROM rust:1.75-bookworm as builder

WORKDIR /app

# Install dependencies for ONNX Runtime
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock* ./

# Create dummy main files to build dependencies
RUN mkdir -p src tools && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > tools/test_producer.rs

# Build dependencies only
RUN cargo build --release && rm -rf src tools

# Copy source code
COPY src ./src
COPY tools ./tools

# Build the application
RUN touch src/main.rs tools/test_producer.rs && \
    cargo build --release

# Runtime image
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/fraud-pipeline /app/fraud-pipeline
COPY --from=builder /app/target/release/test-producer /app/test-producer

# Copy config
COPY config ./config

# Create models directory
RUN mkdir -p models

# Set environment
ENV RUST_LOG=info

# Run the pipeline
CMD ["/app/fraud-pipeline"]

