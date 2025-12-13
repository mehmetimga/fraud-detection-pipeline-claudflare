//! Fraud Detection Pipeline - Main Entry Point
//!
//! Consumes transactions from NATS, runs ML inference, and publishes fraud alerts.
//! Supports parallel transaction processing for high throughput.

use anyhow::Result;
use fraud_detection_pipeline::{
    config::AppConfig, consumer::TransactionConsumer, feature_extractor::FeatureExtractor,
    metrics::{MetricsReporter, PipelineMetrics},
    models::inference::InferenceEngine,
    producer::AlertProducer,
};
use futures::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("fraud_detection_pipeline=info".parse()?),
        )
        .init();

    info!("Starting Fraud Detection Pipeline");

    // Load configuration
    let config = AppConfig::load()?;
    info!("Configuration loaded successfully");
    info!(
        "Detection threshold: {:.2}, Alert levels: low<{:.2}, medium<{:.2}, high<{:.2}",
        config.detection.threshold,
        config.detection.risk_levels.low,
        config.detection.risk_levels.medium,
        config.detection.risk_levels.high
    );

    // Initialize metrics
    let metrics = Arc::new(PipelineMetrics::new());

    // Initialize components
    let feature_extractor = Arc::new(FeatureExtractor::new());
    info!(
        "Feature extractor initialized ({} features)",
        feature_extractor.feature_count()
    );

    // Initialize inference engine with ONNX models
    let inference_engine = Arc::new(InferenceEngine::new(&config)?);
    info!(
        "Inference engine initialized with {} models: {:?}",
        inference_engine.model_count(),
        inference_engine.model_names()
    );

    // Connect to NATS
    let client = async_nats::connect(&config.nats.url).await?;
    info!("Connected to NATS at {}", config.nats.url);

    // Initialize consumer and producer
    let consumer = TransactionConsumer::new(client.clone(), &config.nats.transaction_subject);
    let producer = Arc::new(AlertProducer::new(client.clone(), &config.nats.alert_subject));

    // Parallel processing configuration
    let num_workers = config.pipeline.workers;
    info!(
        "Starting transaction processing loop with {} parallel workers",
        num_workers
    );
    info!(
        "Listening on subject: {}",
        config.nats.transaction_subject
    );
    info!("Publishing alerts to: {}", config.nats.alert_subject);

    // Semaphore to limit concurrent processing
    let semaphore = Arc::new(Semaphore::new(num_workers));
    let processed_count = Arc::new(AtomicU64::new(0));

    // Wrap config in Arc for sharing
    let config = Arc::new(config);

    // Start metrics reporter (prints summary every 30 seconds)
    let metrics_clone = metrics.clone();
    tokio::spawn(async move {
        let reporter = MetricsReporter::new(metrics_clone, 30);
        reporter.start().await;
    });

    // Process transactions in parallel
    let mut subscription = consumer.subscribe().await?;

    while let Some(message) = subscription.next().await {
        // Acquire permit (limits concurrent tasks)
        let permit = semaphore.clone().acquire_owned().await.unwrap();

        // Clone shared resources for the spawned task
        let feature_extractor = feature_extractor.clone();
        let inference_engine = inference_engine.clone();
        let producer = producer.clone();
        let metrics = metrics.clone();
        let config = config.clone();
        let processed_count = processed_count.clone();

        // Spawn task to process this transaction
        tokio::spawn(async move {
            let start_time = Instant::now();

            match serde_json::from_slice::<fraud_detection_pipeline::Transaction>(&message.payload)
            {
                Ok(transaction) => {
                    let tx_id = transaction.transaction_id.clone();

                    // Extract features
                    let features = feature_extractor.extract(&transaction);

                    // Run inference
                    match inference_engine.predict(&features) {
                        Ok(prediction) => {
                            let processing_time = start_time.elapsed();

                            // Record metrics
                            metrics.record_transaction(processing_time, prediction.risk_score);
                            metrics.record_model_agreement(&prediction.model_scores);

                            // Check if fraud alert should be generated
                            if prediction.risk_score >= config.detection.threshold {
                                let alert = prediction
                                    .to_alert(&transaction, &config.detection.risk_levels);

                                // Record alert metrics
                                metrics.record_alert(
                                    &format!("{:?}", alert.risk_level).to_lowercase(),
                                );

                                if let Err(e) = producer.publish(&alert).await {
                                    error!(
                                        transaction_id = %tx_id,
                                        error = %e,
                                        "Failed to publish fraud alert"
                                    );
                                } else {
                                    info!(
                                        transaction_id = %tx_id,
                                        risk_score = prediction.risk_score,
                                        risk_level = ?alert.risk_level,
                                        processing_time_us = processing_time.as_micros(),
                                        "Fraud alert published"
                                    );
                                }
                            } else {
                                debug!(
                                    transaction_id = %tx_id,
                                    risk_score = prediction.risk_score,
                                    processing_time_us = processing_time.as_micros(),
                                    "Transaction processed (below threshold)"
                                );
                            }

                            let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;

                            // Log progress every 100 transactions
                            if count % 100 == 0 {
                                let throughput = metrics.get_throughput();
                                let processing_stats = metrics.get_processing_stats();
                                info!(
                                    processed = count,
                                    throughput = format!("{:.1} tx/s", throughput),
                                    avg_latency_us = processing_stats.mean_us,
                                    "Processing milestone"
                                );
                            }
                        }
                        Err(e) => {
                            error!(
                                transaction_id = %tx_id,
                                error = %e,
                                "Inference failed"
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to deserialize transaction");
                }
            }

            // Release permit when done
            drop(permit);
        });
    }

    // Print final summary
    info!("Pipeline shutting down...");
    metrics.print_summary();

    Ok(())
}
