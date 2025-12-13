//! Fraud Detection Pipeline Library
//!
//! A high-performance, real-time payment fraud detection pipeline
//! inspired by Cloudflare's BLISS architecture.

pub mod config;
pub mod consumer;
pub mod feature_extractor;
pub mod metrics;
pub mod models;
pub mod producer;
pub mod types;

pub use config::AppConfig;
pub use consumer::TransactionConsumer;
pub use feature_extractor::FeatureExtractor;
pub use models::inference::InferenceEngine;
pub use producer::AlertProducer;
pub use types::{alert::FraudAlert, transaction::Transaction};

