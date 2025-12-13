//! Configuration management for the fraud detection pipeline

use crate::types::alert::RiskLevelThresholds;
use anyhow::{Context, Result};
use config::{Config, File};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Main application configuration
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub nats: NatsConfig,
    pub models: ModelsConfig,
    pub detection: DetectionConfig,
    pub pipeline: PipelineConfig,
    pub logging: LoggingConfig,
}

/// NATS connection configuration
#[derive(Debug, Clone, Deserialize)]
pub struct NatsConfig {
    /// NATS server URL
    pub url: String,
    /// Subject for incoming transactions
    pub transaction_subject: String,
    /// Subject for outgoing fraud alerts
    pub alert_subject: String,
}

/// ML models configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    /// Directory containing ONNX model files
    pub models_dir: String,
    /// Model weights for ensemble scoring
    pub weights: HashMap<String, f64>,
    /// Number of threads for ONNX inference per model (default: 1)
    #[serde(default = "default_onnx_threads")]
    pub onnx_threads: usize,
}

fn default_onnx_threads() -> usize {
    1
}

/// Detection configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DetectionConfig {
    /// Risk score threshold for generating alerts
    pub threshold: f64,
    /// Risk level classification thresholds
    pub risk_levels: RiskLevelThresholds,
}

/// Pipeline configuration
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    /// Number of worker threads
    pub workers: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Processing timeout in milliseconds
    pub timeout_ms: u64,
}

/// Logging configuration
#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log format (json, pretty)
    pub format: String,
}

impl AppConfig {
    /// Load configuration from file
    pub fn load() -> Result<Self> {
        Self::load_from_path("config/config.toml")
    }

    /// Load configuration from a specific path
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = Config::builder()
            .add_source(File::from(path.as_ref()))
            .build()
            .context("Failed to build configuration")?;

        config
            .try_deserialize()
            .context("Failed to deserialize configuration")
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("catboost".to_string(), 0.25);
        weights.insert("xgboost".to_string(), 0.25);
        weights.insert("lightgbm".to_string(), 0.20);
        weights.insert("random_forest".to_string(), 0.15);
        weights.insert("isolation_forest".to_string(), 0.15);

        Self {
            nats: NatsConfig {
                url: "nats://localhost:4222".to_string(),
                transaction_subject: "transactions".to_string(),
                alert_subject: "fraud.alerts".to_string(),
            },
            models: ModelsConfig {
                models_dir: "models".to_string(),
                weights,
                onnx_threads: 1,
            },
            detection: DetectionConfig {
                threshold: 0.7,
                risk_levels: RiskLevelThresholds::default(),
            },
            pipeline: PipelineConfig {
                workers: 4,
                batch_size: 32,
                timeout_ms: 1000,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.nats.url, "nats://localhost:4222");
        assert_eq!(config.detection.threshold, 0.7);
        assert_eq!(config.models.weights.len(), 5);
    }
}

