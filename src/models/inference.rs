//! Multi-model inference engine for fraud detection

use crate::config::{AppConfig, InferenceStrategy};
use crate::models::aggregator::ScoreAggregator;
use crate::models::loader::{LoadedModel, ModelLoader};
use crate::types::alert::{FraudAlert, RiskLevel, RiskLevelThresholds};
use crate::types::transaction::Transaction;
use anyhow::{Context, Result};
use ort::memory::Allocator;
use ort::value::{DynMapValueType, DynSequenceValueType, DowncastableTarget};
use std::collections::HashMap;
use std::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Result of model inference
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Aggregated risk score (0.0 - 1.0)
    pub risk_score: f64,
    /// Individual model scores
    pub model_scores: HashMap<String, f64>,
    /// Features that contributed most to the score
    pub triggered_features: Vec<String>,
}

impl PredictionResult {
    /// Convert prediction result to a fraud alert
    pub fn to_alert(
        &self,
        transaction: &Transaction,
        risk_thresholds: &RiskLevelThresholds,
    ) -> FraudAlert {
        let risk_level = RiskLevel::from_score(self.risk_score, risk_thresholds);

        FraudAlert::new(
            transaction.transaction_id.clone(),
            self.risk_score,
            risk_level,
            self.model_scores.clone(),
        )
        .with_transaction_details(
            transaction.limit_bal,
            format!("age_{}", transaction.age),
            format!("edu_{}", transaction.education),
        )
        .with_triggered_features(self.triggered_features.clone())
    }
}

/// Multi-model inference engine using ONNX Runtime
pub struct InferenceEngine {
    /// Loaded ONNX models (wrapped in RwLock for interior mutability)
    models: Vec<RwLock<LoadedModel>>,
    /// Score aggregator for combining model outputs
    aggregator: ScoreAggregator,
    /// Inference strategy: primary model or ensemble
    strategy: InferenceStrategy,
    /// Primary model name (for primary strategy)
    primary_model: String,
    /// Per-model optimal thresholds
    model_thresholds: HashMap<String, f64>,
}

impl InferenceEngine {
    /// Create a new inference engine from configuration
    pub fn new(config: &AppConfig) -> Result<Self> {
        let loader = ModelLoader::with_threads(config.models.onnx_threads)?;
        let models: Vec<RwLock<LoadedModel>> = loader
            .load_all_models(&config.models.models_dir)?
            .into_iter()
            .map(RwLock::new)
            .collect();

        let aggregator = ScoreAggregator::new(config.models.weights.clone());

        info!(
            strategy = ?config.models.strategy,
            primary_model = %config.models.primary_model,
            "Inference engine initialized"
        );

        Ok(Self {
            models,
            aggregator,
            strategy: config.models.strategy.clone(),
            primary_model: config.models.primary_model.clone(),
            model_thresholds: config.models.thresholds.clone(),
        })
    }

    /// Create inference engine with custom models directory
    pub fn with_models_dir(models_dir: &str, weights: HashMap<String, f64>) -> Result<Self> {
        Self::with_models_dir_and_threads(models_dir, weights, 1)
    }

    /// Create inference engine with custom models directory and thread count
    pub fn with_models_dir_and_threads(
        models_dir: &str,
        weights: HashMap<String, f64>,
        onnx_threads: usize,
    ) -> Result<Self> {
        let loader = ModelLoader::with_threads(onnx_threads)?;
        let models: Vec<RwLock<LoadedModel>> = loader
            .load_all_models(models_dir)?
            .into_iter()
            .map(RwLock::new)
            .collect();
        let aggregator = ScoreAggregator::new(weights);

        // Default to primary strategy with xgboost
        let mut thresholds = HashMap::new();
        thresholds.insert("xgboost".to_string(), 0.61);
        thresholds.insert("ensemble".to_string(), 0.56);

        Ok(Self {
            models,
            aggregator,
            strategy: InferenceStrategy::Primary,
            primary_model: "xgboost".to_string(),
            model_thresholds: thresholds,
        })
    }

    /// Get the current inference strategy
    pub fn strategy(&self) -> &InferenceStrategy {
        &self.strategy
    }

    /// Get the optimal threshold for the current strategy
    pub fn optimal_threshold(&self) -> f64 {
        match self.strategy {
            InferenceStrategy::Primary => {
                self.model_thresholds
                    .get(&self.primary_model)
                    .copied()
                    .unwrap_or(0.61)
            }
            InferenceStrategy::Ensemble => {
                self.model_thresholds
                    .get("ensemble")
                    .copied()
                    .unwrap_or(0.56)
            }
        }
    }

    /// Get the number of loaded models
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Get loaded model names
    pub fn model_names(&self) -> Vec<String> {
        self.models
            .iter()
            .filter_map(|m| m.read().ok().map(|m| m.name.clone()))
            .collect()
    }

    /// Run inference on feature vector using the configured strategy
    pub fn predict(&self, features: &[f32]) -> Result<PredictionResult> {
        match self.strategy {
            InferenceStrategy::Primary => self.predict_primary(features),
            InferenceStrategy::Ensemble => self.predict_ensemble(features),
        }
    }

    /// Run inference using only the primary model (fast path)
    fn predict_primary(&self, features: &[f32]) -> Result<PredictionResult> {
        let mut model_scores = HashMap::new();
        let mut primary_score = None;

        // Find and run only the primary model
        for model_lock in &self.models {
            let model_name;
            let is_primary;

            {
                let model = model_lock
                    .read()
                    .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                model_name = model.name.clone();
                is_primary = model_name == self.primary_model;
            }

            if is_primary {
                let mut model = model_lock
                    .write()
                    .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;

                match self.run_single_model(&mut model, features) {
                    Ok(score) => {
                        primary_score = Some(score);
                        model_scores.insert(model_name, score);
                    }
                    Err(e) => {
                        error!(
                            model = %model_name,
                            error = %e,
                            "Primary model inference failed, falling back to ensemble"
                        );
                        // Fall back to ensemble on primary model failure
                        return self.predict_ensemble(features);
                    }
                }
                break;
            }
        }

        let risk_score = primary_score.unwrap_or_else(|| {
            warn!(
                primary_model = %self.primary_model,
                "Primary model not found, falling back to ensemble"
            );
            // Fall back to ensemble if primary model not found
            if let Ok(result) = self.predict_ensemble(features) {
                return result.risk_score;
            }
            0.5
        });

        let triggered_features: Vec<String> = model_scores
            .iter()
            .filter(|(_, &score)| score > self.optimal_threshold())
            .map(|(name, score)| format!("{}:{:.2}", name, score))
            .collect();

        debug!(
            strategy = "primary",
            model = %self.primary_model,
            risk_score = risk_score,
            "Primary model inference complete"
        );

        Ok(PredictionResult {
            risk_score,
            model_scores,
            triggered_features,
        })
    }

    /// Run inference using all models (ensemble - maximum accuracy)
    fn predict_ensemble(&self, features: &[f32]) -> Result<PredictionResult> {
        let mut model_scores = HashMap::new();

        for model_lock in &self.models {
            let model_name;
            let score_result;

            {
                let mut model = model_lock
                    .write()
                    .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                model_name = model.name.clone();
                score_result = self.run_single_model(&mut model, features);
            }

            match score_result {
                Ok(score) => {
                    model_scores.insert(model_name, score);
                }
                Err(e) => {
                    error!(
                        model = %model_name,
                        error = %e,
                        "Model inference failed"
                    );
                    // Use neutral score for failed models
                    model_scores.insert(model_name, 0.5);
                }
            }
        }

        // Aggregate scores using weighted ensemble
        let risk_score = self.aggregator.aggregate(&model_scores);

        // Identify triggered features (models with high scores)
        let ensemble_threshold = self.model_thresholds.get("ensemble").copied().unwrap_or(0.56);
        let triggered_features: Vec<String> = model_scores
            .iter()
            .filter(|(_, &score)| score > ensemble_threshold)
            .map(|(name, score)| format!("{}:{:.2}", name, score))
            .collect();

        debug!(
            strategy = "ensemble",
            risk_score = risk_score,
            model_scores = ?model_scores,
            "Ensemble inference complete"
        );

        Ok(PredictionResult {
            risk_score,
            model_scores,
            triggered_features,
        })
    }

    /// Run inference on a batch of feature vectors
    pub fn predict_batch(&self, features_batch: &[Vec<f32>]) -> Vec<Result<PredictionResult>> {
        features_batch.iter().map(|f| self.predict(f)).collect()
    }

    /// Run a single model on features
    fn run_single_model(&self, model: &mut LoadedModel, features: &[f32]) -> Result<f64> {
        use ort::value::Tensor;

        // Prepare input tensor - shape [1, num_features]
        let shape = vec![1_i64, features.len() as i64];
        let input_tensor =
            Tensor::from_array((shape, features.to_vec())).context("Failed to create input tensor")?;

        // Get model name for error messages
        let model_name = model.name.clone();

        // Run inference
        let outputs = model
            .session
            .run(ort::inputs![&model.input_name => input_tensor])?;

        // Extract probability score - try different output formats
        let score = self.extract_probability(&outputs, &model.output_name, &model_name)?;

        Ok(score)
    }

    /// Extract fraud probability from model output
    /// Handles both tensor outputs (XGBoost, RandomForest) and seq(map) outputs (CatBoost, LightGBM)
    fn extract_probability(
        &self,
        outputs: &ort::session::SessionOutputs,
        output_name: &str,
        model_name: &str,
    ) -> Result<f64> {
        // First, try to get the "probabilities" output by name
        if let Some(output) = outputs.get(output_name) {
            // Check the value type
            let dtype = output.dtype();

            // Try tensor format first (XGBoost, Random Forest)
            if let Ok(tensor) = output.try_extract_tensor::<f32>() {
                let (shape, data) = tensor;
                let prob = self.extract_fraud_prob_from_tensor(&shape, data);
                debug!(model = %model_name, prob = prob, "Extracted from tensor");
                return Ok(prob);
            }

            // Try sequence format (CatBoost, LightGBM) - seq(map(int64, float))
            if DynSequenceValueType::can_downcast(&dtype) {
                if let Ok(prob) = self.extract_from_sequence_map(output, model_name) {
                    return Ok(prob);
                }
            }
        }

        // Fallback: iterate all outputs and try extraction
        for (name, output) in outputs.iter() {
            // Skip "label" output
            if name.contains("label") {
                continue;
            }

            let dtype = output.dtype();

            // Try tensor
            if let Ok(tensor) = output.try_extract_tensor::<f32>() {
                let (shape, data) = tensor;
                let prob = self.extract_fraud_prob_from_tensor(&shape, data);
                debug!(model = %model_name, output = %name, prob = prob, "Extracted from tensor (fallback)");
                return Ok(prob);
            }

            // Try sequence
            if DynSequenceValueType::can_downcast(&dtype) {
                if let Ok(prob) = self.extract_from_sequence_map(&output, model_name) {
                    return Ok(prob);
                }
            }
        }

        warn!(model = %model_name, "Could not extract probability, using default 0.5");
        Ok(0.5)
    }

    /// Extract probability from seq(map(int64, float)) format
    /// This is used by CatBoost and LightGBM ONNX exports
    fn extract_from_sequence_map(
        &self,
        output: &ort::value::DynValue,
        model_name: &str,
    ) -> Result<f64> {
        let allocator = Allocator::default();

        // Downcast to DynSequence
        let sequence = output
            .downcast_ref::<DynSequenceValueType>()
            .map_err(|e| anyhow::anyhow!("Failed to downcast to sequence: {}", e))?;

        // Extract sequence elements (each is a map)
        let maps = sequence.try_extract_sequence::<DynMapValueType>(&allocator)?;

        if maps.is_empty() {
            return Err(anyhow::anyhow!("Empty sequence"));
        }

        // Get the first map (we only have batch_size=1)
        // maps[0] is a ValueRef<DynMapValueType>
        let map_value = &maps[0];

        // Extract key-value pairs as i64 -> f32
        // DynMapValueType implements MapValueTypeMarker which has try_extract_key_values
        let kv_pairs = map_value.try_extract_key_values::<i64, f32>()?;

        // Find class 1 (fraud) probability
        for (class_id, prob) in &kv_pairs {
            if *class_id == 1 {
                debug!(
                    model = %model_name,
                    prob = *prob,
                    "Extracted from seq(map)"
                );
                return Ok(*prob as f64);
            }
        }

        // If no class 1, return class 0 probability inverted (shouldn't happen)
        for (class_id, prob) in &kv_pairs {
            if *class_id == 0 {
                return Ok(1.0 - *prob as f64);
            }
        }

        Err(anyhow::anyhow!("No probability found in map"))
    }

    /// Extract fraud probability from tensor data
    fn extract_fraud_prob_from_tensor(&self, shape: &ort::tensor::Shape, data: &[f32]) -> f64 {
        // Shape contains dimensions
        let dims: Vec<i64> = shape.iter().copied().collect();

        if dims.len() == 2 {
            let num_classes = dims[1] as usize;
            if num_classes >= 2 {
                // [batch, num_classes] - get fraud class probability (index 1)
                return data[1] as f64;
            } else if num_classes == 1 {
                // [batch, 1] - single probability
                return data[0] as f64;
            }
        } else if dims.len() == 1 {
            let num_classes = dims[0] as usize;
            if num_classes >= 2 {
                // [num_classes] - get fraud class
                return data[1] as f64;
            } else if num_classes == 1 {
                return data[0] as f64;
            }
        }

        // Fallback: return last value
        data.last().map(|&v| v as f64).unwrap_or(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_result() {
        let mut scores = HashMap::new();
        scores.insert("catboost".to_string(), 0.8);
        scores.insert("xgboost".to_string(), 0.75);

        let result = PredictionResult {
            risk_score: 0.78,
            model_scores: scores,
            triggered_features: vec!["catboost:0.80".to_string()],
        };

        assert_eq!(result.risk_score, 0.78);
        assert_eq!(result.model_scores.len(), 2);
    }
}
