//! Score aggregation for multi-model ensemble

use std::collections::HashMap;

/// Aggregates scores from multiple models into a single risk score.
pub struct ScoreAggregator {
    /// Model weights for weighted average
    weights: HashMap<String, f64>,
    /// Default weight for models not in the weights map
    default_weight: f64,
}

impl ScoreAggregator {
    /// Create a new score aggregator with model weights.
    pub fn new(weights: HashMap<String, f64>) -> Self {
        Self {
            weights,
            default_weight: 0.1,
        }
    }

    /// Create aggregator with equal weights for all models.
    pub fn equal_weights() -> Self {
        Self {
            weights: HashMap::new(),
            default_weight: 1.0,
        }
    }

    /// Aggregate multiple model scores into a single risk score.
    ///
    /// Uses weighted average where weights are normalized to sum to 1.
    pub fn aggregate(&self, model_scores: &HashMap<String, f64>) -> f64 {
        if model_scores.is_empty() {
            return 0.5; // Neutral score when no models
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (model_name, &score) in model_scores {
            let weight = self
                .weights
                .get(model_name)
                .copied()
                .unwrap_or(self.default_weight);

            weighted_sum += score * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Aggregate with a minimum threshold - at least one model must exceed threshold.
    pub fn aggregate_with_threshold(
        &self,
        model_scores: &HashMap<String, f64>,
        threshold: f64,
    ) -> f64 {
        let any_above_threshold = model_scores.values().any(|&score| score >= threshold);

        if any_above_threshold {
            self.aggregate(model_scores)
        } else {
            // If no model is confident, reduce the aggregated score
            self.aggregate(model_scores) * 0.5
        }
    }

    /// Get the maximum score among all models.
    pub fn max_score(model_scores: &HashMap<String, f64>) -> f64 {
        model_scores
            .values()
            .copied()
            .fold(0.0, |a, b| a.max(b))
    }

    /// Get the minimum score among all models.
    pub fn min_score(model_scores: &HashMap<String, f64>) -> f64 {
        model_scores
            .values()
            .copied()
            .fold(1.0, |a, b| a.min(b))
    }

    /// Get the median score among all models.
    pub fn median_score(model_scores: &HashMap<String, f64>) -> f64 {
        let mut scores: Vec<f64> = model_scores.values().copied().collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if scores.is_empty() {
            return 0.5;
        }

        let mid = scores.len() / 2;
        if scores.len() % 2 == 0 {
            (scores[mid - 1] + scores[mid]) / 2.0
        } else {
            scores[mid]
        }
    }

    /// Calculate voting-based score (fraction of models above threshold).
    pub fn voting_score(model_scores: &HashMap<String, f64>, threshold: f64) -> f64 {
        if model_scores.is_empty() {
            return 0.0;
        }

        let votes_for_fraud = model_scores
            .values()
            .filter(|&&score| score >= threshold)
            .count();

        votes_for_fraud as f64 / model_scores.len() as f64
    }

    /// Set weight for a specific model.
    pub fn set_weight(&mut self, model_name: &str, weight: f64) {
        self.weights.insert(model_name.to_string(), weight);
    }

    /// Get configured weights.
    pub fn get_weights(&self) -> &HashMap<String, f64> {
        &self.weights
    }
}

impl Default for ScoreAggregator {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("catboost".to_string(), 0.25);
        weights.insert("xgboost".to_string(), 0.25);
        weights.insert("lightgbm".to_string(), 0.20);
        weights.insert("random_forest".to_string(), 0.15);
        weights.insert("isolation_forest".to_string(), 0.15);

        Self {
            weights,
            default_weight: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_aggregation() {
        let aggregator = ScoreAggregator::default();

        let mut scores = HashMap::new();
        scores.insert("catboost".to_string(), 0.8);
        scores.insert("xgboost".to_string(), 0.8);
        scores.insert("lightgbm".to_string(), 0.7);
        scores.insert("random_forest".to_string(), 0.6);
        scores.insert("isolation_forest".to_string(), 0.5);

        let aggregated = aggregator.aggregate(&scores);

        // Expected: (0.8*0.25 + 0.8*0.25 + 0.7*0.2 + 0.6*0.15 + 0.5*0.15) / 1.0 = 0.705
        assert!((aggregated - 0.705).abs() < 0.01);
    }

    #[test]
    fn test_equal_weights() {
        let aggregator = ScoreAggregator::equal_weights();

        let mut scores = HashMap::new();
        scores.insert("model1".to_string(), 0.8);
        scores.insert("model2".to_string(), 0.6);

        let aggregated = aggregator.aggregate(&scores);

        // Should be simple average: 0.7
        assert!((aggregated - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_voting_score() {
        let mut scores = HashMap::new();
        scores.insert("model1".to_string(), 0.8);
        scores.insert("model2".to_string(), 0.6);
        scores.insert("model3".to_string(), 0.4);
        scores.insert("model4".to_string(), 0.3);

        // 2 out of 4 models above 0.5 threshold
        let vote = ScoreAggregator::voting_score(&scores, 0.5);
        assert!((vote - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_median_score() {
        let mut scores = HashMap::new();
        scores.insert("model1".to_string(), 0.9);
        scores.insert("model2".to_string(), 0.7);
        scores.insert("model3".to_string(), 0.5);
        scores.insert("model4".to_string(), 0.3);
        scores.insert("model5".to_string(), 0.1);

        let median = ScoreAggregator::median_score(&scores);
        assert!((median - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_empty_scores() {
        let aggregator = ScoreAggregator::default();
        let scores = HashMap::new();

        let aggregated = aggregator.aggregate(&scores);
        assert_eq!(aggregated, 0.5);
    }
}

