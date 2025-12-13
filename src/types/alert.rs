//! Fraud alert data structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskLevel {
    /// Determine risk level from score and thresholds
    pub fn from_score(score: f64, thresholds: &RiskLevelThresholds) -> Self {
        if score >= thresholds.critical {
            RiskLevel::Critical
        } else if score >= thresholds.high {
            RiskLevel::High
        } else if score >= thresholds.medium {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }
}

/// Configurable risk level thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLevelThresholds {
    pub low: f64,
    pub medium: f64,
    pub high: f64,
    pub critical: f64,
}

impl Default for RiskLevelThresholds {
    fn default() -> Self {
        Self {
            low: 0.3,
            medium: 0.5,
            high: 0.7,
            critical: 0.9,
        }
    }
}

/// Fraud alert generated when a transaction exceeds risk threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudAlert {
    /// Unique alert identifier
    pub alert_id: String,

    /// Associated transaction ID
    pub transaction_id: String,

    /// Combined risk score (0.0 - 1.0)
    pub risk_score: f64,

    /// Risk level classification
    pub risk_level: RiskLevel,

    /// Individual model scores
    pub model_scores: HashMap<String, f64>,

    /// Rules or features that contributed to the alert
    pub triggered_features: Vec<String>,

    /// Alert generation timestamp
    pub timestamp: DateTime<Utc>,

    /// Credit limit from the transaction
    pub credit_limit: f64,

    /// Customer info
    pub customer_info: String,

    /// Additional context
    pub context: String,
}

impl FraudAlert {
    /// Create a new fraud alert
    pub fn new(
        transaction_id: String,
        risk_score: f64,
        risk_level: RiskLevel,
        model_scores: HashMap<String, f64>,
    ) -> Self {
        Self {
            alert_id: uuid::Uuid::new_v4().to_string(),
            transaction_id,
            risk_score,
            risk_level,
            model_scores,
            triggered_features: Vec::new(),
            timestamp: Utc::now(),
            credit_limit: 0.0,
            customer_info: String::new(),
            context: String::new(),
        }
    }

    /// Add transaction details to the alert
    pub fn with_transaction_details(
        mut self,
        credit_limit: f64,
        customer_info: String,
        context: String,
    ) -> Self {
        self.credit_limit = credit_limit;
        self.customer_info = customer_info;
        self.context = context;
        self
    }

    /// Add triggered features to the alert
    pub fn with_triggered_features(mut self, features: Vec<String>) -> Self {
        self.triggered_features = features;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_from_score() {
        let thresholds = RiskLevelThresholds::default();

        assert_eq!(RiskLevel::from_score(0.1, &thresholds), RiskLevel::Low);
        assert_eq!(RiskLevel::from_score(0.5, &thresholds), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_score(0.75, &thresholds), RiskLevel::High);
        assert_eq!(RiskLevel::from_score(0.95, &thresholds), RiskLevel::Critical);
    }

    #[test]
    fn test_fraud_alert_serialization() {
        let mut model_scores = HashMap::new();
        model_scores.insert("catboost".to_string(), 0.8);
        model_scores.insert("xgboost".to_string(), 0.75);

        let alert = FraudAlert::new(
            "tx_123".to_string(),
            0.78,
            RiskLevel::High,
            model_scores,
        );

        let json = serde_json::to_string(&alert).unwrap();
        let deserialized: FraudAlert = serde_json::from_str(&json).unwrap();

        assert_eq!(alert.transaction_id, deserialized.transaction_id);
        assert_eq!(alert.risk_score, deserialized.risk_score);
        assert_eq!(alert.risk_level, deserialized.risk_level);
    }
}

