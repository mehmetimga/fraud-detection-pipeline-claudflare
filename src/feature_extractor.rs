//! Feature extraction for credit card default detection model inference.
//!
//! This module extracts features from transactions that match
//! the features used during Python model training.

use crate::types::transaction::Transaction;

/// Feature extractor that transforms transactions into model input features.
///
/// Matches the preprocessing done in Python training pipeline.
/// Features are extracted in the exact order expected by the ONNX models.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Create a new feature extractor.
    pub fn new() -> Self {
        Self
    }

    /// Extract features from a transaction.
    ///
    /// Returns a feature vector matching the training data format (36 features).
    /// Order matches feature_info.json from training.
    pub fn extract(&self, tx: &Transaction) -> Vec<f32> {
        let mut features = Vec::with_capacity(36);

        // Original numeric features (20)
        features.push(tx.limit_bal as f32);
        features.push(tx.age as f32);
        features.push(tx.pay_0 as f32);
        features.push(tx.pay_2 as f32);
        features.push(tx.pay_3 as f32);
        features.push(tx.pay_4 as f32);
        features.push(tx.pay_5 as f32);
        features.push(tx.pay_6 as f32);
        features.push(tx.bill_amt1 as f32);
        features.push(tx.bill_amt2 as f32);
        features.push(tx.bill_amt3 as f32);
        features.push(tx.bill_amt4 as f32);
        features.push(tx.bill_amt5 as f32);
        features.push(tx.bill_amt6 as f32);
        features.push(tx.pay_amt1 as f32);
        features.push(tx.pay_amt2 as f32);
        features.push(tx.pay_amt3 as f32);
        features.push(tx.pay_amt4 as f32);
        features.push(tx.pay_amt5 as f32);
        features.push(tx.pay_amt6 as f32);

        // Engineered features (13)
        let pay_delays = [tx.pay_0, tx.pay_2, tx.pay_3, tx.pay_4, tx.pay_5, tx.pay_6];
        let bill_amts = [
            tx.bill_amt1,
            tx.bill_amt2,
            tx.bill_amt3,
            tx.bill_amt4,
            tx.bill_amt5,
            tx.bill_amt6,
        ];
        let pay_amts = [
            tx.pay_amt1,
            tx.pay_amt2,
            tx.pay_amt3,
            tx.pay_amt4,
            tx.pay_amt5,
            tx.pay_amt6,
        ];

        // avg_pay_delay
        let avg_pay_delay: f32 = pay_delays.iter().map(|&x| x as f32).sum::<f32>() / 6.0;
        features.push(avg_pay_delay);

        // max_pay_delay
        let max_pay_delay: f32 = pay_delays.iter().copied().max().unwrap_or(0) as f32;
        features.push(max_pay_delay);

        // months_delayed
        let months_delayed: f32 = pay_delays.iter().filter(|&&x| x > 0).count() as f32;
        features.push(months_delayed);

        // pay_trend
        let pay_trend: f32 = (tx.pay_0 - tx.pay_6) as f32;
        features.push(pay_trend);

        // avg_bill_amt
        let avg_bill_amt: f32 = bill_amts.iter().sum::<f64>() as f32 / 6.0;
        features.push(avg_bill_amt);

        // bill_trend
        let bill_trend: f32 = (tx.bill_amt1 - tx.bill_amt6) as f32;
        features.push(bill_trend);

        // bill_volatility (std dev)
        let bill_mean = avg_bill_amt as f64;
        let bill_variance: f64 = bill_amts
            .iter()
            .map(|&x| (x - bill_mean).powi(2))
            .sum::<f64>()
            / 6.0;
        let bill_volatility: f32 = bill_variance.sqrt() as f32;
        features.push(bill_volatility);

        // avg_pay_amt
        let avg_pay_amt: f32 = pay_amts.iter().sum::<f64>() as f32 / 6.0;
        features.push(avg_pay_amt);

        // total_pay_amt
        let total_pay_amt: f32 = pay_amts.iter().sum::<f64>() as f32;
        features.push(total_pay_amt);

        // utilization_ratio
        let utilization_ratio: f32 = (tx.bill_amt1 / (tx.limit_bal + 1.0)) as f32;
        features.push(utilization_ratio);

        // over_limit
        let over_limit: f32 = if tx.bill_amt1 > tx.limit_bal { 1.0 } else { 0.0 };
        features.push(over_limit);

        // pay_bill_ratio
        let pay_bill_ratio: f32 = (tx.pay_amt1 / (tx.bill_amt1.abs() + 1.0)) as f32;
        features.push(pay_bill_ratio);

        // age_group
        let age_group: f32 = match tx.age {
            0..=25 => 0.0,
            26..=35 => 1.0,
            36..=45 => 2.0,
            46..=55 => 3.0,
            _ => 4.0,
        };
        features.push(age_group);

        // Categorical features (3)
        features.push(tx.sex as f32);
        features.push(tx.education as f32);
        features.push(tx.marriage as f32);

        features
    }

    /// Get the number of features produced.
    pub fn feature_count(&self) -> usize {
        36
    }

    /// Get feature names (matching Python order).
    pub fn feature_names(&self) -> Vec<&'static str> {
        vec![
            // Original numeric (20)
            "LIMIT_BAL",
            "AGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
            // Engineered (13)
            "avg_pay_delay",
            "max_pay_delay",
            "months_delayed",
            "pay_trend",
            "avg_bill_amt",
            "bill_trend",
            "bill_volatility",
            "avg_pay_amt",
            "total_pay_amt",
            "utilization_ratio",
            "over_limit",
            "pay_bill_ratio",
            "age_group",
            // Categorical (3)
            "SEX",
            "EDUCATION",
            "MARRIAGE",
        ]
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::new();
        let tx = Transaction::new("test_001".to_string(), 50000.0, 35);

        let features = extractor.extract(&tx);

        assert_eq!(features.len(), extractor.feature_count());
        assert_eq!(features[0], 50000.0); // limit_bal
        assert_eq!(features[1], 35.0); // age
    }

    #[test]
    fn test_feature_count() {
        let extractor = FeatureExtractor::new();
        assert_eq!(extractor.feature_count(), 36);
        assert_eq!(extractor.feature_names().len(), 36);
    }
}
