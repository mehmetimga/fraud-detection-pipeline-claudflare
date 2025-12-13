//! Transaction data structures for credit card default detection

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a credit card transaction to be analyzed for default risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Unique transaction/record identifier
    #[serde(alias = "ID")]
    pub transaction_id: String,

    /// Credit limit
    #[serde(alias = "LIMIT_BAL")]
    pub limit_bal: f64,

    /// Sex (1 = male, 2 = female)
    #[serde(alias = "SEX")]
    pub sex: i32,

    /// Education level (1 = graduate, 2 = university, 3 = high school, 4 = others)
    #[serde(alias = "EDUCATION")]
    pub education: i32,

    /// Marital status (1 = married, 2 = single, 3 = others)
    #[serde(alias = "MARRIAGE")]
    pub marriage: i32,

    /// Age in years
    #[serde(alias = "AGE")]
    pub age: i32,

    /// Payment status month 0 (-1 = pay duly, 1 = payment delay 1 month, etc.)
    #[serde(alias = "PAY_0")]
    pub pay_0: i32,

    /// Payment status month 2
    #[serde(alias = "PAY_2")]
    pub pay_2: i32,

    /// Payment status month 3
    #[serde(alias = "PAY_3")]
    pub pay_3: i32,

    /// Payment status month 4
    #[serde(alias = "PAY_4")]
    pub pay_4: i32,

    /// Payment status month 5
    #[serde(alias = "PAY_5")]
    pub pay_5: i32,

    /// Payment status month 6
    #[serde(alias = "PAY_6")]
    pub pay_6: i32,

    /// Bill amount month 1
    #[serde(alias = "BILL_AMT1")]
    pub bill_amt1: f64,

    /// Bill amount month 2
    #[serde(alias = "BILL_AMT2")]
    pub bill_amt2: f64,

    /// Bill amount month 3
    #[serde(alias = "BILL_AMT3")]
    pub bill_amt3: f64,

    /// Bill amount month 4
    #[serde(alias = "BILL_AMT4")]
    pub bill_amt4: f64,

    /// Bill amount month 5
    #[serde(alias = "BILL_AMT5")]
    pub bill_amt5: f64,

    /// Bill amount month 6
    #[serde(alias = "BILL_AMT6")]
    pub bill_amt6: f64,

    /// Payment amount month 1
    #[serde(alias = "PAY_AMT1")]
    pub pay_amt1: f64,

    /// Payment amount month 2
    #[serde(alias = "PAY_AMT2")]
    pub pay_amt2: f64,

    /// Payment amount month 3
    #[serde(alias = "PAY_AMT3")]
    pub pay_amt3: f64,

    /// Payment amount month 4
    #[serde(alias = "PAY_AMT4")]
    pub pay_amt4: f64,

    /// Payment amount month 5
    #[serde(alias = "PAY_AMT5")]
    pub pay_amt5: f64,

    /// Payment amount month 6
    #[serde(alias = "PAY_AMT6")]
    pub pay_amt6: f64,

    /// Timestamp (optional, for real-time processing)
    #[serde(default = "Utc::now")]
    pub timestamp: DateTime<Utc>,
}

impl Transaction {
    /// Create a new transaction with required fields
    pub fn new(transaction_id: String, limit_bal: f64, age: i32) -> Self {
        Self {
            transaction_id,
            limit_bal,
            sex: 1,
            education: 1,
            marriage: 1,
            age,
            pay_0: 0,
            pay_2: 0,
            pay_3: 0,
            pay_4: 0,
            pay_5: 0,
            pay_6: 0,
            bill_amt1: 0.0,
            bill_amt2: 0.0,
            bill_amt3: 0.0,
            bill_amt4: 0.0,
            bill_amt5: 0.0,
            bill_amt6: 0.0,
            pay_amt1: 0.0,
            pay_amt2: 0.0,
            pay_amt3: 0.0,
            pay_amt4: 0.0,
            pay_amt5: 0.0,
            pay_amt6: 0.0,
            timestamp: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_serialization() {
        let tx = Transaction::new("tx_123".to_string(), 50000.0, 30);

        let json = serde_json::to_string(&tx).unwrap();
        let deserialized: Transaction = serde_json::from_str(&json).unwrap();

        assert_eq!(tx.transaction_id, deserialized.transaction_id);
        assert_eq!(tx.limit_bal, deserialized.limit_bal);
        assert_eq!(tx.age, deserialized.age);
    }
}
