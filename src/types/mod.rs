//! Type definitions for the fraud detection pipeline

pub mod alert;
pub mod transaction;

pub use alert::{FraudAlert, RiskLevel};
pub use transaction::Transaction;

