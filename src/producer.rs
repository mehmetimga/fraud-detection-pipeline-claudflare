//! NATS message producer for fraud alerts

use crate::types::alert::FraudAlert;
use anyhow::Result;
use async_nats::Client;
use tracing::{debug, error};

/// Producer for publishing fraud alerts to NATS
#[derive(Clone)]
pub struct AlertProducer {
    client: Client,
    subject: String,
}

impl AlertProducer {
    /// Create a new alert producer
    pub fn new(client: Client, subject: &str) -> Self {
        Self {
            client,
            subject: subject.to_string(),
        }
    }

    /// Publish a fraud alert
    pub async fn publish(&self, alert: &FraudAlert) -> Result<()> {
        let payload = serde_json::to_vec(alert)?;

        self.client
            .publish(self.subject.clone(), payload.into())
            .await?;

        debug!(
            alert_id = %alert.alert_id,
            transaction_id = %alert.transaction_id,
            risk_score = alert.risk_score,
            "Published fraud alert"
        );

        Ok(())
    }

    /// Publish multiple alerts in batch
    pub async fn publish_batch(&self, alerts: &[FraudAlert]) -> Result<()> {
        for alert in alerts {
            if let Err(e) = self.publish(alert).await {
                error!(
                    alert_id = %alert.alert_id,
                    error = %e,
                    "Failed to publish alert"
                );
            }
        }
        Ok(())
    }

    /// Get the subject name
    pub fn subject(&self) -> &str {
        &self.subject
    }
}

#[cfg(test)]
mod tests {
    // Integration tests would require a running NATS server
}

