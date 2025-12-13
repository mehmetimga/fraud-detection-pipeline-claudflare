//! NATS message consumer for incoming transactions

use anyhow::Result;
use async_nats::{Client, Subscriber};
use tracing::info;

/// Consumer for receiving transactions from NATS
pub struct TransactionConsumer {
    client: Client,
    subject: String,
}

impl TransactionConsumer {
    /// Create a new transaction consumer
    pub fn new(client: Client, subject: &str) -> Self {
        Self {
            client,
            subject: subject.to_string(),
        }
    }

    /// Subscribe to the transaction subject
    pub async fn subscribe(&self) -> Result<Subscriber> {
        let subscriber = self.client.subscribe(self.subject.clone()).await?;
        info!(subject = %self.subject, "Subscribed to transaction subject");
        Ok(subscriber)
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

