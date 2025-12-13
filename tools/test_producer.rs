//! Test Transaction Producer
//!
//! Generates and publishes test transactions to NATS for pipeline testing.

use chrono::Utc;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, warn};

/// Transaction structure matching the pipeline's expected format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    transaction_id: String,
    amount: f64,
    currency: String,
    merchant_id: String,
    merchant_category: String,
    card_hash: String,
    ip_address: String,
    device_fingerprint: String,
    timestamp: chrono::DateTime<Utc>,
    country: String,
    is_international: bool,
    email_domain: String,
    hour_of_day: u8,
    day_of_week: u8,
    is_online: bool,
    card_present: bool,
    pin_entered: bool,
    tx_count_last_hour: u32,
    tx_amount_last_hour: f64,
    tx_count_last_24h: u32,
    tx_amount_last_24h: f64,
    unique_merchants_24h: u32,
    avg_tx_amount: f64,
    failed_tx_last_hour: u32,
    distance_from_last_tx: f64,
    time_since_last_tx: u32,
    account_age_days: u32,
}

/// Transaction generator for testing
struct TransactionGenerator {
    rng: rand::rngs::ThreadRng,
    transaction_counter: u64,
}

impl TransactionGenerator {
    fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
            transaction_counter: 0,
        }
    }

    /// Generate a random legitimate transaction
    fn generate_legitimate(&mut self) -> Transaction {
        self.transaction_counter += 1;
        let now = Utc::now();

        Transaction {
            transaction_id: format!("tx_{:012}", self.transaction_counter),
            amount: self.rng.gen_range(10.0..500.0),
            currency: self.random_choice(&["USD", "EUR", "GBP", "CAD"]).to_string(),
            merchant_id: format!("merchant_{}", self.rng.gen_range(1..1000)),
            merchant_category: self
                .random_choice(&["5411", "5812", "5541", "5912", "5999"])
                .to_string(),
            card_hash: format!("card_{:08x}", self.rng.gen::<u32>()),
            ip_address: format!(
                "{}.{}.{}.{}",
                self.rng.gen_range(1..255),
                self.rng.gen_range(0..255),
                self.rng.gen_range(0..255),
                self.rng.gen_range(1..255)
            ),
            device_fingerprint: format!("fp_{:016x}", self.rng.gen::<u64>()),
            timestamp: now,
            country: self.random_choice(&["US", "UK", "CA", "DE", "FR"]).to_string(),
            is_international: self.rng.gen_bool(0.1),
            email_domain: self
                .random_choice(&["gmail.com", "yahoo.com", "outlook.com", "other"])
                .to_string(),
            hour_of_day: now.format("%H").to_string().parse().unwrap_or(12),
            day_of_week: now.format("%u").to_string().parse::<u8>().unwrap_or(1) - 1,
            is_online: self.rng.gen_bool(0.6),
            card_present: self.rng.gen_bool(0.4),
            pin_entered: self.rng.gen_bool(0.3),
            tx_count_last_hour: self.rng.gen_range(0..3),
            tx_amount_last_hour: self.rng.gen_range(0.0..300.0),
            tx_count_last_24h: self.rng.gen_range(0..10),
            tx_amount_last_24h: self.rng.gen_range(0.0..2000.0),
            unique_merchants_24h: self.rng.gen_range(0..5),
            avg_tx_amount: self.rng.gen_range(50.0..200.0),
            failed_tx_last_hour: self.rng.gen_range(0..1),
            distance_from_last_tx: self.rng.gen_range(0.0..100.0),
            time_since_last_tx: self.rng.gen_range(300..7200),
            account_age_days: self.rng.gen_range(30..1000),
        }
    }

    /// Generate a suspicious/fraudulent transaction
    fn generate_suspicious(&mut self) -> Transaction {
        self.transaction_counter += 1;
        let now = Utc::now();

        Transaction {
            transaction_id: format!("tx_{:012}", self.transaction_counter),
            amount: self.rng.gen_range(1000.0..10000.0), // High amount
            currency: self.random_choice(&["USD", "EUR"]).to_string(),
            merchant_id: format!("merchant_{}", self.rng.gen_range(1..1000)),
            merchant_category: self.random_choice(&["5999", "5912"]).to_string(), // Suspicious categories
            card_hash: format!("card_{:08x}", self.rng.gen::<u32>()),
            ip_address: format!(
                "{}.{}.{}.{}",
                self.rng.gen_range(1..255),
                self.rng.gen_range(0..255),
                self.rng.gen_range(0..255),
                self.rng.gen_range(1..255)
            ),
            device_fingerprint: format!("fp_{:016x}", self.rng.gen::<u64>()),
            timestamp: now,
            country: self
                .random_choice(&["RU", "CN", "US"])
                .to_string(), // Include high-risk countries
            is_international: true,                         // International
            email_domain: self
                .random_choice(&["tempmail.com", "gmail.com"])
                .to_string(),
            hour_of_day: self.rng.gen_range(0..6), // Night time
            day_of_week: self.rng.gen_range(0..7),
            is_online: true,            // Online
            card_present: false,        // Card not present
            pin_entered: false,         // No PIN
            tx_count_last_hour: self.rng.gen_range(5..15), // Many recent transactions
            tx_amount_last_hour: self.rng.gen_range(2000.0..10000.0),
            tx_count_last_24h: self.rng.gen_range(15..50),
            tx_amount_last_24h: self.rng.gen_range(5000.0..50000.0),
            unique_merchants_24h: self.rng.gen_range(8..20), // Many different merchants
            avg_tx_amount: self.rng.gen_range(50.0..150.0),
            failed_tx_last_hour: self.rng.gen_range(1..5), // Failed attempts
            distance_from_last_tx: self.rng.gen_range(500.0..5000.0), // Large distance
            time_since_last_tx: self.rng.gen_range(60..300), // Very short time
            account_age_days: self.rng.gen_range(1..30),  // New account
        }
    }

    fn random_choice<'a>(&mut self, choices: &[&'a str]) -> &'a str {
        choices[self.rng.gen_range(0..choices.len())]
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("test_producer=info".parse()?),
        )
        .init();

    info!("Starting Test Transaction Producer");

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let nats_url = args.get(1).map(|s| s.as_str()).unwrap_or("nats://localhost:4222");
    let subject = args.get(2).map(|s| s.as_str()).unwrap_or("transactions");
    let count: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let fraud_rate: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let delay_ms: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100);

    info!(
        nats_url = %nats_url,
        subject = %subject,
        count = count,
        fraud_rate = fraud_rate,
        delay_ms = delay_ms,
        "Configuration loaded"
    );

    // Connect to NATS
    let client = match async_nats::connect(nats_url).await {
        Ok(c) => {
            info!("Connected to NATS");
            c
        }
        Err(e) => {
            warn!(error = %e, "Failed to connect to NATS. Running in dry-run mode.");
            // Continue in dry-run mode
            return run_dry_mode(count, fraud_rate, delay_ms).await;
        }
    };

    // Generate and publish transactions
    let mut generator = TransactionGenerator::new();
    let mut rng = rand::thread_rng();

    info!("Starting to publish {} transactions...", count);

    let mut legitimate_count = 0;
    let mut suspicious_count = 0;

    for i in 0..count {
        let transaction = if rng.gen_bool(fraud_rate) {
            suspicious_count += 1;
            generator.generate_suspicious()
        } else {
            legitimate_count += 1;
            generator.generate_legitimate()
        };

        let payload = serde_json::to_vec(&transaction)?;

        client.publish(subject.to_string(), payload.into()).await?;

        if (i + 1) % 10 == 0 {
            info!(
                "Published {}/{} transactions ({} legitimate, {} suspicious)",
                i + 1,
                count,
                legitimate_count,
                suspicious_count
            );
        }

        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }

    info!(
        "Completed! Published {} transactions ({} legitimate, {} suspicious)",
        count, legitimate_count, suspicious_count
    );

    Ok(())
}

async fn run_dry_mode(count: u64, fraud_rate: f64, delay_ms: u64) -> anyhow::Result<()> {
    info!("Running in dry-run mode (no NATS connection)");

    let mut generator = TransactionGenerator::new();
    let mut rng = rand::thread_rng();

    for i in 0..count {
        let transaction = if rng.gen_bool(fraud_rate) {
            generator.generate_suspicious()
        } else {
            generator.generate_legitimate()
        };

        let json = serde_json::to_string_pretty(&transaction)?;

        if (i + 1) % 10 == 0 || i == 0 {
            info!("Sample transaction {}:\n{}", i + 1, json);
        }

        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }

    Ok(())
}

