//! Performance metrics and statistics tracking for the fraud detection pipeline.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};
use tracing::info;

/// Metrics collector for pipeline performance
pub struct PipelineMetrics {
    /// Total transactions processed
    pub transactions_processed: AtomicU64,
    /// Total alerts generated
    pub alerts_generated: AtomicU64,
    /// Alerts by risk level
    alerts_by_level: RwLock<HashMap<String, u64>>,
    /// Processing times (in microseconds)
    processing_times: RwLock<Vec<u64>>,
    /// Model inference times (in microseconds)
    model_times: RwLock<HashMap<String, Vec<u64>>>,
    /// Risk score distribution buckets
    score_buckets: RwLock<[u64; 10]>,
    /// Start time for rate calculation
    start_time: Instant,
    /// Model agreement tracking (how often models agree)
    model_agreements: RwLock<Vec<f64>>,
}

impl PipelineMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            transactions_processed: AtomicU64::new(0),
            alerts_generated: AtomicU64::new(0),
            alerts_by_level: RwLock::new(HashMap::new()),
            processing_times: RwLock::new(Vec::with_capacity(1000)),
            model_times: RwLock::new(HashMap::new()),
            score_buckets: RwLock::new([0; 10]),
            start_time: Instant::now(),
            model_agreements: RwLock::new(Vec::with_capacity(1000)),
        }
    }

    /// Record a processed transaction
    pub fn record_transaction(&self, processing_time: Duration, risk_score: f64) {
        self.transactions_processed.fetch_add(1, Ordering::Relaxed);

        // Record processing time
        if let Ok(mut times) = self.processing_times.write() {
            times.push(processing_time.as_micros() as u64);
            // Keep only last 10000 for memory efficiency
            if times.len() > 10000 {
                times.drain(0..5000);
            }
        }

        // Record score bucket
        let bucket = (risk_score * 10.0).min(9.0) as usize;
        if let Ok(mut buckets) = self.score_buckets.write() {
            buckets[bucket] += 1;
        }
    }

    /// Record an alert
    pub fn record_alert(&self, risk_level: &str) {
        self.alerts_generated.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut by_level) = self.alerts_by_level.write() {
            *by_level.entry(risk_level.to_string()).or_insert(0) += 1;
        }
    }

    /// Record model inference time
    pub fn record_model_time(&self, model_name: &str, duration: Duration) {
        if let Ok(mut times) = self.model_times.write() {
            let model_times = times.entry(model_name.to_string()).or_insert_with(Vec::new);
            model_times.push(duration.as_micros() as u64);
            // Keep only last 1000 per model
            if model_times.len() > 1000 {
                model_times.drain(0..500);
            }
        }
    }

    /// Record model agreement (std dev of scores)
    pub fn record_model_agreement(&self, model_scores: &HashMap<String, f64>) {
        if model_scores.len() < 2 {
            return;
        }

        let scores: Vec<f64> = model_scores.values().copied().collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        // Agreement = 1 - std_dev (higher = more agreement)
        let agreement = 1.0 - std_dev.min(1.0);

        if let Ok(mut agreements) = self.model_agreements.write() {
            agreements.push(agreement);
            if agreements.len() > 1000 {
                agreements.drain(0..500);
            }
        }
    }

    /// Get processing time statistics
    pub fn get_processing_stats(&self) -> ProcessingStats {
        let times = self.processing_times.read().unwrap();
        if times.is_empty() {
            return ProcessingStats::default();
        }

        let mut sorted: Vec<u64> = times.clone();
        sorted.sort();

        let sum: u64 = sorted.iter().sum();
        let count = sorted.len();

        ProcessingStats {
            count: count as u64,
            mean_us: sum / count as u64,
            p50_us: sorted[count / 2],
            p95_us: sorted[(count as f64 * 0.95) as usize],
            p99_us: sorted[(count as f64 * 0.99) as usize],
            max_us: *sorted.last().unwrap_or(&0),
        }
    }

    /// Get model performance stats
    pub fn get_model_stats(&self) -> HashMap<String, ModelStats> {
        let times = self.model_times.read().unwrap();
        let mut stats = HashMap::new();

        for (model, model_times) in times.iter() {
            if model_times.is_empty() {
                continue;
            }

            let mut sorted: Vec<u64> = model_times.clone();
            sorted.sort();

            let sum: u64 = sorted.iter().sum();
            let count = sorted.len();

            stats.insert(
                model.clone(),
                ModelStats {
                    calls: count as u64,
                    mean_us: sum / count as u64,
                    p50_us: sorted[count / 2],
                    p99_us: sorted[(count as f64 * 0.99) as usize],
                },
            );
        }

        stats
    }

    /// Get average model agreement
    pub fn get_avg_agreement(&self) -> f64 {
        let agreements = self.model_agreements.read().unwrap();
        if agreements.is_empty() {
            return 0.0;
        }
        agreements.iter().sum::<f64>() / agreements.len() as f64
    }

    /// Get current throughput (transactions per second)
    pub fn get_throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.transactions_processed.load(Ordering::Relaxed) as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get score distribution
    pub fn get_score_distribution(&self) -> [u64; 10] {
        *self.score_buckets.read().unwrap()
    }

    /// Get alerts by risk level
    pub fn get_alerts_by_level(&self) -> HashMap<String, u64> {
        self.alerts_by_level.read().unwrap().clone()
    }

    /// Print summary statistics
    pub fn print_summary(&self) {
        let tx_count = self.transactions_processed.load(Ordering::Relaxed);
        let alert_count = self.alerts_generated.load(Ordering::Relaxed);
        let alert_rate = if tx_count > 0 {
            (alert_count as f64 / tx_count as f64) * 100.0
        } else {
            0.0
        };

        let processing = self.get_processing_stats();
        let throughput = self.get_throughput();
        let agreement = self.get_avg_agreement();
        let alerts_by_level = self.get_alerts_by_level();
        let score_dist = self.get_score_distribution();

        info!("╔══════════════════════════════════════════════════════════════╗");
        info!("║           FRAUD DETECTION PIPELINE - METRICS SUMMARY         ║");
        info!("╠══════════════════════════════════════════════════════════════╣");
        info!(
            "║ Transactions Processed: {:>8}  │  Throughput: {:>6.1} tx/s ║",
            tx_count, throughput
        );
        info!(
            "║ Alerts Generated:       {:>8}  │  Alert Rate: {:>6.1}%     ║",
            alert_count, alert_rate
        );
        info!("╠══════════════════════════════════════════════════════════════╣");
        info!(
            "║ Processing Time (μs): mean={:>5} p50={:>5} p95={:>5} p99={:>5} ║",
            processing.mean_us, processing.p50_us, processing.p95_us, processing.p99_us
        );
        info!(
            "║ Model Agreement: {:>5.1}% (higher = models agree more)        ║",
            agreement * 100.0
        );
        info!("╠══════════════════════════════════════════════════════════════╣");
        info!("║ Alerts by Risk Level:                                        ║");
        for (level, count) in &alerts_by_level {
            let pct = if alert_count > 0 {
                (*count as f64 / alert_count as f64) * 100.0
            } else {
                0.0
            };
            info!("║   {:10}: {:>6} ({:>5.1}%)                                ║", level, count, pct);
        }
        info!("╠══════════════════════════════════════════════════════════════╣");
        info!("║ Risk Score Distribution:                                     ║");
        let total: u64 = score_dist.iter().sum();
        for (i, &count) in score_dist.iter().enumerate() {
            let pct = if total > 0 { (count as f64 / total as f64) * 100.0 } else { 0.0 };
            let bar_len = (pct / 2.0) as usize;
            let bar: String = "█".repeat(bar_len.min(20));
            info!(
                "║   {:.1}-{:.1}: {:>6} ({:>5.1}%) {}",
                i as f64 / 10.0,
                (i + 1) as f64 / 10.0,
                count,
                pct,
                bar
            );
        }
        info!("╚══════════════════════════════════════════════════════════════╝");

        // Model-specific stats
        let model_stats = self.get_model_stats();
        if !model_stats.is_empty() {
            info!("Model Inference Times (μs):");
            for (model, stats) in &model_stats {
                info!(
                    "  {}: mean={} p50={} p99={} (calls={})",
                    model, stats.mean_us, stats.p50_us, stats.p99_us, stats.calls
                );
            }
        }
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Processing time statistics
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub count: u64,
    pub mean_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
}

/// Model-specific statistics
#[derive(Debug)]
pub struct ModelStats {
    pub calls: u64,
    pub mean_us: u64,
    pub p50_us: u64,
    pub p99_us: u64,
}

/// Real-time metrics reporter that prints periodic summaries
pub struct MetricsReporter {
    metrics: std::sync::Arc<PipelineMetrics>,
    interval_secs: u64,
}

impl MetricsReporter {
    pub fn new(metrics: std::sync::Arc<PipelineMetrics>, interval_secs: u64) -> Self {
        Self {
            metrics,
            interval_secs,
        }
    }

    /// Start the periodic reporting task
    pub async fn start(self) {
        let mut interval = tokio::time::interval(Duration::from_secs(self.interval_secs));
        loop {
            interval.tick().await;
            self.metrics.print_summary();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = PipelineMetrics::new();

        metrics.record_transaction(Duration::from_micros(100), 0.5);
        metrics.record_transaction(Duration::from_micros(200), 0.8);
        metrics.record_alert("high");
        metrics.record_alert("low");

        assert_eq!(metrics.transactions_processed.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.alerts_generated.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_model_agreement() {
        let metrics = PipelineMetrics::new();

        // High agreement (all similar)
        let mut scores = HashMap::new();
        scores.insert("model1".to_string(), 0.8);
        scores.insert("model2".to_string(), 0.82);
        scores.insert("model3".to_string(), 0.79);
        metrics.record_model_agreement(&scores);

        let agreement = metrics.get_avg_agreement();
        assert!(agreement > 0.9); // Should be high agreement
    }
}
