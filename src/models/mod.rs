//! ML model inference components

pub mod aggregator;
pub mod inference;
pub mod loader;

pub use aggregator::ScoreAggregator;
pub use inference::InferenceEngine;
pub use loader::ModelLoader;

