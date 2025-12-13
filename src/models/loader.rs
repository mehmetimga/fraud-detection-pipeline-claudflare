//! ONNX model loader

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;
use tracing::info;

/// Loaded ONNX model with metadata
pub struct LoadedModel {
    /// Model name
    pub name: String,
    /// ONNX Runtime session
    pub session: Session,
    /// Input name for the model
    pub input_name: String,
    /// Output name for probabilities
    pub output_name: String,
}

/// Loader for ONNX models
pub struct ModelLoader {
    /// Number of threads for ONNX inference
    onnx_threads: usize,
}

impl ModelLoader {
    /// Create a new model loader with default settings (1 thread)
    pub fn new() -> Result<Self> {
        Self::with_threads(1)
    }

    /// Create a new model loader with specified number of threads
    pub fn with_threads(onnx_threads: usize) -> Result<Self> {
        // Initialize ONNX Runtime
        ort::init().commit()?;
        info!(onnx_threads = onnx_threads, "ONNX Runtime initialized");
        Ok(Self { onnx_threads })
    }

    /// Load a single ONNX model from file
    pub fn load_model<P: AsRef<Path>>(&self, path: P, name: &str) -> Result<LoadedModel> {
        let path = path.as_ref();

        info!(model = %name, path = %path.display(), threads = self.onnx_threads, "Loading ONNX model");

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.onnx_threads)?
            .commit_from_file(path)
            .context(format!("Failed to load model from {:?}", path))?;

        // Get input/output names
        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "float_input".to_string());

        let output_name = session
            .outputs
            .iter()
            .find(|o| o.name.contains("prob") || o.name.contains("output"))
            .map(|o| o.name.clone())
            .unwrap_or_else(|| {
                session
                    .outputs
                    .last()
                    .map(|o| o.name.clone())
                    .unwrap_or_else(|| "probabilities".to_string())
            });

        info!(
            model = %name,
            input = %input_name,
            output = %output_name,
            "Model loaded successfully"
        );

        Ok(LoadedModel {
            name: name.to_string(),
            session,
            input_name,
            output_name,
        })
    }

    /// Load all models from a directory
    pub fn load_all_models<P: AsRef<Path>>(&self, models_dir: P) -> Result<Vec<LoadedModel>> {
        let models_dir = models_dir.as_ref();
        let mut models = Vec::new();

        let model_files = [
            ("catboost", "catboost.onnx"),
            ("xgboost", "xgboost.onnx"),
            ("lightgbm", "lightgbm.onnx"),
            ("random_forest", "random_forest.onnx"),
            ("isolation_forest", "isolation_forest.onnx"),
        ];

        for (name, filename) in &model_files {
            let path = models_dir.join(filename);
            if path.exists() {
                match self.load_model(&path, name) {
                    Ok(model) => models.push(model),
                    Err(e) => {
                        tracing::warn!(model = %name, error = %e, "Failed to load model, skipping");
                    }
                }
            } else {
                tracing::warn!(model = %name, path = %path.display(), "Model file not found");
            }
        }

        if models.is_empty() {
            anyhow::bail!("No models loaded from {}", models_dir.display());
        }

        info!(
            count = models.len(),
            "Loaded {} models from {}",
            models.len(),
            models_dir.display()
        );

        Ok(models)
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self { onnx_threads: 1 }
    }
}

