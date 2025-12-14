# Install required R packages for the fraud detection pipeline
# Usage: Rscript install_packages.R

cat("=== Installing R Packages for Fraud Detection Pipeline ===\n\n")

# Required packages
packages <- c(
  "tidyverse",      # Data manipulation
  "caret",          # ML utilities
  "xgboost",        # XGBoost model
  "lightgbm",       # LightGBM model
  "randomForest",   # Random Forest model
  "pROC",           # ROC curves
  "MLmetrics"       # ML metrics
)

# CatBoost requires special installation
catboost_required <- TRUE

# Install CRAN packages
cat("Installing CRAN packages...\n")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("  Installing", pkg, "...\n")
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  } else {
    cat("  ✓", pkg, "already installed\n")
  }
}

# Install CatBoost (requires devtools or remotes)
if (catboost_required) {
  cat("\nInstalling CatBoost...\n")
  
  if (!require("catboost", quietly = TRUE)) {
    cat("  CatBoost not found. Installing...\n")
    
    # Try installing from GitHub
    if (!require("devtools", quietly = TRUE)) {
      install.packages("devtools", repos = "https://cloud.r-project.org", quiet = TRUE)
    }
    
    tryCatch({
      devtools::install_github("catboost/catboost", subdir = "catboost/R-package", quiet = TRUE)
      cat("  ✓ CatBoost installed from GitHub\n")
    }, error = function(e) {
      cat("  ⚠ CatBoost installation failed. You may need to install it manually.\n")
      cat("  See: https://catboost.ai/docs/installation/r-installation-binary-installation\n")
    })
  } else {
    cat("  ✓ CatBoost already installed\n")
  }
}

# Verify installations
cat("\n=== Verification ===\n")
all_ok <- TRUE

for (pkg in packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "- OK\n")
  } else {
    cat("✗", pkg, "- FAILED\n")
    all_ok <- FALSE
  }
}

if (require("catboost", quietly = TRUE)) {
  cat("✓ catboost - OK\n")
} else {
  cat("✗ catboost - FAILED (optional, pipeline can run without it)\n")
}

if (all_ok) {
  cat("\n=== All packages installed successfully! ===\n")
} else {
  cat("\n=== Some packages failed. Please install them manually. ===\n")
}

