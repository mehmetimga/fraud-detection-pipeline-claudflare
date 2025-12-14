# ============================================================
# Fraud Detection - Interactive Analysis Script
# Run this script in RStudio to explore the results
# ============================================================

# Load required libraries
library(tidyverse)
library(caret)
library(pROC)

# Set working directory to script location (if running in RStudio)
if (interactive()) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}

cat("=== FRAUD DETECTION MODEL ANALYSIS ===\n\n")

# ------------------------------------------------------------
# 1. Load Results
# ------------------------------------------------------------

cat("Loading results...\n")

# Load model comparison results
results <- read_csv("results/model_comparison.csv", show_col_types = FALSE)
thresholds <- read_csv("results/optimal_thresholds.csv", show_col_types = FALSE)

cat("\n📊 MODEL COMPARISON RESULTS:\n")
print(results)

cat("\n🎯 OPTIMAL THRESHOLDS:\n")
print(thresholds)

# ------------------------------------------------------------
# 2. Load Test Data
# ------------------------------------------------------------

cat("\n\nLoading test data...\n")

X_test <- read_csv("results/X_test.csv", show_col_types = FALSE)
y_test <- read_csv("results/y_test.csv", show_col_types = FALSE)$default

cat(sprintf("Test samples: %d\n", nrow(X_test)))
cat(sprintf("Default rate: %.1f%%\n", mean(y_test) * 100))

# ------------------------------------------------------------
# 3. Load Models (if available)
# ------------------------------------------------------------

if (file.exists("results/models.rds")) {
  cat("\nLoading trained models...\n")
  models <- readRDS("results/models.rds")
  cat(sprintf("Loaded %d models: %s\n", length(models), paste(names(models), collapse = ", ")))
}

if (file.exists("results/features.rds")) {
  features <- readRDS("results/features.rds")
  cat(sprintf("Features: %d\n", length(features)))
}

# ------------------------------------------------------------
# 4. Visualization Functions
# ------------------------------------------------------------

#' Plot model comparison bar chart
plot_model_comparison <- function(results, metric = "roc_auc") {
  results %>%
    filter(model != "ensemble") %>%
    ggplot(aes(x = reorder(model, !!sym(metric)), y = !!sym(metric), fill = model)) +
    geom_col() +
    geom_text(aes(label = sprintf("%.3f", !!sym(metric))), hjust = -0.1) +
    coord_flip() +
    labs(
      title = sprintf("Model Comparison - %s", toupper(metric)),
      x = "Model",
      y = metric
    ) +
    theme_minimal() +
    theme(legend.position = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
}

#' Plot ROC curves for all models
plot_roc_curves <- function(models, X_test, y_test, features) {
  # Get predictions
  X_matrix <- as.matrix(X_test[, features])
  
  roc_data <- list()
  
  # XGBoost
  if ("xgboost" %in% names(models)) {
    pred <- predict(models$xgboost, xgboost::xgb.DMatrix(X_matrix))
    roc_data$XGBoost <- roc(y_test, pred, quiet = TRUE)
  }
  
  # LightGBM
  if ("lightgbm" %in% names(models)) {
    pred <- predict(models$lightgbm, X_matrix)
    roc_data$LightGBM <- roc(y_test, pred, quiet = TRUE)
  }
  
  # CatBoost
  if ("catboost" %in% names(models)) {
    pool <- catboost::catboost.load_pool(X_matrix, label = NULL)
    pred <- catboost::catboost.predict(models$catboost, pool, prediction_type = "Probability")
    roc_data$CatBoost <- roc(y_test, pred, quiet = TRUE)
  }
  
  # Random Forest
  if ("random_forest" %in% names(models)) {
    pred <- predict(models$random_forest, X_test[, features], type = "prob")[, 2]
    roc_data$RandomForest <- roc(y_test, pred, quiet = TRUE)
  }
  
  # Plot
  plot(roc_data[[1]], col = 1, main = "ROC Curves - All Models")
  for (i in 2:length(roc_data)) {
    lines(roc_data[[i]], col = i)
  }
  legend("bottomright", 
         legend = paste(names(roc_data), sprintf("(AUC: %.3f)", sapply(roc_data, auc))),
         col = 1:length(roc_data), lwd = 2)
}

# ------------------------------------------------------------
# 5. Display Visualizations
# ------------------------------------------------------------

cat("\n\n📈 GENERATING VISUALIZATIONS...\n")

# Plot ROC-AUC comparison
p1 <- plot_model_comparison(results, "roc_auc")
print(p1)

# Plot F1 comparison
p2 <- plot_model_comparison(results, "f1")
print(p2)

# ------------------------------------------------------------
# 6. Summary Statistics
# ------------------------------------------------------------

cat("\n\n📋 SUMMARY STATISTICS:\n")
cat(strrep("=", 60), "\n")

# Best model by ROC-AUC
best_auc <- results %>% 
  filter(model != "ensemble") %>%
  arrange(desc(roc_auc)) %>%
  slice(1)

cat(sprintf("\n🥇 Best Model (ROC-AUC): %s\n", toupper(best_auc$model)))
cat(sprintf("   ROC-AUC: %.4f\n", best_auc$roc_auc))
cat(sprintf("   F1 Score: %.4f\n", best_auc$f1))
cat(sprintf("   Precision: %.4f\n", best_auc$precision))
cat(sprintf("   Recall: %.4f\n", best_auc$recall))

# Ensemble performance
ensemble <- results %>% filter(model == "ensemble")
if (nrow(ensemble) > 0) {
  cat(sprintf("\n🔗 Ensemble Performance:\n"))
  cat(sprintf("   ROC-AUC: %.4f\n", ensemble$roc_auc))
  cat(sprintf("   F1 Score: %.4f\n", ensemble$f1))
  cat(sprintf("   Accuracy: %.1f%%\n", ensemble$accuracy * 100))
}

cat("\n✅ Analysis complete! Use the functions above to explore further.\n")
cat("\nAvailable functions:\n")
cat("  - plot_model_comparison(results, 'roc_auc')\n")
cat("  - plot_model_comparison(results, 'f1')\n")
cat("  - plot_roc_curves(models, X_test, y_test, features)\n")
