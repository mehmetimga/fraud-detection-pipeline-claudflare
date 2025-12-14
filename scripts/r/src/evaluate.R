# Evaluate models on test set
# Usage: Rscript src/evaluate.R

library(tidyverse)
library(caret)
library(xgboost)
library(lightgbm)
library(randomForest)
library(pROC)
library(MLmetrics)

# Try to load CatBoost (optional)
CATBOOST_AVAILABLE <- requireNamespace("catboost", quietly = TRUE)
if (CATBOOST_AVAILABLE) {
  library(catboost)
}

# Prepare data (same as train.R)
prepare_data <- function(X, features) {
  features_present <- features[features %in% colnames(X)]
  X_prep <- X %>% select(all_of(features_present))
  X_matrix <- as.matrix(X_prep)
  return(X_matrix)
}

# Get predictions from each model
get_predictions <- function(models, X_test, features) {
  predictions <- list()
  
  # XGBoost
  if ("xgboost" %in% names(models)) {
    cat("  XGBoost predictions...\n")
    dtest <- xgb.DMatrix(data = X_test)
    predictions$xgboost <- predict(models$xgboost, dtest)
  }
  
  # LightGBM
  if ("lightgbm" %in% names(models)) {
    cat("  LightGBM predictions...\n")
    predictions$lightgbm <- predict(models$lightgbm, X_test)
  }
  
  # CatBoost
  if ("catboost" %in% names(models) && CATBOOST_AVAILABLE) {
    cat("  CatBoost predictions...\n")
    test_pool <- catboost.load_pool(data = X_test, feature_names = as.list(features))
    predictions$catboost <- catboost.predict(models$catboost, test_pool, prediction_type = "Probability")
  }
  
  # Random Forest
  if ("random_forest" %in% names(models)) {
    cat("  Random Forest predictions...\n")
    df_test <- as.data.frame(X_test)
    predictions$random_forest <- predict(models$random_forest, df_test, type = "prob")[, "1"]
  }
  
  return(predictions)
}

# Calculate metrics for a model
calculate_metrics <- function(y_true, y_proba, model_name, threshold = 0.5) {
  y_pred <- ifelse(y_proba >= threshold, 1, 0)
  
  # Confusion matrix
  cm <- confusionMatrix(as.factor(y_pred), as.factor(y_true), positive = "1")
  
  # Metrics
  metrics <- data.frame(
    model = model_name,
    accuracy = cm$overall["Accuracy"],
    precision = cm$byClass["Precision"],
    recall = cm$byClass["Recall"],
    f1 = cm$byClass["F1"],
    roc_auc = as.numeric(auc(roc(y_true, y_proba, quiet = TRUE))),
    fraud_detection_rate = cm$byClass["Recall"],
    false_alarm_rate = 1 - cm$byClass["Specificity"]
  )
  
  rownames(metrics) <- NULL
  return(metrics)
}

# Find optimal threshold
find_optimal_threshold <- function(y_true, y_proba) {
  thresholds <- seq(0.1, 0.9, by = 0.01)
  best_f1 <- 0
  best_threshold <- 0.5
  
  for (thresh in thresholds) {
    y_pred <- ifelse(y_proba >= thresh, 1, 0)
    f1 <- F1_Score(y_true, y_pred, positive = "1")
    if (!is.na(f1) && f1 > best_f1) {
      best_f1 <- f1
      best_threshold <- thresh
    }
  }
  
  return(list(threshold = best_threshold, f1 = best_f1))
}

# Main function
main <- function() {
  cat("=== MODEL EVALUATION ===\n\n")
  
  # Load data
  cat("Loading test data...\n")
  X_test <- read_csv("results/X_test.csv", show_col_types = FALSE)
  y_test <- read_csv("results/y_test.csv", show_col_types = FALSE)$default
  
  cat("Test samples:", nrow(X_test), "\n")
  cat("Default rate:", round(mean(y_test) * 100, 1), "%\n\n")
  
  # Load models
  cat("Loading models...\n")
  models <- readRDS("results/models.rds")
  features <- readRDS("results/features.rds")
  
  # Prepare test data
  X_test_mat <- prepare_data(X_test, features)
  
  # Get predictions
  cat("\nGetting predictions...\n")
  predictions <- get_predictions(models, X_test_mat, features)
  
  # Calculate metrics for each model
  cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
  cat("TEST SET RESULTS\n")
  cat(paste(rep("=", 60), collapse = ""), "\n\n")
  
  all_metrics <- data.frame()
  optimal_thresholds <- list()
  
  for (model_name in names(predictions)) {
    y_proba <- predictions[[model_name]]
    
    # Find optimal threshold
    opt <- find_optimal_threshold(y_test, y_proba)
    optimal_thresholds[[model_name]] <- opt$threshold
    
    # Calculate metrics at optimal threshold
    metrics <- calculate_metrics(y_test, y_proba, model_name, opt$threshold)
    all_metrics <- rbind(all_metrics, metrics)
    
    cat("--- ", toupper(model_name), " ---\n", sep = "")
    cat("  Optimal threshold:", round(opt$threshold, 3), "\n")
    cat("  Accuracy:", round(metrics$accuracy, 4), "\n")
    cat("  Precision:", round(metrics$precision, 4), "\n")
    cat("  Recall:", round(metrics$recall, 4), "\n")
    cat("  F1 Score:", round(metrics$f1, 4), "\n")
    cat("  ROC-AUC:", round(metrics$roc_auc, 4), "\n")
    cat("\n")
  }
  
  # Rank by ROC-AUC
  all_metrics <- all_metrics %>% arrange(desc(roc_auc))
  
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("MODEL RANKING (by ROC-AUC)\n")
  cat(paste(rep("=", 60), collapse = ""), "\n\n")
  
  for (i in 1:nrow(all_metrics)) {
    cat("#", i, " ", all_metrics$model[i], "\n", sep = "")
    cat("   ROC-AUC:", round(all_metrics$roc_auc[i], 4), "\n")
    cat("   F1:", round(all_metrics$f1[i], 4), "\n\n")
  }
  
  # Ensemble prediction (average)
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("ENSEMBLE (Average of all models)\n")
  cat(paste(rep("=", 60), collapse = ""), "\n\n")
  
  # Average all available predictions
  pred_matrix <- do.call(cbind, predictions)
  ensemble_proba <- rowMeans(pred_matrix)
  
  ensemble_opt <- find_optimal_threshold(y_test, ensemble_proba)
  ensemble_metrics <- calculate_metrics(y_test, ensemble_proba, "ensemble", ensemble_opt$threshold)
  
  cat("  Optimal threshold:", round(ensemble_opt$threshold, 3), "\n")
  cat("  Accuracy:", round(ensemble_metrics$accuracy, 4), "\n")
  cat("  Precision:", round(ensemble_metrics$precision, 4), "\n")
  cat("  Recall:", round(ensemble_metrics$recall, 4), "\n")
  cat("  F1 Score:", round(ensemble_metrics$f1, 4), "\n")
  cat("  ROC-AUC:", round(ensemble_metrics$roc_auc, 4), "\n")
  
  # Save results
  cat("\n--- Saving Results ---\n")
  
  # Add ensemble to metrics
  all_metrics <- rbind(all_metrics, ensemble_metrics)
  
  write_csv(all_metrics, "results/model_comparison.csv")
  
  # Save optimal thresholds
  thresholds_df <- data.frame(
    model = names(optimal_thresholds),
    optimal_threshold = unlist(optimal_thresholds)
  )
  thresholds_df <- rbind(thresholds_df, data.frame(
    model = "ensemble",
    optimal_threshold = ensemble_opt$threshold
  ))
  write_csv(thresholds_df, "results/optimal_thresholds.csv")
  
  cat("\n✓ Results saved to results/model_comparison.csv\n")
  cat("✓ Thresholds saved to results/optimal_thresholds.csv\n")
}

# Run if executed directly
if (!interactive()) {
  main()
}

