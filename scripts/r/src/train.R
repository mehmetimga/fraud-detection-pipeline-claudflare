# Train models for credit card default prediction
# Models: XGBoost, LightGBM, CatBoost (optional), Random Forest (same as Rust pipeline)
# Usage: Rscript src/train.R

library(tidyverse)
library(caret)
library(xgboost)
library(lightgbm)
library(randomForest)
library(pROC)

# Try to load CatBoost (optional - harder to install)
CATBOOST_AVAILABLE <- requireNamespace("catboost", quietly = TRUE)
if (CATBOOST_AVAILABLE) {
  library(catboost)
  cat("CatBoost available\n")
} else {
  cat("CatBoost not available (optional) - skipping\n")
}

# Feature groups
NUMERIC_FEATURES <- c(
  "LIMIT_BAL", "AGE",
  "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
  "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
  "avg_bill", "avg_payment", "payment_ratio", "credit_util"
)

ORDINAL_FEATURES <- c(
  "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
  "times_late", "max_months_late"
)

CATEGORICAL_FEATURES <- c("SEX", "EDUCATION", "MARRIAGE")

# Prepare data for modeling
prepare_data <- function(X, y = NULL) {
  # Select features
  features <- c(NUMERIC_FEATURES, ORDINAL_FEATURES, CATEGORICAL_FEATURES)
  features <- features[features %in% colnames(X)]
  
  X_prep <- X %>% select(all_of(features))
  
  # Convert to matrix for XGBoost/LightGBM
  X_matrix <- as.matrix(X_prep)
  
  if (!is.null(y)) {
    return(list(X = X_matrix, y = as.numeric(y), features = features, df = X_prep))
  }
  return(list(X = X_matrix, features = features, df = X_prep))
}

# Train XGBoost
train_xgboost <- function(X_train, y_train, X_val = NULL, y_val = NULL) {
  cat("\n--- Training XGBoost ---\n")
  
  # Calculate scale_pos_weight for class imbalance
  scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)
  cat("Scale pos weight:", round(scale_pos_weight, 2), "\n")
  
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight
  )
  
  watchlist <- list(train = dtrain)
  if (!is.null(X_val)) {
    dval <- xgb.DMatrix(data = X_val, label = y_val)
    watchlist <- list(train = dtrain, val = dval)
  }
  
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 200,
    watchlist = watchlist,
    early_stopping_rounds = 50,
    verbose = 1,
    print_every_n = 50
  )
  
  cat("✓ XGBoost trained (", model$best_iteration, " rounds)\n", sep = "")
  return(model)
}

# Train LightGBM
train_lightgbm <- function(X_train, y_train, X_val = NULL, y_val = NULL) {
  cat("\n--- Training LightGBM ---\n")
  
  dtrain <- lgb.Dataset(data = X_train, label = y_train)
  
  params <- list(
    objective = "binary",
    metric = "auc",
    learning_rate = 0.1,
    max_depth = 6,
    num_leaves = 31,
    is_unbalance = TRUE,
    verbose = -1
  )
  
  valids <- list(train = dtrain)
  if (!is.null(X_val)) {
    dval <- lgb.Dataset(data = X_val, label = y_val, reference = dtrain)
    valids <- list(train = dtrain, val = dval)
  }
  
  model <- lgb.train(
    params = params,
    data = dtrain,
    nrounds = 200,
    valids = valids,
    early_stopping_rounds = 50,
    verbose = 1
  )
  
  cat("✓ LightGBM trained (", model$best_iter, " rounds)\n", sep = "")
  return(model)
}

# Train CatBoost
train_catboost <- function(X_train, y_train, X_val = NULL, y_val = NULL, feature_names = NULL) {
  cat("\n--- Training CatBoost ---\n")
  
  # Convert feature_names to list (CatBoost requirement)
  feat_list <- if (!is.null(feature_names)) as.list(feature_names) else NULL
  
  train_pool <- catboost.load_pool(
    data = X_train,
    label = y_train,
    feature_names = feat_list
  )
  
  params <- list(
    loss_function = "Logloss",
    eval_metric = "AUC",
    iterations = 200,
    learning_rate = 0.1,
    depth = 6,
    auto_class_weights = "Balanced",
    verbose = 50
  )
  
  if (!is.null(X_val)) {
    val_pool <- catboost.load_pool(
      data = X_val,
      label = y_val,
      feature_names = feat_list
    )
    model <- catboost.train(train_pool, val_pool, params = params)
  } else {
    model <- catboost.train(train_pool, params = params)
  }
  
  cat("✓ CatBoost trained\n")
  return(model)
}

# Train Random Forest
train_random_forest <- function(X_train, y_train) {
  cat("\n--- Training Random Forest ---\n")
  
  # Convert to data frame for randomForest
  df <- as.data.frame(X_train)
  df$target <- as.factor(y_train)
  
  model <- randomForest(
    target ~ .,
    data = df,
    ntree = 200,
    mtry = floor(sqrt(ncol(X_train))),
    classwt = c("0" = 1, "1" = sum(y_train == 0) / sum(y_train == 1)),
    importance = TRUE
  )
  
  cat("✓ Random Forest trained (", model$ntree, " trees)\n", sep = "")
  return(model)
}

# Main training function
main <- function() {
  cat("=== MODEL TRAINING ===\n")
  cat("Models: XGBoost, LightGBM, CatBoost, Random Forest\n\n")
  
  # Load data
  cat("Loading processed data...\n")
  X_train <- read_csv("results/X_train.csv", show_col_types = FALSE)
  y_train <- read_csv("results/y_train.csv", show_col_types = FALSE)$default
  
  cat("Train samples:", nrow(X_train), "\n")
  cat("Default rate:", round(mean(y_train) * 100, 1), "%\n")
  
  # Prepare data
  prep <- prepare_data(X_train, y_train)
  X_mat <- prep$X
  y_vec <- prep$y
  features <- prep$features
  
  cat("Features:", length(features), "\n\n")
  
  # Split for validation
  set.seed(42)
  val_idx <- sample(1:nrow(X_mat), size = floor(0.1 * nrow(X_mat)))
  X_train_split <- X_mat[-val_idx, ]
  y_train_split <- y_vec[-val_idx]
  X_val <- X_mat[val_idx, ]
  y_val <- y_vec[val_idx]
  
  cat("Training:", nrow(X_train_split), "samples\n")
  cat("Validation:", nrow(X_val), "samples\n")
  
  # Train models
  models <- list()
  
  # 1. XGBoost
  models$xgboost <- train_xgboost(X_train_split, y_train_split, X_val, y_val)
  
  # 2. LightGBM
  models$lightgbm <- train_lightgbm(X_train_split, y_train_split, X_val, y_val)
  
  # 3. CatBoost (if available)
  if (CATBOOST_AVAILABLE) {
    models$catboost <- train_catboost(X_train_split, y_train_split, X_val, y_val, features)
  } else {
    cat("\n--- Skipping CatBoost (not installed) ---\n")
  }
  
  # 4. Random Forest
  models$random_forest <- train_random_forest(X_train_split, y_train_split)
  
  # Save models
  cat("\n--- Saving Models ---\n")
  saveRDS(models, "results/models.rds")
  saveRDS(features, "results/features.rds")
  
  cat("\n✓ All models saved to results/models.rds\n")
  cat("✓ Feature names saved to results/features.rds\n")
}

# Run if executed directly
if (!interactive()) {
  main()
}

