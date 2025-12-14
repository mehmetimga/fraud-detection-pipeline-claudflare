# Preprocess data for credit card default prediction
# Usage: Rscript src/preprocess.R

library(tidyverse)
library(caret)

# Load and clean the dataset
load_data <- function(path = "data/UCI_Credit_Card.csv") {
  cat("Loading data from", path, "...\n")
  
  df <- read_csv(path, show_col_types = FALSE)
  
  # Rename columns
  df <- df %>%
    select(-ID) %>%
    rename(
      default = `default.payment.next.month`,
      PAY_1 = PAY_0
    )
  
  # Clean education and marriage values
  df <- df %>%
    mutate(
      EDUCATION = ifelse(EDUCATION %in% c(0, 5, 6), 4, EDUCATION),
      MARRIAGE = ifelse(MARRIAGE == 0, 3, MARRIAGE)
    )
  
  cat("Loaded", nrow(df), "records with", ncol(df), "columns\n")
  return(df)
}

# Add engineered features
add_features <- function(df) {
  pay_cols <- c("PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
  bill_cols <- c("BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6")
  amt_cols <- c("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")
  
  df <- df %>%
    mutate(
      # Payment behavior features
      times_late = rowSums(select(., all_of(pay_cols)) > 0),
      max_months_late = pmax(PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6),
      
      # Bill amount features
      avg_bill = rowMeans(select(., all_of(bill_cols))),
      
      # Payment amount features
      avg_payment = rowMeans(select(., all_of(amt_cols))),
      
      # Ratio features
      payment_ratio = avg_payment / (abs(avg_bill) + 1),
      credit_util = BILL_AMT1 / (LIMIT_BAL + 1)
    )
  
  return(df)
}

# Main function
main <- function() {
  cat("=== PREPROCESSING ===\n\n")
  
  # Load data
  df <- load_data()
  
  # Split data
  cat("\nSplitting data (80/20)...\n")
  set.seed(42)
  
  train_index <- createDataPartition(df$default, p = 0.8, list = FALSE)
  train_df <- df[train_index, ]
  test_df <- df[-train_index, ]
  
  cat("Train:", nrow(train_df), "records\n")
  cat("Test:", nrow(test_df), "records\n")
  cat("Default rate (train):", round(mean(train_df$default) * 100, 1), "%\n")
  cat("Default rate (test):", round(mean(test_df$default) * 100, 1), "%\n")
  
  # Add features
  cat("\nAdding engineered features...\n")
  train_df <- add_features(train_df)
  test_df <- add_features(test_df)
  
  # Separate X and y
  X_train <- train_df %>% select(-default)
  y_train <- train_df$default
  X_test <- test_df %>% select(-default)
  y_test <- test_df$default
  
  # Save processed data
  cat("\nSaving processed data...\n")
  dir.create("results", showWarnings = FALSE)
  
  write_csv(X_train, "results/X_train.csv")
  write_csv(X_test, "results/X_test.csv")
  write_csv(data.frame(default = y_train), "results/y_train.csv")
  write_csv(data.frame(default = y_test), "results/y_test.csv")
  
  cat("\n✓ Saved: X_train (", nrow(X_train), "x", ncol(X_train), "), ",
      "X_test (", nrow(X_test), "x", ncol(X_test), ")\n", sep = "")
}

# Run if executed directly
if (!interactive()) {
  main()
}

