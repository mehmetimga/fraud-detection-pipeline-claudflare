# Credit Card Default Prediction Pipeline (R)
# Models: XGBoost, LightGBM, CatBoost, Random Forest
# Usage: make all

.PHONY: all clean preprocess train evaluate install install-conda help

# Use conda Rscript if available, otherwise system Rscript
CONDA_BASE := $(shell conda info --base 2>/dev/null)
RSCRIPT := $(if $(wildcard $(CONDA_BASE)/bin/Rscript),$(CONDA_BASE)/bin/Rscript,Rscript)

all: preprocess train evaluate

# Install using conda (recommended - includes CatBoost)
install-conda:
	conda install -y -c conda-forge r-base r-tidyverse r-caret r-xgboost r-lightgbm r-catboost r-randomforest r-proc
	$(RSCRIPT) -e "install.packages('MLmetrics', repos='https://cloud.r-project.org')"

# Install using R packages (fallback - no CatBoost)
install:
	Rscript -e "install.packages(c('tidyverse', 'caret', 'xgboost', 'lightgbm', 'randomForest', 'pROC', 'MLmetrics'), repos='https://cloud.r-project.org')"
	@echo "Note: CatBoost requires conda: conda install -c conda-forge r-catboost"

preprocess:
	@echo "Using R: $(RSCRIPT)"
	$(RSCRIPT) src/preprocess.R

train: preprocess
	$(RSCRIPT) src/train.R

evaluate: train
	$(RSCRIPT) src/evaluate.R

clean:
	rm -rf results/
	@echo "Cleaned results folder"

help:
	@echo "R Pipeline for Credit Card Default Prediction"
	@echo ""
	@echo "Using Rscript: $(RSCRIPT)"
	@echo ""
	@echo "Commands:"
	@echo "  make install-conda - Install packages via conda (recommended)"
	@echo "  make install       - Install packages via CRAN (no CatBoost)"
	@echo "  make all           - Run full pipeline (preprocess, train, evaluate)"
	@echo "  make preprocess    - Preprocess data"
	@echo "  make train         - Train all 4 models"
	@echo "  make evaluate      - Evaluate models on test set"
	@echo "  make clean         - Remove results folder"
	@echo ""
	@echo "Models: XGBoost, LightGBM, CatBoost, Random Forest"
