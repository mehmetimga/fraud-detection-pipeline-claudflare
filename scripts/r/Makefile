# Credit Card Default Prediction Pipeline (R)
# Models: XGBoost, LightGBM, CatBoost, Random Forest
# Usage: make all

.PHONY: all clean preprocess train evaluate install install-conda help report analysis

# Use conda Rscript if available, otherwise system Rscript
CONDA_BASE := $(shell conda info --base 2>/dev/null)
RSCRIPT := $(if $(wildcard $(CONDA_BASE)/bin/Rscript),$(CONDA_BASE)/bin/Rscript,Rscript)

all: preprocess train evaluate

# Install using conda (recommended - includes CatBoost)
install-conda:
	conda install -y -c conda-forge r-base r-tidyverse r-caret r-xgboost r-lightgbm r-catboost r-randomforest r-proc r-rmarkdown r-knitr r-kableextra
	$(RSCRIPT) -e "install.packages('MLmetrics', repos='https://cloud.r-project.org')"

# Install using R packages (fallback - no CatBoost)
install:
	Rscript -e "install.packages(c('tidyverse', 'caret', 'xgboost', 'lightgbm', 'randomForest', 'pROC', 'MLmetrics', 'rmarkdown', 'knitr', 'kableExtra'), repos='https://cloud.r-project.org')"
	@echo "Note: CatBoost requires conda: conda install -c conda-forge r-catboost"

preprocess:
	@echo "Using R: $(RSCRIPT)"
	$(RSCRIPT) src/preprocess.R

train: preprocess
	$(RSCRIPT) src/train.R

evaluate: train
	$(RSCRIPT) src/evaluate.R

# Generate model report from existing results
report: evaluate
	@echo "Generating model_report.html..."
	$(RSCRIPT) -e "rmarkdown::render('model_report.Rmd')"
	@echo "✓ Report generated: model_report.html"

# Run full analysis in RMarkdown (trains and evaluates in one notebook)
analysis:
	@echo "Running full model analysis..."
	$(RSCRIPT) -e "rmarkdown::render('model_analysis.Rmd')"
	@echo "✓ Analysis complete: model_analysis.html"

clean:
	rm -rf results/ figure/
	rm -f *.html Rplots.pdf
	@echo "Cleaned results and output files"

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
	@echo "  make report        - Generate model_report.html"
	@echo "  make analysis      - Run full analysis notebook (model_analysis.Rmd)"
	@echo "  make clean         - Remove results and output files"
	@echo ""
	@echo "Models: XGBoost, LightGBM, CatBoost, Random Forest"
	@echo ""
	@echo "RMarkdown Files:"
	@echo "  model_report.Rmd   - Results visualization (requires make all first)"
	@echo "  model_analysis.Rmd - Complete pipeline in one notebook"
