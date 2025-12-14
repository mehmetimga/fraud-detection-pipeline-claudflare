# Credit Card Default Prediction Pipeline
# Usage: make all

.PHONY: all clean preprocess train evaluate

all: preprocess train evaluate

preprocess:
	python src/preprocess.py

train: preprocess
	python src/train.py

evaluate: train
	python src/evaluate.py

clean:
	rm -rf results/
	@echo "Cleaned results folder"

