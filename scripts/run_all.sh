#!/bin/bash
set -e

echo "=== [1/2] Starting Exploratory Data Analysis Pipeline ==="
python -m src.eda

echo "=== [2/2] Starting Model Training & Explainability Pipeline ==="
python -m src.models

echo "=== Pipeline Completed Successfully! ==="
