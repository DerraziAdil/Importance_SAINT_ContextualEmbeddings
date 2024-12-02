# Predicting Employee Attrition Using Hybrid Deep Learning and Tree-Based Models: A Performance and Interpretability Analysis

## Project Overview
This repository contains the code and data for the thesis project titled **"Predicting Employee Attrition Using Hybrid Deep Learning and Tree-Based Models: A Performance and Interpretability Analysis"**. The study investigates the effectiveness of various machine learning pipelines, including standalone models (XGBoost, LightGBM, and SAINT) and hybrid pipelines integrating SAINT-generated embeddings with tree-based models, in predicting employee attrition. The analysis emphasizes predictive performance, interpretability, and computational efficiency, providing insights into the suitability of different approaches for HR analytics tasks.

## Language
Python

## Algorithms and Models
- **Tree-Based Models**: XGBoost, LightGBM
- **Deep Learning Models**: Self-Attention and Intersample Attention Transformer (SAINT)
- **Hybrid Pipelines**:
  - SAINT-XGBoost
  - SAINT-LightGBM

## Libraries and Tools
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn, LightGBM, XGBoost, PyTorch
- **Deep Learning**: PyTorch Lightning
- **Model Interpretability**: SHAP (SHapley Additive exPlanations)
- **Data Visualization**: matplotlib, seaborn
- **Hyperparameter Optimization**: Optuna

## Dataset
- **Original Dataset**: HR dataset containing employee features such as job role, satisfaction, and tenure.
- **Processed Dataset**: Custom preprocessing applied, available within this repository.

## Key Features
- **Nested Cross-Validation**: Ensures robust evaluation and prevents data leakage.
- **Model Comparison**: Evaluates predictive performance, interpretability, and computational efficiency.
- **SHAP Analysis**: Examines feature importance and provides interpretability insights for all models.

## Repository Structure
- **`data/`**: Contains processed datasets.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis and pipeline experiments.
- **`src/`**: Python scripts for model training, evaluation, and interpretability analysis.
- **`results/`**: Output files, including SHAP plots and model performance metrics.
- **`README.md`**: Project documentation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/employee-attrition-prediction.git
