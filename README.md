# Importance of Contextual Embeddings in Tabular Data: Combining SAINT Contextual Embeddings and Tree-Based Models for Employee Attrition Prediction

## Project Overview
This repository contains the code and data for the thesis project titled **"Importance of Contextual Embeddings in Tabular Data: Combining SAINT Contextual Embeddings and Tree-Based Models for Employee Attrition Prediction"**. The study investigates the impact of using SAINT (Self-Attention and Intersample Attention Transformer)-generated contextual embeddings in combination with tree-based models like XGBoost and LightGBM for predicting employee attrition. By comparing standalone and hybrid models, the analysis focuses on predictive performance, generalizability, and interpretability.

## Language
Python

## Algorithms and Models
- **Tree-Based Models**: XGBoost, LightGBM
- **Deep Learning Models**: SAINT (Self-Attention and Intersample Attention Transformer)
- **Hybrid Pipelines**:
  - SAINT-XGBoost
  - SAINT-LightGBM

## Libraries and Tools
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch
- **Model Interpretability**: SHAP (SHapley Additive exPlanations)
- **Data Visualization**: matplotlib, seaborn

## Dataset
- **Source**: Employee Attrition Dataset, publicly available on Kaggle.
  - Dataset link: [Employee Attrition Dataset on Kaggle](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
- The dataset includes features such as age, gender, job role, marital status, and attrition status.

## Key Features
- **Nested Cross-Validation**: A 5-fold stratified nested cross-validation approach was employed, with 3 inner folds for hyperparameter tuning.
- **Model Comparison**: Comprehensive evaluation of standalone and hybrid models using metrics such as ROC-AUC, precision, recall, and F1-score.
- **SHAP Analysis**: Feature importance analysis for interpretability using SHAP values.
- **Error Analysis**: Examination of confusion matrices to understand error patterns.

## Repository Structure
- **`Data/`**: Contains the datasets used in the experiments.
- **`Models/`**: Contains the pipelines used in the experiments.
- **`EDA/`**: Contains the code that was used to perform EDA.
- **`DeLong/`**: Contains the code for the DeLong statistical test. 
- **`README.md`**: Project documentation.

