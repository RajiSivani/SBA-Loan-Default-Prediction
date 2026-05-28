# SBA Loan Default Prediction

## Overview

This project builds a machine learning pipeline to predict SBA loan default risk using structured loan, business, and borrower-related data. The goal is to identify high-risk loans before default and support more informed risk decision-making.

The project focuses on classification modeling, leakage-safe preprocessing, feature engineering, encoding strategies, model tuning, threshold optimization, and explainability.

## Problem Statement

Loan default prediction is an imbalanced classification problem where the goal is not only to achieve strong model accuracy, but also to correctly identify risky loans while keeping false positives manageable.

This project answers the question:

> Can machine learning models identify SBA loans with higher default risk using structured application and business features?

## Dataset

The project uses SBA loan records with approximately **799K+ observations**. The target variable represents whether a loan defaulted or was paid in full.

The raw dataset is not included in this repository due to file size and academic/project constraints.

## Tools & Technologies

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* SHAP
* category-encoders
* matplotlib
* seaborn
* Jupyter Notebook

## Project Workflow

The project includes the following steps:

1. Data cleaning and preprocessing
2. Train/test split with leakage prevention
3. Feature engineering using business and loan-level variables
4. Encoding categorical variables using WOE encoding and target encoding
5. Model training using XGBoost
6. Hyperparameter tuning across 60 configurations
7. Model evaluation using AUCPR, macro-F1, and threshold tuning
8. SHAP-based model interpretation
9. Residual/error analysis to diagnose false positives and false negatives
10. Final scoring workflow for prediction output generation

## Model Development

The final model uses XGBoost because it handles nonlinear relationships, feature interactions, and structured tabular data effectively.

Key modeling techniques:

* Stratified train/test split
* WOE encoding
* Target encoding
* XGBoost classification
* AUCPR-focused model selection
* Threshold optimization
* SHAP explainability
* Permutation feature importance
* Residual analysis

## Results

The best model achieved:

* **Test AUCPR:** 0.563
* **Macro-F1:** 0.720 at optimized threshold
* **Model:** XGBoost
* **Tuning:** 60 model configurations

The project prioritized AUCPR because the dataset is imbalanced and default identification is more important than overall accuracy alone.

## Explainability

SHAP and permutation feature importance were used to interpret model behavior and identify the most important default-risk drivers.

The explainability step helped answer:

* Which features contributed most to predicted default risk?
* Where did the model make false-positive and false-negative errors?
* Which patterns could be improved in future feature engineering?

## Repository Structure

```text
SBA-Loan-Default-Prediction/
│
├── notebooks/
│   ├── sba_loan_default_prediction.ipynb
│   └── scoring_notebook.ipynb
│
├── artifacts/
│   ├── features.txt
│   ├── model_xgb.json
│   ├── params_rounds_threshold.json
│   ├── target_encoder.pkl
│   └── woe_encoder.pkl
│
├── outputs/
│   └── submission_final_v1.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

## My Contributions

This was completed as an academic machine learning project. My major contributions included:

* Cleaning and preparing SBA loan data for modeling
* Engineering business-ratio and loan-level features
* Applying leakage-safe preprocessing and encoding
* Building and tuning XGBoost classification models
* Evaluating model performance using AUCPR and macro-F1
* Optimizing the classification threshold
* Interpreting model behavior using SHAP and error analysis
* Preparing the scoring workflow and final prediction output

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/RajiSivani/SBA-Loan-Default-Prediction.git
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Open the main notebook.

```bash
jupyter notebook notebooks/sba_loan_default_prediction.ipynb
```

4. Run the notebook cells in order.

## Notes

The raw dataset is not included in this repository. The repository contains the modeling workflow, saved artifacts, and final output files needed to understand the project structure and machine learning approach.

## Future Improvements

Future improvements could include:

* Testing additional ensemble models
* Adding probability calibration
* Improving rare-event recall through resampling or cost-sensitive learning
* Comparing XGBoost with LightGBM or CatBoost
* Deploying the model through a simple API or Streamlit app
