# Credit Risk Modeling with Explainable Machine Learning

## Project Overview

This project builds a **credit risk scoring system** to estimate the **probability of default (PD)** for loan applicants using supervised machine learning.
The focus is not just predictive performance, but **model interpretability, stability, and risk awareness**, aligning with real-world financial institutions and regulatory expectations.

The project follows a **time-aware modeling pipeline** to avoid data leakage and emphasizes explainability and model validation.

---

## Objectives

* Predict loan default probability using tabular financial data
* Compare traditional statistical models with modern ML models
* Apply explainability techniques suitable for regulated environments
* Evaluate model stability over time

---

## Dataset

**Source**: Lending Club / Home Credit Default Risk (public dataset)
**Target Variable**: Loan default (binary)
**Features**:

* Applicant financial attributes
* Loan characteristics
* Credit history indicators

Time information is preserved to simulate real deployment conditions.

---

## Methodology

### 1. Data Preparation

* Missing value handling
* Feature scaling where appropriate
* Time-based train / validation / test split
* Class imbalance handling

### 2. Models

* Logistic Regression (baseline, interpretable)
* Gradient Boosting (XGBoost / LightGBM)

### 3. Evaluation Metrics

* ROC-AUC
* Precision-Recall AUC
* KS statistic
* Calibration curves

Accuracy is intentionally avoided due to class imbalance.

---

## Explainability & Model Risk

* SHAP values for global and local explanations
* Feature importance consistency checks
* Partial dependence analysis
* Population Stability Index (PSI) for drift detection

This section is designed to mirror **model risk management** practices in banking.

---

## Results Summary

* ML models outperform baseline logistic regression on ROC-AUC
* Logistic regression remains competitive after calibration
* Key risk drivers align with financial intuition
* Model stability varies across time windows

Detailed results and plots are provided in the notebooks.

---

## Limitations

* Public dataset may not reflect full production complexity
* No macroeconomic variables included
* Label noise and reporting bias possible

These limitations are explicitly acknowledged, as required in regulated environments.

---

## Tools & Technologies

* Python
* pandas, numpy
* scikit-learn
* XGBoost / LightGBM
* SHAP
* matplotlib / seaborn

---

## Repository Structure

```
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_explainability.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
├── reports/
│   └── credit_risk_model_report.pdf
├── README.md
```

---

## Key Takeaway

This project demonstrates how machine learning can be applied responsibly to **credit risk assessment**, balancing predictive performance with transparency, stability, and business relevance.

---

## Future Work

* Incorporate macroeconomic stress scenarios
* Add challenger vs champion model framework
* Deploy as a scoring API with monitoring
