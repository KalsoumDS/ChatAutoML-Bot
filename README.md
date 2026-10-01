# ChatAutoML — Conversational No-Code AutoML Platform

> Conversational AutoML assistant enabling non-technical teams (NGOs, SMEs) to transform raw data into optimized predictive models through natural language.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-blueviolet)](https://optuna.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://lbx6ryyhzigbsh3d5uwiyg.streamlit.app/)

**Live application:** https://lbx6ryyhzigbsh3d5uwiyg.streamlit.app/

---

## Overview

Small organizations (NGOs, SMEs, associations) rarely have dedicated data scientists to analyze their data and build predictive models. ChatAutoML acts as a virtual data scientist accessible through natural language.

---

## Features

- **No-Code Ingestion** — Upload a CSV file, preprocessing is fully automatic (missing values, encoding, scaling).
- **Bayesian Hyperparameter Optimization** — Optuna tunes 8 ML algorithms (Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression, SVM, KNN, Decision Tree) within 60 seconds.
- **Automated Reporting** — Export with performance metrics, confusion matrix, feature importance, and plain-language recommendations.
- **Conversational Interface** — Users interact in natural language to drive the ML pipeline without writing code.

---

## Architecture

```
Input CSV
    |
Automatic Preprocessing (imputation, encoding, scaling)
    |
Algorithm Benchmark (8 models, cross-validation)
    |
Optuna Bayesian Optimization (best model tuned)
    |
Explainability (SHAP / feature importance)
    |
Streamlit Interface + Report
```

---

## Performance

- 8 ML algorithms benchmarked per run
- Hyperparameter search via Optuna TPE sampler
- Full pipeline (ingestion to optimized model) under 60 seconds

---

## Installation

```bash
git clone https://github.com/KalsoumDS/ChatAutoML-Bot.git
cd ChatAutoML-Bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Technologies

- Python 3.10+, Optuna, Scikit-learn, XGBoost, LightGBM
- Streamlit, SHAP, pandas, NumPy

---

## Author

Oumou Kaltoum Sall — Data Scientist & ML Engineer  
[Portfolio](https://luxury-sunshine-073627.netlify.app) · [LinkedIn](https://linkedin.com/in/oumou-kaltoum-sall)
