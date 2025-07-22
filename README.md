# End-to-End ML Pipeline-Production: Hospital No-Show Prediction System

<!-- Badges -->
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Framework](https://img.shields.io/badge/Framework-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![Data Science](https://img.shields.io/badge/Data%20Science-Pandas-150458.svg)](https://pandas.pydata.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-InProgress%20-brightgreen.svg)]()

> **Note:** Respecting NDA compliance - showcasing technical implementation and methodology only.

---

## Problem Statement

Missed appointments cost hospitals money, overload staff, and waste resources.  
This project predicts the likelihood of a patient **not showing up** for their appointment using historical booking, demographic, and temporal features — enabling proactive rescheduling and better resource allocation.

---

## Approach & Features

- Cleaned **malformed & duplicate rows**, filtered invalid `AppointmentIds`
- Engineered **temporal features** (e.g., days until appointment, booking delay)
- Extracted:
  - Day-of-week effects
  - Age groups
  - Lead time buckets
  - Branch-wise breakdowns
  - No-show frequency by patient

---

## Models Trained

- `Logistic Regression`: baseline
- `Random Forest`: interpretable and stable
- `XGBoost`: tuned using `scale_pos_weight` for class imbalance

---

## Metrics (Best Model: XGBoost)

| Metric               | Value      |
|----------------------|------------|
| **Recall (No-Show)** | ~90%       |
| Precision            | ~60%       |
| F1 Score             | ~60%       |
| AUC-ROC              | 0.69+      |
| Log Loss             | 0.62       |
| MCC                  | 0.26       |

> **Note:** Optimized for **recall on the No-Show class**, to avoid missing at-risk cases even if precision drops.

---

## Project Structure

```
PredictML-Production/
│
├── data/                      # Anonymized Data
├── notebooks/                 # EDA and model development experiments
├── src/
│   ├── preprocessing.py        # Handles all data cleaning & feature engineering
│   ├── train_model.py          # Handles model training + evaluation logic
│   ├── inference.py            # For generating predictions from new data
│   ├── utils/                  # Utility functions (logger, metrics, encoders)
│
├── config/
│   ├── config.json             # Model parameters, input/output paths, feature flags
│
├── app/
│   └── flask_app.py            # Flask-based API for local inference (optional)
│   
├── Dockerfile                  # For full environment reproducibility
├── Makefile                    # CLI entry points to run training, inference, etc.
├── requirements.txt
├── README.md                   # You're here
```

---

## Tech Stack

| Layer       | Tools Used                                  |
|------------|----------------------------------------------|
| Data        | Pandas, NumPy, Seaborn, Matplotlib           |
| ML Models   | scikit-learn, XGBoost                        |
| Explainability | SHAP                                     |
| Pipeline    | Python, config JSON, modular scripts         |
| Infra       | Docker, Flask UI compatible(though not integrated)   |
| Cloud Ready | Azure AutoML (tested), S3, CI/CD via GitHub Actions |

---
