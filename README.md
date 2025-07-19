# PredictML-Production: Hospital No-Show Prediction System

Note: Choosing to respect the NDA, I will only show my IP.

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
| Infra       | Docker, Flask UI compatible   |
| Cloud Ready | Azure AutoML (tested), S3, CI/CD via GitHub Actions |

---
