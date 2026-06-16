# ❤️ Cardiovascular Disease Screening Center
> A Full-Stack Machine Learning Web Application utilizing a Multi-Classifier Soft-Voting Ensemble Engine to predict the probability of heart disease.

---

## 📊 Project Overview
This project was developed as part of a machine learning internship task. The primary objective is to reliably evaluate cardiovascular risk thresholds based on a patient's demographics, metabolic profile, and specialized cardiac markers. 

Instead of relying on a single algorithm, this system implements a high-performance **Hybrid Soft-Voting Ensemble** that combines four distinct machine learning classifiers. The predictive backend is served via a clean, asynchronous **Flask API pipeline** and presented through a responsive **Bootstrap 5 UI web console**.

### 🔗 Dataset Reference
The predictive models are trained on the official, peer-reviewed **UCI Machine Learning Repository Heart Disease Dataset** (specifically the processed Cleveland clinical cohort).

---

## 🛠️ Tech Stack & Architecture

### Backend & Machine Learning Ecosystem
* **Python 3.11** — Core processing language
* **Scikit-Learn** — Feature scaling pipeline, SVM, Logistic Regression, and Random Forest development
* **XGBoost** — Advanced Gradient Boosting implementation
* **Joblib** — Serialization and optimization of machine learning model weights
* **Pandas & NumPy** — Matrix manipulation, data handling, and tracking cleaning

### Frontend & API Web Framework
* **Flask** — Micro-framework managing API routes and incoming POST prediction payloads
* **Jinja2** — Dynamic server-side HTML rendering engine
* **Bootstrap 5** — Fully responsive grid layouts, custom UI cards, and dynamic state-based alert boxes

---

## 🧠 Multi-Classifier Ensemble Pipeline

To achieve optimal diagnostic accuracy, the engine coordinates the following mandated classifiers simultaneously:

1.  **Support Vector Machine (SVM):** Constructs hyper-dimensional geometric boundary planes maximizing classification separation thresholds.
2.  **Logistic Regression (LR):** Establishes calibrated probabilistic baseline risk maps using log-odds optimization curves.
3.  **Random Forest Classifier (RF):** A bagging ensemble that aggregates un-correlated decision tree estimators to drastically minimize variance.
4.  **XGBoost Classifier (XGB):** An optimized gradient boosting framework that builds sequential trees to sequentially minimize modeling loss functions.

> **Why an Ensemble?** The app combines these models using a **Soft-Voting Classifier**. It calculates the weighted probability vectors across all four algorithms to issue a more resilient, generalized diagnosis than any single standalone model could provide.

---

## 📁 Repository Directory Structure

```text
CodeAlpha_DiseasePrediction/
│
├── disease_ensemble_model.pkl    # Serialized 4-Model Soft-Voting Ensemble weights
├── disease_scaler.pkl            # Trained StandardScaler artifact mapping feature boundaries
├── app.py                        # Core Flask API routing server engine
└── templates/
    └── index.html                # Responsive Bootstrap 5 web interface application
    
