# 📊 Customer Churn Prediction for Telecommunications

## 📁 Project Overview

Customer churn prediction is essential for telecom companies to retain customers in a competitive market with low switching barriers. This project develops machine learning models to predict customer churn using various techniques, including Logistic Regression, XGBoost, KNN, SVM, and AdaBoost.

The best-performing model (Logistic Regression with AdaBoost) was deployed as a web application using Streamlit to enable real-time, user-friendly predictions.

---

## 🎯 Objectives

- Identify key features driving customer churn
- Build and compare predictive models
- Optimize performance using hyperparameter tuning
- Address class imbalance using sampling techniques
- Deploy a predictive web application for practical use

---

## 📊 Dataset

A synthetic telco churn dataset (or real dataset if accessible) with the following attributes:
- Features include `tenure`, `Contract`, `OnlineSecurity`, `MonthlyCharges`, etc.
- Target variable: `Churn` (binary classification)

---

## 🧪 Project Structure

```bash
project-root/
│
├── data/                   # Contains the dataset
├── notebooks/              # Jupyter notebooks for each stage
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_model_tuning.ipynb
│   ├── 05_model_evaluation.ipynb
│
├── app/
│   └── streamlit_app.py    # Streamlit application for deployment
│
├── reports/
│   └── churn_model_performance.md
│
├── requirements.txt        # Python dependencies
├── environment.yml         # Optional Conda environment
└── README.md               # Project overview and setup instructions
