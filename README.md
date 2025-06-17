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

## 🧪 Notebook Structure


- Based on OSEMN Framework
    - Obtain
    - Scrub
    - Explore
        - Categorical Variables
        - Continuous Variables
        - Correlation Heatmaps
    - Model
        - KNN
        - SVC
        - Logistic Regression
        - XGBoost
        - LOGREG + ADABOOST
    - Interpret
        - Comparison for Evaluation
        - Deployment Research
 
---

This repository include:

- README with setup and run instructions
- Notebook file (ChurnPrediction.ipynb) 
- Streamlit Deployment Script (ChurnPredictionApp.py)
- Dataset (telco_data.csv)
- requirements.txt for reproducibility
