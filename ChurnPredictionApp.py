import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Header and intro
st.set_page_config(layout="wide")
st.image("https://download.logo.wine/logo/University_of_Malaya/University_of_Malaya-Logo.wine.png", use_container_width=True)

st.markdown("""
<div style="background-color: #f0f2f6; padding: 10px 20px; border-radius: 10px;">
    <h2 style="color:#333;">Customer Churn Prediction App</h2>
    <p style="color:#555;">
        This interactive dashboard explores churn behavior and compares model performance using KNN, SVM, Logistic Regression, XGBoost,
        and AdaBoost with Logistic Regression. Data balancing and tuning are included.
    </p>
</div>
""", unsafe_allow_html=True)

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your Telco CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    # Label Encoding
    def object_to_int(series):
        if series.dtype == 'object':
            return LabelEncoder().fit_transform(series)
        return series

    df = df.apply(object_to_int)

    # Show class distribution
    st.subheader("Class Distribution")
    st.write(df['Churn'].value_counts())

    # Data Split
    X = df.drop(columns=['Churn'])
    y = df['Churn'].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Data Balancing
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Normalization
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train_res[num_cols] = scaler.fit_transform(X_train_res[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    st.subheader("Model Results")
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "SVM": SVC(random_state=1, probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "AdaBoost + LogReg": AdaBoostClassifier(estimator=LogisticRegression(max_iter=1000, random_state=42), n_estimators=50, learning_rate=1, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "ROC AUC": auc}

        with st.expander(f"{name} Results"):
            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"Precision: {prec:.4f}")
            st.write(f"Recall: {rec:.4f}")
            st.write(f"F1 Score: {f1:.4f}")
            if auc: st.write(f"ROC AUC: {auc:.4f}")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))

    # Model Comparison
    st.subheader("Model Comparison Chart")
    comp_df = pd.DataFrame(results).T
    st.dataframe(comp_df.style.highlight_max(axis=0))
    st.bar_chart(comp_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']])
else:
    st.warning("Please upload a CSV file to continue.")
