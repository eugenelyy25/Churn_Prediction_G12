
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

# Page configuration and banner
st.set_page_config(layout="wide")
st.image("https://download.logo.wine/logo/University_of_Malaya/University_of_Malaya-Logo.wine.png", use_container_width=True)

st.markdown("""
<div style="background-color: #f0f2f6; padding: 10px 20px; border-radius: 10px;">
    <h2 style="color:#333;">Customer Churn Prediction App</h2>
    <p style="color:#555;">
        This interactive dashboard allows users to explore churn data, understand customer behavior,
        and compare the performance of various machine learning models used to predict customer churn.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    st.header("Data Overview")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Handle and clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode categorical columns except 'Churn'
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')
    encoder = OrdinalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Bivariate analysis
    if st.sidebar.checkbox("Show Bivariate Analysis"):
        st.subheader("Bivariate Analysis")

        fig1 = px.histogram(df, x='Contract', color='Churn', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(df, x='Churn', y='MonthlyCharges', color='Churn')
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x='Churn', y='TotalCharges', color='Churn')
        st.plotly_chart(fig3, use_container_width=True)

    # Model preparation
    X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    y = df['Churn'].apply(lambda x: 1 if x == "Yes" or x == 1 else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='rbf', probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "AdaBoost + Logistic Regression": AdaBoostClassifier(estimator=LogisticRegression(max_iter=200))
    }

    st.header("Model Training and Evaluation")
    model_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_results[name] = acc

        with st.expander(f"{name} Results"):
            st.write(f"**Accuracy**: {acc:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Model comparison chart
    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame.from_dict(model_results, orient='index', columns=["Accuracy"]).sort_values(by="Accuracy", ascending=False)
    st.bar_chart(comparison_df)

else:
    st.warning("Please upload a dataset to continue.")
