import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Streamlit config
st.set_page_config(layout="wide")

st.markdown("""
<div style="background-color: #f0f2f6; padding: 10px 20px; border-radius: 10px;">
    <h2 style="color:#333;">Customer Churn Prediction App</h2>
    <p style="color:#555;">
        Explore key insights and predict churn using logistic regression. Balanced with SMOTE and enriched with EDA visuals.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Telco CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    required_cols = ['Churn', 'tenure', 'MonthlyCharges', 'TotalCharges']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode binary categorical columns
    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # Encode 'Contract' as ordinal
    if 'Contract' in df.columns:
        contract_encoder = OrdinalEncoder(categories=[['Month-to-month', 'One year', 'Two year']])
        df['Contract'] = contract_encoder.fit_transform(df[['Contract']])

    # One-hot encode nominal columns
    nominal_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaymentMethod']
    df = pd.get_dummies(df, columns=[col for col in nominal_cols if col in df.columns], drop_first=True)

    # Drop customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # EDA section
    st.header("Exploratory Data Analysis")

    churn_pie = df['Churn'].value_counts()
    fig_pie = px.pie(values=churn_pie.values, names=['No', 'Yes'], title='Churn Distribution', color_discrete_sequence=['skyblue', 'salmon'])
    st.plotly_chart(fig_pie, use_container_width=True)

    fig_hist = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30, title='Monthly Charges by Churn')
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_tenure = px.histogram(df, x='tenure', color='Churn', nbins=30, title='Tenure by Churn')
    st.plotly_chart(fig_tenure, use_container_width=True)

    # Classification preprocessing: 60/20/20
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Balance training data with SMOTE
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # Scale numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train_bal[num_cols] = scaler.fit_transform(X_train_bal[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Logistic Regression Model
    st.header("Logistic Regression Model")
    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    st.write(f"**Accuracy**: {acc:.4f}")
    st.write(f"**Precision**: {prec:.4f}")
    st.write(f"**Recall**: {rec:.4f}")
    st.write(f"**F1 Score**: {f1:.4f}")
    st.write(f"**ROC AUC**: {auc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='Blues'))
    fig_cm.update_layout(margin=dict(t=20, l=40, r=40, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# Footer logo
st.markdown("""
<div style='text-align: center; padding-top: 2em;'>
    <img src='https://download.logo.wine/logo/University_of_Malaya/University_of_Malaya-Logo.wine.png' width='180' style='max-width: 100%; height: auto;'>
</div>
""", unsafe_allow_html=True)
