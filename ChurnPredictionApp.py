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

        # Show suggestions only when required columns are missing
        st.sidebar.markdown("""
        #### Tips for other datasets:
        - Ensure `target` variable is clearly defined
        - Use `StandardScaler` for numerical consistency
        - Consider features like:
            - Feature importances (SHAP, permutation)
            - ROC Curve, Precision-Recall Curve
            - Class Imbalance Dashboard
            - Time-series (trend lines, seasonality)
        - Upload multiple datasets for benchmarking
        """)

        st.stop()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    if 'Contract' in df.columns:
        contract_encoder = OrdinalEncoder(categories=[['Month-to-month', 'One year', 'Two year']])
        df['Contract'] = contract_encoder.fit_transform(df[['Contract']])

    nominal_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaymentMethod']
    df = pd.get_dummies(df, columns=[col for col in nominal_cols if col in df.columns], drop_first=True)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # EDA section (before modeling and encoding target back)
    eda_df = df.copy()
    eda_df['Churn'] = eda_df['Churn'].map({0: 'No', 1: 'Yes'})

    st.header("Exploratory Data Analysis")

    churn_pie = eda_df['Churn'].value_counts()
    fig_pie = px.pie(values=churn_pie.values, names=churn_pie.index, title='Churn Distribution', color_discrete_sequence=['skyblue', 'salmon'])
    st.plotly_chart(fig_pie, use_container_width=True)

    if 'MonthlyCharges' in eda_df.columns:
        fig1 = px.histogram(eda_df, x='MonthlyCharges', color='Churn', nbins=30, title='Monthly Charges by Churn')
        fig1.update_layout(template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)

    if 'tenure' in eda_df.columns:
        fig2 = px.histogram(eda_df, x='tenure', color='Churn', nbins=30, title='Tenure by Churn')
        fig2.update_layout(template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    # Optional: Correlation matrix heatmap
    st.subheader("Feature Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix", aspect="auto")
    fig_corr.update_layout(template='plotly_white')
    st.plotly_chart(fig_corr, use_container_width=True)

    # Model section
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train_bal[num_cols] = scaler.fit_transform(X_train_bal[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

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

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['True 0','True 1'], colorscale='Blues'))
    fig_cm.update_layout(margin=dict(t=20, l=40, r=40, b=20), template='plotly_white')
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# Footer logo
st.markdown("""
<div style='text-align: center; padding-top: 2em;'>
    <img src='https://download.logo.wine/logo/University_of_Malaya/University_of_Malaya-Logo.wine.png' width='180' style='max-width: 100%; height: auto;'>
</div>
""", unsafe_allow_html=True)
