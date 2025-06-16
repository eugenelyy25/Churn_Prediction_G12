import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    eda_df = df.copy()
    eda_df['Churn'] = eda_df['Churn'].replace({'Yes': 1, 'No': 0}) if eda_df['Churn'].dtype == 'object' else eda_df['Churn']
    eda_df['Churn'] = eda_df['Churn'].map({1: 'Yes', 0: 'No'})

    st.header("Exploratory Data Analysis")

    st.subheader("Churn Table")
    churn_table = eda_df['Churn'].value_counts().reset_index()
    churn_table.columns = ['Churn Status', 'Quantity']
    churn_table['Percentage'] = churn_table['Quantity'] / churn_table['Quantity'].sum() * 100
    churn_table['Percentage'] = churn_table['Percentage'].map("{:.2f}%".format)
    st.dataframe(churn_table)

    st.subheader("Monthly Charges Distribution by Churn")
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=eda_df, x='MonthlyCharges', hue='Churn', fill=True)
    plt.title('Monthly Charges Distribution by Churn')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Density')
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Tenure by Churn")
    fig_tenure = px.box(eda_df, x='Churn', y='tenure', color='Churn')
    st.plotly_chart(fig_tenure, use_container_width=True)

    st.subheader("Contract Type vs Churn")
    if 'Contract' in eda_df.columns:
        fig_contract = px.histogram(eda_df, x='Contract', color='Churn', barmode='group')
        st.plotly_chart(fig_contract, use_container_width=True)

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

    st.header("Logistic Regression Model with Hyperparameter Tuning")
    base_model = LogisticRegression(solver='liblinear', random_state=42)
    param_dist = {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    random_search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=0)
    random_search.fit(X_train_bal, y_train_bal)

    best_model = random_search.best_estimator_
    best_params_df = pd.DataFrame([random_search.best_params_])
    st.subheader("Best Hyperparameters")
    st.dataframe(best_params_df)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

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
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# Footer logo
st.markdown("""
<div style='text-align: center; padding-top: 2em;'>
    <img src='https://download.logo.wine/logo/University_of_Malaya/University_of_Malaya-Logo.wine.png' width='180' style='max-width: 100%; height: auto;'>
</div>
""", unsafe_allow_html=True)
