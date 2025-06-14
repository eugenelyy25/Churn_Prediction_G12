
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Telco Churn Data App", layout="wide")
st.title("Telco Customer Churn Analysis")


st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.subheader("Dataset Info")
    st.text(str(df.info()))
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include='all').T)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Example visualization
    st.subheader("Distribution of Total Charges")
    fig = px.histogram(df, x='TotalCharges', nbins=50, title='Total Charges Distribution')
    st.plotly_chart(fig)

    # Encoding categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')
    encoder = OrdinalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Drop rows with missing values
    df = df.dropna()

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])  # only numeric columns
    if numeric_df.shape[1] >= 2:
        fig2, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig2)
    else:
        st.warning("Not enough numeric data to display a correlation heatmap.")

    # ML Section
    st.header("Model Training")

    X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    y = df['Churn'].apply(lambda x: 1 if x == "Yes" or x == 1 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_option = st.selectbox("Choose a model", ["Decision Tree", "Logistic Regression", "Neural Network"])

    if model_option == "Decision Tree":
        model = DecisionTreeClassifier(criterion="gini", random_state=42)
    elif model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    else:
        model = MLPClassifier(max_iter=100)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy**: {acc:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
else:
    st.warning("Please upload a dataset to continue.")
