import streamlit as st
import pandas as pd
import numpy as np
from models.data_loader import load_and_preprocess_data
from models.logistic_regression_model import run_logistic_regression
from models.decision_tree_model import run_decision_tree
from models.knn_model import run_knn
from models.naive_bayes_model import run_naive_bayes
from models.random_forest_model import run_random_forest
from models.xgboost_model import run_xgboost

st.set_page_config(page_title="ML Models Evaluation", layout="wide")
st.title("ML Models Evaluation to Predict Annual Income")

st.markdown("""
* If no test file is uploaded, evaluation will run on the **internal test split**.
* If a test file is uploaded, evaluation will run on the uploaded dataset.
""")

run_models = True

st.sidebar.subheader("Download Test Dataset")
try:
    with open("data/adult.test", "rb") as f:
        adult_test_bytes = f.read()

    st.sidebar.download_button(
        label="Download adult.test data",
        data=adult_test_bytes,
        file_name="adult.test",
        mime="text/csv"
    )

    run_models = False
except FileNotFoundError:
    st.warning("Test dataset not found in repository.")

# Sidebar
st.sidebar.header("Upload Test Data")
uploaded_test_file = st.sidebar.file_uploader(
    "Upload Test file",
    type=["csv", "test"]
)

#st.sidebar.markdown(
#    "If no test file is uploaded, data will be split automatically."
#)
if uploaded_test_file is not None:
    st.success("Using uploaded test dataset")

    # Read file for preview
    df_preview = pd.read_csv(
        uploaded_test_file,
        header=None,
        skiprows=1
    )

    st.subheader("Preview of Uploaded Test Data")
    st.dataframe(df_preview.head(), use_container_width=True)
    uploaded_test_file.seek(0)

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data(uploaded_test_file)
model_choice = st.selectbox(
    "Choose Model",
    (
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes Classifier",
        "Ensemble Model - Random Forest",
        "Ensemble Model - XGBoost"
    )
)
# Run Model
results = None

if run_models:
    if model_choice == "Logistic Regression":
        if uploaded_test_file:
            st.subheader("Running evaluation for Logistic Regression on uploaded test dataset")
        else:
            st.subheader("Running evaluation for Logistic Regression")
        results = run_logistic_regression(X_train, X_test, y_train, y_test)
    elif model_choice == "Decision Tree Classifier":
        if uploaded_test_file:
            st.subheader("Running evaluation for Decision Tree Classifier on uploaded test dataset")
        else:
            st.subheader("Running evaluation for Decision Tree Classifier")
        results = run_decision_tree(X_train, X_test, y_train, y_test)
    elif model_choice == "K-Nearest Neighbor Classifier":
        if uploaded_test_file:
            st.subheader("Running evaluation for K-Nearest Neighbor Classifier on uploaded test dataset")
        else:
            st.subheader("Running evaluation for K-Nearest Neighbor Classifier")
        results = run_knn(X_train, X_test, y_train, y_test)
    elif model_choice == "Naive Bayes Classifier":
        if uploaded_test_file:
            st.subheader("Running evaluation for Naive Bayes Classifier on uploaded test dataset")
        else:
            st.subheader("Running evaluation for Naive Bayes Classifier")
        results = run_naive_bayes(X_train, X_test, y_train, y_test)
    elif model_choice == "Ensemble Model - Random Forest":
        if uploaded_test_file:
            st.subheader("Running evaluation for Ensemble Model - Random Forest on uploaded test dataset")
        else:
            st.subheader("Running evaluation for Ensemble Model - Random Forest")
        results = run_random_forest(X_train, X_test, y_train, y_test)
    elif model_choice == "Ensemble Model - XGBoost":
        if uploaded_test_file:
            st.subheader("Running evaluation for Ensemble Model - XGBoost on uploaded test dataset")
        else:
            st.subheader("Running evaluation for Ensemble Model - XGBoost")
        results = run_xgboost(X_train, X_test, y_train, y_test)

    
    # Display Metrics


    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
    col2.metric("AUC Score", f"{results['auc']:.4f}")
    col3.metric("Precision", f"{results['precision']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{results['recall']:.4f}")
    col5.metric("F1 Score", f"{results['f1']:.4f}")
    col6.metric("MCC", f"{results['mcc']:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write(results["confusion_matrix"])

    # Display classification report
    st.subheader("Classification Report")
    report_dict = results["classification_report"]

    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(4)

    st.dataframe(report_df, use_container_width=True)
