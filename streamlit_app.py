import streamlit as st
from models.data_loader import load_and_preprocess_data
from models.logistic_regression_model import run_logistic_regression
from models.decision_tree_model import run_decision_tree
from models.knn_model import run_knn



st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Adult Income Classification")

# Sidebar
st.sidebar.header("Upload Test Data")
uploaded_test_file = st.sidebar.file_uploader(
    "Upload Test CSV (Optional)",
    type=["csv", "test"]
)

st.sidebar.markdown(
    "If no test file is uploaded, data will be split automatically."
)

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

if model_choice == "Logistic Regression":
    st.subheader("Logistic Regression Results")
    results = run_logistic_regression(X_train, X_test, y_train, y_test)
elif model_choice == "Decision Tree Classifier":
    st.subheader("Decision Tree Classifier")
    results = run_decision_tree(X_train, X_test, y_train, y_test)
elif model_choice == "K-Nearest Neighbor Classifier":
    results = run_knn(X_train, X_test, y_train, y_test)
    
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

# Classification Report
st.subheader("Classification Report")
st.text(results["classification_report"])
