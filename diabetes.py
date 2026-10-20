"""Streamlit diabetes classification and single-patient prediction app.

Run with: streamlit run diabetes_modified.py
Place diabetes.csv in the same directory as this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = Path(__file__).resolve().parent / "diabetes.csv"
CLASS_NAMES = {0: "Non-Diabetes", 1: "Diabetes"}


@st.cache_data
def load_data(file_mtime):
    """Cache CSV loading; file timestamp invalidates the data cache."""
    df = pd.read_csv(DATA_FILE)
    if "Outcome" not in df.columns:
        raise ValueError("diabetes.csv must contain an 'Outcome' column.")
    return df


@st.cache_resource(show_spinner="Training the selected model (first use only)...")
def get_trained_model(model_name, train_percent, params, file_mtime):
    """Train once per model/settings/dataset and reuse on later Streamlit reruns."""
    df = load_data(file_mtime)
    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=train_percent / 100.0,
        random_state=0,
        stratify=y,
    )

    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            criterion=params[0], splitter=params[1], random_state=0
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=params[0], max_iter=params[1], penalty="l2", random_state=0
        )
    else:
        model = RandomForestClassifier(
            n_estimators=params[0],
            max_depth=params[1],
            bootstrap=params[2],
            random_state=0,
            n_jobs=-1,
        )

    model.fit(x_train, y_train)
    return model, x_train, x_test, y_test


def plot_metrics(metrics_list, model, x_test, y_test):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, model.predict(x_test), labels=[0, 1])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)
        plt.close(fig)

    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        pos_index = list(model.classes_).index(1)
        y_prob = model.predict_proba(x_test)[:, pos_index]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)


def single_prediction(model, x_train):
    st.subheader("Single Patient Prediction")
    st.write("Enter the feature values below to predict one patient's outcome.")

    # Use columns and order from training data so that the input matches the model.
    patient = {}
    with st.form("single_patient_form"):
        col1, col2 = st.columns(2)
        for index, feature in enumerate(x_train.columns):
            default = float(x_train[feature].median())
            is_integer = pd.api.types.is_integer_dtype(x_train[feature].dtype)
            with (col1 if index % 2 == 0 else col2):
                if is_integer:
                    patient[feature] = st.number_input(
                        feature, min_value=0, value=max(0, int(round(default))),
                        step=1, key=f"patient_{feature}"
                    )
                else:
                    patient[feature] = st.number_input(
                        feature, min_value=0.0, value=max(0.0, default),
                        step=0.1, format="%.3f", key=f"patient_{feature}"
                    )
        predict_clicked = st.form_submit_button("Predict")

    if predict_clicked:
        patient_df = pd.DataFrame([patient], columns=x_train.columns)
        prediction = int(model.predict(patient_df)[0])
        probabilities = model.predict_proba(patient_df)[0]
        probability_map = dict(zip(model.classes_, probabilities))
        p_no = float(probability_map.get(0, 0.0))
        p_yes = float(probability_map.get(1, 0.0))

        st.markdown(f"**Prediction: {CLASS_NAMES[prediction]}**")
        st.write(f"Non-Diabetes probability: {p_no:.2%}")
        st.write(f"Diabetes probability: {p_yes:.2%}")

        fig, ax = plt.subplots()
        ax.pie(
            [p_no, p_yes],
            labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#66b3ff", "#ff9999"],
        )
        ax.set_title("Prediction Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)


def main():
    st.title("Predicting Diabetes Web App")
    st.markdown("Affected by Diabetes or not?")

    if not DATA_FILE.exists():
        st.error(f"Cannot find {DATA_FILE.name}. Put diabetes.csv next to this script.")
        st.stop()

    try:
        file_mtime = DATA_FILE.stat().st_mtime_ns
        df = load_data(file_mtime)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Cannot load dataset: {exc}")
        st.stop()

    st.sidebar.title("Model Selection Panel")
    st.sidebar.markdown("Choose your model and its parameters")
    train_percent = st.sidebar.slider(
        "Training data proportion (%)", min_value=50, max_value=90,
        value=70, step=5
    )
    st.sidebar.caption(f"Training: {train_percent}% | Testing: {100 - train_percent}%")

    classifier = st.sidebar.selectbox(
        "Classifier", ("Decision Tree", "Logistic Regression", "Random Forest")
    )
    st.sidebar.subheader("Model Parameters")

    if classifier == "Decision Tree":
        criterion = st.sidebar.radio("Criterion", ("gini", "entropy"))
        splitter = st.sidebar.radio("Splitter", ("best", "random"))
        params = (criterion, splitter)
    elif classifier == "Logistic Regression":
        c = st.sidebar.number_input(
            "C (Regularization parameter)", min_value=0.01,
            max_value=10.0, value=1.0, step=0.01
        )
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 1000, 500)
        params = (c, max_iter)
    else:
        n_estimators = st.sidebar.number_input(
            "The number of trees", min_value=10, max_value=5000,
            value=100, step=10
        )
        max_depth = st.sidebar.number_input(
            "Maximum tree depth", min_value=1, max_value=20, value=5, step=1
        )
        bootstrap = st.sidebar.radio("Bootstrap", (True, False))
        params = (n_estimators, max_depth, bootstrap)

    metrics = st.sidebar.multiselect(
        "Select metrics to plot",
        ("Confusion Matrix", "ROC Curve"),
        default=("Confusion Matrix", "ROC Curve"),
    )

    try:
        model, x_train, x_test, y_test = get_trained_model(
            classifier, train_percent, params, file_mtime
        )
    except ValueError as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

    st.subheader(f"{classifier} Results")
    y_pred = model.predict(x_test)
    cols = st.columns(4)
    scores = [
        ("Accuracy", accuracy_score(y_test, y_pred)),
        ("Precision", precision_score(y_test, y_pred, zero_division=0)),
        ("Recall", recall_score(y_test, y_pred, zero_division=0)),
        ("F1 Score", f1_score(y_test, y_pred, zero_division=0)),
    ]
    for col, (name, score) in zip(cols, scores):
        col.metric(name, f"{score:.3f}")

    st.caption(
        f"Training samples: {len(x_train)} | Testing samples: {len(x_test)}. "
        "Models are cached for each training ratio, classifier, and set of parameters."
    )
    plot_metrics(metrics, model, x_test, y_test)
    single_prediction(model, x_train)

    if st.sidebar.checkbox("Show raw data", value=False):
        st.subheader("Diabetes Raw Dataset")
        st.dataframe(df)


if __name__ == "__main__":
    main()
