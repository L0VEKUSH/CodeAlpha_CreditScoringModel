
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"
DATA_PATH = BASE_DIR / "data" / "credit_data.csv"
METRICS_PATH = BASE_DIR / "results" / "model_metrics.json"
COMPARISON_PATH = BASE_DIR / "results" / "model_comparison.csv"

st.set_page_config(
    page_title="Credit Scoring Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .risk-card {
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        background: #f8fafc;
        color: #0f172a;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .risk-card h2 {
        margin-top: 0;
        color: #0f172a;
    }

    .risk-card p {
        color: #334155;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding-top: 30px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD PROJECT FILES
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found: data/credit_data.csv"
        )

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():

    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_comparison():

    if not COMPARISON_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        COMPARISON_PATH,
        index_col=0
    )


try:
    model = load_model()
    df = load_dataset()
    metrics_data = load_metrics()
    comparison_df = load_comparison()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("💳 Credit Intelligence")

    st.caption("CodeAlpha Machine Learning Internship")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Credit Risk Prediction",
            "Model Performance",
            "Dataset Insights",
            "About Project"
        ]
    )

    st.divider()

    st.info(
        "Educational ML demonstration only. "
        "Predictions must not be used to approve "
        "or reject real credit applications."
    )


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def risk_gauge(probability):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "valueformat": ".1f"
            },
            title={
                "text": "Predicted Bad-Risk Probability"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#2563eb"
                },
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#dcfce7"
                    },
                    {
                        "range": [30, 60],
                        "color": "#fef3c7"
                    },
                    {
                        "range": [60, 100],
                        "color": "#fee2e2"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def build_applicant_form():

    st.subheader("Applicant Financial Profile")

    st.caption(
        "Enter a demonstration applicant's information. "
        "Categorical fields use the original German "
        "Credit dataset codes."
    )

    features = df.drop(
        columns=["credit_risk"]
    )

    applicant = {}

    with st.form("credit_prediction_form"):

        col1, col2 = st.columns(2)

        for index, column in enumerate(features.columns):

            target_col = col1 if index % 2 == 0 else col2

            with target_col:

                series = features[column]

                if pd.api.types.is_numeric_dtype(series):

                    minimum = float(series.min())
                    maximum = float(series.max())
                    default = float(series.median())

                    if pd.api.types.is_integer_dtype(series):

                        applicant[column] = st.number_input(
                            column.replace("_", " ").title(),
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(default),
                            step=1
                        )

                    else:

                        applicant[column] = st.number_input(
                            column.replace("_", " ").title(),
                            min_value=minimum,
                            max_value=maximum,
                            value=default
                        )

                else:

                    options = sorted(
                        series.dropna().unique().tolist()
                    )

                    applicant[column] = st.selectbox(
                        column.replace("_", " ").title(),
                        options=options,
                        help=(
                            "Select a category code from "
                            "the German Credit dataset."
                        )
                    )

        submitted = st.form_submit_button(
            "Analyze Credit Risk",
            type="primary",
            use_container_width=True
        )

    return submitted, pd.DataFrame([applicant])


# --------------------------------------------------
# PAGE 1: CREDIT RISK PREDICTION
# --------------------------------------------------

if page == "Credit Risk Prediction":

    st.markdown(
        '<div class="main-title">'
        'Credit Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine learning-based credit risk assessment demo'
        '</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "This is a historical-dataset educational demo, "
        "not a calibrated lending decision system. "
        "Use synthetic applicant profiles only."
    )

    submitted, applicant_df = build_applicant_form()

    if submitted:

        try:

            prediction = int(
                model.predict(applicant_df)[0]
            )

            probability = float(
                model.predict_proba(applicant_df)[0, 1]
            )

            st.divider()

            st.subheader("Prediction Results")

            col1, col2 = st.columns([1, 1])

            with col1:

                if prediction == 1:

                    result = "Bad Credit Risk"

                    st.error(
                        "Model Classification: Bad Credit Risk"
                    )

                else:

                    result = "Good Credit Risk"

                    st.success(
                        "Model Classification: Good Credit Risk"
                    )

                st.markdown(
                    f"""
                    <div class="risk-card">
                        <h2>{result}</h2>
                        <p>
                            Predicted probability of bad credit risk:
                            <strong>{probability:.2%}</strong>
                        </p>
                        <p>
                            This is a model estimate, not a
                            verified probability of default.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric(
                    "Predicted Good-Risk Probability",
                    f"{(1 - probability) * 100:.2f}%"
                )

            with col2:

                st.plotly_chart(
                    risk_gauge(probability),
                    use_container_width=True
                )

            st.info(
                "The classification uses the model's "
                "default decision threshold. The gauge "
                "shows a model score, not an independently "
                "validated credit risk category."
            )

            st.subheader("Applicant Information")

            st.dataframe(
                applicant_df.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )

            report = {
                "project": "CodeAlpha Credit Scoring Model",
                "timestamp": datetime.now().isoformat(),
                "model": metrics_data.get(
                    "selected_model",
                    "Unknown"
                ),
                "prediction": result,
                "bad_risk_probability": probability,
                "good_risk_probability": 1 - probability,
                "applicant": applicant_df.iloc[0].to_dict(),
                "disclaimer": (
                    "Educational demonstration only. "
                    "Not for real lending decisions."
                )
            }

            st.download_button(
                label="Download Prediction Report",
                data=json.dumps(
                    report,
                    indent=4,
                    default=str
                ),
                file_name="credit_prediction_report.json",
                mime="application/json"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance Dashboard")

    st.caption(
        "Evaluation results from the held-out test dataset."
    )

    test_metrics = metrics_data.get(
        "test_metrics",
        {}
    )

    if test_metrics:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{test_metrics.get('accuracy', 0):.2%}"
        )

        col2.metric(
            "Precision",
            f"{test_metrics.get('precision', 0):.2%}"
        )

        col3.metric(
            "Recall",
            f"{test_metrics.get('recall', 0):.2%}"
        )

        col4.metric(
            "ROC-AUC",
            f"{test_metrics.get('roc_auc', 0):.4f}"
        )

        st.metric(
            "F1 Score",
            f"{test_metrics.get('f1_score', 0):.4f}"
        )

        st.success(
            "Selected Model: "
            + metrics_data.get(
                "selected_model",
                "Unknown"
            )
        )

    else:

        st.warning(
            "Model metrics are not available."
        )

    st.divider()

    st.subheader("Cross-Validation Model Comparison")

    if not comparison_df.empty:

        st.dataframe(
            comparison_df.style.format("{:.4f}"),
            use_container_width=True
        )

        st.bar_chart(
            comparison_df[
                ["CV F1", "CV ROC-AUC"]
            ],
            stack=False,
            horizontal=False
        )

    st.divider()

    st.subheader("Confusion Matrix")

    confusion_path = (
        BASE_DIR / "results" / "confusion_matrix.png"
    )

    if confusion_path.exists():

        st.image(
            str(confusion_path),
            use_container_width=True
        )

    st.subheader("ROC Curve")

    roc_path = BASE_DIR / "results" / "roc_curve.png"

    if roc_path.exists():

        st.image(
            str(roc_path),
            use_container_width=True
        )

    st.info(
        "These results describe performance on the "
        "German Credit benchmark test split. "
        "They do not establish performance on "
        "current real-world applicants."
    )


# --------------------------------------------------
# PAGE 3: DATASET INSIGHTS
# --------------------------------------------------

elif page == "Dataset Insights":

    st.title("Dataset Insights")

    st.caption(
        "Exploratory analysis of the German Credit dataset."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Input Features",
        len(df.columns) - 1
    )

    col3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Credit Risk Distribution")

    counts = df["credit_risk"].value_counts()

    # Support both original and converted target labels.
    if set(counts.index) == {1, 2}:

        counts.index = counts.index.map({
            1: "Good Risk",
            2: "Bad Risk"
        })

    elif set(counts.index) == {0, 1}:

        counts.index = counts.index.map({
            0: "Good Risk",
            1: "Bad Risk"
        })

    st.bar_chart(counts)

    st.subheader("Numerical Feature Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Feature Distribution")

    numeric_columns = df.drop(
        columns=["credit_risk"]
    ).select_dtypes(
        include="number"
    ).columns.tolist()

    selected_feature = st.selectbox(
        "Select Numerical Feature",
        numeric_columns
    )

    st.bar_chart(
        df[selected_feature].value_counts().sort_index()
    )


# --------------------------------------------------
# PAGE 4: ABOUT
# --------------------------------------------------

elif page == "About Project":

    st.title("About Credit Scoring Intelligence")

    st.markdown("""
    ### Project Overview

    This project was developed as part of the
    CodeAlpha Machine Learning Internship.

    The objective is to demonstrate how machine
    learning can classify historical credit-risk
    records using financial and demographic features.

    ### Machine Learning Algorithms

    - Logistic Regression
    - Decision Tree
    - Random Forest

    ### Model Evaluation

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Confusion Matrix

    ### Technology Stack

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Plotly

    ### Dataset

    German Credit Data — UCI Machine Learning Repository.

    ### Limitations

    This project uses a small historical benchmark dataset.

    The model has not been validated for modern lending
    populations, probability calibration, fairness,
    regulatory compliance, or real-world deployment.

    Its outputs must not be used to make actual
    credit eligibility decisions.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        CodeAlpha Machine Learning Internship |
        Credit Scoring Intelligence |
        Educational Demonstration
    </div>
    """,
    unsafe_allow_html=True
)