"""
Customer Churn Prediction & Retention Analysis System
-------------------------------------------------------
A Streamlit web application that loads the trained churn model and
preprocessing pipeline (produced by Customer_Churn_Prediction.ipynb) and
serves real-time churn risk predictions through an interactive UI.

Run locally:    streamlit run app.py
Deploy:         push to GitHub, then deploy on share.streamlit.io
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction | Retention Analytics",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CUSTOM CSS FOR A POLISHED, PROFESSIONAL LOOK
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fb;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2E86AB;
    }
    .risk-high {
        background-color: #FDECEC;
        border-left: 6px solid #E63946;
        padding: 1.2rem;
        border-radius: 10px;
    }
    .risk-medium {
        background-color: #FFF6E5;
        border-left: 6px solid #F4A261;
        padding: 1.2rem;
        border-radius: 10px;
    }
    .risk-low {
        background-color: #EAF7ED;
        border-left: 6px solid #2A9D8F;
        padding: 1.2rem;
        border-radius: 10px;
    }
    div.stButton > button {
        background-color: #1a1a2e;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2E86AB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# CACHED RESOURCE LOADING
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the trained model, preprocessor, and metadata once per session."""
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return model, preprocessor, metadata


try:
    model, preprocessor, metadata = load_artifacts()
    ARTIFACTS_LOADED = True
except FileNotFoundError:
    ARTIFACTS_LOADED = False


# --------------------------------------------------------------------------
# FEATURE ENGINEERING (mirrors the notebook's Section 5 exactly)
# --------------------------------------------------------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    """Recreate the exact feature engineering pipeline used in training,
    starting from raw form inputs, so the saved preprocessor sees the
    same schema it was fit on."""
    row = raw.copy()

    tenure = row["tenure"]
    total_charges = row["TotalCharges"]

    row["AvgMonthlySpend"] = total_charges / (tenure if tenure > 0 else 1)

    if tenure <= 12:
        row["TenureGroup"] = "0-1 yr"
    elif tenure <= 24:
        row["TenureGroup"] = "1-2 yr"
    elif tenure <= 48:
        row["TenureGroup"] = "2-4 yr"
    else:
        row["TenureGroup"] = "4+ yr"

    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    row["NumAddOnServices"] = sum(1 for c in addon_cols if row[c] == "Yes")

    row["HasInternet"] = 0 if row["InternetService"] == "No" else 1
    row["IsNewCustomer"] = 1 if tenure <= 3 else 0

    return pd.DataFrame([row])


def risk_tier(prob: float) -> tuple:
    """Bucket a churn probability into a business-facing risk tier."""
    if prob >= 0.6:
        return "High Risk", "risk-high", "#E63946"
    elif prob >= 0.3:
        return "Medium Risk", "risk-medium", "#F4A261"
    else:
        return "Low Risk", "risk-low", "#2A9D8F"


def gauge_chart(prob: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#EAF7ED"},
                {"range": [30, 60], "color": "#FFF6E5"},
                {"range": [60, 100], "color": "#FDECEC"},
            ],
            "threshold": {"line": {"color": "black", "width": 3},
                          "thickness": 0.8, "value": prob * 100},
        },
        title={"text": "Churn Probability"},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# --------------------------------------------------------------------------
# SIDEBAR — PROJECT INFORMATION
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📉 Project Info")
    st.markdown("""
    **Customer Churn Prediction & Retention Analysis System**

    An end-to-end machine learning system that predicts customer churn risk
    and surfaces the key drivers behind each prediction, enabling data-driven
    retention strategy.
    """)

    st.markdown("---")
    st.markdown("### 🧠 Model")
    if ARTIFACTS_LOADED:
        st.success(f"**{metadata['best_model_name']}** (loaded)")
        results = metadata["results"][metadata["best_model_name"]]
        st.metric("ROC-AUC", f"{results['ROC-AUC']:.3f}")
        col_a, col_b = st.columns(2)
        col_a.metric("Recall", f"{results['Recall']:.2f}")
        col_b.metric("Precision", f"{results['Precision']:.2f}")
    else:
        st.error("Model artifacts not found. Run the notebook first to generate `model.pkl`.")

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("""
    IBM Telco Customer Churn Dataset
    7,043 customers · 21 original features
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("`scikit-learn` · `XGBoost` · `SHAP` · `Streamlit` · `Plotly`")

    st.markdown("---")
    st.caption("Built as a portfolio project demonstrating end-to-end ML engineering: "
                "data cleaning, feature engineering, model comparison, explainability, "
                "and deployment.")


# --------------------------------------------------------------------------
# MAIN HEADER
# --------------------------------------------------------------------------
st.markdown('<p class="main-header">📉 Customer Churn Prediction & Retention Analysis</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter a customer\'s profile to predict churn risk and '
            'see the key factors driving that prediction.</p>', unsafe_allow_html=True)

if not ARTIFACTS_LOADED:
    st.warning(
        "⚠️ `model.pkl`, `preprocessor.pkl`, or `model_metadata.json` were not found in the "
        "app directory. Run `Customer_Churn_Prediction.ipynb` end-to-end first — it saves "
        "these files automatically — then restart the app."
    )
    st.stop()

tab_predict, tab_about = st.tabs(["🔮 Predict Churn", "ℹ️ About This Project"])

# --------------------------------------------------------------------------
# TAB 1 — PREDICTION FORM
# --------------------------------------------------------------------------
with tab_predict:
    with st.form("customer_form"):
        st.markdown("### 👤 Customer Profile")

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with c3:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                             float(monthly_charges) * max(tenure, 1), step=10.0)

        st.markdown("### 📡 Services")
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with s2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with s3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn Risk")

    if submitted:
        raw_input = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        input_df = engineer_features(raw_input)
        input_processed = preprocessor.transform(input_df)

        churn_proba = model.predict_proba(input_processed)[0][1]
        tier_label, tier_css, tier_color = risk_tier(churn_proba)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            st.plotly_chart(gauge_chart(churn_proba, tier_color), use_container_width=True)

        with res_col2:
            st.markdown(f"""
            <div class="{tier_css}">
                <h3 style="margin-top:0;">Risk Category: {tier_label}</h3>
                <p style="font-size:1.1rem;">
                    This customer has a <b>{churn_proba*100:.1f}%</b> probability of churning,
                    based on the trained {metadata['best_model_name']} model.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Suggested Action")
            if tier_label == "High Risk":
                st.error("🚨 Prioritize immediate retention outreach — consider a loyalty "
                         "discount, contract upgrade incentive, or personal account review call.")
            elif tier_label == "Medium Risk":
                st.warning("⚠️ Monitor closely and consider a proactive check-in or "
                           "value-add offer (e.g., bundled service discount).")
            else:
                st.success("✅ Low risk — no immediate action needed. Continue standard "
                           "engagement and satisfaction monitoring.")

        # ---- Local feature importance (global importance from the model) ----
        st.markdown("### 📊 What's Driving This Prediction")
        st.caption("Global feature importance from the trained model — the strongest "
                    "overall churn signals across all customers.")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])

        fi_df = pd.DataFrame({
            "Feature": metadata["feature_names"],
            "Importance": importances
        }).sort_values("Importance", ascending=True).tail(10)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker_color="#2E86AB"
        ))
        fig_fi.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 2 — ABOUT
# --------------------------------------------------------------------------
with tab_about:
    st.markdown("""
    ### About This Project

    This application is the deployment layer of an end-to-end **Customer Churn Prediction &
    Retention Analysis System**, built on the IBM Telco Customer Churn dataset (7,043 customers).

    **Pipeline summary:**
    1. Data cleaning & missing value handling
    2. Feature engineering (tenure grouping, add-on counts, spend ratios)
    3. Class imbalance correction with SMOTE
    4. Model comparison: Logistic Regression, Random Forest, XGBoost
    5. Hyperparameter tuning via `RandomizedSearchCV`
    6. Explainability with SHAP
    7. Deployment via this Streamlit app

    **Model performance (test set):**
    """)
    results_table = pd.DataFrame(metadata["results"]).T
    st.dataframe(results_table.style.format("{:.3f}"), use_container_width=True)

    st.markdown("""
    See the full notebook (`Customer_Churn_Prediction.ipynb`) and `README.md` in the project
    repository for the complete methodology, visualizations, and business recommendations.
    """)

st.markdown("---")
st.caption("Customer Churn Prediction & Retention Analysis System · Built with Streamlit, "
            "scikit-learn, XGBoost & SHAP")
