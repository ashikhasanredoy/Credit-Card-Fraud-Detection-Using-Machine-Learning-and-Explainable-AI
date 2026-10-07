import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Fraud Detection in Digital Payment System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .alert-fraud {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .alert-safe {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        color: #166534;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
    }
    div[data-testid="stRadio"] > div {
        background-color: #F1F5F9;
        border-radius: 10px;
        padding: 4px;
        display: inline-flex;
        margin-bottom: 20px;
    }
    div[data-testid="stRadio"] label {
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "models" / "best_fraud_model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_DIR / "best_fraud_model.pkl"

DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8003")

def patch_sklearn_estimator(estimator):
    """Recursively patch deserialized scikit-learn estimators for cross-version compatibility."""
    if estimator is None:
        return estimator
    
    if hasattr(estimator, 'statistics_') and not hasattr(estimator, '_fill_dtype'):
        fit_dtype = getattr(estimator, '_fit_dtype', getattr(estimator.statistics_, 'dtype', np.float64))
        setattr(estimator, '_fill_dtype', fit_dtype)
        
    if hasattr(estimator, 'steps'):
        for _, step in estimator.steps:
            patch_sklearn_estimator(step)
            
    if hasattr(estimator, 'transformers_'):
        for item in estimator.transformers_:
            if len(item) >= 2:
                patch_sklearn_estimator(item[1])
                
    if hasattr(estimator, 'named_steps'):
        for _, step in estimator.named_steps.items():
            patch_sklearn_estimator(step)
            
    if hasattr(estimator, 'estimators_'):
        for est in estimator.estimators_:
            patch_sklearn_estimator(est)
            
    if hasattr(estimator, 'named_estimators_'):
        for _, est in estimator.named_estimators_.items():
            patch_sklearn_estimator(est)
            
    return estimator

@st.cache_resource
def load_local_model():
    if MODEL_PATH.exists():
        try:
            try:
                import sklearn.compose._column_transformer as _ct
                from collections import UserList
                if not hasattr(_ct, "_RemainderColsList"):
                    class _RemainderColsList(UserList):
                        def __init__(self, columns=(), **kwargs):
                            super().__init__(columns)
                    _ct._RemainderColsList = _RemainderColsList
            except Exception:
                pass

            loaded = joblib.load(MODEL_PATH)
            return patch_sklearn_estimator(loaded)
        except Exception as e:
            st.error(f"Error loading local model: {e}")
            return None
    return None

local_model = load_local_model()

st.markdown('<div class="main-title">Fraud Detection in Digital Payment System</div>', unsafe_allow_html=True)

st.sidebar.header("Configuration & Backend")
api_url = st.sidebar.text_input("FastAPI Endpoint URL", value=DEFAULT_API_URL)

api_online = False
try:
    res = requests.get(f"{api_url}/", timeout=1.5)
    if res.status_code == 200:
        data = res.json()
        if isinstance(data, dict) and data.get("service") == "Credit Card Fraud Detection API":
            api_online = True
            st.sidebar.success("FastAPI Backend: Connected")
        else:
            st.sidebar.warning("FastAPI Backend: Connected to non-fraud service")
    else:
        st.sidebar.warning("FastAPI Backend: Unexpected response")
except Exception:
    st.sidebar.info("FastAPI Backend: Offline (Using direct in-process inference)")

detection_threshold = st.sidebar.slider(
    "Decision Threshold (Fraud Sensitivity)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Lower threshold catches more potential fraud (higher recall), while higher threshold minimizes false alarms (higher precision)."
)

@st.cache_data(show_spinner=False)
def convert_df_to_csv(df_to_export):
    """Cached CSV conversion to avoid repeated serializations during UI interactions."""
    return df_to_export.to_csv(index=False).encode("utf-8")

if "app_active_tab" not in st.session_state:
    st.session_state["app_active_tab"] = "Single Transaction Analysis"

selected_tab = st.radio(
    "Navigation Tab",
    ["Single Transaction Analysis", "Batch CSV Prediction"],
    horizontal=True,
    key="app_active_tab",
    label_visibility="collapsed"
)

if selected_tab == "Single Transaction Analysis":
    st.markdown("### Enter Transaction Details")

    with st.form("transaction_form"):
        st.subheader("Key Transaction Attributes")
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=0.0, step=1.0)
        with c2:
            time_val = st.number_input("Time (seconds elapsed)", min_value=0.0, value=0.0, step=1.0)

        st.subheader("PCA Transformed Features (V1 – V28)")
        
        with st.expander("Adjust V1 to V10 (Primary Components)", expanded=True):
            cols_v1 = st.columns(5)
            v_inputs = {}
            for i in range(1, 11):
                col_idx = (i - 1) % 5
                v_inputs[f"V{i}"] = cols_v1[col_idx].number_input(f"V{i}", value=0.0, format="%.4f")

        with st.expander("Adjust V11 to V20 (Secondary Components)", expanded=False):
            cols_v2 = st.columns(5)
            for i in range(11, 21):
                col_idx = (i - 11) % 5
                v_inputs[f"V{i}"] = cols_v2[col_idx].number_input(f"V{i}", value=0.0, format="%.4f")

        with st.expander("Adjust V21 to V28 (Residual Components)", expanded=False):
            cols_v3 = st.columns(4)
            for i in range(21, 29):
                col_idx = (i - 21) % 4
                v_inputs[f"V{i}"] = cols_v3[col_idx].number_input(f"V{i}", value=0.0, format="%.4f")

        submit_btn = st.form_submit_button("Run Fraud Prediction", use_container_width=True)

    if submit_btn:
        payload = {"Time": time_val, **v_inputs, "Amount": amount}
        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        
        fraud_prob = 0.0
        legit_prob = 1.0
        prediction_made = False

        if api_online:
            try:
                resp = requests.post(f"{api_url}/predict?threshold={detection_threshold}", json=payload, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    fraud_prob = data["fraud_probability"]
                    legit_prob = data["legitimate_probability"]
                    prediction_made = True
            except Exception as e:
                st.warning(f"FastAPI request error: {e}. Falling back to in-memory model.")

        if not prediction_made and local_model is not None:
            df_in = pd.DataFrame([[payload[c] for c in feature_order]], columns=feature_order)
            probs = local_model.predict_proba(df_in)[0]
            legit_prob = float(probs[0])
            fraud_prob = float(probs[1])
            prediction_made = True

        if prediction_made:
            is_fraud = fraud_prob >= detection_threshold
            st.markdown("---")
            st.subheader("Prediction Results")

            r1, r2 = st.columns([1, 1])
            with r1:
                if is_fraud:
                    st.markdown(f'''
                    <div class="alert-fraud">
                        FRAUD DETECTED<br>
                        <span style="font-size: 1rem; font-weight: normal;">This transaction has a high risk score exceeding the threshold ({detection_threshold:.2f})</span>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="alert-safe">
                        TRANSACTION LEGITIMATE<br>
                        <span style="font-size: 1rem; font-weight: normal;">Risk score is within safe operating limits (&lt; {detection_threshold:.2f})</span>
                    </div>
                    ''', unsafe_allow_html=True)

            with r2:
                st.metric("Fraud Probability", f"{fraud_prob * 100:.2f}%")
                st.metric("Legitimate Probability", f"{legit_prob * 100:.2f}%")
                st.progress(fraud_prob, text=f"Fraud Risk Meter: {fraud_prob * 100:.1f}%")

else:
    st.markdown("### Batch Transaction File Evaluation")
    uploaded_file = st.file_uploader("Upload CSV file with transactions (must include Time, V1-V28, Amount)", type=["csv"])

    if uploaded_file is None:
        st.session_state["batch_result_df"] = None
        st.session_state["raw_uploaded_df"] = None
        st.session_state["uploaded_file_name"] = None
    else:
        if st.session_state.get("uploaded_file_name") != uploaded_file.name or st.session_state.get("raw_uploaded_df") is None:
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["raw_uploaded_df"] = pd.read_csv(uploaded_file)
            st.session_state["batch_result_df"] = None

        batch_df = st.session_state["raw_uploaded_df"]
        
        st.success(f"File loaded: **{uploaded_file.name}** ({len(batch_df):,} rows, {len(batch_df.columns)} columns)")
        
        st.write(f"Preview of **{uploaded_file.name}** (First 5 records):")
        st.dataframe(batch_df.head(5), use_container_width=True)

        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        missing_cols = [c for c in feature_order if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required features in uploaded file: {missing_cols}")
        else:
            if st.button(f"Analyze {uploaded_file.name}"):
                with st.spinner(f"Analyzing all {len(batch_df):,} transactions in {uploaded_file.name}..."):
                    if local_model is not None:
                        X_input = batch_df[feature_order].copy()
                        probs = local_model.predict_proba(X_input)
                        
                        res_df = batch_df.copy()
                        res_df["Fraud_Probability"] = probs[:, 1].round(5)
                        res_df["Predicted_Class"] = (probs[:, 1] >= detection_threshold).astype(int)
                        res_df["Predicted_Status"] = np.where(res_df["Predicted_Class"] == 1, "FRAUD", "LEGITIMATE")
                        
                        st.session_state["batch_result_df"] = res_df

            if st.session_state.get("batch_result_df") is not None:
                result_df = st.session_state["batch_result_df"]
                
                fraud_mask = result_df["Fraud_Probability"] >= detection_threshold
                total_count = len(result_df)
                total_fraud = int(fraud_mask.sum())
                total_legit = total_count - total_fraud
                fraud_pct = (total_fraud / total_count) * 100 if total_count > 0 else 0

                st.markdown("---")
                st.subheader(f"Analysis Summary for {uploaded_file.name}")

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Transactions", f"{total_count:,}")
                col_m2.metric("Flagged as Fraud", f"{total_fraud:,} ({fraud_pct:.2f}%)")
                col_m3.metric("Verified Legitimate", f"{total_legit:,}")

                st.markdown("### Output Predictions on Uploaded Data")
                
                view_filter = st.radio(
                    "Filter View:",
                    [
                        f"All Transactions ({total_count:,})",
                        f"Fraud Only ({total_fraud:,})",
                        f"Legitimate Only ({total_legit:,})"
                    ],
                    key="batch_view_filter",
                    horizontal=True
                )
                
                if "Fraud Only" in view_filter:
                    filtered_df = result_df[fraud_mask]
                elif "Legitimate Only" in view_filter:
                    filtered_df = result_df[~fraud_mask]
                else:
                    filtered_df = result_df

                display_cols = ["Time", "Amount", "Fraud_Probability", "Predicted_Status"]
                other_cols = [c for c in filtered_df.columns if c not in display_cols and c not in ["Predicted_Class"]]
                ordered_display_cols = display_cols + other_cols
                
                preview_limit = 1000
                if len(filtered_df) > preview_limit:
                    st.dataframe(filtered_df[ordered_display_cols].head(preview_limit), use_container_width=True)
                    st.caption(f"Showing first **{preview_limit:,}** rows of **{len(filtered_df):,}** matches for instant performance. Download CSV for complete dataset.")
                else:
                    st.dataframe(filtered_df[ordered_display_cols], use_container_width=True)
                    st.caption(f"Showing all **{len(filtered_df):,}** rows ({view_filter}).")

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    full_csv = convert_df_to_csv(result_df)
                    st.download_button(
                        label=f"Download Full Analyzed CSV ({total_count:,} rows)",
                        data=full_csv,
                        file_name=f"analyzed_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                with dl_col2:
                    fraud_only_df = result_df[fraud_mask]
                    fraud_csv = convert_df_to_csv(fraud_only_df)
                    st.download_button(
                        label=f"Download Fraud Rows Only ({len(fraud_only_df):,} rows)",
                        data=fraud_csv,
                        file_name=f"fraud_only_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
