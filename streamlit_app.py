import io
import os
import pickle
import sys

import gdown
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

from model import GraphSAGE

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.makedirs("artifacts", exist_ok=True)

PREPROCESS_URL = "https://drive.google.com/uc?id=1Uds7ZTU_8NBCHzE2bMGUKovBLIxX0KRg"
MODEL_PATH = "model.pt"
PREPROCESS_PATH = "artifacts/preprocess.pkl"
EXPECTED_TYPE_COLS = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
FEATURE_COLS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    *EXPECTED_TYPE_COLS,
]

# Image assets (external CDN/Unsplash previews suitable for prototyping)
HERO_IMAGE_URL = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=1400&q=80"
LOGO_URL = "https://images.unsplash.com/photo-1545239351-1141bd82e8a6?auto=format&fit=crop&w=400&q=80"
REPORT_IMAGE_URL = "https://images.unsplash.com/photo-1559526324-593bc073d938?auto=format&fit=crop&w=1200&q=80"

st.set_page_config(page_title="Fraud Risk Intelligence Console", page_icon="🛡️", layout="wide")
st.config.set_option("server.maxUploadSize", 200)


def inject_css():
    st.markdown(f"""
        <style>
            :root {{
                --bg: #07111f;
                --panel: rgba(7, 17, 31, 0.72);
                --panel-strong: rgba(10, 25, 46, 0.95);
                --accent: #4f7cff;
                --accent-2: #22c55e;
                --text: #f6f8fc;
                --muted: #9fb0ca;
                --border: rgba(255, 255, 255, 0.12);
            }}
            .stApp {{
                background: radial-gradient(circle at top left, rgba(79, 124, 255, 0.18), transparent 28%),
                            linear-gradient(135deg, #040816 0%, #07111f 45%, #0d1728 100%);
            }}
            .block-container {{
                padding-top: 88px !important; /* ensure top header does not overlap hero */
                padding-bottom: 2.4rem;
                max-width: 1400px;
            }}
            /* Some Streamlit toolbars overlay the top; add a small negative margin to the header area to ensure hero content is visible */
            .stApp > header {{
                z-index: 9999;
            }}
            .hero-card, .section-card, .sidebar-card {{
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1.1rem 1.2rem;
                background: var(--panel);
                backdrop-filter: blur(12px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25);
            }}
            .hero-card {{
                background: linear-gradient(135deg, rgba(79, 124, 255, 0.22), rgba(34, 197, 94, 0.16));
                margin-bottom: 1rem;
                background-image: url('{HERO_IMAGE_URL}');
                background-size: cover;
                background-position: center center;
                background-blend-mode: overlay;
            }}
            .hero-card h1, .hero-card p {{
                color: var(--text) !important;
            }}
            .kpi-pill {{
                display: inline-block;
                background: rgba(79, 124, 255, 0.12);
                border: 1px solid rgba(79, 124, 255, 0.38);
                border-radius: 999px;
                padding: 0.28rem 0.7rem;
                color: #dfe7ff;
                font-size: 0.82rem;
                margin-right: 0.4rem;
                margin-bottom: 0.4rem;
            }}
            .stButton > button {{
                background: linear-gradient(135deg, var(--accent), #5f8fff);
                color: white;
                border: none;
                border-radius: 999px;
                padding: 0.55rem 1.15rem;
                font-weight: 600;
            }}
            .stButton > button:hover {{
                background: linear-gradient(135deg, #3566eb, #4f7cff);
                box-shadow: 0 10px 25px rgba(79, 124, 255, 0.25);
            }}
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stSelectbox > div > div > div,
            .stFileUploader > section {{
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 12px;
            }}
            .stDataFrame, .stTable {{
                border-radius: 16px;
                overflow: hidden;
            }}
            div[data-testid="stMetric"] {{
                background: rgba(255, 255, 255, 0.045);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 16px;
                padding: 0.7rem 0.9rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_feature_frame(df, preprocessor):
    if preprocessor is None:
        raise ValueError("Preprocessing artifact is unavailable.")

    frame = df.copy()
    if "type" in frame.columns:
        df_type = pd.get_dummies(frame["type"], prefix="type")
        for col in EXPECTED_TYPE_COLS:
            if col not in df_type.columns:
                df_type[col] = 0
        frame = pd.concat([frame.drop("type", axis=1), df_type[EXPECTED_TYPE_COLS]], axis=1)

    numeric_cols = preprocessor["scaler"].feature_names_in_
    frame[numeric_cols] = preprocessor["scaler"].transform(frame[numeric_cols])

    for col in FEATURE_COLS:
        if col not in frame.columns:
            frame[col] = 0

    return frame[FEATURE_COLS].astype(np.float32)


def predict_probabilities(model, frame):
    if model is None:
        raise ValueError("Model is unavailable.")

    X_tensor = torch.tensor(frame.values.astype(np.float32))
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy()
    return probs


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model artifact not found. Add model.pt to the project root before running predictions.")
        return None

    try:
        model = GraphSAGE(in_channels=11, hidden_channels=64)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    except Exception as exc:
        st.warning(f"Unable to load the model: {exc}")
        return None


@st.cache_resource
def load_preprocessor():
    if not os.path.exists(PREPROCESS_PATH):
        try:
            gdown.download(PREPROCESS_URL, PREPROCESS_PATH, quiet=False)
        except Exception as exc:
            st.warning(f"Unable to download preprocessing artifact: {exc}")
            return None

    if not os.path.exists(PREPROCESS_PATH):
        st.warning("Preprocessing artifact is unavailable. Upload artifacts/preprocess.pkl manually if needed.")
        return None

    try:
        with open(PREPROCESS_PATH, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        st.warning(f"Unable to read preprocessing artifact: {exc}")
        return None


def ensure_runtime_ready():
    if "runtime_model" not in st.session_state:
        st.session_state.runtime_model = None
    if "runtime_preprocessor" not in st.session_state:
        st.session_state.runtime_preprocessor = None

    if st.session_state.runtime_model is None:
        st.session_state.runtime_model = load_model()
    if st.session_state.runtime_preprocessor is None:
        st.session_state.runtime_preprocessor = load_preprocessor()

    return st.session_state.runtime_model, st.session_state.runtime_preprocessor


inject_css()


def render_sidebar():
    # Logo + short brand card
    try:
        st.sidebar.image(LOGO_URL, width=140)
    except Exception:
        st.sidebar.markdown("<div class='sidebar-card'><h2>🛡️ SecureGuard AI</h2><p>Enterprise fraud defense for modern financial operations.</p></div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Workspace**")
    app_mode = st.sidebar.radio("Choose a mode", ["Manual Review", "Bulk Analytics"], key="workspace")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Operational readiness")
    st.sidebar.caption("Real-time scoring, high-volume batch review, and downloadable reporting.")
    st.sidebar.metric("Model", "GraphSAGE", "PyTorch")
    st.sidebar.metric("Input", "Transaction vectors", "Risk scored")
    return app_mode


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="kpi-pill">Enterprise-grade monitoring</div>
            <div class="kpi-pill">Explainable scorecards</div>
            <div class="kpi-pill">Secure review workflows</div>
            <h1>Fraud Risk Intelligence Console</h1>
            <p>Deliver a premium, decision-ready experience for fraud analysts with instant risk scoring, batch analytics, and polished reporting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_summary(df, threshold):
    total_transactions = len(df)
    high_risk_count = int((df["predicted_fraud"] == "Fraud").sum())
    average_risk = float(df["fraud_probability"].mean()) if total_transactions else 0.0
    peak_risk = float(df["fraud_probability"].max()) if total_transactions else 0.0
    high_risk_rate = (high_risk_count / total_transactions * 100) if total_transactions else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Transactions reviewed", f"{total_transactions:,}")
    with col2:
        st.metric("High-risk alerts", f"{high_risk_count:,}", f"{high_risk_rate:.1f}% of batch")
    with col3:
        st.metric("Average fraud score", f"{average_risk:.2%}")
    with col4:
        st.metric("Peak score", f"{peak_risk:.2%}")

    st.markdown(
        f"""
        <div class="section-card">
            <strong>Current review threshold:</strong> {threshold:.2f}<br>
            <strong>Analyst guidance:</strong> Items above this threshold are routed into the active review queue.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_manual_review():
    st.markdown(
        """
        <div class="section-card">
            <h3>Single transaction assessment</h3>
            <p>Capture one transaction and receive an executive-ready fraud risk score instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            payment_type = st.selectbox(
                "Payment Type",
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                key="manual_payment_type",
            )
            step = st.number_input("Step", min_value=0, step=1, key="manual_step")
            amount = st.number_input("Amount", min_value=0.0, step=0.01, key="manual_amount")
            oldbalanceOrg = st.number_input("Sender Balance Before", min_value=0.0, step=0.01, key="manual_oldbalanceOrg")
        with col2:
            newbalanceOrig = st.number_input("Sender Balance After", min_value=0.0, step=0.01, key="manual_newbalanceOrig")
            oldbalanceDest = st.number_input("Receiver Balance Before", min_value=0.0, step=0.01, key="manual_oldbalanceDest")
            newbalanceDest = st.number_input("Receiver Balance After", min_value=0.0, step=0.01, key="manual_newbalanceDest")
            manual_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.5, key="manual_threshold_slider")

        submit_manual = st.form_submit_button("Run risk assessment", use_container_width=True)

    if submit_manual:
        model, preprocessor = ensure_runtime_ready()
        if model is None or preprocessor is None:
            st.warning("The scoring runtime is not ready yet. The app will remain usable, but prediction features are currently unavailable.")
            return

        input_frame = pd.DataFrame(
            [
                {
                    "step": step,
                    "amount": amount,
                    "oldbalanceOrg": oldbalanceOrg,
                    "newbalanceOrig": newbalanceOrig,
                    "oldbalanceDest": oldbalanceDest,
                    "newbalanceDest": newbalanceDest,
                    "type": payment_type,
                }
            ]
        )
        prepared = build_feature_frame(input_frame, preprocessor)
        probs = predict_probabilities(model, prepared)
        prob = float(probs[0])
        prediction = "Fraud" if prob > manual_threshold else "Not Fraud"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{prob:.2%}")
        with col2:
            st.metric("Decision", prediction)
        with col3:
            st.metric("Threshold", f"> {manual_threshold:.2f}")

        st.progress(min(prob, 1.0))
        if prob >= 0.8:
            st.error("High-risk transaction. Escalate for immediate review.")
        elif prob >= 0.5:
            st.warning("Moderate risk. Queue for analyst verification.")
        else:
            st.success("Low risk. Continue standard processing flow.")


def render_bulk_analytics():
    st.markdown(
        """
        <div class="section-card">
            <h3>Batch review workflow</h3>
            <p>Upload a transaction file for high-volume scoring, analytics, and export-ready summaries.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            model, preprocessor = ensure_runtime_ready()
            if model is None or preprocessor is None:
                st.warning("The scoring runtime is not ready yet. Uploading data is still possible, but scoring is currently unavailable.")
                return

            uploaded_file.seek(0)
            chunksize = 100000
            results = []
            progress_bar = st.progress(0)
            total_rows = sum(1 for _ in uploaded_file) - 1
            uploaded_file.seek(0)

            processed_rows = 0
            for chunk in pd.read_csv(uploaded_file, chunksize=chunksize):
                processed_rows += len(chunk)
                prepared = build_feature_frame(chunk, preprocessor)
                probs = predict_probabilities(model, prepared)
                chunk["fraud_probability"] = probs
                results.append(chunk)

                if total_rows > 0:
                    progress = min(1.0, processed_rows / total_rows)
                    progress_bar.progress(progress)

            df = pd.concat(results, ignore_index=True)
            threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5)
            df["predicted_fraud"] = np.where(df["fraud_probability"] > threshold, "Fraud", "Not Fraud")

            st.success(f"Processed {len(df):,} transactions successfully.")
            render_dashboard_summary(df, threshold)
            tab_results, tab_analytics, tab_downloads = st.tabs(["Results", "Analytics", "Downloads"])

            with tab_results:
                st.subheader("Priority review queue")
                review_queue = df.sort_values("fraud_probability", ascending=False).copy()
                review_columns = ["fraud_probability", "predicted_fraud"]
                if "amount" in review_queue.columns:
                    review_columns.append("amount")
                review_queue = review_queue[review_columns].head(40)
                st.dataframe(review_queue, use_container_width=True)
                st.subheader("High-risk transactions")
                st.dataframe(df[df["predicted_fraud"] == "Fraud"].head(50), use_container_width=True)

            with tab_analytics:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df["fraud_probability"], bins=40, kde=True, ax=ax)
                ax.set_title("Fraud probability distribution")
                ax.set_xlabel("Fraud probability")
                ax.set_ylabel("Count")
                st.pyplot(fig)

                if "isFraud" in df.columns:
                    y_true = df["isFraud"].astype(int)
                    y_pred = np.where(df["fraud_probability"] > threshold, 1, 0)
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(5.5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Confusion matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                    fpr, tpr, _ = roc_curve(y_true, df["fraud_probability"])
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots(figsize=(5.5, 4))
                    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
                    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                    ax.set_title("ROC curve")
                    ax.set_xlabel("False positive rate")
                    ax.set_ylabel("True positive rate")
                    ax.legend(loc="lower right")
                    st.pyplot(fig)

                    st.subheader("Classification report")
                    st.json(classification_report(y_true, y_pred, output_dict=True))

            with tab_downloads:
                # Executive preview (image) and quick download
                try:
                    st.image(REPORT_IMAGE_URL, caption="Executive summary (preview)", use_column_width=True)
                except Exception:
                    pass

                try:
                    resp = requests.get(REPORT_IMAGE_URL, timeout=8)
                    if resp.status_code == 200:
                        st.download_button(
                            label="Download executive preview (PNG)",
                            data=resp.content,
                            file_name="executive_preview.png",
                            mime="image/png",
                        )
                except Exception:
                    # ignore network errors for preview download
                    pass

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

                buf = io.BytesIO()
                fig = plt.figure(figsize=(7, 4))
                sns.histplot(df["fraud_probability"], bins=40, kde=True)
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    label="Download histogram as PNG",
                    data=buf,
                    file_name="fraud_probability_histogram.png",
                    mime="image/png",
                )

        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Error processing file: {exc}")


app_mode = render_sidebar()
render_header()

if app_mode == "Manual Review":
    render_manual_review()
else:
    render_bulk_analytics()

