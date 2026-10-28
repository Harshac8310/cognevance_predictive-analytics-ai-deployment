"""Streamlit dashboard for churn exploration and interactive predictions."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_churn_demo.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_comparison.csv"

st.set_page_config(page_title="Churn Intelligence", page_icon="📊", layout="wide")
st.title("📊 Churn Intelligence")
st.caption("Predictive analytics demo • Synthetic data by default • Not for consequential decisions")

if not DATA_PATH.exists():
    st.warning("Demo dataset not found. From the project root, run: `python -m src.generate_demo_data`")
    st.stop()
df = pd.read_csv(DATA_PATH)
churn_rate = (df["churn"].astype(str).str.lower() == "yes").mean() * 100
c1, c2, c3 = st.columns(3)
c1.metric("Customers in dataset", f"{len(df):,}")
c2.metric("Observed churn rate", f"{churn_rate:.1f}%")
c3.metric("Features", str(df.shape[1] - 2))

left, right = st.columns(2)
with left:
    st.subheader("Churn distribution")
    churn_counts = df["churn"].value_counts().rename_axis("Churn").reset_index(name="Customers")
    st.plotly_chart(px.pie(churn_counts, names="Churn", values="Customers", hole=0.45), use_container_width=True)
with right:
    st.subheader("Churn by contract")
    contract_churn = df.assign(churn_yes=(df["churn"].astype(str).str.lower() == "yes").astype(int)).groupby("contract", as_index=False)["churn_yes"].mean()
    contract_churn["Churn rate (%)"] = contract_churn["churn_yes"] * 100
    st.plotly_chart(px.bar(contract_churn, x="contract", y="Churn rate (%)", color="contract"), use_container_width=True)

st.subheader("Model comparison")
if METRICS_PATH.exists():
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns if c != "model"}), use_container_width=True)
    image_path = ROOT / "reports" / "figures" / "model_comparison.png"
    if image_path.exists():
        st.image(str(image_path), caption="Metrics calculated on a held-out test set")
else:
    st.info("Train models first to populate metrics: `python -m src.train`")

st.subheader("Try a customer prediction")
if not MODEL_PATH.exists():
    st.warning("No trained model found. Run `python -m src.train` from the project root first.")
else:
    model = joblib.load(MODEL_PATH)
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        tenure = col1.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = col2.number_input("Monthly charges", min_value=1.0, max_value=10000.0, value=75.0)
        total_charges = col3.number_input("Total charges", min_value=0.0, max_value=1000000.0, value=900.0)
        contract = col1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = col2.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        tech_support = col3.selectbox("Tech support", ["Yes", "No", "No internet service"])
        payment_method = col1.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        paperless_billing = col2.selectbox("Paperless billing", ["Yes", "No"])
        senior_citizen = col3.selectbox("Senior citizen", [0, 1])
        submitted = st.form_submit_button("Predict churn", type="primary")
    if submitted:
        row = pd.DataFrame([{
            "tenure": tenure, "monthly_charges": monthly_charges, "total_charges": total_charges,
            "contract": contract, "internet_service": internet_service, "tech_support": tech_support,
            "payment_method": payment_method, "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen
        }])
        prob = float(model.predict_proba(row)[0, 1])
        st.metric("Estimated churn probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.write(f"**Prediction:** {'Likely to churn' if prob >= 0.5 else 'Less likely to churn'}")
        st.caption("Predictions are estimates based on the training data, not guarantees. Synthetic demo data may not represent real customers.")
