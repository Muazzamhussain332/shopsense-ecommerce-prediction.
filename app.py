import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ShopSense | Purchase Prediction", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem;}
.hero {padding: 1.55rem 1.7rem; border-radius: 18px; background: linear-gradient(115deg,#132945,#285b82); color:white; margin-bottom:1.35rem;}
.hero h1 {font-size:2rem; margin:0 0 .3rem 0; color:white;}
.hero p {margin:0; color:#e2edf7; font-size:1.03rem;}
.small-muted {color:#6b7280; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('''<div class="hero"><h1>🛍️ ShopSense</h1><p>E-commerce Purchase Intent • AI-powered session prediction dashboard</p></div>''', unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent / "ecommerce_purchase_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.error("Model file missing. Colab se ecommerce_purchase_model.pkl download karke app.py ke saath same folder mein rakhein.")
    st.stop()

with st.sidebar:
    st.header("Prediction settings")
    threshold = st.slider("Purchase threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Higher threshold generally gives fewer positive predictions; this does not retrain the model.")
    st.caption("Demo MVP • Scores are model estimates, not guaranteed purchases.")

st.subheader("Enter visitor session details")
st.caption("Training data ke 15 input .")

with st.form("prediction_form"):
    st.markdown("#### Browsing activity")
    a, b, c = st.columns(3)
    with a:
        administrative = st.number_input("Administrative pages", 0, 500, 2, step=1)
        administrative_duration = st.number_input("Administrative duration (sec)", 0.0, 1_000_000.0, 30.0, step=10.0)
    with b:
        informational = st.number_input("Informational pages", 0, 500, 0, step=1)
        informational_duration = st.number_input("Informational duration (sec)", 0.0, 1_000_000.0, 0.0, step=10.0)
    with c:
        product_related = st.number_input("Product-related pages", 0, 10000, 20, step=1)
        product_duration = st.number_input("Product-related duration (sec)", 0.0, 1_000_000.0, 600.0, step=30.0)

    st.markdown("#### Engagement & session context")
    d, e, f = st.columns(3)
    with d:
        bounce_rates = st.number_input("Bounce rate", 0.0, 1.0, 0.02, 0.01, format="%.3f")
        exit_rates = st.number_input("Exit rate", 0.0, 1.0, 0.05, 0.01, format="%.3f")
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
    with e:
        operating_systems = st.number_input("Operating system code", min_value=1, max_value=100, value=2)
        browser = st.number_input("Browser code", min_value=1, max_value=100, value=2)
        region = st.number_input("Region code", min_value=1, max_value=100, value=1)
    with f:
        traffic_type = st.number_input("Traffic type code", min_value=1, max_value=100, value=2)
        visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = st.toggle("Weekend session", value=False)

    submitted = st.form_submit_button("Predict purchase intent", type="primary", use_container_width=True)

if submitted:
    sample = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend,
    }])
    try:
        # The saved pipeline includes preprocessing and classifier.
        purchase_probability = float(model.predict_proba(sample)[0, 1])
        prediction = int(purchase_probability >= threshold)
    except Exception as exc:
        st.error(f"Prediction failed. Check your model file and feature schema. Details: {exc}")
        st.stop()

    st.divider()
    st.subheader("Prediction result")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated purchase probability", f"{purchase_probability:.1%}")
        st.metric("Decision threshold", f"{threshold:.0%}")
    with right:
        st.progress(purchase_probability)
        if prediction:
            st.success("Likely to purchase (at the selected threshold)")
        else:
            st.info("Unlikely to purchase (at the selected threshold)")
        st.caption("This is the model's estimated score, not a calibrated guarantee or a statement of customer intent.")

    with st.expander("View the exact input sent to the model"):
        st.dataframe(sample, use_container_width=True, hide_index=True)

st.divider()
st.caption("ShopSense • Educational ML prototype. For production SaaS, add authentication, session ingestion, database, monitoring and privacy protections.")
