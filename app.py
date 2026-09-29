from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.carvalue_utils import make_user_input

st.set_page_config(page_title="CarValue AI", page_icon="🚗", layout="wide")

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
OPTIONS_PATH = PROJECT_ROOT / "artifacts" / "web_options.json"
META_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if not MODEL_PATH.exists() or not OPTIONS_PATH.exists():
    st.error("Model artifacts are missing. Run notebooks 05 and 06 first.")
    st.stop()

model = load_model()
options = load_json(OPTIONS_PATH)
meta = load_json(META_PATH) if META_PATH.exists() else {}

st.title("CarValue AI")
st.subheader("Used Car Asking Price Estimator")
st.caption("Estimate the online asking price of a used car from observable listing attributes.")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("### Vehicle information")
    brands = sorted(options["brand_models"].keys())
    brand = st.selectbox("Brand", brands)
    models = options["brand_models"].get(brand, [])
    model_name = st.selectbox("Model", models)

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input(
            "Model year",
            min_value=int(options["year_min"]),
            max_value=int(options["year_max"]),
            value=min(int(options["reference_year"]), int(options["year_max"])),
            step=1,
        )
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=1_000_000, value=30_000, step=1_000)
        origin = st.selectbox("Origin", options["origin"])
        body_type = st.selectbox("Body type", options["body_type"])
        transmission = st.selectbox("Transmission", options["transmission"])
    with c2:
        fuel_type = st.selectbox("Fuel type", options["fuel_type"])
        engine_default = min(max(2.0, float(options["engine_size_min"])), float(options["engine_size_max"]))
        engine_size = st.number_input(
            "Engine size (L)",
            min_value=0.0,
            max_value=max(10.0, float(options["engine_size_max"]) + 1.0),
            value=float(engine_default),
            step=0.1,
            format="%.1f",
        )
        exterior_color = st.selectbox("Exterior color", options["exterior_color"])
        seat_choices = options.get("seats", [5]) or [5]
        seats = st.selectbox("Seats", seat_choices, index=seat_choices.index(5) if 5 in seat_choices else 0)
        drivetrain = st.selectbox("Drivetrain", options["drivetrain"])

    predict_clicked = st.button("Estimate price", type="primary", use_container_width=True)

with right:
    st.markdown("### Prediction")
    if predict_clicked:
        x = make_user_input(
            reference_year=int(options["reference_year"]),
            brand=brand,
            model=model_name,
            year=int(year),
            mileage_km=float(mileage),
            origin=origin,
            body_type=body_type,
            transmission=transmission,
            fuel_type=fuel_type,
            engine_size_l=float(engine_size),
            exterior_color=exterior_color,
            seats=float(seats),
            drivetrain=drivetrain,
        )
        price = float(model.predict(x)[0])
        st.metric("Estimated asking price", f"{price:,.0f} VND")
        st.write(f"Approximately **{price / 1e6:,.1f} million VND**")
        st.caption(f"Final model: {meta.get('selected_model', getattr(model, 'name', 'Model'))}")

        with st.expander("Model input used"):
            st.dataframe(x, use_container_width=True)
    else:
        st.info("Enter the vehicle information and click **Estimate price**.")

st.divider()
st.caption(
    "Important: this model estimates the ONLINE ASKING PRICE in the Bonbanh marketplace sample. "
    "It does not observe or guarantee the final negotiated transaction price."
)
