import os
import re
import traceback
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Google Reviews Forecast", page_icon="📈", layout="centered")

DATE_COL = "ds"
ID_COL = "unique_id"

MODEL_CANDIDATES = [
    "google_reviews_forecast_model.pkl",
    "Models/Amazon/google_reviews_forecast_model.pkl",
    "model.pkl"
]

FEATURE_CANDIDATES = [
    "google_reviews_feature_columns.pkl",
    "Models/Amazon/google_reviews_feature_columns.pkl",
    "feature_columns.pkl"
]

def find_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None

@st.cache_resource
def load_artifacts():
    model_path = find_existing_path(MODEL_CANDIDATES)
    feature_path = find_existing_path(FEATURE_CANDIDATES)

    if model_path is None:
        raise FileNotFoundError(f"Model file not found. Checked: {MODEL_CANDIDATES}")
    if feature_path is None:
        raise FileNotFoundError(f"Feature file not found. Checked: {FEATURE_CANDIDATES}")

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    if not isinstance(feature_cols, (list, tuple)):
        raise ValueError("Feature columns pickle must contain a list or tuple of feature names.")

    return model, list(feature_cols), model_path, feature_path

def build_feature_row(pred_dt, unique_id, history_values, feature_cols):
    row = {}

    hour = pred_dt.hour
    day = pred_dt.day
    dayofweek = pred_dt.dayofweek
    dayofyear = pred_dt.dayofyear
    weekofyear = int(pred_dt.isocalendar().week)
    month = pred_dt.month
    quarter = pred_dt.quarter
    is_weekend = int(dayofweek in [5, 6])

    row["hour"] = hour
    row["day"] = day
    row["dayofweek"] = dayofweek
    row["dayofyear"] = dayofyear
    row["weekofyear"] = weekofyear
    row["month"] = month
    row["quarter"] = quarter
    row["is_weekend"] = is_weekend

    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    row["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)

    if "unique_id" in feature_cols:
        row["unique_id"] = unique_id

    history_arr = np.array(history_values, dtype=float)

    lag_requirements = {
        "lag_1": 1,
        "lag_2": 2,
        "lag_3": 3,
        "lag_6": 6,
        "lag_12": 12,
        "lag_24": 24,
        "lag_48": 48,
        "lag_72": 72,
    }

    for lag_name, lag_num in lag_requirements.items():
        if lag_name in feature_cols:
            row[lag_name] = history_arr[-lag_num]

    for window in [3, 6, 12, 24]:
        vals = history_arr[-window:]
        if f"rolling_mean_{window}" in feature_cols:
            row[f"rolling_mean_{window}"] = float(np.mean(vals))
        if f"rolling_std_{window}" in feature_cols:
            row[f"rolling_std_{window}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if f"rolling_min_{window}" in feature_cols:
            row[f"rolling_min_{window}"] = float(np.min(vals))
        if f"rolling_max_{window}" in feature_cols:
            row[f"rolling_max_{window}"] = float(np.max(vals))

    final_row = {col: row.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([final_row])

def parse_history_input(history_text):
    values = [float(x) for x in re.split(r"[,\s]+", history_text.strip()) if x]
    return values

def main():
    st.title("📈 Google Reviews Forecast App")
    st.write("Predict the next target value directly from saved `.pkl` files without uploading a CSV.")

    try:
        model, feature_cols, model_path, feature_path = load_artifacts()
    except Exception as e:
        st.error(f"Artifact loading failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

    st.success("Artifacts loaded successfully")

    with st.expander("Loaded artifact details"):
        st.write(f"Model path: {model_path}")
        st.write(f"Feature file path: {feature_path}")
        st.write(f"Total features expected by model: {len(feature_cols)}")
        st.write(feature_cols)

    st.subheader("Prediction Inputs")

    unique_id = st.number_input("Unique ID", min_value=0, value=0, step=1)
    pred_date = st.date_input("Prediction date")
    pred_time = st.time_input("Prediction time")

    st.markdown("### Historical target values")
    st.caption("Enter recent y values separated by commas, spaces, or new lines.")

    history_text = st.text_area(
        "Recent y history",
        value="",
        height=220,
        placeholder="204\n149\n175\n196\n282\n493"
    )

    if st.button("Predict"):
        try:
            if not history_text.strip():
                st.error("Please enter historical y values.")
                st.stop()

            history_values = parse_history_input(history_text)

            min_required = 1
            if "lag_72" in feature_cols:
                min_required = 72
            elif "lag_48" in feature_cols:
                min_required = 48
            elif "lag_24" in feature_cols:
                min_required = 24
            elif "lag_12" in feature_cols:
                min_required = 12
            elif "lag_6" in feature_cols:
                min_required = 6
            elif "lag_3" in feature_cols:
                min_required = 3
            elif "lag_2" in feature_cols:
                min_required = 2

            if len(history_values) < min_required:
                st.error(f"This model needs at least {min_required} historical y values, but you provided {len(history_values)}.")
                st.stop()

            pred_dt = pd.Timestamp.combine(pred_date, pred_time)

            input_df = build_feature_row(pred_dt, unique_id, history_values, feature_cols)
            prediction = model.predict(input_df)[0]

            st.subheader("Prediction Output")
            st.success(f"Predicted y = {prediction:.4f}")

            result_df = pd.DataFrame({
                "unique_id": [unique_id],
                "ds": [pred_dt],
                "predicted_y": [prediction]
            })
            st.dataframe(result_df, use_container_width=True)

            st.subheader("Derived feature values used for prediction")
            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()