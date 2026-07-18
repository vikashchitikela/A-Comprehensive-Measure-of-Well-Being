"""
app.py

HDI Prediction System — Human Development Intelligence
Streamlit front-end for predicting a country's Human Development Index (HDI)
tier from its life expectancy, education, and income indicators.
"""

import os
import joblib
import pandas as pd
import streamlit as st

from hdi_utils import calculate_hdi

MODELS_DIR = "models"

st.set_page_config(
    page_title="HDI Prediction System",
    page_icon="🌍",
    layout="centered",
)

CATEGORY_COLORS = {
    "Very High": "#1a9850",
    "High": "#91cf60",
    "Medium": "#fee08b",
    "Low": "#d73027",
}


@st.cache_resource
def load_models():
    """Load the trained regressor/classifier if present; otherwise fall back
    to the formula-only calculation in hdi_utils."""
    reg_path = os.path.join(MODELS_DIR, "hdi_regressor.joblib")
    clf_path = os.path.join(MODELS_DIR, "hdi_classifier.joblib")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.joblib")

    if os.path.exists(reg_path) and os.path.exists(clf_path):
        regressor = joblib.load(reg_path)
        classifier = joblib.load(clf_path)
        feature_cols = joblib.load(cols_path)
        return regressor, classifier, feature_cols
    return None, None, None


def render_result(hdi_score: float, hdi_category: str, sub_indices: dict):
    color = CATEGORY_COLORS.get(hdi_category, "#333333")

    st.markdown(
        f"""
        <div style="border-radius: 12px; padding: 24px; text-align: center;
                    background-color: {color}20; border: 2px solid {color};">
            <h2 style="margin: 0; color: {color};">{hdi_category} Human Development</h2>
            <p style="font-size: 48px; font-weight: 700; margin: 8px 0; color: {color};">
                {hdi_score:.3f}
            </p>
            <p style="margin: 0; color: #555;">Predicted HDI Score</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Life Expectancy Index", f"{sub_indices['life_expectancy_index']:.3f}")
    col2.metric("Education Index", f"{sub_indices['education_index']:.3f}")
    col3.metric("Income Index", f"{sub_indices['income_index']:.3f}")


def main():
    st.title("🌍 HDI Prediction System")
    st.caption("Human Development Intelligence — predict a country's HDI tier "
               "from life expectancy, education, and income indicators.")

    st.markdown(
        "The **Human Development Index (HDI)** is a composite statistic of life "
        "expectancy, education, and per-capita income indicators used to rank "
        "countries into four tiers: **Very High, High, Medium, and Low** human "
        "development."
    )

    regressor, classifier, feature_cols = load_models()
    use_ml = regressor is not None and classifier is not None

    st.sidebar.header("Model")
    if use_ml:
        st.sidebar.success("ML models loaded (RandomForest)")
        prediction_mode = st.sidebar.radio(
            "Prediction method", ["ML Model", "Formula-based (UNDP method)"], index=0
        )
    else:
        st.sidebar.warning("No trained models found — using formula-based calculation.\n"
                            "Run `python train_model.py` to enable ML predictions.")
        prediction_mode = "Formula-based (UNDP method)"

    st.sidebar.header("Preset Scenarios")
    preset = st.sidebar.selectbox(
        "Load a scenario",
        ["Custom", "Very High — Developed Nation", "Medium — Emerging Economy",
         "Low — Needs Intervention"],
    )

    presets = {
        "Very High — Developed Nation": (82.0, 12.5, 16.5, 55000),
        "Medium — Emerging Economy": (68.0, 7.5, 11.0, 8000),
        "Low — Needs Intervention": (55.0, 3.5, 6.0, 1200),
    }
    default_le, default_mys, default_eys, default_gni = presets.get(
        preset, (70.0, 8.0, 12.0, 15000)
    )

    st.subheader("Country Indicators")
    col1, col2 = st.columns(2)
    with col1:
        life_expectancy = st.slider(
            "Life Expectancy at Birth (years)", 40.0, 90.0, float(default_le), 0.1
        )
        mean_years_schooling = st.slider(
            "Mean Years of Schooling", 0.0, 16.0, float(default_mys), 0.1
        )
    with col2:
        expected_years_schooling = st.slider(
            "Expected Years of Schooling", 0.0, 20.0, float(default_eys), 0.1
        )
        gni_per_capita = st.number_input(
            "GNI per Capita (PPP $)", min_value=100, max_value=150000,
            value=int(default_gni), step=100
        )

    if st.button("Predict HDI", type="primary", use_container_width=True):
        formula_result = calculate_hdi(
            life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita
        )

        if use_ml and prediction_mode == "ML Model":
            input_df = pd.DataFrame([{
                "life_expectancy": life_expectancy,
                "mean_years_schooling": mean_years_schooling,
                "expected_years_schooling": expected_years_schooling,
                "gni_per_capita": gni_per_capita,
            }])[feature_cols]

            hdi_score = float(regressor.predict(input_df)[0])
            hdi_category = classifier.predict(input_df)[0]
        else:
            hdi_score = formula_result["hdi_score"]
            hdi_category = formula_result["hdi_category"]

        render_result(hdi_score, hdi_category, formula_result)

        with st.expander("See raw indicator values"):
            st.json({
                "life_expectancy": life_expectancy,
                "mean_years_schooling": mean_years_schooling,
                "expected_years_schooling": expected_years_schooling,
                "gni_per_capita": gni_per_capita,
                "prediction_method": prediction_mode,
            })

    st.divider()
    st.caption(
        "This tool is for educational and research purposes. HDI classifications "
        "follow UNDP standard bands: Very High ≥ 0.800, High 0.700–0.799, "
        "Medium 0.550–0.699, Low < 0.550."
    )


if __name__ == "__main__":
    main()
