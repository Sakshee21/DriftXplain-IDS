from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="DriftXplain IDS", page_icon="🛡️", layout="wide")


def _resolve_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def _resolve_demo_inputs_csv() -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "demo_inputs.csv"


@st.cache_resource
def _load_artifacts():
    models_dir = _resolve_models_dir()
    model_path = models_dir / "rf_final.pkl"
    scaler_path = models_dir / "scaler.pkl"
    features_path = models_dir / "features.pkl"

    model_obj = joblib.load(model_path)
    scaler_obj = joblib.load(scaler_path)
    features_obj = joblib.load(features_path)
    return model_obj, scaler_obj, features_obj


@st.cache_resource
def _build_explainer(_model_obj):
    return shap.TreeExplainer(_model_obj)


@st.cache_data
def _load_demo_inputs(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _extract_class1_shap(shap_values_raw):
    if isinstance(shap_values_raw, list):
        if len(shap_values_raw) == 2:
            return np.array(shap_values_raw[1])
        return np.array(shap_values_raw[0])

    shap_values_arr = np.array(shap_values_raw)
    if shap_values_arr.ndim == 3:
        return shap_values_arr[:, :, 1]
    return shap_values_arr


def _extract_class1_base_value(explainer_obj):
    base_value = explainer_obj.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        if len(base_value) == 2:
            return float(base_value[1])
        return float(base_value[0])
    return float(base_value)


def _prepare_summary_source(demo_source: pd.DataFrame, feature_names_list):
    if demo_source.empty:
        return pd.DataFrame()
    if "case" in demo_source.columns:
        demo_source = demo_source.drop(columns=["case"])
    if any(feat not in demo_source.columns for feat in feature_names_list):
        return pd.DataFrame()
    return demo_source[feature_names_list].copy()


st.title("Intrusion Detection System")
st.caption("CICIDS2017 inference with threshold tuning and SHAP explainability")

try:
    model, scaler, features_artifact = _load_artifacts()
except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()

if not hasattr(scaler, "feature_names_in_"):
    st.error("Scaler does not expose feature_names_in_. Re-export scaler with named columns.")
    st.stop()

feature_names = list(scaler.feature_names_in_)
artifact_features = list(features_artifact)

if artifact_features != feature_names:
    st.warning("features.pkl differs from scaler schema. Using scaler feature contract for inference.")

if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(feature_names):
    st.error(
        f"Model expects {int(model.n_features_in_)} features, but scaler provides {len(feature_names)}."
    )
    st.stop()

threshold = st.sidebar.slider(
    "Attack decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.40,
    step=0.01,
)
st.sidebar.caption("Lower threshold catches more attacks, higher threshold reduces false alarms.")

demo_df = _load_demo_inputs(str(_resolve_demo_inputs_csv()))
if not demo_df.empty and "case" in demo_df.columns:
    missing_demo_cols = [feat for feat in feature_names if feat not in demo_df.columns]
    if missing_demo_cols:
        st.sidebar.warning("Demo CSV is missing one or more model feature columns.")
    else:
        preset_options = demo_df["case"].astype(str).tolist()
        selected_preset = st.sidebar.selectbox("Demo preset", options=preset_options, index=0)
        if st.sidebar.button("Load preset values"):
            selected_row = demo_df[demo_df["case"].astype(str) == selected_preset].iloc[0]
            for feat in feature_names:
                st.session_state[f"feat__{feat}"] = float(selected_row[feat])
            st.rerun()
elif "case" not in demo_df.columns and not demo_df.empty:
    st.sidebar.warning("Demo CSV found but column 'case' is missing.")

with st.expander("Feature schema diagnostics", expanded=False):
    st.write(f"Active features: {len(feature_names)}")
    st.write("First 5 features:", feature_names[:5])

st.subheader("Input Features")
st.write("Enter values for every model feature.")

input_data = {}
col_a, col_b = st.columns(2)
for idx, feat in enumerate(feature_names):
    target_col = col_a if idx % 2 == 0 else col_b
    field_key = f"feat__{feat}"
    if field_key not in st.session_state:
        st.session_state[field_key] = 0.0
    with target_col:
        input_data[feat] = st.number_input(feat, key=field_key, format="%.6f")

if st.button("Predict", type="primary"):
    try:
        df = pd.DataFrame([input_data], columns=feature_names)
        df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        st.stop()

    try:
        prob = float(model.predict_proba(df_scaled)[:, 1][0])
        pred = int(prob >= threshold)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    confidence = prob if pred == 1 else 1.0 - prob
    st.metric("Attack probability", f"{prob:.3f}")
    st.progress(int(max(0.0, min(1.0, confidence)) * 100))
    st.caption(f"Threshold used: {threshold:.2f}")

    if pred == 1:
        st.error(f"⚠️ ATTACK detected (probability {prob:.3f})")
    else:
        st.success(f"✅ Normal traffic (probability {prob:.3f})")

    st.subheader("SHAP Explanation")
    try:
        explainer = _build_explainer(model)
        shap_values_raw = explainer.shap_values(df_scaled)
        shap_values = _extract_class1_shap(shap_values_raw)
        base_value = _extract_class1_base_value(explainer)
    except Exception as exc:
        st.error(f"SHAP computation failed: {exc}")
        st.stop()

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots._waterfall.waterfall_legacy(
        base_value,
        shap_values[0],
        feature_names=feature_names,
        show=False,
    )
    st.pyplot(fig, clear_figure=True)

    summary_source = _prepare_summary_source(demo_df, feature_names)
    if not summary_source.empty and len(summary_source) >= 2:
        try:
            summary_scaled = pd.DataFrame(
                scaler.transform(summary_source), columns=feature_names
            )
            summary_shap_raw = explainer.shap_values(summary_scaled)
            summary_shap = _extract_class1_shap(summary_shap_raw)

            st.subheader("SHAP Summary (Global Pattern)")
            fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
            shap.summary_plot(
                summary_shap,
                summary_scaled,
                feature_names=feature_names,
                max_display=10,
                show=False,
            )
            st.pyplot(fig_summary, clear_figure=True)
            st.caption("This summary is computed from rows in tests/demo_inputs.csv.")
        except Exception as exc:
            st.warning(f"SHAP summary plot skipped: {exc}")
    else:
        st.info("Add at least two valid rows in tests/demo_inputs.csv to render SHAP summary plot.")

    contrib_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Contribution": shap_values[0],
        }
    )
    top_pos = contrib_df.sort_values("SHAP Contribution", ascending=False).head(5)
    top_neg = contrib_df.sort_values("SHAP Contribution", ascending=True).head(5)

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("**Top Positive Contributions**")
        st.dataframe(top_pos, use_container_width=True)
    with neg_col:
        st.markdown("**Top Negative Contributions**")
        st.dataframe(top_neg, use_container_width=True)