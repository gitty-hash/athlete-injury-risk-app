"""Athlete injury risk predictor - Streamlit inference app (MIS 545 final project, 3rd and Vector).

Loads the saved logistic regression pipeline and predicts whether an athlete's NEXT session
will be labelled high risk, from the current session's readings and their recent history.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="Athlete Injury Risk Predictor", page_icon="🏃", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "injury_risk_model.joblib")
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    return model, info


model, info = load_model()
RANGES = info["ranges"]
ROLL_COLS = info["roll_cols"]

LABELS = {
    "heart_rate": "Heart rate (bpm)",
    "body_temperature": "Body temperature (°C)",
    "hydration_level": "Hydration level (%)",
    "sleep_quality": "Sleep quality (1-10)",
    "recovery_score": "Recovery score (0-100)",
    "stress_level": "Stress level (0-1)",
    "muscle_activity": "Muscle activity (EMG)",
    "joint_angles": "Joint angle (°)",
    "gait_speed": "Gait speed (m/s)",
    "cadence": "Cadence (steps/min)",
    "step_count": "Step count",
    "jump_height": "Jump height (m)",
    "ground_reaction_force": "Ground reaction force (N)",
    "range_of_motion": "Range of motion (°)",
    "ambient_temperature": "Ambient temperature (°C)",
    "humidity": "Humidity (%)",
    "altitude": "Altitude (m)",
    "training_intensity": "Training intensity (1-10)",
    "training_duration": "Training duration (min)",
    "training_load": "Training load",
    "fatigue_index": "Fatigue index",
    "age": "Age",
    "bmi": "BMI",
}
for col in ROLL_COLS:
    base = LABELS[col].split(" (")[0]
    LABELS[f"{col}_roll5"] = f"{base}: average of last 5 sessions"
    LABELS[f"{col}_3ago"] = f"{base}: 3 sessions ago"
    LABELS[f"{col}_trend3"] = f"{base}: change vs 3 sessions ago"

GROUPS = {
    "Recovery & wellness": ["recovery_score", "sleep_quality", "stress_level", "hydration_level",
                            "heart_rate", "body_temperature"],
    "Training": ["training_intensity", "training_duration", "training_load", "fatigue_index"],
    "Biomechanics": ["muscle_activity", "joint_angles", "gait_speed", "cadence", "step_count",
                     "jump_height", "ground_reaction_force", "range_of_motion"],
    "Environment": ["ambient_temperature", "humidity", "altitude"],
    "Recent history": [f"{c}_roll5" for c in ROLL_COLS] + [f"{c}_3ago" for c in ROLL_COLS],
}
SLIDER_COLS = ["age", "bmi"] + [c for cols in GROUPS.values() for c in cols]
CATEGORIES = info["categories"]


# ------------------------------------------------------------------ input state + presets
def preset_values(name):
    """Slider values for a preset: medians of the training data, or a real held-out session."""
    values = {c: RANGES[c]["median"] for c in SLIDER_COLS}
    values.update({"sport_type": CATEGORIES["sport_type"][0], "gender": CATEGORIES["gender"][0],
                   "playing_surface": CATEGORIES["playing_surface"][0]})
    if name in info["examples"]:
        values.update({k: v for k, v in info["examples"][name].items() if k in values and v is not None})
    for c in SLIDER_COLS:  # keep every value inside its slider's range
        r = RANGES[c]
        values[c] = min(max(values[c], r["min"]), r["max"])
        values[c] = int(round(values[c])) if r["is_int"] else float(values[c])
    return values


def apply_preset():
    for k, v in preset_values(st.session_state["preset"]).items():
        st.session_state[f"in_{k}"] = v


PRESETS = ["Typical athlete (training medians)"] + list(info["examples"])
if "preset" not in st.session_state:
    st.session_state["preset"] = PRESETS[0]
    apply_preset()


def slider(col):
    r = RANGES[col]
    if r["is_int"]:
        return st.slider(LABELS[col], int(r["min"]), int(np.ceil(r["max"])), step=1, key=f"in_{col}")
    step = float(f"{(r['max'] - r['min']) / 200:.2g}")
    return st.slider(LABELS[col], float(r["min"]), float(r["max"]), step=step, key=f"in_{col}")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Start from")
    st.selectbox("Preset", PRESETS, key="preset", on_change=apply_preset, label_visibility="collapsed")
    st.caption("Examples are real sessions from athletes the model never saw during training.")

    st.header("Model")
    m = info["test_metrics"]
    st.write(f"**{info['model_name']}**, C = {info['best_params']['C']}")
    c1, c2 = st.columns(2)
    c1.metric("Test F1", f"{m['f1']:.3f}")
    c2.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
    c1.metric("Recall", f"{m['recall']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    st.caption(
        "Chosen as the best of four models (dummy, logistic regression, random forest, XGBoost) "
        "by test F1. Test set: about 20% of athletes, held out entirely."
    )

# ------------------------------------------------------------------ header + result area
st.title("🏃 Athlete Injury Risk Predictor")
st.write(
    "Predicts whether an athlete's **next** training session will be **high risk** for injury, "
    "using today's wearable, training and environment readings plus their recent history. "
    "Adjust the sliders and the prediction updates instantly."
)
result_area = st.container()

# ------------------------------------------------------------------ inputs
st.subheader("Session inputs")
profile, *group_tabs = st.tabs(["Athlete profile"] + list(GROUPS))
with profile:
    a, b = st.columns(2)
    with a:
        st.selectbox("Sport", CATEGORIES["sport_type"], key="in_sport_type")
        st.radio("Gender", CATEGORIES["gender"], key="in_gender", horizontal=True)
        st.selectbox("Playing surface (dataset code)", CATEGORIES["playing_surface"], key="in_playing_surface",
                     format_func=lambda v: f"Surface {v}")
    with b:
        slider("age")
        slider("bmi")

for tab, (group, cols) in zip(group_tabs, GROUPS.items()):
    with tab:
        if group == "Recent history":
            st.caption("The model also looks at how four key measures have moved over the athlete's last few sessions.")
        a, b = st.columns(2)
        for i, col in enumerate(cols):
            with (a if i < (len(cols) + 1) // 2 else b):
                slider(col)

# ------------------------------------------------------------------ build the model input row
row = {c: st.session_state[f"in_{c}"] for c in info["numeric_base"]}
for col in ROLL_COLS:
    row[f"{col}_roll5"] = st.session_state[f"in_{col}_roll5"]
    row[f"{col}_trend3"] = st.session_state[f"in_{col}"] - st.session_state[f"in_{col}_3ago"]
for col in info["categorical_features"]:
    row[col] = st.session_state[f"in_{col}"]
X = pd.DataFrame([row])[info["features"]]

proba = float(model.predict_proba(X)[0, 1])
is_high = proba >= 0.5

# Per-feature contribution to the log-odds: coefficient x (scaled) input value.
prep, lr = model.named_steps["prep"], model.named_steps["model"]
Xt = prep.transform(X)
Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
contrib = pd.Series(Xt.ravel(), index=prep.get_feature_names_out()) * lr.coef_.ravel()
contrib = contrib[contrib != 0]


def nice(name):
    kind, feat = name.split("__", 1)
    if kind == "cat":
        for c in info["categorical_features"]:
            if feat.startswith(c + "_"):
                return f"{c.replace('_', ' ').capitalize()} = {feat[len(c) + 1:]}"
    return LABELS.get(feat, feat)


contrib.index = [nice(n) for n in contrib.index]

# ------------------------------------------------------------------ show the result
with result_area:
    left, right = st.columns([1, 2])
    with left:
        st.metric("High-risk probability", f"{proba:.0%}")
        st.progress(proba)
        if is_high:
            st.error("**HIGH RISK** next session. Consider reducing load and prioritising recovery.")
        else:
            st.success("**Not high risk** next session.")
        st.caption("Flagged as high risk when the probability is 50% or more (the model's default threshold).")
    with right:
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        chart_df = pd.DataFrame({"factor": top.index, "effect": top.values,
                                 "direction": np.where(top.values > 0, "raises risk", "lowers risk")})
        st.write("**What is driving this prediction** (top 8 factors for this athlete)")
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("effect:Q", title="Effect on risk (log-odds)"),
                y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
                color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                range=["#d1495b", "#2e86ab"])),
                tooltip=["factor", alt.Tooltip("effect:Q", format=".3f")],
            )
            .properties(height=alt.Step(30))
        )
        st.altair_chart(chart, width="stretch")

with st.expander("Model input row (34 features, as sent to the model)"):
    st.dataframe(X.T.rename(columns={0: "value"}).astype(str), width="stretch")
