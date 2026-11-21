"""Train and save the best model from the final project (Task 3).

Rebuilds the logistic regression from models(updated).ipynb exactly as the notebook does
(same target, features, athlete-grouped split, preprocessing and grid search), checks it
reaches the notebook's test scores, and saves:

    model/injury_risk_model.joblib   - the full fitted pipeline (preprocessing + model)
    model/model_info.json            - feature list, slider ranges, test scores, examples

Logistic regression is saved because it had the best test F1 (0.4987), and F1 is the
metric the notebook uses to pick settings. Run it locally or paste it into a Colab cell:

    python train_and_save_model.py
"""
import json
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUT_DIR = Path(__file__).resolve().parent / "model"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data + target (notebook sections 0-1)
path = kagglehub.dataset_download("anjalibhegam/multimodal-sports-injury-dataset")
csv_files = sorted(Path(path).rglob("*.csv"))
assert len(csv_files) == 1, csv_files
raw = pd.read_csv(csv_files[0])
ORIGINAL_COLS = list(raw.columns)

df = raw.sort_values(["athlete_id", "session_id"]).reset_index(drop=True)
df["next_label"] = df.groupby("athlete_id")["injury_occurred"].shift(-1)
df = df.dropna(subset=["next_label"]).reset_index(drop=True)
y = (df["next_label"] == 2).astype(int)
assert len(df) == 15264 and int(y.sum()) == 2290

# ---------------------------------------------------------------- features (notebook section 2)
EXCLUDE = {"athlete_id", "session_id", "injury_occurred", "next_label"}
base_features = [c for c in ORIGINAL_COLS if c not in EXCLUDE]
categorical_features = ["sport_type", "gender", "playing_surface"]
numeric_base = [c for c in base_features if c not in categorical_features]

ROLL_COLS = ["recovery_score", "sleep_quality", "fatigue_index", "training_load"]
rolling_features, trend_features = [], []
for col in ROLL_COLS:
    by_athlete = df.groupby("athlete_id")[col]
    df[f"{col}_roll5"] = by_athlete.transform(lambda s: s.rolling(5, min_periods=1).mean())
    df[f"{col}_trend3"] = (df[col] - by_athlete.shift(3)).fillna(0)
    rolling_features.append(f"{col}_roll5")
    trend_features.append(f"{col}_trend3")

numeric_features = numeric_base + rolling_features + trend_features
FEATURES = numeric_features + categorical_features

# ---------------------------------------------------------------- split (notebook section 3)
X, groups = df[FEATURES], df["athlete_id"]
outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(outer_cv.split(X, y, groups))
X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# ---------------------------------------------------------------- pipeline + tuning (sections 4-5)
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])
pipe = Pipeline([
    ("prep", prep),
    ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000,
                                 random_state=RANDOM_STATE)),
])
# scikit-learn 1.8+ deprecated `penalty`; with the default l1_ratio=0 it silently fits L2 even
# when penalty="l1". Use l1_ratio there (1 = L1, 0 = L2) so the grid means the same thing as in Colab.
SK_NEW = tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 8)
penalty_grid = {"model__l1_ratio": [1.0, 0.0]} if SK_NEW else {"model__penalty": ["l1", "l2"]}
search = GridSearchCV(
    pipe,
    {"model__C": [0.01, 0.1, 1, 10], **penalty_grid},
    scoring="f1",
    cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
)
search.fit(X_train, y_train, groups=groups_train)
model = search.best_estimator_
print("Best params:", search.best_params_, "| CV F1:", round(search.best_score_, 4))

# ---------------------------------------------------------------- test check (section 6)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
test_metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
}
print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
if abs(test_metrics["f1"] - 0.4987) > 0.002:
    print("WARNING: test F1 differs from the notebook's 0.4987 (different library versions?)")

# ---------------------------------------------------------------- save
joblib.dump(model, OUT_DIR / "injury_risk_model.joblib", compress=3)

# Slider ranges come from the 1st-99th percentile of the training data, so the sliders cover
# realistic values without letting outliers stretch them.
ranges = {}
for col in numeric_base + rolling_features:
    s = X_train[col].dropna()
    ranges[col] = {
        "min": float(s.quantile(0.01)), "max": float(s.quantile(0.99)),
        "median": float(s.median()), "is_int": bool(pd.api.types.is_integer_dtype(s)),
    }
for col in ROLL_COLS:  # the "3 sessions ago" sliders share the current value's range
    ranges[f"{col}_3ago"] = dict(ranges[col])

# Two real held-out sessions, used as ready-made examples in the app.
test_df = df.iloc[test_idx].assign(proba=proba)
high = test_df[y_test.to_numpy() == 1].sort_values("proba", ascending=False).iloc[len(test_df) // 200]
low = test_df[y_test.to_numpy() == 0].sort_values("proba").iloc[len(test_df) // 200]


def as_example(row):
    out = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in FEATURES}
    for col in ROLL_COLS:  # recover "value 3 sessions ago" from the trend
        out[f"{col}_3ago"] = float(row[col] - row[f"{col}_trend3"])
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in out.items()}


info = {
    "model_name": "Logistic regression (L1, class_weight='balanced')",
    "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    "cv_f1": float(search.best_score_),
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    "sklearn_version": sklearn.__version__,
    "numeric_base": numeric_base,
    "roll_cols": ROLL_COLS,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "features": FEATURES,
    "categories": {c: sorted(X_train[c].dropna().unique().tolist()) for c in categorical_features},
    "ranges": ranges,
    "examples": {"Example: high-risk session": as_example(high), "Example: low-risk session": as_example(low)},
}
(OUT_DIR / "model_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
print("Saved", OUT_DIR / "injury_risk_model.joblib", "and", OUT_DIR / "model_info.json")
