"""
StudyPulse – ML Model Pipeline
Trains models for:
1. Focus Level Prediction (Regression)
2. Peak Hour Classification
3. Burnout Risk Score
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categoricals
    le_subject = LabelEncoder()
    le_tod = LabelEncoder()
    le_dow = LabelEncoder()

    df["subject_enc"]  = le_subject.fit_transform(df["subject"])
    df["tod_enc"]      = le_tod.fit_transform(df["time_of_day"])
    df["dow_enc"]      = le_dow.fit_transform(df["day_of_week"])

    # Cyclical time encoding (hour)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Rolling 7-day averages
    df = df.sort_values("date").reset_index(drop=True)
    df["roll_focus_7d"]  = df["focus_level"].rolling(7, min_periods=1).mean()
    df["roll_sleep_7d"]  = df["sleep_hours"].rolling(7, min_periods=1).mean()
    df["roll_stress_7d"] = df["stress_level"].rolling(7, min_periods=1).mean()

    # Interaction features
    df["sleep_x_focus"]    = df["sleep_hours"] * df["focus_level"]
    df["diff_x_pressure"]  = df["subject_difficulty"] * df["exam_pressure"]
    df["stress_x_sleep"]   = df["stress_level"] * df["sleep_hours"]

    # Peak hour label: top-30% focus sessions → 1
    threshold = df["focus_level"].quantile(0.70)
    df["is_peak_hour"] = (df["focus_level"] >= threshold).astype(int)

    return df, le_subject, le_tod, le_dow


FOCUS_FEATURES = [
    "hour_sin", "hour_cos", "tod_enc", "dow_enc",
    "subject_enc", "subject_difficulty", "sleep_hours",
    "break_duration_min", "exam_pressure", "stress_level",
    "is_weekend", "energy_level", "roll_focus_7d",
    "roll_sleep_7d", "roll_stress_7d",
    "sleep_x_focus", "diff_x_pressure", "stress_x_sleep",
]

BURNOUT_FEATURES = [
    "roll_stress_7d", "roll_sleep_7d", "cumulative_stress",
    "energy_level", "exam_pressure", "subject_difficulty",
    "study_hours", "break_duration_min",
]

PEAK_FEATURES = [
    "hour_sin", "hour_cos", "tod_enc",
    "subject_enc", "subject_difficulty",
    "sleep_hours", "roll_sleep_7d", "roll_focus_7d",
    "stress_level", "energy_level", "exam_pressure",
    "is_weekend",
]


def train_models(df: pd.DataFrame):
    df_feat, le_subject, le_tod, le_dow = engineer_features(df)

    results = {}

    # ── 1. Focus Level Regressor ─────────────────────────────────────────────
    X_focus = df_feat[FOCUS_FEATURES]
    y_focus = df_feat["focus_level"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_focus, y_focus, test_size=0.2, random_state=42)

    focus_model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                        min_samples_leaf=3, random_state=42, n_jobs=-1)
    focus_model.fit(X_tr, y_tr)
    y_pred = focus_model.predict(X_te)
    results["focus_mae"]  = round(mean_absolute_error(y_te, y_pred), 3)
    results["focus_r2"]   = round(r2_score(y_te, y_pred), 3)
    cv_scores = cross_val_score(focus_model, X_focus, y_focus, cv=5, scoring="r2")
    results["focus_cv_r2"] = round(cv_scores.mean(), 3)
    print(f"[Focus Regressor]  MAE={results['focus_mae']}  R²={results['focus_r2']}  CV-R²={results['focus_cv_r2']}")

    # Feature importances
    feat_imp = pd.Series(focus_model.feature_importances_, index=FOCUS_FEATURES).sort_values(ascending=False)
    results["focus_feature_importance"] = feat_imp

    # ── 2. Peak Hour Classifier ──────────────────────────────────────────────
    X_peak = df_feat[PEAK_FEATURES]
    y_peak = df_feat["is_peak_hour"]
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(X_peak, y_peak, test_size=0.2, random_state=42)

    peak_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                             max_depth=4, random_state=42)
    peak_model.fit(Xp_tr, yp_tr)
    yp_pred = peak_model.predict(Xp_te)
    results["peak_accuracy"] = round(accuracy_score(yp_te, yp_pred), 3)
    results["peak_report"]   = classification_report(yp_te, yp_pred)
    print(f"[Peak Classifier]  Accuracy={results['peak_accuracy']}")

    # ── 3. Burnout Risk ──────────────────────────────────────────────────────
    X_burn = df_feat[BURNOUT_FEATURES]
    y_burn = df_feat["burnout_risk"]
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_burn, y_burn, test_size=0.2, random_state=42)

    burnout_model = RandomForestRegressor(n_estimators=150, max_depth=8,
                                           random_state=42, n_jobs=-1)
    burnout_model.fit(Xb_tr, yb_tr)
    yb_pred = burnout_model.predict(Xb_te)
    results["burnout_mae"] = round(mean_absolute_error(yb_te, yb_pred), 3)
    results["burnout_r2"]  = round(r2_score(yb_te, yb_pred), 3)
    print(f"[Burnout Model]    MAE={results['burnout_mae']}  R²={results['burnout_r2']}")

    # ── Package everything ───────────────────────────────────────────────────
    artifacts = {
        "focus_model":    focus_model,
        "peak_model":     peak_model,
        "burnout_model":  burnout_model,
        "le_subject":     le_subject,
        "le_tod":         le_tod,
        "le_dow":         le_dow,
        "focus_features": FOCUS_FEATURES,
        "peak_features":  PEAK_FEATURES,
        "burnout_features": BURNOUT_FEATURES,
        "results":        results,
    }
    return artifacts


def save_models(artifacts: dict, path: str = "models/studypulse_models.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Models saved → {path}")


def load_models(path: str = "models/studypulse_models.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    df = pd.read_csv("data/student_study_log.csv")
    artifacts = train_models(df)
    save_models(artifacts)
    print("\nAll models trained and saved ✓")
