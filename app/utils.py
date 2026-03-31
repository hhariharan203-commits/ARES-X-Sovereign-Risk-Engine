from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import shap
import joblib
import pandas as pd
import streamlit as st

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean_master_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "feature_cols.json"
THRESHOLDS_PATH = BASE_DIR / "models" / "risk_thresholds.json"


# ================= FEATURE NAME CLEANING =================
def humanize_feature(feat: str) -> str:
    feat = feat.replace("_pct", "%")
    feat = feat.replace("_usd", " USD")
    feat = feat.replace("_lag1", " (Short-term)")
    feat = feat.replace("_lag3", " (Medium-term)")
    feat = feat.replace("_lag6", " (Long-term)")
    feat = feat.replace("_z", " (Normalized)")
    feat = feat.replace("_", " ").title()
    return feat


# ================= SAFE HELPERS =================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def safe_get(df: pd.DataFrame, col: str, default=None):
    return df[col] if col in df.columns else default


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> bool:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing columns: {', '.join(missing)}")
        return False
    return True


# ================= LOADERS =================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_csv(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        df = _standardize_columns(df)
        return df
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = load_csv(path, parse_dates=["month"])
    if "country" in df.columns and "month" in df.columns:
        df = df.sort_values(["country", "month"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_cols() -> list[str]:
    with open(FEATURE_COLS_PATH) as f:
        cols = json.load(f)
    return [c.strip().lower() for c in cols]


@lru_cache(maxsize=1)
def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            return json.load(f)
    return {"low": 0.33, "high": 0.66}


@lru_cache(maxsize=1)
def get_dynamic_thresholds() -> dict:
    """
    Compute dynamic percentile-based thresholds using latest month snapshot.
    HIGH: top 20%, LOW: bottom 20%, MEDIUM: middle 60%.
    """
    try:
        df = load_data()
        latest_month = df["month"].max()
        latest_df = df[df["month"] == latest_month]
        if latest_df.empty:
            raise ValueError("No latest month data")

        model = load_model()
        aligned = align_features(latest_df)
        probs = model.predict_proba(aligned)[:, 1]

        p20 = float(np.percentile(probs, 20))
        p80 = float(np.percentile(probs, 80))

        return {"low": p20, "high": p80}
    except Exception:
        # Fallback to static thresholds if anything goes wrong
        fallback = load_thresholds()
        # widen to avoid edge-case mislabeling
        return {"low": fallback.get("low", 0.2), "high": fallback.get("high", 0.8)}


@lru_cache(maxsize=1)
def load_explainer():
    return shap.TreeExplainer(load_model())


# ================= CORE =================
def add_probabilities(df: pd.DataFrame, model=None) -> pd.DataFrame:
    df = df.copy()
    model = model or load_model()
    if df.empty:
        return df
    aligned = align_features(df).fillna(0)
    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)
    try:
        df["crisis_prob"] = model.predict_proba(aligned.astype(float))[:, 1]
    except Exception:
        df["crisis_prob"] = 0.0
    df["risk_level"] = assign_risk_levels(df["crisis_prob"])
    return df


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = load_feature_cols()
    df = _standardize_columns(df.copy())
    df_aligned = df.reindex(columns=cols)
    df_aligned = df_aligned.apply(pd.to_numeric, errors="coerce")
    return df_aligned[cols]


def assign_risk_levels(prob_series: pd.Series) -> pd.Series:
    if prob_series is None or prob_series.empty:
        return pd.Series(dtype=object)
    q_low = prob_series.quantile(0.33)
    q_high = prob_series.quantile(0.66)
    def _label(p):
        if pd.isna(p):
            return "UNKNOWN"
        if p < q_low:
            return "LOW"
        if p < q_high:
            return "MEDIUM"
        return "HIGH"
    return prob_series.apply(_label)


def risk_label(prob: float) -> str:
    try:
        df = load_data()
        if "crisis_prob" in df.columns:
            labels = assign_risk_levels(df["crisis_prob"])
            if not labels.empty:
                q_low = df["crisis_prob"].quantile(0.33)
                q_high = df["crisis_prob"].quantile(0.66)
                if prob < q_low:
                    return "LOW"
                if prob < q_high:
                    return "MEDIUM"
                return "HIGH"
    except Exception:
        pass
    if prob < 0.33:
        return "LOW"
    if prob < 0.66:
        return "MEDIUM"
    return "HIGH"


# ================= PREDICTION =================
def predict_with_explanations(df_row: pd.DataFrame):
    model = load_model()
    explainer = load_explainer()

    full_df = load_data()
    country = df_row["country"].iloc[0]

    country_hist = full_df[full_df["country"] == country]
    global_hist = full_df.copy()

    # Align all
    aligned = align_features(df_row).iloc[[0]]
    aligned_country = align_features(country_hist)
    aligned_global = align_features(global_hist)

    numeric_cols = aligned.columns

    # Track missing ratio before imputation for warning
    missing_ratio = aligned.isnull().mean(axis=1).iloc[0]

    # ✅ STEP 1: forward-fill from country history
    # Gets most recent actual recorded value per feature for this country
    # This is what makes ARG/GBR/TUR unique instead of all getting global mean
    if not aligned_country.empty:
        country_last_known = aligned_country.ffill().tail(1)
        if not country_last_known.empty:
            aligned[numeric_cols] = aligned[numeric_cols].fillna(
                country_last_known[numeric_cols].iloc[0]
            )

    # ✅ STEP 2: country mean fallback
    if not aligned_country.empty:
        aligned[numeric_cols] = aligned[numeric_cols].fillna(
            aligned_country[numeric_cols].mean()
        )

    # ✅ STEP 3: global mean fallback
    aligned[numeric_cols] = aligned[numeric_cols].fillna(
        aligned_global[numeric_cols].mean()
    )

    # ✅ STEP 4: final zero fallback
    aligned = aligned.fillna(0)

    # Model alignment
    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    aligned = aligned.astype(float)

    # Validation
    if not np.isfinite(aligned.values).all():
        return 0.25, ["Data issue — fallback prediction"]

    # Prediction
    try:
        prob = float(model.predict_proba(aligned)[0, 1])
    except Exception:
        return 0.25, ["Model failed — fallback prediction"]

    # Low data flag
    low_data = missing_ratio > 0.5

    # SHAP explanations
    try:
        shap_values = explainer.shap_values(aligned)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.array(shap_values)
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
        shap_values = shap_values.flatten()

        feature_cols = list(load_feature_cols())

        top_idx = np.argsort(np.abs(shap_values))[::-1][:3]

        insights = []
        for i in top_idx:
            idx = int(i)
            if idx >= len(feature_cols) or idx >= len(shap_values):
                continue
            feat = feature_cols[idx]
            val = float(shap_values[idx])
            direction = "↑" if val > 0 else "↓"
            insights.append(f"{humanize_feature(feat)} {direction}")

        if low_data:
            insights.append("⚠️ Sparse data — imputed from history/global mean")

        if not insights:
            insights.append("Explainability unavailable")

    except Exception:
        insights = ["Explainability unavailable"]

    if aligned.std(axis=1).values[0] < 1e-5:
        insights.append("⚠️ Low data variability — low confidence")

    return prob, insights


# ================= EXECUTIVE INSIGHTS =================
def generate_executive_insights(aligned: pd.DataFrame, shap_values) -> dict:
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_array = np.array(shap_values)
    if shap_array.ndim > 1:
        shap_array = shap_array[0]
    shap_array = shap_array.flatten()

    feature_cols = list(aligned.columns)

    order = np.argsort(np.abs(shap_array))[::-1]

    group_map = {
        "gdp":          "Economic Growth",
        "interest":     "Interest Rates",
        "import":       "Import Dependency",
        "export":       "Export Strength",
        "unemployment": "Labor Market",
    }

    seen_groups = set()
    drivers = []
    actions = []

    for idx in order:
        idx = int(idx)
        if idx >= len(feature_cols) or idx >= len(shap_array):
            continue
        feat = feature_cols[idx]
        delta = float(shap_array[idx])

        group = None
        for k in group_map:
            if k in feat:
                group = group_map[k]
                break

        if group and group not in seen_groups:
            direction = "increasing" if delta > 0 else "reducing"
            drivers.append(f"- {group} is {direction} financial risk")
            seen_groups.add(group)

            if group == "Interest Rates":
                actions.append("Review central bank rate policy")
            elif group == "Export Strength":
                actions.append("Boost export competitiveness")
            elif group == "Import Dependency":
                actions.append("Reduce import reliance")
            elif group == "Economic Growth":
                actions.append("Deploy fiscal stimulus")
            elif group == "Labor Market":
                actions.append("Improve employment programs")

        if len(drivers) == 3:
            break

    summary = drivers[0].replace("- ", "").capitalize() if drivers else "Risk stable"

    return {
        "summary": summary,
        "drivers": drivers if drivers else ["No strong risk drivers identified"],
        "actions": list(set(actions)) if actions else ["Maintain current macroeconomic stability"],
    }


# ================= UI =================
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="black", font_size=14, font_color="white"),
        xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), gridcolor="gray"),
        yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), gridcolor="gray"),
    )

    for trace in fig.data:
        if trace.type != "choropleth":
            if hasattr(trace, "opacity"):
                trace.opacity = 0.9
            if hasattr(trace, "line"):
                trace.line.width = 3

    return fig


# ================= SHAP =================
@lru_cache(maxsize=1)
def load_shap():
    shap_path = BASE_DIR / "outputs" / "shap_importance.csv"
    if shap_path.exists():
        return pd.read_csv(shap_path)
    return pd.DataFrame()


def safe_metric(label: str, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        st.metric(label, "N/A")
    else:
        st.metric(label, value)
