"""
explainability.py — Extract and present real feature importances from XGBoost model.
"""

import pandas as pd
import numpy as np
from data_api import load_model, load_feature_cols


def get_feature_importance(top_n: int = 20) -> pd.DataFrame:
    """
    Return a DataFrame of feature importances from the trained XGBoost model.
    Columns: Feature, Importance, Category
    """
    model        = load_model()
    feature_cols = load_feature_cols()

    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Categorize features
    def _categorize(feat):
        feat = feat.lower()
        if "gdp"         in feat: return "GDP"
        if "inflation"   in feat: return "Inflation"
        if "unemployment" in feat: return "Labor"
        if "interest"    in feat: return "Interest Rate"
        if "export"      in feat: return "Trade"
        if "import"      in feat: return "Trade"
        if "vix"         in feat: return "Risk / Sentiment"
        if "sentiment"   in feat: return "Risk / Sentiment"
        if "trade"       in feat: return "Trade"
        return "Other"

    df["Category"] = df["Feature"].apply(_categorize)

    # Normalize to 0–100
    total = df["Importance"].sum()
    df["Importance %"] = (df["Importance"] / total * 100).round(2) if total > 0 else 0.0

    return df.head(top_n)


def get_category_importance() -> pd.DataFrame:
    """Aggregate importances by category."""
    df  = get_feature_importance(top_n=100)
    agg = (
        df.groupby("Category")["Importance %"]
        .sum()
        .reset_index()
        .rename(columns={"Importance %": "Total Importance %"})
        .sort_values("Total Importance %", ascending=False)
        .reset_index(drop=True)
    )
    return agg