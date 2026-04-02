import pandas as pd
import numpy as np
import utils


# ✅ CORE RISK INTELLIGENCE
def compute_risk_intelligence(df):
    results = []

    for _, row in df.iterrows():
        p, pred = utils.predict_risk(row.to_frame().T)

        results.append({
            "country": row["country"],
            "risk_score": p,
            "prediction": pred
        })

    return pd.DataFrame(results)


# ✅ GLOBAL TABLE
def compute_global_risk_table(df):
    return utils.compute_global_risk(df)


# ✅ PORTFOLIO
def compute_portfolio_intelligence(df, weights):
    return utils.portfolio_risk(df, weights)


# ✅ FEATURE IMPORTANCE (FIXED)
def _get_feature_importance(model, feature_names):
    try:
        importance = model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
    except:
        return pd.DataFrame()
