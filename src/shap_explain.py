from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train_model import add_features, load_data


MODEL_PATH = (Path(__file__).resolve().parent / "../models/model.pkl").resolve()
DATA_PATH = (Path(__file__).resolve().parent / "../data/clean_master_dataset.csv").resolve()
OUTPUT_DIR = (Path(__file__).resolve().parent / "../outputs").resolve()


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_data() -> pd.DataFrame:
    """Load data and engineer features exactly as in training."""
    df = load_data()
    df_feat = add_features(df)
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in {"country", "month", "crisis_risk", "crisis_risk_future"}
    ]
    X = df_feat[feature_cols]
    return X, feature_cols


def main() -> None:
    ensure_output_dir()

    # Load model and data
    model = joblib.load(MODEL_PATH)
    X, feature_cols = get_feature_data()

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle classification SHAP output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.array(shap_values)

    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    summary_path = OUTPUT_DIR / "shap_summary.png"
    plt.savefig(summary_path, bbox_inches="tight", dpi=200)
    plt.close()

    # Bar plot (global importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    bar_path = OUTPUT_DIR / "shap_bar.png"
    plt.savefig(bar_path, bbox_inches="tight", dpi=200)
    plt.close()

    # Feature importance CSV (mean |SHAP|)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_abs = np.array(mean_abs).flatten()
    feature_cols = list(feature_cols)
    if len(feature_cols) != len(mean_abs):
        min_len = min(len(feature_cols), len(mean_abs))
        feature_cols = feature_cols[:min_len]
        mean_abs = mean_abs[:min_len]
    imp_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
    imp_path = OUTPUT_DIR / "shap_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    print(f"Saved SHAP summary to {summary_path}")
    print(f"Saved SHAP bar to {bar_path}")
    print(f"Saved SHAP importance to {imp_path}")


if __name__ == "__main__":
    main()
