from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    # Load data
    path = Path(__file__).resolve().parents[1] / "data" / "master_dataset.csv"
    df = pd.read_csv(path)

    print("✅ Data loaded:", df.shape)

    # Convert month
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["country", "month"])

    # -------------------------------
    # 🎯 TARGET VARIABLE
    # -------------------------------
    # Predict future GDP growth (next 3 months)
    df["target"] = df.groupby("country")["gdp_growth"].shift(-3)

    # -------------------------------
    # ⚙️ FEATURE ENGINEERING (CRITICAL)
    # -------------------------------
    features = [
        "gdp_growth",
        "inflation",
        "unemployment",
        "interest_rate",
        "exports",
        "imports",
        "vix",
        "sentiment_mean"
    ]

    # Lag features
    for lag in [1, 2, 3]:
        for col in features:
            df[f"{col}_lag{lag}"] = df.groupby("country")[col].shift(lag)

    # Drop missing
    df = df.dropna()

    print("✅ After feature engineering:", df.shape)

    # -------------------------------
    # 🔥 NORMALIZATION (ADD HERE)
    # -------------------------------
    features_to_scale = [col for col in df.columns if col not in ["country", "month", "target"]]

    df[features_to_scale] = df.groupby("country")[features_to_scale].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # -------------------------------
    # 🎯 MODEL INPUT
    # -------------------------------
    X = df.drop(columns=["country", "month", "target"])
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # -------------------------------
    # 🚀 MODEL (ADVANCED)
    # -------------------------------
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # -------------------------------
    # 📊 EVALUATION
    # -------------------------------
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("\n🔥 MODEL PERFORMANCE")
    print("RMSE:", rmse)

    # -------------------------------
    # ⭐ FEATURE IMPORTANCE
    # -------------------------------
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n📊 TOP FEATURES")
    print(importance.head(10))

    # Save importance
    out_path = Path(__file__).resolve().parents[1] / "data" / "feature_importance.csv"
    importance.to_csv(out_path, index=False)

    # -------------------------------
    # 💾 SAVE MODEL + METRICS
    # -------------------------------
    import joblib
    import json

    # Create folders
    model_dir = Path(__file__).resolve().parents[1] / "models"
    output_dir = Path(__file__).resolve().parents[1] / "outputs"

    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Save model
    joblib.dump(model, model_dir / "model.pkl")

    # Save feature columns
    with open(model_dir / "features_cols.json", "w") as f:
        json.dump(list(X.columns), f)

    # Save metrics
    metrics = {
        "rmse": float(rmse)
    }

    with open(output_dir / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n💾 Model + metrics saved")

    with open(output_dir / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n💾 Model + metrics saved")
    print("\n✅ Training completed & importance saved")


if __name__ == "__main__":
    main()