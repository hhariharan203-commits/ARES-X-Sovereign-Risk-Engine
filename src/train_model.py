import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/master_dataset.csv")

print(f"✅ Data loaded: {df.shape}")

# =========================
# CREATE FUTURE TARGET (t+1)
# =========================
df = df.sort_values(["country", "month"])

df["target"] = df.groupby("country")["gdp_growth"].shift(-1)

# Remove last rows where future not available
df = df.dropna().reset_index(drop=True)

print(f"✅ After target creation: {df.shape}")

# =========================
# REMOVE LEAKAGE FEATURES
# =========================
exclude_cols = [
    "country",
    "month",
    "gdp_growth",   # current GDP (avoid leakage)
    "target",
    "gdp_growth_lag1",
    "gdp_growth_lag2",
    "gdp_growth_lag3"
]

features = [col for col in df.columns if col not in exclude_cols]

X = df[features]
y = df["target"]

print(f"✅ Features used: {len(features)}")

# =========================
# TIME-BASED SPLIT (CRITICAL)
# =========================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"✅ Train shape: {X_train.shape}")
print(f"✅ Test shape: {X_test.shape}")

# =========================
# MODEL
# =========================
model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds, squared=False)

print("\n🔥 MODEL PERFORMANCE (FORECAST MODEL)")
print("RMSE:", round(rmse, 4))

# =========================
# FEATURE IMPORTANCE
# =========================
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n📊 TOP FEATURES (REAL SIGNALS)")
print(importance.head(10))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "../models/model.pkl")

print("\n💾 Model saved successfully")