ARES-X: Sovereign Crisis Intelligence Engine

🔗 Live Demo:
https://ares-x-sovereign-risk-engine-cakd8wraujxlh4dwws3gp7.streamlit.app/

💡 No setup required — runs directly in browser

---

📌 Overview

ARES-X is an end-to-end AI system that predicts sovereign financial crisis risk using macroeconomic data, machine learning, and explainable AI.

It transforms raw economic indicators into actionable intelligence, enabling analysts, investors, and policymakers to monitor risk, simulate scenarios, and understand key economic drivers.

---

🚀 Key Features

- Crisis Risk Prediction using Machine Learning (XGBoost)
- Global Risk Heatmap (Interactive Choropleth)
- Country-Level Intelligence Dashboard
- Forecasting Engine with Scenario Simulation
- Explainable AI using SHAP
- Executive Insights with Suggested Actions
- End-to-End Data Pipeline (Automated)

---

🧠 Model Performance

Metric| Value
Accuracy| 0.723
Precision| 0.475
Recall| 0.705
F1 Score| 0.568
ROC-AUC| 0.780

Interpretation

- High recall ensures crisis signals are captured early
- Strong ROC-AUC indicates reliable risk ranking
- Designed for early warning systems, not exact prediction

---

📊 Data Sources

ARES-X integrates multiple datasets:

- World Bank → GDP, inflation, trade
- Google Trends → Market sentiment proxy
- GDELT News → Global sentiment signals
- FRED → Interest rate / yield data

---

⚙️ Data Pipeline

Run scripts in sequence:

python src/fetch_world_bank.py
python src/fetch_google_trends.py
python src/fetch_gdelt_news.py
python src/fetch_fred_yield.py

python src/merge_datasets.py
python src/clean_master_dataset.py
python src/validate_master.py

---

🤖 Model Training

python src/train_model.py

Outputs:

- "models/model.pkl"
- "outputs/model_metrics.json"

---

🔍 Explainability (SHAP)

python src/shap_explain.py

Outputs:

- SHAP importance CSV
- SHAP summary plot
- SHAP bar chart

---

📊 Dashboard (Streamlit)

streamlit run app/app.py

---

📈 Dashboard Modules

1. Executive Dashboard

- Global risk overview
- Trend analysis
- High-risk countries

2. Country Intelligence

- Country-level indicators
- Crisis probability tracking
- Scenario simulation

3. Global View

- Risk heatmap
- Cross-country comparison

4. Forecast

- Future risk simulation
- Scenario-based predictions
- Executive insights + actions

5. Explainability

- SHAP feature importance
- Key risk drivers

6. Model Performance

- Accuracy, Precision, Recall, ROC-AUC

---

🧠 Explainable AI Logic

- SHAP values processed safely
- Top drivers selected using absolute contribution
- Human-readable insights generated:

↑ increases risk  
↓ decreases risk  

---

🎯 Business Applications

- Sovereign risk monitoring
- Investment decision support
- Hedge fund macro strategies
- Policy advisory systems
- Early warning risk detection

---

📁 Project Structure

ARES-X/
├── app/
├── data/
├── models/
├── outputs/
├── src/
├── README.md
└── requirements.txt

---

📸 Screenshots

Add:

- Global Heatmap
- Forecast Dashboard
- SHAP Insights
- Executive Dashboard

---

🔧 Installation

pip install -r requirements.txt

---

🔒 Limitations

- Covers 24 countries (expandable)
- Macro data includes time-lag effects
- Designed for directional risk, not exact prediction

---

👤 Author

Hariharan
MBA – Finance & Business Analytics
Data Analyst | AI & Financial Analytics

---

⭐ Final Note

ARES-X is not just a model —
it is a decision intelligence system that combines machine learning, macroeconomics, and explainability to support real-world strategic decisions.
