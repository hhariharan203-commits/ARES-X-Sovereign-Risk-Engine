ARES-X: Sovereign Crisis Intelligence Engine

ARES-X is an end-to-end AI system that predicts sovereign financial crisis risk using macroeconomic data, machine learning, and explainable AI.

It converts raw economic indicators into actionable intelligence, enabling analysts and decision-makers to monitor risk, simulate scenarios, and understand key economic drivers.

---

🚀 Features

- Crisis Risk Prediction (ML Model)
- Global Risk Heatmap (Interactive Map)
- Country-Level Intelligence Dashboard
- Forecasting Engine with Trend Simulation
- Explainable AI (SHAP-based Drivers)
- Executive Insights + Suggested Actions
- End-to-End Data Pipeline (Automated)

---

📁 Project Structure

ARES-X/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   └── pages/
│       ├── country_view.py
│       ├── executive_dashboard.py
│       ├── explainability.py
│       ├── forecast.py
│       ├── global_view.py
│       └── model_performance.py
│
├── data/
│   ├── world_bank_data.csv
│   ├── google_trends.csv
│   ├── news_sentiment.csv
│   ├── fred_yield.csv
│   ├── master_dataset.csv
│   ├── clean_master_dataset.csv
│   └── feature_importance.csv
│
├── models/
│   ├── model.pkl
│   ├── feature_cols.json
│   ├── feature_cols.json (backup)
│   └── risk_thresholds.json
│
├── outputs/
│   ├── model_metrics.json
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_importance.csv
│
├── src/
│   ├── fetch_world_bank.py
│   ├── fetch_google_trends.py
│   ├── fetch_gdelt_news.py
│   ├── fetch_fred_yield.py
│   ├── merge_datasets.py
│   ├── clean_master_dataset.py
│   ├── train_model.py
│   ├── shap_explain.py
│   └── validate_master.py
│
├── README.md
└── requirements.txt

---

🧠 Model Performance

Metric| Value
Accuracy| 0.723
Precision| 0.475
Recall| 0.705
F1 Score| 0.568
ROC-AUC| 0.780

Insight:

- High recall → captures most crisis events
- Strong ROC-AUC → reliable risk ranking
- Suitable for early warning systems

---

📊 Data Sources

ARES-X integrates multiple macroeconomic and alternative datasets:

- World Bank (GDP, inflation, trade)
- Google Trends (market sentiment proxy)
- GDELT News (global sentiment signals)
- FRED Yield Data (interest rate proxy)

---

⚙️ Data Pipeline (src/)

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

📊 Dashboard (Streamlit App)

Run:

streamlit run app/app.py

---

📈 Dashboard Modules

1. Executive Dashboard

- Global risk overview
- Trend analysis
- High-risk countries

2. Country View

- Country-specific indicators
- Crisis probability tracking

3. Global View

- Risk heatmap across countries
- Comparative analysis

4. Forecast

- Future risk simulation
- Scenario-based predictions
- Executive insights + actions

5. Explainability

- SHAP feature importance
- Top drivers of risk

6. Model Performance

- Accuracy, Precision, Recall, ROC-AUC

---

🧠 Explainable AI Logic

- SHAP values processed safely
- Top drivers extracted using absolute contribution
- Human-readable insights generated:
  - ↑ increases risk
  - ↓ decreases risk

---

🎯 Business Applications

- Sovereign risk monitoring
- Investment decision support
- Hedge fund macro strategies
- Policy advisory systems
- Early warning risk detection

---

🔗 Live Demo
https://ares-x-sovereign-risk-engine-cakd8wraujxlh4dwws3gp7.streamlit.app/

---

🚀 Key Strengths

- Full ML pipeline (data → model → dashboard)
- Multi-source data integration
- Explainable AI (SHAP)
- Forecast + simulation capability
- Modular, production-style architecture

---

🔮 Future Improvements

- Add real-time API ingestion
- Expand to 100+ countries
- Add deep learning models (LSTM)
- Integrate geopolitical risk signals
- Deploy as SaaS / API service

---

▶️ Installation

pip install -r requirements.txt

---

📸 Screenshots

Add images of:

- Global Heatmap
- Forecast Chart
- SHAP Insights
- Executive Dashboard

---

👤 Author

Hariharan
MBA - Finance
Data Analyst 

---

⭐ Final Note

ARES-X is not just a model — it is a decision intelligence system that bridges machine learning and macroeconomic strategy.
