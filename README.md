# ARES-X: AI-Powered Sovereign Risk & Macro Investment Decision Engine

🔗 Live Demo  
https://ares-x-sovereign-risk-engine-6jv4ugrjxdigb7tqhntzcg.streamlit.app/  

💡 Fully deployed — runs directly in browser (no setup required)

---

## 📌 Overview

ARES-X is a macro intelligence and sovereign risk decision system designed to simulate how institutional investors analyze global economies.

It transforms macroeconomic data into:

- Forward-looking GDP forecasts  
- Sovereign risk scores (0–100)  
- Macro regime classification  
- Investment decisions (STRONG BUY → DEFENSIVE)  
- Portfolio allocation strategies  
- Structured macro intelligence reports (PDF)  

The system integrates **machine learning, macroeconomic signals, and decision logic** into a unified analytical framework.

---

## 🎯 What Makes ARES-X Different

Unlike typical ML projects, ARES-X is:

- A **decision-focused system**, not just a prediction model  
- Designed with **investment logic (buy / hold / defensive)**  
- Built as a **modular macro intelligence pipeline**  
- Structured to **simulate institutional workflows**, not replicate them  

---

## 🧠 System Architecture

ARES-X follows a structured pipeline:

### 1. Data Layer
Macroeconomic indicators:
- GDP growth  
- Inflation  
- Unemployment  
- Trade balance (exports / imports)  
- Global volatility (VIX)  

---

### 2. Forecasting Engine
- XGBoost model predicts GDP growth trends  
- Produces forward-looking macro signals  

---

### 3. Risk Engine
- Composite scoring system (0–100)  
- Normalizes macro indicators into risk signals  
- Weighted model combining growth, inflation, labor, volatility, and trade  

---

### 4. Decision Engine
Transforms macro + risk signals into:

| Decision      | Interpretation |
|--------------|--------------|
| STRONG BUY   | Strong growth, low risk |
| BUY          | Stable macro environment |
| HOLD         | Neutral conditions |
| DEFENSIVE    | Elevated macro risk |

---

### 5. Portfolio Allocation Engine
- Suggests asset allocation based on macro regime  
- Aligns investment strategy with economic conditions  

---

### 6. Report Engine
Generates structured macro reports including:
- Executive summary  
- Forecast outlook  
- Risk assessment  
- Investment recommendation  
- Portfolio allocation  

---

## 📊 Key Outputs

- Country-level risk scores  
- Global risk rankings  
- GDP forecast trends  
- Investment decision classification  
- Portfolio allocation strategy  
- Downloadable macro intelligence reports  

---

## 📈 Model Performance

| Metric | Value |
|------|------|
| R² Score | ~0.90+ |
| RMSE | Optimized |
| MAE | Low |

> Model is optimized for **directional macro forecasting**, not precise point prediction.

---

## 🌍 Dashboard Modules

### 1. Global Risk Terminal
- Cross-country comparison  
- Risk rankings  
- Macro overview  

### 2. Country Intelligence
- Forecast + risk breakdown  
- Decision output  
- Supporting macro drivers  

### 3. Forecast Engine
- GDP predictions  
- Trend direction  

### 4. Decision Terminal
- Investment classification  
- Score-based ranking  

### 5. Report Generator
- Structured macro reports  
- PDF export  

---

## 📑 PDF Reports

ARES-X generates structured reports including:

- Executive summary  
- GDP forecast  
- Risk assessment  
- Investment decision  
- Portfolio allocation  
- Supporting factors  

---

## ⚙️ Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- ReportLab  

---

## 🧠 Design Principles

- Decision-focused (not prediction-only)  
- Modular pipeline (forecast → risk → decision)  
- Interpretable macro signals  
- Scalable and extensible architecture  

---

## 🎯 Use Cases

- Sovereign risk monitoring  
- Investment decision support  
- Macro research automation  
- Portfolio strategy development  
- Financial intelligence systems  

---

## 📁 Project Structure

ARES-X/
├── app/            # Streamlit application  
├── data/           # Processed datasets  
├── models/         # Trained ML models  
├── outputs/        # Metrics and reports  
├── src/            # Data + model pipeline  
├── README.md  
└── requirements.txt  

---

## 🔧 Installation (Local)

pip install -r requirements.txt
streamlit run app/main.py

🔒 Limitations
Dataset size can be expanded
VIX represents global volatility (not country-specific)
Designed as a decision-support prototype, not a production trading system

👤 Author
Hariharan B
MBA – Finance
Data Analyst | AI & Macro Intelligence

⭐ Final Note
ARES-X is not just a machine learning project.
It is a macro decision intelligence system that combines forecasting, risk modeling, and investment logic — designed to simulate real-world analytical workflows used in macro research and investment environments.
