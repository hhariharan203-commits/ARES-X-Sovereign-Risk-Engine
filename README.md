# ARES-X: AI-Powered Sovereign Risk & Macro Investment Decision Engine

🔗 Live Demo  
https://ares-x-sovereign-risk-engine-6jv4ugrjxdigb7tqhntzcg.streamlit.app/  

💡 Fully deployed — runs directly in browser (no setup required)

---

## 📌 Overview

ARES-X is an AI-driven macro intelligence and sovereign risk decision system designed to replicate how institutional investors analyze global economies.

It transforms macroeconomic data into:

- Forward-looking GDP forecasts  
- Sovereign risk scores (0–100)  
- Macro regime classification  
- Investment decisions (STRONG BUY → DEFENSIVE)  
- Portfolio allocation strategies  
- Institutional-grade macro reports (PDF)

The system integrates **machine learning, macroeconomics, and decision logic** into a unified framework for real-world financial analysis.

---

## 🎯 What Makes ARES-X Different

Unlike typical ML projects, ARES-X is:

- A **decision system**, not just a prediction model  
- Designed with **investment logic (buy / hold / defensive)**  
- Built as a **modular macro intelligence pipeline**  
- Focused on **real-world financial use cases**  

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
- Outputs forward-looking macro signal  

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
| STRONG BUY   | High growth, low risk |
| BUY          | Stable macro environment |
| HOLD         | Neutral conditions |
| DEFENSIVE    | High-risk environment |

---

### 5. Portfolio Allocation Engine
- Suggests asset allocation based on macro regime  
- Aligns strategy with economic conditions  

---

### 6. Report Engine
Generates structured macro reports including:
- Executive summary  
- Forecast outlook  
- Risk breakdown  
- Investment recommendation  
- Portfolio allocation  

---

## 📊 Key Outputs

- Country-level risk scores  
- Global risk rankings  
- GDP forecast trends  
- Investment decision classification  
- Portfolio allocation strategy  
- Downloadable investor-grade reports  

---

## 📈 Model Performance

| Metric | Value |
|------|------|
| R² Score | ~0.90+ |
| RMSE | Optimized |
| MAE | Low |

> Model is optimized for **directional macro forecasting**, not exact prediction.

---

## 🌍 Dashboard Modules

### 1. Global Risk Terminal
- Cross-country comparison  
- Risk rankings  
- Macro snapshot  

### 2. Country Intelligence
- Forecast + risk analysis  
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

## 📑 PDF Reports (Investor-Grade)

ARES-X generates reports similar to institutional research notes:

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

- Decision-focused, not prediction-only  
- Modular architecture (forecast → risk → decision)  
- Explainable macro signals  
- Scalable and extensible pipeline  

---

## 🎯 Use Cases

- Sovereign risk analysis  
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

```bash
pip install -r requirements.txt
streamlit run app/main.py
🔒 Limitations
Dataset size can be expanded
VIX represents global volatility (not country-specific)
Designed for decision support, not execution trading
👤 Author
Hariharan
MBA – Finance & Business Analytics
Data Analyst | AI & Macro Intelligence
⭐ Final Note
ARES-X is not just a machine learning project.
It is a macro decision intelligence system that combines forecasting, risk modeling, and investment logic — similar to workflows used in institutional finance and global macro funds.
