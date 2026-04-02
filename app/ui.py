import streamlit as st
import pandas as pd
import utils

# ─────────────────────────────
def render_sidebar():
    st.sidebar.title("ARES-X")

    choice = st.sidebar.radio(
        "Navigation",
        ["Home", "Global Risk", "Country Intelligence"]
    )

    st.session_state["view"] = choice


# ─────────────────────────────
def render_home(system):
    st.title("ARES-X Sovereign Risk Intelligence")
    st.write("System running successfully")


# ─────────────────────────────
def render_global_risk(system):
    st.title("Global Risk Ranking")

    df = system["df"]

    results = []
    for _, row in df.iterrows():
        p, _ = utils.predict_risk(row.to_frame().T)
        results.append({
            "country": row["country"],
            "risk": round(p, 3)
        })

    result_df = pd.DataFrame(results).sort_values("risk", ascending=False)

    st.dataframe(result_df)


# ─────────────────────────────
def render_country(system):
    st.title("Country Intelligence")

    df = system["df"]

    countries = sorted(df["country"].dropna().unique())

    selected = st.selectbox("Select Country", countries)

    d = df[df["country"] == selected].sort_values(["year", "month"])

    latest = d.tail(1)

    p, pred = utils.predict_risk(latest)

    st.metric("Risk Score", round(p, 3))
    st.metric("Prediction", "High Risk" if pred else "Low Risk")

    st.dataframe(d)
