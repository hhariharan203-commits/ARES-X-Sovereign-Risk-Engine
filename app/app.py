import streamlit as st
import utils, ui

st.set_page_config(layout="wide")
ui.apply_theme()

df = utils.load_data()
countries = sorted(df["country"].unique())

with st.sidebar:
    st.title("ARES-X")
    c = st.selectbox("Country", countries)
    st.session_state["country"] = c

st.title("🌍 Global Risk Intelligence")

risk = utils.global_risk(df)
st.dataframe(risk[["country","risk_score"]].head(10))

row = utils.latest(df, c)
score = utils.predict(row)[0]

ui.kpi("Selected Country Risk", round(score,3))
