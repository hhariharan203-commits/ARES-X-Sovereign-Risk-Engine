import streamlit as st
import utils, intelligence, ui

st.set_page_config(layout="wide")
ui.apply_theme()

df = utils.load_data()
countries = sorted(df["country"].unique())

with st.sidebar:
    st.title("ARES-X")
    c = st.selectbox("Country", countries)
    st.session_state["country"] = c

st.title("Sovereign Risk Intelligence System")

row = utils.latest(df, c)
score, tier = utils.predict_full(row)

st.metric("Risk Score", round(score,3))
st.metric("Risk Tier", tier)

intel = intelligence.generate_intelligence(row.iloc[0], score)

st.markdown("### Strategic Intelligence Brief")
st.write("**Regime:**", intel["regime"])

st.write("**Drivers:**")
for d in intel["drivers"]:
    st.write("-", d)

st.write("**Action:**", intel["action"])

st.markdown("### Global Ranking")
st.dataframe(utils.global_risk(df)[["country","risk_score"]].head(10))
