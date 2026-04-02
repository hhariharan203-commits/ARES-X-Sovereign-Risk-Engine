import streamlit as st
import utils
import plotly.express as px

df = utils.add_risk(utils.load_data())

latest = df.sort_values("year").groupby("country").tail(1)

st.title("🌍 Global Risk Landscape")

top = latest.sort_values("risk_score", ascending=False).head(10)

fig = px.bar(top, x="country", y="risk_score", color="risk_score")

st.plotly_chart(fig, use_container_width=True)
