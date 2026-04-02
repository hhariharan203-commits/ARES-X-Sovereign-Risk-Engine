import streamlit as st
import utils, intelligence

df = utils.load_data()

country = st.selectbox("Country", df["country"].unique())

shock = st.slider("Inflation Shock", -5, 10, 0)

row = utils.latest(df, country).copy()

base = utils.predict(row)

row["inflation"] += shock
row["inflation_lag1"] = row["inflation"]

new = utils.predict(row)

st.title("Scenario Intelligence")

st.metric("Base", round(base,3))
st.metric("Scenario", round(new,3))
st.metric("Impact", round(new-base,3))

brief = intelligence.generate_brief(row.iloc[0], new)

st.write("Decision:", brief["decision"])
