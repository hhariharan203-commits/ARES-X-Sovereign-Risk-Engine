import streamlit as st
import utils

df = utils.load_data()
c = st.session_state.get("country","USA")

row = utils.latest(df,c).copy()

infl = st.slider("Inflation Shock", -5,10,2)
gdp  = st.slider("GDP Shock", -10,5,-2)

row["inflation"] += infl
row["gdp_growth"] += gdp

score = utils.predict(row)[0]

st.metric("Scenario Risk", round(score,3))
