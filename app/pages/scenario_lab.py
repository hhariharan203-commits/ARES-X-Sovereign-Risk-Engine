import streamlit as st
import utils

df = utils.load_data()
c = st.session_state.get("country","USA")

row = utils.latest(df,c).copy()

row["inflation"] += st.slider("Inflation Shock",-5,10,2)

score,_ = utils.predict_full(row)

st.metric("Scenario Risk", score)
