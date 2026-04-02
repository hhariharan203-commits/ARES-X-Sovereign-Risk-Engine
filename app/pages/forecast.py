import streamlit as st
import utils

df = utils.load_data()
c = st.session_state.get("country","USA")

row = utils.latest(df,c)

sims = utils.monte_carlo(row)

st.metric("Expected Risk", sims.mean())
