import streamlit as st
import utils

df = utils.load_data()
c = st.session_state.get("country","USA")

d = utils.detect_regime(df,c)

st.write("Current Regime:", d.iloc[-1]["regime"])
