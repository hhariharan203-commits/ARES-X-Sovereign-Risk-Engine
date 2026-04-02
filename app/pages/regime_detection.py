import streamlit as st
import utils

df = utils.load_data()
c = st.session_state.get("country","USA")

st.dataframe(utils.detect_regime(df,c))
