import streamlit as st
import utils

df = utils.load_data()
st.dataframe(utils.global_risk(df))
