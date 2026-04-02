import streamlit as st
import utils
import numpy as np

df = utils.load_data()
c = st.session_state.get("country","USA")

row = utils.latest(df,c)

alloc = st.slider("Exposure %",0,100,50)

score,_ = utils.predict_full(row)

loss = alloc * score

st.metric("Expected Loss %", round(loss,2))
