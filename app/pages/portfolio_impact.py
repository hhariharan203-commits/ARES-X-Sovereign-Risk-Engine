import streamlit as st
import utils

df = utils.load_data()

country = st.selectbox("Country", df["country"].unique())

exposure = st.slider("Exposure %",0,100,50)

row = utils.latest(df, country)

risk = utils.predict(row)

loss = exposure * risk

st.title("Portfolio Impact")

st.metric("Risk", round(risk,3))
st.metric("Expected Loss %", round(loss,2))
