import streamlit as st
import utils

df = utils.load_data()

country = st.selectbox("Country", df["country"].unique())

row = utils.latest(df, country)

st.title("Explainability")

st.write(row[["inflation","gdp_growth","unemployment","interest_rate"]])
