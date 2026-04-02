import streamlit as st
import utils
import plotly.express as px

df = utils.load_data()

st.title("Macro Feature Space")

fig = px.scatter_3d(df,
    x="gdp_growth",
    y="inflation",
    z="unemployment",
    color="interest_rate"
)

st.plotly_chart(fig)
