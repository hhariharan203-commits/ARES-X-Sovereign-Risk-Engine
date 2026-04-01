import streamlit as st
import utils
import plotly.graph_objects as go

df = utils.load_data()
c = st.session_state.get("country","USA")

row = utils.latest(df,c)
sims = utils.monte_carlo(row)

fig = go.Figure()
fig.add_histogram(x=sims)

st.plotly_chart(fig)

st.metric("Expected Risk", round(sims.mean(),3))
