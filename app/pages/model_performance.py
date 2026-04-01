import streamlit as st
import utils
import plotly.graph_objects as go

m = utils.metrics()

st.metric("ROC AUC", m["roc_auc"])

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=m["roc_auc"],
    gauge={'axis':{'range':[0,1]}}
))

st.plotly_chart(fig)
