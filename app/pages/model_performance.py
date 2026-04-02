import streamlit as st
import utils

m = utils.metrics()

st.title("Model Performance")

st.metric("ROC AUC", round(m["roc_auc"],3))
st.metric("F1 Score", round(m["f1"],3))
st.metric("CV AUC", round(m["cv_auc"],3))
