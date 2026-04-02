import streamlit as st
import utils
import numpy as np

df = utils.add_risk(utils.load_data())

df["regime"] = np.where(df["risk_score"]>0.8,"Crisis",
                np.where(df["risk_score"]>0.5,"Stress","Stable"))

st.title("Regime Distribution")

st.bar_chart(df["regime"].value_counts())
