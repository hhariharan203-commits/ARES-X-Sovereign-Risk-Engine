import streamlit as st
import utils, intelligence

df = utils.load_data()

country = st.selectbox("Country", df["country"].unique())

row = utils.latest(df, country)
risk = utils.predict(row)

brief = intelligence.generate_brief(row.iloc[0], risk)

st.title(f"{country} Intelligence")

st.metric("Risk Score", round(risk,3))

st.write(brief["situation"])
st.write("Drivers:", brief["drivers"])
st.write("Impact:", brief["impact"])
st.success(brief["decision"])
