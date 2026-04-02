import streamlit as st
import utils, intelligence

st.set_page_config(layout="wide")

st.title("🛡️ ARES-X Sovereign Intelligence Terminal")

st.markdown("### AI-driven macro risk → Decision engine")

df = utils.load_data()

country = st.selectbox("Country", df["country"].unique())

row = utils.latest(df, country)
risk = utils.predict(row)

brief = intelligence.generate_brief(row.iloc[0], risk)

# KPIs
c1,c2 = st.columns(2)
c1.metric("Risk Score", round(risk,3))

if risk > 0.8:
    c2.error("HIGH RISK")
elif risk > 0.5:
    c2.warning("MODERATE RISK")
else:
    c2.success("LOW RISK")

st.divider()

# EXECUTIVE BRIEF
st.header("🧠 Executive Intelligence")

st.subheader("Situation")
st.write(brief["situation"])

st.subheader("Diagnosis")
for d in brief["drivers"]:
    st.write("-", d)

st.subheader("Market Impact")
st.write(brief["impact"])

st.subheader("Decision")
st.success(brief["decision"])
