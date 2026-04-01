import streamlit as st
import utils, intelligence
import plotly.graph_objects as go
from report import generate_report

df = utils.load_data()
c = st.session_state.get("country","USA")

d = df[df["country"]==c].sort_values(["year","month"])
row = utils.latest(df,c)

score = utils.predict(row)[0]
summary = intelligence.insight(score)

st.title(f"{c} Intelligence")

st.metric("Risk Score", round(score,3))
st.info(summary)

fig = go.Figure()
fig.add_trace(go.Scatter(y=d["gdp_growth"], name="GDP"))
fig.add_trace(go.Scatter(y=d["inflation"], name="Inflation"))
fig.add_trace(go.Scatter(y=d["unemployment"], name="Unemployment"))

st.plotly_chart(fig)

drivers = {
    "Inflation": row["inflation"].values[0],
    "GDP Growth": row["gdp_growth"].values[0],
    "Unemployment": row["unemployment"].values[0],
    "Interest Rate": row["interest_rate"].values[0]
}

st.bar_chart(drivers)

if st.button("Generate Report"):
    path = generate_report(c, score, summary, drivers)

    with open(path, "rb") as f:
        st.download_button("Download PDF", f, file_name="report.pdf")
