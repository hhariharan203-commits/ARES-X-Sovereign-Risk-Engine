import streamlit as st
import utils, intelligence

df = utils.add_risk(utils.load_data())

latest = df.sort_values("year").groupby("country").tail(1)

top = latest.sort_values("risk_score", ascending=False).head(3)

st.title("Executive Risk Brief")

for _,row in top.iterrows():
    brief = intelligence.generate_brief(row, row["risk_score"])

    st.markdown(f"""
    ### {row['country']} ({round(row['risk_score'],2)})

    **Situation:** {brief['situation']}

    **Drivers:** {', '.join(brief['drivers'])}

    **Impact:** {brief['impact']}

    **Decision:** {brief['decision']}
    """)
