import streamlit as st
import utils, intelligence

df = utils.load_data()
c = st.session_state.get("country","USA")

row = utils.latest(df,c)
score, tier = utils.predict_full(row)

intel = intelligence.generate_intelligence(row.iloc[0], score)

st.title(c)
st.metric("Risk", round(score,3))

st.write(intel)
