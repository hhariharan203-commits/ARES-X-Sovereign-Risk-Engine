import streamlit as st
import utils
import pydeck as pdk

df = utils.load_data()
risk = utils.global_risk(df)

risk["lat"] = 20
risk["lon"] = 0

layer = pdk.Layer(
    "ColumnLayer",
    data=risk,
    get_position='[lon, lat]',
    get_elevation='risk_score*1000000',
)

st.pydeck_chart(pdk.Deck(layers=[layer]))
