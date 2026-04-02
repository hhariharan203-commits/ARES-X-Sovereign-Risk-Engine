import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    .stApp { background: #0b0f17; color: white; }
    </style>
    """, unsafe_allow_html=True)
