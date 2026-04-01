import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    body { background:#0b0f17; color:#e6edf3; }
    .card {
        background:linear-gradient(145deg,#111826,#0d1117);
        padding:18px;
        border-radius:14px;
        border:1px solid rgba(255,255,255,0.05);
    }
    .kpi { font-size:28px; font-weight:700; }
    .label { font-size:12px; opacity:0.6; }
    </style>
    """, unsafe_allow_html=True)

def kpi(title, value):
    st.markdown(f"""
    <div class="card">
        <div class="label">{title}</div>
        <div class="kpi">{value}</div>
    </div>
    """, unsafe_allow_html=True)
