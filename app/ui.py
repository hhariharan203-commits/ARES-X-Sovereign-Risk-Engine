import streamlit as st

def apply_theme():

    st.markdown("""
    <style>

    /* ─────────────────────────────
       GLOBAL BACKGROUND
    ───────────────────────────── */
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }

    /* ─────────────────────────────
       HEADINGS
    ───────────────────────────── */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ─────────────────────────────
       METRICS (KEY VISUAL)
    ───────────────────────────── */
    div[data-testid="metric-container"] {
        background: #161b22;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363d;
    }

    /* ─────────────────────────────
       SIDEBAR
    ───────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ─────────────────────────────
       BUTTONS
    ───────────────────────────── */
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }

    .stButton>button:hover {
        background-color: #2ea043;
    }

    /* ─────────────────────────────
       SUCCESS / WARNING / ERROR
    ───────────────────────────── */
    .stAlert-success {
        background-color: #12261f;
        color: #3fb950;
    }

    .stAlert-warning {
        background-color: #2d1f00;
        color: #f2cc60;
    }

    .stAlert-error {
        background-color: #2c1617;
        color: #ff7b72;
    }

    /* ─────────────────────────────
       TABLES
    ───────────────────────────── */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 8px;
    }

    </style>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(f"## {title}")
    st.divider()


def info_block(title, content):
    st.markdown(f"### {title}")
    st.write(content)
