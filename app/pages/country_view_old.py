from __future__ import annotations

import streamlit as st


def main():
    st.title('Deprecated Page')
    st.warning('This legacy page has been retired. Please use the main navigation pages.')
    st.stop()


if __name__ == '__main__':
    main()

