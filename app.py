# =============================================================================
# SheetSense AI — Streamlit Interactive Host
# =============================================================================
# Embeds the Stitch-designed high-density productivity dashboard.
# Avoids heavy Streamlit full-page reloads and clunky AI widgets.
# =============================================================================

import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SheetSense AI — Interactive Spreadsheet Agent",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

# Custom minimal styles to remove Streamlit container padding and header
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding: 0rem !important;
        margin: 0rem !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
        height: 100vh !important;
    }
</style>
""", unsafe_allow_html=True)

html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=960, scrolling=False)
else:
    st.error("Static UI template not found at static/index.html")
