"""Prototipo de clasificación de mariposas del Tolima."""

import streamlit as st

from pages import comun

st.set_page_config(page_title="Mariposas del Tolima", page_icon="🦋", layout="wide")

comun.selectorModelo()

st.navigation([
    st.Page("paginas/clasificar.py", title="Identificar", icon="🦋", default=True),
    st.Page("paginas/modelo.py", title="Métricas del modelo", icon="🧠"),
    st.Page("paginas/corpus.py", title="Métricas del dataset", icon="📚"),
]).run()
