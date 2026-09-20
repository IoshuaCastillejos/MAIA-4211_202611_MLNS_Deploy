#  We ensure proper path handling in Python
import Definitions
import streamlit as st
import pandas as pd

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Clasificador de ODS", page_icon="🌍"
)


### My vars

@st.cache_resource
def cargar_controlador():
    """Carga el modelo una sola vez y lo reutiliza entre interacciones."""
    return ModelController()


ctrl = cargar_controlador()


### My UI starting here

st.title(" Clasificador de textos por ODS")
st.caption(
    "Escribe un texto y el modelo identificara con cual Objetivo de "
    "Desarrollo Sostenible se relaciona."
)

with st.form(key="my_form"):
    texto = st.text_area(
        "Texto a clasificar",
        height=160,
        placeholder="Ej: La energia solar y eolica son clave para reducir "
                    "las emisiones de carbono",
    )
    submit_button = st.form_submit_button(label="Clasificar")

if submit_button:
    if not texto.strip():
        st.warning("Escribe un texto antes de clasificar.")
    else:
        resultados = ctrl.predict_proba(texto)
        ods, nombre, prob = resultados[0]

        st.caption("🎯 Resultado")
        relevantes = [(o, n, p) for o, n, p in resultados]

        if not relevantes:
            relevantes = resultados[:1]

        for o, n, p in relevantes:
            st.write(f"ODS {o} - probabilidad {p:.1%}")

