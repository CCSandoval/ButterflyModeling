"""Prototipo de clasificación con ficha de especie y captura etiquetada."""

import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import streamlit as st
from PIL import Image

from scripts import fichas as catalogo
from scripts import modelRegistry as registro
from scripts.architectures import preprocessFn

CAPTURAS = Path("capturas")
TOP = 5


@st.cache_resource
def cargar():
    runId = registro.readJson(registro.OUTPUTS_DIR / "student.json")["run_id"]
    return (registro.modeloInferencia(runId),
            registro.loadRunConfig(runId)["architecture"],
            registro.loadRunMetrics(runId)["class_names"])


@st.cache_data
def cargarFichas():
    return catalogo.cargar()


@st.cache_data
def rutaImagen(especie):
    return str(catalogo.imagen(especie))


def enlace(texto, url):
    return f"[{texto}]({quote(url, safe=':/#?&=')})"


def mostrarFicha(ficha, especie):
    izquierda, derecha = st.columns([1, 2])
    with izquierda:
        st.image(rutaImagen(especie))
        st.caption(f"{ficha['atribucion']} · {ficha['licencia'].upper()}")
    with derecha:
        st.markdown(f"**Familia** {ficha['familia']}")
        if ficha["resumen_nivel"] != "especie":
            st.info(f"Wikipedia no tiene artículo de esta especie. El texto describe "
                    f"el {ficha['resumen_nivel']} *{ficha['resumen_ambito']}*.")
        st.write(re.sub(r"<[^>]+>", "", ficha["resumen"]))
        enlaces = [enlace("Wikipedia", ficha["wikipedia"])] if ficha["wikipedia"] else []
        st.caption(" · ".join(enlaces + [enlace("iNaturalist", ficha["inaturalist"])]))


st.set_page_config(page_title="Mariposas del Tolima", page_icon="🦋")
st.title("Mariposas del Tolima")

modelo, arquitectura, clases = cargar()
st.caption(f"{arquitectura} · {len(clases)} especies")

archivo = st.file_uploader("Foto de la mariposa", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.stop()

imagen = Image.open(archivo)
st.image(imagen, width=360)

lote = np.asarray(imagen.convert("RGB").resize(modelo.input_shape[1:3]), dtype="float32")
probabilidades = modelo.predict(preprocessFn(arquitectura)(lote[None]), verbose=0)[0]
orden = np.argsort(probabilidades)[::-1][:TOP]
especie = clases[orden[0]]

ficha = cargarFichas()[especie]
st.subheader(ficha["cientifico"])
if ficha["comun"]:
    st.caption(ficha["comun"])
st.progress(float(probabilidades[orden[0]]), text=f"{probabilidades[orden[0]]:.1%}")
mostrarFicha(ficha, especie)

st.dataframe(
    {"especie": [clases[i].replace("_", " ") for i in orden],
     "confianza": [f"{probabilidades[i]:.2%}" for i in orden]},
    hide_index=True, use_container_width=True,
)

# st.divider()
# confirmada = st.selectbox("Especie real", sorted(clases), index=sorted(clases).index(especie))
# if st.button("Guardar captura"):
#     carpeta = CAPTURAS / confirmada
#     carpeta.mkdir(parents=True, exist_ok=True)
#     destino = carpeta / f"{confirmada}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jpg"
#     imagen.convert("RGB").save(destino, quality=95)
#     st.success(f"Guardada en {destino}")
