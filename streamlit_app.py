"""Prototipo de clasificación y recolección de capturas etiquetadas."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from scripts import modelRegistry as registro
from scripts.architectures import IMG_SIZE, preprocessFn

CAPTURAS = Path("capturas")
TOP = 5


@st.cache_resource
def cargar():
    info = registro.readJson(registro.CURRENT_RUN_PATH)
    if info is None:
        return None, None
    ruta = registro.PROMOTED_MODEL_PATH
    if not ruta.exists():
        ruta = registro.rutaModelo(info["run_id"])
    return tf.keras.models.load_model(ruta), info


def predecir(modelo, info, imagen):
    tamano = tuple(modelo.input_shape[1:3])
    lote = np.asarray(imagen.convert("RGB").resize(tamano), dtype="float32")[None]
    probabilidades = modelo.predict(preprocessFn(info["architecture"])(lote), verbose=0)[0]
    orden = np.argsort(probabilidades)[::-1][:TOP]
    return [(info["class_names"][i], float(probabilidades[i])) for i in orden]


def guardar(imagen, especie):
    carpeta = CAPTURAS / especie
    carpeta.mkdir(parents=True, exist_ok=True)
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = carpeta / f"{especie}_{marca}.jpg"
    imagen.convert("RGB").save(destino, quality=95)
    return destino


st.set_page_config(page_title="Mariposas del Tolima", page_icon="🦋")
st.title("Mariposas del Tolima")

modelo, info = cargar()
if modelo is None:
    st.error("No hay modelo promovido. Corre `registro.promoteRun(run_id)` primero.")
    st.stop()

tamano = tuple(modelo.input_shape[1:3])
st.caption(f"{info['architecture']} · {info['num_classes']} especies · {tamano[0]}px · "
           f"promovido {info['promoted_at'][:10]}")
if tamano != IMG_SIZE:
    st.warning(f"El modelo promovido espera {tamano[0]}px y la configuración actual "
               f"es {IMG_SIZE[0]}px: es anterior al corpus vigente. Vuelve a promover.")

archivo = st.file_uploader("Foto de la mariposa", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.stop()

imagen = Image.open(archivo)
st.image(imagen, width=360)

predicciones = predecir(modelo, info, imagen)
especie, confianza = predicciones[0]
st.subheader(especie.replace("_", " "))
st.progress(confianza, text=f"{confianza:.1%}")

st.dataframe(
    {"especie": [e.replace("_", " ") for e, _ in predicciones],
     "confianza": [f"{c:.2%}" for _, c in predicciones]},
    hide_index=True, use_container_width=True,
)

st.divider()
st.caption("Captura etiquetada para el conjunto de prueba del mariposario")
confirmada = st.selectbox("Especie real", sorted(info["class_names"]),
                          index=sorted(info["class_names"]).index(especie))
if st.button("Guardar captura"):
    st.success(f"Guardada en {guardar(imagen, confirmada)}")
