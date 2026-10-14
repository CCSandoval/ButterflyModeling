"""Prototipo de clasificación con ficha de especie y captura etiquetada."""

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import streamlit as st
from PIL import Image

from scripts import fichas as catalogo
from scripts import gradcam
from scripts import modelRegistry as registro
from scripts.architectures import CABEZA_POOL, preprocessFn

CAPTURAS = Path("capturas")
TOP = 5


@st.cache_data
def catalogoModelos():
    """Los runs del mismo corpus que el modelo del sitio, mejor primero."""
    sitio = registro.readJson(registro.OUTPUTS_DIR / "student.json")
    runs = [r for r in registro.listRuns() if r["dataset"] == sitio["dataset"]]
    runs.sort(key=lambda r: -r["test_accuracy"])
    return {r["run_id"]: r for r in runs}, sitio["run_id"]


@st.cache_resource
def cargar(runId):
    return (registro.modeloInferencia(runId),
            registro.loadRunConfig(runId)["architecture"],
            registro.loadRunMetrics(runId)["class_names"])


@st.cache_data(show_spinner="Analizando la foto…")
def analizar(runId, datos):
    """Predicción y mapa de atención de una foto, en una sola pasada cacheada."""
    modelo, arquitectura, clases = cargar(runId)
    preprocesar = preprocessFn(arquitectura)
    imagen = Image.open(io.BytesIO(datos))
    cuadrada = imagen.convert("RGB").resize(modelo.input_shape[1:3])

    lote = np.asarray(cuadrada, dtype="float32")[None]
    probabilidades = modelo.predict(preprocesar(lote), verbose=0)[0]
    orden = np.argsort(probabilidades)[::-1][:TOP]

    mapa = gradcam.computeGradcam(modelo, cuadrada, int(orden[0]), preprocesar)
    return (cuadrada, gradcam.overlayHeatmap(cuadrada, mapa),
            [(clases[i], float(probabilidades[i])) for i in orden])


@st.cache_data
def cargarFichas():
    return catalogo.cargar()


@st.cache_data
def rutaImagen(especie):
    return str(catalogo.imagen(especie))


@st.cache_data
def estructura(runId):
    """Capas del modelo de inferencia, partidas en backbone y cabeza."""
    modelo, _, _ = cargar(runId)
    corte = [c.name for c in modelo.layers].index(CABEZA_POOL)
    return [{"parte": "backbone" if i < corte else "cabeza",
             "capa": c.name,
             "tipo": type(c).__name__,
             "salida": str(tuple(c.output.shape[1:])),
             "parámetros": c.count_params()}
            for i, c in enumerate(modelo.layers)]


def mostrarEstructura(runId, run):
    capas = estructura(runId)
    porParte = {p: sum(c["parámetros"] for c in capas if c["parte"] == p)
                for p in ("backbone", "cabeza")}
    total = sum(porParte.values())

    columnas = st.columns(4)
    columnas[0].metric("Parámetros", f"{total / 1e6:.2f} M")
    columnas[1].metric("Peso float32", f"{total * 4 / 2 ** 20:.1f} MB")
    columnas[2].metric("Capas", len(capas))
    columnas[3].metric("Cabeza", f"{porParte['cabeza'] / 1e6:.2f} M")

    st.caption(f"Backbone preentrenado en ImageNet con {porParte['backbone'] / 1e6:.2f} M "
               f"parámetros; durante el entrenamiento se descongeló su último "
               f"{run['finetuning']:.0%}. La cabeza de clasificación se entrenó entera.")
    st.caption(f"{run['architecture']} · macro-F1 {run['test_macro_f1']:.1%} · "
               f"{run['epochs_ran']} épocas · {run['training_minutes']:.0f} min "
               f"en {run['acelerador']} · run {runId}")
    # st.dataframe(capas, hide_index=True, use_container_width=True, height=320)


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

modelos, predeterminado = catalogoModelos()

def etiqueta(runId):
    run = modelos[runId]
    destilado = " destilado" if run["role"] == "destilado" else ""
    return f"{run['architecture']}{destilado} · {run['test_accuracy']:.1%}"

runId = st.sidebar.selectbox("Modelo", list(modelos), format_func=etiqueta,
                             index=list(modelos).index(predeterminado))
elegido = modelos[runId]
st.sidebar.caption(f"Reconoce {elegido['num_classes']} especies y acierta el "
                   f"{elegido['test_accuracy']:.0%} de las fotos de prueba, que no "
                   f"vio mientras aprendía.")

with st.expander("Estructura del modelo"):
    mostrarEstructura(runId, elegido)

archivo = st.file_uploader("Foto de la mariposa", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.stop()

cuadrada, atencion, predicciones = analizar(runId, archivo.getvalue())
especie, confianza = predicciones[0]

izquierda, derecha = st.columns(2)
izquierda.image(cuadrada, caption="Tu foto")
derecha.image(atencion, caption="En qué se fijó el modelo")

ficha = cargarFichas()[especie]
st.subheader(ficha["cientifico"])
if ficha["comun"]:
    st.caption(ficha["comun"])
st.progress(confianza, text=f"{confianza:.1%}")
mostrarFicha(ficha, especie)

st.dataframe(
    {"especie": [e.replace("_", " ") for e, _ in predicciones],
     "confianza": [f"{c:.2%}" for _, c in predicciones]},
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
