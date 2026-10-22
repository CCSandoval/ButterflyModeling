"""Carga y estado que comparten las páginas."""

import io

import numpy as np
import streamlit as st
from PIL import Image

from scripts import fichas as catalogo
from scripts import gradcam
from scripts import modelRegistry as registro
from scripts import recorte
from scripts.architectures import CABEZA_POOL, preprocessFn

TOP = 5


@st.cache_data
def catalogoModelos():
    """Los runs del mismo corpus que el modelo del sitio, mejor primero."""
    sitio = (registro.readJson(registro.OUTPUTS_DIR / "desplegado.json")
             or registro.readJson(registro.OUTPUTS_DIR / "student.json"))
    runs = [r for r in registro.listRuns() if r["dataset"] == sitio["dataset"]]
    runs.sort(key=lambda r: -r["test_accuracy"])
    return {r["run_id"]: r for r in runs}, sitio["run_id"]


@st.cache_resource
def cargar(runId):
    return (registro.modeloInferencia(runId),
            registro.loadRunConfig(runId)["architecture"],
            registro.loadRunMetrics(runId)["class_names"])


@st.cache_resource
def detector():
    return recorte.cargarDetector()


@st.cache_data(show_spinner="Analizando la foto…")
def analizar(runId, datos, recortarAuto):
    """Recorte, predicción y mapa de atención, en una sola pasada cacheada."""
    modelo, arquitectura, clases = cargar(runId)
    preprocesar = preprocessFn(arquitectura)
    imagen = Image.open(io.BytesIO(datos))

    if recortarAuto:
        cuadrada, pasos = recorte.preparar(detector(), imagen)
    else:
        cuadrada, pasos = imagen.convert("RGB").resize(modelo.input_shape[1:3]), None

    lote = np.asarray(cuadrada, dtype="float32")[None]
    probabilidades = modelo.predict(preprocesar(lote), verbose=0)[0]
    orden = np.argsort(probabilidades)[::-1][:TOP]

    mapa = gradcam.computeGradcam(modelo, cuadrada, int(orden[0]), preprocesar)
    return (cuadrada, gradcam.overlayHeatmap(cuadrada, mapa), pasos,
            [(clases[i], float(probabilidades[i])) for i in orden])


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


@st.cache_data
def cargarFichas():
    return catalogo.cargar()


@st.cache_data
def rutaImagen(especie):
    return str(catalogo.imagen(especie))


@st.cache_data
def manifiestoCorpus():
    return catalogo.manifiesto()


@st.cache_data(show_spinner="Leyendo la auditoría…")
def motivosCorpus():
    return dict(catalogo.motivos())


def selectorModelo():
    """Barra lateral compartida. La dibuja el script de entrada, que corre en
    todas las páginas, para que el widget no pierda su valor al navegar."""
    modelos, predeterminado = catalogoModelos()

    def etiqueta(runId):
        run = modelos[runId]
        destilado = " destilado" if run["role"] == "destilado" else ""
        return f"{run['architecture']}{destilado} · {run['test_accuracy']:.1%}"

    runId = st.sidebar.selectbox("Modelo", list(modelos), format_func=etiqueta,
                                 index=list(modelos).index(predeterminado), key="modelo")
    elegido = modelos[runId]
    st.sidebar.caption(f"Reconoce {elegido['num_classes']} especies y acierta el "
                       f"{elegido['test_accuracy']:.0%} de las fotos de prueba, que no "
                       f"vio mientras aprendía.")


def modeloActual():
    modelos, _ = catalogoModelos()
    runId = st.session_state["modelo"]
    return runId, modelos[runId]
