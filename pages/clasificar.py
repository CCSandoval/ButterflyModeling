"""Identificar una mariposa a partir de una foto."""

import re
from urllib.parse import quote

import streamlit as st
from PIL import Image

from pages import comun
from scripts import recorte


def enlace(texto, url):
    return f"[{texto}]({quote(url, safe=':/#?&=')})"


def mostrarPasos(original, cuadrada, atencion, pasos):
    columnas = st.columns(3)
    columnas[0].image(recorte.conCaja(original, pasos["caja"]) if pasos else original,
                      caption="1 · Tu foto, con la mariposa localizada")
    columnas[1].image(cuadrada, caption=f"2 · Recorte cuadrado a {recorte.LADO_SALIDA} px")
    columnas[2].image(atencion, caption="3 · En qué se fijó el modelo")

    if pasos is None:
        st.caption("Sin recorte automático: la foto entera se deformó a "
                   f"{recorte.LADO_SALIDA} px, que no es como se entrenó el modelo.")
        return
    if pasos["caja"] is None:
        st.warning("El detector no encontró ninguna mariposa, así que se usó el "
                   "cuadrado central de la foto. La predicción es poco fiable.")
        return

    x, y, lado = pasos["cuadrado"]
    escala = "ampliado" if pasos["ampliado"] else "reducido"
    st.caption(
        f"Mariposa localizada con {pasos['confianza']:.0%} de confianza en una foto de "
        f"{pasos['ancho']}×{pasos['alto']} px. El cuadrado toma el lado mayor de la caja "
        f"más {recorte.MARGEN:.0%} de margen a cada lado: {lado} px desde ({x}, {y})"
        + (", desplazado para no salirse del borde" if pasos["desplazado"] else "")
        + f", y {escala} a {recorte.LADO_SALIDA} px.")


def mostrarFicha(ficha, especie):
    izquierda, derecha = st.columns([1, 2])
    with izquierda:
        st.image(comun.rutaImagen(especie))
        st.caption(f"{ficha['atribucion']} · {ficha['licencia'].upper()}")
    with derecha:
        st.markdown(f"**Familia** {ficha['familia']}")
        if ficha["resumen_nivel"] != "especie":
            st.info(f"Wikipedia no tiene artículo de esta especie. El texto describe "
                    f"el {ficha['resumen_nivel']} *{ficha['resumen_ambito']}*.")
        st.write(re.sub(r"<[^>]+>", "", ficha["resumen"]))
        enlaces = [enlace("Wikipedia", ficha["wikipedia"])] if ficha["wikipedia"] else []
        st.caption(" · ".join(enlaces + [enlace("iNaturalist", ficha["inaturalist"])]))


runId, _ = comun.modeloActual()

st.title("Identificar una mariposa")
st.write("Sube una foto y el modelo dirá de qué especie del Tolima se trata.")

recortarAuto = st.checkbox(
    "Localizar la mariposa antes de clasificar", value=True,
    help="Recorta un cuadrado alrededor de la mariposa, igual que en el corpus de "
         "entrenamiento. Sin esto el modelo ve la foto completa deformada.")

archivo = st.file_uploader("Foto de la mariposa", type=["jpg", "jpeg", "png"])
if archivo is None:
    st.stop()

cuadrada, atencion, pasos, predicciones = comun.analizar(
    runId, archivo.getvalue(), recortarAuto)
especie, confianza = predicciones[0]

mostrarPasos(Image.open(archivo), cuadrada, atencion, pasos)

st.divider()
ficha = comun.cargarFichas()[especie]
st.subheader(ficha["cientifico"])
if ficha["comun"]:
    st.caption(ficha["comun"])
st.progress(confianza, text=f"{confianza:.1%}")
mostrarFicha(ficha, especie)

st.divider()
st.caption("Las cinco especies más probables")
st.dataframe(
    {"especie": [e.replace("_", " ") for e, _ in predicciones],
     "confianza": [f"{c:.2%}" for _, c in predicciones]},
    hide_index=True, use_container_width=True,
)
