"""Identificar una mariposa a partir de una foto."""

import re
from urllib.parse import quote

import streamlit as st
from PIL import Image

from paginas import comun
from scripts import recorte

FONDO = (252, 252, 251)     # el backgroundColor del tema, para el relleno

PASOS = (
    ("🔍", "1 · Localizar", "El detector encuentra la mariposa"),
    ("✂️", "2 · Recortar", f"Cuadrado a {recorte.LADO_SALIDA} px, como en el corpus"),
    ("🔥", "3 · Explicar", "Dónde miró el modelo para decidir"),
)


def enlace(texto, url):
    return f"[{texto}]({quote(url, safe=':/#?&=')})"


def aCuadrado(imagen):
    """Centra la imagen en un lienzo cuadrado: los tres pasos vienen con
    proporciones distintas y en fila quedaban desalineados."""
    lado = max(imagen.size)
    lienzo = Image.new("RGB", (lado, lado), FONDO)
    lienzo.paste(imagen.convert("RGB"),
                 ((lado - imagen.size[0]) // 2, (lado - imagen.size[1]) // 2))
    return lienzo


def resultado(ficha, confianza):
    with st.container(border=True):
        retrato, datos = st.columns([1, 4], vertical_alignment="center")
        retrato.image(comun.rutaImagen(especie))
        datos.markdown(f"### *{ficha['cientifico']}*")
        subtitulo = ficha["comun"] or "sin nombre común registrado"
        datos.markdown(f"{subtitulo} &nbsp;&nbsp; :blue-badge[{ficha['familia']}]")
        cifra, barra = datos.columns([1, 3], vertical_alignment="center")
        cifra.metric("Confianza", f"{confianza:.1%}", border=False)
        barra.progress(confianza)
        barra.caption(f"{ficha['atribucion']} · {ficha['licencia'].upper()}")


def pasos(original, cuadrada, atencion, detalle):
    imagenes = (recorte.conCaja(original, detalle["caja"]) if detalle else original,
                cuadrada, atencion)
    for columna, (_, titulo, pie), imagen in zip(st.columns(3), PASOS, imagenes):
        with columna.container(border=True):
            st.markdown(f"**{titulo}**")
            st.image(aCuadrado(imagen))
            st.caption(pie)

    if detalle is None:
        st.caption("Sin recorte automático: la foto entera se deformó a "
                   f"{recorte.LADO_SALIDA} px, que no es como se entrenó el modelo.")
        return
    if detalle["caja"] is None:
        st.warning("El detector no encontró ninguna mariposa, así que se usó el "
                   "cuadrado central. La predicción es poco fiable.")
        return

    x, y, lado = detalle["cuadrado"]
    with st.expander(f"Detalle del recorte · {detalle['confianza']:.0%} de confianza "
                     f"en la detección"):
        st.write(
            f"Foto de {detalle['ancho']}×{detalle['alto']} px. El cuadrado toma el lado "
            f"mayor de la caja más {recorte.MARGEN:.0%} de margen a cada lado: {lado} px "
            f"desde ({x}, {y})"
            + (", desplazado para no salirse del borde" if detalle["desplazado"] else "")
            + f", y {'ampliado' if detalle['ampliado'] else 'reducido'} a "
              f"{recorte.LADO_SALIDA} px.")


def descripcion(ficha, predicciones):
    with st.container(border=True):
        if ficha["resumen_nivel"] != "especie":
            st.info(f"Wikipedia no tiene artículo de esta especie. El texto describe "
                    f"el {ficha['resumen_nivel']} *{ficha['resumen_ambito']}*.")
        st.write(re.sub(r"<[^>]+>", "", ficha["resumen"]))
        enlaces = [enlace("Wikipedia", ficha["wikipedia"])] if ficha["wikipedia"] else []
        st.caption(" · ".join(enlaces + [enlace("iNaturalist", ficha["inaturalist"])]))

    st.caption("Las cinco más probables")
    for nombre, valor in predicciones:
        fila, peso = st.columns([3, 2], vertical_alignment="center")
        fila.markdown(f"*{nombre.replace('_', ' ')}*")
        peso.progress(valor, text=f"{valor:.1%}")


runId, _ = comun.modeloActual()

st.title("Identificar una mariposa")
st.write("Sube una foto y el modelo dirá de qué especie del Tolima se trata.")

izquierda, derecha = st.columns([2, 3], vertical_alignment="bottom")
archivo = izquierda.file_uploader("Foto de la mariposa", type=["jpg", "jpeg", "png"])
recortarAuto = derecha.checkbox(
    "Localizar la mariposa antes de clasificar", value=True,
    help="Recorta un cuadrado alrededor de la mariposa, igual que en el corpus de "
         "entrenamiento. Sin esto el modelo ve la foto completa deformada.")

if archivo is None:
    st.caption("Qué pasa con tu foto")
    for columna, (icono, titulo, pie) in zip(st.columns(3), PASOS):
        with columna.container(border=True):
            st.markdown(f"## {icono}")
            st.markdown(f"**{titulo}**")
            st.caption(pie)

    _, elegido = comun.modeloActual()
    izquierda, derecha = st.columns([3, 2], gap="large")
    derecha.markdown(
        f"El modelo reconoce **{elegido['num_classes']} especies** con presencia "
        f"registrada en el Tolima, y siempre responde con una de ellas: si la mariposa "
        f"de la foto no está entre esas cien, dirá la que más se le parezca. El mapa de "
        f"atención ayuda a notarlo — cuando el modelo se fija en el fondo en vez de en "
        f"el animal, conviene desconfiar."
    )
    with izquierda.container(border=True):
        st.markdown("**Qué foto funciona mejor**")
        st.markdown(
            "- La mariposa nítida y reconocible\n"
            "- Sin contraluz ni zonas quemadas\n"
            "- De cerca, pero con el ejemplar entero"
        )
        st.caption("Son los mismos criterios con que se filtró el corpus.")
    st.stop()

cuadrada, atencion, detalle, predicciones = comun.analizar(
    runId, archivo.getvalue(), recortarAuto)
especie, confianza = predicciones[0]
ficha = comun.cargarFichas()[especie]

resultado(ficha, confianza)

# los pasos y la descripción, lado a lado: la identificación se lee sin bajar
columnaPasos, columnaFicha = st.columns([3, 2], gap="large")
with columnaPasos:
    pasos(Image.open(archivo), cuadrada, atencion, detalle)
with columnaFicha:
    descripcion(ficha, predicciones)
