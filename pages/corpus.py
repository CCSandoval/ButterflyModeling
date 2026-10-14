"""De dónde salieron las imágenes con que se entrenó."""

from collections import Counter

import streamlit as st

from pages import comun

manifiesto = comun.manifiestoCorpus()
conteos, parametros = manifiesto["conteos"], manifiesto["parametros"]
umbrales = parametros["umbrales"]
porEspecie = {especie: sum(len(v) for v in splits.values())
              for especie, splits in manifiesto["reparto"].items()}

st.title("El conjunto de datos")
st.write("Fotos de iNaturalist con licencia abierta, recortadas alrededor de la mariposa.")

columnas = st.columns(4)
columnas[0].metric("Especies", manifiesto["num_clases"])
columnas[1].metric("Imágenes", f"{conteos['total']:,}".replace(",", " "))
columnas[2].metric("Por especie", f"{min(porEspecie.values())}–{max(porEspecie.values())}")
columnas[3].metric("Tamaño", f"{parametros['lado_salida']} px")

st.caption(f"Se reparten en {conteos['train']} de entrenamiento, {conteos['test']} de "
           f"prueba y {conteos['validate']} de validación, con semilla "
           f"{parametros['semilla']}. Corpus del {manifiesto['creado'][:10]}.")

st.divider()
st.subheader("Imágenes por especie")
st.bar_chart(dict(sorted(porEspecie.items(), key=lambda par: -par[1])),
             height=260, x_label="especie", y_label="imágenes")

fichas = comun.cargarFichas().values()
familias = Counter(f["familia"] for f in fichas).most_common()
licencias = Counter(f["licencia"].upper() for f in fichas).most_common()

izquierda, derecha = st.columns(2)
izquierda.subheader("Familias")
izquierda.dataframe({"familia": [k for k, _ in familias],
                     "especies": [v for _, v in familias]},
                    hide_index=True, use_container_width=True)
derecha.subheader("Licencias")
derecha.caption("De las imágenes de referencia, no del corpus completo")
derecha.dataframe({"licencia": [k for k, _ in licencias],
                   "imágenes": [v for _, v in licencias]},
                  hide_index=True, use_container_width=True)

st.divider()
st.subheader("Qué se descartó")
st.caption(f"Pasan el filtro las fotos con nitidez sobre {umbrales['nitidez']:.0f}, un "
           f"recorte de al menos {umbrales['ladoOriginal']} px, menos de "
           f"{umbrales['fracSaturados']:.0%} de píxeles quemados y "
           f"{umbrales['fracNegros']:.0%} en negro.")

if st.checkbox("Ver el detalle", help="Baja 9 MB de auditoría"):
    motivos = comun.motivosCorpus()
    total, aceptadas = sum(motivos.values()), motivos["aceptada"]
    st.caption(f"De {total} fotos candidatas se aceptaron {aceptadas}, "
               f"el {aceptadas / total:.0%}.")
    st.bar_chart({k: v for k, v in sorted(motivos.items(), key=lambda par: -par[1])
                  if k != "aceptada"}, height=260,
                 x_label="motivo del descarte", y_label="imágenes")
