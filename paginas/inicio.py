"""Presentación del proyecto."""

import streamlit as st

from paginas import comun
from scripts import modelRegistry as registro

GALERIA = ("morpho_helenor", "siproeta_stelenes", "heliconius_erato",
           "danaus_plexippus", "heraclides_thoas", "hamadryas_feronia")

eleccion = registro.readJson(registro.OUTPUTS_DIR / "student.json")
modelos, predeterminado = comun.catalogoModelos()
manifiesto = comun.manifiestoCorpus()
fichas = comun.cargarFichas()
sitio, mejor = modelos[predeterminado], modelos[next(iter(modelos))]
destilado = next((r for r in modelos.values() if r["role"] == "destilado"), None)

st.title("🦋 Mariposas del Tolima")
st.subheader("Reconoce 100 especies a partir de una fotografía", divider="green")

izquierda, derecha = st.columns([3, 1], vertical_alignment="center")
izquierda.write(
    "Un clasificador entrenado con fotografías de ciencia ciudadana tomadas en el "
    "departamento del Tolima. Sube una foto y el prototipo localiza la mariposa, propone "
    "una especie, muestra en qué se fijó para decidirlo, y la acompaña de una ficha con "
    "su imagen de referencia."
)
derecha.page_link("paginas/clasificar.py", label="**Probar con una foto**", icon="🦋")

for columna, especie in zip(st.columns(len(GALERIA)), GALERIA):
    columna.image(comun.rutaImagen(especie))
    columna.caption(f"*{fichas[especie]['cientifico']}*")

columnas = st.columns(4)
columnas[0].metric("Especies", manifiesto["num_clases"])
columnas[1].metric("Imágenes", f"{manifiesto['conteos']['total']:,}".replace(",", " "))
columnas[2].metric("Modelos entrenados", len(modelos))
columnas[3].metric("Mejor acierto", f"{mejor['test_accuracy']:.1%}")

st.subheader("La pregunta", divider="green")
izquierda, derecha = st.columns([2, 3])
izquierda.write(
    f"Un modelo que quepa en un teléfono o en un navegador tiene que ser pequeño, y lo "
    f"pequeño suele costar precisión. El proyecto mide cuánto cuesta, y si la destilación "
    f"de conocimiento —entrenar al modelo chico imitando a uno grande— cierra esa "
    f"distancia.\n\n"
    f"La distancia existe pero es angosta. El mejor modelo, **{mejor['architecture']}**, "
    f"acierta el {mejor['test_accuracy']:.1%}; el que sirve este sitio, "
    f"**{sitio['architecture']}**, acierta el {sitio['test_accuracy']:.1%} pesando "
    f"{eleccion['pesos_MB']} MB, el único bajo el presupuesto de "
    f"{eleccion['presupuesto_MB']:.0f} MB. Son "
    f"{100 * (mejor['test_accuracy'] - sitio['test_accuracy']):.1f} puntos."
)
if destilado:
    izquierda.write(
        f"Destilar no cerró esa brecha: la versión destilada del modelo pequeño quedó en "
        f"{destilado['test_accuracy']:.1%}, por debajo del {sitio['test_accuracy']:.1%} "
        f"del mismo modelo entrenado de la forma corriente. Un resultado negativo, y "
        f"parte de lo que el trabajo reporta."
    )
derecha.bar_chart(
    [{"modelo": f"{r['architecture']}"
               f"{' destilado' if r['role'] == 'destilado' else ''}",
      "acierto": r["test_accuracy"]} for r in modelos.values()],
    x="modelo", y="acierto", horizontal=True, height=320,
    x_label="acierto en prueba", y_label="",
)

st.subheader("Cómo funciona", divider="green")
pasos = (
    ("🔍 Localizar", "Un detector encuentra la mariposa y recorta un cuadrado a su "
                     "alrededor, igual que se construyó el corpus. Sin ese paso el modelo "
                     "vería la foto completa, deformada, y acertaría menos."),
    ("🧠 Clasificar", "Una red preentrenada en ImageNet, con la cabeza reemplazada y "
                      "reentrenada sobre las especies del Tolima, reparte su confianza "
                      "entre las 100."),
    ("🔥 Explicar", "Grad-CAM colorea las zonas que más pesaron en la decisión. Sirve "
                    "para distinguir un acierto por el patrón alar de uno por el fondo."),
)
for columna, (titulo, cuerpo) in zip(st.columns(3), pasos):
    with columna.container(border=True):
        st.markdown(f"#### {titulo}")
        st.write(cuerpo)

st.subheader("Explora", divider="green")
izquierda, centro, derecha = st.columns(3)
izquierda.page_link("paginas/clasificar.py", label="**Identificar** · sube una foto",
                    icon="🦋")
centro.page_link("paginas/modelo.py", label="**El modelo** · capa por capa", icon="🧠")
derecha.page_link("paginas/corpus.py", label="**El dataset** · de dónde salió", icon="📚")
st.caption(f"El selector de la barra lateral cambia entre los {len(modelos)} modelos "
           f"entrenados sobre este mismo corpus.")

st.divider()
st.caption(
    "Las fotografías provienen de [iNaturalist](https://www.inaturalist.org) bajo "
    "licencias Creative Commons abiertas, y cada ficha muestra la atribución de su "
    "imagen. El corpus lo produce "
    "[ButterflyDataset](https://github.com/CCSandoval/ButterflyDataset); el entrenamiento "
    "y la evaluación están en "
    "[ButterflyModeling](https://github.com/CCSandoval/ButterflyModeling)."
)
