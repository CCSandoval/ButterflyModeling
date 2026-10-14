"""Estructura del modelo seleccionado."""

import streamlit as st

from pages import comun

runId, run = comun.modeloActual()
capas = comun.estructura(runId)
porParte = {p: sum(c["parámetros"] for c in capas if c["parte"] == p)
            for p in ("backbone", "cabeza")}
total = sum(porParte.values())

st.title("El modelo")
st.write(f"**{run['architecture']}**, una red preentrenada en ImageNet a la que se le "
         f"cambió la cabeza para que distinga {run['num_classes']} especies del Tolima.")

columnas = st.columns(4)
columnas[0].metric("Parámetros", f"{total / 1e6:.2f} M")
columnas[1].metric("Peso float32", f"{total * 4 / 2 ** 20:.1f} MB")
columnas[2].metric("Capas", len(capas))
columnas[3].metric("Aciertos en test", f"{run['test_accuracy']:.1%}")

st.caption(f"El backbone aporta {porParte['backbone'] / 1e6:.2f} M parámetros y durante el "
           f"entrenamiento se descongeló su último {run['finetuning']:.0%}. La cabeza de "
           f"clasificación, {porParte['cabeza'] / 1e6:.2f} M, se entrenó entera.")

st.divider()
st.subheader("Capa por capa")
st.dataframe(capas, hide_index=True, use_container_width=True, height=420)

st.caption(f"macro-F1 {run['test_macro_f1']:.1%} · {run['epochs_ran']} épocas · "
           f"{run['training_minutes']:.0f} min en {run['acelerador']} · run {runId}")
