# ButterflyModeling

Modelado, evaluación comparativa y prototipo Streamlit para clasificación de
especies de mariposas del Tolima. Usa el dataset ya dividido/preprocesado por
[`ButterflyDataset`](../ButterflyDataset) (repo hermano).

Entrena candidatos a estudiante (`01_student`) y a docente (`02_teacher`) con
transfer learning, elige uno de cada rol por criterios distintos —el docente por
macro-F1, el estudiante por costo/calidad— y destila el primero en el segundo
(`03_destilacion_*`). Todo sobre el mismo split (semilla 42, 79 clases).

Ver `notebooks/` para el pipeline completo, de EDA a interpretabilidad
(Grad-CAM), y `streamlit_app.py` para el prototipo interactivo.
