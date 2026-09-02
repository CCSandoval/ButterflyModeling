# ButterflyModeling

Modelado, evaluación comparativa y prototipo Streamlit para clasificación de
especies de mariposas del Tolima. Usa el dataset ya dividido/preprocesado por
[`ButterflyDataset`](../ButterflyDataset) (repo hermano).

Compara 5 arquitecturas con transfer learning (MobileNetV3Small,
EfficientNetV2B0, ResNet50V2, DenseNet121, ConvNeXtTiny) contra 3 modelos
obtenidos por destilación de conocimiento, todos sobre el mismo split
(semilla 42, 79 clases).

Ver `notebooks/` para el pipeline completo, de EDA a interpretabilidad
(Grad-CAM), y `streamlit_app.py` para el prototipo interactivo.
