# ButterflyModeling

Clasificación de especies de mariposas del Tolima con destilación de
conocimiento. La pregunta es si un modelo lo bastante pequeño para desplegar en
un navegador puede acercarse a uno grande, y cuánto de esa distancia la cierra
destilar en vez de simplemente entrenar mejor.

Este repositorio entrena y evalúa; el dataset lo produce
[ButterflyDataset](https://github.com/CCSandoval/ButterflyDataset).

## El dataset

100 especies observadas en el Tolima, 25 306 imágenes recortadas a 448 px, con
el reparto train/test/validate congelado en el manifiesto del repositorio
hermano. Todos los runs comparten ese reparto, así que las diferencias entre
ellos son del modelo y no de los datos.

## El experimento

Tres fases, cada una en su carpeta de `notebooks/`:

- **`01_student`** — cuatro candidatos a estudiante: MobileNetV3Small,
  MobileNetV2, EfficientNetB0 y NASNetMobile. Se elige por calidad **dentro de
  un presupuesto de descarga**, no por calidad sola.
- **`02_teacher`** — cinco candidatos a docente: EfficientNetV2S,
  EfficientNetV2B0, ResNet50V2, DenseNet121 e InceptionV3. Se elige por
  macro-F1, sin mirar costo.
- **`03_destilacion`** — cuatro variantes que transfieren del docente al
  estudiante: respuesta con y sin descongelado, features y atención.

Cierran `04_comparacion_final` y `05_interpretabilidad_gradcam`. El `00_eda`
describe el dataset y el `06`/`07` son diagnósticos que quedaron documentando
por qué el pipeline es como es.

La selección en cada fase usa validación y reporta test, y los runs se filtran
por dataset para que no se mezclen cifras de dos datasets distintos.

## Resultados

Cada entrenamiento deja su `run.json`, `metrics.json`, `history.json` y figuras
en `outputs/runs/<run_id>/`, con los pesos en un release de GitHub. El estado
al día está en `outputs/runs/registry.json`.