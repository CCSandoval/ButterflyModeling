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

## Ejecución en Kaggle

Los notebooks de entrenamiento corren igual en local y en Kaggle: la primera
celda detecta el entorno y, solo en Kaggle, clona este repositorio, monta el
dataset adjunto y publica el run en GitHub al terminar.

Preparación, una vez:

1. Subir una versión del dataset (`procesado/<version>/` y `versiones/`) como
   Kaggle Dataset y adjuntarla a la sesión.
2. Crear un PAT fine-grained con `Contents: read and write` sobre este
   repositorio y guardarlo en Kaggle Secrets como `GITHUB_TOKEN`.
3. Habilitar Internet en el notebook.

El token nunca toca el disco: viaja por el entorno y git lo pide con un
credential helper, así que no queda en el remoto ni en `.git/config`. El clon
va a `/tmp` y no a `/kaggle/working` porque esa carpeta se guarda como salida
del notebook.

No hay nada que configurar: `corpus.py` detecta Kaggle y toma el único dataset
adjunto que traiga `versiones/`.

### Los pesos

`model.keras` está en `.gitignore` y el repo ya no usa LFS. A git van solo
`run.json`, `metrics.json`, `history.json` y las figuras; los pesos van a
**GitHub Releases**, un release por run.

En Kaggle es la única forma de que sobrevivan a la sesión: el clon está en
`/tmp` y la plataforma solo conserva `/kaggle/working`. Los assets de release
no consumen cuota de LFS, y mientras el repo sea público bajarlos no pide
token.

`publicarRun` los sube antes de commitear. Los notebooks que cargan un modelo
ajeno — destilación, comparación, Grad-CAM — usan `registro.rutaModelo(runId)`,
que lo busca en disco, luego en un dataset adjunto, y si no lo baja del
release y lo cachea. La destilación encuentra a su docente sin pasos manuales.

Para entrenar sobre otra versión sin tocar `corpus.json`, exportar
`BUTTERFLY_VERSION`.

## El registro de runs

`outputs/runs/registry.json` es caché, no fuente de verdad: se reconstruye
recorriendo el `run.json` y el `metrics.json` de cada run. Por eso dos sesiones
que entrenan en paralelo no chocan — cada una solo escribe dentro de su propia
carpeta— y por eso borrar el registro no pierde nada.
