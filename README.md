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

El clon usa `GIT_LFS_SKIP_SMUDGE=1`: descargar los pesos consume la cuota
mensual de LFS y no hacen falta para entrenar. Subirlos no consume ancho de
banda, pero igual se quedan fuera del commit — a git van solo `run.json`,
`metrics.json`, `history.json` y las figuras.

El token nunca toca el disco: viaja por el entorno y git lo pide con un
credential helper, así que no queda en el remoto ni en `.git/config`. El clon
va a `/tmp` y no a `/kaggle/working` porque esa carpeta se guarda como salida
del notebook.

No hay nada que configurar: `corpus.py` detecta Kaggle y toma el único dataset
adjunto que traiga `versiones/`.

Para entrenar sobre otra versión sin tocar `corpus.json`, exportar
`BUTTERFLY_VERSION`.

## El registro de runs

`outputs/runs/registry.json` es caché, no fuente de verdad: se reconstruye
recorriendo el `run.json` y el `metrics.json` de cada run. Por eso dos sesiones
que entrenan en paralelo no chocan — cada una solo escribe dentro de su propia
carpeta— y por eso borrar el registro no pierde nada.
