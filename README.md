# ButterflyModeling

Identificación automática de 100 especies de mariposas del Tolima a partir de una fotografía. Este repositorio contiene el prototipo desplegado y el código que entrenó y evaluó los modelos; el corpus lo produce [ButterflyDataset](https://github.com/CCSandoval/ButterflyDataset).

El prototipo localiza la mariposa con un detector, clasifica el recorte, muestra con Grad-CAM en qué zonas se apoyó la decisión y acompaña la predicción con una ficha de la especie.

## Correr el prototipo

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No hace falta nada más. En el primer arranque descarga lo que necesita:

| qué | de dónde | cuánto |
|---|---|---|
| pesos del clasificador | release de este repo | 6 MB |
| detector YOLO-World | release `detector` de este repo | 25 MB |
| fichas y manifiesto del corpus | ButterflyDataset por HTTP | 1 MB |
| imagen de referencia por especie | ButterflyDataset por HTTP | 100 KB, bajo demanda |

El modelo que sirve el prototipo es el que indica `outputs/student.json`. El selector de la barra lateral permite cambiar entre los diez modelos entrenados sobre el mismo corpus.

## Replicar

Hay tres niveles, y **ninguno exige cuenta de Kaggle, token ni crear releases**.

### 1 · Usar el prototipo

Los dos comandos de arriba. Baja los pesos ya entrenados de los releases de este repositorio, que son públicos y anónimos.

### 2 · Verificar los resultados

`notebooks/1. Entrenamiento/Seleccion` y `notebooks/3. Analisis/`, en local, reproducen la tabla de las nueve arquitecturas con sus latencias, la elección del estudiante y del docente, la comparación final y los mapas de atención. Los pesos se bajan solos; hace falta el corpus en disco, que se reconstruye con los notebooks de [ButterflyDataset](https://github.com/CCSandoval/ButterflyDataset) y es la parte lenta.

### 3 · Reentrenar

`notebooks/1. Entrenamiento/` tiene un notebook por arquitectura. En CPU cada uno toma horas, así que conviene una GPU; en este caso se usó Kaggle, pero el notebook es agnóstico, detecta el entorno y si está en Kaggle clona este repo para tener los módulos a mano.

El run queda en `outputs/runs/<run_id>/` **con sus pesos en disco**, y todo lo de abajo los encuentra ahí: `registro.rutaModelo` mira primero el disco local y solo recurre al release si no están. **Quien reentrena no necesita publicar nada.**

Publicar el run en un release es un paso opcional que solo hacen los autores y requiere un secreto `GITHUB_TOKEN`. Sin él los notebooks corren igual y lo avisan al arrancar.

Los notebooks encuentran el corpus por la ruta que declara `corpus.json`, relativa a este repositorio.

## Qué hay en notebooks/

| carpeta | qué contiene |
|---|---|
| `1. Entrenamiento/` | un notebook por arquitectura, nueve en total, y `Seleccion`, que de la misma tabla saca el estudiante (el mejor dentro del presupuesto de descarga) y el docente (el mejor sin restricción) |
| `2. Destilacion/` | destilación de respuesta y, sobre ella, transferencia de atención |
| `3. Analisis/` | comparación final con las destilaciones incluidas, Grad-CAM y la latencia de la aplicación |
| `4. Diagnosticos/` | `Calentamiento`: con y sin fase congelada, en las dos arquitecturas que colapsaban |

Se corren en ese orden. Para reproducir el prototipo basta `1. Entrenamiento/`;lo demás sostiene los resultados del artículo.

## Resultados

El modelo desplegado, MobileNetV3Small, acierta el 90.6% en el conjunto de prueba ocupando 5.7 MB. El mejor de los diez, EfficientNetB0, llega al 94.5% con 19.6 MB.
