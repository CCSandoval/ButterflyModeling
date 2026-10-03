import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def preprocessImage(img: Image.Image, targetSize: tuple, preprocessFn) -> np.ndarray:
    img = img.convert("RGB").resize(targetSize)
    arr = np.asarray(img).astype(np.float32)
    arr = preprocessFn(arr)
    return np.expand_dims(arr, axis=0)


def computeGradcam(model, img: Image.Image, classIndex: int, convLayer: str, preprocessFn) -> np.ndarray:
    gradModel = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(convLayer).output, model.output],
    )

    targetSize = (model.input_shape[2], model.input_shape[1])
    imgArray = preprocessImage(img, targetSize, preprocessFn)

    with tf.GradientTape() as tape:
        convOutputs, predictions = gradModel(imgArray)
        loss = predictions[:, classIndex]

    grads = tape.gradient(loss, convOutputs)
    pooledGrads = tf.reduce_mean(grads, axis=(0, 1, 2))

    convOutputs = convOutputs.numpy()[0]
    pooledGrads = pooledGrads.numpy()

    for i in range(pooledGrads.shape[-1]):
        convOutputs[:, :, i] *= pooledGrads[i]

    heatmap = np.mean(convOutputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    maxVal = np.max(heatmap)
    if maxVal > 0:
        heatmap /= maxVal

    return heatmap


def cajaEnRecorte(fila) -> list:
    """Caja de la mariposa en coordenadas normalizadas del recorte cuadrado.

    `auditoria.csv` guarda la detección de YOLO-World en el espacio de la
    imagen original y el recorte como origen + lado, así que hay que trasladar
    y escalar. Se recorta a [0,1] porque la caja puede salirse del cuadrado.
    """
    lado = fila["ladoOriginal"]
    caja = [
        (fila["x1"] - fila["cuadradoX"]) / lado,
        (fila["y1"] - fila["cuadradoY"]) / lado,
        (fila["x2"] - fila["cuadradoX"]) / lado,
        (fila["y2"] - fila["cuadradoY"]) / lado,
    ]
    return [min(max(v, 0.0), 1.0) for v in caja]


def areaCaja(caja) -> float:
    """Fracción del recorte que ocupa la mariposa: es la línea base contra la
    que se compara la atención. Una atención repartida al azar daría esto."""
    x1, y1, x2, y2 = caja
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def fraccionDentro(heatmap: np.ndarray, caja) -> float:
    """Fracción de la activación de Grad-CAM que cae dentro de la caja.

    Comparar contra `areaCaja`: si coinciden, el modelo no está mirando la
    mariposa más que el fondo.
    """
    alto, ancho = heatmap.shape
    x1, y1, x2, y2 = caja
    i0, i1 = int(round(y1 * alto)), int(round(y2 * alto))
    j0, j1 = int(round(x1 * ancho)), int(round(x2 * ancho))
    i1, j1 = max(i1, i0 + 1), max(j1, j0 + 1)

    total = heatmap.sum()
    if total <= 0:
        return float("nan")
    return float(heatmap[i0:i1, j0:j1].sum() / total)


def overlayHeatmap(img: Image.Image, heatmap: np.ndarray) -> Image.Image:
    alpha = 0.4
    imgRgb = np.asarray(img.convert("RGB"))
    h, w = imgRgb.shape[:2]

    heatmapResized = cv2.resize(heatmap, (w, h))
    heatmapUint8 = np.uint8(255 * heatmapResized)
    heatmapColored = cv2.applyColorMap(heatmapUint8, cv2.COLORMAP_JET)
    heatmapColored = cv2.cvtColor(heatmapColored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(imgRgb, alpha, heatmapColored, 1 - alpha, 0)
    return Image.fromarray(superimposed)
