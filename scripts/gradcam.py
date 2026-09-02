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
