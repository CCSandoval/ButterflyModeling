"""Recorte cuadrado alrededor de la mariposa, igual que 02_preprocesamiento."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts import pesos

ROOT_DIR = Path(__file__).resolve().parent.parent
ETIQUETA = "detector"
ARCHIVO = "yolov8s-worldv2-mariposas.pt"

PROMPTS = ("butterfly", "moth", "insect", "")
CLASES_VALIDAS = 3
CONFIANZA_MIN = 0.05
IMGSZ = 640
MARGEN = 0.15
AREA_MIN_RELATIVA = 0.005
LADO_SALIDA = 448


def rutaDetector():
    local = ROOT_DIR / "outputs" / ARCHIVO
    if local.exists():
        return local
    try:
        return pesos.descargar(ETIQUETA, local, ARCHIVO)
    except FileNotFoundError:
        raise FileNotFoundError("No hay detector horneado publicado") from None


def cargarDetector():
    from ultralytics import YOLOWorld

    return YOLOWorld(str(rutaDetector()))


def hornear():
    """Incrusta los prompts en el checkpoint y lo publica, para no bajar CLIP."""
    from ultralytics import YOLOWorld

    destino = ROOT_DIR / "outputs" / ARCHIVO
    destino.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLOWorld("yolov8s-worldv2.pt")
    detector.set_classes(list(PROMPTS))
    del detector.model.clip_model
    detector.save(str(destino))

    print(f"{destino} ({destino.stat().st_size / 2 ** 20:.0f} MB)")
    pesos.subir(ETIQUETA, destino, ARCHIVO)
    return destino


def mejorCaja(resultado):
    cajas = resultado.boxes
    alto, ancho = resultado.orig_shape
    mejor, mejorConfianza = None, -1.0
    for i in range(len(cajas)):
        if int(cajas.cls[i]) >= CLASES_VALIDAS:
            continue
        x1, y1, x2, y2 = (float(v) for v in cajas.xyxy[i])
        if (x2 - x1) * (y2 - y1) < AREA_MIN_RELATIVA * ancho * alto:
            continue
        if float(cajas.conf[i]) > mejorConfianza:
            mejor, mejorConfianza = (x1, y1, x2, y2), float(cajas.conf[i])
    return mejor, mejorConfianza


def cuadradoDe(caja, ancho, alto):
    """El cuadrado se desplaza ante un borde en vez de rellenarse."""
    x1, y1, x2, y2 = caja
    lado = min(max(x2 - x1, y2 - y1) * (1 + 2 * MARGEN), ancho, alto)
    x = min(max((x1 + x2) / 2 - lado / 2, 0), ancho - lado)
    y = min(max((y1 + y2) / 2 - lado / 2, 0), alto - lado)
    return int(round(x)), int(round(y)), int(round(lado))


def redimensionar(imagen, lado):
    interpolacion = cv2.INTER_AREA if imagen.shape[0] > lado else cv2.INTER_CUBIC
    return cv2.resize(imagen, (lado, lado), interpolation=interpolacion)


def preparar(detector, imagen):
    """Recorte listo para el clasificador, más el detalle de cada paso."""
    rgb = np.asarray(imagen.convert("RGB"))
    alto, ancho = rgb.shape[:2]

    resultado = detector.predict(rgb[:, :, ::-1], imgsz=IMGSZ,
                                 conf=CONFIANZA_MIN, verbose=False)[0]
    caja, confianza = mejorCaja(resultado)

    if caja is None:
        lado = min(ancho, alto)
        x, y = (ancho - lado) // 2, (alto - lado) // 2
        desplazado = False
    else:
        x, y, lado = cuadradoDe(caja, ancho, alto)
        desplazado = (abs((caja[0] + caja[2] - lado) / 2 - x) > 1
                      or abs((caja[1] + caja[3] - lado) / 2 - y) > 1)

    recorte = redimensionar(rgb[y:y + lado, x:x + lado], LADO_SALIDA)
    pasos = {"ancho": ancho, "alto": alto, "caja": caja,
             "confianza": confianza if caja else None,
             "cuadrado": (x, y, lado), "ampliado": lado < LADO_SALIDA,
             "desplazado": desplazado}
    return Image.fromarray(recorte), pasos


def conCaja(imagen, caja):
    copia = imagen.convert("RGB").copy()
    if caja:
        grosor = max(2, round(min(copia.size) / 150))
        ImageDraw.Draw(copia).rectangle(caja, outline=(255, 90, 0), width=grosor)
    return copia
