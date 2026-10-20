import math
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def medirLatencia(model, repeticiones=30, calentamiento=3):
    """
    Mediana y desviación en milisegundos por imagen, con lote de 1.
    Mediana y no media porque la media se puede afectar con valores atípicos. 
    """
    alto, ancho = model.input_shape[1:3]
    entrada = tf.random.uniform((1, alto, ancho, 3), 0, 255)

    for _ in range(calentamiento):
        model(entrada, training=False)

    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        model(entrada, training=False)
        tiempos.append((time.perf_counter() - inicio) * 1000)

    return float(np.median(tiempos)), float(np.std(tiempos))


def evaluar(model, ds, classNames):
    """Predice sobre un Dataset ya preprocesado y devuelve accuracy, top-3, 
    F1 por clase y matriz de confusión."""
    probabilidades = model.predict(ds, verbose=0)
    yReal = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in ds])
    yPred = np.argmax(probabilidades, axis=1)
    ids = list(range(len(classNames)))

    reporte = classification_report(
        yReal, yPred, labels=ids, target_names=classNames,
        output_dict=True, zero_division=0,
    )

    k = min(3, len(classNames))
    mejores = np.argsort(-probabilidades, axis=1)[:, :k]
    top3 = float(np.mean([yReal[i] in mejores[i] for i in range(len(yReal))]))

    return {
        "num_samples": int(len(yReal)),
        "num_classes": len(classNames),
        "accuracy": float(reporte["accuracy"]),
        "top3_accuracy": top3,
        "macro_precision": float(reporte["macro avg"]["precision"]),
        "macro_recall": float(reporte["macro avg"]["recall"]),
        "macro_f1": float(reporte["macro avg"]["f1-score"]),
        "weighted_f1": float(reporte["weighted avg"]["f1-score"]),
        "per_class_f1": {n: reporte[n]["f1-score"] for n in classNames},
        "confusion_matrix": confusion_matrix(yReal, yPred, labels=ids).tolist(),
        "class_names": classNames,
        "y_true": yReal.tolist(),
        "y_pred": yPred.tolist(),
    }


def confusionesPrincipales(metrica):
    matriz = np.array(metrica["confusion_matrix"], dtype=float)
    nombres = metrica["class_names"]
    pares = []
    for i, real in enumerate(nombres):
        total = matriz[i].sum()
        for j, predicha in enumerate(nombres):
            if i != j and matriz[i, j] > 0:
                pares.append({
                    "real": real,
                    "predicha": predicha,
                    "casos": int(matriz[i, j]),
                    "tasa": round(matriz[i, j] / total, 3) if total else 0.0,
                })
    pares.sort(key=lambda p: (p["casos"], p["tasa"]), reverse=True)
    return pares[:15]


def intervaloWilson(aciertos, total, z=1.96):
    """IC de una proporción. Wilson y no normal porque no se degrada cerca de 0 o 1."""
    p = aciertos / total
    denominador = 1 + z ** 2 / total
    centro = p + z ** 2 / (2 * total)
    radio = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return (centro - radio) / denominador, (centro + radio) / denominador


def _macroF1(yReal, yPred, numClases):
    conteo = np.bincount(yReal * numClases + yPred,
                         minlength=numClases ** 2).reshape(numClases, numClases)
    tp = np.diag(conteo).astype(float)
    denominador = 2 * tp + (conteo.sum(axis=0) - tp) + (conteo.sum(axis=1) - tp)
    return float(np.divide(2 * tp, denominador, out=np.zeros_like(tp),
                           where=denominador > 0).mean())


def bootstrapMacroF1(yReal, yPred, replicas=4000, semilla=42, nivel=0.95):
    """IC percentil del macro-F1 remuestreando imágenes. No hay forma cerrada."""
    yReal, yPred = np.asarray(yReal), np.asarray(yPred)
    numClases = int(max(yReal.max(), yPred.max())) + 1
    total = len(yReal)
    generador = np.random.default_rng(semilla)
    muestras = np.empty(replicas)
    for r in range(replicas):
        indices = generador.integers(0, total, total)
        muestras[r] = _macroF1(yReal[indices], yPred[indices], numClases)
    muestras.sort()
    cola = (1 - nivel) / 2
    return float(muestras[int(cola * replicas)]), float(muestras[int((1 - cola) * replicas)])


def mcnemar(yReal, predA, predB):
    """
    Prueba exacta sobre los casos donde A y B difieren.
    """
    yReal, predA, predB = np.asarray(yReal), np.asarray(predA), np.asarray(predB)
    aciertaA, aciertaB = predA == yReal, predB == yReal
    soloA = int(np.sum(aciertaA & ~aciertaB))
    soloB = int(np.sum(~aciertaA & aciertaB))
    discordantes = soloA + soloB
    if not discordantes:
        return {"solo_a": 0, "solo_b": 0, "p": 1.0}
    menor = min(soloA, soloB)
    p = 2 * sum(math.comb(discordantes, i) for i in range(menor + 1)) / 2 ** discordantes
    return {"solo_a": soloA, "solo_b": soloB, "p": min(p, 1.0)}
