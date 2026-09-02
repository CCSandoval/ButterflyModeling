import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


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
