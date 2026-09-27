from pathlib import Path

import tensorflow as tf

RESPALDOS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "respaldos"


def callbacksEntrenamiento(runName, paciencia):
    """Callbacks comunes a todos los entrenamientos del proyecto.

    `BackupAndRestore` guarda pesos y número de época al final de cada una y
    reanuda solo al volver a llamar `fit()`; Se indexa por `runName` y no por
    `run_id` porque cada ejecución del notebook genera un `run_id` nuevo
    """
    respaldo = RESPALDOS_DIR / runName
    respaldo.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=paciencia, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(respaldo / "estado")),
        tf.keras.callbacks.CSVLogger(str(respaldo / "epocas.csv"), append=True),
    ]
