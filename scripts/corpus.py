import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_CONFIG_PATH = ROOT_DIR / "corpus.json"
ENLACE_DATASET = ROOT_DIR / "dataset"
SPLITS = ("train", "test", "validate")
EN_KAGGLE = Path("/kaggle").exists()

DATASET_DIR = Path("/tmp/dataset") if EN_KAGGLE else ENLACE_DATASET


def loadConfig():
    with open(CORPUS_CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def buscarEnEntrada(sufijo):
    return sorted(Path("/kaggle/input").glob(f"datasets/*/*/{sufijo}"))


def repoDataset():
    if not EN_KAGGLE:
        return (ROOT_DIR / loadConfig()["dataset_repo"]).resolve()
    montados = buscarEnEntrada("manifiesto.json")
    if len(montados) != 1:
        adjuntos = sorted(str(d) for d in Path("/kaggle/input").glob("*/*"))
        raise RuntimeError(f"Se esperaba un dataset con manifiesto.json adjunto a "
                           f"la sesión; hay {len(montados)}. Bajo /kaggle/input: "
                           f"{adjuntos[:6]}")
    return montados[0].parent


def manifiesto():
    """Parámetros que produjeron el corpus y qué archivo va a qué split."""
    ruta = repoDataset() / "manifiesto.json"
    if not ruta.exists():
        raise FileNotFoundError(f"No hay manifiesto en {ruta.parent}; corre "
                                f"02_preprocesamiento en ButterflyDataset.")
    with open(ruta, encoding="utf-8") as handle:
        return json.load(handle)


def classNames():
    return sorted(manifiesto()["reparto"].keys())


def classWeights():
    """Peso balanceado por clase a partir de los conteos de train del split
    (mismo criterio que sklearn.compute_class_weight, sin iterar el tf.data)."""
    reparto = manifiesto()["reparto"]
    nombres = classNames()
    conteos = [len(reparto[n]["train"]) for n in nombres]
    y = np.repeat(np.arange(len(nombres)), conteos)
    pesos = compute_class_weight("balanced", classes=np.arange(len(nombres)), y=y)
    return {i: float(p) for i, p in enumerate(pesos)}


def _limpiar(ruta):
    if ruta.is_symlink() or ruta.is_file():
        ruta.unlink()
    elif ruta.is_dir():
        shutil.rmtree(ruta)


def materializar():
    """Symlinks en dataset/<split>/<especie>/. El split ya viene en el manifiesto."""
    reparto = manifiesto()["reparto"]
    origen = repoDataset() / "procesado"
    if not origen.exists():
        raise FileNotFoundError(f"No existe {origen}; corre 02_preprocesamiento "
                                f"en ButterflyDataset.")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _limpiar(DATASET_DIR / split)

    # los notebooks leen "dataset/<split>" relativo a la raíz
    if DATASET_DIR != ENLACE_DATASET:
        _limpiar(ENLACE_DATASET)
        ENLACE_DATASET.symlink_to(DATASET_DIR)

    for especie, splits in reparto.items():
        for split in SPLITS:
            carpeta = DATASET_DIR / split / especie
            carpeta.mkdir(parents=True, exist_ok=True)
            for archivo in splits[split]:
                (carpeta / archivo).symlink_to(origen / especie / archivo)

    return DATASET_DIR
