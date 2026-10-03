import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_CONFIG_PATH = ROOT_DIR / "corpus.json"
DATASET_DIR = ROOT_DIR / "dataset"
SPLITS = ("train", "test", "validate")


def loadConfig():
    with open(CORPUS_CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def repoDataset():
    return (ROOT_DIR / loadConfig()["dataset_repo"]).resolve()


def versionActual():
    return loadConfig()["version"]


def listarVersiones():
    directorio = repoDataset() / "versiones"
    return sorted(p.stem for p in directorio.glob("*.json")) if directorio.exists() else []


def loadSplitManifest():
    """Manifiesto de la versión seleccionada en corpus.json. Trae los parámetros
    que la produjeron y qué archivo va a qué split."""
    nombre = versionActual()
    ruta = repoDataset() / "versiones" / f"{nombre}.json"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe la versión '{nombre}' en {ruta.parent}. Hay: {listarVersiones()}")
    with open(ruta, encoding="utf-8") as handle:
        return json.load(handle)


def classNames():
    return sorted(loadSplitManifest()["reparto"].keys())


def classWeights():
    """Peso balanceado por clase a partir de los conteos de train del split
    (mismo criterio que sklearn.compute_class_weight, sin iterar el tf.data)."""
    reparto = loadSplitManifest()["reparto"]
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
    """Symlinks locales en dataset/<split>/<especie>/ según la versión elegida.

    El split ya viene decidido en el manifiesto; esto no lo recalcula. Soporta
    los dos layouts que puede declarar una versión: `por_split` enlaza las tres
    carpetas de golpe, `plano` enlaza archivo por archivo desde el pool
    compartido, que es lo que permite que varias versiones no dupliquen disco.
    """
    manifiesto = loadSplitManifest()
    origen = repoDataset() / manifiesto["imagenes"]["raiz"]
    layout = manifiesto["imagenes"]["layout"]

    if not origen.exists():
        raise FileNotFoundError(
            f"La versión '{versionActual()}' apunta a {origen}, que no existe. "
            f"Si es una versión nueva, falta poblar el pool en ButterflyDataset:\n"
            f"    from corpus import versiones\n"
            f"    versiones.materializarPool('{versionActual()}')")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _limpiar(DATASET_DIR / split)

    if layout == "por_split":
        for split in SPLITS:
            (DATASET_DIR / split).symlink_to(origen / split)
    elif layout == "plano":
        for especie, splits in manifiesto["reparto"].items():
            for split in SPLITS:
                carpeta = DATASET_DIR / split / especie
                carpeta.mkdir(parents=True, exist_ok=True)
                for archivo in splits[split]:
                    (carpeta / archivo).symlink_to(origen / especie / archivo)
    else:
        raise ValueError(f"Layout desconocido: {layout}")

    return DATASET_DIR
