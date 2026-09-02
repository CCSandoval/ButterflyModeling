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


def loadSplitManifest():
    config = loadConfig()
    path = (ROOT_DIR / config["split_file"]).resolve()
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def classNames():
    return sorted(loadSplitManifest()["reparto"].keys())


def classWeights():
    """Peso balanceado por clase"""
    reparto = loadSplitManifest()["reparto"]
    nombres = classNames()
    conteos = [len(reparto[n]["train"]) for n in nombres]
    y = np.repeat(np.arange(len(nombres)), conteos)
    pesos = compute_class_weight("balanced", classes=np.arange(len(nombres)), y=y)
    return {i: float(p) for i, p in enumerate(pesos)}


def materializar():
    """Symlinks locales hacia ButterflyDataset/Dataset."""
    config = loadConfig()
    origen = (ROOT_DIR / config["dataset_dir"]).resolve()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        enlace = DATASET_DIR / split
        if enlace.is_symlink() or enlace.exists():
            if enlace.is_symlink() or enlace.is_file():
                enlace.unlink()
            else:
                shutil.rmtree(enlace)
        enlace.symlink_to(origen / split)

    return DATASET_DIR
