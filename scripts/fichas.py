"""Fichas de especie, del repo hermano en local y del repo público al desplegar."""

import json
import urllib.request
from pathlib import Path

CRUDO = "https://raw.githubusercontent.com/CCSandoval/ButterflyDataset/main"
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE = ROOT_DIR / "outputs" / "fichas"


def _repoLocal():
    with open(ROOT_DIR / "corpus.json", encoding="utf-8") as handle:
        ruta = (ROOT_DIR / json.load(handle)["dataset_repo"]).resolve()
    return ruta if (ruta / "fichas.json").exists() else None


def cargar():
    local = _repoLocal()
    if local:
        return json.loads((local / "fichas.json").read_text(encoding="utf-8"))
    with urllib.request.urlopen(f"{CRUDO}/fichas.json") as respuesta:
        return json.load(respuesta)


def imagen(especie):
    local = _repoLocal()
    if local:
        return local / "fichas" / f"{especie}.jpg"
    destino = CACHE / f"{especie}.jpg"
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{CRUDO}/fichas/{especie}.jpg", destino)
    return destino
