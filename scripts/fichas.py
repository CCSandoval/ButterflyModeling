"""Datos del corpus: del repo hermano en local y del repo público al desplegar."""

import csv
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path

CRUDO = "https://raw.githubusercontent.com/CCSandoval/ButterflyDataset/main"
ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE = ROOT_DIR / "outputs" / "fichas"


def _repoLocal():
    with open(ROOT_DIR / "corpus.json", encoding="utf-8") as handle:
        ruta = (ROOT_DIR / json.load(handle)["dataset_repo"]).resolve()
    return ruta if (ruta / "fichas.json").exists() else None


def _texto(nombre):
    local = _repoLocal()
    if local:
        return (local / nombre).read_text(encoding="utf-8")
    with urllib.request.urlopen(f"{CRUDO}/{nombre}") as respuesta:
        return respuesta.read().decode("utf-8")


def cargar():
    return json.loads(_texto("fichas.json"))


def manifiesto():
    return json.loads(_texto("manifiesto.json"))


def motivos():
    """Veredictos de la auditoría. Son 9 MB: pedirlo solo cuando se muestren."""
    filas = csv.DictReader(io.StringIO(_texto("auditoria.csv")))
    return Counter(fila["motivo"] for fila in filas)


def imagen(especie):
    local = _repoLocal()
    if local:
        return local / "fichas" / f"{especie}.jpg"
    destino = CACHE / f"{especie}.jpg"
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{CRUDO}/fichas/{especie}.jpg", destino)
    return destino
