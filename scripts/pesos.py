import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO = "CCSandoval/ButterflyModeling"
API = f"https://api.github.com/repos/{REPO}"


def _token():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Falta GITHUB_TOKEN para publicar pesos")
    return token


def _api(url, datos=None, cuerpo=None, tipo=None, metodo=None):
    carga = cuerpo if cuerpo is not None else (json.dumps(datos).encode() if datos else None)
    pedido = urllib.request.Request(url, data=carga, method=metodo)
    pedido.add_header("Authorization", f"Bearer {_token()}")
    pedido.add_header("Accept", "application/vnd.github+json")
    if tipo:
        pedido.add_header("Content-Type", tipo)
    elif datos:
        pedido.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(pedido) as respuesta:
        crudo = respuesta.read()
    return json.loads(crudo) if crudo else {}


def _release(runId):
    """El release del run, creándolo si aún no existe."""
    try:
        return _api(f"{API}/releases/tags/{runId}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    return _api(f"{API}/releases", datos={
        "tag_name": runId, "name": runId,
        "body": f"Pesos entrenados del run {runId}.",
    })


def subir(runId, origen):
    """Sube model.keras como asset. Si ya estaba, lo reemplaza."""
    release = _release(runId)
    for asset in release.get("assets", []):
        if asset["name"] == "model.keras":
            _api(f"{API}/releases/assets/{asset['id']}", metodo="DELETE")
    url = release["upload_url"].split("{")[0] + "?name=model.keras"
    with open(origen, "rb") as archivo:
        _api(url, cuerpo=archivo.read(), tipo="application/octet-stream")
    print(f"pesos de {runId} publicados ({Path(origen).stat().st_size // 2**20} MB)")


def descargar(runId, destino):
    """Baja model.keras del release del run. Sin token si el repo es público."""
    url = f"https://github.com/{REPO}/releases/download/{runId}/model.keras"
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"bajando pesos de {runId}…")
    try:
        urllib.request.urlretrieve(url, destino)
    except urllib.error.HTTPError as error:
        destino.unlink(missing_ok=True)
        if error.code != 404:
            raise
        raise FileNotFoundError(
            f"No hay release para {runId}. Los runs entrenados antes de pasar a "
            f"Kaggle solo tienen los pesos en disco local; para publicarlos, "
            f"pesos.subir(runId, ruta) desde la máquina donde estén.") from None
    return destino
