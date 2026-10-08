"""Publica en GitHub los artefactos de un run entrenado en Kaggle."""

import os
import subprocess
from pathlib import Path

from scripts import pesos

CLON = Path("/tmp/ButterflyModeling")
AYUDA_CREDENCIAL = "!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f"

PUBLICABLES = ("run.json", "metrics.json", "history.json", "imgs")

SUELTOS = ("outputs/current_run.json", "outputs/student.json", "outputs/teacher.json")


def git(*args):
    hecho = subprocess.run(["git", "-c", f"credential.helper={AYUDA_CREDENCIAL}", *args],
                           cwd=CLON, capture_output=True, text=True)
    if hecho.returncode:
        detalle = hecho.stderr.strip() or hecho.stdout.strip() or "sin salida"
        raise RuntimeError(f"git {' '.join(args)} -> {hecho.returncode}\n{detalle}")
    return hecho.stdout.strip()


def publicarArchivos(mensaje, *rutas):
    """Para notebooks que no producen un run: comparaciones y figuras."""
    git("config", "user.email", "kaggle@noreply.local")
    git("config", "user.name", "kaggle")
    for ruta in rutas:
        if (CLON / ruta).exists():
            git("add", str(ruta))
    if git("status", "--porcelain"):
        git("commit", "-m", mensaje)
    git("pull", "--rebase")
    if not git("log", "--oneline", "@{u}..HEAD"):
        print("nada que publicar")
        return
    git("push")
    print(mensaje)


def publicarRun(runId):
    """El run se commitea antes del rebase y el registro después."""
    from scripts import modelRegistry as registro

    git("config", "user.email", "kaggle@noreply.local")
    git("config", "user.name", "kaggle")

    carpeta = Path("outputs/runs") / runId
    for nombre in PUBLICABLES:
        if (CLON / carpeta / nombre).exists():
            git("add", str(carpeta / nombre))
    for suelto in SUELTOS:
        if (CLON / suelto).exists():
            git("add", suelto)

    if git("status", "--porcelain"):
        pesos.subir(runId, CLON / carpeta / "model.keras")
        git("commit", "-m", f"Add run {runId}")

    git("checkout", "--", "outputs/runs/registry.json")
    sucio = git("status", "--porcelain")
    if sucio:
        raise RuntimeError(f"El árbol tiene cambios sin publicar antes del rebase:\n{sucio}")

    git("pull", "--rebase")

    registro.registrar(runId)
    git("add", "outputs/runs/registry.json")
    if git("status", "--porcelain"):
        git("commit", "-m", "Refresh run registry")

    if not git("log", "--oneline", "@{u}..HEAD"):
        print(f"{runId} ya estaba publicado")
        return
    git("push")
    print(f"publicado {runId}")
