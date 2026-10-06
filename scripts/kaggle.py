"""Publica en GitHub los artefactos de un run entrenado en Kaggle.

El token nunca se escribe en disco: viaja por el entorno y git lo pide con un
credential helper, así que no queda en el remoto ni en `.git/config`.
"""

import os
import subprocess
from pathlib import Path

from scripts import pesos

CLON = Path("/tmp/ButterflyModeling")
AYUDA_CREDENCIAL = "!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f"

# los pesos están en .gitignore: a git solo van los artefactos chicos
PUBLICABLES = ("run.json", "metrics.json", "history.json", "imgs")


def git(*args):
    return subprocess.run(["git", "-c", f"credential.helper={AYUDA_CREDENCIAL}", *args],
                          cwd=CLON, check=True, capture_output=True, text=True).stdout.strip()


def publicarRun(runId):
    """El run se commitea antes del rebase y el registro después: así lo único
    que comparten dos sesiones es un archivo que se regenera, y el rebase nunca
    tiene que resolver un conflicto dentro de un JSON."""
    from scripts import modelRegistry as registro

    git("config", "user.email", "kaggle@noreply.local")
    git("config", "user.name", "kaggle")

    carpeta = Path("outputs/runs") / runId
    for nombre in PUBLICABLES:
        if (CLON / carpeta / nombre).exists():
            git("add", str(carpeta / nombre))
    if not git("status", "--porcelain"):
        print("sin cambios que publicar")
        return

    # antes del commit: si la subida falla, el run no queda registrado sin pesos
    pesos.subir(runId, CLON / carpeta / "model.keras")

    git("commit", "-m", f"Add run {runId}")
    git("pull", "--rebase")

    registro.registrar(runId)
    git("add", "outputs/runs/registry.json")
    if git("status", "--porcelain"):
        git("commit", "-m", "Refresh run registry")
    git("push")
    print(f"publicado {runId}")
