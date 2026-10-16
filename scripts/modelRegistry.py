import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts import pesos

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
REGISTRY_PATH = RUNS_DIR / "registry.json"


def buildRunId(name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{name}"


def commitActual():
    """Commit del que salió el código; sufijo -sucio si hay cambios sin commitear."""
    def corre(*args):
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True)

    try:
        hecho = corre("rev-parse", "--short", "HEAD")
    except OSError:
        return None
    if hecho.returncode:
        return None
    commit = hecho.stdout.strip()
    return f"{commit}-sucio" if corre("status", "--porcelain", "-uno").stdout.strip() else commit


def runDir(runId):
    return RUNS_DIR / runId


def createRunDir(runId):
    path = runDir(runId)
    (path / "imgs").mkdir(parents=True, exist_ok=True)
    return path


def writeJson(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def readJson(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def entradaDe(runId):
    config = loadRunConfig(runId)
    if not config:
        return None
    splits = loadRunMetrics(runId).get("splits", {})
    hiper = config.get("hyperparameters", {})
    destilacion = config.get("distillation", {})
    return {
        "run_id": config.get("run_id", runId),
        "name": config.get("name"),
        "dataset": config.get("dataset"),
        "architecture": config.get("architecture"),
        "role": config.get("role"),
        "teacher_run_id": destilacion.get("teacher_run_id"),
        "distillation": destilacion.get("tipo"),
        "finetuning": hiper.get("finetuning", destilacion.get("finetuning")),
        "num_classes": config.get("num_classes"),
        "epochs_ran": hiper.get("epochs_ran"),
        "training_minutes": config.get("training_minutes"),
        "test_accuracy": splits.get("test", {}).get("accuracy"),
        "test_macro_f1": splits.get("test", {}).get("macro_f1"),
        "validate_accuracy": splits.get("validate", {}).get("accuracy"),
        "validate_macro_f1": splits.get("validate", {}).get("macro_f1"),
        "acelerador": config.get("environment", {}).get("acelerador", "cpu"),
        "commit": config.get("environment", {}).get("commit"),
        "notes": config.get("notes"),
        "created_at": config.get("started_at"),
    }


def loadRegistry():
    runs = [e for e in (entradaDe(d.name) for d in sorted(RUNS_DIR.glob("*")) if d.is_dir())
            if e]
    runs.sort(key=lambda r: r.get("run_id", ""), reverse=True)
    return {"runs": runs}


def registrar(runId):
    """Refresca la caché del registro tras escribir run.json y metrics.json."""
    registry = loadRegistry()
    writeJson(REGISTRY_PATH, registry)
    return entradaDe(runId)


def listRuns():
    return loadRegistry().get("runs", [])


def findRun(reference):
    runs = listRuns()
    for run in runs:
        if run.get("run_id") == reference:
            return run
    matches = [
        r for r in runs
        if r.get("run_id", "").endswith(reference) or r.get("name") == reference
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(m["run_id"] for m in matches)
        raise ValueError(f"'{reference}' es ambiguo, coincide con: {ids}")
    raise ValueError(f"No existe el run '{reference}'")


def rutaModelo(runId):
    local = runDir(runId) / "model.keras"
    if local.exists():
        return local
    from scripts.corpus import buscarEnEntrada

    for montado in buscarEnEntrada(f"{runId}/model.keras"):
        return montado
    return pesos.descargar(runId, local)


def loadRunMetrics(runId):
    return readJson(runDir(runId) / "metrics.json", default={})


def loadRunConfig(runId):
    return readJson(runDir(runId) / "run.json", default={})


def modeloInferencia(runId):
    """El modelo sin las capas de augmentation, que el conversor de TF.js no
    acepta. No tienen pesos, así que las dos listas coinciden.

    Reconstruye la topología con buildModel y del archivo lee solo los pesos:
    deserializar la configuración rompe cuando Kaggle guarda con una Keras más
    nueva que la local.
    """
    from scripts.architectures import buildModel

    config = loadRunConfig(runId)
    arquitectura, clases = config["architecture"], config["num_classes"]

    entrenado = buildModel(arquitectura, clases, aumentar=True)
    entrenado.load_weights(rutaModelo(runId))

    limpio = buildModel(arquitectura, clases, aumentar=False)
    limpio.set_weights(entrenado.get_weights())
    return limpio
