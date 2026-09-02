import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
REGISTRY_PATH = RUNS_DIR / "registry.json"
PROMOTED_MODEL_PATH = OUTPUTS_DIR / "model.keras"
CURRENT_RUN_PATH = OUTPUTS_DIR / "current_run.json"


def buildRunId(name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{name}"


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


def loadRegistry():
    return readJson(REGISTRY_PATH, default={"runs": []})


def registerRun(entry):
    registry = loadRegistry()
    runs = [r for r in registry.get("runs", []) if r.get("run_id") != entry["run_id"]]
    runs.append(entry)
    runs.sort(key=lambda r: r.get("run_id", ""), reverse=True)
    registry["runs"] = runs
    writeJson(REGISTRY_PATH, registry)
    return registry


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


def loadRunMetrics(runId):
    return readJson(runDir(runId) / "metrics.json", default={})


def loadRunConfig(runId):
    return readJson(runDir(runId) / "run.json", default={})


def promoteRun(reference):
    run = findRun(reference)
    runId = run["run_id"]
    source = runDir(runId) / "model.keras"
    if not source.exists():
        raise FileNotFoundError(f"El run {runId} no tiene model.keras")
    shutil.copyfile(source, PROMOTED_MODEL_PATH)
    metrics = loadRunMetrics(runId)
    writeJson(
        CURRENT_RUN_PATH,
        {
            "run_id": runId,
            "name": run.get("name"),
            "architecture": run.get("architecture"),
            "num_classes": run.get("num_classes"),
            "class_names": metrics.get("class_names", []),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Run promovido: {runId}")
    return runId


def resolvePromotedRun():
    return readJson(CURRENT_RUN_PATH, default=None)
