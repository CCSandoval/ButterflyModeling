import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"

SERIE_1 = "#2a78d6"
SERIE_2 = "#eb6834"
SERIE_3 = "#1baf7a"

RAMPA_AZUL = [
    "#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec",
    "#3987e5", "#256abf", "#184f95", "#0d366b",
]
CMAP_AZUL = LinearSegmentedColormap.from_list("rampaAzul", RAMPA_AZUL)


def applyStyle():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.size": 11,
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
    })


def limpiarEje(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plotConfusionMatrix(metrica, titulo):
    """Matriz de confusión normalizada por fila."""
    applyStyle()
    matriz = np.array(metrica["confusion_matrix"], dtype=float)
    porFila = matriz.sum(axis=1, keepdims=True)
    normal = np.divide(matriz, porFila, out=np.zeros_like(matriz), where=porFila > 0)
    n = len(metrica["class_names"])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.grid(False)
    imagen = ax.imshow(normal, cmap=CMAP_AZUL, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([0, n - 1], ["1", str(n)])
    ax.set_yticks([0, n - 1], ["1", str(n)])
    ax.set(xlabel="Predicción", ylabel="Especie real", title=titulo)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    barra.outline.set_visible(False)
    fig.tight_layout()
    return fig
