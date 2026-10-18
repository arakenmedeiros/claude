"""Estilo comum das figuras do livro (Matplotlib)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COR = dict(ind="#4F46E5", indD="#312E81", cya="#06B6D4", grn="#10B981", amb="#F59E0B",
           red="#EF4444", vio="#8B5CF6", pnk="#EC4899", ink="#1F2937", gry="#6B7280",
           claro="#EEF2FF")
CICLO = [COR["ind"], COR["amb"], COR["grn"], COR["red"], COR["cya"], COR["vio"], COR["pnk"]]
PASTA_FIGS = os.environ.get("FIGS", os.path.join(os.path.dirname(__file__), "..", "..", "figs"))

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "axes.prop_cycle": matplotlib.cycler(color=CICLO),
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "mathtext.fontset": "dejavusans",
})


def salvar(fig, nome):
    os.makedirs(PASTA_FIGS, exist_ok=True)
    caminho = os.path.join(PASTA_FIGS, nome + ".png")
    fig.savefig(caminho, facecolor="white")
    plt.close(fig)
    return caminho
