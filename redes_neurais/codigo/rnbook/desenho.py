"""Auxiliares de desenho (diagramas de redes, caixas, setas) para as figuras do livro."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from .estilo import COR


def caixa(ax, x, y, w, h, texto, cor=None, fc=None, fs=10, tc=None, lw=1.5, bold=False):
    cor = cor or COR["ind"]
    fc = fc or "#EEF2FF"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=cor, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
            color=tc or COR["ink"], zorder=3, fontweight="bold" if bold else "normal")


def seta(ax, p, q, cor=None, lw=1.6, estilo="-|>", rad=0.0, ms=12, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=estilo, mutation_scale=ms, lw=lw,
                                 color=cor or COR["gry"], connectionstyle=f"arc3,rad={rad}",
                                 zorder=1, linestyle=ls))


def diagrama_rede(ax, camadas, rotulos=None, cores=None, raio=0.12, passo_x=1.6, mostrar_pesos=True,
                  titulos=None, fs=9):
    """Desenha um MLP totalmente conectado. camadas: lista com nº de neurônios por camada."""
    cores = cores or [COR["cya"]] + [COR["ind"]] * (len(camadas) - 2) + [COR["amb"]]
    nmax = max(camadas)
    pos = []
    for l, n in enumerate(camadas):
        ys = ((n - 1) / 2.0 - np.arange(n)) * 0.55   # primeiro neurônio no topo
        pos.append([(l * passo_x, y) for y in ys])
    for l in range(len(camadas) - 1):
        for a in pos[l]:
            for b in pos[l + 1]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=COR["gry"], lw=0.7, alpha=0.55, zorder=1)
    for l, ps in enumerate(pos):
        for k, (x, y) in enumerate(ps):
            ax.add_patch(Circle((x, y), raio, fc=cores[l], ec="white", lw=1.2, zorder=3))
            if rotulos and rotulos[l]:
                ax.text(x, y, rotulos[l][k], color="white", ha="center", va="center", fontsize=fs - 1, zorder=4)
    if titulos:
        for l, t in enumerate(titulos):
            ax.text(l * passo_x, (nmax / 2.0) * 0.55 + 0.35, t, ha="center", va="bottom", fontsize=fs, color=COR["ink"])
    ax.set_xlim(-0.6, (len(camadas) - 1) * passo_x + 0.6)
    ax.set_ylim(-(nmax / 2.0) * 0.55 - 0.5, (nmax / 2.0) * 0.55 + 1.0)
    ax.set_aspect("equal"); ax.axis("off")
    return pos
