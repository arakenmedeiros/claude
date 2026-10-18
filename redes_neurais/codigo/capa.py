"""Capa do livro (794x1123 px)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
W, H = 794, 1123
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
# fundo em degradê
g = np.linspace(0, 1, 256)[:, None]; c0 = np.array([0x1E, 0x1B, 0x4B]) / 255; c1 = np.array([0x4F, 0x46, 0xE5]) / 255
img = (c0 * (1 - g) + c1 * g)[:, None, :].repeat(2, 1)
ax.imshow(img, extent=[0, W, 0, H], aspect="auto", origin="upper", zorder=0)
# rede estilizada
camadas = [4, 6, 6, 3]; xs = [130, 290, 500, 665]; ys0 = 565; rng = np.random.default_rng(1); pos = []
for n, x in zip(camadas, xs):
    ys = np.linspace(ys0 - 160, ys0 + 160, n) if n > 1 else [ys0]; pos.append([(x, y) for y in ys])
for a, b in zip(pos[:-1], pos[1:]):
    for (x1, y1) in a:
        for (x2, y2) in b:
            ax.plot([x1, x2], [y1, y2], color="#A5B4FC", lw=0.6, alpha=0.35 + 0.3 * rng.random(), zorder=1)
for k, L in enumerate(pos):
    for (x, y) in L:
        ax.scatter([x], [y], s=210, color=["#F59E0B", "#06B6D4", "#06B6D4", "#10B981"][k], edgecolor="white", linewidth=1.4, zorder=2)
# curva de erro descendo (descida do gradiente)
t = np.linspace(0, 1, 200); xs_c = 90 + 620 * t; ys_c = 330 - 120 * (1 - np.exp(-5 * t)) * 0 - 110 * (1 - np.exp(-4 * t)) + 12 * np.sin(14 * t) * np.exp(-3 * t)
ax.plot(xs_c, ys_c, color="#FCD34D", lw=2.6, zorder=2, solid_capstyle="round")
ax.scatter([xs_c[0], xs_c[-1]], [ys_c[0], ys_c[-1]], s=60, color="#FCD34D", zorder=3)
# título e autor
ax.text(W / 2, 960, "BACKPROPAGATION", ha="center", va="center", color="white", fontsize=40, fontweight="bold")
ax.text(W / 2, 905, "E REDES NEURAIS", ha="center", va="center", color="white", fontsize=40, fontweight="bold")
ax.plot([180, W - 180], [868, 868], color="#F59E0B", lw=3)
ax.text(W / 2, 822, "Do perceptron ao Levenberg–Marquardt", ha="center", va="center", color="#E0E7FF", fontsize=19)
ax.text(W / 2, 790, "fundamentos matemáticos e implementações em Python", ha="center", va="center", color="#C7D2FE", fontsize=15, style="italic")
ax.text(W / 2, 150, "Araken de Medeiros Santos", ha="center", va="center", color="white", fontsize=27, fontweight="bold")
ax.text(W / 2, 90, "Perceptron · Backpropagation · Quickprop · RPROP · LM · Delta-Bar-Delta", ha="center", va="center", color="#A5B4FC", fontsize=11)
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figs", "capa.png"), dpi=100, facecolor="#1E1B4B"); print("ok")
