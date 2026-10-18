"""Figuras do Capítulo 2. f02_softmax, f02_entropia e f02_memoria são gráficos baseados em dados/cálculos; as demais, esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *


def sm(z, T=1.0):
    z = np.asarray(z, float) / T; z = z - z.max(); e = np.exp(z); return e / e.sum()


def mapa():
    fig, ax = canvas(9, 4.6)
    rbox(ax, 2, 62, 24, 28, "Vetores,\nmatrizes e\ntensores", fc="#E0E7FF", ec=IND, fs=10.5, bold=True)
    rbox(ax, 2, 14, 24, 28, "Python e NumPy\n(formas, memória,\nreprodutibilidade)", fc="#D1FAE5", ec=GRN, fs=9.5, bold=True)
    rbox(ax, 38, 62, 24, 28, "Softmax e\nprobabilidade\ncondicional", fc="#CFFAFE", ec=CYA, fs=10.5, bold=True)
    rbox(ax, 38, 14, 24, 28, "Entropia cruzada,\nKL e perplexidade", fc="#FEF3C7", ec=AMB, fs=10, bold=True)
    rbox(ax, 74, 38, 24, 28, "Gradiente e\ndescida do\ngradiente", fc="#FCE7F3", ec=PNK, fs=10.5, bold=True)
    arrow(ax, (26.5, 76), (37.5, 76), c=INK); arrow(ax, (50, 61.5), (50, 42.5), c=INK); arrow(ax, (62.5, 32), (73.5, 46), c=INK)
    arrow(ax, (26.5, 28), (37.5, 28), c=INK, ls="--"); arrow(ax, (86, 37.5), (50, 42.5), c=GRY, rad=-0.35, ls="--")
    ax.text(88, 20, "treinar =\nrepetir\no ciclo", ha="center", fontsize=9, color=INK, style="italic")
    save(fig, "f02_mapa")


def cosseno():
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    A = np.array([4.0, 1.0]); B = np.array([2.5, 3.2]); C = np.array([-1.0, 3.6])
    for v, c, n in [(A, IND, "a"), (B, GRN, "b"), (C, RED, "c")]:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5)); ax.text(v[0] * 1.05, v[1] * 1.08, n, color=c, fontsize=13, fontweight="bold")
    cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    t = np.linspace(np.arctan2(1, 4), np.arctan2(3.2, 2.5), 30); ax.plot(1.2 * np.cos(t), 1.2 * np.sin(t), c=GRY)
    ax.text(1.35, 1.05, f"cos(a,b) = {cos(A, B):.2f}", fontsize=9, color=INK)
    ax.text(-2.1, 0.5, f"cos(a,c) = {cos(A, C):.2f}", fontsize=9, color=INK)
    ax.set_xlim(-2.2, 5); ax.set_ylim(-0.3, 4.4); ax.set_aspect("equal"); ax.axhline(0, c=MID, lw=1); ax.axvline(0, c=MID, lw=1)
    ax.set_title("Ângulo pequeno = direção parecida"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "f02_cosseno")


def formas():
    fig, ax = canvas(8.4, 4.4, (0, 100), (0, 52))
    def cub(x0, y0, w, h, d, c):
        ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)], fc=c, ec=INK, alpha=0.9))
        ax.add_patch(Polygon([(x0, y0 + h), (x0 + w, y0 + h), (x0 + w + d, y0 + h + d * .6), (x0 + d, y0 + h + d * .6)], fc=c, ec=INK, alpha=0.6))
        ax.add_patch(Polygon([(x0 + w, y0), (x0 + w, y0 + h), (x0 + w + d, y0 + h + d * .6), (x0 + w + d, y0 + d * .6)], fc=c, ec=INK, alpha=0.4))
    for k in range(4):
        cub(8 + k * 3.2, 8 + k * 2, 30, 20, 3, [IND, CYA, GRN, AMB][k])
    ax.text(25, 4, "T tokens", ha="center", fontsize=10, fontweight="bold"); ax.text(4, 22, "d", ha="center", fontsize=10, fontweight="bold")
    ax.text(8, 44, "B sequências (lote)", fontsize=9.5, fontweight="bold")
    rbox(ax, 52, 24, 44, 22, "Tensor de formato (B, T, d)\nex.: B = 4, T = 7, d = 16\n→ 4 × 7 × 16 = 448 números", fc="#EEF2FF", ec=IND, fs=10, bold=True)
    rbox(ax, 52, 4, 44, 16, "Matriz de pesos W: (d, d')\nentrada d → saída d'", fc="#ECFEFF", ec=CYA, fs=10, bold=True)
    save(fig, "f02_formas")


def softmax():
    z = np.array([2.0, 1.0, 0.1]); fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
    for ax, T in zip(axs, (0.5, 1.0, 2.0)):
        p = sm(z, T); b = ax.bar(["A", "B", "C"], p, color=[IND, CYA, AMB], width=0.6)
        for r, v in zip(b, p): ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_title(f"T = {T}"); ax.set_ylim(0, 1.0)
    axs[0].set_ylabel("probabilidade"); fig.suptitle("Softmax dos logits (2,0; 1,0; 0,1) em três temperaturas", fontsize=10, fontweight="bold", y=1.04)
    save(fig, "f02_softmax")


def entropia():
    p = np.linspace(0.001, 0.999, 400); H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, ax = plt.subplots(figsize=(5.6, 3.4)); ax.plot(p, H, c=IND, lw=2.5); ax.scatter([0.5], [1], c=RED, zorder=3); ax.text(0.52, 0.97, "máximo: 1 bit", fontsize=9)
    ax.set_xlabel("probabilidade de cara, p"); ax.set_ylabel("entropia (bits)"); ax.set_title("Entropia de uma moeda"); ax.set_ylim(0, 1.1)
    save(fig, "f02_entropia")


def memoria():
    N = np.array([1e9, 7e9, 70e9]); nomes = ["1 bilhão", "7 bilhões", "70 bilhões"]; fmts = [("float32", 4, GRY), ("float16", 2, IND), ("int8", 1, CYA), ("4 bits", 0.5, GRN)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); w = 0.2
    for k, (n, b, c) in enumerate(fmts):
        v = N * b / 1e9; bars = ax.bar(np.arange(3) + (k - 1.5) * w, v, w, label=n, color=c)
        for r, x in zip(bars, v): ax.text(r.get_x() + r.get_width() / 2, x * 1.12, f"{x:g}", ha="center", fontsize=7.5)
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(["modelo de " + n for n in nomes]); ax.set_ylabel("GB só para os pesos (escala log)"); ax.legend(ncol=4, fontsize=8, loc="upper left"); ax.set_ylim(0.3, 800)
    ax.set_title("Memória mínima dos pesos (modelos hipotéticos)")
    save(fig, "f02_memoria")


if __name__ == "__main__":
    mapa(); cosseno(); formas(); softmax(); entropia(); memoria()
