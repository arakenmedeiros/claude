"""Capa e faixas (banners) dos capítulos. Esquemas decorativos: não representam dados reais."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build")
from fc import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Circle

PARTCOL = {1: (IND, CYA), 2: (IND, CYA), 3: (IND, CYA), 4: (IND, CYA), 5: (CYA, GRN), 6: (CYA, GRN), 7: (CYA, GRN),
           8: (GRN, AMB), 9: (GRN, AMB), 10: (GRN, AMB), 11: (GRN, AMB), 12: (AMB, RED), 13: (AMB, RED), 14: (AMB, RED), 15: (AMB, RED),
           16: (RED, VIO), 17: (RED, VIO), 18: (VIO, PNK)}
TOK = ["Os", "grandes", "modelos", "de", "linguagem", "preveem", "o", "próximo", "token", "."]


def arcs(ax, xs, y, rng, cols, n=14, h=0.5, lw=1.0, alpha=0.35):
    for _ in range(n):
        i, j = sorted(rng.choice(len(xs), 2, replace=False))
        if i == j: continue
        t = np.linspace(0, np.pi, 40); cx = (xs[i] + xs[j]) / 2; r = abs(xs[j] - xs[i]) / 2
        ax.plot(cx + r * np.cos(t), y + h * (r / (xs[-1] - xs[0] + 1e-9) * 2.2) * np.sin(t), color=cols[rng.integers(len(cols))], alpha=alpha, lw=lw + rng.random())


def banner(n):
    c1, c2 = PARTCOL[n]; rng = np.random.default_rng(300 + n)
    fig, ax = plt.subplots(figsize=(10, 1.9)); fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.set_xlim(0, 10); ax.set_ylim(0, 1.9); ax.axis("off")
    g = np.linspace(0, 1, 256)[None, :]; cm = LinearSegmentedColormap.from_list("b", [NAVY, "#1E1B4B", NAVY]); ax.imshow(g, extent=[0, 10, 0, 1.9], aspect="auto", cmap=cm, zorder=0)
    xs = np.linspace(3.2, 9.6, 9) + rng.normal(0, 0.05, 9)
    for k, x in enumerate(xs):
        col = c1 if k % 2 == 0 else c2
        ax.add_patch(FancyBboxPatch((x - 0.27, 0.35), 0.54, 0.34, boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec="none", alpha=0.85, zorder=3))
    arcs(ax, xs, 0.72, rng, [c1, c2, "white"], n=16, h=0.55, alpha=0.4)
    ax.scatter(rng.normal(8.6, 0.6, 30), rng.normal(1.45, 0.18, 30), s=rng.uniform(3, 18, 30), c=c2, alpha=0.7, lw=0)
    ax.text(0.35, 0.95, f"{n:02d}", fontsize=62, fontweight="bold", color="white", alpha=0.92, va="center", ha="left")
    ax.text(0.35, 0.12, "CAPÍTULO", fontsize=9, color=c2, fontweight="bold", va="bottom")
    fig.savefig(OUT + f"banner{n}.png", bbox_inches="tight", pad_inches=0, facecolor=NAVY, dpi=170); plt.close(fig); print("banner", n)


def capa():
    rng = np.random.default_rng(11); W, H = 6.0, 8.5
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); fig.patch.set_facecolor(NAVY)
    yy = np.linspace(0, 1, 512)[:, None]; cm = LinearSegmentedColormap.from_list("c", ["#0B1020", "#1E1B4B", "#312E81", "#0B1020"])
    ax.imshow(yy * np.ones((1, 2)), extent=[0, W, 0, H], aspect="auto", cmap=cm, zorder=0, origin="lower")
    # nuvem de embeddings (decorativa)
    cen = [(1.5, 6.9, IND), (4.3, 7.0, CYA), (2.9, 6.1, PNK), (4.8, 5.9, GRN), (1.1, 5.7, AMB)]
    allp = []
    for (x, y, c) in cen:
        P = rng.normal([x, y], [0.42, 0.34], (55, 2)); allp.append(P)
        ax.scatter(*P.T, s=rng.uniform(5, 45, 55), c=c, alpha=0.9, lw=0, zorder=3)
        for r, a in [(0.9, 0.05), (0.6, 0.07), (0.35, 0.1)]: ax.add_patch(Circle((x, y), r, fc=c, alpha=a, lw=0, zorder=1))
    A = np.vstack(allp)
    from sklearn.neighbors import NearestNeighbors
    nb = NearestNeighbors(n_neighbors=4).fit(A).kneighbors(A)[1]
    for i in range(len(A)):
        for j in nb[i, 1:3]: ax.plot([A[i, 0], A[j, 0]], [A[i, 1], A[j, 1]], c="white", alpha=0.12, lw=0.5, zorder=2)
    # faixa de tokens com arcos de atenção
    xs = np.linspace(0.75, 5.25, 8); labs = ["Os", "modelos", "de", "linguagem", "preveem", "o", "próximo", "token"]
    cols = [IND, CYA, AMB, GRN, PNK, VIO, CYA, IND]
    for x, l, c in zip(xs, labs, cols):
        w = 0.1 + 0.075 * len(l)
        ax.add_patch(FancyBboxPatch((x - w / 2, 4.35), w, 0.3, boxstyle="round,pad=0,rounding_size=0.07", fc=c, ec="none", alpha=0.92, zorder=4))
        ax.text(x, 4.5, l, ha="center", va="center", fontsize=6.3, color="white", fontweight="bold", zorder=5)
    arcs(ax, xs, 4.68, rng, [CYA, PNK, AMB, "white"], n=13, h=0.6, lw=0.8, alpha=0.5)
    ax.text(0.45, 3.25, "GRANDES", fontsize=36, fontweight="bold", color="white", va="center", zorder=6)
    ax.text(0.45, 2.55, "MODELOS DE", fontsize=36, fontweight="bold", color="white", va="center", zorder=6)
    ax.text(0.45, 1.85, "LINGUAGEM", fontsize=36, fontweight="bold", color=CYA, va="center", zorder=6)
    ax.plot([0.47, 2.2], [1.45, 1.45], c=AMB, lw=4, zorder=6)
    ax.text(0.47, 1.12, "Dos fundamentos aos sistemas, com Python", fontsize=11.5, color="#E5E7EB", va="center", zorder=6, style="italic")
    ax.text(0.47, 0.45, "Araken de Medeiros Santos", fontsize=15, fontweight="bold", color="white", va="center", zorder=6)
    ax.text(W - 0.35, 8.15, "2026", fontsize=10, color=GRY, ha="right", va="center", zorder=6)
    fig.savefig(OUT + "capa.png", facecolor=NAVY, dpi=220); plt.close(fig); print("capa")


if __name__ == "__main__":
    capa()
    for n in range(1, 19): banner(n)
