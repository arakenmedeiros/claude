from fc import *
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelSpreading

PART = {}
for n in range(1, 19):
    PART[n] = (IND, CYA) if n <= 3 else (CYA, GRN) if n <= 8 else (GRN, AMB) if n <= 13 else (AMB, RED) if n <= 17 else (VIO, PNK)

def banner(n):
    c1, c2 = PART[n]; rng = np.random.default_rng(100 + n)
    fig, ax = plt.subplots(figsize=(10, 1.9)); fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY); ax.set_xlim(0, 10); ax.set_ylim(0, 1.9); ax.axis("off")
    g = np.linspace(0, 1, 256)[None, :]; cm = LinearSegmentedColormap.from_list("b", [NAVY, "#1E1B4B", NAVY]); ax.imshow(g, extent=[0, 10, 0, 1.9], aspect="auto", cmap=cm, zorder=0)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.9)
    # nuvem de pontos cinza + poucos pontos coloridos "rotulados" que irradiam cor
    centers = [(rng.uniform(4.2, 9.5), rng.uniform(0.4, 1.5)) for _ in range(4)]
    A = []
    for k, c in enumerate(centers):
        P = rng.normal(c, [0.40, 0.24], (46, 2)); P[:, 1] = np.clip(P[:, 1], 0.05, 1.85); A.append(P)
        col = c1 if k % 2 == 0 else c2
        ax.scatter(*P.T, s=rng.uniform(4, 16, 46), c="#9CA3AF", alpha=0.55, lw=0, zorder=2)
        near = np.argsort(((P - np.array(c)) ** 2).sum(1))[:14]
        ax.scatter(P[near, 0], P[near, 1], s=rng.uniform(8, 22, 14), c=col, alpha=0.85, lw=0, zorder=3)
        ax.scatter(*c, s=190, c=col, marker="*", edgecolors="white", lw=0.8, zorder=5)
        ax.scatter(*c, s=2200, c=col, alpha=0.07, lw=0, zorder=1)
    A = np.vstack(A)
    for i in range(0, len(A), 3):
        d = np.linalg.norm(A - A[i], axis=1); j = np.argsort(d)[1]; ax.plot([A[i, 0], A[j, 0]], [A[i, 1], A[j, 1]], c="white", alpha=0.10, lw=0.6, zorder=2)
    ax.text(0.35, 0.95, f"{n:02d}", fontsize=62, fontweight="bold", color="white", alpha=0.92, va="center", ha="left")
    ax.text(0.35, 0.12, "CAPÍTULO", fontsize=9, color=c2, fontweight="bold", va="bottom")
    fig.savefig(OUT + f"banner{n}.png", bbox_inches="tight", pad_inches=0, facecolor=NAVY, dpi=170); plt.close(fig); print("banner", n)

def cover():
    rng = np.random.RandomState(7); W, H = 6.0, 8.5
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); fig.patch.set_facecolor(NAVY)
    yy = np.linspace(0, 1, 512)[:, None]; cm = LinearSegmentedColormap.from_list("c", ["#0B1020", "#1E1B4B", "#312E81", "#0B1020"])
    ax.imshow(yy * np.ones((1, 2)), extent=[0, W, 0, H], aspect="auto", cmap=cm, zorder=0, origin="lower"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    # três grupos: duas luas e um bloco
    M, ym = make_moons(220, noise=0.07, random_state=3)
    M = M * np.array([1.55, 1.25]) + np.array([1.45, 5.3])
    B = rng.normal([4.45, 4.6], [0.55, 0.50], (80, 2))
    P = np.vstack([M, B]); lab = np.r_[ym, np.full(80, 2)]
    # poucos rótulos (estrelas): 1 em cada grupo
    ci = [int(np.argmin(((M[ym == 0] - np.array([0.9, 6.5])) ** 2).sum(1))), int(np.argmin(((M[ym == 1] - np.array([2.6, 5.0])) ** 2).sum(1)))]
    idx0 = np.where(ym == 0)[0][ci[0]]; idx1 = np.where(ym == 1)[0][ci[1]]; idx2 = 220 + int(np.argmin(((B - B.mean(0)) ** 2).sum(1)))
    rot = [idx0, idx1, idx2]; cols = [IND, AMB, GRN]
    y = np.full(len(P), -1); y[idx0] = 0; y[idx1] = 1; y[idx2] = 2
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.9, max_iter=2000).fit(P, y)
    D = ls.label_distributions_; conf = D.max(1); pred = D.argmax(1)
    from sklearn.neighbors import NearestNeighbors
    nb = NearestNeighbors(n_neighbors=4).fit(P).kneighbors(P)[1]
    for i in range(len(P)):
        for j in nb[i, 1:3]: ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c="white", alpha=0.13, lw=0.5, zorder=2)
    gray = np.array(to_rgb("#9CA3AF"))
    for i in range(len(P)):
        c = np.array(to_rgb(cols[pred[i]])); a = np.clip((conf[i] - 0.34) / 0.66, 0, 1) ** 0.6
        col = gray * (1 - a) + c * a
        ax.scatter(P[i, 0], P[i, 1], s=rng.uniform(10, 46), c=[col], alpha=0.95, lw=0, zorder=3)
    for r, c in zip(rot, cols):
        for rad, al in [(1.1, 0.05), (0.7, 0.07), (0.4, 0.10)]: ax.add_patch(Circle(P[r], rad, fc=c, alpha=al, lw=0, zorder=1))
        ax.scatter(*P[r], s=330, c=c, marker="*", edgecolors="white", lw=1.4, zorder=6)
    # pontos "sem rótulo" ao fundo (poeira)
    ax.scatter(rng.uniform(0.2, 5.8, 70), rng.uniform(3.3, 8.2, 70), s=rng.uniform(2, 9, 70), c="#9CA3AF", alpha=0.35, lw=0, zorder=2)
    for r in (0.9, 1.6, 2.3): ax.add_patch(Circle((3.0, 5.6), r * 2, fc="none", ec="white", alpha=0.04, lw=1))
    ax.text(0.45, 2.75, "APRENDIZADO", fontsize=33, fontweight="bold", color="white", va="center", zorder=6)
    ax.text(0.45, 2.12, "SEMISSUPERVISIONADO", fontsize=22.2, fontweight="bold", color=CYA, va="center", zorder=6)
    ax.plot([0.47, 2.2], [1.72, 1.72], c=AMB, lw=4, zorder=6)
    ax.text(0.47, 1.38, "Do zero à prática com Python e scikit-learn", fontsize=11.6, color="#E5E7EB", va="center", zorder=6, style="italic")
    ax.text(0.47, 0.55, "Araken de Medeiros Santos", fontsize=15, fontweight="bold", color="white", va="center", zorder=6)
    ax.text(W - 0.35, 8.15, "2026", fontsize=10, color=GRY, ha="right", va="center", zorder=6)
    fig.savefig(OUT + "capa.png", facecolor=NAVY, dpi=220); plt.close(fig); print("capa")

if __name__ == "__main__":
    cover()
    for n in range(1, 19): banner(n)
