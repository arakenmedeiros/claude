from fe import *
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Wedge
import matplotlib.colors as mc

W, H = 8.27, 11.69
fig = plt.figure(figsize=(W, H), dpi=250)
ax = fig.add_axes([0, 0, 1, 1])
X, Y = 100, 100 * H / W
ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.axis("off")

# fundo em degradê
g = np.linspace(0, 1, 256).reshape(-1, 1)
top = np.array(mc.to_rgb("#FFF6EA")); bot = np.array(mc.to_rgb("#F9D9BE"))
img = (top * (1 - g))[:, None, :] + (bot * g)[:, None, :]
img = np.repeat(img.reshape(256, 1, 3), 2, axis=1)
ax.imshow(img, extent=[0, X, 0, Y], aspect="auto", zorder=-10, origin="upper")

# sol e formas suaves
ax.add_patch(Circle((82, 70), 15, fc="#F6B94F", alpha=0.55, ec="none", zorder=-5))
ax.add_patch(Circle((82, 70), 9, fc="#F2A93B", alpha=0.6, ec="none", zorder=-5))
ax.add_patch(Circle((12, 85), 7, fc="#EE7B64", alpha=0.35, ec="none", zorder=-5))
ax.add_patch(Circle((24, 96), 3, fc="#2A9D8F", alpha=0.45, ec="none", zorder=-5))

# colinas
xs = np.linspace(0, X, 200)
for k, (col, base, amp, ph) in enumerate([("#F3A488", 38, 5, 0.0), ("#E88565", 30, 4, 1.3), ("#C9553A", 22, 3.5, 2.1)]):
    ys = base + amp * np.sin(xs / 14 + ph) + 2.2 * np.sin(xs / 5 + ph)
    ax.fill_between(xs, 0, ys, color=col, zorder=-3 + k, alpha=0.95)

# degraus (trajetória educacional)
niveis = ["Infantil", "Anos\niniciais", "Anos\nfinais", "Médio", "EPT", "Graduação", "Pós-\ngraduação"]
cores = ["#F6C46A", "#F2A93B", "#EE7B64", "#E0603E", "#C9553A", "#8E3B57", "#6D3B6E"]
n = 7; sw = 11.0; x0 = 13.5
tops = [19 + i * 5.6 for i in range(n)]
kinds = ["crianca", "crianca", "crianca", "jovem", "jovem", "adulto", "adulto"]
rng = np.random.default_rng(11)
for i in range(n):
    x = x0 + i * sw
    ax.add_patch(FancyBboxPatch((x, 8), sw - 0.4, tops[i] - 8, boxstyle="round,pad=0,rounding_size=1.2", fc=cores[i], ec="none", zorder=2))
    ax.text(x + (sw - 0.4) / 2, tops[i] - 3.4, niveis[i], ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=4, linespacing=1.05)
    for j in range(2):
        px = x + 2.9 + j * 5.0
        pessoa(ax, px, tops[i], 0.95, skin=int(rng.integers(0, 6)), hair=int(rng.integers(0, 6)),
               cloth=[TEAL, "#FFFFFF", AMBER, PLUM, BLUE, GREEN, ROSE, CORAL][(i * 2 + j) % 8], kind=kinds[i], glasses=(i >= 5 and j == 0))
# degradê de seta de continuidade
ax.annotate("", xy=(93, tops[-1] + 20), xytext=(8, tops[0] + 21), arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2, alpha=0.9, connectionstyle="arc3,rad=-0.28", linestyle=(0, (1, 2.2))), zorder=3)

# professores em primeiro plano
pessoa(ax, 6.5, 6.5, 1.4, skin=3, hair=4, cloth=TEAL, kind="adulto", glasses=True)
pessoa(ax, 95.0, 6.5, 1.4, skin=1, hair=0, cloth=PLUM, kind="adulto")

# rede de IA (pontos de luz conectados)
nodes = rng.uniform([8, 64], [92, 80], (16, 2))
for a in range(len(nodes)):
    for b in range(a + 1, len(nodes)):
        d = np.hypot(*(nodes[a] - nodes[b]))
        if d < 22:
            ax.plot(*zip(nodes[a], nodes[b]), c="#6D3B6E", lw=0.7, alpha=0.35, zorder=1)
for p in nodes:
    ax.add_patch(Circle(p, rng.uniform(0.5, 1.1), fc=rng.choice(["#6D3B6E", "#2A9D8F", "#F2A93B"]), ec="white", lw=0.5, zorder=2))

# balão de conversa, lâmpada e livro flutuantes
rbox(ax, 50, 66, 13, 8, "", fc="white", ec="#2A9D8F", lw=1.5, r=2.4, z=5)
for k in range(3): ax.add_patch(Circle((53.5 + k * 3, 70), 0.7, fc="#2A9D8F", zorder=6))
ax.add_patch(Polygon([[54, 66.2], [57, 66.2], [53, 63]], fc="white", ec="#2A9D8F", lw=1.2, zorder=5))
ax.add_patch(Circle((22, 56), 3, fc="#F2A93B", ec="white", lw=1, zorder=5)); ax.add_patch(Rectangle((20.9, 51.6), 2.2, 1.6, fc="#8E3B57", zorder=5))
for a in np.linspace(0, np.pi, 5): ax.plot([22 + 4 * np.cos(a), 22 + 5.3 * np.cos(a)], [56 + 4 * np.sin(a), 56 + 5.3 * np.sin(a)], c="#F2A93B", lw=1.2, zorder=4)
livro(ax, 30, 60, 8, 5.4, "#2A9D8F")

# título
ax.text(50, Y - 9, "INTELIGÊNCIA\nARTIFICIAL\nAPLICADA À EDUCAÇÃO", ha="center", va="top", fontsize=33, fontweight="bold", color="#4A2523", linespacing=1.12, zorder=10)
ax.plot([22, 78], [Y - 42, Y - 42], c="#F2A93B", lw=4, solid_capstyle="round", zorder=10)
ax.text(50, Y - 47.5, "da educação infantil à pós-graduação", ha="center", va="center", fontsize=17, color="#C9553A", fontweight="bold", style="italic", zorder=10)
ax.text(50, Y - 54, "Diretrizes, práticas, oportunidades,\nbenefícios e riscos", ha="center", va="center", fontsize=11.5, color="#5C4A47", linespacing=1.25, zorder=10)

# autor
ax.add_patch(Rectangle((0, 0), X, 5.2, fc="#4A2523", ec="none", zorder=8))
ax.text(50, 2.7, "ARAKEN DE MEDEIROS SANTOS", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#FFF6EA", zorder=9)
fig.savefig(OUT + "capa.png", dpi=250, facecolor="white")
print("capa ok")
