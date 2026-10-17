"""Biblioteca de figuras do livro 'IA Aplicada à Educação'. Paleta quente e suave, diagramas conceituais."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import textwrap, os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle, Polygon, Wedge, Arc
import warnings
warnings.filterwarnings("ignore")

TERRA = "#D9653B"; CORAL = "#EE7B64"; AMBER = "#F2A93B"; GOLD = "#C98A10"; TEAL = "#2A9D8F"; PLUM = "#6D3B6E"
BLUE = "#3D7EA6"; GREEN = "#6E9A4F"; ROSE = "#D95D7F"; INK = "#3B2A2A"; GRY = "#8F847E"
SAND = "#F6E7D3"; CREAM = "#FFF8EF"; LIGHT = "#F7EFE6"; MID = "#E6D8C8"; WHITE = "#FFFFFF"
PART = {1: TERRA, 2: TEAL, 3: GOLD, 4: PLUM, 5: BLUE, 6: ROSE, 7: GREEN}
CL = [TERRA, TEAL, AMBER, PLUM, BLUE, GREEN, ROSE, CORAL]
SKIN = ["#F1C9A5", "#E0A97D", "#C68A5E", "#9B6440", "#6F4529", "#F6D7BD"]
HAIR = ["#2B1D14", "#5A3825", "#8A5A2B", "#C9A15B", "#1C1C1C", "#7A7A7A"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs") + os.sep

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 100, "savefig.dpi": 200, "axes.grid": False,
})
MAXW = 8.6


def save(fig, name):
    w, h = fig.get_size_inches()
    if w > MAXW:
        fig.set_size_inches(MAXW, h * MAXW / w)
    fig.savefig(OUT + name + ".png", bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("fig", name, flush=True)


def canvas(w, h, xl=(0, 100), yl=(0, 100), bg=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.axis("off")
    if bg:
        ax.add_patch(Rectangle((xl[0], yl[0]), xl[1] - xl[0], yl[1] - yl[0], fc=bg, ec="none", zorder=-5))
    return fig, ax


def wrap(t, n):
    return "\n".join(textwrap.fill(p, n) for p in t.split("\n"))


def rbox(ax, x, y, w, h, text="", fc=LIGHT, ec=None, fs=9, tc=INK, bold=False, r=1.5, lw=1.2, ha="center", wrap_at=None, z=2, alpha=1, va="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)
    if text:
        if wrap_at:
            text = wrap(text, wrap_at)
        tx = x + (w / 2 if ha == "center" else 1.5)
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.22, zorder=z + 1)
    return p


def arrow(ax, p1, p2, c=INK, lw=1.6, style="-|>", rad=0.0, ms=12, ls="-", z=1):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, color=c, lw=lw, connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=z)
    ax.add_patch(a); return a


def label(ax, x, y, t, fs=8.5, c=INK, bold=False, ha="center", va="center", it=False, rot=0, wrap_at=None):
    if wrap_at: t = wrap(t, wrap_at)
    ax.text(x, y, t, fontsize=fs, color=c, ha=ha, va=va, fontweight="bold" if bold else "normal", style="italic" if it else "normal", rotation=rot, linespacing=1.22, zorder=6)


def pill(ax, x, y, text, fc=TERRA, tc="white", fs=8, w=None, h=4.2, bold=True):
    w = w or (len(text) * 1.15 + 4)
    rbox(ax, x - w / 2, y - h / 2, w, h, text, fc=fc, tc=tc, fs=fs, bold=bold, r=h / 2, z=4)


# ------------------------------------------------------------------ pessoas (ícones simples e diversos)
def pessoa(ax, x, y, s=1.0, skin=0, hair=0, cloth=TERRA, kind="adulto", glasses=False, arm=None):
    """Desenha uma pessoa estilizada (cabeça + tronco). (x,y) = base do tronco. kind: crianca, jovem, adulto, idoso."""
    k = {"crianca": 0.62, "jovem": 0.85, "adulto": 1.0, "idoso": 0.95}[kind]
    sc = s * k
    sk = SKIN[skin % len(SKIN)]; hr = HAIR[hair % len(HAIR)]
    body_h = 7.0 * sc; body_w = 5.6 * sc; head_r = 2.5 * sc
    ax.add_patch(FancyBboxPatch((x - body_w / 2, y), body_w, body_h, boxstyle=f"round,pad=0,rounding_size={body_w / 2.2}", fc=cloth, ec="none", zorder=4))
    hy = y + body_h + head_r * 0.85
    ax.add_patch(Circle((x, hy), head_r, fc=sk, ec="none", zorder=5))
    if kind == "idoso":
        ax.add_patch(Wedge((x, hy + head_r * 0.15), head_r * 1.02, 20, 160, width=head_r * 0.45, fc="#CFCFCF", ec="none", zorder=6))
    else:
        ax.add_patch(Wedge((x, hy + head_r * 0.1), head_r * 1.06, 15, 165, width=head_r * 0.7, fc=hr, ec="none", zorder=6))
    if glasses:
        for dx in (-0.8, 0.8):
            ax.add_patch(Circle((x + dx * sc, hy), 0.75 * sc, fc="none", ec=INK, lw=0.8, zorder=7))
    ax.add_patch(Circle((x - 0.8 * sc, hy), 0.18 * sc, fc=INK, ec="none", zorder=7))
    ax.add_patch(Circle((x + 0.8 * sc, hy), 0.18 * sc, fc=INK, ec="none", zorder=7))
    ax.add_patch(Arc((x, hy - 0.5 * sc), 1.6 * sc, 1.0 * sc, theta1=200, theta2=340, ec=INK, lw=0.8, zorder=7))
    return hy + head_r


def grupo(ax, x0, y0, n, s=1.0, gap=7.5, kinds=None, clothes=None, seed=0):
    rng = np.random.default_rng(seed)
    kinds = kinds or ["crianca", "jovem", "adulto", "idoso", "jovem", "crianca"]
    clothes = clothes or CL
    for i in range(n):
        pessoa(ax, x0 + i * gap, y0, s, skin=int(rng.integers(0, 6)), hair=int(rng.integers(0, 6)), cloth=clothes[i % len(clothes)], kind=kinds[i % len(kinds)], glasses=bool(rng.integers(0, 2)) and kinds[i % len(kinds)] in ("adulto", "idoso"))


# ------------------------------------------------------------------ objetos
def livro(ax, x, y, w=6, h=4.5, c=TERRA):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.5", fc=c, ec="none", zorder=3))
    ax.add_patch(Rectangle((x + w * 0.5 - 0.15, y), 0.3, h, fc="white", alpha=0.7, zorder=4))
    for i in range(3):
        ax.plot([x + 0.7, x + w * 0.5 - 0.7], [y + h * (0.7 - i * 0.2)] * 2, c="white", lw=0.8, alpha=0.8, zorder=4)
        ax.plot([x + w * 0.5 + 0.7, x + w - 0.7], [y + h * (0.7 - i * 0.2)] * 2, c="white", lw=0.8, alpha=0.8, zorder=4)


def lousa(ax, x, y, w=24, h=14, c=TEAL, txt=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1", fc=c, ec="#8B5E3C", lw=3, zorder=3))
    if txt: ax.text(x + w / 2, y + h / 2, wrap(txt, 22), ha="center", va="center", color="white", fontsize=8, zorder=5, linespacing=1.2)


def robo(ax, x, y, s=1.0, c=BLUE):
    ax.add_patch(FancyBboxPatch((x - 3 * s, y), 6 * s, 5 * s, boxstyle="round,pad=0,rounding_size=1.2", fc=c, ec="none", zorder=4))
    ax.add_patch(FancyBboxPatch((x - 3.6 * s, y + 5.6 * s), 7.2 * s, 5.6 * s, boxstyle="round,pad=0,rounding_size=1.4", fc=c, ec="none", zorder=4))
    for dx in (-1.6, 1.6):
        ax.add_patch(Circle((x + dx * s, y + 8.4 * s), 0.9 * s, fc="white", ec="none", zorder=5))
        ax.add_patch(Circle((x + dx * s, y + 8.4 * s), 0.4 * s, fc=INK, ec="none", zorder=6))
    ax.plot([x, x], [y + 11.2 * s, y + 13 * s], c=c, lw=2 * s, zorder=4)
    ax.add_patch(Circle((x, y + 13.3 * s), 0.8 * s, fc=AMBER, ec="none", zorder=4))
    ax.plot([x - 1.5 * s, x + 1.5 * s], [y + 6.2 * s, y + 6.2 * s], c="white", lw=1.4 * s, zorder=5)


def balao(ax, x, y, w, h, text, fc=WHITE, ec=TERRA, fs=7.5, tail=(0, -1), wrap_at=26):
    rbox(ax, x, y, w, h, text, fc=fc, ec=ec, fs=fs, r=2, wrap_at=wrap_at, lw=1.4, z=6)
    cx = x + w / 2 + tail[0] * w * 0.28
    ty = y if tail[1] < 0 else y + h
    ax.add_patch(Polygon([[cx - 1.2, ty], [cx + 1.2, ty], [cx + tail[0] * 2.5, ty + tail[1] * 3]], fc=fc, ec=ec, lw=1.4, zorder=5))


# ------------------------------------------------------------------ diagramas genéricos
def mapa_conceitos(name, centro, ramos, w=8.4, h=5.4, cor=TERRA, fs=8.2, centro_w=26, titulo=None):
    """ramos: lista de (rótulo, [subitens]). Mapa radial com até 8 ramos."""
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    cx, cy = 50, H / 2
    n = len(ramos)
    rbox(ax, cx - centro_w / 2, cy - 7, centro_w, 14, centro, fc=cor, tc="white", bold=True, fs=fs + 1.3, r=4, wrap_at=int(centro_w * 0.75), z=5)
    for i, (rot, subs) in enumerate(ramos):
        ang = np.pi / 2 - 2 * np.pi * (i + 0.5) / n
        rx, ry = 36 * np.cos(ang), (H / 2 - 12) * np.sin(ang)
        bx, by = cx + rx, cy + ry
        c = CL[i % len(CL)]
        arrow(ax, (cx + np.cos(ang) * centro_w * 0.42, cy + np.sin(ang) * 7.5), (bx - np.cos(ang) * 6, by - np.sin(ang) * 3), c, lw=1.8, style="-", z=1)
        bw = 24
        rbox(ax, bx - bw / 2, by - 4.2, bw, 8.4, rot, fc=c, tc="white", bold=True, fs=fs, r=2.6, wrap_at=22, z=4)
        for j, sbi in enumerate(subs[:3]):
            dy = -7.2 - 4.6 * j if ry < 0 or abs(ry) < 3 else 7.2 + 4.6 * j
            if np.sin(ang) > 0.2: dy = 6.8 + 4.4 * j
            elif np.sin(ang) < -0.2: dy = -6.8 - 4.4 * j
            else: dy = -6.8 - 4.4 * j
            ax.text(bx, by + dy, "• " + wrap(sbi, 30).replace("\n", "\n  "), ha="center", va="center", fontsize=fs - 1.2, color=INK, zorder=6, linespacing=1.1)
    save(fig, name)


def fluxo(name, passos, w=8.4, cor=TERRA, h=None, fs=8.2, wrap_at=18, notas=None, por_linha=None):
    """Fluxo em sequência com setas. passos: lista de (título, descrição). Quebra em linhas."""
    n = len(passos)
    por = por_linha or (n if n <= 5 else int(np.ceil(n / 2)))
    linhas = int(np.ceil(n / por))
    h = h or (2.6 + 3.2 * linhas)
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    bw = (96 - (por - 1) * 5) / por
    bh = (H - 6) / linhas - 6
    for i, (t, d) in enumerate(passos):
        r, c = divmod(i, por)
        x = 2 + c * (bw + 5)
        y = H - 3 - (r + 1) * (bh + 6) + 3
        col = CL[i % len(CL)] if cor == "multi" else cor
        rbox(ax, x, y, bw, bh, "", fc=WHITE, ec=col, lw=1.8, r=2.5, z=2)
        rbox(ax, x, y + bh - 6.2, bw, 6.2, f"{i + 1}. {t}", fc=col, tc="white", bold=True, fs=fs, r=2.5, wrap_at=wrap_at + 4, z=3)
        ax.text(x + bw / 2, y + (bh - 6.2) / 2, wrap(d, wrap_at), ha="center", va="center", fontsize=fs - 1, zorder=4, linespacing=1.18)
        if c < por - 1 and i < n - 1:
            arrow(ax, (x + bw + 0.3, y + bh / 2), (x + bw + 4.7, y + bh / 2), INK, lw=1.6)
        if c == por - 1 and i < n - 1:
            arrow(ax, (x + bw / 2, y - 0.3), (2 + bw / 2, y - 5.7), INK, lw=1.4, rad=-0.0, ls="--") if False else None
    save(fig, name)


def ciclo(name, etapas, centro="", w=7.2, h=5.6, cor=TERRA, fs=8.2, wrap_at=16):
    fig, ax = canvas(w, h, (-50, 50), (-50 * h / w, 50 * h / w))
    n = len(etapas)
    R = 31
    if centro:
        ax.add_patch(Circle((0, 0), 13, fc=cor, ec="none", zorder=2))
        ax.text(0, 0, wrap(centro, 14), ha="center", va="center", color="white", fontweight="bold", fontsize=fs + 0.5, zorder=3, linespacing=1.15)
    pts = []
    for i, e in enumerate(etapas):
        a = np.pi / 2 - 2 * np.pi * i / n
        pts.append((R * 1.14 * np.cos(a), R * 0.78 * np.sin(a)))
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        arrow(ax, p, q, GRY, lw=1.6, rad=-0.28, ms=12, z=1)
    for i, (e, p) in enumerate(zip(etapas, pts)):
        c = CL[i % len(CL)]
        t, d = e if isinstance(e, tuple) else (e, "")
        rbox(ax, p[0] - 11.5, p[1] - 5.2, 23, 10.4, "", fc=WHITE, ec=c, lw=2, r=3, z=3)
        ax.text(p[0], p[1] + (2.0 if d else 0), wrap(t, wrap_at), ha="center", va="center", fontsize=fs, fontweight="bold", color=c, zorder=5, linespacing=1.1)
        if d: ax.text(p[0], p[1] - 2.6, wrap(d, 22), ha="center", va="center", fontsize=fs - 1.6, zorder=5, linespacing=1.1)
    save(fig, name)


def camadas(name, itens, w=7.6, h=5.0, titulo_esq=None, base_larga=True, fs=8.5):
    """Pirâmide/escada de camadas. itens: lista de (título, descrição) de baixo para cima."""
    n = len(itens)
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    lh = (H - 6) / n
    for i, (t, d) in enumerate(itens):
        wid = 90 - i * (60 / max(1, n - 1)) if base_larga else 30 + i * (60 / max(1, n - 1))
        x = 50 - wid / 2
        y = 3 + i * lh
        rbox(ax, x, y, wid, lh - 1.2, "", fc=CL[i % len(CL)], r=1.8, z=2)
        ax.text(50, y + (lh - 1.2) * 0.64, t, ha="center", va="center", fontsize=fs, fontweight="bold", color="white", zorder=4)
        ax.text(50, y + (lh - 1.2) * 0.28, wrap(d, int(wid * 0.9)), ha="center", va="center", fontsize=fs - 1.7, color="white", zorder=4, linespacing=1.1)
    save(fig, name)


def espectro(name, esq, dir_, itens, w=8.4, h=3.0, fs=8.2, cor_esq=TEAL, cor_dir=TERRA):
    """Linha-espectro com itens posicionados em [0,1]: (pos, rótulo)."""
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    y0 = H * 0.42
    n = 100
    for i in range(n):
        t = i / (n - 1)
        c = np.array(matplotlib.colors.to_rgb(cor_esq)) * (1 - t) + np.array(matplotlib.colors.to_rgb(cor_dir)) * t
        ax.add_patch(Rectangle((4 + i * 0.92, y0 - 1.5), 0.93, 3, fc=c, ec="none"))
    label(ax, 4, y0 + 7, esq, fs, cor_esq, True, ha="left"); label(ax, 96, y0 + 7, dir_, fs, cor_dir, True, ha="right")
    for k, (p, t) in enumerate(itens):
        x = 4 + p * 92
        up = k % 2 == 0
        ax.plot([x, x], [y0, y0 + (-9 if up else 9) * -1], c=GRY, lw=1)
        ax.add_patch(Circle((x, y0), 1.6, fc=WHITE, ec=INK, lw=1.4, zorder=4))
        ty = y0 - 8.5 if up else y0 + 8.5
        ax.plot([x, x], [y0, ty + (2.6 if up else -2.6)], c=GRY, lw=1, zorder=1)
        label(ax, x, ty, t, fs - 0.8, INK, wrap_at=17)
    save(fig, name)


def comparar_cartoes(name, cartoes, w=8.4, h=4.6, fs=8.2, wrap_at=24, colunas=None):
    """Cartões lado a lado: cartoes = [(título, [linhas], cor)]."""
    n = len(cartoes)
    cols = colunas or n
    linhas = int(np.ceil(n / cols))
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    cw = (96 - (cols - 1) * 3) / cols
    ch = (H - 4 - (linhas - 1) * 3) / linhas
    for i, (t, ls, c) in enumerate(cartoes):
        r, k = divmod(i, cols)
        x = 2 + k * (cw + 3); y = H - 2 - (r + 1) * ch - r * 3
        rbox(ax, x, y, cw, ch, "", fc=WHITE, ec=c, lw=2, r=2.5, z=2)
        rbox(ax, x, y + ch - 7.5, cw, 7.5, t, fc=c, tc="white", bold=True, fs=fs + 0.3, r=2.5, wrap_at=int(cw * 0.9), z=3)
        yy = y + ch - 11.0
        for l in ls:
            tt = wrap("• " + l, wrap_at).replace("\n", "\n  ")
            ax.text(x + 2.0, yy, tt, ha="left", va="top", fontsize=fs - 1.1, zorder=4, linespacing=1.15)
            yy -= (tt.count("\n") + 1) * 2.8 + 1.4
    save(fig, name)


def venn2(name, a, b, inter, w=7.0, h=4.2, ca=TERRA, cb=TEAL, fs=8.5, wrap_at=18):
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    r = H * 0.44
    ax.add_patch(Circle((36, H / 2), r, fc=ca, alpha=0.28, ec=ca, lw=2))
    ax.add_patch(Circle((64, H / 2), r, fc=cb, alpha=0.28, ec=cb, lw=2))
    ax.text(25, H / 2, wrap(a, wrap_at), ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)
    ax.text(75, H / 2, wrap(b, wrap_at), ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)
    ax.text(50, H / 2, wrap(inter, 13), ha="center", va="center", fontsize=fs, fontweight="bold", color=INK, linespacing=1.15)
    save(fig, name)


def semaforo(name, linhas, w=8.4, fs=8.2, wrap_at=48):
    """Classificação em verde/amarelo/vermelho. linhas = [(cor('v','a','r'), título, texto)]."""
    h = 1.6 + 1.0 * len(linhas) * 1.05
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    cmap = {"v": (GREEN, "#EAF3E3"), "a": (AMBER, "#FDF1D8"), "r": (ROSE, "#FADDE5")}
    lh = (H - 3) / len(linhas)
    for i, (k, t, d) in enumerate(linhas):
        c, f = cmap[k]
        y = H - 1.5 - (i + 1) * lh
        rbox(ax, 2, y + 0.6, 96, lh - 1.2, "", fc=f, ec=c, lw=1.5, r=2, z=2)
        ax.add_patch(Circle((8, y + lh / 2), min(3.6, lh * 0.32), fc=c, ec="none", zorder=4))
        ax.text(14, y + lh / 2 + (1.1 if d else 0), t, ha="left", va="center", fontsize=fs + 0.3, fontweight="bold", color=INK, zorder=5)
        if d: ax.text(14, y + lh / 2 - 2.0, wrap(d, wrap_at), ha="left", va="center", fontsize=fs - 1.4, zorder=5, linespacing=1.1)
    save(fig, name)


def decisao(name, nos, ligacoes, w=8.4, h=6.0, fs=7.8):
    """Fluxograma de decisão. nos: {id: (x,y,w,h,texto,tipo)} tipo: 'q' (pergunta), 'a' (ação), 'f' (fim bom), 'x' (fim: dispensar). ligacoes: (de, para, rótulo, rad)"""
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    sty = {"q": (SAND, TERRA, INK), "a": (WHITE, TEAL, INK), "f": (GREEN, GREEN, "white"), "x": (ROSE, ROSE, "white"), "i": (TERRA, TERRA, "white")}
    for k, (x, y, bw, bh, t, tp) in nos.items():
        fc, ec, tc = sty[tp]
        if tp == "q":
            ax.add_patch(Polygon([[x, y + bh / 2], [x + bw / 2, y + bh], [x + bw, y + bh / 2], [x + bw / 2, y]], fc=fc, ec=ec, lw=1.8, zorder=3))
            ax.text(x + bw / 2, y + bh / 2, wrap(t, int(bw * 0.42)), ha="center", va="center", fontsize=fs - 0.6, zorder=5, linespacing=1.1, color=tc)
        else:
            rbox(ax, x, y, bw, bh, t, fc=fc, ec=ec, tc=tc, fs=fs, r=2, wrap_at=int(bw * 0.62), lw=1.8, z=3, bold=(tp in ("f", "x", "i")))
    def anchor(k, side):
        x, y, bw, bh, _, _ = nos[k]
        return {"t": (x + bw / 2, y + bh), "b": (x + bw / 2, y), "l": (x, y + bh / 2), "r": (x + bw, y + bh / 2)}[side]
    for (a, sa, b, sb, lab, rad) in ligacoes:
        p, q = anchor(a, sa), anchor(b, sb)
        arrow(ax, p, q, INK, lw=1.4, rad=rad, ms=11, z=1)
        if lab:
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            ax.text(mx, my + 1.2, lab, fontsize=fs - 0.6, fontweight="bold", color=TERRA, ha="center", va="bottom", zorder=7, bbox=dict(fc="white", ec="none", pad=0.6))
    save(fig, name)


def barras_conceituais(name, itens, w=7.6, h=3.6, titulo="", rotulo_y="", cor=TERRA, fs=8.2):
    """Barras ILUSTRATIVAS (não representam dados). itens: [(rótulo, altura 0-1)]"""
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    n = len(itens)
    bw = 80 / n * 0.62
    for i, (t, v) in enumerate(itens):
        x = 14 + i * (80 / n) + (80 / n - bw) / 2
        ax.add_patch(FancyBboxPatch((x, 12), bw, (H - 26) * v, boxstyle="round,pad=0,rounding_size=1.2", fc=CL[i % len(CL)], ec="none", zorder=3))
        ax.text(x + bw / 2, 8.5, wrap(t, 14), ha="center", va="center", fontsize=fs - 0.8, linespacing=1.1)
    ax.plot([12, 12], [12, H - 8], c=INK, lw=1.2); ax.plot([12, 97], [12, 12], c=INK, lw=1.2)
    ax.text(5, H / 2 + 4, rotulo_y, rotation=90, ha="center", va="center", fontsize=fs, color=GRY)
    ax.text(55, H - 3, "Ilustração conceitual — as alturas não representam dados", ha="center", fontsize=fs - 1.2, color=GRY, style="italic")
    save(fig, name)


def faixa(name, texto, sub="", cor=TERRA, w=8.4, h=1.1):
    """Banner de capítulo."""
    fig, ax = canvas(w, h, (0, 100), (0, 100 * h / w))
    H = 100 * h / w
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    ax.add_patch(Rectangle((0, 0), 100, H, fc=cor, ec="none"))
    for i in range(14):
        ax.add_patch(Circle((rng.uniform(40, 105), rng.uniform(-2, H + 2)), rng.uniform(2.5, 9), fc="white", alpha=rng.uniform(0.06, 0.17), ec="none"))
    ax.text(3, H * 0.52, texto, ha="left", va="center", fontsize=17, fontweight="bold", color="white")
    if sub: ax.text(97, H * 0.5, sub, ha="right", va="center", fontsize=8.5, color="white", style="italic")
    fig.savefig(OUT + name + ".png", bbox_inches="tight", pad_inches=0, facecolor=cor)
    plt.close(fig)
