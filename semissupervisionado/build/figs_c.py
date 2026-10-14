from fc import *
from ch3_exp import duas_luas, grupos_cruzados, comparar
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC

def f3_indtrans():
    fig, ax = canvas(9.0, 4.4, (0, 100), (0, 52))
    ax.text(25, 49, "INDUTIVO", ha="center", fontsize=11, fontweight="bold", color=IND)
    ax.text(75, 49, "TRANSDUTIVO", ha="center", fontsize=11, fontweight="bold", color=RED)
    ax.plot([50, 50], [4, 47], c=MID, lw=2, ls="--")
    rng = np.random.default_rng(5)
    for off, col in [(0, IND), (50, RED)]:
        rbox(ax, off + 3, 30, 19, 14, "Treino\n(rotulados +\nsem rótulo)", fc="white", ec=col, fs=7.8, r=1.5, lw=1.8)
        arrow(ax, (off + 22.2, 37), (off + 27.5, 37), INK, lw=1.4, ms=9)
    rbox(ax, 28, 30, 14, 14, "Modelo\ngeral", fc=IND, tc="white", bold=True, fs=8.3, r=1.5)
    arrow(ax, (35, 29.8), (35, 22.5), INK, lw=1.4, ms=9)
    rbox(ax, 26, 12, 18, 10, "Exemplos NOVOS\n(nunca vistos)", fc="#EEF2FF", ec=IND, fs=7.6, r=1.3)
    ax.text(25, 6.5, "O modelo vale para qualquer\nexemplo futuro.", ha="center", fontsize=8, style="italic")
    rbox(ax, 78, 30, 20, 14, "Rótulos\npara o próprio\nconjunto sem rótulo", fc=RED, tc="white", bold=True, fs=7.6, r=1.5)
    arrow(ax, (88, 29.8), (88, 22.5), INK, lw=1.4, ms=9)
    rbox(ax, 76, 12, 24, 10, "Os mesmos exemplos\nsem rótulo, agora\nrotulados", fc="#FEF2F2", ec=RED, fs=7.4, r=1.3)
    ax.text(75, 6.5, "Não há promessa para exemplos\nque ainda não existem.", ha="center", fontsize=8, style="italic")
    save(fig, "f3_indtrans")

def f3_hipoteses():
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.7))
    rng = np.random.default_rng(1)
    # 1 suavidade
    ax = axs[0]
    x = np.linspace(0, 10, 300); f = np.sin(x * 0.7) + 0.2 * x
    ax.plot(x, f, c=GRY, lw=2)
    xs = np.sort(rng.uniform(0.3, 9.7, 24)); ax.scatter(xs, np.sin(xs * 0.7) + 0.2 * xs + rng.normal(0, 0.05, 24), c=GRY, s=14, zorder=3)
    for xp in [1.5, 4.8, 8.4]:
        ax.scatter(xp, np.sin(xp * 0.7) + 0.2 * xp, c=[IND, AMB, GRN][[1.5, 4.8, 8.4].index(xp)], s=110, edgecolors="black", zorder=4, marker="*")
    ax.annotate("vizinhos próximos,\nvalores próximos", xy=(4.9, 0.95), xytext=(5.6, 2.4), fontsize=8, arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_title("1) Suavidade\nperto ⇒ mesmo rótulo"); ax.set_xlabel("x"); ax.set_ylabel("rótulo"); ax.set_yticks([])
    # 2 agrupamento
    ax = axs[1]
    P1 = rng.normal([0, 0], 0.7, (60, 2)); P2 = rng.normal([4, 1.5], 0.7, (60, 2))
    ax.scatter(*P1.T, c=GRY, s=16, edgecolors="white", lw=0.3); ax.scatter(*P2.T, c=GRY, s=16, edgecolors="white", lw=0.3)
    ax.scatter(*P1[0], c=IND, s=140, marker="*", edgecolors="black", zorder=4); ax.scatter(*P2[0], c=AMB, s=140, marker="*", edgecolors="black", zorder=4)
    ax.plot([2, 2.1], [-2, 3.5], c=RED, lw=2.2, ls="--"); ax.text(2.25, -2.2, "fronteira onde\nhá poucos pontos", color=RED, fontsize=8, ha="left")
    ax.add_patch(Ellipse((0, 0), 3.6, 3.6, fc="none", ec=IND, lw=1.4, ls=":")); ax.add_patch(Ellipse((4, 1.5), 3.6, 3.6, fc="none", ec=AMB, lw=1.4, ls=":"))
    ax.set_title("2) Agrupamento (baixa densidade)\nmesmo grupo ⇒ mesmo rótulo"); ax.set_xlim(-2.5, 6.5); ax.set_ylim(-2.5, 4.2); clean(ax, True)
    # 3 variedade
    ax = axs[2]
    t = np.sort(rng.uniform(0.3, 3.8, 160)); R = 0.6 + 0.9 * t
    sp = np.c_[R * np.cos(2.2 * t), R * np.sin(2.2 * t)] + rng.normal(0, 0.05, (160, 2))
    ax.scatter(*sp.T, c=GRY, s=12, edgecolors="white", lw=0.3)
    a, b = sp[3], sp[-5]
    ax.scatter(*a, c=IND, s=140, marker="*", edgecolors="black", zorder=4); ax.scatter(*b, c=AMB, s=140, marker="*", edgecolors="black", zorder=4)
    ax.plot(*zip(a, b), c=RED, lw=2, ls="--"); ax.text(0.0, -0.25, "atalho reto\n(engana)", color=RED, fontsize=8, ha="center")
    ax.plot(sp[:, 0], sp[:, 1], c=GRN, lw=1.2, alpha=0.6)
    ax.text(-3.2, 2.4, "caminho pela curva\n(a variedade)", color=GRN, fontsize=8, fontweight="bold")
    ax.set_title("3) Variedade (manifold)\no que importa é a forma dos dados"); ax.set_aspect("equal"); clean(ax, True)
    save(fig, "f3_hipoteses")

def f3_cenarios():
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    X, y = duas_luas(1)
    ax = axs[0]; ax.scatter(X[:, 0], X[:, 1], c=[CL[0] if t == 0 else CL[1] for t in y], s=10, edgecolors="white", lw=0.2)
    ax.set_title("Cenário 1: duas luas\nclasse = grupo (suposição vale)"); clean(ax, True)
    X2, y2 = grupos_cruzados(1)
    ax = axs[1]; ax.scatter(X2[:, 0], X2[:, 1], c=[CL[0] if t == 0 else CL[1] for t in y2], s=10, edgecolors="white", lw=0.2)
    ax.axvline(0, c=INK, lw=1.2, ls="--")
    ax.set_title("Cenário 2: grupos cruzados\nclasse ≠ grupo (suposição falha)"); clean(ax, True)
    r1 = comparar(duas_luas, lambda: SVC(gamma=1.0), 3)
    r2 = comparar(grupos_cruzados, lambda: LogisticRegression(), 5, com_st=True)
    ax = axs[2]
    vals = [100 * (r1[1] - r1[0]).mean(), 100 * (r2[1] - r2[0]).mean(), 100 * (r2[2] - r2[0]).mean()]
    labs = ["Label Spreading\n(cenário 1)", "Label Spreading\n(cenário 2)", "Self-training\n(cenário 2)"]
    bars = ax.bar(range(3), vals, color=[GRN, RED, RED], width=0.6)
    for i, v in enumerate(vals): ax.text(i, v + (0.8 if v > 0 else -1.6), f"{v:+.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, c=INK, lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("ganho (p.p.)"); ax.set_title("Ganho médio sobre a\nreferência (50 sorteios)")
    ax.set_ylim(-13, 18)
    print("cenarios", vals)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "f3_cenarios")

def f3_curva_u():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    nus = [0, 50, 100, 200, 400, 800, 1207]
    ls_m, ls_s, base = [], [], []
    for nu in nus:
        accs = []
        for seed in range(10):
            r = np.random.RandomState(seed)
            idx = np.concatenate([r.choice(np.where(ytr == c)[0], 5, replace=False) for c in range(10)])
            rest = np.setdiff1d(np.arange(len(ytr)), idx); un = np.random.RandomState(seed).choice(rest, nu, replace=False)
            use = np.r_[idx, un]; yy = np.full(len(use), -1); yy[:len(idx)] = ytr[idx]
            accs.append(LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2).fit(Xtr[use], yy).score(Xte, yte))
        ls_m.append(100 * np.mean(accs)); ls_s.append(100 * np.std(accs))
    for seed in range(10):
        r = np.random.RandomState(seed)
        idx = np.concatenate([r.choice(np.where(ytr == c)[0], 5, replace=False) for c in range(10)])
        base.append(LogisticRegression(max_iter=2000).fit(Xtr[idx], ytr[idx]).score(Xte, yte))
    b = 100 * np.mean(base)
    print("curva_u", [round(v, 1) for v in ls_m], "base", round(b, 1))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    m = np.array(ls_m); s = np.array(ls_s)
    ax.plot(range(len(nus)), m, "-o", c=RED, lw=2.4, label="Label Spreading (50 rótulos + u sem rótulo)")
    ax.fill_between(range(len(nus)), m - s, m + s, color=RED, alpha=0.13)
    ax.axhline(b, c=IND, lw=2, ls="--", label=f"Referência supervisionada (50 rótulos): {b:.1f}%")
    ax.set_xticks(range(len(nus))); ax.set_xticklabels([str(n) for n in nus])
    ax.set_xlabel("número de exemplos sem rótulo (u)"); ax.set_ylabel("acurácia no teste (%)")
    ax.set_title("Mais dados sem rótulo ajudam, mas só a partir de certo ponto"); ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
    ax.set_ylim(65, 100)
    save(fig, "f3_curva_u")

def f3_mismatch():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    cls = [0, 1, 2, 3, 4]; te = np.isin(yte, cls); Xte5, yte5 = Xte[te], yte[te]
    res = {k: [] for k in ["sup", "ls_l", "ls_m", "st_l", "st_m"]}
    for s in range(20):
        r = np.random.RandomState(s)
        idx = np.concatenate([r.choice(np.where(ytr == c)[0], 5, replace=False) for c in cls])
        mc = np.isin(ytr, cls); Xl = Xtr[mc]; yl = np.full(mc.sum(), -1); pos = {i: j for j, i in enumerate(np.where(mc)[0])}
        for i in idx: yl[pos[i]] = ytr[i]
        ym = np.full(len(ytr), -1); ym[idx] = ytr[idx]
        res["sup"].append(LogisticRegression(max_iter=2000).fit(Xtr[idx], ytr[idx]).score(Xte5, yte5))
        res["ls_l"].append(LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2).fit(Xl, yl).score(Xte5, yte5))
        res["ls_m"].append(LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2).fit(Xtr, ym).score(Xte5, yte5))
        res["st_l"].append(SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=0.9).fit(Xl, yl).score(Xte5, yte5))
        res["st_m"].append(SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=0.9).fit(Xtr, ym).score(Xte5, yte5))
    M = {k: 100 * np.mean(v) for k, v in res.items()}; S = {k: 100 * np.std(v) for k, v in res.items()}
    print("mismatch", {k: round(v, 1) for k, v in M.items()})
    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    names = ["Só rotulados\n(referência)", "Label Spreading\nsem rótulo só 0–4", "Label Spreading\nsem rótulo 0–9", "Self-training\nsem rótulo só 0–4", "Self-training\nsem rótulo 0–9"]
    ks = ["sup", "ls_l", "ls_m", "st_l", "st_m"]; cols = [IND, GRN, RED, GRN, RED]
    ax.bar(range(5), [M[k] for k in ks], yerr=[S[k] for k in ks], color=cols, width=0.62, capsize=3, ecolor=INK)
    for i, k in enumerate(ks): ax.text(i, M[k] + S[k] + 0.6, f"{M[k]:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=7.8); ax.set_ylim(80, 102); ax.set_ylabel("acurácia no teste, dígitos 0–4 (%)")
    ax.set_title("Dados sem rótulo \"de fora\" reduzem o ganho"); ax.axhline(M["sup"], c=IND, lw=1, ls="--", alpha=0.6)
    save(fig, "f3_mismatch")
    return M

def f3_taxonomia():
    fig, ax = canvas(9.4, 6.0, (0, 100), (0, 72))
    rbox(ax, 36, 63, 28, 7, "Aprendizado\nsemissupervisionado", fc=NAVY, tc="white", bold=True, fs=9.3, r=1.6)
    rbox(ax, 8, 49, 36, 8.5, "INDUTIVO: produz\num modelo geral", fc=IND, tc="white", bold=True, fs=8.2, r=1.4)
    rbox(ax, 62, 49, 32, 8.5, "TRANSDUTIVO: rotula\no conjunto", fc=RED, tc="white", bold=True, fs=8.2, r=1.4)
    arrow(ax, (46, 63), (26, 57.2), GRY, lw=1.4, ms=9); arrow(ax, (54, 63), (78, 57.2), GRY, lw=1.4, ms=9)
    fam = [("Métodos\nenvoltórios", "self-training\nco-training\ntri-training", "Caps. 4–5", CYA, 1),
           ("Pré-treino\nnão supervis.", "pré-treino,\nextração de\natributos", "Cap. 12", GRN, 8.4),
           ("Intrínsecos\n(semissup.)", "generativos (EM)\nS3VM / TSVM\nconsistência\npseudo-rótulos", "Caps. 6, 7, 10, 11, 13", AMB, 22.6)]
    xs = [1, 17, 33]
    for (t, s, c, col, _), x in zip(fam, xs):
        rbox(ax, x, 26, 15, 11, t, fc=col, tc="white", bold=True, fs=6.4, r=1.2)
        rbox(ax, x, 11, 15, 14, s, fc="white", ec=col, fs=6.3, r=1.2, lw=1.4)
        ax.text(x + 7.5, 7.3, c, ha="center", fontsize=6.0, color=col, fontweight="bold")
        arrow(ax, (26, 50), (x + 7.5, 37.2), GRY, lw=0.9, ms=7)
    rbox(ax, 53, 26, 21, 11, "Métodos em grafos", fc=VIO, tc="white", bold=True, fs=8, r=1.2)
    rbox(ax, 53, 11, 21, 14, "Label Propagation\nLabel Spreading\nGraph Neural Nets", fc="white", ec=VIO, fs=7.2, r=1.2, lw=1.4)
    ax.text(63.5, 7.3, "Cap. 8", ha="center", fontsize=7, color=VIO, fontweight="bold")
    arrow(ax, (78, 50), (63.5, 37.2), GRY, lw=1.2, ms=8)
    rbox(ax, 77, 26, 21, 11, "Outros", fc=GRY, tc="white", bold=True, fs=8, r=1.2)
    rbox(ax, 77, 11, 21, 14, "Hoje, muitos métodos\nde grafos também são\nusados de modo\nindutivo", fc="white", ec=GRY, fs=6.7, r=1.2, lw=1.4)
    arrow(ax, (86, 50), (87.5, 37.2), GRY, lw=0.9, ms=7)
    ax.text(50, 2.2, "Organização inspirada em van Engelen e Hoos (2020). Os métodos modernos (Cap. 9 em diante) reúnem várias ideias.", ha="center", fontsize=7.2, style="italic", color=GRY)
    save(fig, "f3_taxonomia")

def f3_oliver():
    fig, ax = canvas(9.0, 5.0, (0, 100), (0, 58))
    ax.text(50, 55, "Avaliação realista: cinco perguntas antes de acreditar em um ganho", ha="center", fontsize=10.4, fontweight="bold")
    items = [("1", "O modelo de\nreferência foi\nbem ajustado?", "Uma referência fraca\nexagera o ganho.", IND),
             ("2", "Os parâmetros\nforam escolhidos\nsem olhar o teste?", "Escolher no teste\né vazamento.", RED),
             ("3", "Quantos rótulos e\nquantos exemplos\nsem rótulo?", "O ganho muda com\nl e u.", AMB),
             ("4", "Os exemplos sem\nrótulo vêm das\nmesmas classes?", "Classes de fora\nreduzem o ganho.", GRN),
             ("5", "A validação é\ngrande o bastante\npara decidir?", "Validação minúscula\ngera escolhas instáveis.", VIO)]
    pos = [(1, 28), (34, 28), (67, 28), (17.5, 3), (50.5, 3)]
    for (n, q, d, c), (x, y) in zip(items, pos):
        rbox(ax, x, y, 31.5, 21.5, fc="white", ec=c, lw=2, r=2)
        ax.add_patch(Circle((x + 4, y + 17), 2.9, fc=c, ec="white", lw=1.5, zorder=3)); ax.text(x + 4, y + 17, n, color="white", fontweight="bold", ha="center", va="center", fontsize=10, zorder=4)
        ax.text(x + 8.5, y + 15.2, q, fontsize=7.6, fontweight="bold", va="center", linespacing=1.3)
        ax.text(x + 15.75, y + 4.6, d, fontsize=7.3, ha="center", va="center", style="italic", color=GRY, linespacing=1.3)
    ax.text(50, 0.3, "Perguntas baseadas em Oliver et al. (2018)", fontsize=7.4, ha="center", va="center", color=GRY, style="italic")
    save(fig, "f3_oliver")

def f3_escolha():
    fig, ax = canvas(9.4, 5.6, (0, 100), (0, 66))
    def q(x, y, t, w=26, h=8):
        rbox(ax, x, y, w, h, t, fc="#FEF3C7", ec=AMB, fs=7.5, r=1.4, lw=1.6)
    def r(x, y, t, c, w=22, h=7):
        rbox(ax, x, y, w, h, t, fc=c, tc="white", bold=True, fs=7.4, r=1.4)
    ax.text(50, 63, "Guia rápido para escolher por onde começar", ha="center", fontsize=10.4, fontweight="bold")
    q(37, 50, "Os dados são imagens,\ntexto ou áudio, e há GPU?")
    r(4, 50.5, "Parte III: consistência,\nFixMatch, contrastivo", GRN, 27, 7)
    arrow(ax, (37, 54), (31.2, 54), GRY, lw=1.3, ms=9); ax.text(33.8, 55.2, "sim", fontsize=7, color=GRN, fontweight="bold", ha="center")
    arrow(ax, (50, 50), (50, 42.2), GRY, lw=1.3, ms=9); ax.text(51.5, 45.5, "não", fontsize=7, color=RED, fontweight="bold")
    q(37, 34, "Os dados formam grupos\nou curvas bem definidos?")
    r(4, 34.5, "Grafos (Cap. 8):\nLabel Spreading", VIO, 27, 7)
    arrow(ax, (37, 38), (31.2, 38), GRY, lw=1.3, ms=9); ax.text(33.8, 39.2, "sim", fontsize=7, color=GRN, fontweight="bold", ha="center")
    arrow(ax, (50, 34), (50, 26.2), GRY, lw=1.3, ms=9); ax.text(51.5, 29.5, "não", fontsize=7, color=RED, fontweight="bold")
    q(37, 18, "Já tem um classificador\nque dá probabilidades?")
    r(69, 18.5, "Self-training (Cap. 4):\nenvolve o seu modelo", CYA, 27, 7)
    arrow(ax, (63, 22), (68.8, 22), GRY, lw=1.3, ms=9); ax.text(66, 23.2, "sim", fontsize=7, color=GRN, fontweight="bold", ha="center")
    arrow(ax, (50, 18), (50, 10.2), GRY, lw=1.3, ms=9); ax.text(51.5, 13.5, "não", fontsize=7, color=RED, fontweight="bold")
    r(33, 2.5, "Modelos generativos (Cap. 6)\nou S3VM (Cap. 7)", AMB, 34, 7.5)
    ax.text(97, 62, "Em qualquer caso:\ncomece pelo modelo de referência\ne compare com rigor.", fontsize=7.6, ha="right", va="top", style="italic", color=RED, fontweight="bold")
    ax.text(3, 14, "Co-training (Cap. 5): se houver\nduas visões independentes\ndos dados.", fontsize=7.2, ha="left", va="top", color=GRY, style="italic")
    save(fig, "f3_escolha")

if __name__ == "__main__":
    for f in [f3_indtrans, f3_hipoteses, f3_cenarios, f3_curva_u, f3_mismatch, f3_taxonomia, f3_oliver, f3_escolha]:
        f()
