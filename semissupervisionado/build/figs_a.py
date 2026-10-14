from fc import *
from sklearn.datasets import make_moons, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def f1_iceberg():
    """Muitos dados, poucos rótulos."""
    rng = np.random.default_rng(4)
    fig, ax = canvas(8.4, 4.9, (0, 100), (0, 60))
    ax.add_patch(Rectangle((0, 0), 100, 60, fc="#F8FAFC", ec="none"))
    # mar de dados sem rótulo
    ax.add_patch(Rectangle((0, 0), 100, 36, fc="#E0F2FE", ec="none"))
    ax.plot([0, 100], [36, 36], c=CYA, lw=2)
    ax.text(2, 57, "O que está visível: poucos exemplos COM rótulo", fontsize=10, fontweight="bold", color=IND)
    ax.text(2, 33, "O que está submerso: oceano de exemplos SEM rótulo", fontsize=10, fontweight="bold", color="#0369A1")
    # pontos rotulados (acima da linha)
    P = np.c_[rng.uniform(8, 60, 9), rng.uniform(38.5, 49, 9)]
    cols = [IND, AMB, GRN] * 3
    for (x, y), c in zip(P, cols):
        ax.scatter(x, y, s=170, c=c, edgecolors="white", lw=1.2, zorder=3)
        ax.text(x, y + 2.6, "y", ha="center", fontsize=8, color=c, fontweight="bold")
    # pontos sem rótulo (abaixo)
    Q = np.c_[rng.uniform(3, 97, 330), rng.uniform(2, 31, 330)]
    ax.scatter(Q[:, 0], Q[:, 1], s=16, c=GRY, alpha=0.75, edgecolors="white", lw=0.3, zorder=2)
    rbox(ax, 66, 40, 32, 12, "Rotular custa tempo,\ndinheiro e especialistas", fc="white", ec=AMB, fs=8.8, r=2)
    rbox(ax, 66, 14, 32, 12, "Coletar dados brutos\né quase de graça", fc="white", ec=CYA, fs=8.8, r=2)
    save(fig, "f1_iceberg")

def f1_funil():
    fig, ax = canvas(8.4, 4.4, (0, 100), (0, 52))
    levels = [("Dados brutos gerados", "câmeras, sensores, sites, prontuários, e-mails…", CYA, 90),
              ("Dados armazenados", "parte do que é gerado é guardada", IND, 72),
              ("Dados revisados por humanos", "alguém olha, entende e decide", AMB, 56),
              ("Dados rotulados", "com a resposta correta anotada", RED, 44)]
    y = 40
    for i, (t, s, c, w) in enumerate(levels):
        x0 = (100 - w) / 2
        ax.add_patch(Polygon([(x0, y), (x0 + w, y), (x0 + w - 4, y - 9.5), (x0 + 4, y - 9.5)], fc=c, ec="white", lw=2))
        ax.text(50, y - 3.4, t, ha="center", va="center", fontsize=10 if i < 3 else 9.5, color="white", fontweight="bold")
        ax.text(50, y - 7, s, ha="center", va="center", fontsize=7.6, color="white")
        y -= 10.5
    ax.annotate("", xy=(99, 2), xytext=(99, 50), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    ax.text(96.2, 26, "cada etapa\nreduz a quantidade", rotation=90, ha="center", va="center", fontsize=8, color=INK)
    save(fig, "f1_funil")

def f1_tipos():
    fig, ax = canvas(8.4, 5.4, (0, 100), (8, 75))
    cards = [("Supervisionado", "tem X e y (rótulos em todos)", "feedback: gabarito", IND, 0, 52),
             ("Não supervisionado", "tem apenas X (nenhum rótulo)", "feedback: nenhum", GRN, 34, 52),
             ("SEMISSUPERVISIONADO", "X em todos; y em poucos", "feedback: gabarito parcial", RED, 68, 52),
             ("Auto-supervisionado", "rótulos criados dos próprios dados", "feedback: tarefa-pretexto", VIO, 0, 27),
             ("Por reforço", "agente age em um ambiente", "feedback: recompensa", AMB, 34, 27)]
    rng = np.random.default_rng(2)
    w, h = 31, 21
    for i, (t, s1, s2, c, x0, y0) in enumerate(cards):
        rbox(ax, x0 + 0.5, y0, w, h, fc="white", ec=c, lw=2.4 if i == 2 else 1.6, r=2.2)
        rbox(ax, x0 + 0.5, y0 + h - 5.6, w, 5.6, t, fc=c, tc="white", bold=True, fs=8.3, r=2.2)
        ax.text(x0 + 2, y0 + h - 8.6, s1, fontsize=6.9, va="center")
        ax.text(x0 + 2, y0 + h - 11.4, s2, fontsize=6.9, va="center", color=c, fontweight="bold")
        # mini diagrama
        base = np.array([x0 + 16, y0 + 4.2])
        if i <= 2:
            grp = [(-8, 0), (0, 2.5), (8, 0)]
            for k, g in enumerate(grp):
                Pp = rng.normal(base + g, [1.7, 1.3], (9, 2))
                col = CL[k] if i == 0 else GRY
                ax.scatter(Pp[:, 0], Pp[:, 1], c=col, s=11, edgecolors="white", lw=0.3, zorder=3)
                if i == 2:
                    ax.scatter(*(base + g + [0.4, 0.2]), c=CL[k], s=55, edgecolors="white", lw=0.8, zorder=4)
        elif i == 3:
            rbox(ax, x0 + 2.5, y0 + 2.3, 11, 4.6, "A casa é ___", fc=LIGHT, fs=6.6, r=1)
            arrow(ax, (x0 + 14, y0 + 4.6), (x0 + 18, y0 + 4.6), VIO, lw=1.3, ms=8)
            rbox(ax, x0 + 18.5, y0 + 2.3, 12, 4.6, "prever 'azul'", fc="#EDE9FE", ec=VIO, fs=6.6, r=1)
        else:
            rbox(ax, x0 + 2.5, y0 + 2.3, 11, 4.6, "Agente", fc="#FEF3C7", ec=AMB, fs=6.8, r=1)
            rbox(ax, x0 + 19, y0 + 2.3, 11, 4.6, "Ambiente", fc=LIGHT, ec=GRY, fs=6.8, r=1)
            arrow(ax, (x0 + 13.5, y0 + 6), (x0 + 19, y0 + 6), AMB, lw=1.2, ms=7, rad=-0.35)
            arrow(ax, (x0 + 19, y0 + 3.2), (x0 + 13.5, y0 + 3.2), GRN, lw=1.2, ms=7, rad=-0.35)
    # complemento do cartão por reforço e auto-supervisionado (ícones)
    # legenda
    rbox(ax, 68.5, 27, 31, 21, fc="#FEF2F2", ec=RED, lw=1.2, r=2.2)
    ax.text(84, 43, "Onde está este livro", ha="center", fontsize=9, fontweight="bold", color=RED)
    ax.text(84, 36.5, "Um caminho do meio:\nmuitos exemplos sem rótulo\n+ alguns com rótulo\n= modelo melhor", ha="center", va="center", fontsize=7.8, linespacing=1.4)
    ax.text(50, 20, "● colorido = exemplo com rótulo        ● cinza = exemplo sem rótulo", ha="center", fontsize=8.5, color=INK)
    ax.text(50, 14, "O aprendizado semissupervisionado usa os dois tipos ao mesmo tempo.", ha="center", fontsize=9, fontweight="bold", color=RED)
    save(fig, "f1_tipos")

def f1_espectro():
    fig, ax = canvas(8.4, 3.2, (0, 100), (0, 38))
    n = 400
    g = np.linspace(0, 1, n)[None, :]
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("e", [GRN, "#FDE68A", IND])
    ax.imshow(g, extent=[5, 95, 14, 22], aspect="auto", cmap=cm, zorder=1); ax.set_xlim(0, 100); ax.set_ylim(0, 38)
    ax.text(5, 25, "0% de exemplos rotulados", fontsize=8.5, ha="left", fontweight="bold", color=GRN)
    ax.text(95, 25, "100% rotulados", fontsize=8.5, ha="right", fontweight="bold", color=IND)
    ax.text(5, 9.5, "Não supervisionado", fontsize=9, ha="left", fontweight="bold", color=GRN)
    ax.text(95, 9.5, "Supervisionado", fontsize=9, ha="right", fontweight="bold", color=IND)
    ax.add_patch(Polygon([(20, 14), (50, 14), (47, 22), (23, 22)], fc="none", ec=RED, lw=0))
    rbox(ax, 16, 29, 56, 7, "Semissupervisionado: só uma pequena parte é rotulada", fc=RED, tc="white", bold=True, fs=7.8, r=1.8)
    ax.annotate("", xy=(44, 22.5), xytext=(44, 29), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.text(50, 3.5, "Não há uma linha rígida: é um contínuo, conforme a fração de rótulos disponível.", ha="center", fontsize=8.2, style="italic", color=GRY)
    save(fig, "f1_espectro")

def _moons():
    X, y = make_moons(300, noise=0.07, random_state=2)
    def near(p, c):
        i = np.where(y == c)[0]; return i[np.argmin(((X[i] - p) ** 2).sum(1))]
    idx = [near(p, c) for p, c in [((0, 1.0), 0), ((-0.8, 0.5), 0), ((1, -0.5), 1), ((1.8, -0.1), 1)]]
    return X, y, np.array(idx)

def f1_luas():
    X, y, idx = _moons()
    yy = np.full(len(y), -1); yy[idx] = y[idx]
    sup = SVC(kernel="rbf", gamma=1).fit(X[idx], y[idx])
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.8, max_iter=2000).fit(X, yy)
    xx, yg = np.meshgrid(np.linspace(-1.7, 2.7, 300), np.linspace(-1.2, 1.7, 300))
    G = np.c_[xx.ravel(), yg.ravel()]
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.5))
    # 1: dados
    ax = axs[0]
    m = np.ones(len(y), bool); m[idx] = False
    ax.scatter(X[m, 0], X[m, 1], c=GRY, s=14, edgecolors="white", lw=0.3)
    ax.scatter(X[idx, 0], X[idx, 1], c=[CL[0] if c == 0 else CL[1] for c in y[idx]], s=120, edgecolors="black", lw=1.2, marker="*", zorder=4)
    ax.set_title("1) Os dados: 4 rótulos\ne 296 exemplos sem rótulo"); clean(ax, True)
    # 2: supervisionado
    ax = axs[1]
    Z = sup.predict(G).reshape(xx.shape)
    ax.contourf(xx, yg, Z, levels=[-.5, .5, 1.5], colors=["#C7D2FE", "#FDE68A"], alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=[CL[0] if c == 0 else CL[1] for c in y], s=11, edgecolors="white", lw=0.3)
    ax.scatter(X[idx, 0], X[idx, 1], c="white", s=110, edgecolors="black", lw=1.2, marker="*", zorder=4)
    ax.set_title(f"2) Só os 4 rótulos\nacurácia = {sup.score(X, y) * 100:.0f}%"); clean(ax, True)
    # 3: semissupervisionado
    ax = axs[2]
    Z = ls.predict(G).reshape(xx.shape)
    ax.contourf(xx, yg, Z, levels=[-.5, .5, 1.5], colors=["#C7D2FE", "#FDE68A"], alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=[CL[0] if c == 0 else CL[1] for c in y], s=11, edgecolors="white", lw=0.3)
    ax.scatter(X[idx, 0], X[idx, 1], c="white", s=110, edgecolors="black", lw=1.2, marker="*", zorder=4)
    ax.set_title(f"3) Rótulos + exemplos sem rótulo\nacurácia = {ls.score(X, y) * 100:.0f}%"); clean(ax, True)
    print("luas sup", sup.score(X, y), "ss", ls.score(X, y))
    save(fig, "f1_luas")

def f1_curva():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    nls = [10, 20, 50, 100, 200, 300]
    R = {k: [] for k in ["sup", "ls", "st"]}
    for nl in nls:
        r = []
        for seed in range(10):
            rng = np.random.RandomState(seed); per = nl // 10
            idx = np.concatenate([rng.choice(np.where(ytr == c)[0], per, replace=False) for c in range(10)])
            yy = np.full(len(ytr), -1); yy[idx] = ytr[idx]
            a = LogisticRegression(max_iter=2000).fit(Xtr[idx], ytr[idx]).score(Xte, yte)
            b = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100).fit(Xtr, yy).score(Xte, yte)
            c = SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=0.9).fit(Xtr, yy).score(Xte, yte)
            r.append((a, b, c))
        r = np.array(r);
        for j, k in enumerate(["sup", "ls", "st"]): R[k].append((r[:, j].mean(), r[:, j].std()))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for k, lab, c in [("sup", "Supervisionado (regressão logística, só rotulados)", IND), ("st", "Self-training (regressão logística)", AMB), ("ls", "Label Spreading (grafo de vizinhos)", RED)]:
        m = np.array([v[0] for v in R[k]]) * 100; s = np.array([v[1] for v in R[k]]) * 100
        ax.plot(nls, m, "-o", color=c, lw=2.2, ms=5, label=lab); ax.fill_between(nls, m - s, m + s, color=c, alpha=0.13)
    ax.set_xscale("log"); ax.set_xticks(nls); ax.set_xticklabels([str(n) for n in nls])
    ax.set_xlabel("exemplos rotulados (treino com 1.257 exemplos; o restante entra sem rótulo)"); ax.set_ylabel("acurácia no teste (%)")
    ax.set_title("Dígitos manuscritos (UCI): poucos rótulos, grande diferença"); ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
    ax.set_ylim(55, 100)
    ax.text(1.0, -0.22, "média de 10 sorteios; faixa = ±1 desvio padrão", transform=ax.transAxes, ha="right", fontsize=7.5, color=GRY, style="italic")
    save(fig, "f1_curva")

def f1_historia():
    fig, ax = canvas(8.8, 4.6, (0, 100), (0, 56))
    ax.plot([3, 97], [28, 28], c=INK, lw=3, solid_capstyle="round")
    ev = [(5, "1965", "Scudder\nauto-treinamento", IND, 1), (17, "1994", "Shahshahani\ne Landgrebe", CYA, -1), (27, "1995", "Yarowsky\ndesambiguação", GRN, 1),
          (37, "1998", "Blum e Mitchell\nco-training", AMB, -1), (46, "2000", "Nigam et al.\nEM + texto", RED, 1), (55, "2003", "Zhu et al.;\nZhou et al.\ngrafos", VIO, -1),
          (64, "2006", "Livro de\nChapelle et al.", PNK, 1), (73, "2013–17", "Pseudo-label,\nTemporal Ens.,\nMean Teacher", IND, -1),
          (83, "2019–20", "MixMatch,\nFixMatch", CYA, 1), (93, "2020", "SimCLR\ncontrastivo", GRN, -1)]
    for x, yr, txt, c, up in ev:
        ax.scatter(x, 28, s=130, c=c, edgecolors="white", lw=2, zorder=4)
        yb = 28 + up * 8
        ax.plot([x, x], [28, yb + (-1 if up > 0 else 1) * 0.5 + (1.4 if up > 0 else -1.4)], c=c, lw=1.4)
        if up > 0:
            ax.text(x, yb + 3, txt, ha="center", va="bottom", fontsize=6.6, linespacing=1.25)
            ax.text(x, yb + 0.7, yr, ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=c)
        else:
            ax.text(x, yb - 0.7, yr, ha="center", va="top", fontsize=7.8, fontweight="bold", color=c)
            ax.text(x, yb - 3.2, txt, ha="center", va="top", fontsize=6.6, linespacing=1.25)
    save(fig, "f1_historia")

def f1_aplicacoes():
    fig, ax = canvas(8.4, 4.9, (0, 100), (0, 60))
    rbox(ax, 34, 21, 32, 16, "Poucos rótulos\n+\nmuitos dados sem rótulo", fc=RED, tc="white", bold=True, fs=10, r=4)
    items = [("Imagens médicas", "exames são abundantes;\nlaudos especializados, não", 2, 40, IND), ("Fala e áudio", "horas de gravação;\ntranscrever é lento", 36, 44, CYA), ("Texto e e-mail", "milhões de mensagens;\npoucas marcadas como spam", 70, 40, GRN),
             ("Biologia", "sequências de proteínas;\nfunção conhecida em poucas", 2, 4, AMB), ("Sensoriamento remoto", "satélites fotografam tudo;\nmapear o solo é caro", 36, 2, VIO), ("Fraude e segurança", "milhões de transações;\nfraudes confirmadas são raras", 70, 4, PNK)]
    for t, s, x, y, c in items:
        rbox(ax, x, y, 28, 14, fc="white", ec=c, lw=2, r=2.5)
        rbox(ax, x, y + 9, 28, 5, t, fc=c, tc="white", bold=True, fs=8, r=2.5)
        ax.text(x + 14, y + 4.2, s, ha="center", va="center", fontsize=6.9, linespacing=1.3)
    for (x, y) in [(16, 40), (50, 44), (84, 40)]:
        arrow(ax, (x, y), (50 + (x - 50) * 0.25, 37.5 if y > 30 else 21), GRY, lw=1.1, ms=9)
    for (x, y) in [(16, 18), (50, 16), (84, 18)]:
        arrow(ax, (x, y), (50 + (x - 50) * 0.25, 21), GRY, lw=1.1, ms=9)
    save(fig, "f1_aplicacoes")

def f1_projeto():
    fig, ax = canvas(8.8, 3.6, (0, 100), (0, 42))
    steps = [("1", "Entender\no problema", IND), ("2", "Reunir\ndados", CYA), ("3", "Rotular\num pouco", AMB), ("4", "Escolher\nmétodo", GRN),
             ("5", "Treinar", VIO), ("6", "Avaliar\ncom rigor", RED), ("7", "Decidir e\nmonitorar", PNK)]
    w = 12.2
    for i, (n, t, c) in enumerate(steps):
        x = 1.5 + i * 13.9
        rbox(ax, x, 16, w, 17, fc="white", ec=c, lw=2, r=2.5)
        ax.add_patch(Circle((x + w / 2, 31), 3.1, fc=c, ec="white", lw=1.5, zorder=3))
        ax.text(x + w / 2, 31, n, ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=4)
        ax.text(x + w / 2, 22.2, t, ha="center", va="center", fontsize=7.0, fontweight="bold", linespacing=1.3)
        if i < 6: arrow(ax, (x + w + 0.1, 24.5), (x + 13.9 - 0.1, 24.5), GRY, lw=1.3, ms=9)
    arrow(ax, (92, 14), (9, 14), RED, lw=1.4, rad=-0.0, ms=10, ls="--")
    ax.text(50, 9.5, "volte ao início: mais rótulos? outro método? outros dados?", ha="center", fontsize=8.5, color=RED, style="italic")
    ax.text(50, 38.5, "Etapas de um projeto semissupervisionado", ha="center", fontsize=10.5, fontweight="bold", color=INK)
    save(fig, "f1_projeto")

if __name__ == "__main__":
    for f in [f1_iceberg, f1_funil, f1_tipos, f1_espectro, f1_luas, f1_curva, f1_historia, f1_aplicacoes, f1_projeto]:
        f()
