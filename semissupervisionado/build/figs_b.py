from fc import *
from sklearn.datasets import load_digits, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def f2_tabela():
    fig, ax = canvas(8.8, 4.4, (0, 100), (0, 52))
    cols = ["idade", "renda", "visitas", "compras"]
    X = [[34, 5.2, 12, 3], [51, 8.1, 4, 1], [27, 3.9, 20, 6], [45, 6.4, 9, 2], [38, 7.0, 15, 4], [60, 9.3, 2, 0]]
    y = ["sim", "não", "sim", "−1", "−1", "−1"]
    x0, y0, cw, rh = 12, 8, 11, 5.4
    rbox(ax, x0 - 2, y0 + 6 * rh + 0.2, 4 * cw + 24, rh + 1, fc=NAVY, r=1)
    ax.text(x0 - 1, y0 + 6.5 * rh + 1, "exemplo", color="white", fontsize=7.5, fontweight="bold", va="center")
    for j, c in enumerate(cols):
        ax.text(x0 + 6 + j * cw + cw / 2, y0 + 6.5 * rh + 0.7, c, color="white", fontsize=8.5, fontweight="bold", va="center", ha="center")
    ax.text(x0 + 6 + 4 * cw + 6, y0 + 6.5 * rh + 0.7, "y (comprou?)", color="#FDE68A", fontsize=8.5, fontweight="bold", va="center", ha="center")
    for i in range(6):
        yy = y0 + (5 - i) * rh
        lab = y[i] != "−1"
        fc = "#EEF2FF" if lab else "#F3F4F6"
        ax.add_patch(Rectangle((x0 - 2, yy), 4 * cw + 24, rh, fc=fc, ec="white", lw=1.5))
        ax.text(x0 + 1, yy + rh / 2, f"cliente {i + 1}", fontsize=7.8, va="center")
        for j in range(4):
            ax.text(x0 + 6 + j * cw + cw / 2, yy + rh / 2, str(X[i][j]), fontsize=8.5, va="center", ha="center")
        ax.text(x0 + 6 + 4 * cw + 6, yy + rh / 2, y[i] if lab else "?  (−1)", fontsize=9, va="center", ha="center", fontweight="bold", color=IND if lab else RED)
    # chaves
    ax.annotate("", xy=(x0 + 6, y0 - 1.2), xytext=(x0 + 6 + 4 * cw, y0 - 1.2), arrowprops=dict(arrowstyle="<->", color=GRN, lw=1.8))
    ax.text(x0 + 6 + 2 * cw, y0 - 3.8, "X  =  atributos (colunas): conhecidos em TODOS os exemplos", ha="center", fontsize=8.3, color=GRN, fontweight="bold")
    ax.annotate("", xy=(x0 + 6 + 4 * cw + 1, y0 - 1.2), xytext=(x0 + 6 + 4 * cw + 13, y0 - 1.2), arrowprops=dict(arrowstyle="<->", color=IND, lw=1.8))
    ax.text(91, y0 + 3.5 * rh, "cada linha =\num exemplo", fontsize=8.5, ha="center", color=INK, style="italic")
    ax.text(60, 50.5, "A tabela de dados do aprendizado semissupervisionado", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(x0 + 6 + 4 * cw + 6, y0 - 6.6, "y: só alguns\ntêm rótulo", ha="center", fontsize=8.2, color=IND, fontweight="bold", va="center")
    save(fig, "f2_tabela")

def f2_pixels():
    d = load_digits(); k = 3
    img = d.images[k]
    fig = plt.figure(figsize=(9, 3.4))
    ax = fig.add_axes([0.02, 0.05, 0.28, 0.9]); ax.imshow(img, cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Uma imagem 8×8\n(o dígito %d)" % d.target[k])
    ax2 = fig.add_axes([0.37, 0.05, 0.30, 0.9]); ax2.imshow(img, cmap="Blues", vmin=0, vmax=16)
    for i in range(8):
        for j in range(8): ax2.text(j, i, int(img[i, j]), ha="center", va="center", fontsize=6.5, color="white" if img[i, j] > 8 else INK)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("Cada pixel é um número\n(0 = branco, 16 = preto)")
    ax3 = fig.add_axes([0.72, 0.30, 0.27, 0.40]); ax3.axis("off")
    v = img.ravel()
    ax3.imshow(v[None, :], cmap="Blues", vmin=0, vmax=16, aspect="auto"); ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("Vetor com 64 atributos\n(um exemplo = uma linha de X)", fontsize=9)
    for s in ax3.spines.values(): s.set_visible(True); s.set_color(MID)
    ax3.axis("on")
    fig.add_artist(FancyArrowPatch((0.31, 0.5), (0.36, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.6))
    fig.add_artist(FancyArrowPatch((0.68, 0.5), (0.71, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.6))
    fig.savefig(OUT + "f2_pixels.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close(fig); print("fig f2_pixels")

def f2_ecossistema():
    fig, ax = canvas(8.4, 4.6, (0, 100), (0, 56))
    layers = [("Seu código e seus experimentos", "notebooks, scripts, relatórios", VIO, 0),
              ("scikit-learn", "modelos, pipelines, métricas, semissupervisionado", RED, 1),
              ("pandas  ·  matplotlib", "tabelas de dados e gráficos", AMB, 2),
              ("NumPy  (+ SciPy)", "vetores, matrizes e cálculo numérico rápido", GRN, 3),
              ("Python", "a linguagem que une tudo", IND, 4)]
    for i, (t, s, c, k) in enumerate(layers):
        y = 42 - i * 10.5
        inset = i * 3
        rbox(ax, 8 + inset, y, 84 - 2 * inset, 9, fc=c, r=1.6)
        ax.text(50, y + 5.4, t, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(50, y + 2.1, s, ha="center", va="center", color="white", fontsize=7.8)
    ax.text(50, 54.5, "A pilha de ferramentas: cada camada se apoia na de baixo", ha="center", fontsize=10.2, fontweight="bold")
    save(fig, "f2_ecossistema")

def f2_split():
    fig, ax = canvas(8.8, 3.9, (0, 100), (0, 46))
    ax.text(1, 42, "Supervisionado clássico: todos os exemplos têm rótulo", fontsize=8.8, fontweight="bold")
    x = 1
    for w, c, t in [(56, IND, "treino (rotulado)"), (20, AMB, "validação"), (20, RED, "teste")]:
        rbox(ax, x, 33, w, 6.5, t, fc=c, tc="white", fs=7.8, r=0.8, bold=True); x += w + 0.5
    ax.text(1, 28, "Semissupervisionado: o treino se divide em rotulados e sem rótulo", fontsize=8.8, fontweight="bold")
    x = 1
    for w, c, t in [(10, IND, "treino\nrotulado"), (31, "#9CA3AF", "treino sem rótulo"), (20, AMB, "validação\nrotulada"), (20, RED, "teste\nrotulado")]:
        rbox(ax, x, 18, w, 8, t, fc=c, tc="white", fs=7.3, r=0.8, bold=True); x += w + 0.5
    ax.text(1, 12, "Regra de ouro: o conjunto de teste nunca participa do treino, nem como exemplo sem rótulo.", fontsize=8.3, color=RED, fontweight="bold")
    ax.text(1, 7.5, "Cuidado: uma validação rotulada grande demais contradiz a premissa de que rótulos são escassos.", fontsize=7.8, color=GRY, style="italic")
    ax.text(1, 3.5, "(veja o Capítulo 3)", fontsize=7.8, color=GRY, style="italic")
    save(fig, "f2_split")

def f2_kfold():
    fig, ax = canvas(8.2, 3.6, (0, 100), (0, 42))
    ax.text(50, 39.5, "Validação cruzada com 5 dobras (k-fold)", ha="center", fontsize=10.3, fontweight="bold")
    for r in range(5):
        y = 31 - r * 6.2
        ax.text(1, y + 2.1, f"rodada {r + 1}", fontsize=8, va="center")
        for k in range(5):
            c = RED if k == r else IND
            rbox(ax, 14 + k * 16, y, 15.2, 4.6, "teste" if k == r else "treino", fc=c, tc="white", fs=7.5, r=0.6, bold=True)
        ax.text(96, y + 2.1, f"→ nota {r + 1}", fontsize=7.6, va="center", color=GRY, ha="center")
    ax.text(50, -0.5, "Nota final = média das 5 notas. Todo exemplo é usado para teste uma vez e para treino quatro vezes.", ha="center", fontsize=8.2, style="italic", color=GRY)
    ax.set_ylim(-3, 42)
    save(fig, "f2_kfold")

def f2_distancias():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    A, B = np.array([1.0, 1.0]), np.array([5.0, 4.0])
    ax = axs[0]; ax.plot(*zip(A, B), c=IND, lw=2.5); ax.scatter(*zip(A, B), c=[AMB, GRN], s=110, zorder=3, edgecolors="white")
    ax.text(A[0] - 0.2, A[1] - 0.5, "A (1,1)", fontsize=8); ax.text(B[0] - 0.4, B[1] + 0.3, "B (5,4)", fontsize=8)
    ax.set_title("Euclidiana\n√(4² + 3²) = 5"); ax.set_xlim(0, 6.5); ax.set_ylim(0, 5.5); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax = axs[1]; ax.plot([A[0], B[0], B[0]], [A[1], A[1], B[1]], c=RED, lw=2.5); ax.scatter(*zip(A, B), c=[AMB, GRN], s=110, zorder=3, edgecolors="white")
    ax.text(3, 0.5, "4", color=RED, fontweight="bold"); ax.text(5.2, 2.5, "3", color=RED, fontweight="bold")
    ax.set_title("Manhattan\n4 + 3 = 7"); ax.set_xlim(0, 6.5); ax.set_ylim(0, 5.5); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax = axs[2]
    u, v = np.array([4.0, 1.0]), np.array([3.0, 3.0]); w = np.array([1.0, 4.0])
    for vec, c, n in [(u, IND, "u"), (v, GRN, "v"), (w, AMB, "w")]:
        ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2)); ax.text(vec[0] + 0.1, vec[1] + 0.1, n, color=c, fontweight="bold")
    ax.set_title("Cosseno: mede o ângulo\nv é mais parecido com u e w do que u de w"); ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    save(fig, "f2_distancias")

def f2_gauss():
    from scipy.stats import norm
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5))
    x = np.linspace(-5, 9, 400)
    ax = axs[0]
    ax.fill_between(x, norm.pdf(x, 2, 1), color=IND, alpha=0.25); ax.plot(x, norm.pdf(x, 2, 1), c=IND, lw=2.2)
    ax.fill_between(x, norm.pdf(x, 2, 2), color=AMB, alpha=0.18); ax.plot(x, norm.pdf(x, 2, 2), c=AMB, lw=2.2)
    ax.axvline(2, c=GRY, ls="--", lw=1); ax.text(2.1, 0.41, "média μ = 2", fontsize=8, color=INK)
    ax.text(3.4, 0.30, "σ = 1\n(concentrada)", color=IND, fontsize=8.5, fontweight="bold"); ax.text(4.9, 0.11, "σ = 2\n(espalhada)", color=AMB, fontsize=8.5, fontweight="bold")
    ax.set_title("A curva normal (gaussiana)"); ax.set_xlabel("valor do atributo"); ax.set_ylabel("densidade"); ax.set_ylim(0, 0.46)
    ax = axs[1]
    ax.fill_between(x, norm.pdf(x, 0, 1.2), color=IND, alpha=0.25); ax.plot(x, norm.pdf(x, 0, 1.2), c=IND, lw=2.2)
    ax.fill_between(x, norm.pdf(x, 4, 1.2), color=RED, alpha=0.25); ax.plot(x, norm.pdf(x, 4, 1.2), c=RED, lw=2.2)
    ax.axvline(2, c=INK, lw=1.8, ls="--"); ax.text(2.15, 0.36, "fronteira de decisão", fontsize=8.3, rotation=90, va="top")
    ax.text(-3.2, 0.2, "classe A", color=IND, fontweight="bold"); ax.text(5.3, 0.2, "classe B", color=RED, fontweight="bold")
    ax.set_title("Duas classes, uma curva para cada"); ax.set_xlabel("valor do atributo"); ax.set_yticks([])
    save(fig, "f2_gauss")

def f2_bayes():
    fig, ax = canvas(8.6, 4.4, (0, 100), (0, 54))
    ax.text(50, 51.5, "Teorema de Bayes com frequências: 1.000 e-mails", ha="center", fontsize=10.3, fontweight="bold")
    rbox(ax, 30, 40, 40, 7, "1.000 e-mails", fc=NAVY, tc="white", bold=True, fs=9.5, r=1.4)
    rbox(ax, 8, 26, 30, 7, "200 são spam (20%)", fc=RED, tc="white", bold=True, fs=8.8, r=1.4)
    rbox(ax, 62, 26, 30, 7, "800 não são spam", fc=GRN, tc="white", bold=True, fs=8.8, r=1.4)
    arrow(ax, (42, 40), (25, 33.2), GRY, lw=1.4, ms=9); arrow(ax, (58, 40), (75, 33.2), GRY, lw=1.4, ms=9)
    rbox(ax, 1, 10, 22, 12, "160 têm a\npalavra “prêmio”\n(80% dos spams)", fc="#FEE2E2", ec=RED, fs=7.8, r=1.4)
    rbox(ax, 25, 10, 22, 12, "40 não têm", fc=LIGHT, ec=GRY, fs=7.8, r=1.4)
    rbox(ax, 55, 10, 22, 12, "40 têm “prêmio”\n(5% dos não spams)", fc="#D1FAE5", ec=GRN, fs=7.8, r=1.4)
    rbox(ax, 79, 10, 20, 12, "760 não têm", fc=LIGHT, ec=GRY, fs=7.8, r=1.4)
    arrow(ax, (23, 26), (12, 22.2), RED, lw=1.2, ms=8); arrow(ax, (23, 26), (36, 22.2), GRY, lw=1.2, ms=8)
    arrow(ax, (77, 26), (66, 22.2), GRN, lw=1.2, ms=8); arrow(ax, (77, 26), (89, 22.2), GRY, lw=1.2, ms=8)
    ax.text(50, 4.8, "Entre os e-mails com “prêmio”: 160 spam + 40 não spam = 200.   P(spam | prêmio) = 160 / 200 = 80%", ha="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(50, 1.2, "Mesmo sendo só 20% dos e-mails, o spam domina quando a palavra aparece.", ha="center", fontsize=8, style="italic", color=GRY)
    save(fig, "f2_bayes")

def f2_overfit():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 1, 14)); f = lambda t: np.sin(2 * np.pi * t)
    y = f(x) + rng.normal(0, 0.22, 14)
    t = np.linspace(0, 1, 300)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, deg, tt, c in zip(axs, [1, 3, 11], ["Simples demais\n(subajuste)", "Na medida certa", "Complexo demais\n(sobreajuste)"], [AMB, GRN, RED]):
        p = np.polyfit(x, y, deg); ax.plot(t, f(t), c=GRY, lw=1.5, ls="--", label="padrão real"); ax.plot(t, np.clip(np.polyval(p, t), -2.2, 2.2), c=c, lw=2.4, label="modelo")
        ax.scatter(x, y, c=INK, s=22, zorder=3); ax.set_title(tt); ax.set_ylim(-2.2, 2.2); clean(ax, True)
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    save(fig, "f2_overfit")

def f2_confusao():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    p = m.predict(sc.transform(Xte))
    cm = confusion_matrix(yte, p, labels=[1, 0])   # 1 = benigno (positivo), 0 = maligno
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    cols = [["#D1FAE5", "#FEE2E2"], ["#FEE2E2", "#D1FAE5"]]
    names = [["VP", "FN"], ["FP", "VN"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j, 1 - i), 1, 1, fc=cols[i][j], ec="white", lw=3))
            ax.text(j + 0.5, 1 - i + 0.58, str(cm[i, j]), ha="center", va="center", fontsize=22, fontweight="bold")
            ax.text(j + 0.5, 1 - i + 0.2, names[i][j], ha="center", va="center", fontsize=10, color=GRY)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["previsto:\nbenigno", "previsto:\nmaligno"], fontsize=8.5)
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["real:\nbenigno", "real:\nmaligno"], fontsize=8.5); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Matriz de confusão (teste: 171 exames)")
    vp, fn, fp, vn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (vp + vn) / cm.sum(); prec = vp / (vp + fp); rev = vp / (vp + fn); f1 = 2 * prec * rev / (prec + rev)
    ax = axs[1]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    rows = [("Acurácia", "(VP+VN) / total", acc, IND), ("Precisão", "VP / (VP+FP)", prec, GRN), ("Revocação", "VP / (VP+FN)", rev, AMB), ("F1", "média harmônica", f1, RED)]
    for k, (n, f, v, c) in enumerate(rows):
        y0 = 8 - k * 2.3
        rbox(ax, 0.2, y0, 9.4, 1.9, fc="white", ec=c, lw=1.6, r=0.35)
        ax.text(0.6, y0 + 1.25, n, fontsize=10, fontweight="bold", color=c, va="center"); ax.text(0.6, y0 + 0.5, f, fontsize=8, color=GRY, va="center")
        ax.text(9.2, y0 + 0.95, f"{100 * v:.1f}%", fontsize=14, fontweight="bold", ha="right", va="center")
    ax.set_title("Métricas calculadas a partir da matriz")
    print("confusao", cm.tolist(), acc, prec, rev, f1)
    save(fig, "f2_confusao")

def f2_api():
    fig, ax = canvas(8.8, 4.2, (0, 100), (0, 48))
    ax.text(50, 45.5, "O \"contrato\" do scikit-learn: todo objeto segue o mesmo padrão", ha="center", fontsize=10.2, fontweight="bold")
    steps = [("1. Criar", "modelo =\nLogisticRegression()", IND), ("2. Ajustar", "modelo.fit(\nX_treino, y_treino)", GRN), ("3. Prever", "modelo.predict(\nX_novo)", AMB), ("4. Avaliar", "modelo.score(\nX_teste, y_teste)", RED)]
    for i, (t, c, col) in enumerate(steps):
        x = 1.5 + i * 24.5
        rbox(ax, x, 25, 22, 15, fc="white", ec=col, lw=2, r=1.8)
        rbox(ax, x, 35, 22, 5, t, fc=col, tc="white", bold=True, fs=9, r=1.8)
        ax.text(x + 11, 29.5, c, ha="center", va="center", fontsize=6.4, family="monospace", linespacing=1.5)
        if i < 3: arrow(ax, (x + 22.2, 32.5), (x + 24.3, 32.5), GRY, lw=1.4, ms=9)
    rbox(ax, 1.5, 2, 97, 18, fc="#F8FAFC", ec=MID, r=1.5)
    ax.text(4, 17, "Pipeline: encadeia passos, e cada passo segue o mesmo contrato", fontsize=8.5, fontweight="bold", va="center")
    for i, (t, col) in enumerate([("StandardScaler\n(transformar)", AMB), ("LogisticRegression\n(prever)", IND)]):
        rbox(ax, 4 + i * 29, 5, 25, 8, t, fc=col, tc="white", fs=7.0, r=1, bold=True)
    arrow(ax, (29.2, 9), (32.8, 9), INK, lw=1.4, ms=9)
    ax.text(80, 9, "make_pipeline(\n  StandardScaler(),\n  LogisticRegression())", fontsize=6.4, family="monospace", va="center", ha="center", linespacing=1.5)
    save(fig, "f2_api")

def f2_escala():
    X, y = load_breast_cancer(return_X_y=True)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    j1, j2 = 0, 3   # mean radius, mean area
    ax = axs[0]
    ax.scatter(X[:, j1], X[:, j2], c=[CL[0] if t == 0 else CL[1] for t in y], s=12, edgecolors="white", lw=0.3)
    ax.set_xlabel("raio médio (≈ 6 a 28)"); ax.set_ylabel("área média (≈ 140 a 2.500)"); ax.set_title("Escalas muito diferentes")
    ax.set_xlim(0, 2600); ax.set_ylim(0, 2600); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.text(1000, 300, "no eixo comum, o raio\nparece quase constante", fontsize=8, color=RED, ha="center", style="italic")
    ax = axs[1]
    Z = StandardScaler().fit_transform(X[:, [j1, j2]])
    ax.scatter(Z[:, 0], Z[:, 1], c=[CL[0] if t == 0 else CL[1] for t in y], s=12, edgecolors="white", lw=0.3)
    ax.set_xlabel("raio médio (padronizado)"); ax.set_ylabel("área média (padronizada)"); ax.set_title("Depois de padronizar: média 0, desvio 1"); ax.grid(alpha=0.25)
    save(fig, "f2_escala")

if __name__ == "__main__":
    for f in [f2_tabela, f2_pixels, f2_ecossistema, f2_split, f2_kfold, f2_distancias, f2_gauss, f2_bayes, f2_overfit, f2_confusao, f2_api, f2_escala]:
        f()
