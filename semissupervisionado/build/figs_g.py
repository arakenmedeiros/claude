"""Figuras do Capítulo 7 (baixa densidade: S3VM, TSVM, entropia)."""
from fc import *
from figs_d import out, pct
from sklearn.svm import SVC
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_g = {}
exec(open("/home/user/claude/semissupervisionado/cap/code/ch07_tsvm.py").read().split('print("duas luas')[0], _g)
tsvm = _g["tsvm"]
_e = {}
exec(open("/home/user/claude/semissupervisionado/cap/code/ch07_entropia.py").read().split("X, y = load_digits")[0], _e)
logistica_entropia, sigmoide = _e["logistica_entropia"], _e["sigmoide"]

def f7_praca():
    rng = np.random.default_rng(6)
    A = rng.normal([-2.2, 0], [0.9, 1.2], (90, 2)); B = rng.normal([2.2, 0], [0.9, 1.2], (90, 2))
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, x0, t, c in [(axs[0], 2.4, "Fronteira ruim: corta uma região\nmovimentada (muitos pontos)", RED), (axs[1], 0.0, "Fronteira boa: passa pelo \"campo vazio\"\nentre os dois grupos", GRN)]:
        ax.scatter(*A.T, c=GRY, s=14, edgecolors="white", lw=0.3); ax.scatter(*B.T, c=GRY, s=14, edgecolors="white", lw=0.3)
        ax.scatter(*A[0], c=IND, marker="*", s=170, edgecolors="black", zorder=4); ax.scatter(*B[0], c=AMB, marker="*", s=170, edgecolors="black", zorder=4)
        ax.axvline(x0, c=c, lw=3); ax.set_title(t, fontsize=9.2); clean(ax, True); ax.set_xlim(-5, 5); ax.set_ylim(-3.5, 3.5)
    save(fig, "f7_praca")

def f7_perdas():
    f = np.linspace(-2.5, 2.5, 400)
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axs[0]; ax.plot(f, np.maximum(0, 1 - f), c=IND, lw=2.5, label="rotulado, classe +1: max(0, 1 − f)"); ax.plot(f, np.maximum(0, 1 + f), c=AMB, lw=2.5, label="rotulado, classe −1: max(0, 1 + f)")
    ax.set_title("Perda de uma SVM nos exemplos rotulados\n(convexa: um só mínimo)", fontsize=9.2); ax.set_xlabel("f(x): pontuação do classificador"); ax.set_ylabel("perda"); ax.legend(frameon=False, fontsize=7.8); ax.grid(alpha=0.25); ax.axvline(0, c=GRY, lw=1)
    ax = axs[1]; ax.plot(f, np.maximum(0, 1 - np.abs(f)), c=RED, lw=2.8, label="sem rótulo: max(0, 1 − |f|)")
    ax.fill_between([-1, 1], 0, 1.05, color=RED, alpha=0.08); ax.text(0, 0.55, "dentro da margem:\npenalidade", ha="center", fontsize=8.2, color=RED)
    ax.set_title("Perda de uma S3VM nos exemplos sem rótulo\n(não convexa: dois \"vales\", em f = −1 e f = +1)", fontsize=9.2); ax.set_xlabel("f(x): pontuação do classificador"); ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.grid(alpha=0.25); ax.axvline(0, c=GRY, lw=1)
    save(fig, "f7_perdas")

def f7_luas():
    X, y = make_moons(300, noise=0.1, random_state=4); r = np.random.RandomState(4)
    idx = np.r_[r.choice(np.where(y == 0)[0], 3, replace=False), r.choice(np.where(y == 1)[0], 3, replace=False)]
    yp = np.full(len(y), -1); yp[idx] = y[idx]; gamma = 5.0
    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 260), np.linspace(-1.1, 1.6, 260)); G = np.c_[xx.ravel(), yy.ravel()]
    svm = SVC(C=10, gamma=gamma).fit(X[idx], y[idx]); ts = tsvm(X, yp, C=10, gamma=gamma, Cu_final=1.0)
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5))
    ax = axs[0]; ax.scatter(X[:, 0], X[:, 1], c=GRY, s=9, edgecolors="none"); ax.scatter(X[idx, 0], X[idx, 1], c=[CL[0] if t == 0 else CL[1] for t in y[idx]], marker="*", s=150, edgecolors="black", zorder=4)
    ax.set_title("Dados: 6 rótulos e 294 pontos\nsem rótulo", fontsize=9); clean(ax, True)
    for ax, m, t, kind in [(axs[1], svm, "SVM só com os 6 rotulados", "svm"), (axs[2], ts, "TSVM com todos os pontos", "ts")]:
        Z = m.decision_function(G).reshape(xx.shape)
        ax.contourf(xx, yy, Z > 0, levels=[-.5, .5, 1.5], colors=["#FDE68A", "#C7D2FE"], alpha=0.7)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[GRY, INK, GRY], linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
        ax.scatter(X[:, 0], X[:, 1], c=[CL[0] if t_ == 0 else CL[1] for t_ in y], s=8, edgecolors="none", alpha=0.8)
        ax.scatter(X[idx, 0], X[idx, 1], c="white", marker="*", s=130, edgecolors="black", zorder=4)
        pred = (m.predict(X) == 1).astype(int) if kind == "ts" else m.predict(X)
        acc = np.mean((m.predict(X) == (2 * y - 1)) if kind == "ts" else (m.predict(X) == y))
        ax.set_title(f"{t}\nacurácia = {100 * acc:.0f}%", fontsize=9); clean(ax, True)
    for a in axs: a.set_xlim(-1.6, 2.6); a.set_ylim(-1.1, 1.6)
    fig.text(0.5, -0.03, "linha preta = fronteira; tracejadas = margem (f = ±1); azul = classe 1, laranja = classe 0", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f7_luas")

def f7_algoritmo():
    fig, ax = canvas(9.2, 4.6, (0, 100), (0, 54))
    ax.text(50, 51.5, "O TSVM simplificado, passo a passo", ha="center", fontsize=10.8, fontweight="bold")
    pas = [("1. SVM inicial", "treina só com os\nrotulados", IND, 2, 28), ("2. Rótulos provisórios", "dá rótulos aos sem rótulo,\ncom proporção fixa de\npositivos", CYA, 26, 28), ("3. Reforça o peso Cu", "aumenta aos poucos a\nimportância dos sem\nrótulo", AMB, 50, 28), ("4. Troca pares", "troca os rótulos de um par\nque viola a margem, se\nisso reduz a perda", RED, 74, 28)]
    for t, s, c, x, y in pas:
        rbox(ax, x, y, 23, 18, fc="white", ec=c, lw=2, r=2); rbox(ax, x, y + 12.5, 23, 5.5, t, fc=c, tc="white", bold=True, fs=7.6, r=2)
        ax.text(x + 11.5, y + 6, s, ha="center", va="center", fontsize=7.4, linespacing=1.35)
    for x in (25.3, 49.3, 73.3): arrow(ax, (x, 37), (x + 0.9, 37), INK, lw=1.4, ms=9)
    arrow(ax, (85, 27.8), (38, 20), GRN, lw=1.5, ms=10, rad=-0.2)
    ax.text(50, 14, "Repete o passo 4 até não haver trocas; depois volta ao 3 com um Cu maior.", ha="center", fontsize=8.2, fontweight="bold", color=GRN)
    ax.text(50, 6.5, "A proporção fixa de positivos evita a solução trivial \"todos os sem rótulo na mesma classe\".", ha="center", fontsize=8, style="italic", color=GRY)
    save(fig, "f7_algoritmo")

def f7_gamma():
    L = out("ch07_tsvm").splitlines()[2:]
    g, a, b, d = [], [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; g.append(p[0]); a.append(pct(p[1])); b.append(pct(p[2])); d.append(pct(p[3]))
    fig, ax = plt.subplots(figsize=(7.4, 3.8)); x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, a, w, color=IND, label="SVM só com rotulados"); ax.bar(x + w / 2, b, w, color=RED, label="TSVM")
    for i in x: ax.text(x[i] + w / 2, b[i] + 0.6, f"{d[i]:+.1f}".replace(".", ","), ha="center", fontsize=9, fontweight="bold", color=GRN)
    ax.set_xticks(x); ax.set_xticklabels([v.replace(".", ",") for v in g]); ax.set_xlabel("gamma (largura do núcleo RBF)"); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(70, 100)
    ax.set_title("O ganho do TSVM depende do gamma (6 rótulos, 30 sorteios)"); ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    save(fig, "f7_gamma")

def f7_entropia():
    X, y = load_digits(return_X_y=True); par = np.isin(y, [3, 8]); X, y = X[par], (y[par] == 8).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    esc = StandardScaler().fit(X_tr); X_tr = esc.transform(X_tr)
    r = np.random.RandomState(0); idx = np.r_[r.choice(np.where(y_tr == 0)[0], 3, replace=False), r.choice(np.where(y_tr == 1)[0], 3, replace=False)]
    yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]; sem = yp == -1
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.4))
    p = np.linspace(0.001, 0.999, 300); H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    ax = axs[0]; ax.plot(p, H, c=RED, lw=2.6); ax.set_xlabel("prob. prevista da classe 1"); ax.set_ylabel("entropia (bits)"); ax.set_title("Entropia: máxima na dúvida (0,5),\nzero na certeza (0 ou 1)", fontsize=9); ax.grid(alpha=0.25)
    for ax, lam, t in [(axs[1], 0.0, "λ = 0 (sem entropia)"), (axs[2], 0.3, "λ = 0,3 (com entropia)")]:
        w = logistica_entropia(X_tr, yp, lam=lam, l2=1.0); pr = sigmoide(np.c_[X_tr, np.ones(len(X_tr))] @ w)[sem]
        ax.hist(pr, bins=20, range=(0, 1), color=IND if lam == 0 else RED, alpha=0.85); ax.set_xlabel("prob. prevista da classe 1\n(exemplos sem rótulo)"); ax.set_title(t, fontsize=9); ax.set_ylim(0, 140)
    axs[1].set_ylabel("número de exemplos")
    save(fig, "f7_entropia")

def f7_entropia_barras():
    L = out("ch07_entropia").splitlines()[2:]
    lam, acc, d = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; lam.append(p[0]); acc.append(pct(p[1])); d.append(pct(p[2]))
    fig, ax = plt.subplots(figsize=(7.2, 3.7)); x = np.arange(len(lam))
    ax.bar(x, acc, color=[GRY] + [RED] * (len(lam) - 1), width=0.6)
    for i in x: ax.text(x[i], acc[i] + 0.5, f"{acc[i]:.1f}%".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([v.replace(".", ",") for v in lam]); ax.set_xlabel("λ (peso da entropia dos exemplos sem rótulo)"); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(80, 100)
    ax.set_title("Dígitos 3 × 8 com 6 rótulos: um λ moderado é melhor")
    save(fig, "f7_entropia_barras")

def f7_cenarios():
    L = out("ch07_cenarios").splitlines()[1:]
    nomes, sb, so, bal = [], [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0]); sb.append(pct(p[1])); so.append(pct(p[2])); bal.append(pct(p[3].split("%")[0]))
    fig, ax = plt.subplots(figsize=(7.6, 3.9)); x = np.arange(len(nomes)); w = 0.26
    ax.bar(x - w, so, w, color=IND, label="SVM só com rotulados"); ax.bar(x, sb, w, color=GRY, label="TSVM sem fixar a proporção de classes"); ax.bar(x + w, bal, w, color=RED, label="TSVM com proporção fixa")
    for xs, vs in [(x - w, so), (x, sb), (x + w, bal)]:
        for xi, v in zip(xs, vs): ax.text(xi, v + 0.6, f"{v:.1f}".replace(".", ","), ha="center", fontsize=8.3, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(nomes); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(70, 100); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("A restrição de proporção de classes é essencial")
    save(fig, "f7_cenarios")

if __name__ == "__main__":
    for f in [f7_praca, f7_perdas, f7_luas, f7_algoritmo, f7_gamma, f7_entropia, f7_entropia_barras, f7_cenarios]: f()
