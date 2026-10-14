"""Figuras do Capítulo 6 (modelos generativos e EM)."""
from fc import *
import matplotlib.colors
from figs_d import out, pct
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn

_src = open("/home/user/claude/semissupervisionado/cap/code/ch06_em_do_zero.py").read().split("X, y = gerar(0)")[0]
_g = {}
exec(_src, _g)
em_semi, prever, gerar_al = _g["em_semi"], _g["prever"], _g["gerar"]

def elipse(ax, mu, cov, cor, k=2.0, lw=2.2, ls="-", alpha=1.0):
    w, v = np.linalg.eigh(cov); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang, fc="none", ec=cor, lw=lw, ls=ls, alpha=alpha))

def f6_receitas():
    fig, ax = canvas(9.0, 4.6, (0, 100), (0, 56))
    ax.text(50, 53.5, "A abordagem generativa: cada classe tem uma \"receita\"", ha="center", fontsize=10.8, fontweight="bold")
    rbox(ax, 3, 28, 28, 18, "Receita A\n(distribuição da classe A)\nmédia e dispersão", fc="#EEF2FF", ec=IND, fs=8.2, r=2, lw=2)
    rbox(ax, 36, 28, 28, 18, "Receita B\n(distribuição da classe B)\nmédia e dispersão", fc="#FFFBEB", ec=AMB, fs=8.2, r=2, lw=2)
    rbox(ax, 69, 28, 28, 18, "Proporção\nde cada classe\n(quantos A e quantos B)", fc="#F0FDF4", ec=GRN, fs=8.2, r=2, lw=2)
    ax.text(50, 20, "Para classificar um exemplo novo, pergunta-se: de qual receita ele veio com mais probabilidade? (regra de Bayes)", ha="center", fontsize=8.2, fontweight="bold", color=RED)
    rbox(ax, 4, 4, 44, 11, "Poucos exemplos COM rótulo:\nmostram de que receita vieram.", fc="white", ec=IND, fs=8.2, r=2)
    rbox(ax, 52, 4, 44, 11, "Muitos exemplos SEM rótulo:\nrevelam a forma de cada receita e as proporções.", fc="white", ec=RED, fs=8.2, r=2)
    save(fig, "f6_receitas")

def f6_gen_disc():
    X, y = gerar_al(1, 300); xx, yy = np.meshgrid(np.linspace(-7, 7, 200), np.linspace(-4.5, 4.5, 200)); G = np.c_[xx.ravel(), yy.ravel()]
    from sklearn.linear_model import LogisticRegression
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axs[0]; m = LogisticRegression().fit(X, y)
    ax.contourf(xx, yy, m.predict(G).reshape(xx.shape), levels=[-.5, .5, 1.5], colors=["#C7D2FE", "#FDE68A"], alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=[CL[0] if t == 0 else CL[1] for t in y], s=9, edgecolors="white", lw=0.3); ax.set_title("Discriminativo: aprende só a fronteira\n(p(y | x) direto)", fontsize=9.2); clean(ax, True)
    ax = axs[1]
    for c, col in [(0, IND), (1, AMB)]:
        mu = X[y == c].mean(0); cov = np.cov(X[y == c].T)
        for k in (1, 2, 3): elipse(ax, mu, cov, col, k=k, lw=1.6 if k == 2 else 1, alpha=0.9 if k == 2 else 0.5)
        ax.scatter(*mu, c=col, marker="x", s=60)
    ax.scatter(X[:, 0], X[:, 1], c=[CL[0] if t == 0 else CL[1] for t in y], s=9, edgecolors="white", lw=0.3, alpha=0.8)
    ax.set_title("Generativo: aprende como cada classe gera\nos dados (p(x | y) e p(y)); a fronteira vem daí", fontsize=9.2); clean(ax, True)
    for a in axs: a.set_xlim(-7, 7); a.set_ylim(-4.5, 4.5)
    save(fig, "f6_gen_disc")

def f6_em():
    X, y = gerar_al(0); r = np.random.RandomState(0); idx = np.r_[r.choice(np.where(y == 0)[0], 3, replace=False), r.choice(np.where(y == 1)[0], 3, replace=False)]
    yp = np.full(len(y), -1); yp[idx] = y[idx]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, it in zip(axs, [0, 1, 3, 15]):
        m = em_semi(X, yp, 2, iters=it, reg=0.05)
        logp = np.column_stack([np.log(m["pi"][k]) + mvn.logpdf(X, m["mu"][k], m["cov"][k]) for k in range(2)]); p1 = np.exp(logp[:, 1] - np.logaddexp(logp[:, 0], logp[:, 1]))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", [IND, "#E5E7EB", AMB])
        com = yp != -1
        ax.scatter(X[~com, 0], X[~com, 1], c=p1[~com], cmap=cmap, vmin=0, vmax=1, s=9, edgecolors="none")
        ax.scatter(X[com, 0], X[com, 1], c=[CL[0] if t == 0 else CL[1] for t in yp[com]], s=90, marker="*", edgecolors="black", lw=0.8, zorder=4)
        for k, col in enumerate([IND, AMB]): elipse(ax, m["mu"][k], m["cov"][k], col, k=2)
        ax.set_title({0: "Início: só os 6 rotulados", 1: "Iteração 1", 3: "Iteração 3", 15: "Iteração 15"}[it] + f"\nacurácia = {100 * np.mean(prever(m, X) == y):.0f}%", fontsize=9)
        clean(ax, True); ax.set_xlim(-8, 8); ax.set_ylim(-4.5, 4.5)
    fig.text(0.5, -0.03, "cor dos pontos = probabilidade de ser da classe laranja (azul 0, cinza 0,5, laranja 1); elipses = gaussianas estimadas (2 desvios)", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f6_em")
    mm = em_semi(X, yp, 2, iters=15, reg=1e-3)
    fig, ax = plt.subplots(figsize=(6.2, 3.3)); ax.plot(range(len(mm["trilha"])), mm["trilha"], "-o", c=RED, lw=2, ms=4)
    ax.set_xlabel("iteração do EM"); ax.set_ylabel("log-verossimilhança"); ax.set_title("O EM só sobe (e para em um máximo local)"); ax.grid(alpha=0.25)
    save(fig, "f6_ll")

def f6_cruzado():
    from sklearn.model_selection import train_test_split
    def cruzado(semente, n=600):
        rr = np.random.RandomState(semente); g = rr.randint(0, 2, n); x = rr.normal(0, 1.0, n)
        return np.c_[x, np.where(g == 0, -3, 3) + rr.normal(0, 0.6, n)], (x > 0).astype(int)
    g2 = {"cruzado": cruzado, "em_semi": em_semi, "prever": prever}
    X, y = g2["cruzado"](3); X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=3)
    r = np.random.RandomState(3); idx = np.r_[r.choice(np.where(y_tr == 0)[0], 5, replace=False), r.choice(np.where(y_tr == 1)[0], 5, replace=False)]
    yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5))
    ax = axs[0]; ax.scatter(X_tr[:, 0], X_tr[:, 1], c=[CL[0] if t == 0 else CL[1] for t in y_tr], s=8, edgecolors="none"); ax.axvline(0, c=INK, ls="--", lw=1)
    ax.scatter(X_tr[idx, 0], X_tr[idx, 1], s=80, marker="*", c="white", edgecolors="black", zorder=4); ax.set_title("Dados: grupos em cima e embaixo;\nclasses: esquerda (azul) e direita (laranja)", fontsize=8.8); clean(ax, True)
    for ax, it, t in [(axs[1], 0, "Gaussianas com só os 10 rotulados"), (axs[2], 30, "Depois do EM com todos os dados")]:
        m = g2["em_semi"](X_tr, yp, 2, iters=it, reg=0.05)
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=[CL[0] if tt == 0 else CL[1] for tt in y_tr], s=8, edgecolors="none", alpha=0.5)
        for k, col in enumerate([IND, AMB]): elipse(ax, m["mu"][k], m["cov"][k], col, k=2, lw=2.6)
        ax.set_title(t + f"\nacurácia no teste = {100 * np.mean(g2['prever'](m, X_te) == y_te):.0f}%", fontsize=8.8); clean(ax, True)
    for a in axs: a.set_xlim(-4.5, 4.5); a.set_ylim(-6, 6)
    save(fig, "f6_cruzado")

def f6_mal():
    L = out("ch06_mal_especificado").splitlines()[1:]
    d = {"alinhado": ([], []), "cruzado": ([], [])}
    for l in L:
        p = [x.strip() for x in l.split("|")]; d[p[0].strip()][0].append(float(p[1])); d[p[0].strip()][1].append(float(p[3]))
    fig, ax = plt.subplots(figsize=(7.4, 3.8)); x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, d["alinhado"][1], w, color=GRN, label="classe = grupo (suposição vale)"); ax.bar(x + w / 2, d["cruzado"][1], w, color=RED, label="classe ≠ grupo (suposição falha)")
    for i in range(4):
        for off, k, c in [(-w / 2, "alinhado", GRN), (w / 2, "cruzado", RED)]:
            v = d[k][1][i]; ax.text(x[i] + off, v + (0.4 if v >= 0 else -1.4), f"{v:+.1f}".replace(".", ","), ha="center", fontsize=8.3, fontweight="bold")
    ax.axhline(0, c=INK, lw=1); ax.set_xticks(x); ax.set_xticklabels([str(p).replace(".", ",") for p in d["alinhado"][0]]); ax.set_xlabel("peso dos exemplos sem rótulo no EM"); ax.set_ylabel("ganho sobre só rotulados (p.p.)")
    ax.set_title("O mesmo EM: ganha quando a suposição vale, perde quando falha"); ax.legend(frameon=False, fontsize=8.3, loc="lower left"); ax.set_ylim(-18, 9)
    save(fig, "f6_mal")

def f6_nb():
    L = out("ch06_nb_em").splitlines()[1:]
    nomes, v, dv = [], [], []
    for l in L:
        a, b = l.split(":"); m, d = b.split("±"); nomes.append(a.strip()); v.append(pct(m)); dv.append(float(d))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("Naive Bayes + EM em texto")
    pas = [("1. Treina o NB com os poucos rotulados", IND), ("2. E: dá a cada mensagem sem rótulo uma\nprobabilidade de spam", CYA), ("3. M: reestima o NB, contando cada mensagem\ncom peso = λ × probabilidade", AMB), ("4. Repete até estabilizar", GRN)]
    for i, (t, c) in enumerate(pas):
        rbox(ax, 0.2, 7.4 - i * 2.4, 9.6, 1.9, t, fc="white", ec=c, fs=7.6, r=0.3, lw=1.8)
        if i < 3: arrow(ax, (5, 7.4 - i * 2.4), (5, 7.4 - (i + 1) * 2.4 + 1.9), GRY, lw=1.2, ms=9)
    ax = axs[1]; y_ = np.arange(len(nomes))[::-1]
    cores = [GRY, GRY, RED, RED, RED, RED]
    ax.barh(y_, v, xerr=dv, color=cores, height=0.6, capsize=3, ecolor=INK)
    for yy, val in zip(y_, v): ax.text(119, yy, f"{val:.1f}%".replace(".", ","), va="center", ha="right", fontsize=8.6, fontweight="bold")
    ax.set_yticks(y_); ax.set_yticklabels(nomes, fontsize=8.2); ax.set_xlim(0, 120); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xlabel("F1 da classe spam (%)"); ax.set_title("SMS Spam, 40 rótulos"); ax.grid(axis="x", alpha=0.25)
    save(fig, "f6_nb")

def f6_cov():
    r = np.random.RandomState(2)
    X = np.vstack([r.multivariate_normal([-2, -1], [[2.0, 1.2], [1.2, 1.0]], 150), r.multivariate_normal([2, 1.5], [[1.4, -0.9], [-0.9, 1.2]], 150)])
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, ct, t in zip(axs, ["spherical", "diag", "tied", "full"], ["spherical\n(1 número por classe)", "diag\n(eixos alinhados)", "tied\n(forma comum às classes)", "full\n(forma livre por classe)"]):
        g = GaussianMixture(2, covariance_type=ct, random_state=0, means_init=[[-2, -1], [2, 1.5]]).fit(X)
        ax.scatter(X[:, 0], X[:, 1], c=GRY, s=7, edgecolors="none", alpha=0.6)
        for k, col in enumerate([IND, AMB]):
            cov = np.eye(2) * g.covariances_[k] if ct == "spherical" else np.diag(g.covariances_[k]) if ct == "diag" else g.covariances_ if ct == "tied" else g.covariances_[k]
            elipse(ax, g.means_[k], cov, col, k=2, lw=2.4)
        ax.set_title(t, fontsize=8.8); clean(ax, True); ax.set_xlim(-7, 6.5); ax.set_ylim(-5, 6)
    save(fig, "f6_cov")

def f6_digits():
    L = out("ch06_digits_gmm").splitlines()[1:]
    nomes, v, dv = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0]); m, d = p[1].split("±"); v.append(pct(m)); dv.append(float(d))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    ax.set_title("\"Cluster-e-rotula\" com mistura de gaussianas")
    passos = [("64\npixels", GRY), ("PCA\n15 comp.", CYA), ("GMM com\n10 comp.", IND), ("rotula cada\ncomponente\npelo voto", GRN)]
    for i, (t, c) in enumerate(passos):
        rbox(ax, 1 + i * 25, 18, 20, 18, t, fc="white", ec=c, fs=7.4, r=1.5, lw=2)
        if i < 3: arrow(ax, (21.3 + i * 25, 27), (25.5 + i * 25, 27), INK, lw=1.4, ms=9)
    ax.text(50, 8, "médias iniciais = médias dos rotulados; o GMM usa TODOS os exemplos", ha="center", fontsize=8, style="italic", color=RED)
    ax = axs[1]; y_ = np.arange(len(nomes))[::-1]
    ax.barh(y_, v, xerr=dv, color=[GRY, GRY, RED], height=0.55, capsize=3, ecolor=INK)
    for yy, val in zip(y_, v): ax.text(99, yy, f"{val:.1f}%".replace(".", ","), va="center", ha="right", fontsize=8.6, fontweight="bold")
    ax.set_yticks(y_); ax.set_yticklabels([n.replace(" (", "\n(") for n in nomes], fontsize=7.8); ax.set_xlim(60, 100); ax.set_xlabel("acurácia no teste (%)"); ax.set_title("Digits, 50 rótulos"); ax.grid(axis="x", alpha=0.25)
    save(fig, "f6_digits")

if __name__ == "__main__":
    for f in [f6_receitas, f6_gen_disc, f6_em, f6_cruzado, f6_mal, f6_nb, f6_cov, f6_digits]: f()
