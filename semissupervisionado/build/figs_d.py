"""Figuras do Capítulo 4 (self-training). Números lidos das saídas reais em cap/code/*.out."""
import re
from fc import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

CODE = "/home/user/claude/semissupervisionado/cap/code/"
def out(nome): return open(CODE + nome + ".out").read()
def pct(s): return float(s.replace("%", "").replace(",", ".").replace("+", "").strip())

def f4_ciclo():
    fig, ax = canvas(8.8, 4.8, (0, 100), (0, 56))
    ax.text(50, 53.5, "O ciclo do self-training", ha="center", fontsize=11, fontweight="bold")
    pos = [(8, 30, "1. TREINAR", "com os exemplos\nque já têm rótulo", IND), (60, 30, "2. PREVER", "dar um palpite e uma\nconfiança aos sem rótulo", CYA),
           (60, 6, "3. SELECIONAR", "ficar só com os palpites\nmuito confiantes", AMB), (8, 6, "4. INCORPORAR", "transformar em pseudo-rótulos\ne juntar ao treino", GRN)]
    for x, y, t, s, c in pos:
        rbox(ax, x, y, 32, 18, fc="white", ec=c, lw=2.2, r=2.2)
        rbox(ax, x, y + 12, 32, 6, t, fc=c, tc="white", bold=True, fs=9.5, r=2.2)
        ax.text(x + 16, y + 5.5, s, ha="center", va="center", fontsize=8.2, linespacing=1.35)
    arrow(ax, (40.5, 40), (59.5, 40), INK, lw=2, ms=14); arrow(ax, (76, 29.5), (76, 24.5), INK, lw=2, ms=14)
    arrow(ax, (59.5, 15), (40.5, 15), INK, lw=2, ms=14); arrow(ax, (24, 24.5), (24, 29.5), INK, lw=2, ms=14)
    ax.text(50, 27.5, "repete até\nnão haver mais\nconfiança", ha="center", va="center", fontsize=8.2, style="italic", color=RED, fontweight="bold")
    save(fig, "f4_ciclo")

def _dados_blobs(s, n=400):
    r = np.random.RandomState(s); y = r.randint(0, 2, n)
    X = r.normal(0, 1, (n, 2)); X[:, 0] *= 2.2; X[:, 1] += np.where(y == 0, -2.0, 2.0)
    return X, y

def f4_rodadas():
    ganhos = []
    for s in range(40):
        X, y = _dados_blobs(s); r = np.random.RandomState(100 + s)
        idx = np.r_[r.choice(np.where(y == 0)[0], 3, replace=False), r.choice(np.where(y == 1)[0], 3, replace=False)]
        yp = np.full(len(y), -1); yp[idx] = y[idx]
        a = LogisticRegression().fit(X[idx], y[idx]).score(X, y)
        b = SelfTrainingClassifier(LogisticRegression(), threshold=0.9).fit(X, yp).score(X, y)
        ganhos.append((b - a, s))
    ganhos.sort(); s = ganhos[len(ganhos) // 2][1]    # semente de ganho mediano
    X, y = _dados_blobs(s); r = np.random.RandomState(100 + s)
    idx = np.r_[r.choice(np.where(y == 0)[0], 3, replace=False), r.choice(np.where(y == 1)[0], 3, replace=False)]
    xx, yy = np.meshgrid(np.linspace(-8, 8, 200), np.linspace(-5, 5, 200)); G = np.c_[xx.ravel(), yy.ravel()]
    fig, axs = plt.subplots(1, 4, figsize=(10.6, 3.1)); lab = np.full(len(y), -1); lab[idx] = y[idx]
    for k, ax in enumerate(axs):
        com = lab != -1
        m = LogisticRegression().fit(X[com], lab[com])
        ax.contourf(xx, yy, m.predict(G).reshape(xx.shape), levels=[-.5, .5, 1.5], colors=["#C7D2FE", "#FDE68A"], alpha=0.75)
        ax.scatter(X[~com, 0], X[~com, 1], c=GRY, s=7, edgecolors="none", alpha=0.7)
        ax.scatter(X[com, 0], X[com, 1], c=[CL[0] if t == 0 else CL[1] for t in lab[com]], s=22 if k else 90, marker="o" if k else "*", edgecolors="black" if k == 0 else "white", lw=0.5)
        ax.set_title(["Rodada 0: 6 rótulos", "Rodada 1", "Rodada 2", "Rodada 3"][k] + f"\nacurácia = {100 * m.score(X, y):.0f}%", fontsize=9); clean(ax, True)
        ax.set_xlim(-8, 8); ax.set_ylim(-5, 5)
        prob = m.predict_proba(X[~com]); conf = prob.max(1); sem = np.where(~com)[0]; acc = sem[conf >= 0.9]
        lab[acc] = m.classes_[prob[conf >= 0.9].argmax(1)]
    fig.text(0.5, -0.02, "pontos coloridos = com rótulo ou pseudo-rótulo; cinza = ainda sem rótulo (semente de ganho mediano em 40 sorteios)", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f4_rodadas")

def f4_limiar():
    L = [l for l in out("ch04_limiar").splitlines()[1:]]
    lim, n1, ac, fim = [], [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; lim.append(float(p[0])); n1.append(float(p[1])); ac.append(pct(p[2])); fim.append(pct(p[3]))
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axs[0]; ax.bar(range(len(lim)), n1, color=CYA, width=0.6); ax.set_xticks(range(len(lim))); ax.set_xticklabels([f"{x:.2f}".replace(".", ",") for x in lim])
    for i, v in enumerate(n1): ax.text(i, v + 15, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xlabel("limiar de confiança"); ax.set_ylabel("pseudo-rótulos na 1ª rodada"); ax.set_title("Quanto mais exigente, menos exemplos entram")
    ax = axs[1]; ax.plot(lim, ac, "-o", c=RED, lw=2.2, label="acerto dos pseudo-rótulos (1ª rodada)"); ax.plot(lim, fim, "-s", c=IND, lw=2.2, label="acurácia final no teste")
    ax.set_xlabel("limiar de confiança"); ax.set_ylabel("%"); ax.set_title("Exigência alta: rótulos mais limpos"); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(alpha=0.25); ax.set_ylim(84, 101)
    save(fig, "f4_limiar")

def f4_iter():
    rows = [re.findall(r"rodada\s+(\d+): \+\s*(\d+) pseudo-rótulos \| acerto deles:\s*([\d.]+)% \| teste:\s*([\d.]+)%", l) for l in out("ch04_manual").splitlines()]
    rows = [r[0] for r in rows if r]
    it = [int(r[0]) for r in rows]; n = [int(r[1]) for r in rows]; ac = [float(r[2]) for r in rows]; te = [float(r[3]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 3.9)); ax2 = ax.twinx()
    ax.bar(it, n, color=CYA, alpha=0.8, width=0.6, label="pseudo-rótulos adicionados"); ax.set_yscale("log"); ax.set_ylabel("exemplos adicionados (escala log)"); ax.set_xlabel("rodada")
    ax2.plot(it, ac, "-o", c=RED, lw=2.2, label="acerto dos pseudo-rótulos"); ax2.plot(it, te, "-s", c=IND, lw=2.2, label="acurácia no teste"); ax2.set_ylim(-3, 103); ax2.set_ylabel("%"); ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("As primeiras rodadas ajudam; as últimas trazem poucos e menos confiáveis rótulos"); ax.set_xticks(it)
    save(fig, "f4_iter")

def f4_base():
    L = out("ch04_base").splitlines()[1:]
    nomes, a, b = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0]); a.append(pct(p[1])); b.append(pct(p[2]))
    fig, ax = plt.subplots(figsize=(9, 3.9)); x = np.arange(len(nomes)); w = 0.38
    ax.bar(x - w / 2, a, w, color=IND, label="só rotulados"); ax.bar(x + w / 2, b, w, color=AMB, label="self-training")
    for i in range(len(x)):
        d = b[i] - a[i]; ax.text(x[i] + w / 2, b[i] + 1, f"{d:+.1f}".replace(".", ","), ha="center", fontsize=8.5, fontweight="bold", color=GRN if d > 0.05 else RED if d < -0.05 else INK)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in nomes], fontsize=8); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(0, 112)
    ax.set_title("O mesmo self-training, cinco classificadores-base (50 rótulos, Digits)"); ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.text(0.01, 0.97, "número sobre a barra = ganho em p.p.", transform=ax.transAxes, fontsize=7.8, style="italic", color=GRY, va="top")
    save(fig, "f4_base")

def f4_calibra():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    r = np.random.RandomState(0); idx = np.concatenate([r.choice(np.where(ytr == d)[0], 5, replace=False) for d in range(10)])
    sem = np.setdiff1d(np.arange(len(ytr)), idx)
    modelos = [("Regressão logística", LogisticRegression(max_iter=2000)), ("k vizinhos (k=5)", KNeighborsClassifier(5)), ("Floresta aleatória", RandomForestClassifier(100, random_state=0))]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    bins = np.array([0, .4, .6, .8, .9, 1.0001])
    for ax, (nome, m) in zip(axs, modelos):
        m.fit(Xtr[idx], ytr[idx]); p = m.predict_proba(Xtr[sem]); conf = p.max(1); ok = (m.classes_[p.argmax(1)] == ytr[sem])
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = (conf >= lo) & (conf < hi)
            if k.sum() >= 5: xs.append(conf[k].mean()); ys.append(ok[k].mean()); ns.append(k.sum())
        ax.plot([0, 1], [0, 1], c=GRY, ls="--", lw=1.2); ax.plot(xs, ys, "-o", c=RED, lw=2.2)
        for xv, yv, nv in zip(xs, ys, ns): ax.text(xv, yv - 0.08, str(nv), ha="center", fontsize=7, color=INK)
        sel = conf >= 0.9; ax.set_title(f"{nome}\n≥0,9: {sel.sum()} ex., {100 * ok[sel].mean() if sel.any() else 0:.0f}% certos", fontsize=9)
        ax.set_xlabel("confiança média do modelo"); ax.set_xlim(0.2, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=0.25)
    axs[0].set_ylabel("fração de acertos")
    fig.text(0.5, -0.03, "tracejado = confiança perfeita (confiança 90% → 90% de acerto); números = exemplos em cada faixa", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f4_calibra")

def f4_uci():
    L = out("ch04_uci").splitlines()[1:]
    nomes, d = [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0]); d.append(pct(p[6]))
    R = out("ch04_ruido").splitlines()[1:]
    rn, rd = [], []
    for l in R:
        p = [x.strip() for x in l.split("|")]; rn.append(p[0]); rd.append(pct(p[3]))
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.6))
    ax = axs[0]; ax.bar(range(len(d)), d, color=[GRN if v > 0 else RED for v in d], width=0.55)
    for i, v in enumerate(d): ax.text(i, v + (0.15 if v > 0 else -0.45), f"{v:+.1f}".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, c=INK, lw=1); ax.set_xticks(range(len(d))); ax.set_xticklabels([n.replace(" (", "\n(") for n in nomes], fontsize=8); ax.set_ylabel("ganho (p.p.)"); ax.set_ylim(-3, 3); ax.set_title("Três bases da UCI, 20 rótulos")
    ax = axs[1]; ax.bar(range(len(rd)), rd, color=AMB, width=0.55)
    for i, v in enumerate(rd): ax.text(i, v + 0.15, f"{v:+.1f}".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(rn))); ax.set_xticklabels(rn); ax.set_xlabel("proporção de rótulos iniciais errados"); ax.set_ylabel("ganho (p.p.)"); ax.set_title("Digits, 100 rótulos, com ruído"); ax.set_ylim(0, 6)
    save(fig, "f4_uci")

def f4_yarowsky():
    fig, ax = canvas(9.0, 4.6, (0, 100), (0, 56))
    ax.text(50, 53.5, "A ideia de Yarowsky: desambiguar a palavra \"planta\"", ha="center", fontsize=10.5, fontweight="bold")
    rbox(ax, 2, 33, 30, 15, "", fc="#EEF2FF", ec=IND, r=2, lw=1.8)
    ax.text(17, 45, "SEMENTES (poucas regras\nrotuladas à mão)", ha="center", va="center", fontsize=8, fontweight="bold", color=IND)
    ax.text(17, 37.2, "“fábrica” → sentido 1 (indústria)\n“folha” → sentido 2 (vegetal)", ha="center", fontsize=7.6, va="center", linespacing=1.4)
    sents = [("A planta da fábrica emprega 300 pessoas.", IND, 1), ("A planta precisa de sol e de água.", GRN, 2), ("A planta fecha as máquinas à noite.", GRY, 0), ("A planta cresceu depois da chuva.", GRY, 0)]
    for i, (t, c, k) in enumerate(sents):
        y = 44 - i * 5.2
        rbox(ax, 38, y, 58, 4.4, t, fc="white", ec=c if k else MID, fs=7.8, r=1, lw=1.6 if k else 1, ha="left")
        if k: ax.text(94.5, y + 2.2, ["", "sentido 1", "sentido 2"][k], fontsize=7, ha="right", va="center", color=c, fontweight="bold")
    arrow(ax, (32.5, 40), (37.5, 40), INK, lw=1.6, ms=10)
    ax.text(50, 24, "Rodada 2: o modelo descobre novas pistas nas frases já rotuladas e rotula as demais", ha="center", fontsize=8.2, style="italic", color=RED, fontweight="bold")
    for i, (t, c) in enumerate([("“máquinas” aparece com sentido 1", IND), ("“chuva” aparece com sentido 2", GRN)]):
        rbox(ax, 8 + i * 45, 12, 38, 7, t, fc="#FFFFFF", ec=c, fs=7.8, r=1.2, lw=1.8)
    arrow(ax, (27, 22.5), (27, 19.3), RED, lw=1.5, ms=9); arrow(ax, (72, 22.5), (72, 19.3), RED, lw=1.5, ms=9)
    ax.text(50, 6, "Exemplo ilustrativo. Yarowsky (1995) aplicou o princípio a sentidos de palavras em inglês e a textos grandes.", ha="center", fontsize=7.4, color=GRY, style="italic")
    save(fig, "f4_yarowsky")

if __name__ == "__main__":
    for f in [f4_ciclo, f4_rodadas, f4_limiar, f4_iter, f4_base, f4_calibra, f4_uci, f4_yarowsky]: f()
