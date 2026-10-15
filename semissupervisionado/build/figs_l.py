"""Figuras dos Capítulos 12 a 17."""
import sys, os
from fc import *
from figs_d import out, pct
sys.path.insert(0, "/home/user/claude/semissupervisionado/cap/code")
os.chdir("/home/user/claude/semissupervisionado")

def f14_lsa():
    L = [l for l in out("ch14_lsa").splitlines()[3:] if "|" in l]; nomes, V = [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0].replace("LSA com ", "LSA ").replace(" dimensões", "").replace("TF-IDF (todas as palavras)", "TF-IDF")); V.append([pct(x.split("±")[0]) for x in p[1:]])
    V = np.array(V); fig, ax = plt.subplots(figsize=(8.2, 4.0)); x = np.arange(3); w = 0.13
    cores = [GRY, "#C7D2FE", "#A5B4FC", IND, "#6366F1", "#3730A3"]
    for k in range(len(nomes)): ax.bar(x + (k - 2.5) * w, V[:, k] if False else V[k], w, color=cores[k], label=nomes[k])
    ax.set_xticks(x); ax.set_xticklabels(["40 rótulos", "200 rótulos", "1.000 rótulos"]); ax.set_ylim(60, 92); ax.set_ylabel("F1 macro no teste (%)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left"); ax.grid(axis="y", alpha=0.25)
    ax.set_title("Yelp: representações com e sem o texto sem rótulo (LSA)")
    save(fig, "f14_lsa")

def f12_ideia():
    fig, ax = canvas(9.4, 4.2, (0, 100), (0, 46))
    ax.text(50, 43.5, "Aprendizado auto-supervisionado em duas etapas", ha="center", fontsize=10.8, fontweight="bold")
    rbox(ax, 1, 8, 46, 30, fc="#EEF2FF", ec=IND, lw=2, r=2); rbox(ax, 1, 32, 46, 6, "Etapa 1 — pré-treino (SEM rótulos)", fc=IND, tc="white", bold=True, fs=8.8, r=2)
    ax.text(24, 24, "muitas imagens\nsem rótulo\n⟶  tarefa inventada\n(rotação, contraste...)\n⟶  codificador", ha="center", va="center", fontsize=8.6, linespacing=1.5)
    rbox(ax, 53, 8, 46, 30, fc="#ECFDF5", ec=GRN, lw=2, r=2); rbox(ax, 53, 32, 46, 6, "Etapa 2 — poucos rótulos", fc=GRN, tc="white", bold=True, fs=8.8, r=2)
    ax.text(76, 22.5, "codificador pré-treinado\n⟶  sonda linear (congelado)\nou ajuste fino (toda a rede)\n⟶  classificador final", ha="center", va="center", fontsize=8.6, linespacing=1.5)
    arrow(ax, (47.3, 23), (52.7, 23), INK, lw=2, ms=12)
    ax.text(50, 3, "O pré-treino não usa nenhum rótulo; os rótulos só aparecem na etapa 2.", ha="center", fontsize=8.4, style="italic", color=GRY)
    save(fig, "f12_ideia")

def f12_contraste():
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9)); cores = [IND, AMB, GRN, RED, VIO]
    circ = np.array([[0.2, 0.25], [0.78, 0.22], [0.5, 0.72], [0.15, 0.8], [0.85, 0.75]])
    antes = np.array([[0.7, 0.6], [0.3, 0.7], [0.25, 0.45], [0.8, 0.35], [0.45, 0.2]])
    depois = circ + np.array([[0.06, 0.04], [-0.05, 0.05], [0.05, -0.05], [0.06, -0.03], [-0.06, -0.04]])
    for k, (ax, tri) in enumerate(zip(axs, [antes, depois])):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(MID)
        for c, t, col in zip(circ, tri, cores):
            ax.plot([c[0], t[0]], [c[1], t[1]], c=col, lw=1.3, ls="--" if k == 0 else "-", alpha=0.7)
            ax.scatter(*c, c=col, s=170, marker="o", edgecolors="white", lw=1.5, zorder=3); ax.scatter(*t, c=col, s=170, marker="^", edgecolors="white", lw=1.5, zorder=3)
        ax.set_title(["Antes do treino:\npares (círculo e triângulo) distantes", "Depois do treino:\npares juntos e exemplos diferentes separados"][k], fontsize=9)
    save(fig, "f12_contraste")

def f12_aumento():
    L = [l for l in out("ch12_aumento").splitlines()[2:] if "|" in l]; nomes = ["nenhum", "fraco", "forte"]; v = [pct(l.split("|")[1].split("±")[0]) for l in L]; d = [float(l.split("±")[1]) for l in L]
    fig, ax = plt.subplots(figsize=(6.6, 3.8)); ax.bar(range(3), v, yerr=d, color=[GRY, AMB, RED], width=0.55, capsize=4, ecolor=INK)
    for i, val in enumerate(v): ax.text(i, val + d[i] + 1, f"{val:.1f}%".replace(".", ","), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["sem aumento\n(cópias idênticas)", "aumento fraco", "aumento forte"]); ax.set_ylim(60, 100); ax.set_ylabel("sonda linear, 100 rótulos (%)"); ax.set_title("O aumento define o que a representação aprende"); ax.grid(axis="y", alpha=0.25)
    save(fig, "f12_aumento")

def f12_pretreino():
    L = [l for l in out("ch12_pretreino").splitlines()[2:] if "|" in l]; nomes = ["reg. logística\nsobre pixels", "CNN do zero\n(só rotulados)", "sonda linear\nsem treino", "sonda linear\nrotação", "sonda linear\nSimCLR", "ajuste fino\nSimCLR"]
    v = [pct(l.split("|")[1].split("±")[0]) for l in L]; d = [float(l.split("±")[1]) for l in L]
    fig, ax = plt.subplots(figsize=(8.6, 4.0)); ax.bar(range(len(v)), v, yerr=d, color=[GRY, "#6B7280", MID, AMB, RED, IND], width=0.6, capsize=4, ecolor=INK)
    for i, val in enumerate(v): ax.text(i, val + d[i] + 0.8, f"{val:.1f}%".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(v))); ax.set_xticklabels(nomes, fontsize=8); ax.set_ylim(40, 100); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("MNIST com 100 rótulos: o valor do pré-treino sem rótulo"); ax.grid(axis="y", alpha=0.25)
    save(fig, "f12_pretreino")

def f12_curva():
    L = [l for l in out("ch12_curva").splitlines()[2:] if "|" in l]; n = [int(l.split("|")[0].split()[0]) for l in L]; a = [pct(l.split("|")[1]) for l in L]; b = [pct(l.split("|")[2]) for l in L]
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.plot(n, a, "-o", c=GRY, lw=2.4, label="regressão logística sobre pixels"); ax.plot(n, b, "-s", c=RED, lw=2.6, label="sonda linear sobre o SimCLR")
    for x, y1, y2 in zip(n, a, b): ax.text(x, y2 + 2.2, f"{y2:.1f}".replace(".", ","), ha="center", fontsize=8.5, color=RED, fontweight="bold"); ax.text(x, y1 - 5.5, f"{y1:.1f}".replace(".", ","), ha="center", fontsize=8.5, color="#4B5563")
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n]); ax.set_xlabel("número de rótulos (escala logarítmica)"); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25); ax.set_title("A curva dos rótulos com e sem pré-treino")
    save(fig, "f12_curva")

def f17_decisao():
    fig, ax = canvas(9.4, 5.4, (0, 100), (0, 58))
    ax.text(50, 55.5, "Por onde começar? Um fluxo de decisão (a ser verificado no seu problema)", ha="center", fontsize=10.2, fontweight="bold")
    rbox(ax, 36, 44, 28, 8, "Qual é o tipo de dado?", fc=NAVY, tc="white", bold=True, fs=9, r=2)
    itens = [("Imagens", "aumentos confiáveis?\nUDA / FixMatch;\nmuito dado: pré-treino", IND, 1), ("Texto", "representação com\ntexto sem rótulo (LSA,\nmodelo de linguagem)\n+ modelo linear", CYA, 21), ("Tabela", "referência forte (GBM);\ntestar self-training e\nLabel Spreading", GRN, 41), ("Série", "atributos de janelas;\ndivisão no tempo", AMB, 61), ("Grafo", "Label Propagation;\nredes neurais de grafos", RED, 81)]
    for t, d, c, x in itens:
        rbox(ax, x, 24, 18, 14, "", fc="white", ec=c, lw=2, r=2); rbox(ax, x, 33, 18, 5, t, fc=c, tc="white", bold=True, fs=8.6, r=2); ax.text(x + 9, 28, d, ha="center", va="center", fontsize=6.9, linespacing=1.35)
        arrow(ax, (50, 44), (x + 9, 38.3), GRY, lw=1.2, ms=8)
    rbox(ax, 8, 4, 84, 14, "", fc="#F3F4F6", ec=GRY, lw=1.4, r=2)
    ax.text(50, 15, "Em todos os casos:", ha="center", fontsize=8.8, fontweight="bold"); ax.text(50, 9.6, "1) referência supervisionada forte    2) validação sem usar o teste    3) vários sorteios\n4) métricas por classe e subgrupo    5) relatar os resultados negativos", ha="center", va="center", fontsize=7.8, linespacing=1.5)
    arrow(ax, (50, 24), (50, 18.3), GRY, lw=1.2, ms=8)
    save(fig, "f17_decisao")

def f16_serie():
    L = [l for l in out("ch16_serie").splitlines()[3:] if "|" in l]; R = []
    for l in L:
        p = [x.strip() for x in l.split("|")]; R.append((int(p[0]), "alea" if p[1].startswith("aleat") else "temp", [pct(x.split("±")[0]) for x in p[2:5]]))
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True)
    for ax, n in zip(axs, [30, 150]):
        a = [r for r in R if r[0] == n and r[1] == "alea"][0][2]; t = [r for r in R if r[0] == n and r[1] == "temp"][0][2]; x = np.arange(3)
        ax.bar(x - 0.19, a, 0.38, color=AMB, label="divisão aleatória (vaza)"); ax.bar(x + 0.19, t, 0.38, color=IND, label="divisão temporal (honesta)")
        for i in range(3): ax.text(i - 0.19, a[i] + 0.5, f"{a[i]:.1f}".replace(".", ","), ha="center", fontsize=8); ax.text(i + 0.19, t[i] + 0.5, f"{t[i]:.1f}".replace(".", ","), ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["reg. logística", "1-vizinho", "Label\nSpreading"], fontsize=8.6); ax.set_title(f"{n} rótulos", fontsize=9.6); ax.set_ylim(60, 92); ax.grid(axis="y", alpha=0.25)
    axs[0].set_ylabel("acurácia no teste (%)"); axs[0].legend(frameon=False, fontsize=7.8, loc="upper right")
    save(fig, "f16_serie")

def f15_aumento():
    L = [l for l in out("ch15_aumento").splitlines()[2:] if "|" in l]; nomes = ["nenhum", "deslocar\n8%", "espelhar", "moderado", "moderado\n+ espelhar", "forte\n(Cap. 9)"]
    v = [pct(l.split("|")[1].split("±")[0]) for l in L]; d = [float(l.split("±")[1]) for l in L]
    fig, ax = plt.subplots(figsize=(7.8, 3.9)); ax.bar(range(6), v, yerr=d, color=[GRY, AMB, "#FCD34D", IND, "#818CF8", RED], width=0.6, capsize=4, ecolor=INK)
    for i, val in enumerate(v): ax.text(i, val + d[i] + 0.4, f"{val:.1f}".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(6)); ax.set_xticklabels(nomes, fontsize=8.4); ax.set_ylim(60, 76); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("Fashion-MNIST, 100 rótulos: o aumento de dados não ajudou"); ax.grid(axis="y", alpha=0.25)
    save(fig, "f15_aumento")

def f15_metodos():
    L = [l for l in out("ch15_metodos").splitlines()[2:5] if "|" in l]; V = [[pct(x.split("±")[0]) for x in l.split("|")[1:]] for l in L]; V = np.array(V)
    fig, ax = plt.subplots(figsize=(8.4, 4.0)); x = np.arange(3); w = 0.2; nomes = ["só rotulados", "Pseudo-Label", "FixMatch", "UDA"]
    for k, (c, n) in enumerate(zip([GRY, AMB, IND, RED], nomes)):
        ax.bar(x + (k - 1.5) * w, V[:, k], w, color=c, label=n)
        for i in range(3): ax.text(x[i] + (k - 1.5) * w, V[i, k] + 0.3, f"{V[i, k]:.1f}".replace(".", ","), ha="center", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(["100 rótulos", "300 rótulos", "1.000 rótulos"]); ax.set_ylim(65, 87); ax.set_ylabel("acurácia no teste (%)"); ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left"); ax.grid(axis="y", alpha=0.25); ax.set_title("Fashion-MNIST: quanto os dados sem rótulo ajudam (2 sorteios)")
    save(fig, "f15_metodos")

def f15_categorias():
    L = [l for l in out("ch15_metodos").splitlines() if "%" in l and "|" in l and l.split("|")[0].strip() in ["camiseta", "calça", "pulôver", "vestido", "casaco", "sandália", "camisa", "tênis", "bolsa", "bota"]]
    nomes = [l.split("|")[0].strip() for l in L]; V = np.array([[pct(x) for x in l.split("|")[1:]] for l in L]); fig, ax = plt.subplots(figsize=(8.6, 4.2)); x = np.arange(10); w = 0.2
    for k, (c, n) in enumerate(zip([GRY, AMB, IND, RED], ["só rotulados", "Pseudo-Label", "FixMatch", "UDA"])): ax.bar(x + (k - 1.5) * w, V[:, k], w, color=c, label=n)
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=8, rotation=30); ax.set_ylim(0, 108); ax.set_ylabel("acerto no teste (%)"); ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center"); ax.grid(axis="y", alpha=0.25); ax.set_title("Acerto por categoria com 100 rótulos (um sorteio)")
    save(fig, "f15_categorias")

def f15_amostras():
    from utilidades import carregar_fashion
    X, y, _, _ = carregar_fashion(); nomes = ["camiseta", "calça", "pulôver", "vestido", "casaco", "sandália", "camisa", "tênis", "bolsa", "bota"]
    fig, axs = plt.subplots(2, 10, figsize=(9.4, 2.4))
    for d in range(10):
        ims = np.where(y.numpy() == d)[0][:2]
        for r in range(2): axs[r, d].imshow(X[ims[r], 0].numpy(), cmap="gray_r", vmin=0, vmax=1); axs[r, d].set_xticks([]); axs[r, d].set_yticks([])
        axs[0, d].set_title(nomes[d], fontsize=7.4)
    save(fig, "f15_amostras")

def f18_resultados():
    L = out("ch18_integrador").splitlines(); i = [k for k, l in enumerate(L) if l.startswith("método") and "média  |" in l][0]
    R = []
    for l in L[i + 1:]:
        if "|" not in l: break
        p = [x.strip() for x in l.split("|")]; ic = p[2].strip("[]").split(";"); R.append((p[0], pct(p[1]), pct(ic[0]), pct(ic[1])))
    fig, ax = plt.subplots(figsize=(8.0, 3.9)); R = R[::-1]
    for k, (n, m, lo, hi) in enumerate(R):
        cor = GRN if lo > 0 else (RED if hi < 0 else GRY); ax.plot([lo, hi], [k, k], c=cor, lw=3.2); ax.scatter([m], [k], c=cor, s=60, zorder=3, edgecolors="white")
        ax.text(hi + 0.5 if hi > -5 else hi + 0.5, k + 0.28, f"{m:+.1f}".replace(".", ","), fontsize=8, color=cor, fontweight="bold")
    ax.axvline(0, c=INK, lw=1.2, ls="--"); ax.set_ylim(-0.6, len(R) - 0.2); ax.set_yticks(range(len(R))); ax.set_yticklabels([r[0] for r in R], fontsize=8.6); ax.set_xlabel("diferença para a CNN só com rotulados (pontos percentuais; intervalo de 95%)"); ax.grid(axis="x", alpha=0.25)
    ax.set_title("Projeto integrador: quem supera a referência forte? (MNIST, 100 rótulos)", fontsize=9.6)
    save(fig, "f18_resultados")

def f15_abstencao():
    d = np.load("dados/abstencao_curvas.npz"); fig, ax = plt.subplots(figsize=(7.2, 3.9)); x = np.linspace(1 / len(d["FixMatch"]), 1, len(d["FixMatch"])) * 100
    ax.plot(x, 100 * d["só_rotulados"], c=GRY, lw=2.6, label="só rotulados"); ax.plot(x, 100 * d["FixMatch"], c=IND, lw=2.6, label="FixMatch"); ax.axhline(90, c=RED, ls="--", lw=1.4); ax.text(3, 90.4, "meta: 90% de acerto", color=RED, fontsize=8.5)
    ax.set_xlabel("fração das fotos respondidas (as mais confiantes primeiro), %"); ax.set_ylabel("acurácia nas respondidas (%)"); ax.set_ylim(78, 100.5); ax.invert_xaxis() if False else None; ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8.5, loc="lower left"); ax.set_title("Quanto podemos automatizar? (Fashion-MNIST, 1.000 rótulos)")
    save(fig, "f15_abstencao")

if __name__ == "__main__":
    fs = [f12_ideia, f12_contraste, f14_lsa, f17_decisao, f16_serie, f15_aumento, f15_metodos, f15_categorias, f15_amostras, f18_resultados] + ([f15_abstencao] if os.path.exists('dados/abstencao_curvas.npz') else [])
    for nome, f in [("ch12_aumento", f12_aumento), ("ch12_pretreino", f12_pretreino), ("ch12_curva", f12_curva)]:
        if os.path.exists(f"cap/code/{nome}.out"): fs.append(f)
    for f in fs: f()
