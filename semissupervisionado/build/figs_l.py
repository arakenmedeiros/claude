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

if __name__ == "__main__":
    fs = [f12_ideia, f12_contraste, f14_lsa]
    for nome, f in [("ch12_aumento", f12_aumento), ("ch12_pretreino", f12_pretreino), ("ch12_curva", f12_curva)]:
        if os.path.exists(f"cap/code/{nome}.out"): fs.append(f)
    for f in fs: f()
