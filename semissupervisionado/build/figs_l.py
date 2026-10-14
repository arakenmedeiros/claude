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

if __name__ == "__main__":
    for f in [f12_ideia, f12_contraste, f14_lsa]: f()
