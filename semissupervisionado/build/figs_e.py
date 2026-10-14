"""Figuras do Capítulo 5 (co-training). Números lidos de cap/code/*.out."""
from fc import *
import matplotlib.colors
from figs_d import out, pct
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

def f5_ideia():
    fig, ax = canvas(9.0, 4.8, (0, 100), (0, 56))
    ax.text(50, 53.5, "Co-training: dois especialistas, duas visões do mesmo caso", ha="center", fontsize=10.8, fontweight="bold")
    rbox(ax, 36, 38, 28, 10, "Um mesmo paciente\n(exemplo sem rótulo)", fc="white", ec=INK, fs=8.5, r=2, lw=1.6)
    rbox(ax, 4, 20, 34, 12, "Visão 1: radiografia\n→ Médico A (classificador 1)", fc="#EEF2FF", ec=IND, fs=8.3, r=2, lw=2)
    rbox(ax, 62, 20, 34, 12, "Visão 2: exame de sangue\n→ Médico B (classificador 2)", fc="#FFFBEB", ec=AMB, fs=8.3, r=2, lw=2)
    arrow(ax, (42, 38), (22, 32.3), IND, lw=1.6, ms=11); arrow(ax, (58, 38), (78, 32.3), AMB, lw=1.6, ms=11)
    arrow(ax, (38.5, 24), (61.5, 24), RED, lw=2.2, ms=14, rad=-0.35); arrow(ax, (61.5, 28), (38.5, 28), RED, lw=2.2, ms=14, rad=-0.35)
    ax.text(50, 24, "ensinam\num ao outro", ha="center", va="center", fontsize=8.2, color=RED, fontweight="bold")
    rbox(ax, 8, 4, 84, 10, "Quando A tem certeza e B não, A rotula o caso e B aprende com ele (e vice-versa).\nO erro de um é corrigido pelo outro, se as visões forem diferentes o bastante.", fc="#F8FAFC", ec=MID, fs=8.2, r=2)
    save(fig, "f5_ideia")

def f5_condicoes():
    rng = np.random.default_rng(3); n = 260
    y = rng.integers(0, 2, n); s = np.where(y == 1, 1.0, -1.0) * 0.9
    V1 = s[:, None] + rng.normal(0, 1, (n, 2)); V2 = s[:, None] + rng.normal(0, 1, (n, 2))
    m1 = LogisticRegression().fit(V1, y); m2 = LogisticRegression().fit(V2, y)
    p1 = m1.predict_proba(V1)[:, 1]; p2 = m2.predict_proba(V2)[:, 1]
    amb1 = (np.abs(p1 - 0.5) < 0.12) & (np.abs(p2 - 0.5) > 0.3); amb2 = (np.abs(p2 - 0.5) < 0.12) & (np.abs(p1 - 0.5) > 0.3)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, V, amb, t in [(axs[0], V1, amb1, "Visão 1"), (axs[1], V2, amb2, "Visão 2")]:
        ax.scatter(V[:, 0], V[:, 1], c=[CL[0] if k == 0 else CL[1] for k in y], s=14, edgecolors="white", lw=0.3, alpha=0.9)
        ax.scatter(V[amb, 0], V[amb, 1], s=150, facecolors="none", edgecolors=RED, lw=2)
        ax.set_title(f"{t}: o círculo vermelho marca casos\ndifíceis aqui, mas claros na outra visão", fontsize=8.8); clean(ax, True)
    fig.text(0.5, -0.02, "Condições do co-training: (1) cada visão basta, sozinha, para classificar bem; (2) as visões são (quase) independentes dada a classe.", ha="center", fontsize=8, style="italic", color=GRY)
    save(fig, "f5_condicoes")

def f5_algoritmo():
    fig, ax = canvas(9.2, 4.6, (0, 100), (0, 54))
    ax.text(50, 51.5, "Uma rodada de co-training", ha="center", fontsize=10.8, fontweight="bold")
    rbox(ax, 2, 20, 18, 14, "Conjunto\nrotulado L\n(compartilhado)", fc="#EEF2FF", ec=IND, fs=8, r=2, lw=2)
    rbox(ax, 28, 33, 24, 10, "Treina h1 com a\nvisão 1 de L", fc=IND, tc="white", fs=7.8, bold=True, r=1.6)
    rbox(ax, 28, 11, 24, 10, "Treina h2 com a\nvisão 2 de L", fc=AMB, tc="white", fs=7.8, bold=True, r=1.6)
    rbox(ax, 60, 33, 24, 10, "h1 escolhe os mais\nconfiantes por classe", fc="white", ec=IND, fs=7.6, r=1.6, lw=1.8)
    rbox(ax, 60, 11, 24, 10, "h2 escolhe os mais\nconfiantes por classe", fc="white", ec=AMB, fs=7.6, r=1.6, lw=1.8)
    rbox(ax, 88, 20, 11, 14, "Sem\nrótulo\n(U)", fc=LIGHT, ec=GRY, fs=8, r=2)
    arrow(ax, (20.2, 30), (27.8, 38), INK, lw=1.4, ms=10); arrow(ax, (20.2, 24), (27.8, 16), INK, lw=1.4, ms=10)
    arrow(ax, (52.2, 38), (59.8, 38), INK, lw=1.4, ms=10); arrow(ax, (52.2, 16), (59.8, 16), INK, lw=1.4, ms=10)
    arrow(ax, (87.8, 30), (84.3, 38), GRY, lw=1.2, ms=9); arrow(ax, (87.8, 24), (84.3, 16), GRY, lw=1.2, ms=9)
    arrow(ax, (72, 43.3), (11, 34.3), GRN, lw=1.6, ms=11, rad=0.25); arrow(ax, (72, 10.7), (11, 19.7), GRN, lw=1.6, ms=11, rad=-0.25)
    ax.text(50, 1.2, "Os pseudo-rótulos de h1 e de h2 entram no mesmo conjunto L. Repete-se por algumas rodadas.", ha="center", fontsize=8.2, style="italic", color=GRN, fontweight="bold")
    save(fig, "f5_algoritmo")

def f5_digits():
    d = load_digits(); img = d.images[5] / 16.0
    perm = np.random.RandomState(0).permutation(64); va = np.zeros(64, bool); va[perm[:32]] = True
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.4))
    axs[0].imshow(img, cmap="gray_r"); axs[0].set_title("O dígito completo (8×8)"); clean(axs[0], True)
    top = np.zeros((8, 8), bool); top[:4] = True
    ca = np.array(matplotlib.colors.to_rgb(IND)); cb = np.array(matplotlib.colors.to_rgb(AMB))
    for ax, mask, t in [(axs[1], top, "Visões: 4 linhas de cima (azul)\ne 4 linhas de baixo (laranja)"), (axs[2], va.reshape(8, 8), "Visões: 32 pixels sorteados (azul)\ne os outros 32 (laranja)")]:
        rgb = np.where(mask[..., None], ca, cb)
        rgb = 1 - (1 - rgb) * 0.55
        rgb = rgb * (1 - 0.85 * img[..., None])
        ax.imshow(rgb); ax.set_title(t, fontsize=9); clean(ax, True)
    save(fig, "f5_digits")

def f5_resultados():
    L = out("ch05_visoes").splitlines()[1:]
    nomes, v, dv = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0]); m, d = p[1].split("±"); v.append(pct(m)); dv.append(float(d))
    cores = [GRY, GRY, IND, IND, AMB, RED, IND, RED]
    fig, ax = plt.subplots(figsize=(9.4, 4.4)); y_ = np.arange(len(nomes))[::-1]
    ax.barh(y_, v, xerr=dv, color=cores, height=0.62, capsize=3, ecolor=INK)
    for yy, val in zip(y_, v): ax.text(101, yy, f"{val:.1f}%".replace(".", ","), va="center", ha="right", fontsize=8.8, fontweight="bold")
    ax.set_yticks(y_); ax.set_yticklabels(nomes, fontsize=8.5); ax.set_xlim(50, 102); ax.set_xlabel("acurácia no teste (%)")
    ax.set_title("Digits com 50 rótulos: o co-training não superou o self-training"); ax.grid(axis="x", alpha=0.25)
    ax.text(1.0, -0.2, "vermelho = co-training; azul = referências com as mesmas visões", transform=ax.transAxes, ha="right", fontsize=7.6, style="italic", color=GRY)
    save(fig, "f5_resultados")

def f5_indep():
    L = out("ch05_independencia").splitlines()[1:]
    rho, a, b = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; rho.append(float(p[0])); a.append(pct(p[1])); b.append(pct(p[2]))
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); x = np.arange(len(rho)); w = 0.38
    ax.bar(x - w / 2, a, w, color=IND, label="só rotulados (as duas visões)"); ax.bar(x + w / 2, b, w, color=RED, label="co-training")
    for i in x: ax.text(x[i] + w / 2, b[i] + 0.6, f"{b[i] - a[i]:+.1f}".replace(".", ","), ha="center", fontsize=9, fontweight="bold", color=GRN)
    ax.set_xticks(x); ax.set_xticklabels([f"{r:.2f}".replace(".", ",") for r in rho]); ax.set_xlabel("correlação entre o ruído das duas visões (0 = independentes)"); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(70, 100)
    ax.set_title("Visões mais dependentes: menos ganho, mas ainda positivo"); ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    save(fig, "f5_indep")

def f5_sms():
    L = out("ch05_sms").splitlines()[2:]
    nomes, v, dv = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0]); m, d = p[1].split("±"); v.append(pct(m)); dv.append(float(d))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    rbox(ax, 0.2, 6.6, 9.6, 2.6, "“WINNER! Ganhe um prêmio agora, ligue já”", fc="white", ec=INK, fs=8.5, r=0.3, lw=1.5)
    rbox(ax, 0.2, 2.6, 4.5, 2.6, "Visão 1:\nPALAVRAS\nwinner · prêmio · ligue", fc="#EEF2FF", ec=IND, fs=7.4, r=0.3, lw=1.8)
    rbox(ax, 5.3, 2.6, 4.5, 2.6, "Visão 2:\nCARACTERES (2 a 4)\nwi · win · inn · prê…", fc="#FFFBEB", ec=AMB, fs=7.4, r=0.3, lw=1.8)
    arrow(ax, (2.5, 6.6), (2.5, 5.3), IND, lw=1.4, ms=10); arrow(ax, (7.5, 6.6), (7.5, 5.3), AMB, lw=1.4, ms=10)
    ax.text(5, 0.9, "duas \"leituras\" do mesmo texto, mas muito redundantes", ha="center", fontsize=8, style="italic", color=RED)
    ax.set_title("Duas visões de uma mensagem")
    ax = axs[1]; y_ = np.arange(len(nomes))[::-1]
    ax.barh(y_, v, xerr=dv, color=[GRY, GRY, IND, RED], height=0.6, capsize=3, ecolor=INK)
    for yy, val in zip(y_, v): ax.text(99, yy, f"{val:.1f}%".replace(".", ","), va="center", ha="right", fontsize=8.8, fontweight="bold")
    ax.set_yticks(y_); ax.set_yticklabels(nomes, fontsize=8.5); ax.set_xlim(30, 100); ax.set_xlabel("F1 da classe spam (%)"); ax.set_title("SMS Spam, 40 rótulos"); ax.grid(axis="x", alpha=0.25)
    save(fig, "f5_sms")

def f5_tri():
    L = out("ch05_tri").splitlines()
    nomes, v, dv = [], [], []
    for l in L:
        a, b = l.split(":"); m, d = b.split("±"); nomes.append(a.strip()); v.append(pct(m)); dv.append(float(d))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 56)
    ax.text(50, 53, "Tri-training: três classificadores, sem visões", ha="center", fontsize=10, fontweight="bold")
    cols = [IND, AMB, GRN]
    for i, c in enumerate(cols):
        rbox(ax, 4 + i * 31, 36, 27, 10, f"h{i + 1}\nbootstrap {i + 1}", fc=c, tc="white", bold=True, fs=8, r=1.6)
    rbox(ax, 6, 14, 88, 16, "", fc="#F8FAFC", ec=MID, r=2)
    ax.text(50, 26, "Exemplo sem rótulo ganha pseudo-rótulo para h1 quando h2 e h3 CONCORDAM.", ha="center", fontsize=7.8, fontweight="bold")
    ax.text(50, 19.5, "Vale para cada classificador: a concordância dos outros dois é o filtro.", ha="center", fontsize=8.2)
    ax.text(50, 7, "A decisão final é a votação dos três (versão simplificada usada no livro).", ha="center", fontsize=8, style="italic", color=GRY)
    for i in range(3): arrow(ax, (17 + i * 31, 36), (17 + i * 31, 30.4), GRY, lw=1.2, ms=9)
    ax = axs[1]; y_ = np.arange(len(nomes))[::-1]
    ax.barh(y_, v, xerr=dv, color=[IND, AMB, RED], height=0.55, capsize=3, ecolor=INK)
    for yy, val in zip(y_, v): ax.text(99.5, yy, f"{val:.1f}%".replace(".", ","), va="center", ha="right", fontsize=8.8, fontweight="bold")
    ax.set_yticks(y_); ax.set_yticklabels(nomes, fontsize=8.5); ax.set_xlim(70, 100); ax.set_xlabel("acurácia no teste (%)"); ax.set_title("Digits, 50 rótulos"); ax.grid(axis="x", alpha=0.25)
    save(fig, "f5_tri")

if __name__ == "__main__":
    import matplotlib
    for f in [f5_ideia, f5_condicoes, f5_algoritmo, f5_digits, f5_resultados, f5_indep, f5_sms, f5_tri]: f()
