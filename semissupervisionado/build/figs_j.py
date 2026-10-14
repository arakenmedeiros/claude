"""Figuras do Capítulo 10 (consistência)."""
import sys, os
from fc import *
from figs_d import out, pct
sys.path.insert(0, "/home/user/claude/semissupervisionado/cap/code")
import torch
os.chdir("/home/user/claude/semissupervisionado")
from utilidades import carregar_mnist, aumento_fraco, aumento_forte

def f10_ideia():
    Xa, ya, _, _ = carregar_mnist(); i = 7; torch.manual_seed(3)
    x = Xa[[i]].repeat(2, 1, 1, 1); v = aumento_forte(x)
    fig = plt.figure(figsize=(9.2, 3.6))
    for k, (img, t) in enumerate([(Xa[i], "original\n(sem rótulo)"), (v[0], "versão A\n(girada)"), (v[1], "versão B\n(deslocada,\ncom ruído)")]):
        ax = fig.add_axes([0.04 + k * 0.18, 0.12, 0.15, 0.62]); ax.imshow(img[0].numpy(), cmap="gray_r", vmin=0, vmax=1); clean(ax, True); ax.set_title(t, fontsize=8.4)
    ax = fig.add_axes([0.56, 0.05, 0.43, 0.9]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    rbox(ax, 2, 62, 28, 22, "Previsão\nda versão A", fc="#EEF2FF", ec=IND, fs=8.6, lw=1.8); rbox(ax, 2, 20, 28, 22, "Previsão\nda versão B", fc="#FFFBEB", ec=AMB, fs=8.6, lw=1.8)
    arrow(ax, (31, 73), (58, 60), INK, lw=1.6, ms=10); arrow(ax, (31, 31), (58, 45), INK, lw=1.6, ms=10)
    rbox(ax, 59, 38, 39, 28, "devem ser\nIGUAIS", fc=GRN, ec=GRN, fs=11, tc="white", bold=True, r=2)
    ax.text(78, 28, f"Não precisamos saber que é um \"{int(ya[i])}\":\nbasta exigir coerência.", ha="center", fontsize=8, style="italic", color=GRY)
    save(fig, "f10_ideia")

def f10_modelos():
    fig, ax = canvas(9.4, 5.6, (0, 100), (0, 100))
    ax.text(50, 97, "Três jeitos de pedir consistência", ha="center", fontsize=10.8, fontweight="bold")
    def painel(x0, titulo, cor, linhas, aluno2, nota):
        rbox(ax, x0, 80, 30, 9, titulo, fc=cor, ec=cor, tc="white", bold=True, fs=9, r=2)
        rbox(ax, x0 + 9, 66, 12, 8, "exemplo", fc=LIGHT, ec=GRY, fs=8)
        arrow(ax, (x0 + 15, 66), (x0 + 15, 62), INK, lw=1.4, ms=8)
        ax.text(x0 + 15, 59.5, linhas[0], ha="center", fontsize=7.6, color=INK)
        rbox(ax, x0 + 1, 40, 13, 14, "Rede\n(aluno)", fc="#EEF2FF", ec=IND, fs=8, lw=1.6); rbox(ax, x0 + 16, 40, 13, 14, aluno2[0], fc=aluno2[1], ec=aluno2[2], fs=8, lw=1.6)
        arrow(ax, (x0 + 15, 58), (x0 + 7.5, 55), INK, lw=1.1, ms=7); arrow(ax, (x0 + 15, 58), (x0 + 22.5, 55), INK, lw=1.1, ms=7)
        arrow(ax, (x0 + 7.5, 40), (x0 + 15, 33), INK, lw=1.1, ms=7); arrow(ax, (x0 + 22.5, 40), (x0 + 15, 33), INK, lw=1.1, ms=7)
        rbox(ax, x0 + 3, 24, 24, 9, "perda de\nconsistência", fc="#FEF2F2", ec=RED, fs=8, lw=1.6)
        ax.text(x0 + 15, 14, nota, ha="center", va="center", fontsize=7.6, linespacing=1.35)
    painel(1, "Π-model", IND, ["duas perturbações"], ("Rede\n(mesma)", "#EEF2FF", IND), "As duas previsões vêm da\nmesma rede e as duas recebem\no ajuste (gradiente).")
    painel(35, "Mean Teacher", GRN, ["duas perturbações"], ("Professor\n(EMA)", "#ECFDF5", GRN), "O alvo vem do professor, uma\nmédia móvel dos pesos do aluno.\nSó o aluno recebe gradiente.")
    painel(69, "UDA", AMB, ["versão fraca e forte"], ("Rede\n(visão fraca)", "#FFFBEB", AMB), "O alvo (afiado e filtrado pela\nconfiança) vem da versão fraca;\no aluno vê a versão forte.")
    save(fig, "f10_modelos")

def f10_rampa():
    t = np.linspace(0, 1, 200); w = np.exp(-5 * (1 - t) ** 2)
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); ax.plot(t * 100, w, c=RED, lw=2.8); ax.fill_between(t * 100, 0, w, color=RED, alpha=0.1)
    ax.set_xlabel("progresso do treino (%) até o fim da rampa"); ax.set_ylabel("peso da perda\nnão supervisionada"); ax.set_title("A rampa de peso: primeiro aprende-se com os rótulos"); ax.grid(alpha=0.25)
    ax.annotate("no começo, as previsões\nsão ruins e não servem de alvo", (12, 0.04), (3, 0.5), arrowprops=dict(arrowstyle="->", color=GRY), fontsize=8.2, color=GRY)
    save(fig, "f10_rampa")

def _tab(nome, ini):
    L = out(nome).splitlines()[ini:]; R = []
    for l in L:
        p = [x.strip() for x in l.split("|")]
        if len(p) < 3 or not p[1][:1].isdigit(): continue
        R.append((p[0], pct(p[1].split("±")[0]), pct(p[2].split("±")[0])))
    return R

def f10_comparacao():
    R = _tab("ch10_comparacao", 2)[:4]; nomes = ["só rotulados\n(referência)", "Π-model", "Mean\nTeacher", "UDA"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0)); x = np.arange(4); w = 0.36
    b1 = ax.bar(x - w / 2, [r[1] for r in R], w, color=[GRY, IND, GRN, AMB], label="aluno"); b2 = ax.bar(x + w / 2, [r[2] for r in R], w, color=[GRY, IND, GRN, AMB], alpha=0.55, hatch="//", edgecolor="white", label="professor EMA")
    for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.1f}".replace(".", ","), ha="center", fontsize=8.3, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=9); ax.set_ylim(90, 98); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("MNIST, 100 rótulos + 10.000 sem rótulo (média de 3 sorteios)")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=GRY), Rectangle((0, 0), 1, 1, fc=GRY, alpha=0.55, hatch="//", ec="white")], labels=["aluno", "professor (EMA)"], frameon=False, loc="upper left", fontsize=8.5); ax.grid(axis="y", alpha=0.25)
    save(fig, "f10_comparacao")

def f10_peso():
    R = _tab("ch10_peso", 2); Q = _tab("ch10_perturbacao", 2)
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    ax = axs[0]; pesos = [0, 1, 10, 30]; ax.plot(range(4), [r[1] for r in R], "-o", c=IND, lw=2.4, label="aluno"); ax.plot(range(4), [r[2] for r in R], "-s", c=GRN, lw=2.4, label="professor (EMA)")
    ax.set_xticks(range(4)); ax.set_xticklabels([str(p) for p in pesos]); ax.set_xlabel("peso máximo w"); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("Peso da consistência (Mean Teacher)", fontsize=9.4); ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=0.25)
    ax = axs[1]; x = np.arange(3); ax.bar(x - 0.18, [r[1] for r in Q], 0.36, color=IND, label="aluno"); ax.bar(x + 0.18, [r[2] for r in Q], 0.36, color=GRN, label="professor (EMA)")
    ax.set_xticks(x); ax.set_xticklabels(["nenhuma\n(só dropout)", "fraca", "forte"]); ax.set_ylim(93, 96); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("Perturbação nos sem rótulo (Mean Teacher)", fontsize=9.4); ax.grid(axis="y", alpha=0.25)
    for i, r in enumerate(Q): ax.text(i - 0.18, r[1] + 0.04, f"{r[1]:.1f}".replace(".", ","), ha="center", fontsize=8); ax.text(i + 0.18, r[2] + 0.04, f"{r[2]:.1f}".replace(".", ","), ha="center", fontsize=8)
    save(fig, "f10_peso")

def f10_quantidade():
    R = _tab("ch10_quantidade", 2); n = [0, 100, 1000, 10000, 40000]
    fig, ax = plt.subplots(figsize=(7.2, 3.8)); ax.plot(range(5), [r[1] for r in R], "-o", c=AMB, lw=2.6, ms=7, label="aluno"); ax.plot(range(5), [r[2] for r in R], "-s", c=GRN, lw=2.2, label="professor (EMA)")
    for i, r in enumerate(R): ax.text(i, r[1] + (0.12 if i == 0 else -0.45), f"{r[1]:.1f}".replace(".", ","), ha="center", fontsize=8.5, fontweight="bold", color="#B45309")
    ax.set_ylim(92.6, 96.9); ax.set_xticks(range(5)); ax.set_xticklabels(["0\n(referência)", "100", "1.000", "10.000", "40.000"]); ax.set_xlabel("imagens sem rótulo"); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("UDA: quanto ajudam os dados sem rótulo? (um sorteio)"); ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
    save(fig, "f10_quantidade")

if __name__ == "__main__":
    import sys
    for f in [f10_ideia, f10_modelos, f10_rampa, f10_comparacao, f10_peso] + ([f10_quantidade] if os.path.exists("cap/code/ch10_quantidade.out") else []): f()
