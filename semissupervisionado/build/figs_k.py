"""Figuras do Capítulo 11 (pseudo-rótulos modernos)."""
import sys, os
from fc import *
from figs_d import out, pct
sys.path.insert(0, "/home/user/claude/semissupervisionado/cap/code")
import torch
os.chdir("/home/user/claude/semissupervisionado")
from utilidades import carregar_mnist, aumento_fraco, aumento_forte

def f11_fixmatch():
    Xa, ya, _, _ = carregar_mnist(); i = 12; torch.manual_seed(5); x = Xa[[i]]
    fraca, forte = aumento_fraco(x)[0], aumento_forte(x)[0]
    fig = plt.figure(figsize=(9.4, 4.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis("off")
    ax.text(50, 44.5, "FixMatch: um exemplo sem rótulo, duas versões", ha="center", fontsize=10.8, fontweight="bold")
    def img(cx, cy, im, titulo, cor):
        a = fig.add_axes([cx / 100, cy / 47, 0.1, 0.2]); a.imshow(im[0].numpy(), cmap="gray_r", vmin=0, vmax=1); clean(a, True)
        for sp in a.spines.values(): sp.set_visible(True); sp.set_color(cor); sp.set_linewidth(2)
        ax.text(cx + 5, cy + 10.6, titulo, ha="center", fontsize=8.2, fontweight="bold", color=cor)
    img(2, 28, fraca, "versão fraca", IND); img(2, 5, forte, "versão forte", RED)
    arrow(ax, (14, 33), (22, 33), INK); rbox(ax, 22.5, 28, 14, 10, "Rede", fc="#EEF2FF", ec=IND, fs=9, lw=1.8); arrow(ax, (37, 33), (43, 33), INK)
    rbox(ax, 43.5, 27, 18, 12, "probabilidades:\nmáx = 0,98", fc="#EEF2FF", ec=IND, fs=8.4); arrow(ax, (62, 33), (66, 33), INK)
    rbox(ax, 66.5, 26, 15, 14, "máx ≥ 0,95?\nentão vira\npseudo-rótulo", fc="#ECFDF5", ec=GRN, fs=8.2, lw=1.8)
    arrow(ax, (14, 10), (22, 10), INK); rbox(ax, 22.5, 5, 14, 10, "Rede\n(a mesma)", fc="#EEF2FF", ec=IND, fs=9, lw=1.8); arrow(ax, (37, 10), (43, 10), INK)
    rbox(ax, 43.5, 4, 18, 12, "previsão na\nversão forte", fc="#FEF2F2", ec=RED, fs=8.4); arrow(ax, (62, 10), (66, 10), INK)
    rbox(ax, 66.5, 3.5, 15, 13, "entropia cruzada\ncom o\npseudo-rótulo", fc="#FEF2F2", ec=RED, fs=8.2, lw=1.8)
    arrow(ax, (74, 26), (74, 17), GRN, lw=2); ax.text(75.5, 21.5, "alvo", fontsize=8, color=GRN, fontweight="bold")
    save(fig, "f11_fixmatch")

def f11_mixup():
    Xa, ya, _, _ = carregar_mnist(); a, b = Xa[3], Xa[7]
    fig, axs = plt.subplots(1, 5, figsize=(9.2, 2.9))
    for ax, lam in zip(axs, [1.0, 0.8, 0.5, 0.2, 0.0]):
        im = lam * a + (1 - lam) * b; ax.imshow(im[0].numpy(), cmap="gray_r", vmin=0, vmax=1); clean(ax, True)
        ax.set_title(f"{int(lam * 100)}% \"{int(ya[3])}\" + {int(100 - lam * 100)}% \"{int(ya[7])}\"\nrótulo: ({lam:.1f}; {1 - lam:.1f})".replace(".", ","), fontsize=8)
    save(fig, "f11_mixup")

def f11_vies():
    fig, ax = canvas(9.2, 3.8, (0, 100), (0, 42))
    ax.text(50, 39.5, "O viés de confirmação: um erro que se realimenta", ha="center", fontsize=10.8, fontweight="bold")
    passos = [("1. A rede erra", "confunde \"4\" com \"9\"\ncom confiança alta", RED), ("2. O erro vira alvo", "vira pseudo-rótulo\ne passa pelo limiar", AMB), ("3. Aprende o erro", "treina para dizer \"9\"\nem outros \"4\" parecidos", IND), ("4. O erro se espalha", "mais \"4\" viram \"9\"\ne a confiança sobe", RED)]
    for i, (t, s, c) in enumerate(passos):
        x = 1.5 + i * 24.5; rbox(ax, x, 12, 22, 20, fc="white", ec=c, lw=2, r=2); rbox(ax, x, 26, 22, 6, t, fc=c, tc="white", bold=True, fs=7.8, r=2)
        ax.text(x + 11, 18.3, s, ha="center", va="center", fontsize=7.8, linespacing=1.4)
        if i < 3: arrow(ax, (x + 22.2, 22), (x + 24.3, 22), INK, lw=1.6, ms=10)
    ax.text(50, 5, "O limiar de confiança e o aumento forte reduzem esse risco, mas não o eliminam.", ha="center", fontsize=8.5, style="italic", color=GRY)
    save(fig, "f11_vies")

def _linhas(nome, ini):
    R = []
    for l in out(nome).splitlines()[ini:]:
        p = [x.strip() for x in l.split("|")]
        if len(p) >= 3 and p[1][:1].isdigit() or (len(p) >= 3 and p[1].startswith("--")): R.append(p)
    return R

def f11_comparacao():
    R = _linhas("ch11_comparacao", 2)[:4]; nomes = ["só rotulados\n(referência)", "Pseudo-Label", "FixMatch", "MixMatch\n(simplificado)"]
    v = [pct(r[1].split("±")[0]) for r in R]; d = [float(r[1].split("±")[1]) for r in R]
    uda = pct(out("ch10_comparacao").splitlines()[5].split("|")[1].split("±")[0])
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.bar(range(4), v, yerr=d, color=[GRY, AMB, IND, GRN], width=0.55, capsize=4, ecolor=INK)
    for i, val in enumerate(v): ax.text(i, val + d[i] + 0.15, f"{val:.1f}%".replace(".", ","), ha="center", fontsize=9.5, fontweight="bold")
    ax.axhline(uda, c=RED, ls="--", lw=1.6); ax.text(3.45, uda + 0.12, "UDA (Cap. 10): " + f"{uda:.1f}%".replace(".", ","), color=RED, fontsize=8.5, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels(nomes, fontsize=9); ax.set_ylim(90, 97.2); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("MNIST, 100 rótulos + 10.000 sem rótulo (média de 3 sorteios)"); ax.grid(axis="y", alpha=0.25)
    save(fig, "f11_comparacao")

def f11_qualidade():
    R = _linhas("ch11_comparacao", 2)[1:3]
    fr = [pct(r[2]) for r in R]; pr = [pct(r[3]) for r in R]; nomes = ["Pseudo-Label\n(sem limiar)", "FixMatch\n(limiar 0,95)"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6)); x = np.arange(2); ax.bar(x - 0.19, fr, 0.38, color=AMB, label="exemplos sem rótulo que viram pseudo-rótulo"); ax.bar(x + 0.19, pr, 0.38, color=GRN, label="pseudo-rótulos corretos entre os aceitos")
    for i in range(2): ax.text(i - 0.19, fr[i] + 1, f"{fr[i]:.1f}%".replace(".", ","), ha="center", fontsize=9, fontweight="bold"); ax.text(i + 0.19, pr[i] + 1, f"{pr[i]:.1f}%".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(nomes); ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("%"); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1); ax.set_title("Quantidade x qualidade dos pseudo-rótulos")
    save(fig, "f11_qualidade")

def f11_limiar():
    R = _linhas("ch11_limiar", 2); lim = [r[0] for r in R]; fr = [pct(r[1]) for r in R]; pr = [pct(r[2]) for r in R]; ac = [pct(r[3]) for r in R]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5)); x = range(len(R))
    axs[0].plot(x, fr, "-o", c=AMB, lw=2.4, label="fração aceita"); axs[0].plot(x, pr, "-s", c=GRN, lw=2.4, label="precisão dos aceitos")
    axs[0].set_xticks(list(x)); axs[0].set_xticklabels([l.replace(".", ",") for l in lim]); axs[0].set_xlabel("limiar de confiança"); axs[0].set_ylabel("%"); axs[0].set_title("Quantidade x qualidade", fontsize=9.6); axs[0].legend(frameon=False, fontsize=8.3); axs[0].grid(alpha=0.25)
    axs[1].plot(x, ac, "-o", c=IND, lw=2.6); axs[1].set_xticks(list(x)); axs[1].set_xticklabels([l.replace(".", ",") for l in lim]); axs[1].set_xlabel("limiar de confiança"); axs[1].set_ylabel("acurácia no teste (%)"); axs[1].set_title("Efeito na acurácia (um sorteio)", fontsize=9.6); axs[1].grid(alpha=0.25); axs[1].set_ylim(94.5, 96.6)
    for i, a in enumerate(ac): axs[1].text(i, a + 0.12, f"{a:.1f}".replace(".", ","), ha="center", fontsize=8.5, fontweight="bold")
    save(fig, "f11_limiar")

def f11_poucos():
    R = _linhas("ch11_poucos", 2); nomes = [r[0].split("(")[0].strip() + "\n(" + r[0].split("(")[1] for r in R]
    ref = [pct(r[1]) for r in R]; ps = [pct(r[2]) for r in R]; fm = [pct(r[3]) for r in R]
    fig, ax = plt.subplots(figsize=(7.6, 3.9)); x = np.arange(len(R)); w = 0.26
    for k, (v, c, n) in enumerate([(ref, GRY, "só rotulados"), (ps, AMB, "Pseudo-Label"), (fm, IND, "FixMatch")]):
        ax.bar(x + (k - 1) * w, v, w, color=c, label=n)
        for i, val in enumerate(v): ax.text(x[i] + (k - 1) * w, val + 0.8, f"{val:.1f}".replace(".", ","), ha="center", fontsize=7.4)
    ax.set_xticks(x); ax.set_xticklabels([f"{n} por dígito" if False else n for n in nomes], fontsize=8.6); ax.set_ylim(0, 105); ax.set_ylabel("acurácia no teste (%)"); ax.set_xlabel("rótulos no total (por dígito)"); ax.legend(frameon=False, fontsize=8.5, loc="upper left"); ax.set_title("E com ainda menos rótulos? (um sorteio)"); ax.grid(axis="y", alpha=0.25)
    save(fig, "f11_poucos")

if __name__ == "__main__":
    for f in [f11_fixmatch, f11_mixup, f11_vies, f11_comparacao, f11_qualidade] + ([f11_limiar] if os.path.exists("cap/code/ch11_limiar.out") else []) + ([f11_poucos] if os.path.exists("cap/code/ch11_poucos.out") else []): f()
