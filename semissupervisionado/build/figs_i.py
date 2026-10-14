"""Figuras do Capítulo 9 (redes neurais essenciais)."""
import sys
from fc import *
import matplotlib.colors
from figs_d import out, pct
sys.path.insert(0, "/home/user/claude/semissupervisionado/cap/code")
import torch, os
os.chdir("/home/user/claude/semissupervisionado")
from utilidades import carregar_mnist, aumento_fraco, aumento_forte

def f9_linha():
    fig, ax = canvas(9.2, 4.4, (0, 100), (0, 52))
    ax.text(50, 49.5, "Uma rede neural é uma linha de montagem de \"inspetores\"", ha="center", fontsize=10.8, fontweight="bold")
    etapas = [("Entrada", "pixels da imagem", GRY), ("Camada 1", "acha traços simples\n(bordas, curvas)", IND), ("Camada 2", "combina traços\n(cantos, laços)", CYA), ("Camada 3", "combina partes\n(um \"8\", um \"3\")", GRN), ("Saída", "uma probabilidade\npor classe", RED)]
    for i, (t, s, c) in enumerate(etapas):
        x = 1.5 + i * 19.7
        rbox(ax, x, 16, 17, 22, fc="white", ec=c, lw=2, r=2); rbox(ax, x, 32, 17, 6, t, fc=c, tc="white", bold=True, fs=8.2, r=2)
        ax.text(x + 8.5, 23.5, s, ha="center", va="center", fontsize=7.6, linespacing=1.4)
        if i < 4: arrow(ax, (x + 17.2, 27), (x + 19.5, 27), INK, lw=1.6, ms=10)
    ax.text(50, 8, "Cada camada recebe o que a anterior produziu e o transforma. Quem aprende os \"inspetores\" é o treinamento.", ha="center", fontsize=8.3, style="italic", color=GRY)
    save(fig, "f9_linha")

def f9_neuronio():
    fig, ax = canvas(9.2, 4.2, (0, 100), (0, 50))
    ax.text(50, 47.5, "Um neurônio artificial", ha="center", fontsize=10.8, fontweight="bold")
    for i, (y, t) in enumerate([(36, "x₁"), (25, "x₂"), (14, "x₃")]):
        ax.scatter(8, y, s=700, c=LIGHT, edgecolors=GRY, lw=2, zorder=3); ax.text(8, y, t, ha="center", va="center", fontsize=11, fontweight="bold")
        arrow(ax, (12, y), (42, 25), IND, lw=1.8, ms=10); ax.text(25, 25 + (y - 25) * 0.55 + 1.5, ["w₁", "w₂", "w₃"][i], ha="center", fontsize=9, color=IND, fontweight="bold")
    ax.scatter(48, 25, s=2600, c="#EEF2FF", edgecolors=IND, lw=2.5, zorder=3); ax.text(48, 25, "Σ + b", ha="center", va="center", fontsize=12, fontweight="bold", color=IND)
    arrow(ax, (54, 25), (66, 25), INK, lw=1.8, ms=12)
    rbox(ax, 67, 18, 14, 14, "f\n(ativação)", fc="#FFFBEB", ec=AMB, fs=9, r=2, lw=2); arrow(ax, (82, 25), (92, 25), INK, lw=1.8, ms=12); ax.text(95, 25, "saída", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(50, 7, "saída = f( w₁·x₁ + w₂·x₂ + w₃·x₃ + b )     entradas × pesos, somadas com um viés b, passam por uma função de ativação f", ha="center", fontsize=8.2, style="italic")
    save(fig, "f9_neuronio")

def f9_ativacoes():
    z = np.linspace(-5, 5, 400)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, f, t, c in [(axs[0], 1 / (1 + np.exp(-z)), "Sigmoide: comprime entre 0 e 1", IND), (axs[1], np.tanh(z), "Tangente hiperbólica: entre -1 e 1", GRN), (axs[2], np.maximum(0, z), "ReLU: zero se negativo, identidade se positivo", RED)]:
        ax.plot(z, f, c=c, lw=2.8); ax.axhline(0, c=GRY, lw=0.8); ax.axvline(0, c=GRY, lw=0.8); ax.set_title(t, fontsize=8.8); ax.set_xlabel("entrada z"); ax.grid(alpha=0.25)
    save(fig, "f9_ativacoes")

def f9_gradiente():
    f = lambda w: (w - 2) ** 2 + 0.5 * np.sin(3 * w) + 1; df = lambda w: 2 * (w - 2) + 1.5 * np.cos(3 * w)
    w_ = np.linspace(-1.5, 5.5, 400); fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True)
    for ax, lr, t in [(axs[0], 0.02, "Passo pequeno demais: lento"), (axs[1], 0.15, "Passo bom: chega ao vale"), (axs[2], 0.62, "Passo grande demais: oscila")]:
        ax.plot(w_, f(w_), c=GRY, lw=2); w = -1.0; pts = [w]
        for _ in range(12): w = w - lr * df(w); pts.append(w)
        pts = np.array(pts); ax.plot(pts, f(pts), "-o", c=RED, ms=4, lw=1.2); ax.scatter(pts[0], f(pts[0]), c=IND, s=70, zorder=4)
        ax.set_title(f"{t}\ntaxa de aprendizado = {lr}".replace(".", ","), fontsize=8.6); ax.set_xlabel("peso w"); ax.set_ylim(0, 14)
    axs[0].set_ylabel("perda")
    save(fig, "f9_gradiente")

def f9_convolucao():
    Xa, ya, _, _ = carregar_mnist(); img = Xa[7, 0].numpy()
    kern = {"borda vertical": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), "borda horizontal": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])}
    def conv(im, k):
        o = np.zeros((26, 26))
        for i in range(26):
            for j in range(26): o[i, j] = np.sum(im[i:i + 3, j:j + 3] * k)
        return o
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 3.0))
    axs[0].imshow(img, cmap="gray_r"); axs[0].set_title(f"Imagem original (dígito {int(ya[7])})", fontsize=9)
    for ax, (n, k) in zip(axs[1:3], kern.items()):
        ax.imshow(np.abs(conv(img, k)), cmap="magma"); ax.set_title(f"Filtro 3×3: {n}\n(mapa de atributos)", fontsize=9)
    mp = np.abs(conv(img, kern["borda vertical"])); mp = mp[:26, :26].reshape(13, 2, 13, 2).max(axis=(1, 3)); axs[3].imshow(mp, cmap="magma"); axs[3].set_title("Max pooling 2×2\n(resume e reduz)", fontsize=9)
    for a in axs: a.set_xticks([]); a.set_yticks([])
    save(fig, "f9_convolucao")

def f9_aug_exemplos():
    Xa, ya, _, _ = carregar_mnist(); x = Xa[[7]].repeat(6, 1, 1, 1); torch.manual_seed(1)
    fig, axs = plt.subplots(3, 6, figsize=(8.6, 4.6))
    linhas = [("original", [Xa[7]] * 6), ("aumento fraco", list(aumento_fraco(x))), ("aumento forte", list(aumento_forte(x)))]
    for r, (nome, imgs) in enumerate(linhas):
        for c in range(6):
            axs[r, c].imshow(imgs[c][0].numpy(), cmap="gray_r", vmin=0, vmax=1); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, 0].set_ylabel(nome, fontsize=9, rotation=90)
    save(fig, "f9_aug_exemplos")

def f9_overfit():
    L = out("ch09_cnn_mnist").splitlines(); i = [k for k, l in enumerate(L) if l.startswith("época")][0]
    ep, tr, te = [], [], []
    for l in L[i + 1:]:
        p = [x.strip() for x in l.split("|")]
        if len(p) == 3: ep.append(int(p[0])); tr.append(pct(p[1])); te.append(pct(p[2]))
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.plot(ep, tr, "-o", c=IND, lw=2.4, label="acerto nos 100 rotulados (treino)"); ax.plot(ep, te, "-s", c=RED, lw=2.4, label="acerto no teste")
    ax.fill_between(ep, te, tr, color=AMB, alpha=0.15); ax.text(55, 90, "diferença = sobreajuste", color="#B45309", fontsize=9, fontweight="bold")
    ax.set_xlabel("época"); ax.set_ylabel("acurácia (%)"); ax.set_title("CNN com 100 rótulos: decora o treino, generaliza menos"); ax.legend(frameon=False, loc="lower right", fontsize=8.5); ax.grid(alpha=0.25); ax.set_ylim(0, 105)
    save(fig, "f9_overfit")

def f9_aug_barras():
    L = out("ch09_cnn_mnist").splitlines()[2:5]
    nomes, v, dv = [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0].replace(" (", "\n(")); m, d = p[2].split("±"); v.append(pct(m)); dv.append(float(d))
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.bar(range(3), v, yerr=dv, color=[GRY, AMB, RED], width=0.55, capsize=4, ecolor=INK)
    for i, val in enumerate(v): ax.text(i, val + dv[i] + 0.8, f"{val:.1f}%".replace(".", ","), ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(nomes, fontsize=8.6); ax.set_ylim(70, 96); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("O aumento de dados vale muitos rótulos (MNIST, 100 rótulos)")
    save(fig, "f9_aug_barras")

def f9_rotulos():
    L = out("ch09_rotulos").splitlines()[2:]
    n, a = [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; n.append(int(p[0].split()[0])); a.append(pct(p[2]))
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.plot(n, a, "-o", c=RED, lw=2.6, ms=7)
    for x, y in zip(n, a): ax.text(x, y - 1.4, f"{y:.1f}%".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.axhline(88.7, c=IND, ls="--", lw=1.6); ax.text(250, 87.4, "Label Spreading em PCA (Cap. 8), com 100 rótulos: 88,7%", color=IND, fontsize=8)
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n]); ax.set_xlabel("número de rótulos (escala logarítmica)"); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(84, 99)
    ax.set_title("CNN supervisionada com aumento forte: a curva dos rótulos"); ax.grid(alpha=0.25)
    save(fig, "f9_rotulos")

if __name__ == "__main__":
    for f in [f9_linha, f9_neuronio, f9_ativacoes, f9_gradiente, f9_convolucao, f9_aug_exemplos, f9_overfit, f9_aug_barras, f9_rotulos]: f()
