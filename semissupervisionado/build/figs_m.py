"""Figuras do Capítulo 13 (generativos profundos)."""
import sys, os
from fc import *
from figs_d import out, pct
sys.path.insert(0, "/home/user/claude/semissupervisionado/cap/code")
import torch
os.chdir("/home/user/claude/semissupervisionado")
from generativos import *

def f13_vae():
    fig, ax = canvas(9.4, 4.0, (0, 100), (0, 44))
    ax.text(50, 41.5, "Autocodificador variacional (VAE)", ha="center", fontsize=10.8, fontweight="bold")
    rbox(ax, 2, 15, 14, 12, "imagem x", fc=LIGHT, ec=GRY, fs=9); arrow(ax, (16.3, 21), (21.7, 21), INK)
    rbox(ax, 22, 12, 16, 18, "Codificador\n(rede)", fc="#EEF2FF", ec=IND, fs=9, lw=1.8); arrow(ax, (38.3, 21), (43.7, 21), INK)
    rbox(ax, 44, 9, 16, 24, "média μ e\ndesvio σ\n⟶ sorteia\nz = μ + σ·ε", fc="#FFFBEB", ec=AMB, fs=8.4, lw=1.8); arrow(ax, (60.3, 21), (65.7, 21), INK)
    rbox(ax, 66, 12, 14, 18, "Decodificador\n(rede)", fc="#ECFDF5", ec=GRN, fs=9, lw=1.8); arrow(ax, (80.3, 21), (84.7, 21), INK)
    rbox(ax, 85, 15, 13, 12, "imagem\nreconstruída", fc=LIGHT, ec=GRY, fs=8.4)
    ax.text(52, 4.5, "perda = erro de reconstrução + KL (aproxima a distribuição de z de uma normal padrão)", ha="center", fontsize=8.5, style="italic", color=GRY)
    save(fig, "f13_vae")

def f13_gan():
    fig, ax = canvas(9.4, 4.2, (0, 100), (0, 46))
    ax.text(50, 43.5, "GAN semissupervisionada: o falsificador e o detetive", ha="center", fontsize=10.8, fontweight="bold")
    rbox(ax, 1, 28, 12, 9, "ruído z", fc=LIGHT, ec=GRY, fs=9); arrow(ax, (13.3, 32.5), (18.7, 32.5), INK)
    rbox(ax, 19, 26, 17, 13, "Gerador\n(falsificador)", fc="#FEF2F2", ec=RED, fs=9, lw=1.8); arrow(ax, (36.3, 32.5), (45.5, 26), INK)
    rbox(ax, 1, 8, 12, 9, "imagens\nreais", fc=LIGHT, ec=GRY, fs=9); arrow(ax, (13.3, 12.5), (45.5, 20), INK)
    rbox(ax, 46, 15, 22, 17, "Discriminador\n(detetive)\ntambém classifica\nem 10 classes", fc="#EEF2FF", ec=IND, fs=8.6, lw=1.8)
    arrow(ax, (68.3, 23.5), (73.7, 30), INK); arrow(ax, (68.3, 23.5), (73.7, 14), INK)
    rbox(ax, 74, 26, 24, 11, "real ou falsa?", fc="#FFFBEB", ec=AMB, fs=8.8); rbox(ax, 74, 8, 24, 12, "qual dígito?\n(só com os rotulados)", fc="#ECFDF5", ec=GRN, fs=8.4)
    ax.text(50, 2.5, "O gerador tenta enganar o discriminador; o discriminador aprende a separar o real do falso e, de brinde, os dígitos.", ha="center", fontsize=8.2, style="italic", color=GRY)
    save(fig, "f13_gan")

def _carregar_vae(z):
    m = VAE(z); m.load_state_dict(torch.load(f"dados/vae{z}.pt")); m.eval(); return m

def f13_reconstrucao():
    X, y, Xt, yt = carregar_mnist(); m = _carregar_vae(20); torch.manual_seed(0)
    with torch.no_grad(): mu, lv = m.codificar(Xt[:10]); rec = torch.sigmoid(m.dec(mu)).view(-1, 1, 28, 28); z = torch.randn(10, 20); gen = torch.sigmoid(m.dec(z)).view(-1, 1, 28, 28)
    fig, axs = plt.subplots(3, 10, figsize=(9.2, 3.2))
    for i in range(10):
        for r, im in enumerate([Xt[i], rec[i], gen[i]]):
            axs[r, i].imshow(im[0].numpy(), cmap="gray_r", vmin=0, vmax=1); axs[r, i].set_xticks([]); axs[r, i].set_yticks([])
    for r, t in enumerate(["original", "recon.", "gerada"]): axs[r, 0].set_ylabel(t, fontsize=7.6, rotation=90)
    save(fig, "f13_reconstrucao")

def f13_latente():
    X, y, Xt, yt = carregar_mnist(); m = _carregar_vae(2)
    with torch.no_grad(): mu = m.codificar(Xt[:3000])[0].numpy()
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    sc = axs[0].scatter(mu[:, 0], mu[:, 1], c=yt[:3000].numpy(), cmap="tab10", s=5, alpha=0.8); axs[0].set_title("Espaço latente (2 dimensões)", fontsize=9.5); axs[0].set_xlabel("z₁"); axs[0].set_ylabel("z₂")
    g = np.linspace(-2.5, 2.5, 12); img = np.zeros((12 * 28, 12 * 28))
    with torch.no_grad():
        for i, b in enumerate(g[::-1]):
            for j, a in enumerate(g): img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = torch.sigmoid(m.dec(torch.tensor([[a, b]], dtype=torch.float32))).view(28, 28).numpy()
    axs[1].imshow(img, cmap="gray_r"); axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].set_title("Dígitos gerados ao percorrer z", fontsize=9.5)
    save(fig, "f13_latente")

def f13_gan_amostras():
    G = nn.Sequential(nn.Linear(100, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 784), nn.Sigmoid()); G.load_state_dict(torch.load("dados/sgan_G.pt")); G.eval(); torch.manual_seed(1)
    with torch.no_grad(): im = G(torch.randn(40, 100)).view(-1, 28, 28).numpy()
    fig, axs = plt.subplots(4, 10, figsize=(8.0, 3.4))
    for k, ax in enumerate(axs.ravel()): ax.imshow(im[k], cmap="gray_r", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "f13_gan_amostras")

if __name__ == "__main__":
    fs = [f13_vae, f13_gan]
    if os.path.exists("dados/vae20.pt"): fs += [f13_reconstrucao, f13_latente]
    if os.path.exists("dados/sgan_G.pt"): fs += [f13_gan_amostras]
    for f in fs: f()
