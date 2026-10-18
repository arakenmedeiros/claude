"""Figuras do Capítulo 13. Baseadas em dados (sintéticos/embutidos): pixels, patches, contraste, temp, espectro, difusao. Conceituais: mapa, projetor, documento."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from sklearn.datasets import load_digits, make_moons
from llmdidatico.multimodal import *

dg = load_digits(); imagens, rotulos = dg.images, dg.target
X = imagens.reshape(len(imagens), -1) / 16
rng = np.random.default_rng(0); idx = rng.permutation(len(rotulos)); tr, te = idx[:1300], idx[1300:]
med = np.median(X.sum(1)); rng_t = np.random.default_rng(1)
legendas = [descricao(rotulos[i], imagens[i] / 16, rng_t, med) for i in range(len(rotulos))]
voc = vocabulario(legendas); T = bow(legendas, voc); Xi = X - 0.5

def emb(Xa, W):
    Z = Xa @ W; return Z / np.linalg.norm(Z, axis=1, keepdims=True)

def mapa():
    fig, ax = canvas(9.4, 4.4)
    mods = [("Texto", IND), ("Imagem", CYA), ("Áudio", AMB), ("Documento", GRN)]
    for k, (n, c) in enumerate(mods):
        y = 78 - k * 20; rbox(ax, 1, y, 16, 14, n, fc=c, ec=c, tc="white", fs=9, bold=True)
        rbox(ax, 22, y, 20, 14, ["tokens", "pixels e patches", "espectrograma", "OCR + layout"][k], fc="white", ec=c, fs=8.4)
        arrow(ax, (17.4, y + 7), (21.6, y + 7), c=INK, lw=1.2, ms=9); arrow(ax, (42.4, y + 7), (50.6, 52), c=GRY, lw=1.1, ms=9)
    rbox(ax, 51, 38, 20, 28, "Espaço de\nembeddings\n(contrastivo)", fc="#E0E7FF", ec=IND, fs=8.8, bold=True)
    rbox(ax, 78, 66, 21, 22, "LLM\n(via projetor)", fc="#D1FAE5", ec=GRN, fs=8.8, bold=True)
    rbox(ax, 78, 36, 21, 22, "Difusão\n(gera imagens)", fc="#FEF3C7", ec=AMB, fs=8.8, bold=True)
    rbox(ax, 51, 4, 48, 22, "Avaliação por fatia · vieses · alucinação visual\ncusto em tokens · privacidade · acessibilidade", fc="#FCE7F3", ec=PNK, fs=8.4, bold=True)
    arrow(ax, (71.4, 56), (77.6, 74), c=INK, lw=1.3, ms=10); arrow(ax, (71.4, 48), (77.6, 46), c=INK, lw=1.3, ms=10)
    save(fig, "f13_mapa")

def pixels():
    im = imagens[0]
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    ax = axs[0]; ax.imshow(im, cmap="gray_r", vmin=0, vmax=16)
    for i in range(8):
        for j in range(8): ax.text(j, i, int(im[i, j]), ha="center", va="center", fontsize=6.5, color="white" if im[i, j] > 8 else INK)
    ax.set_title("grade 8×8 de números", fontsize=8.5); clean(ax, True)
    g = im / 16
    for ax, (nm, m, cm) in zip(axs[1:], [("canal R", g, "Reds"), ("canal G", 0.5 * g, "Greens"), ("canal B", 0.2 * g, "Blues")]):
        ax.imshow(m, cmap=cm, vmin=0, vmax=1); ax.set_title(nm, fontsize=8.5); clean(ax, True)
    save(fig, "f13_pixels")

def patches_fig():
    img = imagens[0] / 16; P = patches(img, 4)
    Ptr = np.concatenate([patches(imagens[i] / 16, 4) for i in tr]); mu = Ptr.mean(0)
    _, s, Vt = np.linalg.svd(Ptr - mu, full_matrices=False); W = Vt[:8].T
    E = (P - mu) @ W; pos = posicao_senoidal(4, 8)
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.1), gridspec_kw={"width_ratios": [1, 1, 1]})
    ax = axs[0]; ax.imshow(img, cmap="gray_r"); 
    ax.axhline(3.5, c=RED, lw=1.5); ax.axvline(3.5, c=RED, lw=1.5)
    for k, (y, x) in enumerate([(1.5, 1.5), (1.5, 5.5), (5.5, 1.5), (5.5, 5.5)]): ax.text(x, y, str(k + 1), color=RED, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title("4 patches 4×4", fontsize=8.5); clean(ax, True)
    v = max(abs(E).max(), abs(E + pos).max())
    for ax, (M, tt) in zip(axs[1:], [(E, "embeddings (PCA, d=8)"), (E + pos, "embeddings + posição")]):
        ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); ax.set_title(tt, fontsize=8.5)
        ax.set_yticks(range(4)); ax.set_yticklabels([f"patch {k+1}" for k in range(4)], fontsize=7.5); ax.set_xticks(range(8)); ax.set_xticklabels(range(1, 9), fontsize=7.5); ax.set_xlabel("dimensão")
    save(fig, "f13_patches")

def treinar(tau, ep=60):
    return treinar_contrastivo(Xi[tr], T[tr], d=16, tau=tau, epocas=ep, lote=64, lr=0.05, seed=0)

def contraste():
    Wi, Wt, _ = treinar(0.1); yt = rotulos[te]
    sel = [int(np.where(yt == d)[0][0]) for d in (0, 1, 3, 7, 8, 9)]
    r = np.random.default_rng(0); Wi0 = r.normal(0, 0.1, Wi.shape); Wt0 = r.normal(0, 0.1, Wt.shape)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, (A, B, tt) in zip(axs, [(Wi0, Wt0, "antes (aleatório)"), (Wi, Wt, "depois do treino")]):
        S = emb(Xi[te][sel], A) @ emb(T[te][sel], B).T
        im = ax.imshow(S, cmap="viridis", vmin=-1, vmax=1)
        for i in range(6):
            for j in range(6): ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if S[i, j] < 0.4 else INK)
        ax.set_xticks(range(6)); ax.set_yticks(range(6)); lab = [PALAVRAS[yt[i]] for i in sel]
        ax.set_xticklabels(lab, rotation=45, fontsize=7.5); ax.set_yticklabels(lab, fontsize=7.5); ax.set_title(tt, fontsize=9)
        ax.set_xlabel("descrição"); ax.set_ylabel("imagem")
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, label="cosseno")
    save(fig, "f13_contraste", tight=True)

def temp():
    taus = (0.01, 0.05, 0.1, 0.5, 1.0); L, R = [], []
    yt = rotulos[te]
    for tau in taus:
        Wi, Wt, h = treinar(tau); S = emb(Xi[te], Wi) @ emb(T[te], Wt).T
        L.append(h[-1]); R.append(recall_em_k(S, yt, yt, (1,))[1])
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axs[0].plot(range(5), L, "o-", c=IND); axs[0].axhline(np.log(64), c=GRY, ls="--"); axs[0].text(0.05, np.log(64) + 0.08, "ln 64 (acaso)", color=GRY, fontsize=8)
    axs[1].plot(range(5), R, "o-", c=GRN); axs[1].set_ylim(0.9, 1.0)
    for ax, yl, tt in zip(axs, ["perda final", "recall@1 imagem→texto"], ["perda (não comparável entre τ)", "recuperação"]):
        ax.set_xticks(range(5)); ax.set_xticklabels([str(t) for t in taus]); ax.set_xlabel("temperatura τ"); ax.set_ylabel(yl); ax.set_title(tt, fontsize=9)
    save(fig, "f13_temp")

def projetor():
    fig, ax = canvas(9.4, 3.6)
    rbox(ax, 1, 40, 14, 22, "Imagem", fc=CYA, ec=CYA, tc="white", fs=9, bold=True)
    rbox(ax, 21, 40, 18, 22, "Codificador\nvisual\n(ViT)", fc="#CFFAFE", ec=CYA, fs=8.6, bold=True)
    rbox(ax, 45, 40, 16, 22, "Projetor\n(linear ou\npequena rede)", fc="#FEF3C7", ec=AMB, fs=8.4, bold=True)
    for k in range(4): rbox(ax, 66 + k * 4.2, 66, 3.6, 10, "", fc=CYA, ec=CYA)
    ax.text(74.5, 80, "tokens visuais", ha="center", fontsize=8, color=CYA)
    for k in range(5): rbox(ax, 66 + k * 4.2, 22, 3.6, 10, "", fc=IND, ec=IND)
    ax.text(76, 14, "tokens da pergunta", ha="center", fontsize=8, color=IND)
    rbox(ax, 66, 38, 33, 24, "LLM  →  texto da resposta", fc="#D1FAE5", ec=GRN, fs=9, bold=True)
    for a, b in [((15.4, 51), (20.6, 51)), ((39.4, 51), (44.6, 51)), ((61.4, 51), (65.6, 51))]: arrow(ax, a, b, c=INK, lw=1.3, ms=10)
    arrow(ax, (74.5, 65.5), (74.5, 62.5), c=GRY, lw=1.1, ms=8); arrow(ax, (76, 33), (76, 37.5), c=GRY, lw=1.1, ms=8)
    save(fig, "f13_projetor")

def espectro():
    fs = 8000; t = np.arange(2 * fs) / fs; rg = np.random.default_rng(0); x = np.zeros_like(t)
    m1 = t < 0.8; x[m1] = np.sin(2 * np.pi * 440 * t[m1]); m2 = t >= 1.2
    x[m2] = np.sin(2 * np.pi * (300 * (t[m2] - 1.2) + 0.5 * (1500 - 300) / 0.8 * (t[m2] - 1.2) ** 2)); x += 0.05 * rg.standard_normal(len(x))
    Z = stft(x, 256, 64); db = 20 * np.log10(np.abs(Z) + 1e-6)
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True)
    axs[0].plot(t, x, c=IND, lw=0.4); axs[0].set_ylabel("amplitude"); axs[0].set_title("forma de onda", fontsize=9)
    axs[1].imshow(db, origin="lower", aspect="auto", extent=[0, 2, 0, fs / 2], cmap="magma", vmin=db.max() - 70); axs[1].set_ylim(0, 2500)
    axs[1].set_ylabel("frequência (Hz)"); axs[1].set_xlabel("tempo (s)"); axs[1].set_title("espectrograma (STFT, dB)", fontsize=9)
    save(fig, "f13_espectro")

def difusao():
    T_ = 1000; beta, alpha, abar = agenda_beta(T_); rd = np.random.default_rng(0); n = 50000
    mu_k, sg, w_k = np.array([-2.0, 2.0]), 0.3, np.array([0.5, 0.5])
    x0 = rd.choice(mu_k, size=n, p=w_k) + sg * rd.standard_normal(n)
    def eps_otimo(xt, t_):
        ab = abar[t_]; v = ab * sg ** 2 + (1 - ab)
        logp = -0.5 * (xt[:, None] - np.sqrt(ab) * mu_k[None, :]) ** 2 / v + np.log(w_k)[None, :]
        r = np.exp(logp - logp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
        x0k = mu_k[None, :] + (np.sqrt(ab) * sg ** 2 / v) * (xt[:, None] - np.sqrt(ab) * mu_k[None, :])
        return (xt - np.sqrt(ab) * (r * x0k).sum(1)) / np.sqrt(1 - ab)
    xs = rd.standard_normal(20000)
    for t_ in range(T_ - 1, -1, -1):
        xs = (xs - beta[t_] / np.sqrt(1 - abar[t_]) * eps_otimo(xs, t_)) / np.sqrt(alpha[t_])
        if t_ > 0: xs = xs + np.sqrt(beta[t_]) * rd.standard_normal(len(xs))
    fig = plt.figure(figsize=(9.2, 5.6)); gs = fig.add_gridspec(2, 5)
    ts = (0, 49, 149, 399, 999); cols = [IND, CYA, GRN, AMB, RED]
    ax = fig.add_subplot(gs[0, :3])
    for t_, c in zip(ts, cols):
        xt, _ = ruido_direto(x0, t_, abar, rd); ax.hist(xt, bins=80, range=(-5, 5), histtype="step", color=c, lw=1.3, label=f"t = {t_ + 1}", density=True)
    ax.legend(fontsize=7, frameon=False); ax.set_title("processo direto (fórmula fechada)", fontsize=9); ax.set_xlabel("x")
    ax = fig.add_subplot(gs[0, 3:]); ax.hist(x0, bins=80, range=(-5, 5), density=True, color=GRY, alpha=0.6, label="dados")
    ax.hist(xs, bins=80, range=(-5, 5), histtype="step", density=True, color=RED, lw=1.5, label="gerado a partir de ruído"); ax.set_ylim(0, 0.95); ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=1)
    ax.set_title("processo reverso (denoiser analítico)", fontsize=9); ax.set_xlabel("x")
    M, _ = make_moons(600, noise=0.05, random_state=0); M = (M - M.mean(0)) / M.std(0)
    for k, t_ in enumerate((0, 49, 149, 399, 999)):
        ax = fig.add_subplot(gs[1, k]); xt, _ = ruido_direto(M, t_, abar, rd); ax.scatter(xt[:, 0], xt[:, 1], s=4, c=cols[k]); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_title(f"t = {t_ + 1}", fontsize=8.5); clean(ax, True)
    save(fig, "f13_difusao")

def documento():
    fig, ax = canvas(9.4, 2.8)
    et = [("Imagem\nda página", CYA), ("Regiões\ne OCR", IND), ("Layout e\ntabelas", VIO), ("Extração\nestruturada", GRN), ("Validações\n(somas)", AMB), ("Revisão\nhumana", RED)]
    for k, (t, c) in enumerate(et):
        x = 1 + k * 16.4; rbox(ax, x, 30, 14, 40, t, fc=c, ec=c, tc="white", fs=8, bold=True)
        if k < 5: arrow(ax, (x + 14.2, 50), (x + 16.2, 50), c=INK, lw=1.2, ms=9)
    ax.text(50, 12, "o erro é silencioso: valide e revise antes de usar", ha="center", fontsize=8.8, color=GRY, style="italic")
    save(fig, "f13_documento")

if __name__ == "__main__":
    mapa(); pixels(); patches_fig(); contraste(); temp(); projetor(); espectro(); difusao(); documento()
