"""Capítulo 13: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 13.1: A ordem dos patches importa?]
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from llmdidatico.multimodal import patches

dg = load_digits(); imagens, y = dg.images / 16, dg.target
idx = np.random.default_rng(0).permutation(len(y)); tr, te = idx[:1300], idx[1300:]

def caracteristicas(p, d=8):
    Ptr = np.concatenate([patches(imagens[i], p) for i in tr]); mu = Ptr.mean(0)
    _, _, Vt = np.linalg.svd(Ptr - mu, full_matrices=False); W = Vt[:d].T
    E = np.stack([(patches(im, p) - mu) @ W for im in imagens])      # (n, N, d)
    return E.mean(1), E.reshape(len(E), -1)

print(f"{'patch':>6} {'N':>4} {'média (sem ordem)':>18} {'concatenação (com ordem)':>25}")
for p in (4, 2):
    media, conc = caracteristicas(p)
    acc = []
    for F in (media, conc):
        clf = LogisticRegression(max_iter=2000).fit(F[tr], y[tr])
        acc.append(clf.score(F[te], y[te]))
    print(f"{p}x{p:<4} {(8 // p) ** 2:>4} {acc[0]:>18.3f} {acc[1]:>25.3f}")


# %% [Exercício 13.2: Temperatura e dimensão do embedding]
import numpy as np
from sklearn.datasets import load_digits
from llmdidatico.multimodal import descricao, vocabulario, bow, treinar_contrastivo, recall_em_k

dg = load_digits(); imagens, y = dg.images, dg.target
X = imagens.reshape(len(imagens), -1) / 16
idx = np.random.default_rng(0).permutation(len(y)); tr, te = idx[:1300], idx[1300:]
med = np.median(X.sum(1)); rt = np.random.default_rng(1)
leg = [descricao(y[i], imagens[i] / 16, rt, med) for i in range(len(y))]
T = bow(leg, vocabulario(leg)); Xi = X - 0.5

def emb(A, W):
    Z = A @ W; return Z / np.linalg.norm(Z, axis=1, keepdims=True)

print(f"{'tau':>5} {'d':>3} {'R@1 i->t (média ± dp, 3 sementes)':>36}")
for tau in (0.05, 0.5):
    for d in (2, 4, 8, 16):
        r = []
        for sem in range(3):
            Wi, Wt, _ = treinar_contrastivo(Xi[tr], T[tr], d=d, tau=tau, epocas=40, lote=64, lr=0.05, seed=sem)
            S = emb(Xi[te], Wi) @ emb(T[te], Wt).T
            r.append(recall_em_k(S, y[te], y[te], (1,))[1])
        print(f"{tau:>5} {d:>3} {np.mean(r):>25.3f} ± {np.std(r):.3f}")


# %% [Exercício 13.3: Janela da STFT: tempo versus frequência]
import numpy as np
from llmdidatico.multimodal import stft

fs = 8000
t = np.arange(2 * fs) / fs
x = np.where(t >= 1.0, np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 470 * t), 0.0)
print(f"{'n_fft':>6} {'resol. (Hz)':>12} {'picos separados':>16} {'transição (ms)':>15}")
for n_fft in (64, 128, 256, 512, 1024):
    salto = n_fft // 4
    Z = np.abs(stft(x, n_fft, salto)); freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    centros = (np.arange(Z.shape[1]) * salto + n_fft / 2) / fs
    esp = Z[:, centros >= 1.2 + n_fft / fs].mean(1)
    sel = (freqs > 300) & (freqs < 620); e = esp[sel]; f = freqs[sel]
    picos = [f[i] for i in range(1, len(e) - 1) if e[i] > e[i - 1] and e[i] > e[i + 1] and e[i] > 0.5 * e.max()]
    separados = len(picos) >= 2 and (max(picos) - min(picos)) >= 20
    energia = Z.max(0); i10 = np.argmax(energia > 0.1 * energia.max()); i90 = np.argmax(energia > 0.9 * energia.max())
    print(f"{n_fft:>6} {fs / n_fft:>12.2f} {str(separados):>16} {(i90 - i10) * salto / fs * 1000:>15.1f}")


# %% [Projeto: Busca imagem–texto com abstenção e avaliação por fatia]
import numpy as np
from sklearn.datasets import load_digits
from llmdidatico.multimodal import descricao, vocabulario, bow, treinar_contrastivo, recall_em_k, PALAVRAS
from llmdidatico import avaliacao as av

dg = load_digits(); imagens, y = dg.images, dg.target
X = imagens.reshape(len(imagens), -1) / 16
idx = np.random.default_rng(0).permutation(len(y)); trn, val, te = idx[:1000], idx[1000:1300], idx[1300:]
med = np.median(X.sum(1)); rt = np.random.default_rng(1)
leg = [descricao(y[i], imagens[i] / 16, rt, med) for i in range(len(y))]
voc = vocabulario(leg); T = bow(leg, voc); Xi = X - 0.5

def emb(A, W):
    Z = A @ W; return Z / np.linalg.norm(Z, axis=1, keepdims=True)

Wi, Wt, h = treinar_contrastivo(Xi[trn], T[trn], d=16, tau=0.1, epocas=60, lote=64, lr=0.05, seed=0)
Up = emb(bow([f"dígito {w}" for w in PALAVRAS], voc), Wt)

# 2. recuperação e classificação
S = emb(Xi[te], Wi) @ emb(T[te], Wt).T
r_i2t, r_t2i = recall_em_k(S, y[te], y[te], (1, 5)), recall_em_k(S.T, y[te], y[te], (1, 5))
print("recall@1/@5 imagem->texto:", round(r_i2t[1], 3), round(r_i2t[5], 3), "| texto->imagem:", round(r_t2i[1], 3), round(r_t2i[5], 3))
acertos = ((emb(Xi[te], Wi) @ Up.T).argmax(1) == y[te]).astype(float)
m, lo, hi = av.bootstrap_ic(acertos, n=1000, seed=0)
print(f"acurácia por descrições de classe: {m:.3f}  IC95% [{lo:.3f}; {hi:.3f}]")

# 3. robustez ao ruído
rg = np.random.default_rng(4)
print("\nsigma   acurácia   IC95%")
for sg in (0.0, 0.1, 0.3, 0.5):
    Xr = np.clip(X[te] + sg * rg.standard_normal(X[te].shape), 0, 1) - 0.5
    a = ((emb(Xr, Wi) @ Up.T).argmax(1) == y[te]).astype(float)
    m, lo, hi = av.bootstrap_ic(a, n=1000, seed=0)
    print(f"{sg:<7} {m:<10.3f} [{lo:.3f}; {hi:.3f}]")

# 4. abstenção
esc_val = (emb(Xi[val], Wi) @ Up.T).max(1)
esc_te = (emb(Xi[te], Wi) @ Up.T)
pred_te, s_te = esc_te.argmax(1), esc_te.max(1)
ruido = np.random.default_rng(9).uniform(0, 1, (300, 64)) - 0.5
s_ru = (emb(ruido, Wi) @ Up.T).max(1)
print("\npercentil  limiar  cobertura  acurácia nas aceitas  ruído aceito")
for q in (0, 5, 10, 20):
    lim = np.percentile(esc_val, q)
    ac = s_te >= lim
    print(f"{q:>9} {lim:>7.3f} {ac.mean():>10.3f} {(pred_te[ac] == y[te][ac]).mean():>20.3f} {(s_ru >= lim).mean():>13.3f}")

