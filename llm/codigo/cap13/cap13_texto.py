"""Capítulo 13: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np
from sklearn.datasets import load_digits

dg = load_digits()
imagens, rotulos = dg.images, dg.target
print("conjunto:", imagens.shape, "| tipo:", imagens.dtype)
print("pixel mínimo e máximo:", imagens.min(), imagens.max())
print("rótulo da imagem 0:", rotulos[0])
print(imagens[0].astype(int))


# %% [bloco 2]
cinza = imagens[0] / 16                                          # normaliza para [0, 1]
rgb = np.stack([cinza, 0.5 * cinza, 0.2 * cinza], axis=-1)       # três canais (R, G, B): tingimos de laranja
print("formato RGB (altura, largura, canais):", rgb.shape)
print("pixel (2, 3) nos canais R, G, B:", rgb[2, 3].round(3))
luminancia = rgb @ np.array([0.299, 0.587, 0.114])               # pesos usuais para converter RGB em cinza
print("formato após converter para cinza:", luminancia.shape)
for h, l in [(8, 8), (224, 224), (1024, 768)]:
    print(f"imagem {h}x{l} em RGB: {h * l * 3} números")


# %% [bloco 3]
from llmdidatico.multimodal import patches, posicao_senoidal

rng = np.random.default_rng(0)
idx = rng.permutation(len(rotulos)); tr, te = idx[:1300], idx[1300:]       # divisão treino/teste usada no capítulo
img = imagens[0] / 16
P = patches(img, 4)
print("patches 4x4:", P.shape, "(N patches, p*p valores)  |  patches 2x2:", patches(img, 2).shape)

Ptr = np.concatenate([patches(imagens[i] / 16, 4) for i in tr])           # todos os patches do treino
mu = Ptr.mean(0)
_, s, Vt = np.linalg.svd(Ptr - mu, full_matrices=False)
d = 8
W = Vt[:d].T                                                              # projeção 16 -> 8 (PCA)
print(f"variância dos patches explicada por {d} componentes: {(s[:d]**2).sum() / (s**2).sum():.3f}")

E = (P - mu) @ W                                                          # embeddings de patch (4, 8)
pos = posicao_senoidal(len(P), d)
cls = rng.normal(0, 0.1, (1, d))                                          # token de classificação (aqui, fixo)
seq = np.vstack([cls, E + pos])
print("sequência entregue ao Transformer:", seq.shape, "(1 token especial + 4 patches)")

def atencao_media(X):                                                     # autoatenção sem parâmetros, saída média
    A = X @ X.T / np.sqrt(X.shape[1]); A = np.exp(A - A.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    return (A @ X).mean(0)
ordem = np.array([3, 1, 0, 2])                                            # embaralha os patches
Eb = (P[ordem] - mu) @ W
print("sem posição, diferença após embaralhar:", np.abs(atencao_media(E) - atencao_media(Eb)).max().round(6))
print("com posição, diferença após embaralhar:", np.abs(atencao_media(E + pos) - atencao_media(Eb + pos)).max().round(6))


# %% [bloco 4]
from llmdidatico.multimodal import (descricao, vocabulario, bow, infonce_simetrica,
                                    treinar_contrastivo, recall_em_k, PALAVRAS)

X = imagens.reshape(len(imagens), -1) / 16                     # 64 pixels por imagem, em [0, 1]
med = np.median(X.sum(1))
rng_t = np.random.default_rng(1)
legendas = [descricao(rotulos[i], imagens[i] / 16, rng_t, med) for i in range(len(rotulos))]
voc = vocabulario(legendas)
T = bow(legendas, voc)
chaves = np.array([f"{rotulos[i]}|" + ",".join(c.split(",")[1:]) for i, c in enumerate(legendas)])   # chave exata: dígito + atributos
print("exemplos de legendas:", legendas[:3])
print("tamanho do vocabulário:", len(voc), "| legendas distintas:", len(set(legendas)))

# verificação do gradiente por diferenças finitas
g = np.random.default_rng(0); A = g.normal(size=(5, 4)); B = g.normal(size=(5, 4))
L0, dA, dB, _ = infonce_simetrica(A, B, 0.3)
A2 = A.copy(); A2[2, 1] += 1e-6
print("gradiente analítico:", round(dA[2, 1], 6), "| numérico:", round((infonce_simetrica(A2, B, 0.3)[0] - L0) / 1e-6, 6))
print("perda esperada com embeddings ao acaso (ln 64):", round(float(np.log(64)), 3))


# %% [bloco 5]
Xi = X - 0.5
Wi, Wt, hist = treinar_contrastivo(Xi[tr], T[tr], d=16, tau=0.1, epocas=60, lote=64, lr=0.05, seed=0)
print(f"perda: época 1 = {hist[0]:.3f} | época 60 = {hist[-1]:.3f}")

def embutir(Xa, Wa):
    Z = Xa @ Wa; return Z / np.linalg.norm(Z, axis=1, keepdims=True)

Ui, Ut = embutir(Xi[te], Wi), embutir(T[te], Wt)
S = Ui @ Ut.T                                                  # similaridades de cosseno, 497 x 497
yt = rotulos[te]
sel = [int(np.where(yt == dgt)[0][0]) for dgt in (0, 1, 3, 7, 8, 9)]   # seis pares de dígitos diferentes
print("matriz de similaridades (linhas: imagens; colunas: descrições dos mesmos pares)")
print("          " + " ".join(f"{PALAVRAS[yt[j]]:>7}" for j in sel))
for i in sel:
    print(f"{PALAVRAS[yt[i]]:>9} " + " ".join(f"{S[i, j]:7.2f}" for j in sel))


# %% [bloco 6]
def avaliar(Wi_, Wt_, ks=(1, 5)):
    Ui_, Ut_ = embutir(Xi[te], Wi_), embutir(T[te], Wt_)
    S_ = Ui_ @ Ut_.T
    a_i2t, a_t2i = recall_em_k(S_, yt, yt, ks), recall_em_k(S_.T, yt, yt, ks)
    ch = chaves[te]
    b_i2t, b_t2i = recall_em_k(S_, ch, ch, ks), recall_em_k(S_.T, ch, ch, ks)
    return a_i2t, a_t2i, b_i2t, b_t2i, S_

a_i2t, a_t2i, b_i2t, b_t2i, _ = avaliar(Wi, Wt)
print(f"{'critério':<22} {'direção':<14} {'R@1':>6} {'R@5':>6}")
for nome, i2t, t2i in (("mesmo dígito", a_i2t, a_t2i), ("dígito + atributos", b_i2t, b_t2i)):
    print(f"{nome:<22} {'imagem->texto':<14} {i2t[1]:>6.3f} {i2t[5]:>6.3f}")
    print(f"{nome:<22} {'texto->imagem':<14} {t2i[1]:>6.3f} {t2i[5]:>6.3f}")


# %% [bloco 7]
print(f"{'tau':>5} {'perda final':>12} {'cos par certo':>14} {'cos par errado':>15} {'R@1 i->t':>9} {'R@1 t->i':>9}")
res_tau = {}
for tau in (0.01, 0.05, 0.1, 0.5, 1.0):
    Wi_, Wt_, h = treinar_contrastivo(Xi[tr], T[tr], d=16, tau=tau, epocas=60, lote=64, lr=0.05, seed=0)
    a1, a2, _, _, S_ = avaliar(Wi_, Wt_)
    n = len(te); mask = np.eye(n, dtype=bool)
    certo, errado = S_[mask].mean(), S_[~mask].mean()
    res_tau[tau] = (h[-1], certo, errado, a1[1], a2[1])
    print(f"{tau:>5} {h[-1]:>12.3f} {certo:>14.3f} {errado:>15.3f} {a1[1]:>9.3f} {a2[1]:>9.3f}")


# %% [bloco 8]
prompts = [f"dígito {w}" for w in PALAVRAS]
Up = embutir(bow(prompts, voc), Wt)                         # 10 x 16
pred = (Ui @ Up.T).argmax(1)
print("acurácia zero-shot (descrições de classe) no teste:", round(float((pred == yt).mean()), 3))
print("acurácia ao acaso:", 0.1)
conf = np.zeros((10, 10), int)
for a, b in zip(yt, pred): conf[a, b] += 1
erros = [(PALAVRAS[a], PALAVRAS[b], int(conf[a, b])) for a in range(10) for b in range(10) if a != b and conf[a, b] >= 3]
print("confusões mais frequentes (verdadeiro, previsto, contagem):", erros)


# %% [bloco 9]
rng_e = np.random.default_rng(5)
E_tok = rng_e.normal(size=(10, 32)); E_tok /= np.linalg.norm(E_tok, axis=1, keepdims=True)   # embeddings de token (fictícios)
Zi_tr = Xi[tr] @ Wi                                                    # features visuais (16 dimensões)
Y_tr = E_tok[rotulos[tr]]                                              # alvo: embedding da palavra do dígito
lam = 1.0
Wp = np.linalg.solve(Zi_tr.T @ Zi_tr + lam * np.eye(16), Zi_tr.T @ Y_tr)   # projetor 16 -> 32 (ridge)
pseudo = (Xi[te] @ Wi) @ Wp
pseudo /= np.linalg.norm(pseudo, axis=1, keepdims=True)
vizinho = (pseudo @ E_tok.T).argmax(1)
print("formato do projetor:", Wp.shape)
print("palavra mais próxima do pseudotoken = dígito verdadeiro:", round(float((vizinho == yt).mean()), 3))
print("exemplo (3 imagens):", [(PALAVRAS[a], PALAVRAS[b]) for a, b in zip(yt[:3], vizinho[:3])])


# %% [bloco 10]
from llmdidatico.multimodal import stft
fs = 8000
t = np.arange(2 * fs) / fs
rg = np.random.default_rng(0)
x = np.zeros_like(t)
m1 = t < 0.8
x[m1] = np.sin(2 * np.pi * 440 * t[m1])                                   # 0 a 0,8 s: tom de 440 Hz
m2 = t >= 1.2
fase = 2 * np.pi * (300 * (t[m2] - 1.2) + 0.5 * (1500 - 300) / 0.8 * (t[m2] - 1.2) ** 2)
x[m2] = np.sin(fase)                                                      # 1,2 a 2,0 s: sobe de 300 a 1500 Hz
x += 0.05 * rg.standard_normal(len(x))                                    # ruído de fundo
print("amostras:", len(x), "| duração:", len(x) / fs, "s | frequência máxima representável:", fs / 2, "Hz")

n_fft, salto = 256, 64
Z = stft(x, n_fft, salto)
mag = np.abs(Z)
print("espectrograma:", Z.shape, "(frequências, quadros) | resolução em frequência:", fs / n_fft, "Hz | passo no tempo:", salto / fs, "s")
freqs = np.fft.rfftfreq(n_fft, 1 / fs)
for tq in (0.4, 1.6):
    q = int((tq * fs - n_fft / 2) / salto)
    print(f"frequência de maior energia perto de t = {tq} s: {freqs[mag[:, q].argmax()]:.1f} Hz")
print("frequência esperada em t = 1,6 s (chirp):", 300 + (1500 - 300) / 0.8 * (1.6 - 1.2), "Hz")

alias = np.sin(2 * np.pi * 5000 * t)                                      # 5 000 Hz amostrado a 8 000 Hz (acima de Nyquist)
fa = np.fft.rfftfreq(len(alias), 1 / fs)[np.abs(np.fft.rfft(alias)).argmax()]
print("tom de 5000 Hz amostrado a 8000 Hz aparece em:", fa, "Hz (aliasing)")
print("quadros por segundo com salto de 10 ms:", round(1 / 0.010), "| em 30 s:", 30 * round(1 / 0.010))


# %% [bloco 11]
from llmdidatico.multimodal import levenshtein
rg = np.random.default_rng(3)
confusao = {"0": "O", "1": "l", "5": "S", "8": "B", ",": "."}
def linha():
    return f"NF {rg.integers(1000, 9999)}  Itens {rg.integers(1, 40)}  Total R$ {rg.integers(1, 99)}.{rg.integers(100, 999)},{rg.integers(10, 99)}"
def ocr_simulado(texto, p, rg):
    return "".join(confusao[c] if (c in confusao and rg.random() < p) else c for c in texto)

linhas = [linha() for _ in range(2000)]
print("exemplo de linha:", linhas[0])
print(f"{'p (por caractere confundível)':<30} {'CER':>7} {'linhas com erro':>16} {'totais alterados':>17}")
for p in (0.005, 0.01, 0.02, 0.05):
    sai = [ocr_simulado(l, p, rg) for l in linhas]
    cer = sum(levenshtein(a, b) for a, b in zip(linhas, sai)) / sum(len(a) for a in linhas)
    com_erro = np.mean([a != b for a, b in zip(linhas, sai)])
    tot = np.mean([a.split("Total R$ ")[1] != b.split("Total R$ ")[1] for a, b in zip(linhas, sai)])
    print(f"{p:<30} {cer:>7.4f} {com_erro:>16.3f} {tot:>17.3f}")


# %% [bloco 12]
from llmdidatico.multimodal import agenda_beta, ruido_direto
T_ = 1000
beta, alpha, abar = agenda_beta(T_)
print("passos T =", T_, "| beta de", beta[0], "a", beta[-1])
for t_ in (0, 99, 499, 999):
    print(f"t = {t_ + 1:4d}: abar = {abar[t_]:.5f} | sqrt(abar) = {np.sqrt(abar[t_]):.4f} | sqrt(1 - abar) = {np.sqrt(1 - abar[t_]):.4f}")

rd = np.random.default_rng(0)
n = 100_000
mu_k, sg, w_k = np.array([-2.0, 2.0]), 0.3, np.array([0.5, 0.5])
x0 = rd.choice(mu_k, size=n, p=w_k) + sg * rd.standard_normal(n)
print(f"dados: média = {x0.mean():.3f}, desvio = {x0.std():.3f}")

x = x0.copy(); passo = {}
for t_ in range(T_):                                                  # passo a passo
    x = np.sqrt(alpha[t_]) * x + np.sqrt(beta[t_]) * rd.standard_normal(n)
    if t_ in (99, 499, 999): passo[t_] = x.copy()
print(f"{'t':>5} {'média passo':>12} {'desvio passo':>13} {'média fórmula':>14} {'desvio fórmula':>15}")
for t_ in (99, 499, 999):
    xf, _ = ruido_direto(x0, t_, abar, rd)
    print(f"{t_ + 1:>5} {passo[t_].mean():>12.3f} {passo[t_].std():>13.3f} {xf.mean():>14.3f} {xf.std():>15.3f}")


# %% [bloco 13]
def eps_otimo(xt, t_):
    """Estimador do ruído para a mistura gaussiana conhecida (denoiser analítico, não é uma rede treinada)."""
    ab = abar[t_]; v = ab * sg ** 2 + (1 - ab)
    logp = -0.5 * (xt[:, None] - np.sqrt(ab) * mu_k[None, :]) ** 2 / v + np.log(w_k)[None, :]
    r = np.exp(logp - logp.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    x0_k = mu_k[None, :] + (np.sqrt(ab) * sg ** 2 / v) * (xt[:, None] - np.sqrt(ab) * mu_k[None, :])
    x0_hat = (r * x0_k).sum(1)
    return (xt - np.sqrt(ab) * x0_hat) / np.sqrt(1 - ab)

xs = rd.standard_normal(20_000)                                       # começa de ruído puro
for t_ in range(T_ - 1, -1, -1):
    eps_hat = eps_otimo(xs, t_)
    xs = (xs - beta[t_] / np.sqrt(1 - abar[t_]) * eps_hat) / np.sqrt(alpha[t_])
    if t_ > 0: xs = xs + np.sqrt(beta[t_]) * rd.standard_normal(len(xs))
print(f"amostras geradas: média = {xs.mean():.3f}, desvio = {xs.std():.3f}")
print("fração em torno do modo -2 / +2:", round(float((xs < 0).mean()), 3), "/", round(float((xs >= 0).mean()), 3))
print("desvio dentro do modo +2 (dados: 0,300):", round(float(xs[xs >= 0].std()), 3))


# %% [bloco 14]
from llmdidatico import avaliacao as av
corretos = (pred == yt).astype(float)
cat = np.array([[a.strip() for a in c.split(",")[1:]] for c in np.array(legendas)[te]])   # [traço, lado] de cada imagem de teste
print(f"{'fatia':<16} {'n':>4} {'acurácia':>9} {'IC95% (bootstrap)':>20}")
for nome, mk in [("traço forte", cat[:, 0] == "traço forte"), ("traço leve", cat[:, 0] == "traço leve"),
                 ("à esquerda", cat[:, 1] == "à esquerda"), ("ao centro", cat[:, 1] == "ao centro"), ("à direita", cat[:, 1] == "à direita")]:
    m, lo, hi = av.bootstrap_ic(corretos[mk], n=1000, seed=0)
    print(f"{nome:<16} {mk.sum():>4} {m:>9.3f} {f'[{lo:.3f}; {hi:.3f}]':>20}")

ro = np.random.default_rng(9)
ruido = ro.uniform(0, 1, size=(300, 64)) - 0.5                            # "imagens" fora do domínio
sim_in = (Ui @ Up.T).max(1)
sim_out = (embutir(ruido, Wi) @ Up.T).max(1)
limiar = np.percentile(sim_in, 5)
print(f"similaridade máxima: dentro do domínio (média) = {sim_in.mean():.3f} | ruído (média) = {sim_out.mean():.3f}")
print(f"limiar (percentil 5 dentro do domínio) = {limiar:.3f}")
print(f"ruído aceito sem abstenção: 1.000 | ruído aceito com limiar: {(sim_out >= limiar).mean():.3f} | dígitos reais rejeitados: {(sim_in < limiar).mean():.3f}")


# %% [bloco 15]
print(f"{'imagem':<14} {'patch':>6} {'tokens visuais':>15}")
for (h, l, p) in [(224, 224, 16), (224, 224, 32), (448, 448, 16), (1024, 768, 16), (1024, 768, 32)]:
    print(f"{f'{h}x{l}':<14} {p:>6} {(h // p) * (l // p):>15}")
tok_texto = 200
for tok_img in (196, 784, 3072):
    print(f"1 imagem de {tok_img} tokens + texto de {tok_texto}: {tok_img + tok_texto} tokens de entrada; "
          f"em 10 000 requisições: {(tok_img + tok_texto) * 10_000:,} tokens".replace(",", "."))

