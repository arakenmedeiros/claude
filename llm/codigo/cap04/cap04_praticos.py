"""Capítulo 4: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 4.1: Atenção do zero, máscara e teste de vazamento]
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def atencao(Q, K, V, mascara=None):
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    if mascara is not None:
        S = np.where(mascara, S, -1e9)
    A = softmax(S)
    return A @ V, A

rng = np.random.default_rng(0)
T, d = 6, 8
X = rng.normal(size=(T, d))
WQ, WK, WV = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
causal = np.tril(np.ones((T, T), dtype=bool))

saida, A = atencao(X @ WQ, X @ WK, X @ WV, causal)
print("linhas somam 1:", np.allclose(A.sum(axis=1), 1))
print("pesos com máscara causal:\n", A.round(2))

X2 = X.copy(); X2[-1] = rng.normal(size=d)
saida2, _ = atencao(X2 @ WQ, X2 @ WK, X2 @ WV, causal)
print("tokens 1-5 inalterados ao mudar o token 6:", np.allclose(saida[:5], saida2[:5]))

# versão incorreta: máscara aplicada depois do softmax, sem renormalizar
def atencao_errada(Q, K, V, mascara):
    A = softmax(Q @ K.T / np.sqrt(Q.shape[-1])) * mascara
    return A @ V, A
_, Ae = atencao_errada(X @ WQ, X @ WK, X @ WV, causal)
print("versão errada: as linhas ainda somam 1?", np.allclose(Ae.sum(axis=1), 1), "| soma da linha 1:", round(float(Ae[0].sum()), 3))


# %% [Exercício 4.2: Cabeças de atenção: mesmo custo em parâmetros, perfis diferentes]
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def multicabeca(X, WQ, WK, WV, WO, h):
    T, d = X.shape; dh = d // h
    sep = lambda M: M.reshape(T, h, dh).transpose(1, 0, 2)
    Q, K, V = sep(X @ WQ), sep(X @ WK), sep(X @ WV)
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(dh))
    Z = (A @ V).transpose(1, 0, 2).reshape(T, d)
    return Z @ WO, A

d, T = 64, 10
rng = np.random.default_rng(1)
X = rng.normal(size=(T, d))
WQ, WK, WV, WO = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))

print(f"{'h':>3} {'dh':>4} {'parâmetros':>11} {'formato dos pesos':>18} {'entropia média':>15} {'máx. = ln T':>12}")
for h in (1, 2, 4, 8):
    Z, A = multicabeca(X, WQ, WK, WV, WO, h)
    ent = -(A * np.log(A + 1e-12)).sum(-1).mean()
    print(f"{h:>3} {d // h:>4} {4 * d * d:>11} {str(A.shape):>18} {ent:>15.3f} {np.log(T):>12.3f}")


# %% [Exercício 4.3: Quando a ordem importa: saco de palavras versus posição]
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
letras = list("abcdefgh")
T, N = 6, 2000
seqs, y = [], []
for _ in range(N):
    s = list(rng.choice(letras[:6], size=T))
    px, py = rng.choice(T, size=2, replace=False)
    s[px], s[py] = "x", "y"
    seqs.append(s); y.append(int(px < py))
y = np.array(y)
vocab = letras[:6] + ["x", "y"]; ix = {t: i for i, t in enumerate(vocab)}

# (a) saco de palavras: contagens por token
Xbow = np.zeros((N, len(vocab)))
for n, s in enumerate(seqs):
    for t in s: Xbow[n, ix[t]] += 1

# (b) posição de x e de y em one-hot
Xpos = np.zeros((N, 2 * T))
for n, s in enumerate(seqs):
    Xpos[n, s.index("x")] = 1
    Xpos[n, T + s.index("y")] = 1

for nome, X in [("saco de palavras", Xbow), ("com posições", Xpos)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
    print(f"{nome:<18} acurácia no teste = {acc:.3f}")
print("proporção de rótulo 1:", y.mean().round(3))


# %% [Projeto: Medindo o custo da atenção]
import time
import numpy as np
from sklearn.linear_model import LinearRegression

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def multicabeca(X, WQ, WK, WV, WO, h):
    T, d = X.shape; dh = d // h
    sep = lambda M: M.reshape(T, h, dh).transpose(1, 0, 2)
    Q, K, V = sep(X @ WQ), sep(X @ WK), sep(X @ WV)
    A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(dh))
    return ((A @ V).transpose(1, 0, 2).reshape(T, d)) @ WO

d, h = 64, 4
rng = np.random.default_rng(0)
WQ, WK, WV, WO = (rng.normal(size=(d, d)).astype(np.float32) / 8 for _ in range(4))

Ts = [64, 128, 256, 512, 1024]
tempos = []
print(f"{'T':>6} {'tempo mediano (ms)':>19} {'memória das pontuações (MB, float32)':>38}")
for T in Ts:
    X = rng.normal(size=(T, d)).astype(np.float32)
    multicabeca(X, WQ, WK, WV, WO, h)                      # aquecimento
    rep = []
    for _ in range(5):
        t0 = time.perf_counter(); multicabeca(X, WQ, WK, WV, WO, h); rep.append(time.perf_counter() - t0)
    tempos.append(float(np.median(rep)))
    print(f"{T:>6} {1000 * tempos[-1]:>19.2f} {h * T * T * 4 / 1e6:>38.2f}")

reg = LinearRegression().fit(np.log(Ts).reshape(-1, 1), np.log(tempos))
print(f"expoente estimado (inclinação log-log): {reg.coef_[0]:.2f} | esperado: entre 1 (domínio linear) e 2 (domínio quadrático)")

