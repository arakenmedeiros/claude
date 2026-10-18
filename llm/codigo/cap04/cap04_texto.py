"""Capítulo 4: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def atencao(Q, K, V, mascara=None):
    """Atenção por produto escalar escalonado. Q:(T,dk) K:(T,dk) V:(T,dv). Retorna (saída, pesos)."""
    dk = Q.shape[-1]
    S = Q @ K.T / np.sqrt(dk)
    if mascara is not None:
        S = np.where(mascara, S, -1e9)
    A = softmax(S, axis=-1)
    return A @ V, A

Q = K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
saida, pesos = atencao(Q, K, V)
print("pontuações escalonadas:\n", (Q @ K.T / np.sqrt(2)).round(3))
print("pesos (cada linha soma 1):\n", pesos.round(3))
print("saída:\n", saida.round(3))


# %% [bloco 2]
rng = np.random.default_rng(0)
print("dk    var(q·k)  var(q·k/√dk)")
for dk in (4, 16, 64, 256, 1024):
    q = rng.normal(size=(20000, dk)); k = rng.normal(size=(20000, dk))
    d = (q * k).sum(axis=1)
    print(f"{dk:<5} {d.var():>8.1f} {(d / np.sqrt(dk)).var():>12.2f}")

print("\npeso máximo médio do softmax sobre 10 chaves:")
print("dk    sem escala  com escala")
for dk in (4, 64, 1024):
    q = rng.normal(size=(2000, 1, dk)); k = rng.normal(size=(2000, 10, dk))
    s = (q * k).sum(-1)
    print(f"{dk:<5} {softmax(s).max(1).mean():>10.3f} {softmax(s / np.sqrt(dk)).max(1).mean():>11.3f}")


# %% [bloco 3]
T = 3
mascara = np.tril(np.ones((T, T), dtype=bool))      # True onde o token pode olhar (j <= i)
saida_c, pesos_c = atencao(Q, K, V, mascara)
print("máscara:\n", mascara.astype(int))
print("pesos com máscara causal:\n", pesos_c.round(3))
print("saída com máscara:\n", saida_c.round(3))


# %% [bloco 4]
def atencao_multicabeca(X, WQ, WK, WV, WO, h, mascara=None):
    """X:(T,d). WQ,WK,WV,WO:(d,d). Divide d em h cabeças de tamanho d/h."""
    T, d = X.shape
    dh = d // h
    Q, K, V = X @ WQ, X @ WK, X @ WV                      # (T, d)
    def separa(M):                                         # (T, d) -> (h, T, dh)
        return M.reshape(T, h, dh).transpose(1, 0, 2)
    Qh, Kh, Vh = separa(Q), separa(K), separa(V)
    S = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(dh)           # (h, T, T)
    if mascara is not None:
        S = np.where(mascara, S, -1e9)
    A = softmax(S, axis=-1)                                # pesos por cabeça
    Z = A @ Vh                                             # (h, T, dh)
    Z = Z.transpose(1, 0, 2).reshape(T, d)                 # concatena as cabeças
    return Z @ WO, A

rng = np.random.default_rng(1)
T, d, h = 6, 16, 4
X = rng.normal(size=(T, d))
WQ, WK, WV, WO = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
Z, A = atencao_multicabeca(X, WQ, WK, WV, WO, h, mascara=np.tril(np.ones((T, T), dtype=bool)))
print("entrada:", X.shape, "| saída:", Z.shape, "| pesos:", A.shape, "(cabeças, T, T)")
print("cada linha de pesos soma 1:", np.allclose(A.sum(-1), 1))
print("parâmetros do bloco de atenção (sem vieses):", 4 * d * d, "= 4·d²")


# %% [bloco 5]
def posicional(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / (10000 ** (2 * i / d))
    PE = np.zeros((T, d))
    PE[:, 0::2] = np.sin(ang)
    PE[:, 1::2] = np.cos(ang)
    return PE

PE = posicional(10, 16)
print("PE[0, :6] =", PE[0, :6].round(3))
print("PE[1, :6] =", PE[1, :6].round(3))

# a atenção sem posições é equivariante à permutação
rng = np.random.default_rng(2)
T, d, h = 6, 16, 4
X = rng.normal(size=(T, d))
WQ, WK, WV, WO = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))
perm = rng.permutation(T)
Z1, _ = atencao_multicabeca(X, WQ, WK, WV, WO, h)
Z2, _ = atencao_multicabeca(X[perm], WQ, WK, WV, WO, h)
print("sem posições: saída(embaralhado) == embaralhada(saída)?", np.allclose(Z2, Z1[perm]))
X_pos = X + posicional(T, d)
Z3, _ = atencao_multicabeca(X_pos, WQ, WK, WV, WO, h)
Z4, _ = atencao_multicabeca(X[perm] + posicional(T, d), WQ, WK, WV, WO, h)   # mesmos tokens embaralhados, posições fixas
print("com posições, a ordem passa a importar:", not np.allclose(Z4, Z3[perm]))


# %% [bloco 6]
def layernorm(x, gamma=None, beta=None, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    if gamma is not None: y = y * gamma
    if beta is not None: y = y + beta
    return y

def rmsnorm(x, gamma=None, eps=1e-5):
    y = x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return y * gamma if gamma is not None else y

x = np.array([[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]])
y = layernorm(x)
print("LayerNorm linha 1:", y[0].round(3), "| média ≈", y[0].mean().round(6), "| desvio ≈", y[0].std().round(3))
print("LayerNorm linha 2:", y[1].round(3))
print("RMSNorm   linha 1:", rmsnorm(x)[0].round(3))


# %% [bloco 7]
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

class BlocoTransformer:
    def __init__(self, d, h, dff, rng):
        s = 1 / np.sqrt(d)
        self.h = h
        self.WQ, self.WK, self.WV, self.WO = (rng.normal(size=(d, d)) * s for _ in range(4))
        self.W1 = rng.normal(size=(d, dff)) * s; self.b1 = np.zeros(dff)
        self.W2 = rng.normal(size=(dff, d)) / np.sqrt(dff); self.b2 = np.zeros(d)

    def __call__(self, X, mascara):
        a, _ = atencao_multicabeca(layernorm(X), self.WQ, self.WK, self.WV, self.WO, self.h, mascara)
        X = X + a                                                    # residual 1
        f = gelu(layernorm(X) @ self.W1 + self.b1) @ self.W2 + self.b2
        return X + f                                                 # residual 2

    def n_parametros(self):
        return sum(p.size for p in (self.WQ, self.WK, self.WV, self.WO, self.W1, self.b1, self.W2, self.b2))

rng = np.random.default_rng(3)
T, d, h, dff = 8, 32, 4, 128
bloco = BlocoTransformer(d, h, dff, rng)
causal = np.tril(np.ones((T, T), dtype=bool))
X = rng.normal(size=(T, d))
Y = bloco(X, causal)
print("formato entrada/saída:", X.shape, Y.shape)
print("parâmetros do bloco:", bloco.n_parametros(), "| 12·d² =", 12 * d * d, "(diferença = vieses)")

X2 = X.copy(); X2[5:] = rng.normal(size=(T - 5, d))                 # muda só os tokens 6, 7 e 8
Y2 = bloco(X2, causal)
print("saídas dos tokens 1-5 inalteradas (causal)?", np.allclose(Y[:5], Y2[:5]))
print("saídas dos tokens 6-8 mudaram?", not np.allclose(Y[5:], Y2[5:]))
Yb = bloco(X, None); Yb2 = bloco(X2, None)
print("sem máscara, os tokens 1-5 também mudam?", not np.allclose(Yb[:5], Yb2[:5]))


# %% [bloco 8]
def parametros_decoder(N, d, V, dff=None, compartilhado=True):
    dff = dff or 4 * d
    por_bloco = 4 * d * d + 2 * d * dff
    emb = V * d * (1 if compartilhado else 2)
    return N * por_bloco + emb

n = parametros_decoder(12, 768, 50_000)
print(f"parâmetros aproximados: {n:,}".replace(",", "."), "| memória em float16:", round(n * 2 / 1e6), "MB")


# %% [bloco 9]
def memoria_scores_gb(T, h, bytes_por_elem=2):
    return T * T * h * bytes_por_elem / 1e9

for T in (1024, 4096, 16384):
    print(f"T = {T:>6}: matriz de pontuações (h=32, float16, uma camada) = {memoria_scores_gb(T, 32):8.2f} GB")

