"""Capítulo 2: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print("produto escalar:", a @ b)
print("normas:", np.linalg.norm(a).round(3), np.linalg.norm(b).round(3))

def cosseno(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

print("cosseno(a, b):", round(cosseno(a, b), 4))
print("cosseno(a, -a):", round(cosseno(a, -a), 4))
print("cosseno((1,0), (0,1)):", cosseno(np.array([1.0, 0.0]), np.array([0.0, 1.0])))

W = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)   # formato (3, 2)
x = np.array([10.0, 1.0])                              # formato (2,)
print("W @ x =", W @ x, "| formato:", (W @ x).shape)

T = np.zeros((4, 7, 16))                               # lote 4, 7 tokens, dimensão 16
print("formato do tensor:", T.shape, "| elementos:", T.size)


# %% [bloco 2]
def softmax_ingenuo(z):
    e = np.exp(z)
    return e / e.sum()

def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max()                       # estabilidade numérica
    e = np.exp(z)
    return e / e.sum()

z = np.array([2.0, 1.0, 0.1])
print("softmax(z)       :", softmax(z).round(4), "| soma =", softmax(z).sum().round(6))
for T in (0.5, 1.0, 2.0):
    print(f"temperatura {T:>3}:", softmax(z, T).round(4))

grandes = np.array([1000.0, 999.0, 998.0])
with np.errstate(all="ignore"):
    print("ingênuo com logits grandes:", softmax_ingenuo(grandes))
print("estável com logits grandes:", softmax(grandes).round(4))


# %% [bloco 3]
def entropia(p):
    p = np.asarray(p, float)
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())

def entropia_cruzada(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(-(p * np.log(q)).sum())

def kl(p, q):
    return entropia_cruzada(p, q) - entropia(p)

p_mod = softmax(np.array([2.0, 1.0, 0.1]))
print("surpresa por token (nats):", (-np.log(p_mod)).round(3))
print("H(p_mod) =", round(entropia(p_mod), 3), "| máximo ln 3 =", round(np.log(3), 3))

p = np.array([0.7, 0.2, 0.1]); q = np.array([0.5, 0.3, 0.2])
print("H(p) =", round(entropia(p), 3), "| H(p,q) =", round(entropia_cruzada(p, q), 3), "| KL(p||q) =", round(kl(p, q), 3), "| KL(q||p) =", round(kl(q, p), 3))

perda_media = -np.log(p_mod[0])
print("perda =", round(float(perda_media), 3), "| perplexidade = exp(perda) =", round(float(np.exp(perda_media)), 3))


# %% [bloco 4]
def perda_ce(z, k):
    return -np.log(softmax(z)[k])

z = np.array([2.0, 1.0, 0.1]); k = 0
y = np.eye(3)[k]
grad_analitico = softmax(z) - y

def grad_numerico(f, z, h=1e-6):
    g = np.zeros_like(z)
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += h; zm[i] -= h
        g[i] = (f(zp) - f(zm)) / (2 * h)
    return g

grad_num = grad_numerico(lambda zz: perda_ce(zz, k), z)
print("gradiente p - y      :", grad_analitico.round(4))
print("gradiente numérico   :", grad_num.round(4))
print("diferença máxima     :", float(np.abs(grad_analitico - grad_num).max()))

# descida do gradiente nos próprios logits: a perda cai e p[k] sobe
zt = z.copy()
for passo in range(4):
    zt = zt - 0.5 * (softmax(zt) - y)
    print(f"passo {passo + 1}: p[k] = {softmax(zt)[k]:.3f} | perda = {perda_ce(zt, k):.3f}")


# %% [bloco 5]
def memoria_gb(n_parametros, bytes_por_parametro):
    return n_parametros * bytes_por_parametro / 1e9

N = 7e9   # modelo hipotético
for nome, b in [("float32", 4), ("float16", 2), ("int8", 1), ("4 bits", 0.5)]:
    print(f"{nome:<8}: {memoria_gb(N, b):6.1f} GB só para os pesos")

M = np.arange(12, dtype=np.float32).reshape(3, 4)
v = np.array([100, 200, 300, 400], dtype=np.float32)
print("broadcasting (3,4)+(4,):\n", M + v)
print("tipo e bytes por elemento:", M.dtype, M.itemsize)


# %% [bloco 6]
import json, platform, sys

def registrar_experimento(nome, parametros, metricas, caminho=None):
    registro = {
        "experimento": nome,
        "python": sys.version.split()[0],
        "plataforma": platform.system(),
        "numpy": np.__version__,
        "parametros": parametros,
        "metricas": metricas,
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2)
    if caminho:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(texto)
    return registro

rng = np.random.default_rng(0)
amostra = rng.normal(size=5).round(3)
reg = registrar_experimento("demo-semente", {"semente": 0, "n": 5}, {"amostra": amostra.tolist()})
print(json.dumps({k: reg[k] for k in ("experimento", "parametros", "metricas")}, ensure_ascii=False))
print("mesma semente, mesma amostra:", np.allclose(np.random.default_rng(0).normal(size=5).round(3), amostra))

