"""Capítulo 2: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 2.1: Vizinhos mais próximos por cosseno]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llmdidatico import DOCS

ids = [d[0] for d in DOCS]
titulos = {d[0]: d[2] for d in DOCS}
textos = [d[2] + ". " + d[3] for d in DOCS]

X = TfidfVectorizer(sublinear_tf=True).fit_transform(textos)
S = cosine_similarity(X)
print("Formato de X:", X.shape)
print("Normas das 5 primeiras linhas:", np.linalg.norm(X[:5].toarray(), axis=1).round(3))
print("Cosseno == produto escalar?", np.allclose(S, (X @ X.T).toarray()))

for alvo in ("D12", "D02", "D37"):
    i = ids.index(alvo)
    ordem = [j for j in np.argsort(-S[i]) if j != i][:3]
    print(f"\nVizinhos de {alvo} ({titulos[alvo]}):")
    for j in ordem:
        print(f"  {S[i, j]:.3f}  {ids[j]}  {titulos[ids[j]]}")


# %% [Exercício 2.2: Softmax, entropia cruzada e perplexidade do zero]
import numpy as np
from scipy.special import softmax as sp_softmax
from sklearn.metrics import log_loss

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def entropia_cruzada(Z, alvos):
    P = softmax(Z)
    return float(-np.log(P[np.arange(len(alvos)), alvos]).mean())

rng = np.random.default_rng(0)
N, V = 1000, 50
Z = rng.normal(size=(N, V))
alvos = rng.integers(0, V, size=N)

print("softmax igual ao do SciPy:", np.allclose(softmax(Z), sp_softmax(Z, axis=1)))
ce = entropia_cruzada(Z, alvos)
print("entropia cruzada (NumPy):", round(ce, 4))
print("log_loss (scikit-learn): ", round(log_loss(alvos, softmax(Z), labels=list(range(V))), 4))
print("perplexidade:", round(float(np.exp(ce)), 2))
print("perda do modelo uniforme (ln V):", round(float(np.log(V)), 4), "| perplexidade V =", V)


# %% [Exercício 2.3: Treinando uma regressão logística multiclasse com gradientes]
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

X, y = make_blobs(n_samples=600, centers=3, cluster_std=2.0, random_state=0)
Y = np.eye(3)[y]

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def treinar(eta, epocas=200):
    W = np.zeros((2, 3)); b = np.zeros(3); hist = []
    for _ in range(epocas):
        P = softmax(X @ W + b)
        hist.append(float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean()))
        W -= eta * X.T @ (P - Y) / len(y)
        b -= eta * (P - Y).mean(axis=0)
    acc = float(((X @ W + b).argmax(axis=1) == y).mean())
    return hist, acc

print(f"{'eta':>7} {'perda inicial':>14} {'perda final':>12} {'acurácia':>9}")
for eta in (0.001, 0.05, 1.0, 50.0):
    h, acc = treinar(eta)
    print(f"{eta:>7} {h[0]:>14.4f} {h[-1]:>12.4f} {acc:>9.3f}")

ref = LogisticRegression(max_iter=1000).fit(X, y)
print("scikit-learn LogisticRegression: acurácia =", round(float(ref.score(X, y)), 3))


# %% [Projeto: Mini-laboratório de otimização reprodutível]
import json, sys
import numpy as np, sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True)
X = X / 16.0

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def treinar(Xtr, ytr, eta=0.5, epocas=300):
    K = 10; W = np.zeros((Xtr.shape[1], K)); b = np.zeros(K); Y = np.eye(K)[ytr]
    for _ in range(epocas):
        P = softmax(Xtr @ W + b)
        W -= eta * Xtr.T @ (P - Y) / len(ytr)
        b -= eta * (P - Y).mean(axis=0)
    P = softmax(Xtr @ W + b)
    perda = float(-np.log(P[np.arange(len(ytr)), ytr] + 1e-12).mean())
    return W, b, perda

registro = []; acc_prop = []; acc_ref = []
for semente in range(5):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=semente)
    W, b, perda = treinar(Xtr, ytr)
    acc = float(((Xte @ W + b).argmax(axis=1) == yte).mean())
    ref = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    acc_prop.append(acc); acc_ref.append(float(ref.score(Xte, yte)))
    registro.append({"semente": semente, "perda_treino": round(perda, 4), "acuracia_teste": round(acc, 4)})
    print(registro[-1])

print(f"NumPy (softmax + gradiente): {np.mean(acc_prop):.4f} ± {np.std(acc_prop, ddof=1):.4f}")
print(f"scikit-learn LogisticRegression: {np.mean(acc_ref):.4f} ± {np.std(acc_ref, ddof=1):.4f}")

saida = {"python": sys.version.split()[0], "numpy": np.__version__, "scikit-learn": sklearn.__version__,
         "parametros": {"eta": 0.5, "epocas": 300, "teste": 0.3}, "execucoes": registro}
with open("registro_digits.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)
print("registro salvo em registro_digits.json")

