# Capítulo 11 — solução de referência (PROJ)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
X = np.hstack([X, rng.normal(size=(len(y), 70))])
y = np.where(rng.random(len(y)) < 0.10, 1 - y, y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, stratify=ytr, random_state=0)
sc = StandardScaler().fit(Xa); Xa, Xv, Xt = sc.transform(Xa), sc.transform(Xv), sc.transform(Xte)
D = X.shape[1]
LARG, ATIV = [4, 8, 16, 32, 64], ["relu", "tanh"]
print("Tamanho do espaço:", len(ATIV) * sum(len(LARG) ** L for L in range(1, 4)))
cache = {}
def avalia(a):
    if a not in cache:
        m = MLPClassifier(a[0], activation=a[1], max_iter=100, random_state=0).fit(Xa, ya)
        d = [D, *a[0], 2]; cache[a] = (m.score(Xv, yv), sum(i * o + o for i, o in zip(d[:-1], d[1:])), m.score(Xt, yte))
    return cache[a]
def sorteia(r): return (tuple(int(r.choice(LARG)) for _ in range(int(r.integers(1, 4)))), str(r.choice(ATIV)))
def muta(a, r):
    h, at = list(a[0]), a[1]; op = int(r.integers(4))
    if op == 0: h[int(r.integers(len(h)))] = int(r.choice(LARG))
    elif op == 1 and len(h) < 3: h.insert(int(r.integers(len(h) + 1)), int(r.choice(LARG)))
    elif op == 2 and len(h) > 1: h.pop(int(r.integers(len(h))))
    else: at = "tanh" if at == "relu" else "relu"
    return (tuple(h), at)
B = 40; vistas = set(); melhores = {"aleatória": [], "hill climbing": []}
for s in range(3):
    r = np.random.default_rng(s); h = [sorteia(r) for _ in range(B)]
    melhores["aleatória"].append(max(avalia(a)[0] for a in h)); vistas.update(h)
    r = np.random.default_rng(50 + s); atual = sorteia(r); melhor = avalia(atual)[0]; vistas.add(atual)
    for _ in range(B - 1):
        f = muta(atual, r); vistas.add(f)
        if avalia(f)[0] > melhor: atual, melhor = f, avalia(f)[0]
    melhores["hill climbing"].append(melhor)
for k, v in melhores.items(): print(f"{k:<14s} melhor validação (3 sementes): {np.round(v, 3).tolist()}  média = {np.mean(v):.3f}")
T = [(a,) + avalia(a) for a in vistas]            # (arq, validação, parâmetros, teste)
front = [t for t in T if not any((u[1] >= t[1] and u[2] <= t[2]) and (u[1] > t[1] or u[2] < t[2]) for u in T)]
front.sort(key=lambda t: t[2])
print("Arquiteturas distintas avaliadas:", len(T), "| na fronteira:", len(front))
for a, v, p, _ in front[-6:]: print(f"  {str(a[0]):<14s} {a[1]:<5s} parâmetros = {p:>5d}  validação = {v:.3f}")
melhor = max(T, key=lambda t: t[1]); pequena = min([t for t in T if t[1] >= melhor[1] - 0.01], key=lambda t: t[2])
print(f"\nMais acurada na validação : {melhor[0][0]} {melhor[0][1]} | {melhor[2]} parâmetros | val = {melhor[1]:.3f} | TESTE = {melhor[3]:.3f}")
print(f"Menor dentro de 1 ponto    : {pequena[0][0]} {pequena[0][1]} | {pequena[2]} parâmetros | val = {pequena[1]:.3f} | TESTE = {pequena[3]:.3f}")
print(f"Redução de parâmetros: {100 * (1 - pequena[2] / melhor[2]):.0f}%  (teste = {len(yte)} exemplos)")
