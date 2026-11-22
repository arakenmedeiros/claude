# Capítulo 7 — solução de referência (PROJ)
import copy
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

X0, y0 = load_breast_cancer(return_X_y=True)
rng_d = np.random.default_rng(0)
idx = rng_d.choice(len(y0), 250, replace=False)
X = np.hstack([X0[idx], rng_d.normal(size=(250, 100))]); y = y0[idx]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
cv = StratifiedKFold(3, shuffle=True, random_state=0)
KS = [5, 10, 20, 40, 0]                                   # 0 = todos os atributos

def sortear_modelo(rng, m):
    if m == "logistica": return {"C": float(10 ** rng.uniform(-3, 2))}
    if m == "knn":       return {"kv": int(rng.integers(1, 25))}
    return {"arvores": int(rng.integers(20, 120)), "prof": int(rng.integers(2, 15))}

def sortear(rng):
    g = {"escala": str(rng.choice(["nenhuma", "padrao"])), "k": int(rng.choice(KS)), "modelo": str(rng.choice(["logistica", "knn", "floresta"]))}
    g.update(sortear_modelo(rng, g["modelo"])); return g

def construir(g):
    passos = [StandardScaler()] if g["escala"] == "padrao" else []
    if g["k"] > 0: passos.append(SelectKBest(f_classif, k=g["k"]))
    m = g["modelo"]
    if m == "logistica": passos.append(LogisticRegression(C=g["C"], max_iter=500))
    elif m == "knn":     passos.append(KNeighborsClassifier(g["kv"]))
    else:                passos.append(RandomForestClassifier(g["arvores"], max_depth=g["prof"], random_state=0, n_jobs=-1))
    return make_pipeline(*passos)

def n_atributos(g): return g["k"] if g["k"] > 0 else X.shape[1]

def aptidao(g, cache):
    chave = str(sorted(g.items()))
    if chave not in cache: cache[chave] = float(cross_val_score(construir(g), Xtr, ytr, cv=cv).mean())
    return cache[chave]

def cruzar(a, b, rng):
    f = {k: (a[k] if rng.random() < 0.5 else b[k]) for k in ("escala", "k")}
    pai = a if rng.random() < 0.5 else b
    f.update({k: v for k, v in pai.items() if k not in ("escala", "k")}); return f

def mutar(g, rng, p=0.3):
    g = copy.deepcopy(g)
    if rng.random() < p: g["escala"] = str(rng.choice(["nenhuma", "padrao"]))
    if rng.random() < p: g["k"] = int(rng.choice(KS))
    if rng.random() < p:
        g = {k: g[k] for k in ("escala", "k")}; g["modelo"] = str(rng.choice(["logistica", "knn", "floresta"])); g.update(sortear_modelo(rng, g["modelo"]))
    return g

def evoluir(semente, pop=12, geracoes=5, elite=2):
    rng = np.random.default_rng(semente); cache = {}
    P = [sortear(rng) for _ in range(pop)]; todos = []
    for ger in range(geracoes + 1):
        f = np.array([aptidao(g, cache) for g in P]); todos += list(zip(P, f))
        if ger == geracoes: break
        ordem = np.argsort(-f); nova = [copy.deepcopy(P[i]) for i in ordem[:elite]]
        torneio = lambda: P[max(rng.choice(pop, 3, replace=False), key=lambda i: f[i])]
        while len(nova) < pop: nova.append(mutar(cruzar(torneio(), torneio(), rng), rng))
        P = nova
    return cache, todos

def aleatoria(semente, n):
    rng = np.random.default_rng(500 + semente); cache = {}
    return max(aptidao(sortear(rng), cache) for _ in range(n))

resultados = []
for s in range(3):
    cache, todos = evoluir(s)
    resultados.append((max(cache.values()), aleatoria(s, len(cache)), len(cache)))
    if s == 0: todos0 = todos
for s, (a, b, n) in enumerate(resultados): print(f"semente {s}: {n} avaliações | genético = {a:.3f} | aleatória = {b:.3f}")

unicos = {str(sorted(g.items())): (g, float(f)) for g, f in todos0}
pontos = [(n_atributos(g), f, g) for g, f in unicos.values()]
domina = lambda a, b: a[0] <= b[0] and a[1] >= b[1] and (a[0] < b[0] or a[1] > b[1])
fronteira = sorted([p for p in pontos if not any(domina(q, p) for q in pontos)], key=lambda p: p[0])
fronteira = list({(n, round(f, 4)): (n, f, g) for n, f, g in fronteira}.values())          # remove duplicatas (mesmos objetivos)
print("\nFronteira de Pareto (semente 0): atributos | acurácia de validação | modelo")
for n, f, g in fronteira: print(f"  {n:9d} | {f:.3f} | {g['modelo']}, escala={g['escala']}")

melhor = max(fronteira, key=lambda p: p[1]); limite = melhor[1] - 0.01
enxuto = min([p for p in fronteira if p[1] >= limite], key=lambda p: p[0])
for nome, p in [("mais acurado", melhor), ("enxuto", enxuto)]:
    teste = construir(p[2]).fit(Xtr, ytr).score(Xte, yte)
    print(f"Pipeline {nome:12s}: {p[0]:3d} atributos | validação = {p[1]:.3f} | TESTE FINAL = {teste:.3f}")
