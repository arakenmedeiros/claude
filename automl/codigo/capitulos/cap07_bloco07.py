# Capítulo 7 — Métodos evolutivos e programação genética
# Bloco 7. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
import copy
import numpy as np
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Dados: 300 imagens de Digits + 200 colunas de ruído
X0, y0 = load_digits(return_X_y=True)
X0 = X0 / 16.0
rng_dados = np.random.default_rng(0)
idx = rng_dados.choice(len(y0), 300, replace=False)
X = np.hstack([X0[idx], rng_dados.normal(size=(300, 200))])
y = y0[idx]
print("Formato dos dados:", X.shape, "(64 colunas úteis + 200 de ruído)")

def sortear_modelo(rng, modelo):
    """Sorteia os hiperparâmetros do modelo escolhido."""
    if modelo == "knn":       return {"k": int(rng.integers(1, 30))}
    if modelo == "svm":       return {"C": float(10 ** rng.uniform(-2, 3)), "gamma": float(10 ** rng.uniform(-5, 0))}
    if modelo == "logistica": return {"C": float(10 ** rng.uniform(-3, 3))}
    return {"arvores": int(rng.integers(20, 150)), "prof": int(rng.integers(3, 20))}

def sortear_individuo(rng):
    g = {"escala": str(rng.choice(["nenhuma", "padrao", "minmax"])),
         "pca": int(rng.choice([0, 0, 10, 20, 30, 40])),
         "modelo": str(rng.choice(["knn", "svm", "logistica", "floresta"]))}
    g.update(sortear_modelo(rng, g["modelo"]))
    return g
def construir(g):
    """Traduz o genoma em um pipeline do scikit-learn."""
    passos = []
    if g["escala"] == "padrao": passos.append(StandardScaler())
    if g["escala"] == "minmax": passos.append(MinMaxScaler())
    if g["pca"] > 0:            passos.append(PCA(g["pca"], random_state=0))
    m = g["modelo"]
    if m == "knn":        passos.append(KNeighborsClassifier(g["k"]))
    elif m == "svm":      passos.append(SVC(C=g["C"], gamma=g["gamma"]))
    elif m == "logistica": passos.append(LogisticRegression(C=g["C"], max_iter=300))
    else:                 passos.append(RandomForestClassifier(g["arvores"], max_depth=g["prof"], random_state=0, n_jobs=-1))
    return make_pipeline(*passos)

def aptidao(g, X, y, cache):
    chave = str(sorted(g.items()))
    if chave not in cache:
        cache[chave] = float(cross_val_score(construir(g), X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0)).mean())
    return cache[chave]
def cruzar(a, b, rng):
    filho = {}
    for k in ("escala", "pca"):                                 # cada gene vem de um dos pais
        filho[k] = a[k] if rng.random() < 0.5 else b[k]
    pai = a if rng.random() < 0.5 else b                        # o modelo (com seus hiperparâmetros) vem de um pai
    filho["modelo"] = pai["modelo"]
    filho.update({k: v for k, v in pai.items() if k not in ("escala", "pca", "modelo")})
    return filho

def mutar(g, rng, p=0.3):
    g = copy.deepcopy(g)
    if rng.random() < p: g["escala"] = str(rng.choice(["nenhuma", "padrao", "minmax"]))
    if rng.random() < p: g["pca"] = int(rng.choice([0, 0, 10, 20, 30, 40]))
    if rng.random() < p * 0.7:                                  # às vezes troca o modelo inteiro
        g["modelo"] = str(rng.choice(["knn", "svm", "logistica", "floresta"]))
        for k in [k for k in g if k not in ("escala", "pca", "modelo")]: del g[k]
        g.update(sortear_modelo(rng, g["modelo"]))
    else:                                                       # ou ajusta levemente os hiperparâmetros
        for k, v in list(g.items()):
            if k in ("escala", "pca", "modelo") or rng.random() >= 0.6: continue
            if k in ("C", "gamma"): g[k] = float(np.clip(v * 10 ** rng.normal(0, 0.4), 1e-5 if k == "gamma" else 1e-3, 1 if k == "gamma" else 1e3))
            elif k == "k":          g[k] = int(np.clip(v + rng.integers(-3, 4), 1, 40))
            elif k == "arvores":    g[k] = int(np.clip(v + rng.integers(-20, 21), 10, 200))
            elif k == "prof":       g[k] = int(np.clip(v + rng.integers(-3, 4), 2, 25))
    return g
def evoluir(X, y, semente, pop=14, geracoes=5, elite=2):
    rng = np.random.default_rng(semente)
    cache, populacao, historico, todos = {}, [sortear_individuo(rng) for _ in range(pop)], [], []
    for ger in range(geracoes + 1):
        notas = np.array([aptidao(g, X, y, cache) for g in populacao])
        todos += list(zip(populacao, notas))
        historico.append((len(cache), float(notas.max()), float(notas.mean())))
        if ger == geracoes:
            break
        ordem = np.argsort(-notas)
        nova = [copy.deepcopy(populacao[i]) for i in ordem[:elite]]                 # elitismo
        torneio = lambda: populacao[max(rng.choice(pop, 3, replace=False), key=lambda i: notas[i])]
        while len(nova) < pop:
            nova.append(mutar(cruzar(torneio(), torneio(), rng), rng))
        populacao = nova
    return historico, todos

historico, todos = evoluir(X, y, semente=0)
print("geração | pipelines avaliados | melhor nota | nota média da população")
for ger, (n, melhor, media) in enumerate(historico):
    print(f"{ger:7d} | {n:19d} | {melhor:11.3f} | {media:.3f}")
# --- fim dos blocos anteriores ---
import numpy as np

# Cada indivíduo avaliado: (genoma, nota). Objetivo 1: nota (maximizar). Objetivo 2: nº de atributos (minimizar).
avaliados = {}
for g, nota in todos:
    avaliados[str(sorted(g.items()))] = (g, float(nota))
pontos = [(g["pca"] if g["pca"] > 0 else X.shape[1], nota, g) for g, nota in avaliados.values()]

def domina(a, b):
    """a domina b: não é pior em nenhum objetivo e é melhor em pelo menos um."""
    return a[0] <= b[0] and a[1] >= b[1] and (a[0] < b[0] or a[1] > b[1])

fronteira = [p for p in pontos if not any(domina(q, p) for q in pontos)]
fronteira.sort(key=lambda p: p[0])
print(f"Pipelines distintos avaliados: {len(pontos)} | na fronteira de Pareto: {len(fronteira)}")
print("\natributos | acurácia | pipeline")
for n_atr, nota, g in fronteira:
    ajustes = {k: round(v, 3) if isinstance(v, float) else v for k, v in g.items() if k not in ("escala", "pca", "modelo")}
    descricao = f"escala={g['escala']}, PCA={g['pca']}, {g['modelo']} {ajustes}"
    print(f"{n_atr:9d} | {nota:8.3f} | {descricao}")
