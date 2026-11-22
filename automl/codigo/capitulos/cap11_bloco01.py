# Capítulo 11 — Busca de arquitetura neural (NAS)
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import itertools, json, time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os; os.makedirs("resultados", exist_ok=True)

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, stratify=ytr, random_state=0)
sc = StandardScaler().fit(Xa); Xa, Xv, Xt = sc.transform(Xa), sc.transform(Xv), sc.transform(Xte)

LARGURAS = [8, 16, 32, 64, 128]; ATIVACOES = ["relu", "tanh"]
ARQS = [(h, a) for L in (1, 2, 3) for h in itertools.product(LARGURAS, repeat=L) for a in ATIVACOES]
print("Tamanho do espaço:", len(ARQS), "arquiteturas")

def n_parametros(arq):
    d = [64] + list(arq[0]) + [10]
    return sum(i * o + o for i, o in zip(d[:-1], d[1:]))

def treina(arq, iteracoes):
    return MLPClassifier(arq[0], activation=arq[1], max_iter=iteracoes, random_state=0).fit(Xa, ya)

tabela = []; t0 = time.perf_counter()
for arq in ARQS:
    m10, m40 = treina(arq, 10), treina(arq, 40)
    tabela.append({"arq": [list(arq[0]), arq[1]], "params": n_parametros(arq),
                   "v10": m10.score(Xv, yv), "v40": m40.score(Xv, yv), "teste": m40.score(Xt, yte)})
json.dump(tabela, open("resultados/f11_cache.json", "w"))
custo = time.perf_counter() - t0
print("Custo de construir a tabela: %d treinos (%d arquiteturas x 2 fidelidades) em %.0f s" % (2 * len(ARQS), len(ARQS), custo))

v = np.array([r["v40"] for r in tabela]); i = int(v.argmax())
print("Validação: pior = %.3f | mediana = %.3f | melhor = %.3f" % (v.min(), np.median(v), v.max()))
print("Melhor arquitetura:", tuple(tabela[i]["arq"][0]), tabela[i]["arq"][1], "| parâmetros:", tabela[i]["params"])
print("Seu acerto no TESTE: %.3f" % tabela[i]["teste"])
