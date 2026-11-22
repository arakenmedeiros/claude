# Capítulo 11 — Busca de arquitetura neural (NAS)
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import itertools, json, time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import os; os.makedirs("resultados", exist_ok=True)

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, stratify=ytr, random_state=0)
sc = StandardScaler().fit(Xa); Xa, Xv, Xt = sc.transform(Xa), sc.transform(Xv), sc.transform(Xte)

T = json.load(open("resultados/f11_cache.json"))
ARQS = [(tuple(r["arq"][0]), r["arq"][1]) for r in T]
VAL = np.array([[r["v40"] for r in T], [0.0] * len(T), [0.0] * len(T)])         # linhas = sementes 0, 1, 2
TES = np.array([[r["teste"] for r in T], [0.0] * len(T), [0.0] * len(T)])
t0 = time.perf_counter()
for i, (h, ativ) in enumerate(ARQS):
    for k, sem in ((1, 1), (2, 2)):                                                # duas sementes de treino a mais
        m = MLPClassifier(h, activation=ativ, max_iter=40, random_state=sem).fit(Xa, ya)
        VAL[k, i], TES[k, i] = m.score(Xv, yv), m.score(Xt, yte)
print("Custo das duas sementes extras: %d treinos em %.0f s" % (2 * len(ARQS), time.perf_counter() - t0))
verdade, teste = VAL.mean(0), TES.mean(0)                                          # médias das 3 sementes
dp = VAL.std(0)
print("Desvio entre sementes de treino (310 arquiteturas): mediano = %.3f | p90 = %.3f | máx. = %.3f" % (np.median(dp), np.percentile(dp, 90), dp.max()))
print("Spearman entre as notas das sementes 0 e 1: %.2f | sementes 0 e 2: %.2f" % (spearmanr(VAL[0], VAL[1])[0], spearmanr(VAL[0], VAL[2])[0]))
print("Melhor do espaço pela média das 3 sementes: %.3f | pela semente 0: %.3f" % (verdade.max(), VAL[0].max()))

idx = {a: i for i, a in enumerate(ARQS)}; LARGURAS = [8, 16, 32, 64, 128]; ATIVACOES = ["relu", "tanh"]
posto = lambda i: 1 + int((verdade > verdade[i]).sum())

def mutacao(a, r):
    h, ativ = list(a[0]), a[1]; op = r.integers(4)
    if op == 0: h[r.integers(len(h))] = int(r.choice(LARGURAS))
    elif op == 1 and len(h) < 3: h.insert(r.integers(len(h) + 1), int(r.choice(LARGURAS)))
    elif op == 2 and len(h) > 1: h.pop(r.integers(len(h)))
    else: ativ = [x for x in ATIVACOES if x != ativ][0]
    return (tuple(h), ativ)

def busca(estrategia, B, r, P=8, S=3):
    nota = lambda a: VAL[r.integers(3), idx[a]]                                    # avaliação RUIDOSA: semente sorteada
    if estrategia == "aleatória":
        h = [ARQS[i] for i in r.choice(len(ARQS), B, replace=False)]; return max(h, key=nota)
    vistas, hist, pop = set(), [], []
    while len(hist) < P:
        a = ARQS[r.integers(len(ARQS))]
        if a not in vistas: vistas.add(a); hist.append((a, nota(a))); pop.append((a, hist[-1][1]))
    while len(hist) < B:
        pai = max([pop[i] for i in r.choice(len(pop), S, replace=False)], key=lambda t: t[1])[0]
        for _ in range(50):
            filha = mutacao(pai, r)
            if filha not in vistas: break
        else: filha = ARQS[r.integers(len(ARQS))]
        if filha in vistas: continue
        vistas.add(filha); v = nota(filha); hist.append((filha, v)); pop.append((filha, v)); pop.pop(0)
    return max(hist, key=lambda t: t[1])[0]

print("\nBusca com avaliação RUIDOSA (30 avaliações, 300 repetições); verdade = média das 3 sementes:")
for nome in ("aleatória", "evolução"):
    esc = [idx[busca(nome, 30, np.random.default_rng(s))] for s in range(300)]
    ps = np.array([posto(i) for i in esc]); tt = np.array([teste[i] for i in esc]); vv = np.array([verdade[i] for i in esc])
    print(f"{nome:<10s} nota verdadeira = {np.median(vv):.3f} | posto mediano = {np.median(ps):.1f} | top 5% = {100 * np.mean(ps <= 16):.0f}% | teste = {tt.mean():.3f}")
json.dump({"desvio_mediano": float(np.median(dp)), "maximo": float(dp.max())}, open("resultados/f11_sementes.json", "w"))
