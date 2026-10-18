"""Capítulo 5: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 5.1: Curvas de perda com pouco e com muito dado]
import numpy as np
from llmdidatico import corpus, modelos

frases = corpus.frases_lm(12000, seed=0)
val = frases[-1000:]
vocab = sorted({t for f in frases for t in f} | {"<s>"})

def perda(m, dados):
    X, y = m.dados(dados); _, _, z = m.frente(X)
    return float(-modelos.log_softmax(z)[np.arange(len(y)), y].mean())

for n in (200, 3000):
    m = modelos.MLPLM(vocab, ctx=3, d=16, h=64, seed=0)
    curva = []
    for ep in range(20):
        m.treinar(frases[:n], epocas=1, lote=64, lr=0.01, seed=ep)
        curva.append((perda(m, frases[:n]), perda(m, val)))
    print(f"\ntreino com {n} frases")
    print(f"{'época':>6} {'treino':>8} {'validação':>10}")
    for ep in (1, 5, 10, 15, 20):
        print(f"{ep:>6} {curva[ep - 1][0]:>8.3f} {curva[ep - 1][1]:>10.3f}")
    print(f"lacuna final (validação - treino): {curva[-1][1] - curva[-1][0]:.3f}")


# %% [Exercício 5.2: Capacidade do modelo: tamanho da camada oculta]
import time
import numpy as np
from llmdidatico import corpus, modelos

frases = corpus.frases_lm(12000, seed=0)
val = frases[-1000:]; treino = frases[:800]
vocab = sorted({t for f in frases for t in f} | {"<s>"})

def perda(m, dados):
    X, y = m.dados(dados); _, _, z = m.frente(X)
    return float(-modelos.log_softmax(z)[np.arange(len(y)), y].mean())

print(f"{'h':>5} {'parâmetros':>11} {'tempo (s)':>10} {'treino':>8} {'validação':>10}")
for h in (4, 16, 64, 256):
    m = modelos.MLPLM(vocab, ctx=3, d=16, h=h, seed=0)
    t0 = time.perf_counter()
    m.treinar(treino, epocas=25, lote=64, lr=0.01)
    dt = time.perf_counter() - t0
    print(f"{h:>5} {sum(p.size for p in m.P.values()):>11} {dt:>10.2f} {perda(m, treino):>8.3f} {perda(m, val):>10.3f}")


# %% [Exercício 5.3: Curvas de aprendizagem com scikit-learn]
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True)
X = X / 16.0
fracs = np.linspace(0.1, 1.0, 5)
for nome, modelo in [("Árvore de decisão", DecisionTreeClassifier(random_state=0)), ("Regressão logística", LogisticRegression(max_iter=2000))]:
    tam, tr, va = learning_curve(modelo, X, y, train_sizes=fracs, cv=5, random_state=0, shuffle=True)
    print(f"\n{nome}")
    print(f"{'exemplos':>9} {'treino':>8} {'validação':>10} {'lacuna':>8}")
    for t, a, b in zip(tam, tr.mean(1), va.mean(1)):
        print(f"{t:>9} {a:>8.3f} {b:>10.3f} {a - b:>8.3f}")


# %% [Projeto: Auditoria de dados: deduplicação e contaminação]
import re
from llmdidatico import DOCS, modelos

D = {d[0]: d[3] for d in DOCS}
treino = [(i, D[i]) for i in [f"D{k:02d}" for k in range(1, 41)]]
for i in ("D05", "D12", "D27"):
    treino.append((i + "_copia", D[i]))
alterados = {"D10": ("90 dias", "60 dias"), "D18": ("R$ 400,00", "R$ 450,00"), "D33": ("30 dias", "45 dias")}
for i, (a, b) in alterados.items():
    assert a in D[i], (i, a)
    treino.append((i + "_alterado", D[i].replace(a, b)))
print("documentos no corpus de treino bruto:", len(treino))

def shingles(t, k=3):
    p = t.lower().split()
    return {tuple(p[i:i + k]) for i in range(len(p) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

S = [shingles(t) for _, t in treino]
manter, removidos = [], []
for i, s in enumerate(S):
    dup = [treino[j][0] for j in manter if jaccard(s, S[j]) >= 0.6]
    if dup:
        removidos.append((treino[i][0], dup[0]))
    else:
        manter.append(i)
print("mantidos:", len(manter), "| removidos:", len(removidos))
for r, de in removidos:
    print(f"  removido {r:<14} (duplicata de {de})")

treino_dedup = [treino[i] for i in manter]
ng_treino = [{tuple(w[i:i + 4]) for i in range(len(w) - 3)} for w in (modelos.tokenizar(t) for _, t in treino_dedup)]
teste = [(f"D{k:02d}", D[f"D{k:02d}"]) for k in range(41, 51)] + [("D12", D["D12"]), ("D18", D["D18"]), ("D33", D["D33"])]
print(f"\n{'item de teste':<14} {'sobreposição máx. de 4-gramas':>30}")
suspeitos = 0
for nome, t in teste:
    w = modelos.tokenizar(t); ng = {tuple(w[i:i + 4]) for i in range(len(w) - 3)}
    sobre = max(len(ng & g) / len(ng) for g in ng_treino)
    suspeitos += sobre >= 0.5
    print(f"{nome:<14} {sobre:>30.2f}{'   <- suspeito' if sobre >= 0.5 else ''}")
print(f"\ntaxa de itens suspeitos: {suspeitos}/{len(teste)} = {suspeitos / len(teste):.0%}")

