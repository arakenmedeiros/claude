import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

def co_treino(X1, X2, y_parcial, rodadas=10, por_classe=10, C=1.0):
    """Co-training com duas visões. Rótulos desconhecidos = -1. Devolve os dois classificadores."""
    y = y_parcial.copy()
    for _ in range(rodadas):
        com = y != -1
        h1 = LogisticRegression(max_iter=2000, C=C).fit(X1[com], y[com])      # classificador da visão 1
        h2 = LogisticRegression(max_iter=2000, C=C).fit(X2[com], y[com])      # classificador da visão 2
        sem = np.where(~com)[0]
        if len(sem) == 0: break
        novos = {}
        for h, X in ((h1, X1), (h2, X2)):                                    # cada um escolhe os seus melhores palpites
            prob = h.predict_proba(X[sem]); conf = prob.max(axis=1); classe = h.classes_[prob.argmax(axis=1)]
            for c in h.classes_:
                cand = np.where(classe == c)[0]
                for i in cand[np.argsort(-conf[cand])[:por_classe]]:
                    novos.setdefault(sem[i], c)                               # o primeiro a escolher fica com o exemplo
        for i, c in novos.items(): y[i] = c
    com = y != -1
    return LogisticRegression(max_iter=2000, C=C).fit(X1[com], y[com]), LogisticRegression(max_iter=2000, C=C).fit(X2[com], y[com])

def acuracia(h1, h2, X1, X2, y):                                              # decide somando as probabilidades das duas visões
    return np.mean(h1.classes_[(h1.predict_proba(X1) + h2.predict_proba(X2)).argmax(axis=1)] == y)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
topo = np.arange(64).reshape(8, 8)[:4].ravel(); base = np.arange(64).reshape(8, 8)[4:].ravel()   # visão 1 = 4 linhas de cima; visão 2 = 4 de baixo
perm = np.random.RandomState(0).permutation(64); va, vb = perm[:32], perm[32:]      # visões formadas por pixels sorteados
res = {k: [] for k in ["visão 1 (cima)", "visão 2 (baixo)", "duas visões, um só modelo", "duas visões, só rotulados", "self-training (um só modelo)", "co-training (cima/baixo)", "visões aleatórias, só rotulados", "co-training (visões aleatórias)"]}
for s in range(10):
    r = np.random.RandomState(s)
    idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
    yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    res["visão 1 (cima)"].append(LogisticRegression(max_iter=2000).fit(X_tr[idx][:, topo], y_tr[idx]).score(X_te[:, topo], y_te))
    res["visão 2 (baixo)"].append(LogisticRegression(max_iter=2000).fit(X_tr[idx][:, base], y_tr[idx]).score(X_te[:, base], y_te))
    res["duas visões, um só modelo"].append(LogisticRegression(max_iter=2000).fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
    g1 = LogisticRegression(max_iter=2000).fit(X_tr[idx][:, topo], y_tr[idx]); g2 = LogisticRegression(max_iter=2000).fit(X_tr[idx][:, base], y_tr[idx])
    res["duas visões, só rotulados"].append(acuracia(g1, g2, X_te[:, topo], X_te[:, base], y_te))
    res["self-training (um só modelo)"].append(SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=0.9).fit(X_tr, yp).score(X_te, y_te))
    h1, h2 = co_treino(X_tr[:, topo], X_tr[:, base], yp)
    res["co-training (cima/baixo)"].append(acuracia(h1, h2, X_te[:, topo], X_te[:, base], y_te))
    g1 = LogisticRegression(max_iter=2000).fit(X_tr[idx][:, va], y_tr[idx]); g2 = LogisticRegression(max_iter=2000).fit(X_tr[idx][:, vb], y_tr[idx])
    res["visões aleatórias, só rotulados"].append(acuracia(g1, g2, X_te[:, va], X_te[:, vb], y_te))
    h1, h2 = co_treino(X_tr[:, va], X_tr[:, vb], yp)
    res["co-training (visões aleatórias)"].append(acuracia(h1, h2, X_te[:, va], X_te[:, vb], y_te))
print("método                          | acurácia no teste (média ± desvio, 10 sorteios, 50 rótulos)")
for k, v in res.items(): print(f"{k:31s} | {100 * np.mean(v):5.1f}% ± {100 * np.std(v):.1f}")
