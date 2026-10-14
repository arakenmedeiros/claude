import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def gerar(semente, rho, n=1500, d=6):
    """Duas visões. O ruído da visão 2 tem correlação rho com o ruído da visão 1 (rho=0: independentes dadas as classes)."""
    r = np.random.RandomState(semente)
    y = r.randint(0, 2, n)
    sinal = np.where(y[:, None] == 1, 1.0, -1.0) * 0.45
    e1 = r.normal(0, 1, (n, d)); e2 = rho * e1 + np.sqrt(1 - rho ** 2) * r.normal(0, 1, (n, d))
    return sinal + e1, sinal + e2, y

def co_treino(X1, X2, yp, rodadas=10, por_classe=15):
    y = yp.copy()
    for _ in range(rodadas):
        com = y != -1
        h1 = LogisticRegression().fit(X1[com], y[com]); h2 = LogisticRegression().fit(X2[com], y[com])
        sem = np.where(~com)[0]
        if len(sem) == 0: break
        novos = {}
        for h, X in ((h1, X1), (h2, X2)):
            p = h.predict_proba(X[sem]); conf = p.max(1); cl = h.classes_[p.argmax(1)]
            for c in h.classes_:
                cand = np.where(cl == c)[0]
                for i in cand[np.argsort(-conf[cand])[:por_classe]]: novos.setdefault(sem[i], c)
        for i, c in novos.items(): y[i] = c
    com = y != -1
    return LogisticRegression().fit(X1[com], y[com]), LogisticRegression().fit(X2[com], y[com])

print("correlação entre as visões | só rotulados (2 visões) | co-training | ganho (p.p.) | melhor/pior em 40")
for rho in [0.0, 0.5, 0.9, 0.99]:
    a, b = [], []
    for s in range(40):
        V1, V2, y = gerar(s, rho)
        i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.4, random_state=s)
        r = np.random.RandomState(s)
        idx = np.r_[r.choice(i_tr[y[i_tr] == 0], 5, replace=False), r.choice(i_tr[y[i_tr] == 1], 5, replace=False)]
        yp = np.full(len(y), -1); yp[idx] = y[idx]
        yp = yp[i_tr]; pos = {g: k for k, g in enumerate(i_tr)}
        X1, X2 = V1[i_tr], V2[i_tr]; rot = np.array([pos[g] for g in idx])
        j1 = LogisticRegression().fit(X1[rot], y[idx]); j2 = LogisticRegression().fit(X2[rot], y[idx])
        a.append(np.mean(j1.classes_[(j1.predict_proba(V1[i_te]) + j2.predict_proba(V2[i_te])).argmax(1)] == y[i_te]))
        h1, h2 = co_treino(X1, X2, yp)
        b.append(np.mean(h1.classes_[(h1.predict_proba(V1[i_te]) + h2.predict_proba(V2[i_te])).argmax(1)] == y[i_te]))
    d = np.array(b) - np.array(a)
    print(f"{rho:26.2f} | {100 * np.mean(a):23.1f}% | {100 * np.mean(b):10.1f}% | {100 * d.mean():+11.1f} | {(d > 0).sum()}/{(d < 0).sum()}")
