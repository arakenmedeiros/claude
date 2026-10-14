import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def tsvm(X, y_parcial, C=10.0, gamma=1.0, Cu_final=1.0, passos=5, max_trocas=40, prop_pos=None, balancear=True):
    """TSVM simplificado, inspirado em Joachims (1999). Rótulos 0/1; -1 = sem rótulo.
    1) treina uma SVM só com os rotulados e dá rótulos provisórios aos demais, respeitando a proporção de positivos;
    2) aumenta aos poucos o peso Cu dos exemplos sem rótulo e, a cada passo, troca o rótulo de pares que violam a margem."""
    rot = np.where(y_parcial != -1)[0]; sem = np.where(y_parcial == -1)[0]
    Xl, yl = X[rot], 2 * y_parcial[rot] - 1; Xu = X[sem]
    if prop_pos is None: prop_pos = np.mean(yl == 1)
    m = SVC(C=C, kernel="rbf", gamma=gamma).fit(Xl, yl)
    f = m.decision_function(Xu)
    if balancear:
        yu = -np.ones(len(sem)); yu[np.argsort(-f)[:int(round(prop_pos * len(sem)))]] = 1      # proporção fixa de positivos
    else:
        yu = np.where(f > 0, 1.0, -1.0)
    Xa = np.vstack([Xl, Xu])
    for Cu in np.geomspace(Cu_final / 100, Cu_final, passos):                                   # peso crescente dos sem rótulo
        for _ in range(max_trocas):
            sw = np.r_[np.ones(len(rot)), np.full(len(sem), Cu)]
            m = SVC(C=C, kernel="rbf", gamma=gamma).fit(Xa, np.r_[yl, yu], sample_weight=sw)
            folga = np.maximum(0, 1 - yu * m.decision_function(Xu))                              # quanto cada exemplo viola a margem
            pos = np.where((yu == 1) & (folga > 0))[0]; neg = np.where((yu == -1) & (folga > 0))[0]
            if len(pos) == 0 or len(neg) == 0: break
            i, j = pos[np.argmax(folga[pos])], neg[np.argmax(folga[neg])]
            if folga[i] + folga[j] <= 2: break                                                   # trocar não compensa
            yu[i], yu[j] = -1, 1                                                                 # troca os rótulos do par
    return m

print("duas luas, 6 rótulos, 30 sorteios (C = 10, Cu = 1): SVM só com rotulados x TSVM, conforme gamma")
print("gamma | SVM só rotulados | TSVM | ganho (p.p.) | melhor/pior")
for gamma in [0.5, 1, 2, 5]:
    a, b = [], []
    for s in range(30):
        X, y = make_moons(300, noise=0.1, random_state=s); X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=s)
        r = np.random.RandomState(s); idx = np.r_[r.choice(np.where(y_tr == 0)[0], 3, replace=False), r.choice(np.where(y_tr == 1)[0], 3, replace=False)]
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        a.append(SVC(C=10, gamma=gamma).fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
        b.append(np.mean(tsvm(X_tr, yp, C=10, gamma=gamma, Cu_final=1.0).predict(X_te) == 2 * y_te - 1))
    d = np.array(b) - np.array(a)
    print(f"{gamma:5.1f} | {100 * np.mean(a):15.1f}% | {100 * np.mean(b):4.1f}% | {100 * d.mean():+11.1f} | {(d > 0).sum()}/{(d < 0).sum()}")
