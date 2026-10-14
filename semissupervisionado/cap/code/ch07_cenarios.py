import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
exec(open("cap/code/ch07_tsvm.py").read().split("print(\"duas luas")[0])           # reaproveita a função tsvm

def duas_luas(s, n=300): return make_moons(n, noise=0.1, random_state=s)
def grupos_cruzados(s, n=300):
    r = np.random.RandomState(s); g = r.randint(0, 2, n); x = r.normal(0, 1.0, n)
    return np.c_[x, np.where(g == 0, -3, 3) + r.normal(0, 0.6, n)], (x > 0).astype(int)

print("cenário | TSVM sem balancear | só rotulados | TSVM balanceado (ganho sobre só rotulados, p.p.; melhor/pior em 30)")
for nome, gerar, nl, gamma in [("duas luas       ", duas_luas, 3, 5.0), ("grupos cruzados ", grupos_cruzados, 5, 0.5)]:
    a, b, c = [], [], []
    for s in range(30):
        X, y = gerar(s); X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=s)
        r = np.random.RandomState(s); idx = np.r_[r.choice(np.where(y_tr == 0)[0], nl, replace=False), r.choice(np.where(y_tr == 1)[0], nl, replace=False)]
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        a.append(SVC(C=10, gamma=gamma).fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
        b.append(np.mean(tsvm(X_tr, yp, gamma=gamma, balancear=False).predict(X_te) == 2 * y_te - 1))
        c.append(np.mean(tsvm(X_tr, yp, gamma=gamma, balancear=True).predict(X_te) == 2 * y_te - 1))
    a, b, c = map(np.array, (a, b, c))
    print(f"{nome}| {100 * b.mean():17.1f}% | {100 * a.mean():11.1f}% | {100 * c.mean():.1f}% ({100 * (c - a).mean():+.1f}; {(c > a).sum()}/{(c < a).sum()})")
