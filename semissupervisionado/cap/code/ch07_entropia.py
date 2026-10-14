import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def sigmoide(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def logistica_entropia(X, y_parcial, lam=1.0, l2=1.0, w0=None):
    """Regressão logística binária que minimiza: perda logística nos rotulados + lam * entropia das previsões nos sem rótulo + l2 * ||w||^2 (Grandvalet e Bengio, 2004)."""
    rot = y_parcial != -1; Xa = np.c_[X, np.ones(len(X))]
    def objetivo(w):
        z = Xa @ w; p = sigmoide(z)
        perda = -np.sum(y_parcial[rot] * np.log(p[rot] + 1e-12) + (1 - y_parcial[rot]) * np.log(1 - p[rot] + 1e-12))
        ent = -np.sum(p[~rot] * np.log(p[~rot] + 1e-12) + (1 - p[~rot]) * np.log(1 - p[~rot] + 1e-12))
        g = Xa[rot].T @ (p[rot] - y_parcial[rot]) + lam * Xa[~rot].T @ (-z[~rot] * p[~rot] * (1 - p[~rot])) + 2 * l2 * np.r_[w[:-1], 0]
        return perda + lam * ent + l2 * np.sum(w[:-1] ** 2), g
    w0 = np.zeros(Xa.shape[1]) if w0 is None else w0
    return minimize(objetivo, w0, jac=True, method="L-BFGS-B").x

X, y = load_digits(return_X_y=True)
par = np.isin(y, [3, 8]); X, y = X[par], (y[par] == 8).astype(int)               # problema binário: dígito 3 contra dígito 8
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
esc = StandardScaler().fit(X_tr); X_tr, X_te = esc.transform(X_tr), esc.transform(X_te)

print("Dígitos 3 x 8, 6 rótulos (3 por classe), 30 sorteios: acurácia no teste")
print("lambda (peso da entropia) | acurácia | ganho sobre lambda=0 (p.p.) | melhor/pior")
res = {}
for lam in [0.0, 0.1, 0.3, 1.0, 3.0]:
    acc = []
    for s in range(30):
        r = np.random.RandomState(s); idx = np.r_[r.choice(np.where(y_tr == 0)[0], 3, replace=False), r.choice(np.where(y_tr == 1)[0], 3, replace=False)]
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        w = logistica_entropia(X_tr, yp, lam=lam, l2=1.0)
        acc.append(np.mean((np.c_[X_te, np.ones(len(X_te))] @ w > 0) == y_te))
    res[lam] = np.array(acc)
    d = res[lam] - res[0.0]
    print(f"{lam:25.1f} | {100 * res[lam].mean():7.1f}% | {100 * d.mean():+26.1f} | {(d > 0).sum()}/{(d < 0).sum()}")
