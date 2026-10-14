import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split

def em_semi(X, y_parcial, n_classes, iters=30, reg=1e-3, peso_u=1.0):
    """Mistura de gaussianas semissupervisionada (uma gaussiana por classe) com o algoritmo EM.
    y_parcial: -1 = sem rótulo. peso_u: peso dos exemplos sem rótulo (1 = peso total)."""
    n, d = X.shape; rot = y_parcial != -1
    R = np.zeros((n, n_classes)); R[rot, y_parcial[rot]] = 1.0                       # exemplos rotulados: responsabilidade fixa
    pi = np.ones(n_classes) / n_classes; mu = np.zeros((n_classes, d)); cov = np.array([np.eye(d)] * n_classes)
    trilha = []
    for it in range(iters + 1):
        # M-step: estima proporções, médias e covariâncias com as responsabilidades (rotulados: 1; sem rótulo: o que o E-step calculou)
        w = R.copy(); w[~rot] *= peso_u
        Nk = w.sum(axis=0)
        pi = Nk / Nk.sum(); mu = (w.T @ X) / Nk[:, None]
        for k in range(n_classes):
            Xc = X - mu[k]; cov[k] = (w[:, k, None] * Xc).T @ Xc / Nk[k] + reg * np.eye(d)
        # E-step: probabilidade de cada classe para cada exemplo sem rótulo
        logp = np.column_stack([np.log(pi[k]) + mvn.logpdf(X, mu[k], cov[k]) for k in range(n_classes)])
        mx = logp.max(axis=1, keepdims=True); post = np.exp(logp - mx); soma = post.sum(axis=1, keepdims=True); post /= soma
        trilha.append(float(np.sum(np.log(soma[~rot]) + mx[~rot]) * peso_u + np.sum(logp[rot, y_parcial[rot]])))   # log-verossimilhança
        R[~rot] = post[~rot]
    return {"pi": pi, "mu": mu, "cov": cov, "trilha": trilha}

def prever(m, X):
    logp = np.column_stack([np.log(m["pi"][k]) + mvn.logpdf(X, m["mu"][k], m["cov"][k]) for k in range(len(m["pi"]))])
    return logp.argmax(axis=1)

def gerar(semente, n=600):
    r = np.random.RandomState(semente); y = r.randint(0, 2, n)
    X = r.normal(0, 1, (n, 2)) * np.array([2.0, 0.7]) + np.where(y[:, None] == 0, [-1.5, -1.0], [1.5, 1.0])   # duas gaussianas alongadas
    return X, y

X, y = gerar(0)
r = np.random.RandomState(0); idx = np.r_[r.choice(np.where(y == 0)[0], 3, replace=False), r.choice(np.where(y == 1)[0], 3, replace=False)]
yp = np.full(len(y), -1); yp[idx] = y[idx]
m = em_semi(X, yp, 2, iters=15)
print("log-verossimilhança a cada iteração do EM (deve crescer):")
print("  " + "  ".join(f"{v:.0f}" for v in m["trilha"][:8]) + "  ...  " + f"{m['trilha'][-1]:.0f}")

print("\nGaussianas com 6 rótulos: só rotulados x EM com dados sem rótulo (30 conjuntos de dados)")
a, b = [], []
for s in range(30):
    X, y = gerar(s); X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=s)
    r = np.random.RandomState(s); idx = np.r_[r.choice(np.where(y_tr == 0)[0], 3, replace=False), r.choice(np.where(y_tr == 1)[0], 3, replace=False)]
    yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    a.append(np.mean(prever(em_semi(X_tr, yp, 2, iters=0, reg=0.05), X_te) == y_te))      # iters=0: só os rotulados
    b.append(np.mean(prever(em_semi(X_tr, yp, 2, iters=30, reg=0.05), X_te) == y_te))
d = np.array(b) - np.array(a)
print(f"  só rotulados : {100 * np.mean(a):.1f}%\n  EM semissup. : {100 * np.mean(b):.1f}%  (ganho {100 * d.mean():+.1f} p.p.; melhor/pior em 30: {(d > 0).sum()}/{(d < 0).sum()})")
