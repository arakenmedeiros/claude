import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 3
print(f"MNIST, 100 rótulos + 10.000 sem rótulo, {PASSOS} passos, {SEMENTES} sorteios: limiares fixo e adaptativos (versões simplificadas)")
print("método                                    | aluno (teste)  | peso/fração dos alvos | precisão dos alvos")
for nome, m in [("FixMatch (limiar fixo 0,95)", "fixmatch"), ("FlexMatch (limiar por classe, crescente)", "flexmatch"), ("FreeMatch (limiar global e por classe, por EMA)", "freematch"), ("SoftMatch (peso suave, sem limiar)", "softmatch")]:
    a, f, pr = [], [], []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 10, s); u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
        _, _, h = treinar_ssl(m, X[idx], y[idx], X[u], passos=PASSOS, yu=y[u], semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
        a.append(h[0][1]); f.append(h[0][3]); pr.append(h[0][4])
    print(f"{nome:41s} | {100 * np.mean(a):.1f}% ± {100 * np.std(a):.1f} | {100 * np.mean(f):19.1f}% | {100 * np.mean(pr):.1f}%")
