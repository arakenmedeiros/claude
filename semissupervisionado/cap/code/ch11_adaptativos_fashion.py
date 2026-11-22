import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_fashion()
moderado = lambda x: aumentar(x, rotacao=10, escala=0.1, deslocamento=0.1)
PASSOS, SEMENTES = 2000, 2
print(f"Fashion-MNIST, 300 rótulos (30 por classe) + 10.000 sem rótulo, {PASSOS} passos, {SEMENTES} sorteios: limiares fixo e adaptativos")
print("método                                    | aluno (teste)  | peso/fração dos alvos | precisão dos alvos")
for nome, m in [("só rotulados", "sup"), ("FixMatch (limiar fixo 0,95)", "fixmatch"), ("FlexMatch", "flexmatch"), ("FreeMatch", "freematch"), ("SoftMatch", "softmatch")]:
    a, f, pr = [], [], []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 30, s); u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
        _, _, h = treinar_ssl(m, X[idx], y[idx], X[u], passos=PASSOS, forte=moderado, fraco=aumento_fraco, yu=y[u], semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
        a.append(h[0][1]); f.append(h[0][3]); pr.append(h[0][4])
    fp = lambda v: f"{100 * np.nanmean(v):19.1f}%" if not np.all(np.isnan(v)) else "                 --"
    print(f"{nome:41s} | {100 * np.mean(a):.1f}% ± {100 * np.std(a):.1f} | {fp(f)} | " + (f"{100 * np.nanmean(pr):.1f}%" if not np.all(np.isnan(pr)) else "--"))
