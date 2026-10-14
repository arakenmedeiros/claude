import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 3
print(f"MNIST, 100 rótulos + 10.000 imagens sem rótulo, CNN, {PASSOS} passos, {SEMENTES} sorteios dos rotulados")
print("método                           | aluno (teste)  | alvos que passam | precisão dos alvos")
res = {}
for nome, metodo in [("só rotulados (referência)", "sup"), ("Pseudo-Label (sem limiar)", "pseudo"), ("FixMatch (limiar 0,95)", "fixmatch"), ("MixMatch (versão simplificada)", "mixmatch")]:
    a, f, pr = [], [], []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 10, s)
        u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
        _, _, h = treinar_ssl(metodo, X[idx], y[idx], X[u], passos=PASSOS, peso=10.0 if metodo == "mixmatch" else 1.0, yu=y[u], semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
        a.append(h[0][1]); f.append(h[0][3]); pr.append(h[0][4])
    res[nome] = a
    fp = lambda v: f"{100 * np.mean(v):14.1f}%" if not np.isnan(np.mean(v)) else "              --"
    print(f"{nome:32s} | {100 * np.mean(a):.1f}% ± {100 * np.std(a):.1f} | {fp(f)} | {fp(pr)}")
base = np.array(res["só rotulados (referência)"])
print("\nganho do aluno sobre a referência, por sorteio (pontos percentuais):")
for nome, a in list(res.items())[1:]:
    print(f"{nome:32s} | " + "  ".join(f"{100 * (x - b):+.1f}" for x, b in zip(a, base)))
