import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 3
print(f"MNIST, 100 rótulos + 10.000 imagens sem rótulo, CNN, {PASSOS} passos de treino, {SEMENTES} sorteios dos rotulados")
print("método                          | aluno (teste)   | professor EMA (teste)")
res = {}
for nome, metodo in [("só rotulados (referência)", "sup"), ("Π-model", "pi"), ("Mean Teacher", "mt"), ("UDA (afiada + confiança)", "uda")]:
    a, p = [], []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 10, s)
        u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
        _, _, h = treinar_ssl(metodo, X[idx], y[idx], X[u], passos=PASSOS, semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
        a.append(h[0][1]); p.append(h[0][2])
    res[nome] = (a, p)
    print(f"{nome:31s} | {100 * np.mean(a):.1f}% ± {100 * np.std(a):.1f}  | {100 * np.mean(p):.1f}% ± {100 * np.std(p):.1f}")
base = np.array(res["só rotulados (referência)"][0])
print("\nganho do aluno sobre a referência, por sorteio (pontos percentuais):")
for nome, (a, p) in list(res.items())[1:]:
    print(f"{nome:31s} | " + "  ".join(f"{100 * (x - b):+.1f}" for x, b in zip(a, base)))
