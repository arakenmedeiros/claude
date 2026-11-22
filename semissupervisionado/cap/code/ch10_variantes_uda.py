import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 3
print(f"MNIST, 100 rótulos + 10.000 sem rótulo, {PASSOS} passos, {SEMENTES} sorteios (os mesmos do experimento principal): variantes da UDA")
print("variante                                                      | aluno (teste)  | professor EMA | alvos que passam | precisão dos alvos")
variantes = [("UDA adaptada (professor EMA, máscara após o afiamento) [a do livro]", dict(mascara="apos", professor=True)),
             ("professor EMA, máscara ANTES do afiamento", dict(mascara="antes", professor=True)),
             ("UDA fiel ao artigo (sem professor, máscara antes do afiamento)", dict(mascara="antes", professor=False))]
for nome, kw in variantes:
    a, p, f, pr = [], [], [], []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 10, s); u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
        _, _, h = treinar_ssl("uda", X[idx], y[idx], X[u], passos=PASSOS, yu=y[u], semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,), **kw)
        a.append(h[0][1]); p.append(h[0][2]); f.append(h[0][3]); pr.append(h[0][4])
    print(f"{nome:62s} | {100 * np.mean(a):.1f}% ± {100 * np.std(a):.1f} | {100 * np.mean(p):.1f}% ± {100 * np.std(p):.1f} | {100 * np.mean(f):14.1f}% | {100 * np.mean(pr):.1f}%")
