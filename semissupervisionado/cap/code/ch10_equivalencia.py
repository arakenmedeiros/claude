import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 3
print(f"MNIST, CNN só com rotulados, aumento forte, {PASSOS} passos, {SEMENTES} sorteios: a curva da referência")
print("rótulos (por dígito) | acerto no teste")
curva = {}
for por in [10, 30, 100]:
    v = [treinar_ssl("sup", X[sortear_rotulados(y, por, s)], y[sortear_rotulados(y, por, s)], X[:10], passos=PASSOS, semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))[2][0][1] for s in range(SEMENTES)]
    curva[10 * por] = np.mean(v); print(f"{10 * por:5d} ({por:3d})        | {100 * np.mean(v):.1f}% ± {100 * np.std(v):.1f}")
n = np.array(sorted(curva)); a = np.array([curva[k] for k in n])
for alvo in [0.949, 0.954, 0.960]:
    if alvo <= a.max(): print(f"acerto de {100 * alvo:.1f}%: equivale a cerca de {np.exp(np.interp(alvo, a, np.log(n))):.0f} rótulos da referência")
    else: print(f"acerto de {100 * alvo:.1f}%: acima do maior valor medido ({100 * a.max():.1f}% com {n.max()} rótulos); a equivalência exigiria mais de {n.max()} rótulos (extrapolação não feita)")
