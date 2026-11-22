import numpy as np
from avancado import *

Xbr, Xteb, ytr, yte = carregar_digits(bruto=True)
K = 30
R = {}
for s in range(5):
    partes = particao_dirichlet(ytr, K, 0.1, s)
    treino, teste = dividir_local(partes, 0.3, s)
    ok = [i for i in range(K) if len(teste[i]) >= 5 and len(treino[i]) > 0]
    treino, teste = [treino[i] for i in ok], [teste[i] for i in ok]
    Xtr, Xte = padronizar_clientes(Xbr, Xteb, treino)
    disp = np.where(classe_dominante(treino, ytr) < 5, 0.95, 0.05)     # quem tem os dígitos 0 a 4 quase sempre está disponível
    for pol in ("uniforme", "disponivel", "ponderado", "poder"):
        W = fedavg_selecao(treino, Xtr, ytr, Xte, yte, pol, disp, seed=s)
        por_cli = acuracias_locais([W] * len(treino), teste, Xtr, ytr)
        por_classe = [np.mean(np.argmax(Xte @ W, 1)[yte == c] == c) for c in range(10)]
        R.setdefault(pol, []).append((acuracia(W, Xte, yte), por_cli.mean(), np.percentile(por_cli, 10), min(por_classe)))

print(f"{'política':<12}{'acc global':>11}{'média por cliente':>19}{'percentil 10':>14}{'pior classe':>13}")
for pol, v in R.items():
    m = np.mean(v, axis=0)
    print(f"{pol:<12}{m[0]:>11.3f}{m[1]:>19.3f}{m[2]:>14.3f}{m[3]:>13.3f}")
