import numpy as np
from avancado import *

Xbr, Xteb, ytr, yte = carregar_digits(bruto=True)
K, T = 20, 60
tempos = np.where(np.arange(K) < 14, 1.0, 6.0)             # 14 clientes rápidos (1 unidade de tempo) e 6 lentos (6 unidades)
resultado, curvas = {}, {}
for s in range(3):
    rng = np.random.default_rng(s)
    partes = [[] for _ in range(K)]
    for i in range(len(ytr)):                              # os clientes lentos concentram os dígitos 0, 1 e 2
        c = rng.integers(14, 20) if (ytr[i] <= 2 and rng.random() < 0.85) else rng.integers(0, 20)
        partes[c].append(i)
    treino, teste = dividir_local([np.array(p) for p in partes], 0.3, s)
    Xtr, Xte = padronizar_clientes(Xbr, Xteb, treino)
    rapidos = np.concatenate(teste[:14]); lentos = np.concatenate(teste[14:])
    rodadas = {"síncrono": lambda: simular_sincrono(treino, Xtr, ytr, Xte, yte, tempos, T=T, seed=s),
               "FedAsync": lambda: simular_assincrono(treino, Xtr, ytr, Xte, yte, "fedasync", tempos, T=T, seed=s),
               "FedBuff (K=5)": lambda: simular_assincrono(treino, Xtr, ytr, Xte, yte, "fedbuff", tempos, T=T, K=5, seed=s)}
    for nome, f in rodadas.items():
        W, curva = f()
        t90 = next((t for t, a in curva if a >= 0.90), np.inf)
        resultado.setdefault(nome, []).append((t90, acuracia(W, Xte, yte), acuracia(W, Xtr[rapidos], ytr[rapidos]), acuracia(W, Xtr[lentos], ytr[lentos])))
        curvas.setdefault(nome, []).append(curva[:T, 1])

print(f"{'método':<15}{'tempo p/ 90%':>13}{'acc global':>11}{'clientes rápidos':>18}{'clientes lentos':>17}")
for nome, v in resultado.items():
    m = np.mean(v, axis=0)
    print(f"{nome:<15}{m[0]:>13.1f}{m[1]:>11.3f}{m[2]:>18.3f}{m[3]:>17.3f}")
