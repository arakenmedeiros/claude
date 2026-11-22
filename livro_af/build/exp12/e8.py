import numpy as np
from avancado import *
from personalizacao import ifca

Xtr, Xte, ytr, yte = carregar_digits()
K, R0 = 20, 25
H = {"continuar": [], "reiniciar": []}; disparos = []
for s in range(3):
    perm = np.random.default_rng(s).permutation(10)               # a "nova linguagem": os dígitos trocam de nome
    partes = particao_dirichlet(ytr, K, 1.0, s)
    for est in H:
        _, h, d = fedavg_com_deriva(partes, Xtr, ytr, Xte, yte, perm, R0, None, est, seed=s)
        H[est].append(h)
        if est == "reiniciar": disparos.append(d)
print("rodada do disparo do detector:", disparos)
print(f"{'rodada':>8}{'continuar':>11}{'reiniciar':>11}")
for r in (R0 - 1, R0, R0 + 2, R0 + 5, R0 + 10, R0 + 20):
    print(f"{r:>8}{np.mean([h[r] for h in H['continuar']]):>11.3f}{np.mean([h[r] for h in H['reiniciar']]):>11.3f}")

# deriva parcial: só metade dos clientes muda de conceito
res = {"continuar (um modelo)": [], "IFCA com K = 2": []}
for s in range(3):
    perm = np.random.default_rng(s).permutation(10)
    partes = particao_dirichlet(ytr, K, 1.0, s); metade = list(range(0, K, 2))
    W, _, _ = fedavg_com_deriva(partes, Xtr, ytr, Xte, yte, perm, R0, metade, "continuar", seed=s)
    res["continuar (um modelo)"].append((acuracia(W, Xte, yte), acuracia(W, Xte, perm[yte])))
    y_novo = ytr.copy()
    for c in metade: y_novo[partes[c]] = perm[ytr[partes[c]]]
    M, esc = ifca(partes, Xtr, y_novo, K=2, seed=s)
    mA = np.bincount([esc[c] for c in range(K) if c not in metade], minlength=2).argmax()
    mB = np.bincount([esc[c] for c in metade], minlength=2).argmax()
    res["IFCA com K = 2"].append((acuracia(M[mA], Xte, yte), acuracia(M[mB], Xte, perm[yte])))
print(f"\n{'estratégia':<24}{'conceito antigo':>16}{'conceito novo':>15}")
for nome, v in res.items():
    m = np.mean(v, axis=0); print(f"{nome:<24}{m[0]:>16.3f}{m[1]:>15.3f}")
