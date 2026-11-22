import numpy as np
from avancado import *

Xtr, Xte, ytr, yte = carregar_digits()
Y, Yte = rotulos_multi(ytr), rotulos_multi(yte)
K = 20
R = {}
for s in range(5):
    rng = np.random.default_rng(s); partes = particao_iid(ytr, K, s)
    anota = [np.eye(3)[rng.permutation(3)[:rng.integers(1, 3)]].sum(0) for _ in range(K)]      # cada clínica anota 1 ou 2 rótulos
    anota[0], anota[1] = np.ones(3), np.array([0, 0, 1.0])
    casos = {"rótulos completos (teto)": (dict(agregacao="global"), [np.ones(3)] * K),
             "ausente tratado como 0": (dict(ausente_como_zero=True, agregacao="global"), anota),
             "máscara + média global": (dict(agregacao="global"), anota),
             "máscara + média por rótulo": (dict(agregacao="por_rotulo"), anota)}
    for nome, (kw, an) in casos.items():
        W = fedavg_multi(partes, an, Xtr, Y, seed=s, **kw)
        acc, f1 = metricas_multi(W, Xte, Yte); R.setdefault(nome, []).append(f1)
print(f"{'estratégia':<28}{'F1 par':>8}{'F1 ≥ 5':>8}{'F1 primo':>10}")
for nome, v in R.items():
    m = np.mean(v, axis=0); print(f"{nome:<28}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>10.3f}")

# hierarquia: o dígito (filho) deve ser coerente com a paridade (pai)
partes = particao_iid(ytr, K, 0)
Wd, _ = fedavg(partes, Xtr, ytr, Xte, yte, rodadas=40, epocas=3, seed=0)
Wm = fedavg_multi(partes, [np.ones(3)] * K, Xtr, Y, seed=0)
digito = np.argmax(Xte @ Wd, 1); par = sigmoide(Xte @ Wm[:, 0]) > 0.5
print("\nincoerência (dígito par, mas 'ímpar' previsto, ou o contrário):", round(float(np.mean((digito % 2 == 0) != par)), 3))
print("acurácia do pai, previsto direto :", round(float(np.mean(par == (yte % 2 == 0))), 3))
print("acurácia do pai, derivado do filho:", round(float(np.mean((digito % 2 == 0) == (yte % 2 == 0))), 3))
