import numpy as np
from avancado import *

Xtr, Xte, ytr, yte = carregar_digits()
K, ALVO = 20, 9
def particao_exclusiva(seed):
    rng = np.random.default_rng(seed); partes = [[] for _ in range(K)]
    for i in range(len(ytr)):                                  # só o cliente 0 possui exemplos do dígito 9
        partes[0 if ytr[i] == ALVO else rng.integers(0, K)].append(i)
    return [np.array(p) for p in partes]

def medir(W, partes):
    pred = np.argmax(Xte @ W, 1); m9 = yte == ALVO
    p0 = partes[0]
    return (np.mean(pred[m9] == ALVO), np.mean(pred[~m9] == yte[~m9]), perda(W, Xtr[p0], ytr[p0]))

linhas = {}
custo = {"original": 30 * 5 * 20, "só parar (+10 rodadas)": 10 * 5 * 19, "subtrair contribuição": 0,
         "calibração (FedEraser)": 30 * 1 * 19, "retreino exato": 30 * 5 * 19}
for s in range(3):
    partes = particao_exclusiva(s)
    W, modelos, deltas = fedavg_historico(partes, Xtr, ytr, rodadas=30, seed=s)
    linhas.setdefault("original", []).append(medir(W, partes))
    W1 = fedavg_historico(sem_cliente(partes, 0), Xtr, ytr, rodadas=10, seed=s + 100, W0=W)[0]
    linhas.setdefault("só parar (+10 rodadas)", []).append(medir(W1, partes))
    linhas.setdefault("subtrair contribuição", []).append(medir(subtrair_contribuicao(W, partes, 0, deltas), partes))
    linhas.setdefault("calibração (FedEraser)", []).append(medir(calibrar_federaser(partes, 0, Xtr, ytr, modelos, deltas, seed=s), partes))
    Wr = fedavg_historico(sem_cliente(partes, 0), Xtr, ytr, rodadas=30, seed=s)[0]
    linhas.setdefault("retreino exato", []).append(medir(Wr, partes) )
print(f"{'método':<26}{'acc dígito 9':>13}{'acc demais':>12}{'perda no cliente':>18}{'épocas-cliente':>16}")
for k, v in linhas.items():
    m = np.mean(v, axis=0)
    print(f"{k:<26}{m[0]:>13.3f}{m[1]:>12.3f}{m[2]:>18.2f}{custo[k]:>16}")
