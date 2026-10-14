import numpy as np, torch
from contrastivo import *
from ssl_profundo import treinar_ssl

X, y, X_te, y_te = carregar_mnist()
Xu = X[np.random.RandomState(100).choice(len(X), 10000, replace=False)]        # 10.000 imagens usadas SEM seus rótulos
PASSOS = 1500
enc_sim = carregar_ou_treinar("simclr_forte", lambda: pre_treinar(Xu, "simclr", passos=PASSOS, lote=128))
enc_rot = carregar_ou_treinar("rotacao", lambda: pre_treinar(Xu, "rotacao", passos=PASSOS, lote=128))
torch.manual_seed(0); enc_aleat = CNN()                                           # codificador sem nenhum treino

print(f"MNIST, 100 rótulos, pré-treino com 10.000 imagens sem rótulo ({PASSOS} passos), 3 sorteios dos rotulados")
print("abordagem                                        | acerto no teste (média ± desvio)")
res = {}
for s in range(3):
    idx = sortear_rotulados(y, 10, s); Xl, yl = X[idx], y[idx]
    res.setdefault("regressão logística sobre os pixels", []).append(LogisticRegression(max_iter=3000).fit(Xl.flatten(1).numpy(), yl.numpy()).score(X_te.flatten(1).numpy(), y_te.numpy()))
    res.setdefault("CNN do zero, só rotulados (1.500 passos)", []).append(treinar_ssl("sup", Xl, yl, Xu[:10], passos=PASSOS, semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))[2][0][1])
    res.setdefault("sonda linear, codificador sem treino", []).append(sonda_linear(enc_aleat, Xl, yl, X_te, y_te))
    res.setdefault("sonda linear, pré-treino por rotação", []).append(sonda_linear(enc_rot, Xl, yl, X_te, y_te))
    res.setdefault("sonda linear, pré-treino SimCLR", []).append(sonda_linear(enc_sim, Xl, yl, X_te, y_te))
    res.setdefault("ajuste fino, pré-treino SimCLR", []).append(ajuste_fino(enc_sim, Xl, yl, X_te, y_te, epocas=100, semente=s))
for k, v in res.items(): print(f"{k:48s} | {100 * np.mean(v):.1f}% ± {100 * np.std(v):.1f}")
