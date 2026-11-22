import numpy as np, torch
from contrastivo import *
from ssl_profundo import treinar_ssl

Xf, yf, Xft, yft = carregar_fashion(); X, y, X_te, y_te = carregar_mnist()
def treinar_fonte():
    idx = sortear_rotulados(yf, 1000, 0); torch.manual_seed(0); m = CNN()
    treinar_supervisionado(m, Xf[idx], yf[idx], epocas=8, aug=aumento_fraco, semente=0, X_teste=Xft, y_teste=yft, relatar=())
    return m
fonte = carregar_ou_treinar("fonte_fashion", treinar_fonte)
print(f"Tarefa-fonte: CNN treinada com 10.000 imagens ROTULADAS do Fashion-MNIST (acerto no teste do Fashion-MNIST: {100 * acuracia(fonte.eval(), Xft, yft):.1f}%)")
print("Tarefa-alvo: MNIST, 100 rótulos, 3 sorteios (acerto no teste)")
res = {}; torch.manual_seed(0); aleat = CNN()
for s in range(3):
    idx = sortear_rotulados(y, 10, s)
    res.setdefault("CNN do zero, 1.500 passos (referência forte)", []).append(treinar_ssl("sup", X[idx], y[idx], X[:10], passos=1500, semente=s, X_teste=X_te, y_teste=y_te, relatar=(1500,))[2][0][1])
    res.setdefault("extrator congelado da tarefa-fonte + sonda linear", []).append(sonda_linear(fonte, X[idx], y[idx], X_te, y_te))
    res.setdefault("extrator sem treino + sonda linear", []).append(sonda_linear(aleat, X[idx], y[idx], X_te, y_te))
    res.setdefault("ajuste fino a partir da tarefa-fonte, 1.500 passos", []).append(ajuste_fino(fonte, X[idx], y[idx], X_te, y_te, epocas=375, semente=s))
for k, v in res.items(): print(f"{k:55s} | {100 * np.mean(v):.1f}% ± {100 * np.std(v):.1f}")
