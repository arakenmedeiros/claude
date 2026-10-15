import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_fashion()
moderado = lambda x: aumentar(x, rotacao=10, escala=0.1, deslocamento=0.1)
idx = sortear_rotulados(y, 100, 0); u = np.random.RandomState(100).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
print("Fashion-MNIST, 1.000 rótulos (100 por classe), 10.000 sem rótulo, 2.000 passos, 1 sorteio: confiança e abstenção")
curvas = {}
print("método       | acerto geral | confiança média | ECE   | acerto se responder 80% | 60% | 40% | maior fração com >= 90% de acerto")
for nome, m in [("só rotulados", "sup"), ("FixMatch", "fixmatch")]:
    aluno, _, _ = treinar_ssl(m, X[idx], y[idx], X[u], passos=2000, forte=moderado, fraco=aumento_fraco, semente=0, X_teste=X_te, y_teste=y_te, relatar=())
    aluno.eval()
    with torch.no_grad(): P = torch.cat([F.softmax(aluno(X_te[i:i + 2000]), 1) for i in range(0, len(X_te), 2000)])
    conf, pred = P.max(1); ok = (pred == y_te).float(); ece = 0.0; f = np.linspace(0, 1, 11)
    for a, b in zip(f[:-1], f[1:]):
        k = (conf > a) & (conf <= b)
        if k.any(): ece += float(k.float().mean() * abs(ok[k].mean() - conf[k].mean()))
    c = np.cumsum(ok[torch.argsort(-conf)].numpy()) / np.arange(1, len(ok) + 1); n = len(c); curvas[nome] = c[::100]; idx90 = np.where(c >= 0.9)[0]
    print(f"{nome:12s} | {100 * ok.mean():10.1f}% | {100 * conf.mean():13.1f}% | {100 * ece:4.1f}% | {100 * c[int(0.8 * n) - 1]:22.1f}% | {100 * c[int(0.6 * n) - 1]:.1f}% | {100 * c[int(0.4 * n) - 1]:.1f}% | {100 * ((idx90.max() + 1) / n if len(idx90) else 0):.1f}%")
np.savez("dados/abstencao_curvas.npz", **{k.replace(" ", "_"): v for k, v in curvas.items()})   # só para desenhar a figura
