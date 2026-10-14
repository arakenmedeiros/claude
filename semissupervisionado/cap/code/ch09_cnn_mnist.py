import numpy as np, torch
from utilidades import *

X, y, X_te, y_te = carregar_mnist()
print("MNIST, 100 rótulos (10 por dígito), CNN treinada só com os rotulados (100 épocas); teste com 10.000 imagens")
print("aumento de dados | acerto no treino | acerto no teste (média ± desvio, 3 sorteios)")
for nome, aug in [("nenhum", None), ("fraco (deslocar até 8%)", aumento_fraco), ("forte (gira, escala, apaga, ruído)", aumento_forte)]:
    tr, te = [], []
    for s in range(3):
        idx = sortear_rotulados(y, 10, s); torch.manual_seed(s); m = CNN()
        h = treinar_supervisionado(m, X[idx], y[idx], epocas=100, aug=aug, semente=s, X_teste=X_te, y_teste=y_te, relatar=(100,))
        tr.append(h[0][1]); te.append(h[0][2])
    print(f"{nome:36s} | {100 * np.mean(tr):15.1f}% | {100 * np.mean(te):.1f}% ± {100 * np.std(te):.1f}")

print("\nCurva de aprendizado sem aumento de dados (um sorteio): o que acontece ao longo das épocas")
print("época | acerto no treino | acerto no teste")
idx = sortear_rotulados(y, 10, 0); torch.manual_seed(0); m = CNN()
for ep, a_tr, a_te in treinar_supervisionado(m, X[idx], y[idx], epocas=100, X_teste=X_te, y_teste=y_te, relatar=(1, 3, 5, 10, 20, 50, 100)):
    print(f"{ep:5d} | {100 * a_tr:15.1f}% | {100 * a_te:.1f}%")
