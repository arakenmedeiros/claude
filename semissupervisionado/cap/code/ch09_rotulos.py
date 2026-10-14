import numpy as np, torch
from utilidades import *

X, y, X_te, y_te = carregar_mnist()
print("CNN supervisionada com aumento forte: quantos rótulos fazem falta? (MNIST, teste com 10.000 imagens; 2 sorteios)")
print("rótulos (por dígito) | épocas | acerto no teste")
for por_classe, epocas in [(10, 100), (30, 60), (100, 30), (300, 15)]:
    te = []
    for s in range(2):
        idx = sortear_rotulados(y, por_classe, s); torch.manual_seed(s); m = CNN()
        h = treinar_supervisionado(m, X[idx], y[idx], epocas=epocas, aug=aumento_forte, semente=s, X_teste=X_te, y_teste=y_te, relatar=(epocas,))
        te.append(h[0][2])
    print(f"{10 * por_classe:5d} ({por_classe:3d})         | {epocas:6d} | {100 * np.mean(te):.1f}%")
