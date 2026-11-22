# Capítulo 12 — AutoML para séries temporais, texto e imagens
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import shift, rotate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
Xtr, ytr = Xtr[:300], ytr[:300]                                          # poucos dados de treino
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, stratify=ytr, random_state=0)

def distorce(Xs, r, dmax, rot, ruido):
    saida = []
    for x in Xs:
        im = shift(x.reshape(8, 8), r.uniform(-dmax, dmax, 2), order=1, mode="constant")
        if rot > 0: im = rotate(im, r.uniform(-rot, rot), reshape=False, order=1, mode="constant")
        saida.append((im + r.normal(0, ruido, im.shape)).ravel())
    return np.array(saida)

Xte_d = distorce(Xte, np.random.default_rng(123), 1.0, 10, 0.0)            # teste deslocado/girado
Xv_d = distorce(Xv, np.random.default_rng(5), 1.0, 10, 0.0)                # validação com a mesma natureza

def treina(pol, k=4, seed=0):
    r = np.random.default_rng(seed); dmax, rot, ru = pol; Xs, ys = [Xa], [ya]
    if max(pol) > 0:
        for _ in range(k): Xs.append(distorce(Xa, r, dmax, rot, ru)); ys.append(ya)
    return SVC(C=10, gamma="scale").fit(np.vstack(Xs), np.concatenate(ys))

m0 = treina((0, 0, 0))
print(f"Sem ampliação      : validação distorcida = {m0.score(Xv_d, yv):.3f} | teste limpo = {m0.score(Xte, yte):.3f} | teste distorcido = {m0.score(Xte_d, yte):.3f}")

r = np.random.default_rng(0); melhor = None; todas = []
for _ in range(15):
    pol = (round(float(r.uniform(0, 2)), 2), round(float(r.uniform(0, 25)), 1), round(float(r.uniform(0, 2)), 2))
    m = treina(pol); v = m.score(Xv_d, yv); todas.append(v)
    if melhor is None or v > melhor[0]: melhor = (v, pol, m)
v, pol, m = melhor
print(f"Políticas testadas : validação de {min(todas):.3f} a {max(todas):.3f}")
print(f"Melhor política    : deslocamento={pol[0]}, rotação={pol[1]}°, ruído={pol[2]}")
print(f"Com a melhor política: validação distorcida = {v:.3f} | teste limpo = {m.score(Xte, yte):.3f} | teste distorcido = {m.score(Xte_d, yte):.3f}")
