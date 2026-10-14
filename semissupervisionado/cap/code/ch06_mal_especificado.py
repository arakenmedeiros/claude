import numpy as np
from sklearn.model_selection import train_test_split
exec(open("cap/code/ch06_em_do_zero.py").read().split("X, y = gerar(0)")[0])      # reaproveita em_semi e prever

def alinhado(semente, n=600):              # classe = grupo: a suposição do modelo vale
    r = np.random.RandomState(semente); y = r.randint(0, 2, n)
    return r.normal(0, 1, (n, 2)) * [2.0, 0.7] + np.where(y[:, None] == 0, [-1.5, -1.0], [1.5, 1.0]), y

def cruzado(semente, n=600):               # grupos em cima e embaixo, mas a classe é esquerda x direita: a suposição NÃO vale
    r = np.random.RandomState(semente); g = r.randint(0, 2, n); x = r.normal(0, 1.0, n)
    return np.c_[x, np.where(g == 0, -3, 3) + r.normal(0, 0.6, n)], (x > 0).astype(int)

print("cenário   | peso dos sem rótulo | acurácia | ganho sobre 'só rotulados' (p.p.)")
for nome, gerar_ in [("alinhado", alinhado), ("cruzado ", cruzado)]:
    for peso in [0.0, 0.1, 0.5, 1.0]:
        acc, base = [], []
        for s in range(30):
            X, y = gerar_(s); X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=s)
            r = np.random.RandomState(s); idx = np.r_[r.choice(np.where(y_tr == 0)[0], 5, replace=False), r.choice(np.where(y_tr == 1)[0], 5, replace=False)]
            yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
            base.append(np.mean(prever(em_semi(X_tr, yp, 2, iters=0, reg=0.05), X_te) == y_te))
            acc.append(np.mean(prever(em_semi(X_tr, yp, 2, iters=30, reg=0.05, peso_u=peso), X_te) == y_te))
        print(f"{nome} | {peso:19.1f} | {100 * np.mean(acc):7.1f}% | {100 * (np.mean(acc) - np.mean(base)):+.1f}")
