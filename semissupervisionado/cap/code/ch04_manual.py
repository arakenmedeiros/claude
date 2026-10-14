import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

def auto_treino(modelo, X, y_parcial, limiar=0.9, max_iter=10, y_oculto=None, X_teste=None, y_teste=None):
    """Self-training do zero. Rótulos desconhecidos = -1. y_oculto e teste servem só para registrar o que acontece."""
    y = y_parcial.copy()
    for it in range(1, max_iter + 1):
        com = y != -1
        modelo.fit(X[com], y[com])                       # 1) treina com o que tem rótulo
        sem = np.where(~com)[0]
        if len(sem) == 0: break
        prob = modelo.predict_proba(X[sem])              # 2) prevê os exemplos sem rótulo
        conf = prob.max(axis=1)
        aceitos = sem[conf >= limiar]                    # 3) aceita só os confiantes
        if len(aceitos) == 0: break
        y[aceitos] = modelo.classes_[prob[conf >= limiar].argmax(axis=1)]   # 4) pseudo-rótulos entram no treino
        msg = f"rodada {it}: +{len(aceitos):4d} pseudo-rótulos"
        if y_oculto is not None: msg += f" | acerto deles: {100 * np.mean(y[aceitos] == y_oculto[aceitos]):5.1f}%"
        if X_teste is not None: msg += f" | teste: {100 * modelo.score(X_teste, y_teste):5.1f}%"
        print(msg)
    modelo.fit(X[y != -1], y[y != -1])
    return modelo

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
rng = np.random.RandomState(0)
idx = np.concatenate([rng.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
y_parcial = np.full(len(y_tr), -1); y_parcial[idx] = y_tr[idx]

base = LogisticRegression(max_iter=2000).fit(X_tr[idx], y_tr[idx])
print("Referência (só 50 rótulos): %.1f%%" % (100 * base.score(X_te, y_te)))
meu = auto_treino(LogisticRegression(max_iter=2000), X_tr, y_parcial, y_oculto=y_tr, X_teste=X_te, y_teste=y_te)
print("Meu self-training          : %.1f%%" % (100 * meu.score(X_te, y_te)))
sk = SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=0.9).fit(X_tr, y_parcial)
print("SelfTrainingClassifier     : %.1f%%" % (100 * sk.score(X_te, y_te)))
