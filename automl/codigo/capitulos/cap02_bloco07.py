# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 7. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(6000, 20, n_informative=6, n_redundant=2, weights=[0.95, 0.05], flip_y=0.01, class_sep=0.8, random_state=0)
Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=1)
print("Positivos: %.1f%% | no treino: %d | na validação: %d | no teste: %d" % (100 * y.mean(), ytr.sum(), yva.sum(), yte.sum()))

modelo = lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))
def relatorio(nome, p, pred):
    print(f"{nome:<34s} {(pred == yte).mean():6.3f} {precision_score(yte, pred, zero_division=0):6.3f} {recall_score(yte, pred):6.3f} "
          f"{f1_score(yte, pred):6.3f} {roc_auc_score(yte, p):6.3f} {average_precision_score(yte, p):6.3f}")

print(f"\n{'método (avaliado no teste)':<34s} {'acc':>6s} {'prec':>6s} {'rev':>6s} {'F1':>6s} {'AUC':>6s} {'AP':>6s}")
m0 = modelo().fit(Xtr, ytr); p0 = m0.predict_proba(Xte)[:, 1]
relatorio("sem tratamento (limiar 0,5)", p0, (p0 >= 0.5).astype(int))
mw = modelo(class_weight="balanced").fit(Xtr, ytr); pw = mw.predict_proba(Xte)[:, 1]
relatorio("pesos de classe (limiar 0,5)", pw, (pw >= 0.5).astype(int))

def oversample(X, y, r):                                                  # repete positivos até igualar o número de negativos
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    i = np.r_[neg, pos, r.choice(pos, len(neg) - len(pos), replace=True)]
    return X[i], y[i]
r = np.random.default_rng(0)
mo = modelo().fit(*oversample(Xtr, ytr, r)); po = mo.predict_proba(Xte)[:, 1]
relatorio("oversampling no treino (limiar 0,5)", po, (po >= 0.5).astype(int))

cand = np.linspace(0.02, 0.9, 89); pv = m0.predict_proba(Xva)[:, 1]            # limiar escolhido NA VALIDAÇÃO (maximiza o F1)
t = cand[int(np.argmax([f1_score(yva, (pv >= c).astype(int)) for c in cand]))]
relatorio(f"sem tratamento + limiar {t:.2f} (valid.)", p0, (p0 >= t).astype(int))

def f1_cv(Xs, ys, reamostra_antes):
    notas = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys):
        Xt_, yt_ = Xs[tr], ys[tr]
        if not reamostra_antes: Xt_, yt_ = oversample(Xt_, yt_, r)        # certo: só no treino da dobra
        notas.append(f1_score(ys[va], modelo().fit(Xt_, yt_).predict(Xs[va])))
    return np.mean(notas)
print("\nF1 em validação cruzada (5 partes) no treino:")
print("  oversampling ANTES de dividir (errado)  : %.3f" % f1_cv(*oversample(Xtr, ytr, r), True))
print("  oversampling DENTRO da dobra (certo)    : %.3f" % f1_cv(Xtr, ytr, False))
