# Capítulo 8 — solução de referência (EX 3)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

Xw, y = load_wine(return_X_y=True)
rng = np.random.default_rng(0)
X = np.hstack([Xw, rng.normal(size=(len(y), 100))])                  # 13 reais + 100 de ruído

def novo_pipeline():
    sel = RFECV(LogisticRegression(max_iter=3000), step=5, cv=StratifiedKFold(5, shuffle=True, random_state=1))
    return make_pipeline(StandardScaler(), sel, LogisticRegression(max_iter=3000))

pipe = novo_pipeline().fit(X, y)
suporte = pipe[1].support_
print(f"RFECV escolheu {suporte.sum()} atributos, dos quais {suporte[:13].sum()} estão entre os 13 reais")
externa = cross_val_score(novo_pipeline(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
print(f"Acurácia da validação externa do pipeline completo: {externa:.3f}")
