# Capítulo 2 — solução de referência (EX 2)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
mal = rng.choice(np.where(y == 0)[0], 15, replace=False)    # 15 malignos (classe rara)
ben = np.where(y == 1)[0]                                     # todos os benignos
idx = np.concatenate([mal, ben]); X, y = X[idx], y[idx]
y = 1 - y                                                     # agora 1 = maligno (positivo, raro)
print("Proporção de casos raros:", round(y.mean(), 3))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
modelos = {
    "Sempre benigno": DummyClassifier(strategy="most_frequent"),
    "Logística comum": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Logística balanceada": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
}
for nome, m in modelos.items():
    m.fit(Xtr, ytr); p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
    print(f"{nome:21s} acur={accuracy_score(yte,p):.3f} prec={precision_score(yte,p,zero_division=0):.3f} "
          f"revoc={recall_score(yte,p):.3f} AUC={roc_auc_score(yte,pr):.3f}")
