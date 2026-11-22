# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# Dados simulados: apenas 5% de casos positivos (doença rara)
X, y = make_classification(n_samples=4000, n_features=10, n_informative=4, weights=[0.95, 0.05], class_sep=1.5, flip_y=0.0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

bobo = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
logis = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)

for nome, m in [("Sempre 'saudável'", bobo), ("Regressão logística", logis)]:
    p = m.predict(Xte)
    prob = m.predict_proba(Xte)[:, 1]
    print(f"{nome:20s} acurácia={accuracy_score(yte, p):.3f}  revocação={recall_score(yte, p):.3f}  F1={f1_score(yte, p):.3f}  AUC={roc_auc_score(yte, prob):.3f}")
