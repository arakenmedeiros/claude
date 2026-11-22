# Capítulo 2 — solução de referência (PROJ)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

X, y = load_digits(return_X_y=True)
print("Formato:", X.shape, "| exemplos por classe:", np.bincount(y))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

cv = StratifiedKFold(5, shuffle=True, random_state=0)
base = cross_val_score(DummyClassifier(strategy="most_frequent"), Xtr, ytr, cv=cv).mean()
print(f"Linha de base: {base:.3f}")

candidatos = {
    "Árvore": DecisionTreeClassifier(random_state=0),
    "k-vizinhos": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Reg. logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "Floresta": RandomForestClassifier(200, random_state=0),
}
res = {}
for nome, m in candidatos.items():
    n = cross_val_score(m, Xtr, ytr, cv=cv)
    res[nome] = n.mean()
    print(f"{nome:15s} validação = {n.mean():.3f} (±{n.std():.3f})")

vencedor = max(res, key=res.get)
modelo = candidatos[vencedor].fit(Xtr, ytr)
p = modelo.predict(Xte)
print(f"\nVencedor: {vencedor}")
print(f"Teste final: acurácia = {accuracy_score(yte, p):.3f} | F1 macro = {f1_score(yte, p, average='macro'):.3f}")

ConfusionMatrixDisplay(confusion_matrix(yte, p)).plot(cmap="Blues", colorbar=False)
plt.title(f"Matriz de confusão no teste final ({vencedor})"); plt.show()
cm = confusion_matrix(yte, p); np.fill_diagonal(cm, 0)
i, j = np.unravel_index(cm.argmax(), cm.shape)
print(f"Confusão mais frequente: dígito {i} previsto como {j} ({cm[i, j]} vezes)")
