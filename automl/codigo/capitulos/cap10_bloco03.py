# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import json, time
import numpy as np
from flaml import AutoML
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os; os.makedirs("resultados", exist_ok=True)

X, y = load_digits(return_X_y=True)
res = {}
for b in (2, 5, 10, 20):
    acc = []
    for s in (0, 1):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
        a = AutoML()
        a.fit(Xtr, ytr, task="classification", time_budget=b, metric="accuracy", eval_method="cv",
              n_splits=3, seed=s, verbose=0)
        acc.append((a.predict(Xte) == yte).mean())
    res[b] = acc
    print(f"orçamento {b:>2d} s : acurácia no teste = {np.mean(acc):.3f} (divisões: {acc[0]:.3f}, {acc[1]:.3f})")
json.dump({str(k): v for k, v in res.items()}, open("resultados/f10_orcamento.json", "w"))
