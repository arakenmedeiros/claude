# Capítulo 10 — solução de referência (EX 1)
import warnings; warnings.filterwarnings("ignore")
from flaml import AutoML
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
a = AutoML()
a.fit(Xtr, ytr, task="classification", estimator_list=["lgbm", "rf"], time_budget=5, metric="accuracy",
      eval_method="cv", n_splits=3, seed=0, verbose=0)
print("Vencedor            :", a.best_estimator)
print("Erro de validação   : %.4f (acurácia %.4f)" % (a.best_loss, 1 - a.best_loss))
print("Acurácia no teste   : %.4f" % (a.predict(Xte) == yte).mean())
