# Capítulo 10 — solução de referência (EX 2)
import warnings; warnings.filterwarnings("ignore")
from flaml import AutoML
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

def roda(**kw):
    a = AutoML()
    a.fit(Xtr, ytr, task="classification", metric="accuracy", eval_method="cv", n_splits=3, seed=0, verbose=0,
          estimator_list=["lgbm", "rf", "extra_tree"], **kw)
    return len(a.config_history), a.best_estimator, round(a.best_loss, 4)

print("(a) por tempo (3 s)   :", roda(time_budget=3), "|", roda(time_budget=3))
print("(b) por tentativas (12):", roda(time_budget=60, max_iter=12), "|", roda(time_budget=60, max_iter=12))
