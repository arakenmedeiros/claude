# Capítulo 9 — Meta-aprendizado, warm start e ensembles
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import skew, entropy
from sklearn.datasets import make_classification, load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss

NOMES = ["log(n)", "log(d)", "n/d", "classes", "entropia", "assimetria", "correlação", "landmarker"]

def meta_atributos(X, y):
    n, d = X.shape; c = len(np.unique(y)); p = np.bincount(y) / n
    assim = np.abs(skew(X, axis=0))                                      # assimetria de cada atributo
    cor = np.abs(np.corrcoef(X.T)); np.fill_diagonal(cor, 0)             # correlação entre atributos
    landmarker = cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=0), X, y, cv=3).mean()
    return [np.log(n), np.log(d), n / d, c, entropy(p) / np.log(c), np.nanmean(assim), np.nanmean(cor), landmarker]

print(f"{'conjunto':8s} | " + " | ".join(f"{n:>11s}" for n in NOMES))
for nome, carregar in [("Iris", load_iris), ("Wine", load_wine), ("Cancer", load_breast_cancer), ("Digits", load_digits)]:
    X, y = carregar(return_X_y=True)
    print(f"{nome:8s} | " + " | ".join(f"{v:11.2f}" for v in meta_atributos(X, y)))
