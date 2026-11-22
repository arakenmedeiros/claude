# Capítulo 9 — Meta-aprendizado, warm start e ensembles
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
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
def gerar_conjunto(rng, i):
    n = int(rng.integers(150, 500)); d = int(rng.integers(8, 40)); inf = int(rng.integers(4, min(d - 2, 12) + 1)); c = int(rng.integers(2, 5))
    return make_classification(n_samples=n, n_features=d, n_informative=inf, n_redundant=int(rng.integers(0, (d - inf) // 2 + 1)),
                               n_classes=c, n_clusters_per_class=int(rng.integers(1, 3)), class_sep=float(rng.uniform(0.4, 2.0)),
                               flip_y=float(rng.uniform(0, 0.1)), random_state=i)

ALGORITMOS = {"knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
              "svm": lambda: make_pipeline(StandardScaler(), SVC()),
              "logistica": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "floresta": lambda: RandomForestClassifier(60, random_state=0, n_jobs=-1)}

def avaliar_todos(X, y):
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    return {nome: cross_val_score(f(), X, y, cv=cv).mean() for nome, f in ALGORITMOS.items()}
# --- fim dos blocos anteriores ---
C_VALORES = [0.1, 1, 10, 100, 1000]
G_RELATIVOS = [1e-3, 1e-2, 1e-1, 1, 10]
CONFIGS = [(c, g) for c in C_VALORES for g in G_RELATIVOS]

def desempenho_svm(X, y):
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    return np.array([cross_val_score(make_pipeline(StandardScaler(), SVC(C=c, gamma=g / X.shape[1])), X, y, cv=cv).mean() for c, g in CONFIGS])

rng = np.random.default_rng(5)
origem = []
for i in range(40):
    n = int(rng.integers(150, 500)); d = int(rng.integers(8, 40)); inf = int(rng.integers(4, min(d - 2, 12) + 1)); c = int(rng.integers(2, 5))
    X, y = make_classification(n_samples=n, n_features=d, n_informative=inf, n_redundant=int(rng.integers(0, (d - inf) // 2 + 1)),
                               n_classes=c, n_clusters_per_class=int(rng.integers(1, 3)), class_sep=float(rng.uniform(0.4, 2.0)),
                               flip_y=float(rng.uniform(0, 0.1)), random_state=1000 + i)
    origem.append(desempenho_svm(X, y))
origem = np.array(origem)

portfolio = []
for _ in range(3):                                           # escolha gulosa
    candidatos = [j for j in range(len(CONFIGS)) if j not in portfolio]
    portfolio.append(max(candidatos, key=lambda j: np.mean(np.max(origem[:, portfolio + [j]], axis=1))))
print("Portfólio (C, gamma relativo):", [CONFIGS[j] for j in portfolio])

print("\nMelhor acurácia após k tentativas, no conjunto alvo (real):")
print(f"{'alvo':8s} | portfólio: 1, 2, 3 tentativas | aleatório: 1, 3, 6 tentativas | melhor possível")
for nome, carregar in [("Iris", load_iris), ("Wine", load_wine), ("Cancer", load_breast_cancer), ("Digits", load_digits)]:
    X, y = carregar(return_X_y=True); p = desempenho_svm(X, y); r = np.random.default_rng(0)
    port = [p[portfolio[0]], p[portfolio[:2]].max(), p[portfolio].max()]
    aleat = [np.mean([p[r.choice(len(CONFIGS), k, replace=False)].max() for _ in range(500)]) for k in (1, 3, 6)]
    print(f"{nome:8s} |        {port[0]:.3f}, {port[1]:.3f}, {port[2]:.3f}      |       {aleat[0]:.3f}, {aleat[1]:.3f}, {aleat[2]:.3f}      |      {p.max():.3f}")
