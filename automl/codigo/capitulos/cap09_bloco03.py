# Capítulo 9 — Meta-aprendizado, warm start e ensembles
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
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
import collections

rng = np.random.default_rng(0)
M, R = [], []
for i in range(150):
    X, y = gerar_conjunto(rng, i)
    M.append(meta_atributos(X, y)); R.append(avaliar_todos(X, y))
M = np.array(M); melhor = np.array([max(r, key=r.get) for r in R])
print("Quantas vezes cada algoritmo foi o melhor nos 150 conjuntos:", dict(collections.Counter(melhor)))

acertos, perdas = [], []
for i in range(len(M)):
    treino = np.arange(len(M)) != i
    previsto = RandomForestClassifier(200, random_state=0).fit(M[treino], melhor[treino]).predict(M[i:i+1])[0]
    acertos.append(previsto == melhor[i]); perdas.append(max(R[i].values()) - R[i][previsto])

frequente = collections.Counter(melhor).most_common(1)[0][0]
perdas_freq = [max(r.values()) - r[frequente] for r in R]
print(f"\nAcerto do meta-modelo ao indicar o melhor algoritmo : {np.mean(acertos):.3f}")
print(f"Acerto de 'sempre {frequente}' (linha de base)           : {collections.Counter(melhor)[frequente] / len(M):.3f}")
print(f"Perda média de acurácia frente ao melhor: meta-modelo = {np.mean(perdas):.4f} | sempre {frequente} = {np.mean(perdas_freq):.4f}")

meta_modelo = RandomForestClassifier(300, random_state=0).fit(M, melhor)
print("\nTeste em conjuntos REAIS (nunca vistos pelo meta-modelo):")
for nome, carregar in [("Iris", load_iris), ("Wine", load_wine), ("Cancer", load_breast_cancer), ("Digits", load_digits)]:
    X, y = carregar(return_X_y=True); notas = avaliar_todos(X, y)
    rec = meta_modelo.predict(np.array(meta_atributos(X, y))[None])[0]
    real = max(notas, key=notas.get)
    print(f"  {nome:7s} recomendado = {rec:9s} ({notas[rec]:.3f}) | melhor de verdade = {real:9s} ({notas[real]:.3f})")
