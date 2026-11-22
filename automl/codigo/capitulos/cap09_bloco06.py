# Capítulo 9 — Meta-aprendizado, warm start e ensembles
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
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
from sklearn.ensemble import VotingClassifier, StackingClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, n_redundant=4, n_classes=3,
                           n_clusters_per_class=3, class_sep=0.9, flip_y=0.08, random_state=1)
Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)      # Xte: teste final

base = [("svm", SVC(C=10, gamma=0.02, probability=True, random_state=0)),
        ("knn", KNeighborsClassifier(5, weights="distance")),
        ("lr", LogisticRegression(C=1, max_iter=2000)),
        ("rf", RandomForestClassifier(100, random_state=0, n_jobs=-1))]
cv = StratifiedKFold(5, shuffle=True, random_state=0)

print("Validação cruzada (5 partes) no conjunto de treino:")
for nome, modelo in base:
    print(f"  {nome:20s} acurácia = {cross_val_score(modelo, Xa, ya, cv=cv).mean():.3f}")
print(f"  {'votação (soft)':20s} acurácia = {cross_val_score(VotingClassifier(base, voting='soft'), Xa, ya, cv=cv).mean():.3f}")
print(f"  {'stacking':20s} acurácia = {cross_val_score(StackingClassifier(base, final_estimator=LogisticRegression(max_iter=2000), cv=5), Xa, ya, cv=cv).mean():.3f}")
# --- fim dos blocos anteriores ---
Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.33, stratify=ya, random_state=0)
rng = np.random.default_rng(0)
biblioteca = []
for i in range(40):
    tipo = str(rng.choice(["svm", "knn", "lr", "rf", "hgb"]))
    if tipo == "svm":   m = SVC(C=10 ** rng.uniform(-1, 3), gamma=10 ** rng.uniform(-4, -0.5), probability=True, random_state=0)
    elif tipo == "knn": m = KNeighborsClassifier(int(rng.integers(1, 25)), weights="distance")
    elif tipo == "lr":  m = LogisticRegression(C=10 ** rng.uniform(-3, 2), max_iter=2000)
    elif tipo == "rf":  m = RandomForestClassifier(int(rng.integers(20, 120)), max_depth=int(rng.integers(3, 20)), random_state=0, n_jobs=-1)
    else:               m = HistGradientBoostingClassifier(learning_rate=10 ** rng.uniform(-1.7, -0.3), max_iter=60, max_leaf_nodes=int(rng.integers(4, 30)), random_state=0)
    m.fit(Xtr, ytr)
    biblioteca.append((tipo, m.predict_proba(Xva), m.predict_proba(Xte)))

nota_val = [accuracy_score(yva, pv.argmax(1)) for _, pv, _ in biblioteca]
nota_teste = [accuracy_score(yte, pt.argmax(1)) for _, _, pt in biblioteca]
melhor = int(np.argmax(nota_val))

# Seleção gulosa para frente, com reposição, minimizando a perda logarítmica na validação
escolhidos, atual = [], None
for passo in range(25):
    melhor_j, melhor_perda = None, np.inf
    for j, (_, pv, _) in enumerate(biblioteca):
        candidato = pv if not escolhidos else (atual * len(escolhidos) + pv) / (len(escolhidos) + 1)
        perda = log_loss(yva, candidato, labels=range(3))
        if perda < melhor_perda: melhor_j, melhor_perda = j, perda
    atual = biblioteca[melhor_j][1] if not escolhidos else (atual * len(escolhidos) + biblioteca[melhor_j][1]) / (len(escolhidos) + 1)
    escolhidos.append(melhor_j)

pesos = np.bincount(escolhidos, minlength=len(biblioteca))
ensemble_teste = sum(pesos[j] * biblioteca[j][2] for j in range(len(biblioteca))) / pesos.sum()
media_todos = sum(pt for _, _, pt in biblioteca) / len(biblioteca)

print(f"Melhor modelo escolhido pela validação ({biblioteca[melhor][0]}): validação = {nota_val[melhor]:.3f} | TESTE = {nota_teste[melhor]:.3f}")
print(f"Média simples dos 40 modelos                        : TESTE = {accuracy_score(yte, media_todos.argmax(1)):.3f}")
print(f"Ensemble de Caruana ({(pesos > 0).sum()} modelos diferentes)        : TESTE = {accuracy_score(yte, ensemble_teste.argmax(1)):.3f}")
print(f"(Melhor modelo individual no teste, com informação do futuro: {max(nota_teste):.3f})")
print("Modelos mais usados:", [(biblioteca[j][0], int(pesos[j])) for j in np.argsort(-pesos)[:5] if pesos[j] > 0])
