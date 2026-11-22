# Capítulo 9 — solução de referência (EX 2)
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, n_redundant=4, n_classes=3,
                           n_clusters_per_class=3, class_sep=0.9, flip_y=0.08, random_state=1)
base = [("svm", SVC(C=10, gamma=0.02, probability=True, random_state=0)), ("knn", KNeighborsClassifier(5, weights="distance")),
        ("lr", LogisticRegression(C=1, max_iter=2000)), ("rf", RandomForestClassifier(100, random_state=0, n_jobs=-1))]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for tipo in ("hard", "soft"):
    print(f"votação {tipo:4s}: acurácia = {cross_val_score(VotingClassifier(base, voting=tipo), X, y, cv=cv).mean():.3f}")
