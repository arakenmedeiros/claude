# Capítulo 3 — solução de referência (EX 4)
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

X, y = load_wine(return_X_y=True)
pipe = Pipeline([("escala", StandardScaler()), ("knn", KNeighborsClassifier())])
grade = {"escala": [StandardScaler(), MinMaxScaler(), "passthrough"], "knn__n_neighbors": [1, 3, 5, 9, 15]}
busca = GridSearchCV(pipe, grade, cv=5).fit(X, y)

res = pd.DataFrame(busca.cv_results_)[["param_escala", "param_knn__n_neighbors", "mean_test_score"]]
res["param_escala"] = res["param_escala"].astype(str).str.replace("()", "", regex=False)
print(res.sort_values("mean_test_score", ascending=False).head(3).to_string(index=False))
print("Pior combinação:", round(res["mean_test_score"].min(), 3), "| melhor:", round(res["mean_test_score"].max(), 3))
