# Capítulo 14 — solução de referência (EX 1)
import hashlib, json, os, sys, tempfile
import joblib, numpy as np, sklearn
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
h = lambda A: hashlib.sha256(A.tobytes()).hexdigest()[:16]
pasta = tempfile.mkdtemp(); joblib.dump(pipe, os.path.join(pasta, "m.joblib"))
ficha = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "semente": 0, "hash_treino": h(Xtr)}
print("Ficha:", json.dumps(ficha))
print("Previsões idênticas após recarregar:", np.array_equal(pipe.predict(Xte), joblib.load(os.path.join(pasta, "m.joblib")).predict(Xte)))
Xalt = Xtr.copy(); Xalt[0, 0] += 0.001
print("Hash depois de alterar um valor em 0,001:", h(Xalt), "| igual ao original?", h(Xalt) == h(Xtr))
