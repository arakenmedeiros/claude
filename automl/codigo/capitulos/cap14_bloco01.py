# Capítulo 14 — MLOps, implantação e projeto integrador
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import hashlib, json, os, sys, tempfile
import joblib, numpy as np, sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
pipe = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=0)).fit(Xtr, ytr)

pasta = tempfile.mkdtemp()
joblib.dump(pipe, os.path.join(pasta, "modelo.joblib"))
ficha = {
    "python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "numpy": np.__version__,
    "semente": 0, "divisao_teste": 0.25, "hash_treino": hashlib.sha256(Xtr.tobytes()).hexdigest()[:16],
    "acuracia_teste": round(float(pipe.score(Xte, yte)), 4),
}
json.dump(ficha, open(os.path.join(pasta, "ficha.json"), "w"), indent=2)

recarregado = joblib.load(os.path.join(pasta, "modelo.joblib"))
identicas = np.array_equal(pipe.predict_proba(Xte), recarregado.predict_proba(Xte))
print("Ficha do modelo:", json.dumps(ficha, ensure_ascii=False))
print("Previsões idênticas após recarregar:", identicas)
