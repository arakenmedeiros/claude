# Capítulo 6 — solução de referência (EX 4)
import math
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def carregar_dados(semente=0):
    X, y = load_digits(return_X_y=True)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=semente)
    e = StandardScaler().fit(Xtr); return e.transform(Xtr), e.transform(Xva), ytr, yva

class Candidato:
    def __init__(self, rng, dados):
        self.hp = {"alpha": 10 ** rng.uniform(-8, 1), "eta0": 10 ** rng.uniform(-6, 1), "loss": str(rng.choice(["hinge", "log_loss", "modified_huber"]))}
        self.modelo = SGDClassifier(loss=self.hp["loss"], alpha=self.hp["alpha"], learning_rate="constant", eta0=self.hp["eta0"], random_state=int(rng.integers(1e6)))
        self.dados, self.epocas, self.nota = dados, 0, 0.0
    def treinar_ate(self, epocas):
        Xtr, Xva, ytr, yva = self.dados
        while self.epocas < epocas:
            self.modelo.partial_fit(Xtr, ytr, classes=np.arange(10)); self.epocas += 1; self.nota = self.modelo.score(Xva, yva)
        return self.nota

def sucessivo(candidatos, r_min, eta, r_max):
    r, gasto = r_min, 0
    while True:
        for c in candidatos:
            antes = c.epocas; c.treinar_ate(int(min(r, r_max))); gasto += c.epocas - antes     # nunca passa de r_max
        if len(candidatos) <= 1 or r >= r_max: break
        candidatos = sorted(candidatos, key=lambda c: -c.nota)[:max(1, len(candidatos) // eta)]; r *= eta
    return max(candidatos, key=lambda c: c.nota), gasto

dados = carregar_dados(0)
for eta in (2, 3, 5):
    gastos, notas = [], []
    for s in range(8):
        cands = [Candidato(np.random.default_rng(2000 + s * 1000 + i), dados) for i in range(81)]
        v, g = sucessivo(cands, 1, eta, 81); gastos.append(g); notas.append(v.nota)
    print(f"η = {eta}: gasto médio = {np.mean(gastos):6.0f} épocas | acurácia média = {np.mean(notas):.3f} (±{np.std(notas):.3f})")
