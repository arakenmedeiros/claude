# Capítulo 6 — Métodos multifidelidade: Successive Halving, Hyperband e BOHB
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import math
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def carregar_dados(semente=0):
    X, y = load_digits(return_X_y=True)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=semente)
    escala = StandardScaler().fit(Xtr)
    return escala.transform(Xtr), escala.transform(Xva), ytr, yva

class Candidato:
    """Uma configuração que pode ser treinada aos poucos (pausar e retomar)."""
    def __init__(self, rng, dados):
        self.hp = {"alpha": 10 ** rng.uniform(-8, 1), "eta0": 10 ** rng.uniform(-6, 1),
                   "loss": str(rng.choice(["hinge", "log_loss", "modified_huber"]))}
        self.modelo = SGDClassifier(loss=self.hp["loss"], alpha=self.hp["alpha"], learning_rate="constant",
                                    eta0=self.hp["eta0"], random_state=int(rng.integers(1e6)))
        self.dados, self.epocas, self.nota, self.curva = dados, 0, 0.0, []

    def treinar_ate(self, epocas):
        Xtr, Xva, ytr, yva = self.dados
        while self.epocas < epocas:
            self.modelo.partial_fit(Xtr, ytr, classes=np.arange(10))      # uma época
            self.epocas += 1
            self.nota = self.modelo.score(Xva, yva)                        # acurácia na validação
            self.curva.append(self.nota)
        return self.nota
