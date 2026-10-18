"""Perceptron de Rosenblatt (implementação didática do livro).

Rótulos aceitos: {0,1} ou {-1,+1}. Internamente usamos y in {-1,+1}.
Regra (por erro): se y_i * (w.x_i + b) <= 0, então  w <- w + eta*y_i*x_i ; b <- b + eta*y_i.
(Equivale à regra  w <- w + eta*(t - y_hat)*x  com t, y_hat in {0,1} e saída 1 se z>=0.)
"""
import numpy as np


class PerceptronDidatico:
    def __init__(self, eta=1.0, max_epocas=100, embaralhar=False, semente=0):
        self.eta, self.max_epocas, self.embaralhar, self.semente = eta, max_epocas, embaralhar, semente

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.where(np.asarray(y) > 0, 1, -1)
        rng = np.random.default_rng(self.semente)
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        self.historico_erros, self.trajetoria = [], [(self.w.copy(), self.b)]
        self.n_atualizacoes, self.convergiu = 0, False
        for epoca in range(self.max_epocas):
            ordem = rng.permutation(len(y)) if self.embaralhar else np.arange(len(y))
            erros = 0
            for i in ordem:
                if y[i] * (X[i] @ self.w + self.b) <= 0:       # classificação errada (ou na fronteira)
                    self.w = self.w + self.eta * y[i] * X[i]
                    self.b = self.b + self.eta * y[i]
                    erros += 1
                    self.n_atualizacoes += 1
                    self.trajetoria.append((self.w.copy(), self.b))
            self.historico_erros.append(erros)
            if erros == 0:
                self.convergiu = True
                break
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)   # 1 se z>0
