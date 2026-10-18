"""Rede MLP didática: forward, backpropagation vetorizado, resíduos e Jacobiana.

Implementação didática do livro (não é a API do scikit-learn).

Convenções (usadas em todo o livro):
  * amostras nas LINHAS: X tem forma (N, n0); alvos T têm forma (N, m).
  * camada l: z^(l) = W^(l) a^(l-1) + b^(l);  a^(l) = phi_l(z^(l)),
    com W^(l) de forma (n_l, n_{l-1}) e b^(l) de forma (n_l,).
    Em lote: Z = A_prev @ W.T + b.
  * vetor de parâmetros theta: concatena, camada a camada, W (linha a linha) e b.
  * resíduo r = y - t (predição MENOS alvo), empilhado na ordem (amostra, saída):
    índice p*m + k.   Jacobiana J = dr/dtheta, de forma (N*m, P).
  * perda: E = c * sum(r**2), com c = 1/2 ('sse') ou c = 1/(N*m) ('mse').
    Gradiente: grad E = 2 c J^T r.
"""
import numpy as np
from .ativacoes import ATIVACOES


class MLP:
    def __init__(self, camadas, ativacoes, semente=0, init="xavier"):
        """camadas: [n0, n1, ..., nL]; ativacoes: lista com L nomes."""
        assert len(ativacoes) == len(camadas) - 1
        self.camadas = list(camadas)
        self.nomes_ativ = list(ativacoes)
        self.phi = [ATIVACOES[a][0] for a in ativacoes]
        self.dphi = [ATIVACOES[a][1] for a in ativacoes]
        self.L = len(camadas) - 1
        self.formas = []
        for l in range(self.L):
            self.formas.append(((camadas[l + 1], camadas[l]), (camadas[l + 1],)))
        self.n_params = sum(w[0] * w[1] + b[0] for w, b in self.formas)
        self.rng = np.random.default_rng(semente)
        self.theta = self.inicializar(init)

    # ------------------------------------------------------------ parâmetros
    def inicializar(self, metodo="xavier"):
        partes = []
        for l, (fw, fb) in enumerate(self.formas):
            fan_out, fan_in = fw
            if metodo == "xavier":      # Glorot & Bengio (2010), versão uniforme
                lim = np.sqrt(6.0 / (fan_in + fan_out))
                W = self.rng.uniform(-lim, lim, size=fw)
            elif metodo == "he":        # He et al. (2015), versão normal
                W = self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=fw)
            elif metodo == "pequena":
                W = self.rng.normal(0.0, 0.5, size=fw)
            else:
                raise ValueError(metodo)
            partes += [W.ravel(), np.zeros(fb)]
        return np.concatenate(partes)

    def desempacotar(self, theta):
        """theta -> lista de (W, b) (visões, sem cópia)."""
        out, i = [], 0
        for fw, fb in self.formas:
            nw = fw[0] * fw[1]
            W = theta[i:i + nw].reshape(fw)
            i += nw
            b = theta[i:i + fb[0]]
            i += fb[0]
            out.append((W, b))
        return out

    def fatias(self):
        """Índices (ini, fim) de W e b de cada camada dentro de theta."""
        out, i = [], 0
        for fw, fb in self.formas:
            nw = fw[0] * fw[1]
            out.append(((i, i + nw), (i + nw, i + nw + fb[0])))
            i += nw + fb[0]
        return out

    # --------------------------------------------------------------- forward
    def forward(self, X, theta=None):
        theta = self.theta if theta is None else theta
        A = [np.atleast_2d(X)]
        Z = []
        for l, (W, b) in enumerate(self.desempacotar(theta)):
            z = A[-1] @ W.T + b
            Z.append(z)
            A.append(self.phi[l](z))
        return A, Z

    def prever(self, X, theta=None):
        return self.forward(X, theta)[0][-1]

    # ----------------------------------------------------------------- perda
    @staticmethod
    def _c(perda, N, m):
        if perda == "sse":
            return 0.5
        if perda == "mse":
            return 1.0 / (N * m)
        raise ValueError(perda)

    def residuos(self, theta, X, T):
        """r = y - t, achatado na ordem (amostra, saída)."""
        return (self.prever(X, theta) - T).reshape(-1)

    def perda(self, theta, X, T, perda="sse"):
        N, m = T.shape
        r = self.residuos(theta, X, T)
        return self._c(perda, N, m) * float(r @ r)

    # ---------------------------------------------------------- backprop vet.
    def grad(self, theta, X, T, perda="sse"):
        """Gradiente de E por backpropagation vetorizado (retorna vetor P)."""
        A, Z = self.forward(X, theta)
        N, m = T.shape
        c = self._c(perda, N, m)
        params = self.desempacotar(theta)
        delta = 2.0 * c * (A[-1] - T) * self.dphi[-1](Z[-1])   # delta^(L)
        g = np.empty_like(theta)
        sl = self.fatias()
        for l in range(self.L - 1, -1, -1):
            (iw, fw), (ib, fb) = sl[l]
            g[iw:fw] = (delta.T @ A[l]).ravel()
            g[ib:fb] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ params[l][0]) * self.dphi[l - 1](Z[l - 1])
        return g

    def perda_e_grad(self, theta, X, T, perda="sse"):
        return self.perda(theta, X, T, perda), self.grad(theta, X, T, perda)

    # ------------------------------------------------- Jacobiana dos resíduos
    def jacobiana(self, theta, X):
        """J = dr/dtheta, forma (N*m, P).  Como r = y - t e t não depende de
        theta, J = dy/dtheta.  Um backward por saída k, com 'semente' unitária."""
        A, Z = self.forward(X, theta)
        N = A[0].shape[0]
        m = self.camadas[-1]
        params = self.desempacotar(theta)
        sl = self.fatias()
        J = np.zeros((N, m, self.n_params))
        for k in range(m):
            semente = np.zeros((N, m))
            semente[:, k] = 1.0
            delta = semente * self.dphi[-1](Z[-1])
            for l in range(self.L - 1, -1, -1):
                (iw, fw), (ib, fb) = sl[l]
                # gradiente por amostra: delta_p (x) a_{l-1,p}
                J[:, k, iw:fw] = (delta[:, :, None] * A[l][:, None, :]).reshape(N, -1)
                J[:, k, ib:fb] = delta
                if l > 0:
                    delta = (delta @ params[l][0]) * self.dphi[l - 1](Z[l - 1])
        return J.reshape(N * m, self.n_params)
