"""Diferenciação automática (modo reverso) didática, em NumPy (Capítulo 9).

Cada operação cria um nó com valor, pais e uma função que propaga o gradiente (vetor–Jacobiana).
Contribuições de vários filhos SOMAM-SE em 'grad' (ramificações); um mesmo nó usado duas vezes
(compartilhamento de pesos) acumula as duas contribuições. Implementação didática do livro."""
import numpy as np


def _desfazer_broadcast(g, forma):
    while g.ndim > len(forma):
        g = g.sum(axis=0)
    for i, d in enumerate(forma):
        if d == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class No:
    def __init__(self, valor, pais=(), voltar=None):
        self.valor = np.asarray(valor, dtype=float)
        self.grad = np.zeros_like(self.valor)
        self.pais, self.voltar = pais, voltar

    # operações ---------------------------------------------------------------------------------
    def __add__(self, o):
        o = _no(o)
        return No(self.valor + o.valor, (self, o), lambda g: ((self, _desfazer_broadcast(g, self.valor.shape)),
                                                             (o, _desfazer_broadcast(g, o.valor.shape))))

    __radd__ = __add__

    def __sub__(self, o):
        return self + (_no(o) * -1.0)

    def __mul__(self, o):
        o = _no(o)
        return No(self.valor * o.valor, (self, o), lambda g: ((self, _desfazer_broadcast(g * o.valor, self.valor.shape)),
                                                             (o, _desfazer_broadcast(g * self.valor, o.valor.shape))))

    __rmul__ = __mul__

    def __matmul__(self, o):
        o = _no(o)
        return No(self.valor @ o.valor, (self, o), lambda g: ((self, g @ o.valor.T), (o, self.valor.T @ g)))

    @property
    def T(self):
        return No(self.valor.T, (self,), lambda g: ((self, g.T),))

    def tanh(self):
        t = np.tanh(self.valor)
        return No(t, (self,), lambda g: ((self, g * (1 - t * t)),))

    def sigmoide(self):
        s = 1 / (1 + np.exp(-self.valor))
        return No(s, (self,), lambda g: ((self, g * s * (1 - s)),))

    def relu(self):
        return No(np.maximum(self.valor, 0), (self,), lambda g: ((self, g * (self.valor > 0)),))

    def soma(self):
        return No(self.valor.sum(), (self,), lambda g: ((self, np.full_like(self.valor, g)),))

    def media(self, eixo=None, keepdims=False):
        n = self.valor.size if eixo is None else self.valor.shape[eixo]
        return No(self.valor.mean(axis=eixo, keepdims=keepdims), (self,),
                  lambda g: ((self, np.broadcast_to(np.asarray(g) if keepdims or eixo is None else np.expand_dims(g, eixo), self.valor.shape) / n),))

    def quadrado(self):
        return No(self.valor ** 2, (self,), lambda g: ((self, 2 * self.valor * g),))

    def raiz(self):
        r = np.sqrt(self.valor)
        return No(r, (self,), lambda g: ((self, g / (2 * r)),))

    def inverso(self):
        return No(1.0 / self.valor, (self,), lambda g: ((self, -g / self.valor ** 2),))

    # retropropagação ---------------------------------------------------------------------------
    def retropropagar(self):
        ordem, visto = [], set()

        def topo(n):
            if id(n) in visto:
                return
            visto.add(id(n))
            for p in n.pais:
                topo(p)
            ordem.append(n)
        topo(self)
        for n in ordem:
            n.grad = np.zeros_like(n.valor)
        self.grad = np.ones_like(self.valor)
        for n in reversed(ordem):
            if n.voltar is not None:
                for pai, g in n.voltar(n.grad):
                    pai.grad = pai.grad + g          # SOMA das contribuições (ramificações)


def _no(x):
    return x if isinstance(x, No) else No(x)


def normalizacao_de_camada(h, eps=1e-5):
    """Layer norm sem parâmetros: (h - média)/sqrt(var + eps) ao longo do último eixo (usa só operações do grafo)."""
    mu = h.media(eixo=1, keepdims=True)
    c = h - mu
    var = c.quadrado().media(eixo=1, keepdims=True)
    return c * (var + eps).raiz().inverso()
