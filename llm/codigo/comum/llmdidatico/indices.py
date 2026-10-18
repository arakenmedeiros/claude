"""Índices vetoriais e fusão de rankings (Capítulo 10). NumPy puro; versões didáticas, sem otimização.
Não substituem bibliotecas maduras de busca vetorial."""
import numpy as np


def normalizar_l2(X):
    """Divide cada linha pela sua norma euclidiana (linhas nulas permanecem nulas)."""
    X = np.asarray(X, float)
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(n == 0, 1.0, n)


def pontuar(Q, D, metrica="cosseno"):
    """Matriz de pontuações (maior = mais parecido) entre consultas Q (m,d) e documentos D (N,d)."""
    Q = np.atleast_2d(np.asarray(Q, float)); D = np.asarray(D, float)
    if metrica == "produto":
        return Q @ D.T
    if metrica == "cosseno":
        return normalizar_l2(Q) @ normalizar_l2(D).T
    if metrica == "euclidiana":                      # negativa da distância: maior = mais perto
        d2 = (Q ** 2).sum(1)[:, None] - 2 * Q @ D.T + (D ** 2).sum(1)[None, :]
        return -np.sqrt(np.maximum(d2, 0))
    raise ValueError(metrica)


def knn_exato(q, D, k=10, metrica="cosseno"):
    """Busca exaustiva: devolve (índices dos k melhores, pontuações), em ordem decrescente."""
    s = pontuar(q, D, metrica)[0]
    k = min(k, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    return top, s[top]


def rrf(rankings, k=60, topo=None):
    """Fusão por rank recíproco. 'rankings' é uma lista de listas de ids (do melhor ao pior).
    Devolve a lista fundida e o dicionário de pontuações: RRF(d) = soma 1/(k + posição)."""
    pont = {}
    for rk in rankings:
        for pos, d in enumerate(rk, 1):
            pont[d] = pont.get(d, 0.0) + 1.0 / (k + pos)
    ordem = sorted(pont, key=lambda d: (-pont[d], str(d)))
    return (ordem if topo is None else ordem[:topo]), pont


def chunk_palavras(texto, tamanho, sobreposicao=0):
    """Janelas de 'tamanho' palavras, com 'sobreposicao' palavras repetidas entre janelas vizinhas.
    Devolve lista de (inicio, fim, texto) em índices de palavras."""
    pal = texto.split()
    passo = max(1, tamanho - sobreposicao)
    saida = []
    for i in range(0, len(pal), passo):
        saida.append((i, min(i + tamanho, len(pal)), " ".join(pal[i:i + tamanho])))
        if i + tamanho >= len(pal):
            break
    return saida


def colecao_sintetica(N=20000, d=64, n_centros=200, ruido=1.6, seed=0, n_consultas=200):
    """Vetores unitários em torno de centros (aglomerados), imitando embeddings de tópicos.
    As consultas são perturbações de documentos. É um conjunto SINTÉTICO, só para medir índices."""
    rng = np.random.default_rng(seed)
    C = normalizar_l2(rng.normal(size=(n_centros, d)))
    ids = rng.integers(0, n_centros, N)
    D = normalizar_l2(C[ids] + ruido * rng.normal(size=(N, d)) / np.sqrt(d))
    base = rng.integers(0, N, n_consultas)
    Q = normalizar_l2(D[base] + 0.5 * ruido * rng.normal(size=(n_consultas, d)) / np.sqrt(d))
    return D, Q


class IndiceIVF:
    """Arquivo invertido com k-means (IVF): agrupa os vetores em 'n_listas' grupos e, na consulta,
    examina só os 'nprobe' grupos de centroide mais próximo."""
    def __init__(self, n_listas=64, iteracoes=15, seed=0):
        self.n_listas, self.iteracoes, self.seed = n_listas, iteracoes, seed

    def treinar(self, D):
        rng = np.random.default_rng(self.seed)
        self.D = normalizar_l2(D)
        C = self.D[rng.choice(len(D), self.n_listas, replace=False)].copy()
        for _ in range(self.iteracoes):
            rot = np.argmax(self.D @ C.T, axis=1)
            for j in range(self.n_listas):
                m = self.D[rot == j]
                if len(m): C[j] = m.mean(0)
            C = normalizar_l2(C)
        self.C = C
        self.rot = np.argmax(self.D @ C.T, axis=1)
        self.listas = [np.where(self.rot == j)[0] for j in range(self.n_listas)]
        return self

    def buscar(self, q, k=10, nprobe=4):
        """Devolve (índices dos k melhores entre os candidatos, número de candidatos examinados)."""
        q = normalizar_l2(q)[0] if np.ndim(q) > 1 else normalizar_l2(q)
        grupos = np.argsort(-(self.C @ q))[:nprobe]
        cand = np.concatenate([self.listas[j] for j in grupos])
        s = self.D[cand] @ q
        top = cand[np.argsort(-s, kind="stable")[:k]]
        return top, len(cand)

    def adicionar(self, v):
        """Inclui um vetor novo na lista do centroide mais próximo (sem retreinar). Devolve seu índice."""
        v = normalizar_l2(v)
        self.D = np.vstack([self.D, v]); j = int(np.argmax(self.C @ v[0]))
        i = len(self.D) - 1; self.listas[j] = np.append(self.listas[j], i)
        return i


class IndiceLSH:
    """LSH com hiperplanos aleatórios: cada tabela usa 'n_bits' hiperplanos; o vetor recebe como código
    os sinais de suas projeções. Na consulta, candidatos = vetores com o mesmo código em ao menos uma tabela."""
    def __init__(self, n_tabelas=8, n_bits=10, seed=0):
        self.n_tabelas, self.n_bits, self.seed = n_tabelas, n_bits, seed

    def _codigos(self, X, t):
        bits = (X @ self.H[t]) > 0
        return bits @ (1 << np.arange(self.n_bits))

    def treinar(self, D):
        rng = np.random.default_rng(self.seed)
        self.D = normalizar_l2(D)
        self.H = rng.normal(size=(self.n_tabelas, self.D.shape[1], self.n_bits))
        self.tabelas = []
        for t in range(self.n_tabelas):
            cod = self._codigos(self.D, t); tab = {}
            for i, c in enumerate(cod): tab.setdefault(int(c), []).append(i)
            self.tabelas.append(tab)
        return self

    def buscar(self, q, k=10):
        q = normalizar_l2(q)[0] if np.ndim(q) > 1 else normalizar_l2(q)
        cand = set()
        for t in range(self.n_tabelas):
            cand.update(self.tabelas[t].get(int(self._codigos(q[None, :], t)[0]), []))
        cand = np.array(sorted(cand), dtype=int)
        if len(cand) == 0: return cand, 0
        s = self.D[cand] @ q
        return cand[np.argsort(-s, kind="stable")[:k]], len(cand)
