"""Capítulo 3: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 14]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(docs):
    toks = [d.split() for d in docs]
    vocab = sorted({t for d in toks for t in d})
    N = len(docs)
    df = np.array([sum(t in d for d in toks) for t in vocab])
    idf = np.log(N / df) + 1
    M = np.array([[d.count(t) for t in vocab] for d in toks], float) * idf
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return vocab, M

docs = ["um gato dorme", "um cachorro dorme", "um gato corre"]
vocab, M = tfidf(docs)
vec = TfidfVectorizer(smooth_idf=False)
X = vec.fit_transform(docs).toarray()
assert list(vec.get_feature_names_out()) == vocab
assert np.allclose(M, X)
print(M.round(3))

# %% [Questão discursiva 15]
from collections import Counter

def bpe_treinar(palavras, n_fusoes):
    vocab = Counter(tuple(p) + ("</w>",) for p in palavras)
    fusoes = []
    for _ in range(n_fusoes):
        pares = Counter()
        for w, f in vocab.items():
            for a, b in zip(w, w[1:]):
                pares[(a, b)] += f
        if not pares:
            break
        melhor = max(pares, key=lambda p: (pares[p], tuple(-ord(c) for c in "".join(p))))
        fusoes.append(melhor)
        novo = Counter()
        for w, f in vocab.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == melhor:
                    out.append(w[i] + w[i + 1]); i += 2
                else:
                    out.append(w[i]); i += 1
            novo[tuple(out)] += f
        vocab = novo
    return fusoes

def bpe_codificar(palavra, fusoes):
    toks = list(palavra) + ["</w>"]
    for a, b in fusoes:
        out, i = [], 0
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(toks[i]); i += 1
        toks = out
    return toks

corpus = ["lua"] * 4 + ["luas"] * 2 + ["sol"] * 3
f = bpe_treinar(corpus, 2)
assert f == [("l", "u"), ("lu", "a")]
assert bpe_codificar("luas", f) == ["lua", "s", "</w>"]
print(f, bpe_codificar("luas", f), bpe_codificar("sol", f))

# %% [Questão discursiva 16]
import numpy as np

def embedding_sentenca(frase, E, indice):
    ids = [indice[p] for p in frase.split() if p in indice]
    if not ids:
        return np.zeros(E.shape[1])
    v = E[ids].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def buscar(consulta_vec, docs_vetores, k=2):
    D = docs_vetores / np.linalg.norm(docs_vetores, axis=1, keepdims=True)
    q = consulta_vec / np.linalg.norm(consulta_vec)
    return list(np.argsort(-(D @ q))[:k])

indice = {"gato": 0, "cachorro": 1, "carro": 2, "moto": 3}
E = np.array([[1.0, 0.1], [0.9, 0.2], [0.1, 1.0], [0.2, 0.9]])
docs = ["gato cachorro", "carro moto", "gato moto"]
DV = np.array([embedding_sentenca(d, E, indice) for d in docs])
q = embedding_sentenca("cachorro", E, indice)
assert buscar(q, DV, k=1) == [0]
print(buscar(q, DV, k=3))

# %% [Questão discursiva 17]
def hit_at_k(rankings, relevantes, k):
    vals = [float(bool(set(r[:k]) & set(rel))) for r, rel in zip(rankings, relevantes) if rel]
    return sum(vals) / len(vals)

def mrr(rankings, relevantes):
    vals = []
    for r, rel in zip(rankings, relevantes):
        if not rel:
            continue
        pos = next((i for i, d in enumerate(r, 1) if d in rel), None)
        vals.append(1 / pos if pos else 0.0)
    return sum(vals) / len(vals)

rankings = [["a", "b", "c"], ["x", "y", "z"], ["m", "n", "o"]]
relevantes = [["b"], ["z"], []]       # a terceira consulta não tem resposta e é ignorada
assert hit_at_k(rankings, relevantes, 1) == 0.0
assert hit_at_k(rankings, relevantes, 3) == 1.0
assert abs(mrr(rankings, relevantes) - (1/2 + 1/3) / 2) < 1e-12
print(hit_at_k(rankings, relevantes, 3), round(mrr(rankings, relevantes), 4))
