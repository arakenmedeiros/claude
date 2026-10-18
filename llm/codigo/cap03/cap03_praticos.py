"""Capítulo 3: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 3.1: Fertilidade do BPE em texto novo]
import re
import numpy as np
from llmdidatico import DOCS, corpus, modelos

pal = lambda t: re.findall(r"[a-zçáàâãéêíóôõú]+", t.lower())
treino = pal(" ".join(d[2] + " " + d[3] for d in DOCS))
msgs, _ = corpus.intencoes(40, seed=1)
novo = pal(" ".join(msgs))
print("palavras de treino:", len(treino), "| palavras do texto novo:", len(novo))

def fert(palavras, fusoes):
    return float(np.mean([len(modelos.codificar_bpe(w, fusoes)) for w in palavras]))

print(f"{'fusões':>7} {'treino':>8} {'texto novo':>11}")
for n in (50, 200, 600):
    f = modelos.treinar_bpe(treino, n)
    print(f"{n:>7} {fert(treino, f):>8.2f} {fert(novo, f):>11.2f}")
f = modelos.treinar_bpe(treino, 200)
print("reembolso ->", modelos.codificar_bpe("reembolso", f))


# %% [Exercício 3.2: Embeddings LSA e agrupamento das intenções]
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics import adjusted_rand_score
from llmdidatico import corpus

X, y = corpus.intencoes(120, seed=0)
T = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode").fit_transform(X)
Z = TruncatedSVD(n_components=20, random_state=0).fit_transform(T)
Z = Normalizer().fit_transform(Z)

g_svd = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(Z)
g_tf = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(T)
print("ARI com SVD (20 dim.):", round(adjusted_rand_score(y, g_svd), 3))
print("ARI sobre o TF-IDF  :", round(adjusted_rand_score(y, g_tf), 3))
print()
print(pd.crosstab(pd.Series(y, name="intenção"), pd.Series(g_svd, name="grupo")))


# %% [Exercício 3.3: O efeito da normalização de acentos na busca]
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from llmdidatico import DOCS, CONSULTAS, avaliacao as av

def sem_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

ids = [d[0] for d in DOCS]
textos = [d[2] + ". " + d[3] for d in DOCS]
diretas = [(q, rel) for q, rel, a, t in CONSULTAS if t == "direta"]

def hit3(vec, consultas):
    X = vec.fit_transform(textos)
    hits = []
    for q, rel in consultas:
        s = (X @ vec.transform([q]).T).toarray().ravel()
        hits.append(av.hit_at_k([ids[i] for i in np.argsort(-s, kind="stable")], rel, 3))
    return float(np.mean(hits))

orig = diretas
sema = [(sem_acentos(q), rel) for q, rel in diretas]
print(f"{'índice':<22}{'consultas originais':>21}{'consultas sem acento':>22}")
print(f"{'sem normalização':<22}{hit3(TfidfVectorizer(), orig):>21.3f}{hit3(TfidfVectorizer(), sema):>22.3f}")
print(f"{'com strip_accents':<22}{hit3(TfidfVectorizer(strip_accents='unicode'), orig):>21.3f}{hit3(TfidfVectorizer(strip_accents='unicode'), sema):>22.3f}")


# %% [Projeto: Explorando embeddings por coocorrência]
import numpy as np
from llmdidatico import corpus

frases = corpus.frases_lm(3000, seed=0)
vocab = sorted({t for f in frases for t in f})
ix = {t: i for i, t in enumerate(vocab)}

CAT = {
    "sujeito": ["estudante", "professor", "professora", "coordenadora", "bibliotecário"],
    "verbo": ["lê", "escreve", "entrega", "empresta", "corrige", "explica", "devolve", "estuda", "procura", "assina", "organiza", "prepara"],
    "objeto": ["livro", "artigo", "relatório", "prova", "trabalho"],
    "tempo_lugar": ["hoje", "ontem", "casa", "biblioteca", "laboratório", "atenção"],
}
cat_de = {w: c for c, ws in CAT.items() for w in ws if w in ix}

def embeddings(J, d):
    V = len(vocab); C = np.zeros((V, V))
    for f in frases:
        for i, t in enumerate(f):
            for j in range(max(0, i - J), min(len(f), i + J + 1)):
                if j != i:
                    C[ix[t], ix[f[j]]] += 1
    tot = C.sum(); pw = C.sum(1, keepdims=True) / tot; pc = C.sum(0, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / tot) / (pw * pc))
    ppmi = np.where(np.isfinite(pmi), np.maximum(pmi, 0), 0.0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    E = U[:, :d] * np.sqrt(S[:d])
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

def precisao_vizinho(E):
    acertos = []
    for w, c in cat_de.items():
        sim = E @ E[ix[w]]; sim[ix[w]] = -2
        acertos.append(cat_de.get(vocab[int(sim.argmax())]) == c)
    return float(np.mean(acertos))

print(f"palavras categorizadas: {len(cat_de)}")
print(f"{'janela':>7} | " + " ".join(f"d={d:>2}" for d in (2, 4, 8, 16)))
for J in (1, 2, 4):
    print(f"{J:>7} | " + " ".join(f"{precisao_vizinho(embeddings(J, d)):>4.2f}" for d in (2, 4, 8, 16)))

