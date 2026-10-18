"""Capítulo 3: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import unicodedata

for w in ["ação", "coração", "pão", "Brasil"]:
    nfc = unicodedata.normalize("NFC", w)
    nfd = unicodedata.normalize("NFD", w)
    print(f"{w:<8} NFC: {len(nfc)} caracteres, {len(nfc.encode('utf-8'))} bytes | NFD: {len(nfd)} caracteres, {len(nfd.encode('utf-8'))} bytes")

a, b = "ação", unicodedata.normalize("NFD", "ação")
print("Parecem iguais:", a, b, "| são iguais para o Python?", a == b)
print("Iguais após normalizar:", unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b))


# %% [bloco 2]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["um gato dorme", "um cachorro dorme", "um gato corre"]
vec = TfidfVectorizer(smooth_idf=False, norm="l2")
X = vec.fit_transform(docs)
print("vocabulário:", list(vec.get_feature_names_out()))
print("idf:", dict(zip(vec.get_feature_names_out(), vec.idf_.round(3).tolist())))
print("vetor de d2:", dict(zip(vec.get_feature_names_out(), X[1].toarray()[0].round(3).tolist())))
print("norma de d2:", round(float(np.linalg.norm(X[1].toarray())), 3))
S = (X @ X.T).toarray()
print("similaridades de cosseno:\n", S.round(3))


# %% [bloco 3]
import re
from llmdidatico import DOCS, modelos

texto = " ".join(d[2] + " " + d[3] for d in DOCS).lower()
palavras = re.findall(r"[a-zçáàâãéêíóôõú]+", texto)
print("palavras no corpus:", len(palavras), "| distintas:", len(set(palavras)))

fusoes = modelos.treinar_bpe(palavras, n_fusoes=200)
print("primeiras fusões:", fusoes[:6])

for w in ["trancamento", "matrículas", "desligamentos", "biblioteca", "estudante", "inconstitucionalmente", "xyz"]:
    print(f"{w:<22} -> {modelos.codificar_bpe(w, fusoes)}")

print("\nfusões  tokens por palavra (em média)")
for n in (0, 50, 100, 200, 400):
    f = modelos.treinar_bpe(palavras, n)
    media = np.mean([len(modelos.codificar_bpe(w, f)) for w in palavras])
    print(f"{n:>6}  {media:>8.2f}")


# %% [bloco 4]
from llmdidatico import corpus

frases = corpus.frases_lm(3000, seed=0)
vocab = sorted({t for f in frases for t in f})
ix = {t: i for i, t in enumerate(vocab)}
V = len(vocab)

C = np.zeros((V, V))                       # coocorrências numa janela de 2 tokens
for f in frases:
    for i, t in enumerate(f):
        for j in range(max(0, i - 2), min(len(f), i + 3)):
            if j != i:
                C[ix[t], ix[f[j]]] += 1

total = C.sum()
pw = C.sum(1, keepdims=True) / total
pc = C.sum(0, keepdims=True) / total
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((C / total) / (pw * pc))
ppmi = np.where(np.isfinite(pmi), np.maximum(pmi, 0), 0.0)

U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
d = 8
E = U[:, :d] * np.sqrt(S[:d])               # embeddings de dimensão 8
En = E / np.linalg.norm(E, axis=1, keepdims=True)

def vizinhos(palavra, k=4):
    sim = En @ En[ix[palavra]]
    ordem = np.argsort(-sim)
    return [(vocab[i], round(float(sim[i]), 2)) for i in ordem[1:k + 1]]

print("formato da tabela de embeddings:", E.shape)
for w in ["professor", "lê", "relatório", "hoje"]:
    print(f"{w:<10} -> {vizinhos(w)}")


# %% [bloco 5]
from sklearn.feature_extraction.text import TfidfVectorizer
from llmdidatico import CONSULTAS, avaliacao as av

ids = [d[0] for d in DOCS]
textos = [d[2] + ". " + d[3] for d in DOCS]

tfidf = TfidfVectorizer(tokenizer=modelos.tokenizar_sem_stop, lowercase=False, token_pattern=None, sublinear_tf=True)
Xd = tfidf.fit_transform(textos)
lsa = modelos.LSAEmbedder(dim=40).ajustar(textos)
Dl = lsa.transformar(textos)

def rank_tfidf(q):
    return np.argsort(-(Xd @ tfidf.transform([q]).T).toarray().ravel(), kind="stable")

def rank_lsa(q):
    return np.argsort(-(Dl @ lsa.transformar([q])[0]), kind="stable")

print(f"{'método':<8}{'tipo':<11}{'n':>3}{'hit@1':>8}{'hit@3':>8}")
for nome, f in [("TF-IDF", rank_tfidf), ("LSA", rank_lsa)]:
    for tipo in ("direta", "parafrase"):
        h1, h3 = [], []
        for q, rel, ans, t in CONSULTAS:
            if t != tipo: continue
            rk = [ids[i] for i in f(q)]
            h1.append(av.hit_at_k(rk, rel, 1)); h3.append(av.hit_at_k(rk, rel, 3))
        print(f"{nome:<8}{tipo:<11}{len(h1):>3}{np.mean(h1):>8.2f}{np.mean(h3):>8.2f}")


# %% [bloco 6]
import unicodedata

def sem_acentos(s):
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

def busca_literal(consulta, textos):
    return [i for i, t in enumerate(textos) if consulta in t]

print("busca 'matricula' (sem acento) no texto original:", busca_literal("matricula", [d[3].lower() for d in DOCS])[:5])
print("busca 'matricula' após normalizar:", busca_literal("matricula", [sem_acentos(d[3]) for d in DOCS])[:5])
print("sem_acentos('Ação, coração e pão') =", sem_acentos("Ação, coração e pão"))


# %% [bloco 7]
print("hoje   ~ ontem  :", round(float(En[ix["hoje"]] @ En[ix["ontem"]]), 3))
print("lê     ~ escreve:", round(float(En[ix["lê"]] @ En[ix["escreve"]]), 3))
print("lê     ~ prova  :", round(float(En[ix["lê"]] @ En[ix["prova"]]), 3))
print("hoje   ~ livro  :", round(float(En[ix["hoje"]] @ En[ix["livro"]]), 3))

