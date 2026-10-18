"""Capítulo 1: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 1.1: Contando n-gramas e medindo a esparsidade]
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from llmdidatico import corpus

frases = corpus.frases_lm(3000, seed=0)
textos = [" ".join(f) for f in frases]

vec = CountVectorizer(ngram_range=(3, 3), token_pattern=r"\S+", lowercase=False)
X = vec.fit_transform(textos)
cont = np.asarray(X.sum(axis=0)).ravel()
nomes = vec.get_feature_names_out()

print("Trigramas mais frequentes:")
for i in np.argsort(-cont)[:5]:
    print(f"  {cont[i]:5d}  {nomes[i]}")

V = len({t for f in frases for t in f})
distintos = len(nomes)
print("Vocabulário:", V, "| trigramas distintos observados:", distintos)
print("Trigramas possíveis (V^3):", V ** 3)
print(f"Fração observada: {distintos / V**3:.4%}")


# %% [Exercício 1.2: Um classificador simples como linha de base]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from llmdidatico import corpus

X, y = corpus.intencoes(120, seed=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print("Acurácia no teste:", round(accuracy_score(yte, pred), 4))
print(classification_report(yte, pred, digits=3))

proba = clf.predict_proba(Xte)
classes = list(clf.classes_)
p_correta = np.array([proba[i, classes.index(yte[i])] for i in range(len(yte))])
print("Três casos mais difíceis:")
for i in np.argsort(p_correta)[:3]:
    print(f"  P(correta)={p_correta[i]:.2f} | real={yte[i]} | previsto={pred[i]} | {Xte[i]}")


# %% [Exercício 1.3: Plausível não é o mesmo que verdadeiro]
from llmdidatico import corpus, modelos

frases = corpus.frases_lm(3000, seed=0)
treino = frases[:2400]
lm = modelos.NgramaLM(n=3, k=0.05).ajustar(treino)

candidatas = {
    "verdadeira no mundo, fora do padrão": [
        "o estudante corrige a prova .",
        "a professora empresta o livro .",
        "o bibliotecário escreve o artigo .",
    ],
    "sem sentido factual, dentro do padrão": [
        "o estudante lê o livro hoje .",
        "a professora explica a prova antes da prova .",
        "o estudante entrega o relatório na biblioteca .",
    ],
}
for grupo, fs in candidatas.items():
    print(grupo)
    for f in fs:
        toks = f.split()
        lp = lm.logprob_frase(toks) / len(toks)
        print(f"   lnP/token = {lp:6.2f} | {f}")


# %% [Projeto: Comparando três formas de achar o documento certo]
import numpy as np
from llmdidatico import DOCS, CONSULTAS, modelos, avaliacao as av

ids = [d[0] for d in DOCS]
textos = [d[2] + ". " + d[3] for d in DOCS]
toks_docs = [set(modelos.tokenizar_sem_stop(t)) for t in textos]
bm = modelos.BM25([modelos.tokenizar_sem_stop(t) for t in textos])
emb = modelos.LSAEmbedder(dim=40).ajustar(textos)
D = emb.transformar(textos)

def rank_palavras(q):
    qs = set(modelos.tokenizar_sem_stop(q))
    return np.argsort(-np.array([len(qs & t) for t in toks_docs]), kind="stable")

def rank_bm25(q):
    return np.argsort(-bm.pontuar(modelos.tokenizar_sem_stop(q)), kind="stable")

def rank_lsa(q):
    return np.argsort(-(D @ emb.transformar([q])[0]), kind="stable")

metodos = {"palavras em comum": rank_palavras, "BM25": rank_bm25, "LSA (40 dim.)": rank_lsa}
falhas = {m: [] for m in metodos}
print(f"{'método':<18}{'tipo':<11}{'n':>3}{'hit@1':>8}{'hit@3':>8}")
for nome, f in metodos.items():
    for tipo in ("direta", "parafrase"):
        h1, h3 = [], []
        for q, rel, ans, t in CONSULTAS:
            if t != tipo: continue
            rk = [ids[i] for i in f(q)]
            h1.append(av.hit_at_k(rk, rel, 1)); h3.append(av.hit_at_k(rk, rel, 3))
            if not h1[-1]: falhas[nome].append((q, rk[0], rel))
        print(f"{nome:<18}{tipo:<11}{len(h1):>3}{np.mean(h1):>8.2f}{np.mean(h3):>8.2f}")
for nome, lst in falhas.items():
    print(f"\nFalhas de {nome} no topo (duas primeiras):")
    for q, top, rel in lst[:2]:
        print(f"  {q!r} -> recuperou {top}, esperado {rel}")

