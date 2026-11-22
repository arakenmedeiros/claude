# Capítulo 12 — AutoML para séries temporais, texto e imagens
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split

r = np.random.default_rng(0)
POS = ["ótimo", "bom", "excelente", "rápido", "perfeito", "útil"]; NEG = ["ruim", "péssimo", "lento", "fraco", "horrível", "inútil"]
SUJ = ["o produto", "a entrega", "o atendimento", "o aplicativo", "a embalagem"]; VERB = ["é", "foi", "achei", "parece"]
FIL = ["no geral", "sinceramente", "para mim", "nesta semana", "de verdade", "como esperado"]
def gera(n):
    docs, ys = [], []
    for _ in range(n):
        pos = r.random() < 0.5; adj = r.choice(POS if pos else NEG); neg = r.random() < 0.4
        if neg: pos = not pos                                    # a negação inverte o sentido
        f = f"{r.choice(SUJ)} {r.choice(VERB)} {'não ' if neg else ''}{adj}"
        if r.random() < 0.6: f = r.choice(FIL) + " " + f
        docs.append(f); ys.append(int(pos))
    return docs, np.array(ys)

D, Y = gera(1200)
Dtr, Dte, ytr, yte = train_test_split(D, Y, test_size=0.25, stratify=Y, random_state=0)
print("Exemplos de treino:")
for d, l in list(zip(Dtr, ytr))[:3]: print(f"  [{'positivo' if l else 'negativo'}] {d}")

CLASS = {"LogReg": lambda: LogisticRegression(max_iter=2000), "SVM linear": lambda: LinearSVC(), "Naive Bayes": lambda: MultinomialNB()}
res = []
for ng, cl, sub in itertools.product([(1, 1), (1, 2), (1, 3)], CLASS, [False, True]):
    p = make_pipeline(TfidfVectorizer(ngram_range=ng, sublinear_tf=sub), CLASS[cl]())
    res.append((cross_val_score(p, Dtr, ytr, cv=5).mean(), ng, cl, sub))
res.sort(reverse=True)
print("\nValidação cruzada (acurácia), melhores e piores de 18 configurações:")
for a, ng, cl, sub in res[:3] + res[-3:]: print(f"  {a:.3f}  n-gramas={ng}  {cl:<11s}  sublinear={sub}")
a, ng, cl, sub = res[0]
p = make_pipeline(TfidfVectorizer(ngram_range=ng, sublinear_tf=sub), CLASS[cl]()).fit(Dtr, ytr)
p1 = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000)).fit(Dtr, ytr)
print(f"\nTESTE com a melhor configuração : {p.score(Dte, yte):.3f}")
print(f"TESTE com o pipeline padrão     : {p1.score(Dte, yte):.3f}  (palavras isoladas, regressão logística)")
