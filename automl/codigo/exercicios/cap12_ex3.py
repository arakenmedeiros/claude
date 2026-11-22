# Capítulo 12 — solução de referência (EX 3)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

POS = ["ótimo", "bom", "excelente", "rápido", "perfeito", "útil"]; NEG = ["ruim", "péssimo", "lento", "fraco", "horrível", "inútil"]
SUJ = ["o produto", "a entrega", "o atendimento", "o aplicativo", "a embalagem"]

def gera(n, p_neg, seed=0):
    r = np.random.default_rng(seed); docs, ys = [], []
    for _ in range(n):
        pos = r.random() < 0.5; adj = r.choice(POS if pos else NEG); neg = r.random() < p_neg
        if neg: pos = not pos
        docs.append(f"{r.choice(SUJ)} é {'não ' if neg else ''}{adj}"); ys.append(int(pos))
    return docs, np.array(ys)

for p_neg in (0.0, 0.4):
    D, Y = gera(1200, p_neg); Dtr, Dte, ytr, yte = train_test_split(D, Y, test_size=0.25, stratify=Y, random_state=0)
    for ng in ((1, 1), (1, 2)):
        m = make_pipeline(TfidfVectorizer(ngram_range=ng), LogisticRegression(max_iter=2000)).fit(Dtr, ytr)
        print(f"negação = {int(p_neg * 100):>2d}% | n-gramas {ng}: teste = {m.score(Dte, yte):.3f}")
