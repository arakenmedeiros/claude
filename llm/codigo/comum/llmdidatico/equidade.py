"""Capítulo 17: dados SINTÉTICOS de triagem com atributo protegido e métricas de equidade (mesmo código do texto do capítulo)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



def gerar_candidatos(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.random(n) < 0.35                                          # True = grupo B
    aptidao = rng.normal(np.where(g, -0.4, 0.0), 1.0)                  # taxas-base diferentes
    y = (aptidao + rng.normal(0, 0.5, n) > 0.2).astype(int)           # "qualificado de fato"
    nota = aptidao + rng.normal(0, 0.6, n) - np.where(g, 0.3, 0.0)    # medida enviesada contra B
    horas = 0.5 * aptidao + rng.normal(0, 1.0, n)
    return np.c_[nota, horas], y, g.astype(int)

def por_grupo(y, pred, g):
    out = {}
    for k, nome in ((0, "A"), (1, "B")):
        m = g == k; tn, fp, fn, tp = confusion_matrix(y[m], pred[m], labels=[0, 1]).ravel()
        out[nome] = dict(sel=pred[m].mean(), tpr=tp / (tp + fn), fpr=fp / (fp + tn), ppv=tp / (tp + fp), base=y[m].mean())
    return out

def limiar_para(score, y, g, alvo, k, tipo):
    m = g == k; cands = np.unique(np.quantile(score[m], np.linspace(0.01, 0.99, 197)))
    def f(t):
        p = score[m] >= t
        return p.mean() if tipo == "sel" else (p & (y[m] == 1)).sum() / (y[m] == 1).sum()
    return min(cands, key=lambda t: abs(f(t) - alvo))

