"""Experimento do Capítulo 3: quando os dados sem rótulo ajudam e quando prejudicam."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.model_selection import train_test_split

def duas_luas(semente, n=600):
    return make_moons(n, noise=0.1, random_state=semente)

def grupos_cruzados(semente, n=600):
    """Dois grupos (em cima e embaixo), mas a classe é o lado (esquerda/direita)."""
    r = np.random.RandomState(semente)
    grupo = r.randint(0, 2, n)
    x = r.normal(0, 1.0, n)
    X = np.c_[x, np.where(grupo == 0, -3, 3) + r.normal(0, 0.6, n)]
    return X, (x > 0).astype(int)

def comparar(gerar, referencia, n_por_classe, sementes=50, com_st=False):
    ref_, ls_, st_ = [], [], []
    for s in range(sementes):
        X, y = gerar(s)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=s)
        r = np.random.RandomState(s)
        idx = np.r_[r.choice(np.where(y_tr == 0)[0], n_por_classe, replace=False),
                    r.choice(np.where(y_tr == 1)[0], n_por_classe, replace=False)]
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        ref_.append(referencia().fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
        ls_.append(LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.8, max_iter=2000).fit(X_tr, yp).score(X_te, y_te))
        if com_st:
            st_.append(SelfTrainingClassifier(referencia(), threshold=0.9).fit(X_tr, yp).score(X_te, y_te))
    return np.array(ref_), np.array(ls_), np.array(st_)

def relatorio(nome, ref_, ls_, st_):
    print(f"{nome}")
    print(f"  referência supervisionada : {100 * ref_.mean():5.1f}%")
    for rot, v in [("Label Spreading", ls_), ("Self-training", st_)]:
        if len(v) == 0: continue
        d = v - ref_
        print(f"  {rot:16s}: {100 * v.mean():5.1f}%  (ganho {100 * d.mean():+5.1f} p.p.; melhor em {(d > 0).sum()} de {len(d)}, pior em {(d < 0).sum()})")

if __name__ == "__main__":
    relatorio("CENÁRIO 1 - duas luas (a suposição de agrupamento vale)", *comparar(duas_luas, lambda: SVC(gamma=1.0), 3))
    relatorio("CENÁRIO 2 - grupos cruzados (a suposição NÃO vale)", *comparar(grupos_cruzados, lambda: LogisticRegression(), 5, com_st=True))
