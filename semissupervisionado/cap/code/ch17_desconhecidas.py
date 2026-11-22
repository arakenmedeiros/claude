import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import roc_auc_score

X, y = load_digits(return_X_y=True); X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
conhecida = y_te <= 4
print("Digits: 25 rótulos, só dos dígitos 0 a 4 (5 por dígito). Sem rótulo: TODO o treino, inclusive os dígitos 5 a 9. Teste: dígitos 0 a 4 (conhecidos) e 5 a 9 (desconhecidos). 10 sorteios")
print("método                        | acerto nos conhecidos | AUROC (desconhecido x conhecido) | desconhecidos rejeitados | conhecidos rejeitados")
def prob(m, A):
    P = m.predict_proba(A); return np.where(np.isnan(P), 1.0 / P.shape[1], P)       # o Label Spreading devolve NaN para pontos sem massa de rótulo: tratamos como 'sem informação'
res = {k: [] for k in ["lr", "ls"]}
for s in range(10):
    r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y_tr == k)[0], 5, replace=False) for k in range(5)]); yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    lr = LogisticRegression(max_iter=3000).fit(X_tr[idx], y_tr[idx]); ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=200).fit(X_tr, yp)
    for nome, m in [("lr", lr), ("ls", ls)]:
        P = prob(m, X_te); conf = P.max(1); pred = m.classes_[P.argmax(1)]
        # limiar escolhido SEM usar o teste: 10% de rejeição entre os próprios rotulados, por validação cruzada
        cs = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(idx, y_tr[idx]):
            if nome == "lr": cs.append(LogisticRegression(max_iter=3000).fit(X_tr[idx[a]], y_tr[idx[a]]).predict_proba(X_tr[idx[b]]).max(1))
            else:
                yq = np.full(len(y_tr), -1); yq[idx[a]] = y_tr[idx[a]]; cs.append(prob(LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=200).fit(X_tr, yq), X_tr[idx[b]]).max(1))
        lim = np.percentile(np.concatenate(cs), 10)
        res[nome].append([np.mean(pred[conhecida] == y_te[conhecida]), roc_auc_score((~conhecida).astype(int), -conf), np.mean(conf[~conhecida] < lim), np.mean(conf[conhecida] < lim)])
for nome, rot in [("lr", "reg. logística (só rotulados)"), ("ls", "Label Spreading (usa os sem rótulo)")]:
    v = np.array(res[nome]); print(f"{rot:29s} | {100 * v[:, 0].mean():20.1f}% | {v[:, 1].mean():.3f} ± {v[:, 1].std():.3f}              | {100 * v[:, 2].mean():22.1f}% | {100 * v[:, 3].mean():.1f}%")
