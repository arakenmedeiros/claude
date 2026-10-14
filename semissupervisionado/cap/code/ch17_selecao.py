import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading

X, y = load_digits(return_X_y=True); X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
# 12 configurações candidatas de Label Spreading (k, alpha) e self-training (limiar) a serem comparadas
cands = [("LS", k, a) for k in [3, 5, 7, 10, 15, 30] for a in [0.2, 0.8]]
def ajustar(c, idx, yp):
    return LabelSpreading(kernel="knn", n_neighbors=c[1], alpha=c[2], max_iter=200).fit(X_tr, yp)
print("Digits, 50 rótulos, 20 sorteios. Escolha de 1 entre 12 configurações do Label Spreading (k e alpha)")
print("Acurácia no teste da configuração escolhida: (a) olhando o próprio teste (errado); (b) por validação cruzada apenas nos 50 rotulados (honesto); (c) a média das 12 configurações")
A, B, C = [], [], []
for s in range(20):
    r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)]); yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    teste = [ajustar(c, idx, yp).score(X_te, y_te) for c in cands]
    # validação cruzada de 5 dobras nos 50 rotulados: esconde 10 rótulos por vez e mede a acurácia nesses 10
    cv = []
    for c in cands:
        acc = []
        for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=s).split(idx, y_tr[idx]):
            yq = np.full(len(y_tr), -1); yq[idx[tr_i]] = y_tr[idx[tr_i]]
            m = ajustar(c, idx[tr_i], yq); acc.append(np.mean(m.transduction_[idx[va_i]] == y_tr[idx[va_i]]))
        cv.append(np.mean(acc))
    A.append(max(teste)); B.append(teste[int(np.argmax(cv))]); C.append(np.mean(teste))
print(f"(a) escolhida no teste       : {100 * np.mean(A):.1f}%")
print(f"(b) escolhida por validação  : {100 * np.mean(B):.1f}%")
print(f"(c) média das 12 configurações: {100 * np.mean(C):.1f}%")
print(f"excesso de otimismo de (a) sobre (b): {100 * (np.mean(A) - np.mean(B)):.1f} pontos percentuais")
