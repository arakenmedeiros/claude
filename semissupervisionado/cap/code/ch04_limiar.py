import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

print("limiar | pseudo-rótulos na 1ª rodada | acerto deles | acurácia final no teste")
for limiar in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    n1, ac1, fim = [], [], []
    for s in range(10):
        r = np.random.RandomState(s)
        idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        st = SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=limiar).fit(X_tr, yp)
        novos = st.labeled_iter_ == 1
        n1.append(novos.sum()); ac1.append(np.mean(st.transduction_[novos] == y_tr[novos]) if novos.any() else np.nan)
        fim.append(st.score(X_te, y_te))
    print(f"{limiar:6.2f} | {np.mean(n1):26.0f} | {100 * np.nanmean(ac1):11.1f}% | {100 * np.mean(fim):21.1f}%")
