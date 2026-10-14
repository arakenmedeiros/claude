import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print("pseudo-rótulos por classe (média de 10 sorteios; o ideal seria ~120 por dígito; zeros = nenhum atingiu o limiar)")
print("modelo base             | " + " ".join(f"{d:4d}" for d in range(10)) + " | acerto dos pseudo-rótulos")
for nome, fab in [("Regressão logística", lambda: LogisticRegression(max_iter=2000)), ("k vizinhos (k=5)", lambda: KNeighborsClassifier(5)),
                    ("Floresta aleatória", lambda: RandomForestClassifier(100, random_state=0)), ("SVM (probability=True)", lambda: SVC(probability=True, random_state=0))]:
    cont = np.zeros(10); acerto = []
    for s in range(10):
        r = np.random.RandomState(s)
        idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        st = SelfTrainingClassifier(fab(), threshold=0.9).fit(X_tr, yp)
        novos = st.labeled_iter_ > 0
        cont += np.bincount(st.transduction_[novos], minlength=10)
        acerto.append(np.mean(st.transduction_[novos] == y_tr[novos]) if novos.any() else np.nan)
    print(f"{nome:23s} | " + " ".join(f"{c / 10:4.0f}" for c in cont) + " | " + ("   —" if np.all(np.isnan(acerto)) else f"{100 * np.nanmean(acerto):5.1f}%"))
