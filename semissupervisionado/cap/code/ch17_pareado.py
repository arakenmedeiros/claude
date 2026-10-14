import numpy as np
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading

X, y = load_digits(return_X_y=True); X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
SORTEIOS = 30
print(f"Digits, 50 rótulos, {SORTEIOS} sorteios pareados (mesmos rotulados para todos os métodos). Diferença em relação à regressão logística")
ref, st, ls = [], [], []
for s in range(SORTEIOS):
    r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)]); yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    ref.append(LogisticRegression(max_iter=3000).fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
    st.append(SelfTrainingClassifier(LogisticRegression(max_iter=3000), threshold=0.7).fit(X_tr, yp).score(X_te, y_te))
    ls.append(LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=200).fit(X_tr, yp).score(X_te, y_te))
ref, st, ls = map(np.array, (ref, st, ls))
print(f"referência (reg. logística): {100 * ref.mean():.1f}% ± {100 * ref.std():.1f}")
print("método                | média da diferença | IC 95% (bootstrap) | vitórias/derrotas | p (Wilcoxon) | p (teste t pareado)")
rng = np.random.RandomState(0)
for nome, m in [("self-training (0,7)", st), ("Label Spreading (k=10)", ls)]:
    d = m - ref; boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]; lo, hi = np.percentile(boot, [2.5, 97.5])
    pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0; pt = stats.ttest_rel(m, ref).pvalue
    print(f"{nome:21s} | {100 * d.mean():+7.1f} p.p.       | [{100 * lo:+5.1f}; {100 * hi:+5.1f}]   | {int((d > 0).sum()):2d} / {int((d < 0).sum()):2d}         | {pw:.4f}       | {pt:.4f}")
