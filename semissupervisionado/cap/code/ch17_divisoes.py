import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading

X, y = load_digits(return_X_y=True); X = X / 16.0
DIVISOES, SORTEIOS = 10, 10
print(f"Digits, 50 rótulos. Label Spreading (k = 10) menos regressão logística, em {DIVISOES} divisões treino-teste diferentes x {SORTEIOS} sorteios de rotulados")
dif = np.zeros((DIVISOES, SORTEIOS))
for d in range(DIVISOES):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=d)
    for s in range(SORTEIOS):
        r = np.random.RandomState(1000 * d + s); idx = np.concatenate([r.choice(np.where(y_tr == k)[0], 5, replace=False) for k in range(10)]); yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        dif[d, s] = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=200).fit(X_tr, yp).score(X_te, y_te) - LogisticRegression(max_iter=3000).fit(X_tr[idx], y_tr[idx]).score(X_te, y_te)
rng = np.random.RandomState(0)
print("divisão | diferença média nos 10 sorteios (p.p.)")
for d in range(DIVISOES): print(f"{d:7d} | {100 * dif[d].mean():+.1f}")
m_div = dif.mean(axis=1)
def boot_dentro(): return dif[0].mean()
b_fixa = [rng.choice(dif[0], SORTEIOS).mean() for _ in range(5000)]
b_ambos = [np.mean([rng.choice(dif[k], SORTEIOS).mean() for k in rng.choice(DIVISOES, DIVISOES)]) for _ in range(5000)]
print(f"\nSó variando os rotulados (divisão 0 fixa, {SORTEIOS} sorteios): {100 * dif[0].mean():+.1f} p.p., IC 95% [{100 * np.percentile(b_fixa, 2.5):+.1f}; {100 * np.percentile(b_fixa, 97.5):+.1f}]")
print(f"Variando também a divisão treino-teste ({DIVISOES} divisões): {100 * dif.mean():+.1f} p.p., IC 95% [{100 * np.percentile(b_ambos, 2.5):+.1f}; {100 * np.percentile(b_ambos, 97.5):+.1f}]; desvio entre as médias das divisões: {100 * m_div.std():.1f} p.p.")
