# Capítulo 14 — solução de referência (EX 3)
import numpy as np
from scipy.stats import ks_2samp
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def psi(ref, novo, k=10):
    cortes = np.quantile(ref, np.linspace(0, 1, k + 1)[1:-1])
    a = np.bincount(np.searchsorted(cortes, ref), minlength=k) / len(ref)
    b = np.bincount(np.searchsorted(cortes, novo), minlength=k) / len(novo)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))

d = load_wine(); Xr, Xp = train_test_split(d.data, test_size=0.3, random_state=0)
psis = [psi(Xr[:, j], Xp[:, j]) for j in range(Xr.shape[1])]
print("Sem deriva - três maiores PSI:", ", ".join(f"{d.feature_names[j]} ({psis[j]:.3f})" for j in np.argsort(psis)[::-1][:3]))
j = d.feature_names.index("alcohol"); Xd = Xp.copy(); Xd[:, j] *= 1.3
print(f"alcohol sem deriva : KS = {ks_2samp(Xr[:, j], Xp[:, j]).statistic:.3f} | PSI = {psi(Xr[:, j], Xp[:, j]):.3f}")
print(f"alcohol com +30%   : KS = {ks_2samp(Xr[:, j], Xd[:, j]).statistic:.3f} | PSI = {psi(Xr[:, j], Xd[:, j]):.3f}")
