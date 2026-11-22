# Capítulo 5 — solução de referência (EX 4)
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck

f = lambda x: 0.6 * np.exp(-((x - 3) / 1.2) ** 2) + 1.0 * np.exp(-((x - 8.2) / 0.35) ** 2)
candidatos = np.linspace(0, 10, 501)[:, None]

def bo_ucb(semente, kappa, n_total=15, n_inicial=3):
    rng = np.random.default_rng(semente)
    X = list(rng.uniform(0, 10, n_inicial)); Y = [f(x) for x in X]
    for _ in range(n_total - n_inicial):
        k = Ck(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-3))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=semente).fit(np.array(X)[:, None], Y)
        mu, sd = gp.predict(candidatos, return_std=True)
        x = float(candidatos[np.argmax(mu + kappa * sd), 0]); X.append(x); Y.append(f(x))
    return max(Y)

print("kappa | melhor médio | % que achou a agulha")
for kappa in (0.5, 2, 6):
    r = np.array([bo_ucb(s, kappa) for s in range(30)])
    print(f"{kappa:5} | {r.mean():12.3f} | {100*(r > 0.95).mean():5.0f}%")
