# Capítulo 5 — solução de referência (EX 1)
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck

g = lambda x: np.sin(3 * x) * np.exp(-0.25 * x) + 0.5 * np.sin(9 * x)
malha = np.linspace(0, 6, 601)[:, None]
maximo = g(malha[:, 0]).max()
print("Máximo verdadeiro (malha fina):", round(maximo, 3))

def bo(semente, n_total=12, n_inicial=3):
    rng = np.random.default_rng(semente)
    X = list(rng.uniform(0, 6, n_inicial)); Y = [g(x) for x in X]
    for _ in range(n_total - n_inicial):
        k = Ck(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-3))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=semente).fit(np.array(X)[:, None], Y)
        mu, sd = gp.predict(malha, return_std=True); sd = np.maximum(sd, 1e-9)
        z = (mu - max(Y) - 0.01) / sd; ei = (mu - max(Y) - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
        x = float(malha[np.argmax(ei), 0]); X.append(x); Y.append(g(x))
    return np.maximum.accumulate(Y)

def aleatoria(semente, n_total=12):
    return np.maximum.accumulate(g(np.random.default_rng(500 + semente).uniform(0, 6, n_total)))

B = np.array([bo(s) for s in range(20)]); A = np.array([aleatoria(s) for s in range(20)])
print("avaliações | melhor médio (BO x aleatória) | % a menos de 0,05 do máximo")
for k in (4, 8, 12):
    print(f"{k:10d} | {B[:, k-1].mean():.3f} x {A[:, k-1].mean():.3f}            | "
          f"{100*(B[:, k-1] > maximo-0.05).mean():3.0f}% x {100*(A[:, k-1] > maximo-0.05).mean():3.0f}%")
