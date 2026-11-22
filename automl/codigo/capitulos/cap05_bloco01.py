# Capítulo 5 — Otimização bayesiana
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck

def f(x):                                   # função "cara": um morro largo e uma agulha estreita
    return 0.6 * np.exp(-((x - 3) / 1.2) ** 2) + 1.0 * np.exp(-((x - 8.2) / 0.35) ** 2)

candidatos = np.linspace(0, 10, 501)[:, None]

def otimizacao_bayesiana(semente, n_total=15, n_inicial=3):
    rng = np.random.default_rng(semente)
    X = list(rng.uniform(0, 10, n_inicial)); Y = [f(x) for x in X]       # pontos iniciais aleatórios
    for _ in range(n_total - n_inicial):
        # 1) modelo substituto: processo gaussiano treinado no histórico
        kernel = Ck(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-3))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=semente)
        gp.fit(np.array(X)[:, None], Y)
        # 2) função de aquisição: melhoria esperada (EI) em todos os candidatos
        media, desvio = gp.predict(candidatos, return_std=True)
        desvio = np.maximum(desvio, 1e-9)
        z = (media - max(Y) - 0.01) / desvio
        ei = (media - max(Y) - 0.01) * norm.cdf(z) + desvio * norm.pdf(z)
        # 3) escolher o ponto de maior aquisição e 4) avaliar de verdade
        x_novo = float(candidatos[np.argmax(ei), 0])
        X.append(x_novo); Y.append(f(x_novo))                            # 5) atualizar o histórico
    return np.maximum.accumulate(Y)

def busca_aleatoria(semente, n_total=15):
    rng = np.random.default_rng(1000 + semente)
    return np.maximum.accumulate(f(rng.uniform(0, 10, n_total)))

bo = np.array([otimizacao_bayesiana(s) for s in range(40)])
al = np.array([busca_aleatoria(s) for s in range(40)])
print("avaliações | melhor valor médio (bayesiana x aleatória) | % que achou a agulha (> 0,95)")
for k in (5, 10, 15):
    print(f"{k:10d} | {bo[:, k-1].mean():.3f} x {al[:, k-1].mean():.3f}"
          f"                      | {100*(bo[:, k-1] > 0.95).mean():3.0f}% x {100*(al[:, k-1] > 0.95).mean():3.0f}%")
