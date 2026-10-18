"""Cap. 1 - 'Olá, rede': primeira rede treinada com a infraestrutura do livro.
Ajusta y = sen(2*pi*x) com ruído, MLP 1-8-1 (tanh), treinada por Levenberg-Marquardt
(implementação didática do livro). Semente fixa; imprime números citados no texto."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rnbook import MLP
from rnbook import otimizadores as O
from rnbook.estilo import plt, COR, salvar

rng = np.random.default_rng(2026)
N = 40
x = np.sort(rng.uniform(0, 1, N))[:, None]
t = np.sin(2 * np.pi * x) + rng.normal(0, 0.15, size=x.shape)

rede = MLP([1, 8, 1], ["tanh", "identidade"], semente=7)
print("parâmetros:", rede.n_params, "(= 8*1+8 + 1*8+1)")
E0 = rede.perda(rede.theta, x, t)
res = lambda th: rede.residuos(th, x, t)
jac = lambda th: rede.jacobiana(th, x)
r = O.levenberg_marquardt(rede.theta, res, jac, max_iter=60)
th = r.theta
E = rede.perda(th, x, t)
print(f"perda inicial (sse) = {E0:.4f}; perda final = {E:.4f}; iterações = {r.n_iter}; parada = {r.motivo}")
rmse = np.sqrt(np.mean((rede.prever(x, th) - t) ** 2))
print(f"RMSE no treino = {rmse:.4f} (ruído verdadeiro: 0.15)")

xx = np.linspace(0, 1, 300)[:, None]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(x, t, s=22, color=COR["gry"], label="dados com ruído", zorder=3)
ax[0].plot(xx, np.sin(2 * np.pi * xx), "--", color=COR["gry"], label="função verdadeira")
ax[0].plot(xx, rede.prever(xx, rede.theta), color=COR["red"], alpha=.8, label="rede antes do treino")
ax[0].plot(xx, rede.prever(xx, th), color=COR["ind"], lw=2.2, label="rede depois do treino")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].legend(); ax[0].set_title("Aprendendo uma curva")
ax[1].semilogy(r.historico["perda"], color=COR["ind"], lw=2)
ax[1].set_xlabel("iteração"); ax[1].set_ylabel("perda E (escala log)"); ax[1].set_title("A perda cai a cada iteração")
salvar(fig, "f1_primeira_rede")
