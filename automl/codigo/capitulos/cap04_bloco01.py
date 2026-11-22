# Capítulo 4 — Busca em grade e aleatória
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from scipy.stats import loguniform, uniform

rng = np.random.default_rng(0)
uni = uniform(0.01, 1000 - 0.01).rvs(10000, random_state=0)
log = loguniform(0.01, 1000).rvs(10000, random_state=0)

print("Fração dos valores abaixo de 1:")
print(f"  sorteio uniforme    : {np.mean(uni < 1):.3f}")
print(f"  sorteio logarítmico : {np.mean(log < 1):.3f}")
