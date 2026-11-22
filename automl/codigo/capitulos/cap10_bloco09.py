# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 9. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import json
import numpy as np
from scipy.stats import friedmanchisquare, rankdata
import os; os.makedirs("resultados", exist_ok=True)

d = json.load(open("resultados/f10_cache.json"))
metodos, res = d["metodos"], d["res"]
bases = list(res)
M = np.array([[np.mean(res[b][m]) for m in metodos] for b in bases])     # bases x métodos

print("Acurácia média por base e método:")
print(" " * 9 + "  ".join(f"{m[:11]:>11s}" for m in metodos))
for b, lin in zip(bases, M):
    print(f"{b:<9s}" + "  ".join(f"{v:11.3f}" for v in lin))

postos = np.array([rankdata(-lin) for lin in M])     # posto 1 = maior acurácia; empates recebem a média
print("\nPosto médio (1 = melhor):")
for m, p in zip(metodos, postos.mean(0)): print(f"  {m:<16s} {p:.2f}")

est, p = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
print(f"\nTeste de Friedman: estatística = {est:.2f}, valor-p = {p:.3f}  (k = {M.shape[1]} métodos, N = {M.shape[0]} bases)")
d["postos"] = postos.mean(0).tolist(); d["friedman_p"] = float(p)
json.dump(d, open("resultados/f10_cache.json", "w"))
