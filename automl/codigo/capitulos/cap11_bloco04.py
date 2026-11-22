# Capítulo 11 — Busca de arquitetura neural (NAS)
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import json
import numpy as np
from scipy.stats import spearmanr
import os; os.makedirs("resultados", exist_ok=True)

T = json.load(open("resultados/f11_cache.json"))
v10 = np.array([r["v10"] for r in T]); v40 = np.array([r["v40"] for r in T])
rho, _ = spearmanr(v10, v40)
print(f"Correlação de Spearman entre a nota com 10 e com 40 iterações: {rho:.3f}")

melhores40 = set(np.argsort(-v40)[:15])                    # as 15 melhores (5%) com treino "completo"
print("\nSe avaliamos todas com 10 iterações e treinamos mais só as k melhores:")
print(" k mantidas | % do custo | melhor rede do espaço sobrevive? | nota final da melhor mantida")
for k in (10, 30, 60, 120):
    mantidas = np.argsort(-v10)[:k]
    custo = (310 * 10 + k * 40) / (310 * 40)              # em iterações de treino, relativo a treinar tudo com 40
    ok = int(np.argmax(v40)) in set(mantidas)
    print(f"   {k:>4d}     |   {100 * custo:4.0f}%    |   {'sim' if ok else 'não':>3s}                           | {v40[mantidas].max():.3f}")
print("\nMelhor do espaço (40 iterações): %.3f" % v40.max())
json.dump({"v10": v10.tolist(), "v40": v40.tolist(), "rho": float(rho)}, open("resultados/f11_fid.json", "w"))
