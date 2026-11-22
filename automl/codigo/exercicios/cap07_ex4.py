# Capítulo 7 — solução de referência (EX 4)
import numpy as np

f = lambda x: 0.6 * np.exp(-((x - 3) / 1.2) ** 2) + 1.0 * np.exp(-((x - 8.2) / 0.35) ** 2)

def ga(semente, sigma, elitismo, pop=10, geracoes=8):
    rng = np.random.default_rng(semente)
    P = rng.uniform(0, 10, pop); melhor = 0.0
    for g in range(geracoes + 1):
        aptidao = f(P); melhor = max(melhor, aptidao.max())
        if g == geracoes: break
        nova = [P[aptidao.argmax()]] if elitismo else []
        while len(nova) < pop:
            a, b = (P[max(rng.choice(pop, 3, replace=False), key=lambda i: aptidao[i])] for _ in range(2))
            w = rng.uniform(); filho = w * a + (1 - w) * b + rng.normal(0, sigma) if sigma > 0 else w * a + (1 - w) * b
            nova.append(float(np.clip(filho, 0, 10)))
        P = np.array(nova)
    return melhor

print("mutação σ | elitismo | melhor médio | % que achou a agulha")
for sigma, elit in [(0, False), (0, True), (0.5, False), (0.5, True)]:
    r = np.array([ga(s, sigma, elit) for s in range(40)])
    print(f"{sigma:9} | {str(elit):8s} | {r.mean():12.3f} | {100 * (r > 0.95).mean():5.0f}%")
