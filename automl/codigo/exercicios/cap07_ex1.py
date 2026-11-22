# Capítulo 7 — solução de referência (EX 1)
import numpy as np

rng = np.random.default_rng(0)
aptidoes = np.arange(1, 21)                     # 1 = pior, 20 = melhor
for k in (2, 3, 5):
    escolhidos = np.array([max(rng.choice(20, k, replace=False), key=lambda i: aptidoes[i]) for _ in range(20000)])
    freq = np.bincount(escolhidos, minlength=20) / 20000
    print(f"k={k}: melhor escolhido em {100*freq[19]:4.1f}% | pior escolhido em {100*freq[0]:4.1f}% | aptidão média dos escolhidos = {aptidoes[escolhidos].mean():.2f}")
print("Se a escolha fosse totalmente aleatória: 5,0% para cada indivíduo e aptidão média 10,50")
