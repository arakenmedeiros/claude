import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_mnist()
PASSOS, SEMENTES = 1500, 2
prop = 0.1 ** (np.arange(10) / 9); prop = prop / prop.sum()          # proporção das classes entre os SEM rótulo: 10 vezes mais zeros que noves
print(f"MNIST, 100 rótulos balanceados; os sem rótulo seguem proporções desiguais (classe 0: {100 * prop[0]:.0f}%, classe 9: {100 * prop[9]:.0f}%); teste balanceado; {PASSOS} passos, {SEMENTES} sorteios")
print("método                                  | acerto médio (teste balanceado) | acerto nas classes 7 a 9 | acerto nas classes 0 a 2 | acerto ponderado pela proporção dos sem rótulo")
for nome, m, kw in [("só rotulados", "sup", {}), ("FixMatch", "fixmatch", {}), ("FixMatch + alinhamento uniforme", "fixmatch", dict(alinhar=True))]:
    g, raro, freq, pond = [], [], [], []
    for s in range(SEMENTES):
        idx = sortear_rotulados(y, 10, s); r = np.random.RandomState(100 + s); resto = np.setdiff1d(np.arange(len(X)), idx)
        u = np.concatenate([r.choice(resto[y.numpy()[resto] == c], int(round(10000 * prop[c])), replace=True) for c in range(10)])
        aluno, _, _ = treinar_ssl(m, X[idx], y[idx], X[u], passos=PASSOS, semente=s, X_teste=X_te, y_teste=y_te, relatar=(), **kw); aluno.eval()
        with torch.no_grad(): pred = torch.cat([aluno(X_te[i:i + 2000]).argmax(1) for i in range(0, len(X_te), 2000)])
        ac = np.array([(pred[y_te == c] == c).float().mean().item() for c in range(10)])
        g.append(ac.mean()); raro.append(ac[7:].mean()); freq.append(ac[:3].mean()); pond.append((ac * prop).sum())
    print(f"{nome:39s} | {100 * np.mean(g):.1f}% ± {100 * np.std(g):.1f}                  | {100 * np.mean(raro):.1f}%                   | {100 * np.mean(freq):.1f}%                   | {100 * np.mean(pond):.1f}%")
