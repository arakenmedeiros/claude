import numpy as np, torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph
from sklearn.semi_supervised import LabelSpreading
from ssl_profundo import *
from contrastivo import carregar_ou_treinar, pre_treinar, sonda_linear

X, y, X_te, y_te = carregar_mnist()
SORTEIOS, PASSOS = 5, 1500
Xu_pre = X[np.random.RandomState(100).choice(len(X), 10000, replace=False)]

# ---------- 1. Diagnóstico: os dados formam grupos? (usa os rótulos verdadeiros só para diagnosticar) ----------
pca = PCA(50, random_state=0).fit(X.flatten(1).numpy()[:10000]); Z = pca.transform(X.flatten(1).numpy()[:10000]); Zt = pca.transform(X_te.flatten(1).numpy())
G = kneighbors_graph(Z, 10); i, j = G.nonzero(); print(f"Diagnóstico: com 10 vizinhos no PCA de 50 dimensões, {100 * np.mean(y.numpy()[:10000][i] == y.numpy()[:10000][j]):.1f}% das arestas ligam imagens da mesma classe")

# ---------- 2. Os métodos, com os mesmos rotulados e os mesmos não rotulados ----------
enc = carregar_ou_treinar("simclr_forte", lambda: pre_treinar(Xu_pre, "simclr", passos=1500, lote=128))
nomes = ["regressão logística, pixels", "Label Spreading, PCA 50", "CNN só com rotulados", "Pseudo-Label", "FixMatch", "UDA", "SimCLR + sonda linear"]
res = {n: [] for n in nomes}; modelos0 = {}
for s in range(SORTEIOS):
    idx = sortear_rotulados(y, 10, s); u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
    Zl = pca.transform(X[idx].flatten(1).numpy()); Zu = pca.transform(X[u].flatten(1).numpy())
    res[nomes[0]].append(LogisticRegression(max_iter=3000).fit(X[idx].flatten(1).numpy(), y[idx].numpy()).score(X_te.flatten(1).numpy(), y_te.numpy()))
    yp = np.r_[y[idx].numpy(), -np.ones(len(u), int)]; ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=100).fit(np.vstack([Zl, Zu]), yp); res[nomes[1]].append(ls.score(Zt, y_te.numpy()))
    for nome, m in [(nomes[2], "sup"), (nomes[3], "pseudo"), (nomes[4], "fixmatch"), (nomes[5], "uda")]:
        aluno, _, h = treinar_ssl(m, X[idx], y[idx], X[u], passos=PASSOS, semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,)); res[nome].append(h[0][1])
        if s == 0: modelos0[nome] = aluno
    res[nomes[6]].append(sonda_linear(enc, X[idx], y[idx], X_te, y_te))

print(f"\nMNIST, 100 rótulos, 10.000 imagens sem rótulo, {SORTEIOS} sorteios pareados. Acerto no teste")
print("método                       | média ± desvio")
for n in nomes: print(f"{n:28s} | {100 * np.mean(res[n]):.1f}% ± {100 * np.std(res[n]):.1f}")

# ---------- 3. Comparação pareada com a referência forte (CNN só com rotulados) ----------
ref = np.array(res["CNN só com rotulados"]); rng = np.random.RandomState(0)
print("\nDiferença em relação à CNN só com rotulados (pontos percentuais)")
print("método                       | média  | IC 95% (bootstrap) | vitórias/derrotas | p (teste t pareado)")
for n in nomes:
    if n == "CNN só com rotulados": continue
    d = np.array(res[n]) - ref; boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]; lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{n:28s} | {100 * d.mean():+5.1f}  | [{100 * lo:+5.1f}; {100 * hi:+5.1f}]    | {int((d > 0).sum())} / {int((d < 0).sum())}             | {stats.ttest_rel(res[n], ref).pvalue:.4f}")

# ---------- 4. Análise por classe (sorteio 0): onde o melhor método ajuda ----------
melhor = max(["Pseudo-Label", "FixMatch", "UDA"], key=lambda n: np.mean(res[n])); print(f"\nAcerto por dígito no teste (sorteio 0): CNN só com rotulados x {melhor}")
pred = {}
for n in ["CNN só com rotulados", melhor]:
    m = modelos0[n]; m.eval()
    with torch.no_grad(): pred[n] = torch.cat([m(X_te[i:i + 2000]).argmax(1) for i in range(0, len(X_te), 2000)])
for d in range(10): print(f"  dígito {d}: {100 * (pred['CNN só com rotulados'][y_te == d] == d).float().mean():5.1f}% x {100 * (pred[melhor][y_te == d] == d).float().mean():5.1f}%")
