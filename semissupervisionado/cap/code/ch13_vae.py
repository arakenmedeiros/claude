import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from generativos import *

X, y, X_te, y_te = carregar_mnist()
Xu = X[np.random.RandomState(100).choice(len(X), 10000, replace=False)]          # 10.000 imagens usadas SEM seus rótulos
print("VAE treinado com 10.000 imagens sem rótulo (4.000 passos, lotes de 128)")
for z in [2, 20]:
    caminho = f"dados/vae{z}.pt"
    m = VAE(z)
    if os.path.exists(caminho): m.load_state_dict(torch.load(caminho))
    else: m = treinar_vae(Xu, z=z, passos=4000); torch.save(m.state_dict(), caminho)
    m.eval()
    with torch.no_grad(): s, mu, lv = m(X_te); p, rec, kl = perda_vae(s, X_te, mu, lv)
    print(f"  z = {z:2d}: no teste, reconstrução = {rec:.1f} nats, KL = {kl:.1f} nats, perda total = {p:.1f}")
    if z == 20: m20 = m

print("\nO VAE aprendeu algo útil para classificar? Atributos = média do codificador (z = 20), 100 rótulos, 3 sorteios")
with torch.no_grad(): Ft = m20.codificar(X_te)[0].numpy()
res = {}
for s in range(3):
    idx = sortear_rotulados(y, 10, s); yl = y[idx].numpy(); yt = y_te.numpy()
    with torch.no_grad(): Fl = m20.codificar(X[idx])[0].numpy()
    res.setdefault("regressão logística sobre os pixels", []).append(LogisticRegression(max_iter=3000).fit(X[idx].flatten(1).numpy(), yl).score(X_te.flatten(1).numpy(), yt))
    res.setdefault("regressão logística sobre z (VAE)", []).append(LogisticRegression(max_iter=3000, C=10.0).fit(Fl, yl).score(Ft, yt))
    res.setdefault("1-vizinho sobre os pixels", []).append(KNeighborsClassifier(1).fit(X[idx].flatten(1).numpy(), yl).score(X_te.flatten(1).numpy(), yt))
    res.setdefault("1-vizinho sobre z (VAE)", []).append(KNeighborsClassifier(1).fit(Fl, yl).score(Ft, yt))
for k, v in res.items(): print(f"  {k:38s} | {100 * np.mean(v):.1f}% ± {100 * np.std(v):.1f}")
