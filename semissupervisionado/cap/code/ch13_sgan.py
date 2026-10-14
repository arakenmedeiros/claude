import numpy as np, torch
from generativos import *

X, y, X_te, y_te = carregar_mnist()
Xu = X[np.random.RandomState(100).choice(len(X), 10000, replace=False)]
PASSOS = (500, 1000, 2000)
print("GAN semissupervisionada (discriminador de 10 classes + gerador, feature matching) x o mesmo discriminador só com os rotulados (MNIST, 100 rótulos, 3 sorteios)")
print("passos | GAN semissupervisionada | só rotulados (mesmo discriminador, mesmo dropout)")
S, R = [], []
for s in range(3):
    idx = sortear_rotulados(y, 10, s)
    D, G, h = treinar_sgan(X[idx], y[idx], Xu, passos=PASSOS[-1], semente=s, X_teste=X_te, y_teste=y_te, relatar=PASSOS); S.append([a for _, a in h])
    if s == 0: os.makedirs("dados", exist_ok=True); torch.save(G.state_dict(), "dados/sgan_G.pt")
    torch.manual_seed(s); rng = np.random.RandomState(s); Db = Discriminador(); opt = torch.optim.Adam(Db.parameters(), lr=3e-4, betas=(0.5, 0.999)); acc = []
    for passo in range(1, PASSOS[-1] + 1):
        Db.train(); il = torch.tensor(rng.choice(len(idx), 100)); l, _ = Db(X[idx][il]); opt.zero_grad(); F.cross_entropy(l, y[idx][il]).backward(); opt.step()
        if passo in PASSOS:
            Db.eval()
            with torch.no_grad(): acc.append((Db(X_te)[0].argmax(1) == y_te).float().mean().item())
    R.append(acc)
S, R = np.array(S), np.array(R)
for i, p in enumerate(PASSOS): print(f"{p:6d} | {100 * S[:, i].mean():16.1f}% ± {100 * S[:, i].std():.1f}  | {100 * R[:, i].mean():.1f}% ± {100 * R[:, i].std():.1f}")
