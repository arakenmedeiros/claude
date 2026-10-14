"""Modelos generativos profundos para o aprendizado semissupervisionado (Capítulo 13). Salve na mesma pasta de utilidades.py."""
from utilidades import *

# ---------- VAE ----------
class VAE(nn.Module):
    """Autocodificador variacional: o codificador devolve a média e o log da variância de z; o decodificador reconstrói a imagem."""
    def __init__(self, z=10, oculta=400):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, oculta), nn.ReLU()); self.mu = nn.Linear(oculta, z); self.lv = nn.Linear(oculta, z)
        self.dec = nn.Sequential(nn.Linear(z, oculta), nn.ReLU(), nn.Linear(oculta, 784))
    def codificar(self, x):
        h = self.enc(x); return self.mu(h), self.lv(h)
    def forward(self, x):
        mu, lv = self.codificar(x); z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)   # truque da reparametrização
        return self.dec(z), mu, lv

def perda_vae(saida, x, mu, lv, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(saida, x.flatten(1), reduction="none").sum(1)
    kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
    return (recon + beta * kl).mean(), recon.mean(), kl.mean()

def treinar_vae(Xu, z=10, passos=4000, lote=128, lr=1e-3, semente=0):
    torch.manual_seed(semente); rng = np.random.RandomState(semente); m = VAE(z); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(passos):
        x = Xu[torch.tensor(rng.choice(len(Xu), lote))]; s, mu, lv = m(x); p, _, _ = perda_vae(s, x, mu, lv)
        opt.zero_grad(); p.backward(); opt.step()
    return m

# ---------- VAE semissupervisionado (M2, Kingma et al., 2014), versão simplificada ----------
def mlp(entrada, saida, oculta=500): return nn.Sequential(nn.Linear(entrada, oculta), nn.ReLU(), nn.Linear(oculta, saida))

class M2(nn.Module):
    """q(y|x): classificador; q(z|x,y): codificador; p(x|y,z): decodificador."""
    def __init__(self, z=20):
        super().__init__(); self.z = z; self.clf = mlp(784, 10); self.enc = mlp(784 + 10, 2 * z); self.dec = mlp(z + 10, 784)
    def elbo_neg(self, x, y1):
        """Menos o limite inferior (ELBO) de x dado o rótulo y (em one-hot): reconstrução + KL de z."""
        mu, lv = self.enc(torch.cat([x, y1], 1)).chunk(2, 1); z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        recon = F.binary_cross_entropy_with_logits(self.dec(torch.cat([z, y1], 1)), x, reduction="none").sum(1)
        return recon - 0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)

def treinar_m2(Xl, yl, Xu, passos=3000, lote_u=128, alfa=100.0, peso_u=1.0, lr=1e-3, semente=0, X_teste=None, y_teste=None, relatar=()):
    torch.manual_seed(semente); rng = np.random.RandomState(semente); m = M2(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    xl = Xl.flatten(1); Xu2 = Xu.flatten(1); hist = []
    for passo in range(1, passos + 1):
        # rotulados: ELBO com o rótulo conhecido + perda de classificação
        yl1 = F.one_hot(yl, 10).float(); perda_l = m.elbo_neg(xl, yl1).mean() + alfa * F.cross_entropy(m.clf(xl), yl)
        # não rotulados: soma sobre os 10 rótulos possíveis, ponderada por q(y|x), menos a entropia de q(y|x)
        xu = Xu2[torch.tensor(rng.choice(len(Xu2), lote_u))]; q = F.softmax(m.clf(xu), 1); n = len(xu)
        L = torch.stack([m.elbo_neg(xu, F.one_hot(torch.full((n,), c), 10).float()) for c in range(10)], 1)
        perda_u = ((q * L).sum(1) + (q * torch.log(q + 1e-8)).sum(1)).mean()
        opt.zero_grad(); (perda_l + peso_u * perda_u).backward(); opt.step()
        if passo in relatar:
            m.eval()
            with torch.no_grad(): hist.append((passo, (m.clf(X_teste.flatten(1)).argmax(1) == y_teste).float().mean().item()))
    return m, hist

def treinar_mlp(Xl, yl, passos=3000, lr=1e-3, semente=0, X_teste=None, y_teste=None):
    """Referência: a mesma rede do classificador q(y|x) do M2, treinada só com os rotulados."""
    torch.manual_seed(semente); m = mlp(784, 10); opt = torch.optim.Adam(m.parameters(), lr=lr); xl = Xl.flatten(1)
    for _ in range(passos): opt.zero_grad(); F.cross_entropy(m(xl), yl).backward(); opt.step()
    with torch.no_grad(): return (m(X_teste.flatten(1)).argmax(1) == y_teste).float().mean().item()

# ---------- GAN semissupervisionada (Salimans et al., 2016) ----------
class Discriminador(nn.Module):
    """Classificador de 10 classes cujo valor 'real vs falso' vem da soma das exponenciais das pontuações."""
    def __init__(self):
        super().__init__(); self.corpo = nn.Sequential(nn.Dropout(0.2), nn.Linear(784, 1000), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(1000, 500), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(500, 250), nn.LeakyReLU(0.2)); self.fim = nn.Linear(250, 10)
    def forward(self, x):
        f = self.corpo(x.flatten(1)); return self.fim(f), f

def treinar_sgan(Xl, yl, Xu, passos=4000, lote=100, lr=3e-4, semente=0, X_teste=None, y_teste=None, relatar=()):
    torch.manual_seed(semente); rng = np.random.RandomState(semente); D = Discriminador(); G = nn.Sequential(nn.Linear(100, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 500), nn.BatchNorm1d(500), nn.ReLU(), nn.Linear(500, 784), nn.Sigmoid())
    oD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999)); oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999)); hist = []; Xu2 = Xu.flatten(1)
    for passo in range(1, passos + 1):
        D.train(); il = torch.tensor(rng.choice(len(Xl), lote)); xl = Xl[il].flatten(1); xu = Xu2[torch.tensor(rng.choice(len(Xu2), lote))]; xg = G(torch.randn(lote, 100))
        ll, _ = D(xl); lu, _ = D(xu); lg, _ = D(xg.detach())
        sup = F.cross_entropy(ll, yl[il])
        lse_u, lse_g = torch.logsumexp(lu, 1), torch.logsumexp(lg, 1)       # log Z(x); D(x) = Z / (Z + 1)
        unsup = -(lse_u - F.softplus(lse_u)).mean() - (-F.softplus(lse_g)).mean()
        oD.zero_grad(); (sup + unsup).backward(); oD.step()
        _, fu = D(xu); _, fg = D(G(torch.randn(lote, 100)))                    # gerador: casar a média dos atributos (feature matching)
        perda_g = ((fu.detach().mean(0) - fg.mean(0)) ** 2).mean(); oG.zero_grad(); perda_g.backward(); oG.step()
        if passo in relatar:
            D.eval()
            with torch.no_grad(): hist.append((passo, (D(X_teste)[0].argmax(1) == y_teste).float().mean().item()))
    return D, G, hist
