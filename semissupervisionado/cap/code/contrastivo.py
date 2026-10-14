"""Aprendizado auto-supervisionado (Capítulo 12). Salve na mesma pasta de utilidades.py."""
import copy
from utilidades import *
from sklearn.linear_model import LogisticRegression

def ntxent(z1, z2, tau=0.5):
    """Perda contrastiva NT-Xent (SimCLR). z1[i] e z2[i] são duas versões do mesmo exemplo (positivos); todas as outras são negativos."""
    n = z1.shape[0]; z = F.normalize(torch.cat([z1, z2]), dim=1); sim = z @ z.T / tau
    sim.fill_diagonal_(-1e9)
    alvo = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, alvo)

class Codificador(nn.Module):
    """A CNN do livro como codificador, com uma 'cabeça de projeção' usada só no pré-treino."""
    def __init__(self):
        super().__init__(); self.cnn = CNN(); self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
    def forward(self, x): return self.proj(self.cnn.atributos(x))

def pre_treinar(Xu, metodo="simclr", passos=1500, lote=256, tau=0.5, lr=1e-3, semente=0, aug=aumento_forte):
    """Pré-treina o codificador SOMENTE com imagens sem rótulo. metodo: 'simclr' (contraste entre duas versões) ou 'rotacao' (prever a rotação)."""
    torch.manual_seed(semente); rng = np.random.RandomState(semente)
    enc = Codificador(); cab = nn.Linear(128, 4); opt = torch.optim.Adam(list(enc.parameters()) + list(cab.parameters()), lr=lr)
    for passo in range(passos):
        x = Xu[torch.tensor(rng.choice(len(Xu), lote))]; enc.train()
        if metodo == "simclr": perda = ntxent(enc(aug(x)), enc(aug(x)), tau)
        else:
            k = torch.randint(0, 4, (lote,)); xr = torch.stack([torch.rot90(x, j, (2, 3)) for j in range(4)])[k, torch.arange(lote)]
            perda = F.cross_entropy(cab(enc.cnn.atributos(xr)), k)
        opt.zero_grad(); perda.backward(); opt.step()
    return enc.cnn

@torch.no_grad()
def atributos(cnn, X, lote=2000):
    cnn.eval(); return torch.cat([cnn.atributos(X[i:i + lote]) for i in range(0, len(X), lote)]).numpy()

def sonda_linear(cnn, Xl, yl, X_te, y_te):
    """Congela o codificador e treina só uma regressão logística sobre os atributos (a 'sonda linear')."""
    fl, ft = atributos(cnn, Xl), atributos(cnn, X_te)
    m = max(fl.std(), 1e-6); lr = LogisticRegression(max_iter=3000).fit(fl / m, yl.numpy())
    return lr.score(ft / m, y_te.numpy())

def ajuste_fino(cnn, Xl, yl, X_te, y_te, epocas=100, semente=0, aug=aumento_forte):
    """Continua o treino de toda a rede com os rotulados, a partir do codificador pré-treinado."""
    modelo = copy.deepcopy(cnn)
    h = treinar_supervisionado(modelo, Xl, yl, epocas=epocas, aug=aug, lr=3e-4, semente=semente, X_teste=X_te, y_teste=y_te, relatar=(epocas,))
    return h[0][2]

def carregar_ou_treinar(nome, fabricar):
    """Pré-treinar leva alguns minutos; guardamos o codificador em dados/ para reaproveitar."""
    caminho = f"dados/{nome}.pt"
    if os.path.exists(caminho):
        cnn = CNN(); cnn.load_state_dict(torch.load(caminho)); return cnn
    cnn = fabricar(); os.makedirs("dados", exist_ok=True); torch.save(cnn.state_dict(), caminho); return cnn
