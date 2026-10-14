"""Utilidades usadas nos capítulos de aprendizado profundo (9 a 13). Salve este arquivo na mesma pasta dos outros códigos."""
import os, gzip, pickle, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino): urllib.request.urlretrieve(url, destino)
    return destino

def carregar_mnist():
    """MNIST (espelho público). Devolve tensores (N, 1, 28, 28) com valores de 0 a 1."""
    arq = baixar("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz", "dados/mnist.pkl.gz")
    (Xa, ya), _, (Xt, yt) = pickle.load(gzip.open(arq), encoding="latin1")
    f = lambda X: torch.tensor(X, dtype=torch.float32).view(-1, 1, 28, 28)
    return f(Xa), torch.tensor(ya), f(Xt), torch.tensor(yt)

def carregar_fashion():
    """Fashion-MNIST (espelho público). Mesmo formato do MNIST."""
    R = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    img = lambda n: np.frombuffer(gzip.open(baixar(R + n, "dados/fashion_" + n)).read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 255.0
    lab = lambda n: np.frombuffer(gzip.open(baixar(R + n, "dados/fashion_" + n)).read(), np.uint8, offset=8).astype(np.int64)
    return (torch.tensor(img("train-images-idx3-ubyte.gz"), dtype=torch.float32), torch.tensor(lab("train-labels-idx1-ubyte.gz")),
            torch.tensor(img("t10k-images-idx3-ubyte.gz"), dtype=torch.float32), torch.tensor(lab("t10k-labels-idx1-ubyte.gz")))

def sortear_rotulados(y, por_classe, semente):
    """Índices de 'por_classe' exemplos de cada classe."""
    r = np.random.RandomState(semente); y = np.asarray(y)
    return np.concatenate([r.choice(np.where(y == c)[0], por_classe, replace=False) for c in range(10)])

class CNN(nn.Module):
    """Rede convolucional pequena: 2 blocos de convolução + 2 camadas densas."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1); self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64 * 7 * 7, 128); self.f2 = nn.Linear(128, n_classes); self.drop = nn.Dropout(0.3)
    def atributos(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.f1(x.flatten(1)))
    def forward(self, x): return self.f2(self.drop(self.atributos(x)))

def aumentar(x, rotacao=0.0, escala=0.0, deslocamento=0.0, apagar=0, ruido=0.0):
    """Aumento de dados vetorizado: rotação (graus), escala (fração), deslocamento (fração da imagem), 'apagar' = lado de um quadrado
    apagado em posição aleatória (cutout) e ruído gaussiano. Cada imagem recebe sorteios próprios."""
    n = x.shape[0]
    ang = (torch.rand(n) * 2 - 1) * rotacao * np.pi / 180; esc = 1 + (torch.rand(n) * 2 - 1) * escala
    dx, dy = (torch.rand(n) * 2 - 1) * deslocamento * 2, (torch.rand(n) * 2 - 1) * deslocamento * 2
    th = torch.stack([torch.stack([torch.cos(ang) / esc, -torch.sin(ang) / esc, dx], 1), torch.stack([torch.sin(ang) / esc, torch.cos(ang) / esc, dy], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)
    if apagar > 0:
        cx, cy = torch.randint(0, 28, (n,)), torch.randint(0, 28, (n,)); ii = torch.arange(28)
        linhas = ((ii.view(1, 28) >= (cy - apagar // 2).view(n, 1)) & (ii.view(1, 28) < (cy + apagar // 2).view(n, 1)))
        colunas = ((ii.view(1, 28) >= (cx - apagar // 2).view(n, 1)) & (ii.view(1, 28) < (cx + apagar // 2).view(n, 1)))
        x = x * (~(linhas.view(n, 1, 28, 1) & colunas.view(n, 1, 1, 28))).float()
    if ruido > 0: x = (x + ruido * torch.randn_like(x)).clamp(0, 1)
    return x

def aumento_fraco(x): return aumentar(x, rotacao=0, escala=0.0, deslocamento=0.08)
def aumento_forte(x): return aumentar(x, rotacao=20, escala=0.2, deslocamento=0.15, apagar=10, ruido=0.1)

@torch.no_grad()
def acuracia(modelo, X, y, lote=2000):
    modelo.eval(); ok = 0
    for i in range(0, len(X), lote): ok += (modelo(X[i:i + lote]).argmax(1) == y[i:i + lote]).sum().item()
    return ok / len(X)

def treinar_supervisionado(modelo, X, y, epocas=100, aug=None, lr=1e-3, lote=32, semente=0, X_teste=None, y_teste=None, relatar=()):
    """Treino supervisionado simples (só dados rotulados). Devolve o histórico (época, acurácia de treino, acurácia de teste) nas épocas pedidas."""
    torch.manual_seed(semente); opt = torch.optim.Adam(modelo.parameters(), lr=lr); hist = []
    for ep in range(1, epocas + 1):
        modelo.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), lote):
            b = perm[i:i + lote]; xb = aug(X[b]) if aug else X[b]
            opt.zero_grad(); F.cross_entropy(modelo(xb), y[b]).backward(); opt.step()
        if ep in relatar: hist.append((ep, acuracia(modelo, X, y), acuracia(modelo, X_teste, y_teste)))
    return hist
