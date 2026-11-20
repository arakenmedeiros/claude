"""
Livro: Aprendizado por Reforço — Apêndice G — Introdução ao PyTorch
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.as_tensor(np.array([4.0, 5.0, 6.0]), dtype=torch.float32)
print(a + b, (a * b).sum(), a.shape, a.dtype)
print(a.numpy(), torch.zeros(2, 3).shape)
M = torch.arange(6.0).reshape(2, 3)
print(M @ torch.ones(3), M.T.shape, M.max(dim=1).values, M.argmax(dim=1))

# ======================================================================
# Bloco 2
# ======================================================================
w = torch.tensor(2.0, requires_grad=True)
perda = (3.0 * w - 12.0) ** 2        # perda = (3w − 12)²; mínimo em w = 4
perda.backward()
print("d perda / d w =", w.grad.item())   # 2 (3w − 12) × 3 = 2(−6)(3) = −36

# ======================================================================
# Bloco 3
# ======================================================================
import torch.nn as nn

torch.manual_seed(0)
x = torch.linspace(-3, 3, 200).reshape(-1, 1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)

modelo = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.Adam(modelo.parameters(), lr=1e-2)

for epoca in range(600):
    pred = modelo(x)                                   # 1) prever
    perda = nn.functional.mse_loss(pred, y)            # 2) perda
    opt.zero_grad()                                    # 3) zerar gradientes
    perda.backward()                                   # 4) retropropagar
    opt.step()                                         # 5) atualizar
    if epoca % 200 == 0: print(f"época {epoca:3d} | perda = {perda.item():.4f}")
print(f"perda final = {perda.item():.4f}")

# ======================================================================
# Bloco 4
# ======================================================================
class RedeQ(nn.Module):
    def __init__(self, n_obs, n_acoes, ocultas=64):
        super().__init__()
        self.camadas = nn.Sequential(nn.Linear(n_obs, ocultas), nn.ReLU(),
                                     nn.Linear(ocultas, ocultas), nn.ReLU(),
                                     nn.Linear(ocultas, n_acoes))
    def forward(self, estado):
        return self.camadas(estado)

q = RedeQ(4, 2)
estado = torch.tensor([[0.0, 0.1, -0.05, 0.0]])
print("valores Q:", q(estado).detach().numpy().round(3))
print("nº de parâmetros:", sum(p.numel() for p in q.parameters()))

# ======================================================================
# Bloco 5
# ======================================================================
q_alvo = RedeQ(4, 2); q_alvo.load_state_dict(q.state_dict())       # cópia da rede
with torch.no_grad():
    Q = q_alvo(torch.randn(5, 4))                                   # 5 estados
    print("máximo por estado:", Q.max(dim=1).values.numpy().round(2))
    escolhidas = torch.tensor([0, 1, 1, 0, 1])
    print("Q das ações escolhidas:", Q.gather(1, escolhidas.unsqueeze(1)).squeeze(1).numpy().round(2))

