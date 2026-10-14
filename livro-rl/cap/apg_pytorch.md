# Apêndice G — Introdução ao PyTorch
O **PyTorch** é uma biblioteca para construir e treinar **redes neurais** (PASZKE et al., 2019). Seu grande trunfo é a **diferenciação automática**: você escreve a conta, e ele calcula os gradientes sozinho.

## G.1 Tensores

Um **tensor** é como um array do NumPy, mas pode rodar em GPU e guardar o histórico para derivadas.

```python
import numpy as np
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.as_tensor(np.array([4.0, 5.0, 6.0]), dtype=torch.float32)
print(a + b, (a * b).sum(), a.shape, a.dtype)
print(a.numpy(), torch.zeros(2, 3).shape)
M = torch.arange(6.0).reshape(2, 3)
print(M @ torch.ones(3), M.T.shape, M.max(dim=1).values, M.argmax(dim=1))
```

## G.2 Gradientes automáticos

Marque um tensor com `requires_grad=True`; depois de uma conta, chame `.backward()` e leia `.grad`.

```python
w = torch.tensor(2.0, requires_grad=True)
perda = (3.0 * w - 12.0) ** 2        # perda = (3w − 12)²; mínimo em w = 4
perda.backward()
print("d perda / d w =", w.grad.item())   # 2 (3w − 12) × 3 = 2(−6)(3) = −36
```

## G.3 Treinando um modelo

O ciclo de treino tem sempre os mesmos 5 passos:

1. **Prever** (passo "para frente", *forward*).
2. Calcular a **perda** (erro).
3. **Zerar** os gradientes antigos.
4. **Retropropagar** (`backward`): calcula os gradientes.
5. **Atualizar** os pesos (`opt.step()`).

```python
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
```

## G.4 Redes como classes (nn.Module)

Para arquiteturas maiores, defina uma classe:

```python
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
```

## G.5 Ações importantes

| Comando | Para quê |
|---|---|
| `torch.no_grad()` | Desliga o cálculo de gradientes (usar ao escolher ações e calcular alvos) |
| `.detach()` | Separa um tensor do grafo de gradientes |
| `.gather(1, idx)` | Escolhe, em cada linha, o valor da coluna `idx` (valor Q da ação tomada) |
| `nn.utils.clip_grad_norm_` | Limita o tamanho do gradiente (estabiliza) |
| `load_state_dict` | Copia pesos de uma rede para outra (rede-alvo) |
| `torch.save(modelo.state_dict(), "m.pt")` | Salva o modelo |
| `modelo.eval()` / `.train()` | Alterna modos (importante em dropout e batch-norm) |

```python
q_alvo = RedeQ(4, 2); q_alvo.load_state_dict(q.state_dict())       # cópia da rede
with torch.no_grad():
    Q = q_alvo(torch.randn(5, 4))                                   # 5 estados
    print("máximo por estado:", Q.max(dim=1).values.numpy().round(2))
    escolhidas = torch.tensor([0, 1, 1, 0, 1])
    print("Q das ações escolhidas:", Q.gather(1, escolhidas.unsqueeze(1)).squeeze(1).numpy().round(2))
```

:::dica GPU
Para usar a placa de vídeo, mova modelo e dados: `dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, `modelo.to(dispositivo)`, `x = x.to(dispositivo)`. Para redes pequenas, como as deste livro, a CPU basta.
:::

## G.6 Funções de perda e otimizadores frequentes

- **Perda quadrática** (`mse_loss`): padrão em regressão.
- **Perda de Huber** (`smooth_l1_loss`): quadrática perto de zero e linear longe; menos sensível a erros enormes (usada no DQN).
- **Entropia cruzada** (`cross_entropy`): classificação e políticas discretas.
- **SGD**: o clássico. **Adam**: adapta o passo de cada parâmetro (KINGMA; BA, 2015); é a escolha padrão.
