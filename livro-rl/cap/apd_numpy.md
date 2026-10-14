# Apêndice D — NumPy, pandas e Matplotlib
## D.1 NumPy: arrays e contas rápidas

O **NumPy** trabalha com **arrays** (vetores e matrizes) e faz contas em todos os elementos de uma vez, sem laços.

```python
import numpy as np

v = np.array([1.0, 0.0, 0.0, 5.0])
print(v.shape, v.dtype, v.mean(), v.sum(), v.max(), v.argmax())

gammas = 0.9 ** np.arange(4)               # [1, 0.9, 0.81, 0.729]
print("Retorno =", (gammas * v).sum())     # produto elemento a elemento, e soma

M = np.array([[1, 2, 3], [4, 5, 6]])       # matriz 2×3
print(M.shape, M[1, 2], M[:, 0], M.T.shape)
print(M @ np.array([1, 0, -1]))            # produto matriz-vetor
```

### D.1.1 Criando arrays

```python
print(np.zeros((2, 3)))                    # zeros
print(np.ones(3), np.full(4, 0.25))        # uns e valor constante
print(np.arange(0, 10, 3), np.linspace(0, 1, 5))
print(np.eye(3))                           # matriz identidade
```

### D.1.2 Indexação e máscaras

```python
x = np.array([5, 3, 8, 1, 9])
print(x[x > 4])                            # elementos maiores que 4
x[x < 4] = 0                               # altera os que são menores que 4
print(x, np.where(x > 4, "alto", "baixo"))
```

### D.1.3 Números aleatórios

```python
rng = np.random.default_rng(42)            # gerador com semente
print(rng.random(3))                       # uniformes em [0, 1)
print(rng.integers(0, 4, size=5))          # inteiros em [0, 4)
print(rng.normal(0, 1, size=3))            # normal(média 0, desvio 1)
print(rng.choice(3, p=[0.2, 0.5, 0.3]))    # sorteio com probabilidades
```

### D.1.4 Broadcasting e `einsum`

O NumPy "estica" arrays de formas compatíveis automaticamente (*broadcasting*):

```python
A = np.arange(6).reshape(2, 3)             # [[0 1 2],[3 4 5]]
print(A + np.array([10, 20, 30]))          # soma a cada linha
P = np.random.default_rng(0).random((4, 3, 4)); P /= P.sum(axis=2, keepdims=True)   # P[s, a, s']
V = np.arange(4.0)
print((P @ V).shape)                       # Σ_s' P[s,a,s'] V[s'] → matriz [s, a]
```

## D.2 pandas: tabelas

O **pandas** manipula tabelas (*DataFrames*), como planilhas.

```python
import pandas as pd

df = pd.DataFrame({
    "cliente": [1, 1, 2, 2, 3],
    "mes": ["jan", "fev", "jan", "fev", "jan"],
    "gasto": [10.0, 0.0, 55.0, 30.0, 5.0]})
print(df.head())
print(df.describe())
print(df.groupby("cliente")["gasto"].sum())
print(df.pivot_table(index="cliente", columns="mes", values="gasto", aggfunc="sum", fill_value=0))
print(df[df["gasto"] > 8])
df["gasto_log"] = np.log1p(df["gasto"])
print(df.dtypes)
```

Funções úteis: `read_csv`, `to_csv`, `dropna`, `fillna`, `merge`, `sort_values`, `value_counts`, `shift`, `groupby`.

## D.3 Matplotlib: gráficos

O **Matplotlib** desenha gráficos. Dois passos: desenhar e mostrar.

```python
import matplotlib
matplotlib.use("Agg")                      # (só para rodar sem janela; no notebook, remova)
import matplotlib.pyplot as plt

t = np.arange(0, 31)
plt.figure(figsize=(6, 3))
for g, cor in [(0.5, "tab:red"), (0.9, "tab:orange"), (0.99, "tab:green")]:
    plt.plot(t, g ** t, color=cor, label=f"γ = {g}")
plt.xlabel("passos no futuro"); plt.ylabel("peso da recompensa"); plt.legend(); plt.title("Efeito do desconto")
plt.tight_layout()
plt.savefig("desconto.png", dpi=120)       # no notebook, use plt.show()
print("gráfico salvo em desconto.png")
```

Outros gráficos úteis: `plt.hist` (histograma), `plt.scatter` (dispersão), `plt.imshow` (mapas de calor), `plt.fill_between` (faixas de incerteza).

:::dica Dica de ouro para curvas de aprendizado
Curvas de AR são ruidosas. Use **médias móveis**: `np.convolve(r, np.ones(50)/50, mode="valid")`, e mostre várias sementes com uma faixa mín–máx.
:::
