# Apêndice C — Introdução ao Python e ao ambiente de trabalho
Este apêndice é para quem **nunca programou**. Ele traz o mínimo necessário para acompanhar o livro. Não tente decorar: consulte quando precisar.

## C.1 Por que Python?

**Python** é a linguagem mais usada em ciência de dados e aprendizado de máquina. Sua sintaxe é limpa, quase como texto em inglês, e há bibliotecas prontas para quase tudo: NumPy (contas), pandas (tabelas), Matplotlib (gráficos), scikit-learn (aprendizado de máquina), Gymnasium (ambientes de AR) e PyTorch (redes neurais).

## C.2 Como começar sem instalar nada: Google Colab

O **Google Colab** é um ambiente gratuito que roda no navegador. Basta ter uma conta Google.

1. Acesse o Colab e crie um **novo notebook**.
2. Cada **célula** de código roda isoladamente com `Shift + Enter`.
3. As bibliotecas mais comuns já vêm instaladas. Para as demais, escreva uma célula com `!pip install` (veja abaixo).

:::dica Um notebook é um caderno interativo
Você escreve um pedaço de código, executa e vê o resultado logo abaixo. Os blocos de código deste livro foram escritos para serem executados **em ordem**, célula a célula.
:::

## C.3 Instalação no seu computador

Opção recomendada: instalar o **Miniconda** ou usar o `venv` do próprio Python. Depois, crie um **ambiente virtual** (uma "caixa" isolada de bibliotecas) e instale os pacotes:

```bash
python -m venv venv-rl
source venv-rl/bin/activate          # no Windows: venv-rl\Scripts\activate
pip install numpy pandas matplotlib scikit-learn scipy
pip install gymnasium ucimlrepo
pip install torch stable-baselines3
pip install jupyterlab               # para abrir notebooks: jupyter lab
```

No Colab, escreva `!pip install ucimlrepo stable-baselines3` numa célula.

:::atencao Versões
O ecossistema muda com frequência. Neste livro, os códigos foram executados com Python 3.13, NumPy 2, pandas 3, scikit-learn 1.9, Gymnasium 1.4, PyTorch 2.14 e Stable-Baselines3 2.9. Se algum comando falhar com versões diferentes, leia a mensagem de erro: quase sempre é só um nome que mudou (por exemplo, `Taxi-v3` virou `Taxi-v4`).
:::

## C.4 Variáveis e tipos de dados

Uma **variável** é um nome que guarda um valor.

```python
nome = "Ana"                 # texto (string)
idade = 31                   # número inteiro (int)
altura = 1.68                # número decimal (float)
estudante = True             # lógico (bool): True ou False
print(nome, idade, altura, estudante)
print(type(idade), type(altura))
```

Operações aritméticas: `+  -  *  /  **` (potência) `//` (divisão inteira) `%` (resto).

```python
gamma = 0.9
print(1 + gamma * 5)         # 5.5
print(gamma ** 3)            # 0.729
print(7 // 2, 7 % 2)         # 3 1
```

## C.5 Listas, tuplas e dicionários

```python
recompensas = [1, 0, 0, 5]            # lista: ordenada e alterável
print(recompensas[0], recompensas[-1], len(recompensas))   # índices começam em 0
recompensas.append(2)
print(sum(recompensas), recompensas[1:3])                  # fatia: posições 1 e 2

estado = (4, 7, False)                # tupla: ordenada e NÃO alterável
soma, crupie, as_usavel = estado      # "desempacotar"

valores = {"alta": 31.5, "baixa": 28.35}   # dicionário: pares chave -> valor
valores["media"] = 29.9
print(valores["alta"], list(valores.keys()))
```

## C.6 Condições e laços

```python
epsilon = 0.1
sorteio = 0.05
if sorteio < epsilon:
    print("explorar")
else:
    print("aproveitar")

G = 0.0
for r in reversed([1, 0, 0, 5]):      # percorre de trás para frente
    G = r + 0.9 * G
print("Retorno =", round(G, 3))

contador = 0
while contador < 3:
    contador += 1
print("Fim com contador =", contador)
```

**Atenção ao recuo (indentação):** em Python, os espaços no início da linha definem o que está **dentro** do `if`, `for` ou `while`.

## C.7 Funções

Uma **função** é um bloco de código com nome, que você chama quantas vezes quiser.

```python
def retorno_descontado(recompensas, gamma=0.9):
    """Calcula o retorno descontado de uma lista de recompensas."""
    G = 0.0
    for r in reversed(recompensas):
        G = r + gamma * G
    return G

print(retorno_descontado([1, 0, 0, 5]))
print(retorno_descontado([1, 0, 0, 5], gamma=0.5))
```

A *docstring* (texto entre aspas triplas) documenta o que a função faz.

### C.7.1 Funções anônimas (lambda) e compreensão de listas

```python
dobro = lambda x: 2 * x
print(dobro(21))

quadrados = [k ** 2 for k in range(6)]          # compreensão de lista
pares = [k for k in range(10) if k % 2 == 0]
print(quadrados, pares)
```

## C.8 Importando bibliotecas

```python
import math
import numpy as np                 # "np" é o apelido padrão
print(math.sqrt(16), np.sqrt(16))
```

## C.9 Classes (o essencial)

Uma **classe** agrupa dados e funções que operam sobre eles. Usaremos classes para criar ambientes (Apêndice F) e redes neurais (Apêndice G).

```python
class Contador:
    def __init__(self, inicio=0):      # roda ao criar o objeto
        self.valor = inicio
    def somar(self, n=1):
        self.valor += n
        return self.valor

c = Contador(10)
c.somar(); c.somar(5)
print(c.valor)
```

## C.10 Lendo mensagens de erro

Erros são normais. Leia a **última linha** da mensagem: ela diz o tipo (`NameError`, `TypeError`, `IndexError`...) e o motivo. Depois olhe a linha indicada acima.

| Erro | Significa |
|---|---|
| `NameError` | Usou um nome que não foi definido (talvez esqueceu de executar uma célula anterior) |
| `IndexError` | Índice fora do tamanho da lista |
| `TypeError` | Operação com tipos incompatíveis (ex.: texto + número) |
| `ModuleNotFoundError` | Biblioteca não instalada |

## C.11 Boas práticas em experimentos de AR

- **Fixe a semente aleatória** (`np.random.default_rng(0)`) para reproduzir resultados.
- **Repita** com várias sementes e reporte média e variação.
- **Salve** os resultados importantes (`np.save`, `to_csv`).
- **Anote** versões das bibliotecas e hiperparâmetros usados.
