"""
Livro: Aprendizado por Reforço — Apêndice C — Introdução ao Python e ao ambiente de trabalho
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
nome = "Ana"                 # texto (string)
idade = 31                   # número inteiro (int)
altura = 1.68                # número decimal (float)
estudante = True             # lógico (bool): True ou False
print(nome, idade, altura, estudante)
print(type(idade), type(altura))

# ======================================================================
# Bloco 2
# ======================================================================
gamma = 0.9
print(1 + gamma * 5)         # 5.5
print(gamma ** 3)            # 0.729
print(7 // 2, 7 % 2)         # 3 1

# ======================================================================
# Bloco 3
# ======================================================================
recompensas = [1, 0, 0, 5]            # lista: ordenada e alterável
print(recompensas[0], recompensas[-1], len(recompensas))   # índices começam em 0
recompensas.append(2)
print(sum(recompensas), recompensas[1:3])                  # fatia: posições 1 e 2

estado = (4, 7, False)                # tupla: ordenada e NÃO alterável
soma, crupie, as_usavel = estado      # "desempacotar"

valores = {"alta": 31.5, "baixa": 28.35}   # dicionário: pares chave -> valor
valores["media"] = 29.9
print(valores["alta"], list(valores.keys()))

# ======================================================================
# Bloco 4
# ======================================================================
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

# ======================================================================
# Bloco 5
# ======================================================================
def retorno_descontado(recompensas, gamma=0.9):
    """Calcula o retorno descontado de uma lista de recompensas."""
    G = 0.0
    for r in reversed(recompensas):
        G = r + gamma * G
    return G

print(retorno_descontado([1, 0, 0, 5]))
print(retorno_descontado([1, 0, 0, 5], gamma=0.5))

# ======================================================================
# Bloco 6
# ======================================================================
dobro = lambda x: 2 * x
print(dobro(21))

quadrados = [k ** 2 for k in range(6)]          # compreensão de lista
pares = [k for k in range(10) if k % 2 == 0]
print(quadrados, pares)

# ======================================================================
# Bloco 7
# ======================================================================
import math
import numpy as np                 # "np" é o apelido padrão
print(math.sqrt(16), np.sqrt(16))

# ======================================================================
# Bloco 8
# ======================================================================
class Contador:
    def __init__(self, inicio=0):      # roda ao criar o objeto
        self.valor = inicio
    def somar(self, n=1):
        self.valor += n
        return self.valor

c = Contador(10)
c.somar(); c.somar(5)
print(c.valor)

