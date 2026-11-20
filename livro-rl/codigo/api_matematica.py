"""
Livro: Aprendizado por Reforço — Apêndice I — Matemática essencial, sem medo
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np
x = np.array([4, 6, 5])
print("soma =", x.sum(), "| média =", x.mean())

# ======================================================================
# Bloco 2
# ======================================================================
print(0.9 ** 3, np.sqrt(16), np.exp(1), np.log(np.exp(2)))

# ======================================================================
# Bloco 3
# ======================================================================
faces = np.arange(1, 7); p = np.full(6, 1 / 6)
E = (faces * p).sum(); var = (p * (faces - E) ** 2).sum()
print("E =", E, "| variância =", round(var, 3), "| desvio-padrão =", round(np.sqrt(var), 3))
rng = np.random.default_rng(0)
amostra = rng.integers(1, 7, 100_000)
print("média simulada de 100 mil dados:", amostra.mean().round(3))    # lei dos grandes números

# ======================================================================
# Bloco 4
# ======================================================================
w = np.array([0.5, -1.0, 2.0]); x = np.array([2.0, 1.0, 0.5])
print("w · x =", w @ x)
P = np.array([[0.9, 0.1], [0.5, 0.5]])         # matriz de transição: linhas somam 1
v = np.array([10.0, 0.0])
print("P v =", P @ v)                           # média de v no próximo passo, por estado
print("(I − γP)⁻¹ r =", np.linalg.solve(np.eye(2) - 0.9 * P, np.array([1.0, 0.0])).round(2))

# ======================================================================
# Bloco 5
# ======================================================================
f = lambda w: (3 * w - 12) ** 2             # mínimo em w = 4
w = 0.0
for _ in range(50):
    grad = 2 * (3 * w - 12) * 3             # derivada analítica
    w -= 0.02 * grad
print("w após 50 passos de gradiente:", round(w, 3))

# ======================================================================
# Bloco 6
# ======================================================================
def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T); return e / e.sum()
z = np.array([2.0, 1.0, 0.1])
for T in [0.2, 1.0, 5.0]:
    p = softmax(z, T); print(f"T = {T:3.1f} | p = {p.round(3)} | entropia = {-(p * np.log(p)).sum():.3f}")

# ======================================================================
# Bloco 7
# ======================================================================
p = np.array([0.5, 0.3, 0.2]); q = np.array([0.4, 0.4, 0.2])
print("KL(p‖q) =", round((p * np.log(p / q)).sum(), 4), "| KL(p‖p) =", (p * np.log(p / p)).sum())

