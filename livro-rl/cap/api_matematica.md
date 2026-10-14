# Apêndice I — Matemática essencial, sem medo
Este apêndice reúne **tudo o que o livro usa** de matemática, com exemplos pequenos. Se uma fórmula do texto assustar, volte aqui.

## I.1 Somatório e média

O símbolo **Σ** (sigma maiúsculo) significa "some". Σ_(i=1)^3 x_i = x₁ + x₂ + x₃. A **média** de n números é a soma dividida por n.

```python
import numpy as np
x = np.array([4, 6, 5])
print("soma =", x.sum(), "| média =", x.mean())
```

## I.2 Potências, raízes, exponencial e logaritmo

- **γᵏ**: multiplica γ por ele mesmo k vezes (0,9³ = 0,729).
- **√x** (raiz quadrada): número que multiplicado por si dá x.
- **eˣ** (exponencial) cresce rápido; **ln x** (logaritmo natural) é o inverso: ln(eˣ) = x.
- Logaritmos transformam produtos em somas: ln(ab) = ln a + ln b.

```python
print(0.9 ** 3, np.sqrt(16), np.exp(1), np.log(np.exp(2)))
```

## I.3 Probabilidade

Uma **probabilidade** é um número entre 0 e 1 que mede a chance de algo ocorrer. A soma das probabilidades de todos os resultados possíveis é **1**.

- **Condicional:** P(A | B) = "chance de A **dado** que B ocorreu".
- **Independência:** A e B são independentes se saber B não muda a chance de A.
- **Variável aleatória:** número que depende do acaso (o lado de um dado).

## I.4 Esperança e variância

A **esperança** (ou valor esperado) é a média ponderada pelas probabilidades:

$$ E[X] = Σ_x x × P(X = x) $$

Para um dado justo: E = (1 + 2 + 3 + 4 + 5 + 6)/6 = 3,5. A **variância** mede a dispersão: Var(X) = E[(X − E[X])²]. O **desvio-padrão** é a raiz da variância.

```python
faces = np.arange(1, 7); p = np.full(6, 1 / 6)
E = (faces * p).sum(); var = (p * (faces - E) ** 2).sum()
print("E =", E, "| variância =", round(var, 3), "| desvio-padrão =", round(np.sqrt(var), 3))
rng = np.random.default_rng(0)
amostra = rng.integers(1, 7, 100_000)
print("média simulada de 100 mil dados:", amostra.mean().round(3))    # lei dos grandes números
```

**Lei dos grandes números:** a média de muitas amostras se aproxima da esperança. É por isso que o Monte Carlo funciona.

A **média amostral** de n valores tem desvio-padrão σ/√n: para dividir o erro por dois, é preciso **quadruplicar** as amostras.

## I.5 Distribuições usadas no livro

| Distribuição | O que descreve | Onde aparece |
|---|---|---|
| **Bernoulli(p)** | Sucesso (1) com probabilidade p, fracasso (0) caso contrário | Bandits (cliques) |
| **Uniforme** | Todos os valores igualmente prováveis | Sorteio de ações em ε-gulosa |
| **Normal (gaussiana)** | Curva em sino, com média μ e desvio σ | Ruído nas recompensas, políticas contínuas |
| **Beta(a, b)** | Crença sobre uma probabilidade | Thompson Sampling |
| **Poisson(λ)** | Contagem de eventos com média λ | Demanda de produtos |
| **Categórica** | Uma entre várias opções, com probabilidades dadas | Políticas discretas |

## I.6 Vetores e matrizes

Um **vetor** é uma lista de números; uma **matriz**, uma tabela. O **produto interno** de dois vetores soma os produtos elemento a elemento: w · x = w₁x₁ + w₂x₂ + …. O **produto matriz-vetor** faz isso para cada linha.

```python
w = np.array([0.5, -1.0, 2.0]); x = np.array([2.0, 1.0, 0.5])
print("w · x =", w @ x)
P = np.array([[0.9, 0.1], [0.5, 0.5]])         # matriz de transição: linhas somam 1
v = np.array([10.0, 0.0])
print("P v =", P @ v)                           # média de v no próximo passo, por estado
print("(I − γP)⁻¹ r =", np.linalg.solve(np.eye(2) - 0.9 * P, np.array([1.0, 0.0])).round(2))
```

A **inversa** de uma matriz A, A⁻¹, "desfaz" A. Na prática, resolvemos A z = b com `np.linalg.solve` em vez de inverter.

## I.7 Derivadas e gradiente

A **derivada** mede o quanto uma função muda quando sua entrada muda um pouco: é a **inclinação** da curva. Regras úteis:

- (x²)′ = 2x; (c·x)′ = c; (eˣ)′ = eˣ; (ln x)′ = 1/x.
- **Regra da cadeia:** a derivada de f(g(x)) é f′(g(x)) × g′(x).

O **gradiente** ∇f é o vetor de derivadas em relação a cada parâmetro. Ele aponta para onde a função **cresce mais depressa**. Descer o gradiente (ir no sentido contrário) reduz a função.

$$ w ← w − α ∇ perda(w) $$

```python
f = lambda w: (3 * w - 12) ** 2             # mínimo em w = 4
w = 0.0
for _ in range(50):
    grad = 2 * (3 * w - 12) * 3             # derivada analítica
    w -= 0.02 * grad
print("w após 50 passos de gradiente:", round(w, 3))
```

## I.8 Softmax e entropia

A função **softmax** transforma números quaisquer em probabilidades:

$$ softmax(z)_i = e^(z_i) / Σ_j e^(z_j) $$

Ela é usada para políticas: valores maiores recebem probabilidades maiores. Um parâmetro de **temperatura** controla o quão "gulosa" ela é.

A **entropia** de uma distribuição, H = −Σ p ln p, mede a **incerteza**: é máxima quando todas as opções são igualmente prováveis. Algoritmos como o SAC (Capítulo 10) a usam para incentivar exploração.

```python
def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T); return e / e.sum()
z = np.array([2.0, 1.0, 0.1])
for T in [0.2, 1.0, 5.0]:
    p = softmax(z, T); print(f"T = {T:3.1f} | p = {p.round(3)} | entropia = {-(p * np.log(p)).sum():.3f}")
```

## I.9 Divergência KL

A **divergência de Kullback-Leibler**, KL(p ‖ q) = Σ p ln(p/q), mede o quanto uma distribuição q "difere" de p. Vale 0 se forem iguais e é positiva caso contrário. Ela aparece em TRPO, PPO e RLHF (Capítulos 9, 10 e 13) para impedir que a política mude demais.

```python
p = np.array([0.5, 0.3, 0.2]); q = np.array([0.4, 0.4, 0.2])
print("KL(p‖q) =", round((p * np.log(p / q)).sum(), 4), "| KL(p‖p) =", (p * np.log(p / p)).sum())
```
