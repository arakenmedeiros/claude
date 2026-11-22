# Capítulo 1 — solução de referência (EX 4)
from math import prod

espaco = {"max_depth": 10, "min_samples_leaf": 8, "criterion": 2}
n = prod(espaco.values())
print("Combinações:", n, "| tempo total:", n * 3, "s =", round(n * 3 / 60, 1), "min")

espaco["max_features"] = 10
n = prod(espaco.values())
print("Com 4 hiperparâmetros:", n, "| tempo total:", n * 3, "s =", round(n * 3 / 3600, 2), "h")
