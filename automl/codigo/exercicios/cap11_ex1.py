# Capítulo 11 — solução de referência (EX 1)
import itertools

larguras, ativacoes, max_camadas = [16, 32, 64], ["relu", "tanh", "logistic"], 4
formula = len(ativacoes) * sum(len(larguras) ** L for L in range(1, max_camadas + 1))
enumerado = sum(1 for L in range(1, max_camadas + 1) for _ in itertools.product(larguras, repeat=L) for _ in ativacoes)
print("Pela fórmula :", formula)
print("Enumerando   :", enumerado)
print("Tempo total  : %.1f horas (2 min por arquitetura)" % (formula * 2 / 60))

larguras2 = larguras + [128]
novo = len(ativacoes) * sum(len(larguras2) ** L for L in range(1, max_camadas + 1))
print("Com largura 128 também:", novo, f"(x{novo / formula:.1f})")
