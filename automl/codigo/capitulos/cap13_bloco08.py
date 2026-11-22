# Capítulo 13 — Explicabilidade, ética e limitações do AutoML
# Bloco 8. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
estrategias = {
    "Grade 5x5x5 com validação cruzada de 5 partes": 5 * 5 * 5 * 5,
    "Aleatória: 60 sorteios, validação cruzada de 5": 60 * 5,
    "Bayesiana (TPE): 60 tentativas, validação cruzada de 5": 60 * 5,
    "Aleatória: 60 sorteios, holdout (1 divisão)": 60 * 1,
    "Halving/Hyperband: ~30 treinos completos equivalentes": 30,
    "Modelo padrão, sem busca": 1,
}
minutos = 2
print(f"{'estratégia':<58s} {'treinos':>8s} {'horas*':>8s}")
for nome, k in estrategias.items(): print(f"{nome:<58s} {k:>8d} {k * minutos / 60:>8.1f}")
print(f"\n* supondo {minutos} minutos por treino; multiplique pelo consumo do seu hardware para estimar a energia.")
