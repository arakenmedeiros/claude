# Capítulo 11 — Busca de arquitetura neural (NAS)
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import json
import numpy as np
import os; os.makedirs("resultados", exist_ok=True)

T = json.load(open("resultados/f11_cache.json"))
P = np.array([r["params"] for r in T]); A = np.array([r["v40"] for r in T])
# uma rede está na fronteira se ninguém tem (acurácia >= e parâmetros <=) com ao menos uma desigualdade estrita
fronteira = [i for i in range(len(T)) if not any((A[j] >= A[i] and P[j] <= P[i]) and (A[j] > A[i] or P[j] < P[i]) for j in range(len(T)))]
fronteira.sort(key=lambda i: P[i])
print("Fronteira de Pareto (menos parâmetros → mais acurácia):")
for i in fronteira:
    print(f"  {str(tuple(T[i]['arq'][0])):<16s} {T[i]['arq'][1]:<5s} parâmetros = {P[i]:>6d}   validação = {A[i]:.3f}")
melhor = int(A.argmax())
bom = [i for i in fronteira if A[i] >= A[melhor] - 0.01]
j = min(bom, key=lambda i: P[i])
print(f"\nMais acurada: {P[melhor]} parâmetros ({A[melhor]:.3f}). Menor dentro de 1 ponto da melhor: {P[j]} parâmetros ({A[j]:.3f}).")
print(f"Redução de tamanho: {100 * (1 - P[j] / P[melhor]):.0f}%")
json.dump({"fronteira": fronteira}, open("resultados/f11_pareto.json", "w"))
