# Capítulo 6 — solução de referência (EX 1)
def tabela_sh(n, r, eta):
    linhas, total = [], 0
    while True:
        custo = n * r
        linhas.append((n, r, custo)); total += custo
        if n <= 1: break
        n, r = max(1, n // eta), r * eta
    return linhas, total

for n, r, eta in [(64, 1, 4), (81, 1, 3)]:
    linhas, total = tabela_sh(n, r, eta)
    r_max = linhas[-1][1]
    print(f"n={n}, r={r}, η={eta}")
    for cand, rec, custo in linhas:
        print(f"   {cand:3d} candidatos x {rec:3d} épocas = {custo:4d}")
    print(f"   custo total = {total} | treinar todos até {r_max} épocas = {n * r_max} | economia = {100 * (1 - total / (n * r_max)):.0f}%")
