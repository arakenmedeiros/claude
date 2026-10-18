"""Cap. 6: rastreio do passo de cada variante em uma variável (E(w) = 0.5 (w-1)^2 ... e uma função com cruzamento)."""
import numpy as np
def rastro(var, E, g, w0, delta0=0.5, eta_mais=1.2, eta_menos=0.5, dmin=1e-6, dmax=50.0, n=9):
    w, delta, g_ant, dw_ant, E_ant = w0, delta0, 0.0, 0.0, np.inf; linhas = []
    for t in range(1, n + 1):
        gt, Et = g(w), E(w); prod = g_ant * gt; regra = "="
        if prod > 0: delta = min(delta * eta_mais, dmax); regra = ">0"
        elif prod < 0: delta = max(delta * eta_menos, dmin); regra = "<0"
        if var == "rprop+":
            dw = -dw_ant if prod < 0 else -np.sign(gt) * delta
            if prod < 0: gt = 0.0
        elif var == "rprop-": dw = -np.sign(gt) * delta
        elif var == "irprop-":
            if prod < 0: gt = 0.0
            dw = -np.sign(gt) * delta
        elif var == "irprop+":
            if prod < 0:
                dw = -dw_ant if Et > E_ant else 0.0; gt = 0.0
            else: dw = -np.sign(gt) * delta
        linhas.append((t, w, g(w), E(w), regra, delta, dw)); w = w + dw; g_ant, dw_ant, E_ant = gt, dw, Et
    return linhas, w
E = lambda w: 0.5 * (w - 1.0) ** 2 + 0.0; g = lambda w: w - 1.0
for var in ("rprop-", "rprop+", "irprop-", "irprop+"):
    L, wf = rastro(var, E, g, 3.0, n=10)
    print(var); [print("  t=%2d w=%+.4f g=%+.4f E=%.4f prod %s delta=%.4f dw=%+.4f" % l) for l in L]; print("  w final %.4f" % wf)
