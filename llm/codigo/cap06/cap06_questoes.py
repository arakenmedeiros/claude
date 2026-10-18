"""Capítulo 6: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 15]
import numpy as np

def filtrar_top_k(p, k):
    p = np.asarray(p, float).copy()
    p[np.argsort(p)[::-1][k:]] = 0
    return p / p.sum()

def filtrar_top_p(p, top_p):
    p = np.asarray(p, float)
    ordem = np.argsort(p)[::-1]
    acum = np.cumsum(p[ordem])
    n = int(np.searchsorted(acum, top_p - 1e-12) + 1)
    q = np.zeros_like(p)
    q[ordem[:n]] = p[ordem[:n]]
    return q / q.sum()

p = [0.50, 0.30, 0.10, 0.05, 0.05]
assert np.allclose(filtrar_top_k(p, 2), [0.625, 0.375, 0, 0, 0])
assert np.allclose(filtrar_top_p(p, 0.9), [0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0, 0])
print(filtrar_top_k(p, 2).round(3), filtrar_top_p(p, 0.9).round(3))

# %% [Questão discursiva 16]
def montar_historico(mensagens, tokens, limite):
    usados, escolhidas = 0, []
    for msg, n in zip(reversed(mensagens), reversed(tokens)):
        if usados + n > limite:
            break
        escolhidas.append(msg)
        usados += n
    return list(reversed(escolhidas)), usados

msgs = ["m1", "m2", "m3", "m4", "m5"]
toks = [400, 300, 500, 200, 250]
h, u = montar_historico(msgs, toks, 1000)
assert h == ["m3", "m4", "m5"] and u == 950
h2, u2 = montar_historico(msgs, toks, 100)
assert h2 == [] and u2 == 0
print(h, u)

# %% [Questão discursiva 17]
import math

def memoria_kv_gb(camadas, T, n_kv, d_cab, bytes_num=2):
    return 2 * camadas * T * n_kv * d_cab * bytes_num / 1e9

def max_sequencias(gb_disponiveis, camadas, T, n_kv, d_cab, bytes_num=2):
    return math.floor(gb_disponiveis / memoria_kv_gb(camadas, T, n_kv, d_cab, bytes_num))

m = memoria_kv_gb(28, 8192, 8, 128)
assert abs(m - 0.939524096) < 1e-9
assert max_sequencias(15.0, 28, 8192, 8, 128) == 15
assert max_sequencias(16.0, 28, 8192, 8, 128) == 17
print(round(m, 3), max_sequencias(16.0, 28, 8192, 8, 128))
