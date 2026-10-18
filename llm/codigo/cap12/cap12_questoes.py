"""Capítulo 12: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 10]
def params_lora(d, k, r):
    return r * (d + k)

def fracao_lora(d, k, r):
    return params_lora(d, k, r) / (d * k)

assert params_lora(4096, 4096, 8) == 65536
f = fracao_lora(4096, 4096, 8)
assert abs(f - 0.00390625) < 1e-12
print(params_lora(4096, 4096, 8), f"{f:.4%}")
for r in (1, 4, 16, 64):
    print(r, params_lora(4096, 4096, r), f"{fracao_lora(4096, 4096, r):.3%}")

# %% [Questão discursiva 11]
import numpy as np

def quantizar(w, bloco=64, bits=8):
    w = np.asarray(w, float); pad = (-len(w)) % bloco
    vb = np.concatenate([w, np.zeros(pad)]).reshape(-1, bloco)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(vb).max(1) / qmax
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.rint(vb / s[:, None]), -qmax, qmax).astype(np.int8)
    return q, s, pad

def desquantizar(q, s, pad):
    v = (q * s[:, None]).ravel()
    return v[:-pad] if pad else v

rng = np.random.default_rng(0)
w = rng.normal(0, 0.5, 1000)
q, s, pad = quantizar(w)
erro = np.abs(w - desquantizar(q, s, pad)).max()
print(f"erro máximo: {erro:.5f} | metade da maior escala: {s.max() / 2:.5f}")
assert erro <= s.max() / 2 + 1e-12

# %% [Questão discursiva 12]
import numpy as np
rng = np.random.default_rng(1)
d, k, r, alfa = 20, 30, 4, 8
W = rng.normal(size=(d, k)); A = rng.normal(size=(r, k)); B = rng.normal(size=(d, r))
x = rng.normal(size=(7, k))
saida_lora = x @ W.T + (alfa / r) * (x @ A.T) @ B.T
W_mesc = W + (alfa / r) * B @ A
saida_mesc = x @ W_mesc.T
dif = np.abs(saida_lora - saida_mesc).max()
print(f"diferença máxima: {dif:.2e} | posto de BA: {np.linalg.matrix_rank(B @ A)}")
assert dif < 1e-10 and np.linalg.matrix_rank(B @ A) == r

# %% [Questão discursiva 13]
def memoria_gb(N, modo, n_adapt=0):
    if modo == "completo":
        return 16 * N / 1e9
    if modo == "lora":
        return (2 * N + 16 * n_adapt) / 1e9
    if modo == "qlora":
        return (4.127 / 8 * N + 16 * n_adapt) / 1e9
    raise ValueError(modo)

N, na = 7e9, 40e6
valores = {m: memoria_gb(N, m, na) for m in ("completo", "lora", "qlora")}
print({k: round(v, 1) for k, v in valores.items()})
assert round(valores["completo"], 1) == 112.0 and round(valores["lora"], 1) == 14.6 and round(valores["qlora"], 1) == 4.3
