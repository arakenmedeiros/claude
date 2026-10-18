"""Capítulo 7: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 15]
import numpy as np

def perda_sft(perdas, mascara):
    perdas = np.asarray(perdas, float)
    mascara = np.asarray(mascara, float)
    if mascara.sum() == 0:
        raise ValueError("máscara sem tokens de resposta")
    return float((perdas * mascara).sum() / mascara.sum())

assert perda_sft([1, 2, 3, 4], [0, 0, 1, 1]) == 3.5
assert perda_sft([1, 2, 3, 4], [1, 1, 1, 1]) == 2.5
print(perda_sft([1, 2, 3, 4], [0, 0, 1, 1]))

# %% [Questão discursiva 16]
import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

def perda_dpo(lp_w, lp_l, lref_w, lref_l, beta):
    lp_w, lp_l, lref_w, lref_l = map(np.asarray, (lp_w, lp_l, lref_w, lref_l))
    margem = beta * ((lp_w - lref_w) - (lp_l - lref_l))
    return float(-np.log(sig(margem)).mean())

l1 = perda_dpo([-0.8], [-3.0], [-1.2], [-2.4], beta=0.1)
l2 = perda_dpo([-0.8], [-3.0], [-1.2], [-2.4], beta=1.0)
assert abs(l1 - 0.6444) < 1e-3 and abs(l2 - 0.3133) < 1e-3
assert abs(perda_dpo([-1.0], [-2.0], [-1.0], [-2.0], beta=1.0) - np.log(2)) < 1e-12
print(round(l1, 4), round(l2, 4))

# %% [Questão discursiva 17]
import numpy as np

def politica_otima(pref, r, beta):
    pref, r = np.asarray(pref, float), np.asarray(r, float)
    z = np.log(pref) + r / beta
    z = z - z.max()                    # estabilidade numérica
    q = np.exp(z)
    return q / q.sum()

pref = np.array([0.6, 0.3, 0.1]); r = np.array([0.0, 1.0, 2.0])
assert abs(politica_otima(pref, r, 1.0).sum() - 1) < 1e-12
assert np.allclose(politica_otima(pref, r, 1e6), pref, atol=1e-4)
assert politica_otima(pref, r, 0.01).argmax() == 2 and politica_otima(pref, r, 0.01)[2] > 0.999
assert np.allclose(politica_otima(pref, r, 1.0), [0.279, 0.379, 0.343], atol=1e-3)
print(politica_otima(pref, r, 1.0).round(3))
