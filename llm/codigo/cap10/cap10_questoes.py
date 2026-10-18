"""Capítulo 10: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 10]
import numpy as np

def knn_cosseno(q, D, k):
    q = np.asarray(q, float); D = np.asarray(D, float)
    nq = np.linalg.norm(q); nD = np.linalg.norm(D, axis=1)
    cos = (D @ q) / np.where(nD * nq == 0, 1.0, nD * nq)
    return list(np.argsort(-cos, kind="stable")[:k])

D = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [5.0, 5.0]])
q = np.array([0.8, 0.6])
res = knn_cosseno(q, D, 2)
assert res == [3, 1]
print(res)

# %% [Questão discursiva 11]
def rrf(rankings, k=60):
    pont = {}
    for rk in rankings:
        for pos, d in enumerate(rk, 1):
            pont[d] = pont.get(d, 0.0) + 1.0 / (k + pos)
    return sorted(pont, key=lambda d: (-pont[d], d)), pont

A = ["D1", "D2", "D3", "D4"]
B = ["D3", "D1", "D5", "D2"]
ordem, pont = rrf([A, B])
assert ordem == ["D1", "D3", "D2", "D5", "D4"]
print(ordem, round(pont["D1"], 5))

# %% [Questão discursiva 12]
def dividir(texto, tamanho, sobreposicao=0):
    if sobreposicao >= tamanho:
        raise ValueError("a sobreposição deve ser menor que o tamanho")
    palavras = texto.split()
    passo = tamanho - sobreposicao
    janelas = []
    for i in range(0, len(palavras), passo):
        janelas.append(" ".join(palavras[i:i + tamanho]))
        if i + tamanho >= len(palavras):
            break
    return janelas

texto = "a b c d e f g h i j"
j = dividir(texto, 4, 2)
assert j == ["a b c d", "c d e f", "e f g h", "g h i j"]
assert dividir(texto, 4, 0) == ["a b c d", "e f g h", "i j"]
print(j)

# %% [Questão discursiva 13]
import hashlib

def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def a_reindexar(registro, documentos):
    novos = [d for d in documentos if d not in registro]
    alterados = [d for d, t in documentos.items() if d in registro and registro[d] != sha(t)]
    removidos = [d for d in registro if d not in documentos]
    return novos, alterados, removidos

antigo = {"D1": sha("texto um"), "D2": sha("texto dois"), "D3": sha("texto tres")}
atual = {"D1": "texto um", "D2": "texto dois revisado", "D4": "texto quatro"}
n, a, r = a_reindexar(antigo, atual)
assert n == ["D4"] and a == ["D2"] and r == ["D3"]
print(n, a, r)
