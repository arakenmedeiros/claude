"""Capítulo 8: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 14]
import re
import unicodedata

def normalizar(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()

def verificar_citacoes(citacoes, documento):
    doc = normalizar(documento)
    ok = [normalizar(c) in doc for c in citacoes]
    return ok, (sum(ok) / len(ok) if ok else None)

doc = "O atraso na devolução gera multa de R$ 1,50 por dia e por exemplar."
ok, frac = verificar_citacoes(["multa de R$ 1,50  por dia", "multa de R$ 2,00 por dia"], doc)
assert ok == [True, False] and frac == 0.5
print(ok, frac)

# %% [Questão discursiva 15]
import json

def validar_saida(texto, campos_obrigatorios, enums):
    try:
        obj = json.loads(texto)
    except json.JSONDecodeError as e:
        return None, f"JSON inválido: {e.msg}"
    if not isinstance(obj, dict):
        return None, "esperado um objeto JSON"
    faltando = [c for c in campos_obrigatorios if c not in obj]
    if faltando:
        return None, f"campos ausentes: {faltando}"
    for campo, permitidos in enums.items():
        if campo in obj and obj[campo] not in permitidos:
            return None, f"valor inválido em {campo}: {obj[campo]!r}"
    return obj, None

enums = {"intencao": ["elogio", "cancelamento", "indefinido"]}
entradas = ['{"intencao": "elogio"}', '{"intencao": "xyz"}', 'texto solto {"intencao": "elogio"}', '{"outra": 1}']
res = [validar_saida(t, ["intencao"], enums) for t in entradas]
assert res[0][0] == {"intencao": "elogio"} and all(r[0] is None for r in res[1:])
print([r[1] for r in res])

# %% [Questão discursiva 16]
def com_tentativas(gerar, validar, max_tentativas=3):
    erro, historico = None, []
    for k in range(1, max_tentativas + 1):
        obj, erro = validar(gerar(erro))
        if obj is not None:
            return obj, k, historico
        historico.append(erro)
    return None, max_tentativas, historico

validar = lambda t: ({"ok": True}, None) if t == "bom" else (None, "formato inválido")
saidas = iter(["ruim", "bom"])
obj, k, h = com_tentativas(lambda e: next(saidas), validar)
assert obj == {"ok": True} and k == 2 and h == ["formato inválido"]
obj2, k2, h2 = com_tentativas(lambda e: "ruim", validar, max_tentativas=3)
assert obj2 is None and k2 == 3 and len(h2) == 3
print(k, k2)

# %% [Questão discursiva 17]
import numpy as np

def comparar_pareado(a, b, n_boot=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    medias = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
    return a.mean(), b.mean(), d.mean(), float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))

rng = np.random.default_rng(1)
n = 300
a = (rng.random(n) < 0.70).astype(float)
b = np.where(rng.random(n) < 0.5, a, (rng.random(n) < 0.80).astype(float))   # B parcialmente correlacionado com A
ma, mb, dm, lo, hi = comparar_pareado(a, b)
assert lo <= dm <= hi
print(round(ma, 3), round(mb, 3), round(dm, 3), (round(lo, 3), round(hi, 3)))
