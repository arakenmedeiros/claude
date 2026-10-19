"""Capítulo 18: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 6]
import math

def wilson(k, n, z=1.96):
    p = k / n; c = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / c
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / c
    return centro - meia, centro + meia

def veredito(lo, hi, lim):
    return "atende" if lo >= lim else "não atende" if hi < lim else "inconclusivo"

for k, n in [(14, 20), (70, 100)]:
    lo, hi = wilson(k, n)
    print(f"{k}/{n}: [{lo:.3f}; {hi:.3f}] -> {veredito(lo, hi, 0.70)}")
lo, hi = wilson(14, 20)
assert abs(lo - 0.4812) < 0.005 and abs(hi - 0.8541) < 0.005
lo, hi = wilson(70, 100)
assert abs(lo - 0.6038) < 0.005 and abs(hi - 0.7820) < 0.005
assert veredito(*wilson(14, 20), 0.70) == "inconclusivo" and veredito(*wilson(70, 100), 0.70) == "inconclusivo"

# %% [Questão discursiva 7]
import math

def n_necessario(p, margem, z=1.96):
    return math.ceil(z * z * p * (1 - p) / margem ** 2)

n80, n95 = n_necessario(0.80, 0.05), n_necessario(0.95, 0.05)
print(n80, n95)
assert n80 == 246 and n95 == 73

# %% [Questão discursiva 8]
conversas, frac_humano = 120_000, 0.45
c_agente, c_clf = 0.0005, 0.00002
sem = conversas * c_agente
com = conversas * c_clf + conversas * (1 - frac_humano) * c_agente
print(f"sem: {sem:.1f} | com: {com:.1f} | economia: {sem - com:.1f} ({(sem - com) / sem:.0%})")
assert abs(sem - 60.0) < 1e-9 and abs(com - 35.4) < 1e-9

# %% [Questão discursiva 9]
from math import comb

b, c = 3, 1
d = b + c
p = min(1.0, 2 * sum(comb(d, i) for i in range(0, min(b, c) + 1)) / 2 ** d)
print(f"diferença de taxa: {(b - c) / 40:+.3f} | McNemar exato p = {p:.3f}")
assert abs(p - 0.625) < 1e-9 and abs((b - c) / 40 - 0.05) < 1e-12

# %% [Questão discursiva 10]
def veredito(lo, hi, limiar, maior_melhor=True):
    if maior_melhor:
        return "atende" if lo >= limiar else "não atende" if hi < limiar else "inconclusivo"
    return "atende" if hi <= limiar else "não atende" if lo > limiar else "inconclusivo"

assert veredito(0.85, 0.95, 0.80) == "atende"
assert veredito(0.40, 0.70, 0.80) == "não atende"
assert veredito(0.70, 0.90, 0.80) == "inconclusivo"
assert veredito(0.00, 0.08, 0.10, maior_melhor=False) == "atende"
assert veredito(0.12, 0.30, 0.10, maior_melhor=False) == "não atende"
assert veredito(0.05, 0.20, 0.10, maior_melhor=False) == "inconclusivo"
print("ok")

# %% [Questão discursiva 11]
def vigente_em(meta, data):
    return meta["inicio"] <= data and (meta["fim"] is None or data < meta["fim"])

m = {"inicio": "2024-01-01", "fim": "2026-03-01"}
assert vigente_em(m, "2024-01-01")            # o dia do início vale
assert vigente_em(m, "2026-02-28")
assert not vigente_em(m, "2026-03-01")        # o dia do fim já não vale
assert not vigente_em(m, "2023-12-31")
assert vigente_em({"inicio": "2025-02-01", "fim": None}, "2030-01-01")
print("ok")

# %% [Questão discursiva 12]
import hashlib, json

class Trilha:
    def __init__(self):
        self.eventos = []
    def registrar(self, tipo, **dados):
        ant = self.eventos[-1]["hash"] if self.eventos else "0" * 16
        corpo = {"seq": len(self.eventos), "tipo": tipo, "dados": dados}
        h = hashlib.sha256((ant + json.dumps(corpo, sort_keys=True)).encode()).hexdigest()[:16]
        self.eventos.append({**corpo, "hash": h, "ant": ant})
    def verificar(self):
        ant = "0" * 16
        for e in self.eventos:
            corpo = {k: e[k] for k in ("seq", "tipo", "dados")}
            h = hashlib.sha256((ant + json.dumps(corpo, sort_keys=True)).encode()).hexdigest()[:16]
            if e["ant"] != ant or e["hash"] != h:
                return False
            ant = h
        return True

t = Trilha()
for i in range(3):
    t.registrar("consulta", n=i)
assert t.verificar()
t.eventos[1]["dados"]["n"] = 99
assert not t.verificar()
print("ok")

# %% [Questão discursiva 13]
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class Pedido(BaseModel):
    model_config = ConfigDict(extra="forbid")
    codigo: str = Field(pattern=r"^\d{4}$")
    quantidade: int = Field(ge=1, le=10)
    entrega: Literal["retirada", "correios", "motoboy"]

ok = Pedido.model_validate_json('{"codigo": "1042", "quantidade": 3, "entrega": "correios"}')
assert ok.quantidade == 3
for ruim in ['{"codigo": "42", "quantidade": 3, "entrega": "correios"}',
             '{"codigo": "1042", "quantidade": 11, "entrega": "correios"}',
             '{"codigo": "1042", "quantidade": 3, "entrega": "drone"}',
             '{"codigo": "1042", "quantidade": 3, "entrega": "correios", "extra": 1}']:
    try:
        Pedido.model_validate_json(ruim)
        raise AssertionError("deveria ter sido rejeitado")
    except ValidationError:
        pass
print("ok")
