"""Capítulo 9: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 13]
def esperas_recuo(n, t0=1.0, teto=30.0):
    esperas = [min(teto, t0 * 2 ** k) for k in range(n)]
    return esperas, sum(esperas)

esperas, total = esperas_recuo(5, t0=2, teto=20)
assert esperas == [2, 4, 8, 16, 20] and total == 50
esperas6, total6 = esperas_recuo(6, t0=1, teto=30)
assert esperas6[-1] == 30 and total6 == 61
print(esperas, total, total6)

# %% [Questão discursiva 14]
def classificar_status(codigo):
    if codigo in (408, 429) or 500 <= codigo <= 599:
        return "transitorio"
    return "permanente"

def deve_repetir(codigo, tentativa, max_tentativas):
    return classificar_status(codigo) == "transitorio" and tentativa < max_tentativas

casos = [429, 503, 408, 401, 404, 400]
classes = [classificar_status(c) for c in casos]
assert classes == ["transitorio", "transitorio", "transitorio", "permanente", "permanente", "permanente"]
assert deve_repetir(429, 1, 3) and not deve_repetir(429, 3, 3) and not deve_repetir(401, 1, 3)
print(dict(zip(casos, classes)))

# %% [Questão discursiva 15]
import re

PADROES = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("TELEFONE", re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b")),
]

def mascarar(texto):
    achados = {}
    for nome, rx in PADROES:
        def troca(m, nome=nome):
            achados[nome] = achados.get(nome, 0) + 1
            return f"[{nome}]"
        texto = rx.sub(troca, texto)
    return texto, achados

entrada = "Contato: ana@exemplo.com, CPF 111.222.333-44, tel (61) 98888-7777."
saida, achados = mascarar(entrada)
assert "@" not in saida and "111.222" not in saida and "98888" not in saida
assert achados == {"EMAIL": 1, "CPF": 1, "TELEFONE": 1}
assert mascarar(saida) == (saida, {})
neutro = "A multa é de R$ 1,50 por dia."
assert mascarar(neutro) == (neutro, {})
print(saida, achados)

# %% [Questão discursiva 16]
def memoria_pesos_gb(n_params, bits):
    return n_params * bits / 8 / 1e9

def memoria_kv_gb(camadas, T, n_kv, d_cab, bytes_num=2):
    return 2 * camadas * T * n_kv * d_cab * bytes_num / 1e9

def max_usuarios(n_params, bits, camadas, T, n_kv, d_cab, gb, margem=0.15):
    livre = gb * (1 - margem) - memoria_pesos_gb(n_params, bits)
    if livre <= 0:
        return 0
    return int(livre // memoria_kv_gb(camadas, T, n_kv, d_cab))

u24 = max_usuarios(7e9, 4, 32, 4096, 8, 128, 24)
u16 = max_usuarios(7e9, 4, 32, 4096, 8, 128, 16)
assert u24 == 31 and u16 == 18
assert max_usuarios(70e9, 16, 80, 4096, 8, 128, 24) == 0
print(u24, u16)

# %% [Questão discursiva 17]
import json
import statistics

def resumir_log(linhas):
    regs = [json.loads(l) for l in linhas if l.strip()]
    if not regs:
        return {"chamadas": 0, "taxa_sucesso": None, "latencia_mediana_s": None, "tokens_saida": 0}
    return {
        "chamadas": len(regs),
        "taxa_sucesso": sum(r["status"] == "ok" for r in regs) / len(regs),
        "latencia_mediana_s": statistics.median(r["latencia_s"] for r in regs),
        "tokens_saida": sum(r["tokens_saida"] for r in regs),
    }

linhas = [json.dumps({"status": s, "latencia_s": l, "tokens_saida": t})
          for s, l, t in [("ok", 0.4, 20), ("ok", 0.6, 30), ("ok", 0.5, 25), ("erro", 2.0, 0), ("ok", 0.3, 15)]]
linhas.append("")
r = resumir_log(linhas)
assert r["chamadas"] == 5 and r["taxa_sucesso"] == 0.8
assert r["latencia_mediana_s"] == 0.5 and r["tokens_saida"] == 90
assert resumir_log([])["chamadas"] == 0
print(r)
