"""Capítulo 8: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import hashlib
import json

CATEGORIAS = ["segunda_via", "cancelamento", "suporte_tecnico", "alteracao_cadastro", "rastreio_pedido", "elogio", "reclamacao", "duvida_plano"]

def montar_prompt(mensagem, papel=True, categorias=True, formato_json=True, exemplos=()):
    partes = []
    if papel:
        partes.append("Você é um assistente de triagem de mensagens de atendimento ao cliente.")
    tarefa = "Classifique a mensagem do cliente"
    tarefa += f" em UMA das categorias: {', '.join(CATEGORIAS)}. Se nenhuma servir, use 'indefinido'." if categorias else "."
    partes.append(tarefa)
    for msg, cat in exemplos:
        partes.append(f"Exemplo:\nMensagem: <<<{msg}>>>\nResposta: {json.dumps({'intencao': cat}, ensure_ascii=False)}")
    if formato_json:
        partes.append('Responda apenas com JSON: {"intencao": "<categoria>", "evidencia": "<trecho literal da mensagem>"}')
    partes.append(f"Mensagem: <<<{mensagem}>>>")
    return "\n\n".join(partes)

p = montar_prompt("quero cancelar meu plano hoje", exemplos=[("a fatura não chegou", "segunda_via")])
print(p)
print("\nidentificador do prompt (hash curto):", hashlib.sha1(p.encode("utf-8")).hexdigest()[:10])
print("número de caracteres:", len(p))


# %% [bloco 2]
def montar_prompt_documentos(pergunta, documentos):
    corpo = "\n".join(f"<documento id=\"{i}\">\n{t}\n</documento>" for i, t in documentos)
    return (
        "Responda à pergunta usando APENAS os documentos abaixo. Cite o id do documento que sustenta a resposta. "
        "Se a resposta não estiver nos documentos, responda exatamente: não consta.\n\n"
        f"{corpo}\n\nPergunta: {pergunta}\nResposta:"
    )

from llmdidatico import DOCS
print(montar_prompt_documentos("Qual a multa por atraso?", [(DOCS[11][0], DOCS[11][3])]))


# %% [bloco 3]
import re
import unicodedata

def normalizar(s):
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"\s+", " ", s).strip()

def citacao_existe(citacao, documento):
    return normalizar(citacao) in normalizar(documento)

doc = DOCS[11][3]     # documento sobre multas da biblioteca
respostas = [
    {"resposta": "R$ 1,50 por dia e por exemplar", "citacao": "multa de R$ 1,50 por dia e por exemplar"},
    {"resposta": "R$ 2,00 por dia", "citacao": "multa de R$ 2,00 por dia"},                       # citação inventada
    {"resposta": "suspensão ao passar de R$ 30,00", "citacao": "Quando a dívida passa de R$ 30,00, o empréstimo é suspenso"},
]
for r in respostas:
    print(f"citação verificada={citacao_existe(r['citacao'], doc)!s:<5} | resposta: {r['resposta']}")

def conferir_calculo(atraso_dias, exemplares, multa_dia, total_informado, tol=0.01):
    esperado = atraso_dias * exemplares * multa_dia
    return abs(esperado - total_informado) <= tol, esperado

for total in (31.50, 21.00):
    ok, esp = conferir_calculo(atraso_dias=7, exemplares=3, multa_dia=1.50, total_informado=total)
    print(f"total informado R$ {total:.2f} | esperado R$ {esp:.2f} | correto={ok}")


# %% [bloco 4]
import json
from typing import Optional, Literal
from pydantic import BaseModel, ValidationError, Field
import pydantic
print("versão do pydantic:", pydantic.VERSION)

class Extracao(BaseModel):
    intencao: Literal["segunda_via", "cancelamento", "suporte_tecnico", "alteracao_cadastro", "rastreio_pedido", "elogio", "reclamacao", "duvida_plano", "indefinido"]
    evidencia: Optional[str] = Field(default=None, max_length=200)

def validar(texto):
    try:
        return Extracao.model_validate_json(texto), None
    except (ValidationError, ValueError) as e:
        return None, str(e).splitlines()[0]

saidas = [
    '{"intencao": "cancelamento", "evidencia": "quero cancelar"}',   # válida
    '{"intencao": "cancelar", "evidencia": "quero cancelar"}',        # valor fora da enumeração
    'Claro! Aqui está o JSON: {"intencao": "elogio"}',                # texto extra antes do JSON
    '{"intencao": "elogio"}',                                         # válida, sem evidência
]
for s in saidas:
    obj, erro = validar(s)
    print("OK  " if obj else "FALHA", "|", s[:55].ljust(55), "|", obj.intencao if obj else erro[:60])


# %% [bloco 5]
def com_novas_tentativas(gerar, max_tentativas=3):
    """gerar(erro_anterior) -> texto. Retorna (objeto, tentativas, histórico de erros)."""
    erros, erro = [], None
    for k in range(1, max_tentativas + 1):
        texto = gerar(erro)
        obj, erro = validar(texto)
        if obj:
            return obj, k, erros
        erros.append(erro)
    return None, max_tentativas, erros

respostas_simuladas = iter(['Claro! {"intencao": "elogio"}', '{"intencao": "elogio", "evidencia": "parabéns"}'])
obj, k, erros = com_novas_tentativas(lambda e: next(respostas_simuladas))
print("objeto:", obj, "| tentativas:", k, "| erros intermediários:", len(erros))


# %% [bloco 6]
import numpy as np
from llmdidatico import corpus, avaliacao as av
from llmdidatico.simulador import SimuladorLLM, INTENCOES

def interpretar(versao, texto_resposta):
    """Extrai o rótulo da resposta conforme o formato esperado de cada versão; devolve (rótulo, formato_valido)."""
    if versao == "v3":
        obj, _ = validar(texto_resposta)
        return (obj.intencao, True) if obj else (None, False)
    if versao == "v2":
        m = re.fullmatch(r"rotulo: (\w+)", texto_resposta.strip())
        return (m.group(1), True) if m else (None, False)
    achados = [c for c in INTENCOES if c.replace("_", " ") in texto_resposta]
    return (achados[0], False) if achados else (None, False)       # v1: formato livre, nunca "válido"

X, y = corpus.intencoes(60, seed=11)          # 480 mensagens de teste (semente diferente da usada nos exemplos)
resultados = {}
print(f"{'versão':<7} {'acerto':>7} {'IC95% (bootstrap)':>20} {'formato válido':>15}")
for v in ("v1", "v2", "v3"):
    sim = SimuladorLLM(v, semente=0)
    acertos, validos = [], []
    for texto, rotulo in zip(X, y):
        pred, ok = interpretar(v, sim.responder(texto))
        acertos.append(float(pred == rotulo)); validos.append(float(ok))
    m, lo, hi = av.bootstrap_ic(acertos, n=1000, seed=0)
    resultados[v] = np.array(acertos)
    print(f"{v:<7} {m:>7.3f} {f'[{lo:.3f}; {hi:.3f}]':>20} {np.mean(validos):>15.3f}")

dif = resultados["v3"] - resultados["v2"]
m, lo, hi = av.bootstrap_ic(dif, n=1000, seed=0)
print(f"\ndiferença pareada v3 - v2: {m:+.3f}  IC95% [{lo:+.3f}; {hi:+.3f}]")


# %% [bloco 7]
import datetime

registro = {
    "data": datetime.date.today().isoformat(),
    "prompt_id": hashlib.sha1(montar_prompt("x").encode()).hexdigest()[:10],
    "modelo": "SimuladorLLM (regras; não é um LLM)",
    "parametros": {"temperatura": None, "semente": 0},
    "teste": {"conjunto": "corpus.intencoes(60, seed=11)", "n": len(X)},
    "metricas": {v: round(float(resultados[v].mean()), 3) for v in resultados},
}
print(json.dumps(registro, ensure_ascii=False, indent=2))

