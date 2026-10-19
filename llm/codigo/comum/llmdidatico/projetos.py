"""projetos.py: componentes reutilizáveis dos SEIS projetos integradores do Capítulo 18.

AVISO (leia): nenhum componente aqui chama um LLM. Onde o livro diria 'o modelo', usamos os SIMULADORES do pacote
(`rag.GeradorSimulado`, `agente.ModeloSimulado`, `seguranca.AssistenteSimulado`) ou regras escritas por nós. As taxas medidas descrevem
esses simuladores e o desenho do sistema ao redor deles, e NÃO o comportamento de modelos reais. Dados: sintéticos e fictícios.
Preços e latências usados nas estimativas de custo são HIPOTÉTICOS (constantes declaradas abaixo), não medidas nem tabelas de fornecedor.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import modelos, rag, agente, pii
from .aval_llm import ic_wilson, mcnemar, bootstrap_pareado

# ======================================================================================= utilidades estatísticas e de relato
def prop_ic(valores):
    """(proporção, limite inferior, limite superior, n) com intervalo de Wilson; valores = sequência de 0/1 (None é ignorado)."""
    v = [int(x) for x in valores if x is not None]
    if not v:
        return (float("nan"),) * 3 + (0,)
    k, n = sum(v), len(v)
    lo, hi = ic_wilson(k, n)
    return k / n, lo, hi, n


def fmt_ic(t):
    p, lo, hi, n = t
    return f"{p:.3f} [{lo:.3f}; {hi:.3f}] (n={n})"


def veredito(lo, hi, limiar, maior_melhor=True):
    """Compara o INTERVALO com o critério de aceitação: 'atende' (todo o IC do lado bom), 'não atende' (todo o IC do lado ruim) ou
    'inconclusivo' (o IC atravessa o limiar: faltam dados)."""
    if maior_melhor:
        return "atende" if lo >= limiar else "não atende" if hi < limiar else "inconclusivo"
    return "atende" if hi <= limiar else "não atende" if lo > limiar else "inconclusivo"


def dif_prop_ic(a, b, n=2000, seed=0):
    """Diferença de proporções (a - b) entre dois grupos INDEPENDENTES, com IC95% bootstrap (reamostra cada grupo)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def sem_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


# ======================================================================================= PROJETO 1 e base dos demais: RAG com abstenção
def montar_rag(docs, meta_extra=None, limiar=0.5, validar=True, k=3, metodo="hibrido", filtro=None, pre_filtro=True,
               modo_fragmento="documento", vulneravel=False, ciente=True, semente=0):
    """Devolve (pipeline, indice, idf). Junta fragmentação, índice híbrido, gerador SIMULADO e validação de citação."""
    frs = rag.fragmentar(docs, modo=modo_fragmento, meta_extra=meta_extra)
    ix = rag.Indice(frs)
    idf = rag.idf_do_corpus(docs)
    ger = rag.GeradorSimulado(idf, limiar=limiar, vulneravel=vulneravel, ciente_de_delimitacao=ciente, semente=semente)
    pipe = rag.PipelineRAG(ix, ger, metodo=metodo, k=k, filtro=filtro, pre_filtro=pre_filtro, validar=validar)
    return pipe, ix, idf


def metricas_rag(linhas):
    """Métricas do Projeto 1 a partir das linhas de `rag.avaliar`. Cada uma: (p, lo, hi, n) de Wilson."""
    com = [l for l in linhas if l["rel"]]
    sem = [l for l in linhas if not l["rel"]]
    return {
        "recuperacao": prop_ic([l["acerto_rec"] for l in com]),
        "resposta_correta": prop_ic([l["rotulo"] == "ok" for l in com]),
        "abstencao_correta": prop_ic([l["rotulo"] == "ok_abstencao" for l in sem]),
        "alucinacao": prop_ic([l["rotulo"] == "alucinacao_sem_resposta" for l in sem]),
        "fidelidade": prop_ic([l["fiel"] for l in linhas]),
        "abstencao_indevida": prop_ic([l["rotulo"] == "abstencao_indevida" for l in com]),
    }


# ======================================================================================= PROJETO 2: triagem informativa de saúde
_URG = [r"dor forte no peito", r"falta de ar", r"desmai", r"nao acorda", r"sangue e nao para", r"sangramento", r"engasg", r"ficando roxo",
        r"\bavc\b", r"boca (ta|esta) torta", r"me machucar", r"nao aguento mais", r"convuls", r"socorro", r"quero morrer"]
_CLIN = [r"\bdose\b", r"posso tomar", r"que remedio (eu )?(tomo|dou|devo)", r"o que (eu )?tenho", r"o que dou", r"e grave", r"tem cura",
         r"significa que", r"parar de tomar", r"estou doente", r"junto com o meu", r"dor de cabeca", r"\bfebre\b", r"\btosse\b", r"mancha"]
MSG_URGENCIA = ("Isto pode ser uma emergência. Procure imediatamente a UPA mais próxima ou acione o serviço de atendimento móvel de urgência "
                "(telefone 192 no Brasil; confirme o número vigente). Este assistente não faz atendimento de saúde.")
MSG_CLINICA = ("Não posso orientar sobre sintomas, diagnóstico, medicamentos ou condutas: isso exige um profissional de saúde. "
               "Agende uma consulta na UBS (aplicativo Saúde Aurora ou recepção) ou procure a unidade; se houver piora ou risco, procure a urgência.")
RODAPE_SERVICO = " (Informação de serviço; não substitui orientação de profissional de saúde.)"


def rota_saude(pergunta):
    """Roteador por regras: 'urgencia' tem prioridade sobre 'encaminhar_profissional' e sobre 'servico'. Erra por omissão (paráfrases não previstas)."""
    t = sem_acentos(pergunta)
    if any(re.search(p, t) for p in _URG):
        return "urgencia"
    if any(re.search(p, t) for p in _CLIN):
        return "encaminhar_profissional"
    return "servico"


class TriagemSaude:
    """Triagem INFORMATIVA: (1) mascara dados pessoais, (2) roteia urgência/clínica por regras (sem consultar RAG nem modelo),
    (3) responde só perguntas de serviço, com citação, abstenção e rodapé. Registra apenas texto MASCARADO na auditoria."""

    def __init__(self, docs, limiar=0.5, auditoria=None, usar_guardrails=True):
        self.pipe, self.ix, self.idf = montar_rag(docs, limiar=limiar, validar=True)
        self.aud = auditoria or agente.Auditoria()
        self.guard = usar_guardrails

    def atender(self, pergunta):
        achados = pii.detectar(pergunta)
        mascarada = pii.mascarar(pergunta, achados)
        rota = rota_saude(pergunta) if self.guard else "servico"
        if rota == "urgencia":
            resp, rec = MSG_URGENCIA, []
        elif rota == "encaminhar_profissional":
            resp, rec = MSG_CLINICA, []
        else:
            r = self.pipe.responder(mascarada)
            resp, rec = r["resposta"], r["incluidos"]
            if rag.eh_abstencao(resp):
                rota = "abstencao"
                resp = "Não encontrei essa informação nos documentos de serviços da rede. Procure a recepção da UBS ou a ouvidoria."
            else:
                resp += RODAPE_SERVICO
        self.aud.registrar("atendimento", pergunta=mascarada, rota=rota, docs=rec, pii=[a[2] for a in achados])
        return {"rota": rota, "resposta": resp, "docs": rec, "pii": [a[2] for a in achados], "pergunta_mascarada": mascarada}


# ======================================================================================= PROJETO 3: normas com vigência, perfis e trilha
def vigente_em(meta, data):
    """Intervalo [inicio, fim): datas ISO comparam como texto."""
    return meta["inicio"] <= data and (meta["fim"] is None or data < meta["fim"])


def perfil_ok(meta, perfil, nivel):
    return nivel[perfil] >= nivel[meta["perfil"]]


def pseudonimo(id_usuario, sal="projeto3"):
    return "u-" + hashlib.sha256((sal + id_usuario).encode()).hexdigest()[:8]


class ConsultaNormas:
    """modo: 'ingenuo' (sem vigência nem perfil), 'pos_filtro' (filtra DEPOIS de escolher os k melhores) ou 'pre_filtro' (correto)."""

    def __init__(self, normas, meta, nivel, modo="pre_filtro", limiar=0.5, auditoria=None, k=3):
        self.meta, self.nivel, self.modo, self.k = meta, nivel, modo, k
        extra = {i: dict(m) for i, m in meta.items()}
        self.pipe, self.ix, self.idf = montar_rag(normas, meta_extra=extra, limiar=limiar, validar=True, k=k)
        self.aud = auditoria or agente.Auditoria()

    def consultar(self, pergunta, data, perfil, id_usuario="anonimo"):
        if self.modo == "ingenuo":
            filtro, pre = None, True
        else:
            filtro = lambda m: vigente_em(m, data) and perfil_ok(m, perfil, self.nivel)
            pre = self.modo == "pre_filtro"
        res = self.ix.buscar(pergunta, "hibrido", self.k, filtro, pre, vigentes=False)
        frs = [self.ix.fr[i] for i, _ in res]
        ctx, incl, _ = rag.montar_contexto(frs)
        incl_frs = [f for f in frs if f.doc_id in incl]
        resp = self.pipe.ger.gerar(rag.montar_prompt(pergunta, ctx))
        if not rag.verificar_resposta(resp, incl_frs)["fiel"]:
            resp = "Não consta nos documentos fornecidos."
        if rag.eh_abstencao(resp):
            resp = "Não consta nos documentos disponíveis para o seu perfil na data consultada."
        else:                                   # acrescenta a identificação da norma usada (citação humana)
            did = re.findall(r"\[(D\d+)\]", resp)[0]
            resp += f" (norma {self.meta[did]['numero']}, vigente em {data})"
        self.aud.registrar("consulta", usuario=pseudonimo(id_usuario), perfil=perfil, data_ref=data, pergunta=agente.mascarar_pii(pergunta),
                           docs=[f.doc_id for f in frs], resposta_sha=hashlib.sha256(resp.encode()).hexdigest()[:12])
        return {"resposta": resp, "docs": [f.doc_id for f in frs], "incluidos": incl, "ctx": incl_frs}


def avaliar_normas(sistema, consultas):
    """Por caso: rótulo do RAG, vazamento (documento acima do perfil no contexto) e versão errada (documento fora da vigência)."""
    out = []
    for q in consultas:
        pergunta, data, perfil, rel, trecho, tipo = q
        r = sistema.consultar(pergunta, data, perfil, "teste")
        rot, _ = rag.classificar_caso((pergunta, rel, trecho, tipo), r["resposta"], r["incluidos"], r["ctx"])
        vaz = any(not perfil_ok(sistema.meta[d], perfil, sistema.nivel) for d in r["docs"])
        velha = any(not vigente_em(sistema.meta[d], data) for d in r["docs"])
        ok = rot in ("ok", "ok_abstencao")
        out.append(dict(tipo=tipo, rotulo=rot, ok=ok, vazamento=vaz, fora_da_vigencia=velha, resposta=r["resposta"], pergunta=pergunta, data=data, perfil=perfil))
    return out
