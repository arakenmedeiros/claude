"""Capítulo 9: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 9.1: Recuo exponencial, jitter e sincronização de clientes]
import random
import numpy as np

class ErroTransitorio(Exception):
    pass

def com_recuo(funcao, max_tentativas=5, t0=1.0, teto=30.0, jitter=0.25, dormir=lambda s: None, rng=None):
    rng = rng or random.Random(0)
    esperas = []
    for k in range(max_tentativas):
        try:
            return funcao(), k + 1, esperas
        except ErroTransitorio:
            if k == max_tentativas - 1:
                raise
            e = min(teto, t0 * 2 ** k) * (1 + rng.uniform(0, jitter))
            esperas.append(round(e, 2)); dormir(e)

# (a) esperas sem jitter, com dormir injetável
registro = []
falhas = [3]
def servico():
    if falhas[0] > 0:
        falhas[0] -= 1
        raise ErroTransitorio("429")
    return "ok"
res, tent, esp = com_recuo(servico, jitter=0.0, dormir=registro.append)
print("esperas pedidas a dormir:", registro, "| tentativas:", tent)

# (b) 1000 clientes falham juntos no instante 0: quando fazem a 4ª tentativa?
def instantes(n_clientes, falhas, jitter, semente=0):
    rng = random.Random(semente)
    out = []
    for _ in range(n_clientes):
        t = 0.0
        for k in range(falhas):
            t += min(30.0, 1.0 * 2 ** k) * (1 + rng.uniform(0, jitter))
        out.append(t)
    return np.array(out)

print(f"{'jitter':>7} {'média':>7} {'desvio':>7} {'mín':>6} {'máx':>6} {'pico (clientes por janela de 0,5 s)':>38}")
for j in (0.0, 0.25, 1.0):
    t = instantes(1000, 3, j)
    contagem, _ = np.histogram(t, bins=np.arange(0, t.max() + 1.0, 0.5))
    print(f"{j:>7.2f} {t.mean():>7.2f} {t.std():>7.2f} {t.min():>6.2f} {t.max():>6.2f} {contagem.max():>38d}")


# %% [Exercício 9.2: Mascaramento de dados pessoais: cobertura e falsos positivos]
import random
import re
ANTIGOS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("TELEFONE", re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b")),
]
NOVOS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CPF", re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")),
    ("TELEFONE", re.compile(r"(?<![\d-])(?:\(?\d{2}\)?\s?)?9?\d{4}-\d{4}(?![\d-])")),
]
def mascarar(texto, padroes):
    for nome, rx in padroes:
        texto = rx.sub(f"[{nome}]", texto)
    return texto
rng = random.Random(7)

def d(n): return "".join(str(rng.randrange(10)) for _ in range(n))
fmts = {
 "cpf com pontos": lambda: f"{d(3)}.{d(3)}.{d(3)}-{d(2)}",
 "cpf sem pontos": lambda: d(11),
 "cpf misto": lambda: f"{d(3)}.{d(3)}.{d(3)}{d(2)}",
 "tel (DD) 9XXXX-XXXX": lambda: f"({d(2)}) 9{d(4)}-{d(4)}",
 "tel (DD)9XXXX-XXXX": lambda: f"({d(2)})9{d(4)}-{d(4)}",
 "tel DD 9XXXX-XXXX": lambda: f"{d(2)} 9{d(4)}-{d(4)}",
 "tel sem DDD": lambda: f"9{d(4)}-{d(4)}",
 "tel fixo (DD) XXXX-XXXX": lambda: f"({d(2)}) {d(4)}-{d(4)}",
 "email": lambda: f"{rng.choice(['ana','joao.s','m_lima'])}{d(2)}@{rng.choice(['exemplo.com','mail.org.br'])}",
}
print(f"{'formato':<26}{'antigo':>7}{'novo':>6}   (mascarados de 50)")
for nome, f in fmts.items():
    a=n=0
    for _ in range(50):
        v=f(); t=f"Meu dado é {v}, por favor confirme."
        a+= v not in mascarar(t,ANTIGOS); n+= v not in mascarar(t,NOVOS)
    print(f"{nome:<26}{a:>7}{n:>6}")
neutros=["Período de 2020-2023 para a matrícula.","A multa é de R$ 1.500,00 por dia.","O protocolo 2024-0312 foi aberto em 12/03/2024.","Resolução 12/2023, art. 5º, prazo de 30 dias.","O documento D12 trata do auxílio de R$ 300,00.","Código do curso 1234-5678 no catálogo.", "Ligue para 0800-1234 (central)."]
fp_antigo = sum(mascarar(t, ANTIGOS) != t for t in neutros)
fp_novo = [mascarar(t, NOVOS) for t in neutros if mascarar(t, NOVOS) != t]
print(f"\ntextos neutros alterados (de {len(neutros)}): antigo = {fp_antigo}; novo = {len(fp_novo)}")
for t in fp_novo:
    print("  falso positivo:", t)


# %% [Exercício 9.3: Quantos usuários cabem na GPU?]
import math

def pesos_gb(N, bits):
    return N * bits / 8 / 1e9

def kv_usuario_gb(camadas, T, n_kv, d_cab, bytes_num=2):
    return 2 * camadas * T * n_kv * d_cab * bytes_num / 1e9

def max_usuarios(N, bits, camadas, T, n_kv, d_cab, gb, margem=0.15):
    livre = gb * (1 - margem) - pesos_gb(N, bits)
    return 0 if livre <= 0 else int(livre // kv_usuario_gb(camadas, T, n_kv, d_cab))

MODELOS = {"7 bi": (7e9, 32), "13 bi": (13e9, 40), "70 bi": (70e9, 80)}   # hipotéticos: (parâmetros, camadas)
for gb in (24, 80):
    print(f"\nGPU de {gb} GB (margem de 15%), 8 cabeças KV de 128 dimensões; usuários simultâneos:")
    print(f"{'modelo':<7} {'bits':>4} " + " ".join(f"{'T=' + str(T):>7}" for T in (2048, 4096, 8192)))
    for nome, (N, L) in MODELOS.items():
        for bits in (16, 8, 4):
            print(f"{nome:<7} {bits:>4} " + " ".join(f"{max_usuarios(N, bits, L, T, 8, 128, gb):>7d}" for T in (2048, 4096, 8192)))

N, L = MODELOS["7 bi"]
for n_kv in (8, 32):
    kv = kv_usuario_gb(L, 4096, n_kv, 128)
    n_cruza = math.ceil(pesos_gb(N, 4) / kv)
    print(f"\n7 bi, 4 bits, T=4096, {n_kv} cabeças KV: cache de {kv:.2f} GB por usuário; o cache supera os pesos ({pesos_gb(N, 4):.1f} GB) a partir de {n_cruza} usuários; cabem {max_usuarios(N, 4, L, 4096, n_kv, 128, 24)} usuários em 24 GB")


# %% [Projeto: Aplicação em camadas com provedor simulado]
import hashlib, itertools, json, os, random, re, tempfile, time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Protocol
from pydantic import BaseModel, Field, ValidationError, field_validator
from llmdidatico import CONSULTAS, DOCS, modelos

# ---------- camada de domínio: tipos e interface ----------
@dataclass
class Mensagem:
    papel: str
    conteudo: str

@dataclass
class Parametros:
    temperatura: float = 0.0
    max_tokens: int = 200
    tempo_limite_s: float = 30.0

@dataclass
class Resposta:
    texto: str
    tokens_entrada: int = 0
    tokens_saida: int = 0
    modelo: str = ""

class ProvedorLLM(Protocol):
    nome: str
    def gerar(self, mensagens: list[Mensagem], params: Parametros) -> Resposta: ...

class ErroTransitorio(Exception):
    pass

class ErroPermanente(Exception):
    pass

# ---------- camada de provedores (todos simulados, sem rede) ----------
class ProvedorSimulado:
    nome = "simulado"
    def gerar(self, mensagens, params):
        usuario = next(m.conteudo for m in reversed(mensagens) if m.papel == "user")
        ctx, _, pergunta = usuario.partition("\nPERGUNTA: ")
        frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", ctx.replace("CONTEXTO:", "")) if f.strip()]
        texto = "não consta"
        if frases:
            q = set(modelos.tokenizar_sem_stop(pergunta))
            pontos = [len(q & set(modelos.tokenizar_sem_stop(f))) for f in frases]
            if max(pontos) > 0:
                texto = frases[pontos.index(max(pontos))]
        return Resposta(texto, len(usuario.split()), len(texto.split()), "simulado-v1")

class ProvedorPrimeiraFrase:
    nome = "primeira-frase"
    def gerar(self, mensagens, params):
        usuario = next(m.conteudo for m in reversed(mensagens) if m.papel == "user")
        ctx = usuario.partition("\nPERGUNTA: ")[0].replace("CONTEXTO:", "").strip()
        texto = re.split(r"(?<=[.!?])\s+", ctx)[0] if ctx else "não consta"
        return Resposta(texto, len(usuario.split()), len(texto.split()), "primeira-frase-v1")

class ProvedorRoteirizado:
    """Envolve outro provedor e falha conforme um roteiro cíclico de eventos por tentativa."""
    def __init__(self, base, roteiro):
        self.base, self.nome, self._ev, self.tentativas = base, base.nome + "+roteiro", itertools.cycle(roteiro), 0
    def gerar(self, mensagens, params):
        self.tentativas += 1
        ev = next(self._ev)
        if ev == "ok":
            return self.base.gerar(mensagens, params)
        if ev in ("429", "503", "timeout"):
            raise ErroTransitorio(ev)
        raise ErroPermanente(ev)

class ProvedorComVazamento:
    nome = "com-vazamento"
    def gerar(self, mensagens, params):
        t = "Ligue para (61) 98888-7777 para saber mais."
        return Resposta(t, 10, len(t.split()), "vazamento-v0")

class ProvedorEspiao:
    """Guarda as mensagens recebidas, para testar o que realmente sai da aplicação."""
    nome = "espiao"
    def __init__(self):
        self.vistas = []
    def gerar(self, mensagens, params):
        self.vistas.append([m.conteudo for m in mensagens])
        return Resposta("não consta", 1, 2, "espiao-v0")

# ---------- camada de segurança: mascaramento ----------
PADROES = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
           ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
           ("TELEFONE", re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b"))]

def mascarar(texto):
    achados = {}
    for nome, rx in PADROES:
        def troca(m, nome=nome):
            achados[nome] = achados.get(nome, 0) + 1
            return f"[{nome}]"
        texto = rx.sub(troca, texto)
    return texto, achados

# ---------- camada de validação: pydantic ----------
class RespostaAtendimento(BaseModel):
    resposta: str = Field(min_length=1, max_length=500)
    fonte: Optional[str] = None
    @field_validator("resposta")
    @classmethod
    def sem_dados_pessoais(cls, v):
        if mascarar(v)[1]:
            raise ValueError("a resposta contém dados pessoais")
        return v

# ---------- camada de robustez: recuo exponencial ----------
def com_recuo(funcao, max_tentativas=4, t0=1.0, teto=30.0, jitter=0.25, dormir=time.sleep, rng=None):
    rng = rng or random.Random(0)
    esperas = []
    for k in range(max_tentativas):
        try:
            return funcao(), k + 1, esperas
        except ErroPermanente:
            raise
        except ErroTransitorio:
            if k == max_tentativas - 1:
                raise
            e = min(teto, t0 * 2 ** k) * (1 + rng.uniform(0, jitter))
            esperas.append(round(e, 2)); dormir(e)

# ---------- camada de registro: JSON Lines ----------
class Registro:
    def __init__(self, arquivo):
        self.arquivo = arquivo
    def escrever(self, **campos):
        with open(self.arquivo, "a", encoding="utf-8") as f:
            f.write(json.dumps(campos, ensure_ascii=False) + "\n")
    def ler(self):
        with open(self.arquivo, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

# ---------- camada de aplicação ----------
PROMPT_ID = "atendimento-v1"
PADRAO_HUMANO = "Não consegui responder com segurança; encaminhei sua pergunta a um atendente."

class Atendimento:
    def __init__(self, provedor, registro, dormir=time.sleep):
        self.provedor, self.registro, self.dormir = provedor, registro, dormir
        self.bm25 = modelos.BM25([modelos.tokenizar_sem_stop(d[3]) for d in DOCS])
        self.rng = random.Random(0)

    def responder(self, pergunta):
        limpa, achados = mascarar(pergunta)
        s = self.bm25.pontuar(modelos.tokenizar_sem_stop(limpa))
        melhor = int(s.argmax())
        contexto = DOCS[melhor][3] if s[melhor] > 0 else ""
        fonte = DOCS[melhor][0] if contexto else None
        msgs = [Mensagem("system", "Responda usando apenas o contexto. Se não houver resposta, diga: não consta."),
                Mensagem("user", f"CONTEXTO: {contexto}\nPERGUNTA: {limpa}")]
        params = Parametros()
        tentativas, esperas, r = 0, [], None
        try:
            r, tentativas, esperas = com_recuo(lambda: self.provedor.gerar(msgs, params), dormir=self.dormir, rng=self.rng)
            ok = RespostaAtendimento(resposta=r.texto, fonte=fonte)
            status = "abstencao" if ok.resposta == "não consta" else "ok"
            saida = {"resposta": ok.resposta, "fonte": ok.fonte}
        except (ErroTransitorio, ErroPermanente):
            status, saida = "plano_b_provedor", {"resposta": PADRAO_HUMANO, "fonte": None}
        except ValidationError:
            status, saida = "plano_b_validacao", {"resposta": PADRAO_HUMANO, "fonte": None}
        self.registro.escrever(
            provedor=self.provedor.nome, modelo=r.modelo if r else None, prompt_id=PROMPT_ID, status=status,
            tentativas=tentativas, espera_total_s=round(sum(esperas), 2),
            tokens_entrada=r.tokens_entrada if r else 0, tokens_saida=r.tokens_saida if r else 0,
            itens_mascarados=sum(achados.values()), hash_entrada=hashlib.sha256(limpa.encode("utf-8")).hexdigest()[:16])
        return {**saida, "status": status}

# ---------- testes (sem rede, sem espera real) ----------
def novo(provedor, d):
    esperas = []
    return Atendimento(provedor, Registro(os.path.join(d, "log.jsonl")), dormir=esperas.append), esperas

PERGUNTA = "Qual o valor da multa por atraso na devolução de livros?"

def teste_recuo(d):
    prov = ProvedorRoteirizado(ProvedorSimulado(), ["429", "503", "ok"])
    app, esperas = novo(prov, d)
    r = app.responder(PERGUNTA)
    assert r["status"] == "ok" and prov.tentativas == 3
    assert len(esperas) == 2 and 1.0 <= esperas[0] <= 1.25 and 2.0 <= esperas[1] <= 2.5

def teste_erro_permanente_nao_repete(d):
    prov = ProvedorRoteirizado(ProvedorSimulado(), ["401"])
    app, esperas = novo(prov, d)
    r = app.responder(PERGUNTA)
    assert r["status"] == "plano_b_provedor" and prov.tentativas == 1 and esperas == []

def teste_mascara_antes_de_enviar(d):
    esp = ProvedorEspiao()
    app, _ = novo(esp, d)
    app.responder("Meu CPF é 123.456.789-09 e meu e-mail é maria@exemplo.com. Qual a multa da biblioteca?")
    enviado = " ".join(esp.vistas[0])
    assert "123.456" not in enviado and "maria@" not in enviado and "[CPF]" in enviado and "[EMAIL]" in enviado
    assert app.registro.ler()[0]["itens_mascarados"] == 2

def teste_validacao_bloqueia_vazamento(d):
    app, _ = novo(ProvedorComVazamento(), d)
    r = app.responder(PERGUNTA)
    assert r["status"] == "plano_b_validacao" and "98888" not in r["resposta"]

def teste_abstencao(d):
    app, _ = novo(ProvedorSimulado(), d)
    r = app.responder("Qual a capital da Austrália?")
    assert r["resposta"] == "não consta" and r["fonte"] is None and r["status"] == "abstencao"

def teste_log_sem_texto_integral(d):
    app, _ = novo(ProvedorSimulado(), d)
    app.responder(PERGUNTA)
    bruto = open(app.registro.arquivo, encoding="utf-8").read()
    reg = app.registro.ler()
    assert len(reg) == 1 and "multa" not in bruto and len(reg[0]["hash_entrada"]) == 16
    assert {"provedor", "modelo", "prompt_id", "status", "tentativas", "tokens_entrada"} <= set(reg[0])

def teste_troca_de_provedor(d):
    for prov in (ProvedorSimulado(), ProvedorPrimeiraFrase()):
        app, _ = novo(prov, d)
        r = app.responder(PERGUNTA)
        assert r["fonte"] == "D12" and "R$ 1,50" in r["resposta"]

testes = [teste_recuo, teste_erro_permanente_nao_repete, teste_mascara_antes_de_enviar, teste_validacao_bloqueia_vazamento,
          teste_abstencao, teste_log_sem_texto_integral, teste_troca_de_provedor]
passou = 0
for t in testes:
    with tempfile.TemporaryDirectory() as d:
        try:
            t(d); passou += 1; print("passou  ", t.__name__)
        except AssertionError:
            print("FALHOU  ", t.__name__)
print(f"{passou}/{len(testes)} testes passaram")

# ---------- experimento: 56 consultas com um provedor instável ----------
roteiro = ["ok", "ok", "429", "ok", "ok", "ok", "503", "503", "ok", "ok", "ok", "ok", "ok", "ok", "401"]
with tempfile.TemporaryDirectory() as d:
    prov = ProvedorRoteirizado(ProvedorSimulado(), roteiro)
    app, esperas = novo(prov, d)
    saidas = []
    for i, (q, rel, ouro, tipo) in enumerate(CONSULTAS):
        if i < 5:
            q = "Meu CPF é 123.456.789-09. " + q
        saidas.append((app.responder(q), rel, tipo))
    reg = app.registro.ler()
print()
print("status:", dict(sorted(Counter(r["status"] for r in reg).items())))
print(f"chamadas ao provedor: {prov.tentativas} para {len(reg)} perguntas | espera total simulada: {sum(r['espera_total_s'] for r in reg):.2f} s")
print("itens mascarados no log:", sum(r["itens_mascarados"] for r in reg), "| tokens de saída (soma):", sum(r["tokens_saida"] for r in reg))
resp = [(s, rel) for s, rel, t in saidas if t != "sem_resposta" and s["status"] in ("ok", "abstencao")]
print(f"fonte correta nas respondíveis atendidas: {sum(s['fonte'] in rel for s, rel in resp)}/{len(resp)}")
sr = [s for s, rel, t in saidas if t == "sem_resposta"]
print(f"sem_resposta: {sum(s['status'] == 'abstencao' for s in sr)}/{len(sr)} abstenções; {sum(s['status'] == 'ok' for s in sr)} respostas indevidas; {sum(s['status'].startswith('plano_b') for s in sr)} no plano B")

