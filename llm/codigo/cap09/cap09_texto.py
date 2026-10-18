"""Capítulo 9: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class Mensagem:
    papel: str          # "system", "user" ou "assistant"
    conteudo: str

@dataclass
class Parametros:
    temperatura: float = 0.0
    max_tokens: int = 300
    tempo_limite_s: float = 30.0

@dataclass
class Resposta:
    texto: str
    tokens_entrada: int = 0
    tokens_saida: int = 0
    modelo: str = ""
    extra: dict = field(default_factory=dict)

class ProvedorLLM(Protocol):
    nome: str
    def gerar(self, mensagens: list[Mensagem], params: Parametros) -> Resposta: ...


# %% [bloco 2]
import re
from llmdidatico import DOCS, modelos

class ProvedorSimulado:
    nome = "simulado"
    def gerar(self, mensagens, params):
        usuario = next(m.conteudo for m in reversed(mensagens) if m.papel == "user")
        ctx, _, pergunta = usuario.partition("\nPERGUNTA: ")
        frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", ctx.replace("CONTEXTO:", "")) if f.strip()]
        if not frases:
            texto = "não consta"
        else:
            q = set(modelos.tokenizar_sem_stop(pergunta))
            pontos = [len(q & set(modelos.tokenizar_sem_stop(f))) for f in frases]
            texto = frases[pontos.index(max(pontos))] if max(pontos) > 0 else "não consta"
        return Resposta(texto=texto, tokens_entrada=len(usuario.split()), tokens_saida=len(texto.split()), modelo="simulado-v1")

class Assistente:
    """Lógica da aplicação: recupera contexto, monta o prompt, chama o provedor e valida a saída."""
    def __init__(self, provedor: ProvedorLLM):
        self.provedor = provedor
        self.bm25 = modelos.BM25([modelos.tokenizar_sem_stop(d[3]) for d in DOCS])

    def responder(self, pergunta):
        s = self.bm25.pontuar(modelos.tokenizar_sem_stop(pergunta))
        melhor = int(s.argmax())
        contexto = DOCS[melhor][3] if s[melhor] > 0 else ""
        msgs = [Mensagem("system", "Responda usando apenas o contexto. Se não houver resposta, diga: não consta."),
                Mensagem("user", f"CONTEXTO: {contexto}\nPERGUNTA: {pergunta}")]
        r = self.provedor.gerar(msgs, Parametros(temperatura=0.0, max_tokens=200))
        return {"resposta": r.texto, "fonte": DOCS[melhor][0] if contexto else None, "modelo": r.modelo}

app = Assistente(ProvedorSimulado())
for p in ["Qual o valor da multa por atraso na devolução de livros?", "Até quando posso trancar a matrícula?", "Qual a capital da Austrália?"]:
    print(app.responder(p))


# %% [bloco 3]
import random

class ErroTransitorio(Exception):
    """Falha que pode passar (rede, 429, 5xx, tempo esgotado)."""

class ErroPermanente(Exception):
    """Falha que não passa (400, 401, 403, 404)."""

def com_recuo(funcao, max_tentativas=5, t0=1.0, teto=30.0, jitter=0.25, dormir=lambda s: None, rng=None):
    """Chama funcao() repetindo só em ErroTransitorio. 'dormir' é injetável para testar sem esperar."""
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
            espera = min(teto, t0 * 2 ** k) * (1 + rng.uniform(0, jitter))
            esperas.append(round(espera, 2)); dormir(espera)

class ServicoInstavel:
    def __init__(self, falhas):
        self.falhas = falhas
    def __call__(self):
        if self.falhas > 0:
            self.falhas -= 1
            raise ErroTransitorio("429 limite de taxa")
        return "ok"

resultado, tentativas, esperas = com_recuo(ServicoInstavel(falhas=3))
print("resultado:", resultado, "| tentativas:", tentativas, "| esperas (s):", esperas)

try:
    com_recuo(ServicoInstavel(falhas=10), max_tentativas=4)
except ErroTransitorio as e:
    print("esgotou as tentativas:", e)

def servico_com_chave_invalida():
    raise ErroPermanente("401 credencial inválida")
try:
    com_recuo(servico_com_chave_invalida)
except ErroPermanente as e:
    print("erro permanente, sem repetição:", e)


# %% [bloco 4]
def memoria_pesos_gb(n_params, bits):
    return n_params * bits / 8 / 1e9

def memoria_kv_gb(camadas, T, n_kv, d_cab, bytes_num=2):
    return 2 * camadas * T * n_kv * d_cab * bytes_num / 1e9

def cabe(n_params, bits, camadas, T, n_kv, d_cab, gb_disponiveis, margem=0.15):
    total = memoria_pesos_gb(n_params, bits) + memoria_kv_gb(camadas, T, n_kv, d_cab)
    return total, total <= gb_disponiveis * (1 - margem)

for nome, N, bits in [("7 bi, 16 bits", 7e9, 16), ("7 bi, 8 bits", 7e9, 8), ("7 bi, 4 bits", 7e9, 4), ("13 bi, 4 bits", 13e9, 4)]:
    total, ok = cabe(N, bits, camadas=32, T=4096, n_kv=8, d_cab=128, gb_disponiveis=16)
    print(f"{nome:<15} pesos+cache ≈ {total:5.1f} GB | cabe em 16 GB com 15% de margem? {ok}")


# %% [bloco 5]
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

entrada = "Sou Maria, CPF 123.456.789-09, e-mail maria.silva@exemplo.com.br, telefone (61) 98888-7777. Quero cancelar meu plano."
limpo, achados = mascarar(entrada)
print(limpo)
print("itens mascarados:", achados)

def aparar_historico(mensagens, tokens_por_mensagem, limite):
    escolhidas, usados = [], 0
    for m, n in zip(reversed(mensagens), reversed(tokens_por_mensagem)):
        if usados + n > limite:
            break
        escolhidas.append(m); usados += n
    return list(reversed(escolhidas)), usados

hist, usados = aparar_historico(["m1", "m2", "m3", "m4"], [300, 200, 400, 150], 800)
print("histórico mantido:", hist, "| tokens:", usados)


# %% [bloco 6]
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

class RespostaAtendimento(BaseModel):
    resposta: str = Field(min_length=1, max_length=500)
    fonte: Optional[str] = None

    @field_validator("resposta")
    @classmethod
    def sem_dados_pessoais(cls, v):
        _, achados = mascarar(v)
        if achados:
            raise ValueError(f"a resposta contém dados pessoais: {achados}")
        return v

def validar_resposta(dicionario):
    try:
        return RespostaAtendimento(**dicionario), None
    except ValidationError as e:
        return None, e.errors()[0]["msg"]

for d in [{"resposta": "A multa é de R$ 1,50 por dia.", "fonte": "D12"},
          {"resposta": "", "fonte": "D12"},
          {"resposta": "Ligue para (61) 98888-7777 para saber mais.", "fonte": None}]:
    obj, erro = validar_resposta(d)
    print("OK    " if obj else "FALHA ", "|", d["resposta"][:42].ljust(42), "|", erro or "válida")


# %% [bloco 7]
class Orcamento:
    def __init__(self, max_tokens_total, preco_por_mil_tokens=None):
        self.max, self.usado, self.preco = max_tokens_total, 0, preco_por_mil_tokens
    def registrar(self, tokens):
        self.usado += tokens
        if self.usado > self.max:
            raise RuntimeError(f"orçamento excedido: {self.usado} > {self.max} tokens")
    def custo(self):
        return None if self.preco is None else self.usado / 1000 * self.preco

class Disjuntor:
    def __init__(self, limite_falhas=3):
        self.limite, self.falhas, self.aberto = limite_falhas, 0, False
    def chamar(self, funcao, plano_b):
        if self.aberto:
            return plano_b(), "plano_b (disjuntor aberto)"
        try:
            r = funcao(); self.falhas = 0; return r, "principal"
        except ErroTransitorio:
            self.falhas += 1
            if self.falhas >= self.limite:
                self.aberto = True
            return plano_b(), "plano_b (falha)"

orc = Orcamento(max_tokens_total=500, preco_por_mil_tokens=2.0)       # preço hipotético
for n in (200, 200):
    orc.registrar(n)
print("tokens usados:", orc.usado, "| custo hipotético:", orc.custo())
try:
    orc.registrar(200)
except RuntimeError as e:
    print("bloqueado:", e)

d = Disjuntor(limite_falhas=2)
def sempre_falha(): raise ErroTransitorio("indisponível")
print([d.chamar(sempre_falha, lambda: "resposta padrão")[1] for _ in range(4)])


# %% [bloco 8]
import os

def obter_credencial(nome_variavel):
    valor = os.environ.get(nome_variavel)
    if not valor:
        raise RuntimeError(f"variável de ambiente ausente: {nome_variavel} (defina-a fora do código; nunca versione)")
    return valor

def mascarar_chave(chave):
    return "****" if len(chave) <= 8 else chave[:3] + "…" + chave[-2:]

PADROES_SEGREDO = [re.compile(r"(?i)(chave|api[_-]?key|token|senha|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}")]

def procurar_segredos(texto):
    return [m.group(0)[:30] + "…" for rx in PADROES_SEGREDO for m in rx.finditer(texto)]

os.environ["CHAVE_DEMO_FALSA"] = "valor-falso-abcdef123456"      # valor FALSO, só para demonstração
print("chave mascarada para exibição:", mascarar_chave(obter_credencial("CHAVE_DEMO_FALSA")))
try:
    obter_credencial("VARIAVEL_QUE_NAO_EXISTE")
except RuntimeError as e:
    print("erro claro:", e)

codigo_ruim = 'cliente = Servico(api_key="abcdef1234567890XYZ")\nprint("olá")'
print("segredos encontrados no código:", procurar_segredos(codigo_ruim))
print("segredos encontrados no código bom:", procurar_segredos('cliente = Servico(api_key=os.environ["CHAVE"])'))


# %% [bloco 9]
import hashlib, json, tempfile, time, os, statistics
from dataclasses import asdict

def registrar_chamada(arquivo, provedor, prompt_id, params, resposta, latencia_s, status, entrada_texto):
    linha = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provedor": provedor, "modelo": resposta.modelo if resposta else None,
        "prompt_id": prompt_id, "params": asdict(params),
        "tokens_entrada": resposta.tokens_entrada if resposta else 0, "tokens_saida": resposta.tokens_saida if resposta else 0,
        "latencia_s": round(latencia_s, 4), "status": status,
        "hash_entrada": hashlib.sha256(entrada_texto.encode("utf-8")).hexdigest()[:16],
    }
    with open(arquivo, "a", encoding="utf-8") as f:
        f.write(json.dumps(linha, ensure_ascii=False) + "\n")

class AssistenteRegistrado(Assistente):
    def __init__(self, provedor, arquivo_log):
        super().__init__(provedor); self.log = arquivo_log
    def responder(self, pergunta):
        t0 = time.perf_counter()
        r = super().responder(pergunta)
        resp = Resposta(texto=r["resposta"], tokens_entrada=len(pergunta.split()), tokens_saida=len(r["resposta"].split()), modelo=r["modelo"])
        registrar_chamada(self.log, self.provedor.nome, "prompt-v1", Parametros(), resp, time.perf_counter() - t0, "ok", pergunta)
        return r

with tempfile.TemporaryDirectory() as d:
    log = os.path.join(d, "chamadas.jsonl")
    app2 = AssistenteRegistrado(ProvedorSimulado(), log)
    from llmdidatico import CONSULTAS
    for q, rel, a, t in CONSULTAS[:20]:
        app2.responder(q)
    linhas = [json.loads(l) for l in open(log, encoding="utf-8")]
print("chamadas registradas:", len(linhas))
print("exemplo de registro:", {k: linhas[0][k] for k in ("provedor", "modelo", "prompt_id", "status", "tokens_entrada", "hash_entrada")})
print("latência mediana (s):", statistics.median(l["latencia_s"] for l in linhas), "| tokens de saída (soma):", sum(l["tokens_saida"] for l in linhas))


# %% [bloco 10]
class ProvedorPrimeiraFrase:
    nome = "primeira-frase"
    def gerar(self, mensagens, params):
        usuario = next(m.conteudo for m in reversed(mensagens) if m.papel == "user")
        ctx = usuario.partition("\nPERGUNTA: ")[0].replace("CONTEXTO:", "").strip()
        texto = re.split(r"(?<=[.!?])\s+", ctx)[0] if ctx else "não consta"
        return Resposta(texto=texto, tokens_entrada=len(usuario.split()), tokens_saida=len(texto.split()), modelo="primeira-frase-v1")

def teste_abstencao(provedor):
    r = Assistente(provedor).responder("Qual a capital da Austrália?")
    assert r["resposta"] == "não consta" and r["fonte"] is None

def teste_cita_fonte(provedor):
    r = Assistente(provedor).responder("Qual o valor da multa por atraso na devolução de livros?")
    assert r["fonte"] == "D12" and "R$ 1,50" in r["resposta"]

for prov in (ProvedorSimulado(), ProvedorPrimeiraFrase()):
    resultados = []
    for teste in (teste_abstencao, teste_cita_fonte):
        try:
            teste(prov); resultados.append("passou")
        except AssertionError:
            resultados.append("FALHOU")
    print(f"{prov.nome:<15} -> {dict(zip(['abstenção', 'cita fonte'], resultados))}")


# %% [bloco 11]
import importlib

def versao_e_nomes(pacote, nomes):
    try:
        mod = importlib.import_module(pacote)
    except ImportError:
        return f"{pacote}: não instalado"
    return f"{pacote} {getattr(mod, '__version__', '?')}: " + ", ".join(f"{n}={'sim' if hasattr(mod, n) else 'NÃO'}" for n in nomes)

print(versao_e_nomes("anthropic", ["Anthropic", "RateLimitError", "APIStatusError", "APIConnectionError", "APITimeoutError"]))
print(versao_e_nomes("transformers", ["AutoTokenizer", "AutoModelForCausalLM", "pipeline"]))
try:
    import transformers
    print("apply_chat_template em tokenizadores:", hasattr(transformers.PreTrainedTokenizerBase, "apply_chat_template"))
    print("generate nos modelos:", hasattr(transformers.GenerationMixin, "generate"))
except Exception as e:
    print("verificação adicional indisponível:", type(e).__name__)

