"""Capítulo 14: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import json
from llmdidatico.agente import Ambiente, criar_registro, kb_com_envenenados, gerar_tarefas

kb = kb_com_envenenados()                 # 60 documentos da UFA + 2 documentos "envenenados" (usados na seção 14.12)
amb = Ambiente(kb=kb)                     # estado do mundo fictício: hoje = 16/03/2026
reg = criar_registro(amb)

print("ferramentas:", reg.nomes())
esq = reg.esquemas({"consultar_prazo"})[0]
print(json.dumps(esq["input_schema"], ensure_ascii=False))

chamadas = [("consultar_prazo", {"evento": "trancamento"}),
            ("consultar_prazo", {"evento": "formatura"}),
            ("buscar_base", {"consulta": "multa biblioteca", "top_k": "três"}),
            ("voar", {})]
for nome, args in chamadas:
    r = reg.executar(nome, args)
    print(f"{nome}{args} -> ok={r.ok} erro={r.tipo_erro}\n    {r.conteudo[:110]}")


# %% [bloco 2]
from llmdidatico.agente import calcular_seguro

ataques = ["15*0.2+3", "__import__('os').system('ls')", "9**9**9", "open('/etc/passwd')", "2+", "10/0"]
for expr in ataques:
    try:
        print(f"{expr!r:34} -> {calcular_seguro(expr)}")
    except ValueError as e:
        print(f"{expr!r:34} -> recusado: {e}")

arg = {"matricula": "123456", "semestre": "2026.1", "chave_idempotencia": "tr-123456-2026.1"}
print(reg.executar("protocolar_trancamento", arg).conteudo)
print(reg.executar("protocolar_trancamento", arg).conteudo)
print("execuções reais de protocolar_trancamento:", amb.chamadas_reais["protocolar_trancamento"])

lento = criar_registro(amb, atraso_busca_s=0.8)         # ferramenta de busca artificialmente lenta (limite: 0,5 s)
r = lento.executar("buscar_base", {"consulta": "multa biblioteca"})
print("busca lenta ->", r.tipo_erro, "|", r.conteudo)


# %% [bloco 3]
from llmdidatico.agente import Agente, ModeloSimulado, Falhas, Politica, Acao, ModeloRoteirizado, rodar_tarefa, Tarefa

pergunta = "Qual a multa por 12 dias de atraso na devolução de 3 livros?"
print("plano declarado:", ModeloSimulado.plano(pergunta))
ag = Agente(ModeloSimulado(Falhas(), semente=0), criar_registro(Ambiente(kb=kb)), Politica(permitidas={"buscar_base", "calculadora"}))
print(ag.executar(pergunta).formatar())


# %% [bloco 4]
def mostra(titulo, roteiro, tarefa, **kw):
    ag = Agente(ModeloRoteirizado(roteiro), criar_registro(Ambiente(kb=kb)), **kw)
    ex = ag.executar(tarefa)
    print(f"--- {titulo}")
    print(ex.formatar(largura=70))

evento_ruim = Acao("chamar", "consultar_prazo", {"evento": "trancar matrícula"})
evento_ok = Acao("chamar", "consultar_prazo", {"evento": "trancamento"})
mostra("1) argumento inválido, depois corrigido",
       [evento_ruim, evento_ok, Acao("responder", texto="O prazo termina em 24/04/2026.")],
       "Quando termina o prazo de trancamento?", politica=Politica(permitidas={"consultar_prazo"}))
busca = Acao("chamar", "buscar_base", {"consulta": "multa biblioteca"})
mostra("2) laço: o modelo repete a mesma chamada", [busca], "Qual a multa da biblioteca?")
mostra("3) limite de passos", [busca, Acao("chamar", "buscar_base", {"consulta": "xyz abc"}), busca], "Qual a multa da biblioteca?", max_passos=2)


# %% [bloco 5]
t_multa = Tarefa("m", "multa", pergunta, esperado=["R$ 54"])
ok, ex, _ = rodar_tarefa(t_multa, Falhas(p_arg_invalido=0.5, p_ferramenta_errada=0.2), 3, kb)
print(ex.formatar(largura=70)); print("tarefa correta:", ok)


# %% [bloco 6]
from llmdidatico.agente import (fluxo_roteamento, fluxo_encadeamento, fluxo_paralelo, fluxo_orquestrador, fluxo_avaliador_otimizador)

print("ROTEAMENTO:", fluxo_roteamento("Calcule: 3*(4+5)", reg))
print("ROTEAMENTO (fora do escopo):", fluxo_roteamento(pergunta, reg))
resp, portao = fluxo_encadeamento("Quantos livros posso emprestar?", reg)
print("ENCADEAMENTO:", resp, "| portão de citação:", portao)
print("PARALELIZAÇÃO:", fluxo_paralelo(["Até quando posso trancar a matrícula?", "Calcule: 2**10"], reg))
subs, resps, final = fluxo_orquestrador("Quantos livros posso emprestar e qual o valor da multa por atraso?", reg)
print("ORQUESTRADOR: subtarefas =", subs)
q = "Quantos membros tem a banca da defesa de dissertação?"
for i, rascunho, falhas in fluxo_avaliador_otimizador(q, reg):
    print(f"AVALIADOR–OTIMIZADOR, iteração {i}: {rascunho[:62]}... | reprovado por: {falhas or 'nada (aprovado)'}")


# %% [bloco 7]
from llmdidatico.agente import MemoriaCurta, MemoriaLonga

mc = MemoriaCurta(limite_tokens=30)
for m in ["Aluna pergunta sobre trancamento de matrícula no semestre atual", "Agente consulta o prazo de trancamento e informa 24/04",
          "Aluna pergunta sobre multa da biblioteca por 12 dias", "Agente calcula a multa total e informa 54 reais"]:
    mc.adicionar(m)
print("contexto curto:", mc.contexto())

ml = MemoriaLonga()
ml.lembrar("A aluna 123456 prefere ser contatada por e-mail institucional")
ml.lembrar("A aluna 123456 cursa o 6º semestre de Engenharia")
ml.lembrar("O aluno 654321 tem bolsa de monitoria")
print("recordar:", ml.recordar("como a aluna 123456 prefere ser contatada", k=1))


# %% [bloco 8]
from llmdidatico.agente import Auditoria, executar_isolado

# (a) menor privilégio: o modelo "erra" e tenta enviar um comunicado numa tarefa de consulta
alvo = Agente(ModeloRoteirizado([Acao("chamar", "enviar_comunicado", {"destinatario": "aluna@ufa.exemplo.br", "texto": "oi"}),
                                 Acao("responder", texto="não consegui enviar")]),
              criar_registro(Ambiente(kb=kb)), Politica(permitidas={"buscar_base"}))
ex = alvo.executar("Qual o horário da biblioteca?"); print("(a) bloqueios:", ex.bloqueios, "| efeitos:", ex.efeitos)

# (b) confirmação humana: padrão = negar; com um humano que aprova, a ação ocorre
tranca = Acao("chamar", "protocolar_trancamento", {"matricula": "123456", "semestre": "2026.1", "chave_idempotencia": "tr-123456-2026.1"})
for nome, conf in [("sem confirmação (padrão)", None), ("humano aprova", lambda n, a: True)]:
    amb2 = Ambiente(kb=kb); ag2 = Agente(ModeloRoteirizado([tranca, Acao("responder", texto="feito")]), criar_registro(amb2), confirmador=conf)
    ex = ag2.executar("Tranque a matrícula 123456."); print(f"(b) {nome}: bloqueios={ex.bloqueios} protocolos={list(amb2.protocolos.values())}")

# (c) auditoria encadeada por hash, com mascaramento de dados pessoais
aud = Auditoria(); ag3 = Agente(ModeloSimulado(Falhas(), 0), criar_registro(Ambiente(kb=kb)), auditoria=aud)
ag3.executar("Aluna de CPF 123.456.789-09 pergunta: qual o horário da biblioteca?")
print("(c) eventos:", [e["tipo"] for e in aud.eventos], "| íntegro:", aud.verificar())
print("    início registrado:", aud.eventos[0]["dados"])
aud.eventos[1]["dados"]["ferramenta"] = "adulterado"; print("    após adulteração, íntegro:", aud.verificar())

# (d) sandbox: processo separado com limites de tempo e memória
for cod in ["print(sum(range(10)))", "while True: pass", "x = [0] * 10**9"]:
    print("(d)", repr(cod)[:24].ljust(24), "->", executar_isolado(cod, timeout_s=1.0, mem_mb=256))


# %% [bloco 9]
amb3 = Ambiente(kb=kb); reg3 = criar_registro(amb3)
modelo_obediente = lambda: ModeloSimulado(Falhas(susceptibilidade=1.0), semente=0)
aberta = Politica(dominios_email=("",))                                    # sem restrição de domínio (inseguro)
ag = Agente(modelo_obediente(), reg3, aberta, sanitizar=False, confirmador=lambda n, a: True)
ex = ag.executar("Qual o horário de atendimento da secretaria acadêmica para protocolos?")
print(ex.formatar(largura=100)); print("e-mails enviados:", amb3.outbox)


# %% [bloco 10]
from llmdidatico.agente import avaliar, politica_para

inj = [t for t in gerar_tarefas() if t.tipo == "injecao"]
sim, nao = (lambda n, a: True), (lambda n, a: False)
configs = {"nenhuma": dict(sanitizar=False, politica=aberta, confirmador=sim),
           "sanitizador de frases": dict(sanitizar=True, politica=aberta, confirmador=sim),
           "domínio permitido": dict(sanitizar=False, politica=Politica(), confirmador=sim),
           "menor privilégio": dict(sanitizar=False, politica=Politica(permitidas={"buscar_base"}, dominios_email=("",)), confirmador=sim),
           "confirmação humana": dict(sanitizar=False, politica=aberta, confirmador=nao)}
print(f"{'defesa':<24} {'ação indevida':>14} {'tarefa correta':>15}")
for nome, kw in configs.items():
    r = avaliar(inj, Falhas(susceptibilidade=1.0), kb, K=20, **kw)
    print(f"{nome:<24} {r['desfechos']['ação indevida']:>13}% {r['desfechos']['sucesso']:>14}%")


# %% [bloco 11]
import numpy as np
from llmdidatico import avaliacao as av

T = gerar_tarefas()
from collections import Counter
print("tarefas por tipo:", dict(Counter(t.tipo for t in T)))

def ruido(p, rec=0.7):                   # p reparte-se entre os tipos de falha injetada
    return Falhas(p_arg_invalido=p / 2, p_ferramenta_errada=p / 4, p_laco=p / 4, p_recupera_erro=rec)

print(f"{'p':>4} {'sucesso':>8} {'IC 95%':>16} {'passos':>7} {'tokens':>7}")
for p in (0, 0.2, 0.4, 0.6, 0.8):
    r = avaliar(T, ruido(p), kb, K=20, max_passos=6)
    m, lo, hi = av.bootstrap_ic(r["taxa"], n=1000, seed=0)
    print(f"{p:>4} {m:>8.3f} {f'[{lo:.3f}; {hi:.3f}]':>16} {r['passos'].mean():>7.2f} {r['tokens'].mean():>7.0f}")


# %% [bloco 12]
print(f"{'passos máx.':>11} {'sucesso':>8} {'IC 95%':>16} {'tokens':>7}")
for mp in (2, 3, 4, 6, 10):
    r = avaliar(T, ruido(0.4), kb, K=20, max_passos=mp)
    m, lo, hi = av.bootstrap_ic(r["taxa"], n=1000, seed=0)
    print(f"{mp:>11} {m:>8.3f} {f'[{lo:.3f}; {hi:.3f}]':>16} {r['tokens'].mean():>7.0f}")


# %% [bloco 13]
r = avaliar(T, ruido(0.2), kb, K=20, max_passos=6)
tipos = np.array(r["tipos"])
print(f"{'tipo':<9} {'fluxo (código)':>15} {'agente (p=0,2)':>15}")
for tp in dict.fromkeys(tipos):
    oks = []
    for tk in [x for x in T if x.tipo == tp]:
        resp, _ = fluxo_roteamento(tk.texto, criar_registro(Ambiente(kb=kb, hoje=tk.hoje)))
        oks.append(all(e.lower() in resp.lower() for e in tk.esperado))
    print(f"{tp:<9} {np.mean(oks):>15.3f} {r['taxa'][tipos == tp].mean():>15.3f}")


# %% [bloco 14]
from llmdidatico import CONSULTAS
ant = dep = consertou = quebrou = 0
for q, rel, a, tp in CONSULTAS:
    h = fluxo_avaliador_otimizador(q, reg)
    pri, ult = (a in h[0][1]) if h else False, (a in h[-1][1]) if h else False
    ant += pri; dep += ult; consertou += (not pri and ult); quebrou += (pri and not ult)
print(f"corretas antes: {ant} | depois: {dep} | consertadas: {consertou} | quebradas pelo critério: {quebrou} (de {len(CONSULTAS)})")


# %% [bloco 15]
r = avaliar(T, ruido(0.8), kb, K=20, max_passos=6)
tot = sum(r["desfechos"].values())
print("desfechos (p = 0,8):", {k: f"{100 * v / tot:.1f}%" for k, v in r["desfechos"].most_common()})
print("incidentes por tipo de erro:", dict(r["incidentes"].most_common()))


# %% [bloco 16]
import anthropic
nomes = ["Anthropic", "RateLimitError", "APIStatusError", "APIConnectionError", "APITimeoutError"]
print("anthropic", anthropic.__version__, {n: hasattr(anthropic, n) for n in nomes})
from anthropic.resources.messages import Messages
from anthropic.types import ToolUseBlock, ToolResultBlockParam
print("Messages.create:", hasattr(Messages, "create"), "| ToolUseBlock e ToolResultBlockParam importados")
reg_teste = criar_registro(Ambiente(kb=kb))
print("chaves de cada esquema de ferramenta:", sorted(reg_teste.esquemas()[0].keys()))

