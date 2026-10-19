"""Capítulo 18: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import json
import numpy as np
from collections import Counter, defaultdict
from llmdidatico import DOCS, CONSULTAS, rag
from llmdidatico.avaliacao import bootstrap_ic
from llmdidatico import projetos as P

print("Quanto o tamanho da amostra muda a conclusão? (critério de aceitação: taxa >= 0,80)")
for k, n in [(8, 10), (27, 30), (90, 100), (270, 300)]:
    t = P.prop_ic([1] * k + [0] * (n - k))
    print(f"  {k}/{n}: taxa {t[0]:.2f}, IC95% de Wilson [{t[1]:.3f}; {t[2]:.3f}] -> {P.veredito(t[1], t[2], 0.80)}")


# %% [bloco 2]
pipe, ix, idf = P.montar_rag(DOCS, limiar=0.5)
linhas = rag.avaliar(pipe, CONSULTAS)
m = P.metricas_rag(linhas)
criterios_p1 = {"recuperacao": (0.85, True), "resposta_correta": (0.70, True), "abstencao_correta": (0.90, True),
                "alucinacao": (0.10, False), "fidelidade": (0.95, True)}
print(f"{len(DOCS)} documentos, {len(CONSULTAS)} consultas; gerador SIMULADO (não é um LLM)\n")
print(f"{'métrica':<20}{'taxa [IC95% de Wilson] (n)':<36}{'critério':>9}  veredito")
for nome, (lim, maior) in criterios_p1.items():
    t = m[nome]
    print(f"{nome:<20}{P.fmt_ic(t):<36}{('>= ' if maior else '<= ') + format(lim, '.2f'):>9}  {P.veredito(t[1], t[2], lim, maior)}")
print("\nrótulos de erro (todas as consultas):", dict(Counter(l["rotulo"] for l in linhas)))


# %% [bloco 3]
varredura = []
print(f"{'limiar':>7} {'resposta correta':>17} {'abstenção correta':>18} {'abstenção indevida':>19}")
for lim in (0.3, 0.4, 0.5, 0.6, 0.7):
    p_, _, _ = P.montar_rag(DOCS, limiar=lim)
    mm = P.metricas_rag(rag.avaliar(p_, CONSULTAS))
    varredura.append((lim, mm["resposta_correta"], mm["abstencao_correta"], mm["abstencao_indevida"]))
    print(f"{lim:>7.1f} {mm['resposta_correta'][0]:>17.3f} {mm['abstencao_correta'][0]:>18.3f} {mm['abstencao_indevida'][0]:>19.3f}")


# %% [bloco 4]
print("Análise de erros: onde o sistema falha, por tipo de pergunta")
for tipo in ("direta", "parafrase"):
    c = Counter(l["rotulo"] for l in linhas if l["tipo"] == tipo)
    print(f"  {tipo:<10} {dict(c)}")
print("\nExemplos de abstenção indevida (o documento estava na base e foi recuperado):")
for l in [l for l in linhas if l["rotulo"] == "abstencao_indevida"][:4]:
    print(f"  - {l['pergunta']}  | relevante: {l['rel']} | recuperados: {l['recuperados']}")
print("\nExemplos de falha de recuperação (o documento relevante não chegou ao contexto):")
for l in [l for l in linhas if l["rotulo"] == "falha_recuperacao"][:3]:
    print(f"  - {l['pergunta']}  | relevante: {l['rel']} | recuperados: {l['recuperados']}")


# %% [bloco 5]
from llmdidatico.aval_llm import mcnemar, bootstrap_pareado
com = [q for q in CONSULTAS if q[1]]
ok = {}
for metodo in ("bm25", "hibrido"):
    p_, _, _ = P.montar_rag(DOCS, metodo=metodo)
    ok[metodo] = [int(l["rotulo"] == "ok") for l in rag.avaliar(p_, com)]
a, b = ok["hibrido"], ok["bm25"]
d, lo, hi = bootstrap_pareado(a, b, seed=0)
mc = mcnemar(a, b)
print(f"resposta correta: híbrido {np.mean(a):.3f} | BM25 {np.mean(b):.3f} | diferença {d:+.3f} (IC95% pareado [{lo:+.3f}; {hi:+.3f}])")
print(f"discordâncias: híbrido acerta e BM25 erra = {mc['b']}; o inverso = {mc['c']}; McNemar exato p = {mc['p_exato']:.3f}  (n = {len(com)})")


# %% [bloco 6]
from llmdidatico.saude_sintetica import DOCS_SAUDE, CONSULTAS_SAUDE, CONSULTAS_SAUDE_CEGAS, consultas_com_dados_pessoais
triagem = P.TriagemSaude(DOCS_SAUDE)
dev = CONSULTAS_SAUDE + consultas_com_dados_pessoais()
res = [(q, triagem.atender(q[0])) for q in dev]                     # desenvolvimento: as regras foram escritas olhando estas perguntas
res_cego = [(q, triagem.atender(q[0])) for q in CONSULTAS_SAUDE_CEGAS]   # cego: escrito depois de congelar as regras
guarda = lambda r: r["rota"] if r["rota"] in ("urgencia", "encaminhar_profissional") else "servico"
esperado_guarda = lambda q: {"urgencia": "urgencia", "clinica": "encaminhar_profissional"}.get(q[3], "servico")
print(f"{len(DOCS_SAUDE)} documentos fictícios; informativo, sem diagnóstico. Desenvolvimento: {len(res)} perguntas | cego: {len(res_cego)}\n")
print(f"{'camada de guardrails':<34}{'desenvolvimento':<32}{'conjunto cego'}")
for nome, alvo in [("urgência: revocação", "urgencia"), ("clínica: revocação", "encaminhar_profissional"),
                   ("serviço não desviado", "servico")]:
    cel = []
    for conj in (res, res_cego):
        cel.append(P.fmt_ic(P.prop_ic([int(guarda(r) == alvo) for q, r in conj if esperado_guarda(q) == alvo])))
    print(f"{nome:<34}{cel[0]:<32}{cel[1]}")
print("\nErros do roteador no conjunto cego:")
for q, r in res_cego:
    if guarda(r) != esperado_guarda(q):
        print(f"  esperado {esperado_guarda(q):<24} obtido {guarda(r):<24} | {q[0]}")


# %% [bloco 7]
print("Respostas-modelo (a mensagem de urgência vem de texto fixo, nunca do gerador):")
for i in (42, 46, 0):
    q, r = res[i]
    print(f"\n[{q[3]}] {q[0]}\n  -> ({r['rota']}) {r['resposta'][:230]}")
print("\nServiço: qualidade das respostas (rótulos do RAG) e abstenção:")
serv = [(q, r) for q, r in res if q[3] in ("direta", "parafrase")]
rot = [rag.classificar_caso(q[:4], r["bruta"], r["docs"], [f for f in triagem.ix.fr if f.doc_id in r["docs"]])[0] for q, r in serv]
print("  ", dict(Counter(rot)))
print(f"  resposta correta: {P.fmt_ic(P.prop_ic([x == 'ok' for x in rot]))}")
sem = [r for q, r in res if q[3] == "sem_resposta"]
print(f"  perguntas sem resposta nos documentos que receberam abstenção: {sum(r['rota'] == 'abstencao' for r in sem)}/{len(sem)}")
print("\nPrivacidade: o que foi para o registro (auditoria encadeada)?")
for q, r in res[-2:]:
    print("  pergunta mascarada:", r["pergunta_mascarada"])
log = json.dumps(triagem.aud.eventos, ensure_ascii=False)
print("  CPF ou telefone das perguntas aparecem no registro?", any(s in log for q, _ in res[-2:] for s in ("98888-1234", q[0].split("CPF é ")[-1][:14])))
print("  registro íntegro (hash encadeado)?", triagem.aud.verificar())


# %% [bloco 8]
print("Equidade por variedade de registro (formal x coloquial), conjunto de desenvolvimento")
def correto(q, r):
    return rag.classificar_caso(q[:4], r["bruta"], r["docs"], [f for f in triagem.ix.fr if f.doc_id in r["docs"]])[0] == "ok"
acerto_registro = {"formal": [], "coloquial": []}
estrato = defaultdict(list)
for q, r in res:
    if q[3] in ("direta", "parafrase"):
        acerto_registro[q[4]].append(int(correto(q, r))); estrato[(q[3], q[4])].append(int(correto(q, r)))
for grupo, v in acerto_registro.items():
    print(f"  serviço respondido corretamente, {grupo:<9}: {P.fmt_ic(P.prop_ic(v))}")
d, lo, hi = P.dif_prop_ic(acerto_registro["formal"], acerto_registro["coloquial"])
print(f"  diferença bruta formal - coloquial: {d:+.2f} (IC95% bootstrap [{lo:+.2f}; {hi:+.2f}])")
print("  mesma comparação, separando o tipo da pergunta:")
for tipo in ("direta", "parafrase"):
    for grupo in ("formal", "coloquial"):
        v = estrato[(tipo, grupo)]
        print(f"    {tipo:<10} {grupo:<10} {sum(v)}/{len(v)}")
from llmdidatico.pii import gerar_textos, contar, prf
for nome, dados in [("fácil", gerar_textos(300, 7)), ("difícil", gerar_textos(300, 7, adversarial=True))]:
    tot, _ = contar(dados)
    tp, fp, fn = (sum(v[i] for v in tot.values()) for i in range(3))
    p_, r_ = prf(tp, fp, fn)
    print(f"  detector de dados pessoais, textos {nome}: precisão {p_:.3f}, revocação {r_:.3f}")


# %% [bloco 9]
from llmdidatico.normas_sinteticas import NORMAS, META, NIVEL, CONSULTAS_NORMAS
print(f"{len(NORMAS)} normas fictícias (com vigência e perfis); {len(CONSULTAS_NORMAS)} consultas com data de referência e perfil\n")
print(f"{'modo':<12}{'k':>2} {'respostas ok':>13} {'vazamentos':>11} {'fora da vigência':>17}")
tab3 = {}
for modo in ("ingenuo", "pos_filtro", "pre_filtro"):
    for k in (1, 3):
        sis = P.ConsultaNormas(NORMAS, META, NIVEL, modo=modo, k=k)
        L = P.avaliar_normas(sis, CONSULTAS_NORMAS)
        tab3[(modo, k)] = L
        print(f"{modo:<12}{k:>2} {sum(l['ok'] for l in L):>10}/{len(L)} {sum(l['vazamento'] for l in L):>11} {sum(l['fora_da_vigencia'] for l in L):>17}")


# %% [bloco 10]
sis = P.ConsultaNormas(NORMAS, META, NIVEL, modo="pre_filtro", k=3)
for pergunta, data, perfil in [("Qual a taxa de alvará de funcionamento?", "2025-06-01", "publico"),
                               ("Qual a taxa de alvará de funcionamento?", "2026-06-01", "publico"),
                               ("Qual o prazo da sindicância investigativa?", "2026-10-01", "publico"),
                               ("Qual o prazo da sindicância investigativa?", "2026-10-01", "juridico")]:
    r = sis.consultar(pergunta, data, perfil, id_usuario="maria.silva")
    print(f"[{perfil:<8} em {data}] {pergunta}\n   -> {r['resposta']}")
print("\nTrilha de auditoria (3 primeiros eventos, com usuário pseudonimizado):")
for e in sis.aud.eventos[:3]:
    d = e["dados"]
    print(f"  #{e['seq']} {d['usuario']} {d['perfil']:<8} {d['data_ref']} docs={d['docs']} hash={e['hash']}")
print("trilha íntegra?", sis.aud.verificar())
sis.aud.eventos[1]["dados"]["perfil"] = "juridico"           # alguém altera o registro depois
print("após adulterar um evento, íntegra?", sis.aud.verificar())


# %% [bloco 11]
print("Análise de erros do modo correto (pre_filtro, k=3), por tipo de consulta:")
L = tab3[("pre_filtro", 3)]
for tipo in ("simples", "vigencia", "restrito", "sem_resposta"):
    v = [l["ok"] for l in L if l["tipo"] == tipo]
    print(f"  {tipo:<13} {sum(v)}/{len(v)}")
print("\nNo modo ingênuo (sem vigência nem perfil), exemplos de falha:")
for l in [l for l in tab3[("ingenuo", 3)] if l["vazamento"] or l["fora_da_vigencia"]][:3]:
    print(f"  [{l['perfil']} em {l['data']}] {l['pergunta']}\n     -> {l['resposta'][:120]}")


# %% [bloco 12]
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM

Xtr, ytr = corpus.intencoes(100, seed=1); Xte, yte = corpus.intencoes(40, seed=2)
Xn = [t for t, _ in P.MENSAGENS_NOVAS]; yn = [r for _, r in P.MENSAGENS_NOVAS]
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(Xtr), ytr)
classificar = lambda t: str(clf.predict(vec.transform([t]))[0])
def sim(versao):
    s = SimuladorLLM(versao, semente=0)
    def f(t):
        r = s.responder(t)
        return json.loads(r)["intencao"] if versao == "v3" else (r.split(":")[-1].split("(")[0].strip() if r.startswith("rotulo") else "indefinido")
    return f
print(f"{'classificador':<26}{'modelos de frase (n=320)':<30}{'mensagens novas (n=24)'}")
for nome, f in [("TF-IDF + regressão logística", classificar), ("SimuladorLLM v2 (regras)", sim("v2")), ("SimuladorLLM v3 (regras)", sim("v3"))]:
    a = P.prop_ic([f(t) == r for t, r in zip(Xte, yte)]); b = P.prop_ic([f(t) == r for t, r in zip(Xn, yn)])
    print(f"{nome:<26}{P.fmt_ic(a):<30}{P.fmt_ic(b)}")
print("\nErros do TF-IDF + LR nas mensagens novas:")
for t, r in P.MENSAGENS_NOVAS:
    if classificar(t) != r:
        print(f"  '{t}' -> previu {classificar(t)} (correto: {r})")


# %% [bloco 13]
casos = P.gerar_casos_atendimento(10, seed=0)
K = 5
print(f"{len(casos)} casos (8 intenções); {sum(c['intencao'] in P.INTENCOES_AUTOMATIZAVEIS for c in casos)} automatizáveis; {K} sementes por caso\n")
print(f"{'plano B':<9}{'falha da ferramenta':>20}{'sucesso por caso [IC95% bootstrap]':>40}{'enviados a humano':>19}")
res4 = {}
for plano_b in (False, True):
    for pf in (0.0, 0.3):
        runs = [P.rodar_atendimento(casos, classificar, plano_b=plano_b, p_falha=pf, semente=s) for s in range(K)]
        por_caso = np.mean([[r["ok"] for r in run] for run in runs], axis=0)
        m_, lo, hi = bootstrap_ic(por_caso, n=1000, seed=0)
        hum = np.mean([[r["humano"] for r in run] for run in runs])
        res4[(plano_b, pf)] = runs
        print(f"{'com' if plano_b else 'sem':<9}{pf:>20.1f}{f'{m_:.3f} [{lo:.3f}; {hi:.3f}]':>40}{hum:>19.3f}")
sem_, com_ = (np.mean([[r["ok"] for r in run] for run in res4[(pb, 0.3)]], axis=0) for pb in (False, True))
d, lo, hi = bootstrap_pareado(com_, sem_, seed=0)
print(f"\ncom - sem plano B (falha 0,3), pareado por caso: {d:+.3f} [{lo:+.3f}; {hi:+.3f}]")
novos = P.gerar_casos_atendimento(seed=0, fonte="novas")
r_novas = P.rodar_atendimento(novos, classificar, plano_b=True, p_falha=0.0)
print(f"mensagens NOVAS (n={len(novos)}, sem falha de ferramenta, com plano B): sucesso {P.fmt_ic(P.prop_ic([r['ok'] for r in r_novas]))}")
exec_ = [r for runs_ in res4.values() for run in runs_ for r in run]
print(f"cancelamentos executados SEM confirmação, em {len(exec_)} execuções: {sum(r['sem_confirmacao'] for r in exec_)}")


# %% [bloco 14]
runs = res4[(True, 0.3)]
auto = lambda r, c: c["intencao"] in P.INTENCOES_AUTOMATIZAVEIS
A = [(np.mean([run[i]["custo"] for run in runs]), np.mean([run[i]["latencia"] for run in runs])) for i in range(len(casos))]
custo_A = np.mean([a[0] for a in A]); lat_A = np.mean([a[1] for a in A])
B = [(a[0], a[1]) if auto(None, c) else (0.0, P.LAT_CLASSIFICADOR_S) for a, c in zip(A, casos)]
custo_B = np.mean([b[0] for b in B]); lat_B = np.mean([b[1] for b in B])
print("Custo e latência por conversa (preços e tempos HIPOTÉTICOS; não são medidas)")
print(f"  A) agente em todas as conversas        : custo por milhão de conversas {custo_A * 1e6:7.1f} un. | latência {lat_A:.2f} s")
print(f"  B) classificador antes; agente só se automatizável: custo por milhão de conversas {custo_B * 1e6:7.1f} un. | latência {lat_B:.2f} s")
print(f"  economia de B sobre A: custo {1 - custo_B / custo_A:.0%}, latência {1 - lat_B / lat_A:.0%}  (vem de não chamar o modelo onde o destino é sempre um humano)")
print(f"  projeção para 100 mil conversas: A = {custo_A * 1e5:.1f} | B = {custo_B * 1e5:.1f} unidades hipotéticas")


# %% [bloco 15]
from llmdidatico.doctec_sintetica import gerar_paginas
from llmdidatico.modelos import BM25, tokenizar_sem_stop
from llmdidatico.aval_llm import rouge_l
paginas = gerar_paginas(36, seed=0)
bm = BM25([tokenizar_sem_stop(p["texto"]) for p in paginas])
SINONIMO = {"criar": "incluir", "listar": "ver todos os", "obter": "consultar um", "atualizar": "editar", "remover": "apagar", "arquivar": "guardar"}
consultas = []
for k, p in enumerate(paginas):
    acao, rec = p["gold"]["nome"].split("_")
    consultas.append((f"como {acao} {rec} pela API?" if k % 2 == 0 else f"como {SINONIMO[acao]} {rec} pela API?", p["id"], "literal" if k % 2 == 0 else "sinônimo"))
pos = []
for q, alvo, _ in consultas:
    ordem = [paginas[i]["id"] for i in np.argsort(-bm.pontuar(tokenizar_sem_stop(q)), kind="stable")]
    pos.append(ordem.index(alvo) + 1)
print(f"{len(paginas)} páginas sintéticas ({dict(Counter(p['estilo'] for p in paginas))})")
for tipo in ("literal", "sinônimo"):
    for k in (1, 3):
        print(f"busca BM25, consulta {tipo:<8}, página certa entre os {k} primeiros: {P.fmt_ic(P.prop_ic([x <= k for x, c in zip(pos, consultas) if c[2] == tipo]))}")
for n in (1, 2):
    r = [rouge_l(p["resumo"], P.resumo_lead(p["texto"], n)) for p in paginas]
    r = [x["f"] for x in r]
    print(f"resumo extrativo (primeiras {n} frase(s)) x resumo de referência: ROUGE-L F1 médio {np.mean(r):.3f}")


# %% [bloco 16]
print(f"{'extrator':<10}{'reparo':<8}{'JSON válido':<30}{'todos os campos corretos':<30}")
res5 = {}
for versao in ("v1", "v2"):
    for reparo in (False, True):
        out = [P.extrair_validado(p, versao, reparo, 0.25, i) for i, p in enumerate(paginas)]
        val = [o is not None for o, _ in out]
        tudo = [all(P.acerto_campos(o, p["gold"]).values()) for (o, _), p in zip(out, paginas)]
        res5[(versao, reparo)] = (out, val, tudo)
        print(f"{versao:<10}{'sim' if reparo else 'não':<8}{P.fmt_ic(P.prop_ic(val)):<30}{P.fmt_ic(P.prop_ic(tudo)):<30}")
out, val, tudo = res5[("v2", True)]
print("\nv2 com reparo, por estilo da página:")
for est in ("tabela", "prosa", "ruidoso"):
    v = [t for t, p in zip(tudo, paginas) if p["estilo"] == est]
    print(f"  {est:<8} {sum(v)}/{len(v)} páginas totalmente corretas")
print("motivos de rejeição pelo esquema:", dict(Counter(m for _, m in out if m != "valido")))


# %% [bloco 17]
from pydantic import ValidationError
ruim = '{"nome": "criar_alerta", "metodo": "FETCH", "caminho": "/v2/alertas", "parametros": [], "versao_minima": "2", "autenticacao": "sim", "limite_por_min": 0}'
try:
    P.EndpointDoc.model_validate_json(ruim)
except ValidationError as e:
    print(f"{e.error_count()} erros de validação (o esquema barra o que o texto fluente esconderia):")
    for d in e.errors():
        print(f"  campo {d['loc'][0]:<15} tipo={d['type']}")


# %% [bloco 18]
from llmdidatico import modelos
from llmdidatico.lora import MLPLMLoRA, frases_dominio_novo, vocabulario
ant = corpus.frases_lm(1500, seed=0); ant_te = corpus.frases_lm(400, seed=99)
nov = frases_dominio_novo(800, seed=1); nov_va = frases_dominio_novo(200, seed=2); nov_te = frases_dominio_novo(400, seed=3)
base = modelos.MLPLM(vocabulario(ant, nov), ctx=3, d=16, h=64, seed=0); base.treinar(ant, epocas=10, lote=64, lr=0.01)
lora = MLPLMLoRA(base, r=4, alfa=8, seed=0)
antes = (lora.perplexidade(nov_te), lora.perplexidade(ant_te))
lora.treinar(nov, epocas=6, lr=0.005, val=nov_va)
print(f"parâmetros treináveis: {lora.n_treinaveis()} de {lora.n_total_base()} ({lora.n_treinaveis() / lora.n_total_base():.1%})")
print(f"perplexidade no estilo novo : {antes[0]:.3f} -> {lora.perplexidade(nov_te):.3f}")
print(f"perplexidade no estilo antigo: {antes[1]:.3f} -> {lora.perplexidade(ant_te):.3f}  (esquecimento parcial)")


# %% [bloco 19]
casos6 = P.casos_bancada(semente=0)
print(dict(Counter(c["estrato"] for c in casos6)), "| total:", len(casos6))
res_v1 = P.executar_bancada(lambda s: P.SistemaIntegrado(P.CONFIG_V1, semente=s), casos6)
res_v2 = P.executar_bancada(lambda s: P.SistemaIntegrado(P.CONFIG_V2, semente=s), casos6)
print()
print(P.relatorio_automatico("assistente integrado da secretaria (simulado)", P.CONFIG_V2, res_v2, res_v1, "v1 (sem camadas)", "v2 (com camadas)"))


# %% [bloco 20]
print("Portão de regressão: mudar só o limiar de abstenção (0,5 -> 0,6) e comparar com a versão aprovada")
cfg_novo = P.ConfigSistema(limiar=0.6)
res_novo = P.executar_bancada(lambda s: P.SistemaIntegrado(cfg_novo, semente=s), casos6)
for est in ("rag_direta", "rag_parafrase", "rag_sem_resposta", "legitima_bloqueada"):
    a = [r["ok"] for r in res_novo if r["estrato"] == est]; b = [r["ok"] for r in res_v2 if r["estrato"] == est]
    d, lo, hi = bootstrap_pareado(a, b, seed=0)
    print(f"  {est:<20} v2: {np.mean(b):.3f} | limiar 0,6: {np.mean(a):.3f} | diferença {d:+.3f} [{lo:+.3f}; {hi:+.3f}]")
print("\nRobustez: indisponibilidade de 30% do provedor, com e sem nova tentativa (somente consultas RAG):")
for t in (1, 2, 3):
    cfg_t = P.ConfigSistema(p_falha_provedor=0.3, tentativas=t)
    r = [x for x in P.executar_bancada(lambda s: P.SistemaIntegrado(cfg_t, semente=s), [c for c in casos6 if c["estrato"] == "rag_direta"])]
    plano_b = sum("plano_b" in x["bloqueios"] for x in r)
    print(f"  tentativas={t}: respostas ok {np.mean([x['ok'] for x in r]):.3f}; encaminhadas ao atendente (plano B): {plano_b}/{len(r)}")


# %% [bloco 21]
print(json.dumps(P.cartao_sistema(P.CONFIG_V2, res_v2, casos6), ensure_ascii=False, indent=1))


# %% [bloco 22]
print(f"{'atividade':<44}{'responsável':<22}{'aprova':<22}{'frequência'}")
for g in P.plano_governanca():
    print(f"{g['atividade']:<44}{g['responsavel']:<22}{g['aprova']:<22}{g['frequencia']}")
print("\nRegistro de reprodutibilidade:", json.dumps(P.registro_reprodutibilidade(P.CONFIG_V2, 0, casos6), ensure_ascii=False))


# %% [bloco 23]
from llmdidatico.aval_llm import ic_wilson
for n in (20, 40, 60, 80, 100, 150, 200):
    lo, hi = ic_wilson(round(0.95 * n), n)
    print(f"n={n:>3}: taxa {round(0.95 * n) / n:.2f}, IC95% [{lo:.3f}; {hi:.3f}] -> {P.veredito(lo, hi, 0.90)}")


# %% [bloco 24]
for k, n in [(23, 25), (230, 250)]:
    lo, hi = ic_wilson(k, n)
    print(f"{k}/{n}: taxa {k / n:.2f}, IC95% [{lo:.3f}; {hi:.3f}] -> {P.veredito(lo, hi, 0.85)}")


# %% [bloco 25]
conversas, frac_humano, custo_agente = 100_000, 0.5, 0.0006
economia = conversas * frac_humano * custo_agente
print(f"economia mensal: {economia:.1f} unidades hipotéticas de {conversas * custo_agente:.1f} ({frac_humano:.0%})")

