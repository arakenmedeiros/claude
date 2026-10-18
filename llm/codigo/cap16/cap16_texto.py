"""Capítulo 16: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
import math
import numpy as np
from llmdidatico import corpus
from llmdidatico.modelos import NgramaLM

treino = corpus.frases_lm(3000, seed=0)
teste = corpus.frases_lm(500, seed=1)
bi = NgramaLM(n=2, k=0.1).ajustar(treino)
tri = NgramaLM(n=3, k=0.1).ajustar(treino)
print(f"perplexidade por TOKEN (palavra): bigramas {bi.perplexidade(teste):.2f} | trigramas {tri.perplexidade(teste):.2f}")

def texto(f): return " ".join(f)
chars_treino = [list(texto(f)) for f in treino]; chars_teste = [list(texto(f)) for f in teste]
car = NgramaLM(n=5, k=0.05).ajustar(chars_treino)
print(f"perplexidade por CARACTERE: modelo de caracteres (5-gramas) {car.perplexidade(chars_teste):.2f}")

n_car = sum(len(texto(f)) for f in teste)
def bits_por_caractere(lp_nats): return -lp_nats / math.log(2) / n_car
for nome, m, frases in (("bigramas (palavras)", bi, teste), ("trigramas (palavras)", tri, teste), ("5-gramas de caracteres", car, chars_teste)):
    lp = sum(m.logprob_frase(f) for f in frases)
    print(f"{nome:<24} bits por caractere = {bits_por_caractere(lp):.3f}")


# %% [bloco 2]
import numpy as np
from scipy.stats import chi2 as qui
from llmdidatico.aval_llm import n_proporcao, ic_wilson, n_pareado

print("n para estimar uma proporção (95%), por margem de erro:")
print(f"{'p':>5} " + " ".join(f"±{m*100:>4.1f}pp" for m in (0.10, 0.05, 0.03, 0.02)))
for p in (0.5, 0.8, 0.9, 0.95):
    print(f"{p:>5.2f} " + " ".join(f"{n_proporcao(p, m):>7d}" for m in (0.10, 0.05, 0.03, 0.02)))

print("\nIntervalo de Wilson para 'nenhum erro':")
for n in (10, 30, 100, 300):
    lo, hi = ic_wilson(n, n)
    print(f"  {n:>3} acertos em {n:>3}: IC95% [{lo:.3f}; 1,000]")

print("\nn para detectar diferença pareada (McNemar, poder 80%, alfa 5%):")
print(f"{'delta':>6} {'psi=0,10':>9} {'psi=0,20':>9} {'psi=0,30':>9}")
for delta in (0.02, 0.03, 0.05, 0.10):
    print(f"{delta:>6.2f} " + " ".join(f"{n_pareado(delta, psi):>9d}" for psi in (0.10, 0.20, 0.30)))

# verificação por simulação: delta=0,05 e psi=0,20
rng = np.random.default_rng(0)
delta, psi = 0.05, 0.20
n = n_pareado(delta, psi)
cont = rng.multinomial(n, [(psi + delta) / 2, (psi - delta) / 2, 1 - psi], size=5000)
b, c = cont[:, 0], cont[:, 1]
chi2 = (np.abs(b - c) - 1) ** 2 / np.maximum(b + c, 1)
print(f"\nsimulação (5000 repetições): n={n}, poder empírico = {np.mean(qui.sf(chi2, 1) < 0.05):.3f} (alvo 0,80)")
cont0 = rng.multinomial(n, [psi / 2, psi / 2, 1 - psi], size=5000)
b, c = cont0[:, 0], cont0[:, 1]
chi2 = (np.abs(b - c) - 1) ** 2 / np.maximum(b + c, 1)
print(f"sem diferença real: falsos positivos = {np.mean(qui.sf(chi2, 1) < 0.05):.3f} (alvo no máximo 0,05)")


# %% [bloco 3]
import numpy as np
from llmdidatico import corpus
from llmdidatico.aval_llm import tokens

Xdev, ydev = corpus.intencoes(60, seed=5)
Xte, yte = corpus.intencoes(60, seed=11)
def ngr(t, n=4):
    w = tokens(t); return {tuple(w[i:i+n]) for i in range(max(1, len(w)-n+1))}
vistos = set().union(*(ngr(t) for t in Xdev))
def sobreposicao(t):
    g = ngr(t); return len(g & vistos) / len(g)
s = np.array([sobreposicao(t) for t in Xte])
print(f"mensagens de teste com 80% ou mais dos 4-gramas já vistos no desenvolvimento: {np.mean(s >= 0.8):.3f}")
memoria = {" ".join(tokens(t)): r for t, r in zip(Xdev, ydev)}
pred = [memoria.get(" ".join(tokens(t)), "indefinido") for t in Xte]
ok = np.array([p == r for p, r in zip(pred, yte)])
exato = np.array([" ".join(tokens(t)) in memoria for t in Xte])
print(f"mensagens de teste idênticas a alguma do desenvolvimento: {exato.mean():.3f}")
print(f"acerto de um 'memorizador' no teste inteiro: {ok.mean():.3f}")
print(f"  nas mensagens repetidas: {ok[exato].mean():.3f} (n={exato.sum()}) | nas inéditas: {ok[~exato].mean():.3f} (n={(~exato).sum()})")


# %% [bloco 4]
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM, INTENCOES
from llmdidatico.aval_llm import ic_wilson

X, y = corpus.intencoes(60, seed=21)
sim = SimuladorLLM("v3", semente=0)
pred = [json.loads(sim.responder(t))["intencao"] for t in X]
rotulos = INTENCOES + ["indefinido"]
print(classification_report(y, pred, labels=INTENCOES, zero_division=0, digits=3))
print(f"acurácia {accuracy_score(y, pred):.3f} | F1 micro (todos os rótulos) {f1_score(y, pred, average='micro', labels=rotulos):.3f} | "
      f"F1 macro (8 categorias) {f1_score(y, pred, average='macro', labels=INTENCOES):.3f}")
k = sum(a == b for a, b in zip(y, pred)); lo, hi = ic_wilson(k, len(y))
print(f"acurácia com IC de Wilson 95%: {k/len(y):.3f} [{lo:.3f}; {hi:.3f}]")
cm = confusion_matrix(y, pred, labels=rotulos)
print("\nmatriz de confusão (linhas = verdadeiro; colunas = previsto):")
print(" " * 20 + " ".join(f"{r[:5]:>5}" for r in rotulos))
for r, linha in zip(rotulos, cm):
    print(f"{r:<20}" + " ".join(f"{v:>5d}" for v in linha))

y_des = np.array(["normal"] * 950 + ["fraude"] * 50)
sempre_normal = np.array(["normal"] * 1000)
print(f"\ndesbalanceado (5% de 'fraude'), classificador que diz sempre 'normal': acurácia {accuracy_score(y_des, sempre_normal):.3f}, "
      f"F1 macro {f1_score(y_des, sempre_normal, average='macro', zero_division=0):.3f}")


# %% [bloco 5]
import re
from sklearn.metrics import precision_recall_fscore_support
from llmdidatico import DOCS

D = {d[0]: d[3] for d in DOCS}
ids = ["D03", "D07", "D10", "D11", "D12", "D16", "D18", "D59"]
ouro = {
 ("D03", "PRAZO", "6ª semana"),
 ("D10", "PRAZO", "90 dias"), ("D10", "VALOR", "R$ 85,00"),
 ("D11", "PRAZO", "14 dias"), ("D11", "PRAZO", "21 dias"),
 ("D12", "VALOR", "R$ 1,50"), ("D12", "VALOR", "R$ 30,00"),
 ("D16", "VALOR", "R$ 3,00"), ("D16", "VALOR", "R$ 8,00"),
 ("D18", "VALOR", "R$ 400,00"), ("D18", "PRAZO", "um semestre"),
 ("D59", "VALOR", "R$ 12,00"),
}
def extrair(i):
    t = D[i]; s = set()
    for m in re.findall(r"R\$ ?\d[\d.]*,\d{2}", t): s.add((i, "VALOR", m))
    for m in re.findall(r"\d+ (?:dias|semanas|horas|semestres)", t): s.add((i, "PRAZO", m))
    return s
pred = set().union(*(extrair(i) for i in ids))
universo = sorted(ouro | pred)
yt = [int(e in ouro) for e in universo]; yp = [int(e in pred) for e in universo]
p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary")
print(f"entidades (casamento exato): ouro={len(ouro)} previstas={len(pred)} acertos={len(ouro & pred)}")
print(f"precisão {p:.3f} | revocação {r:.3f} | F1 {f:.3f}")
print("falsos positivos:", sorted(pred - ouro))
print("falsos negativos:", sorted(ouro - pred))


# %% [bloco 6]
import numpy as np
from llmdidatico import DOCS
from llmdidatico.modelos import LSAEmbedder
from llmdidatico.aval_llm import correspondencia_exata, f1_tokens, rouge_n, rouge_l, bleu_simples

ref = "O atraso na devolução gera multa de R$ 1,50 por dia e por exemplar."
candidatos = {
 "paráfrase correta":   "A multa por atraso é de R$ 1,50 por dia, para cada exemplar.",
 "número errado":       "O atraso na devolução gera multa de R$ 5,00 por dia e por exemplar.",
 "incompleta":          "Há multa por atraso.",
 "irrelevante":         "O estudante pode renovar o livro até 3 vezes.",
 "cópia literal":       ref,
}
emb = LSAEmbedder(dim=40).ajustar([d[2] + ". " + d[3] for d in DOCS])
zr = emb.transformar([ref])[0]
print(f"{'candidato':<18} {'EM':>4} {'F1tok':>6} {'R1-F':>6} {'RL-F':>6} {'BLEU2':>6} {'cos(LSA)':>9}")
for nome, c in candidatos.items():
    cos = float(emb.transformar([c])[0] @ zr)
    print(f"{nome:<18} {correspondencia_exata(ref, c):>4.0f} {f1_tokens(ref, c):>6.2f} {rouge_n(ref, c)['f']:>6.2f} "
          f"{rouge_l(ref, c)['f']:>6.2f} {bleu_simples(ref, c):>6.2f} {cos:>9.2f}")


# %% [bloco 7]
import re
import numpy as np
from llmdidatico import DOCS, CONSULTAS, avaliacao as av
from llmdidatico.modelos import BM25, tokenizar_sem_stop
from llmdidatico.aval_llm import tokens

ids = [d[0] for d in DOCS]; textos = [f"{d[2]}. {d[3]}" for d in DOCS]; por_id = {d[0]: d[3] for d in DOCS}
bm = BM25([tokenizar_sem_stop(t) for t in textos])
def recuperar(q, k=3):
    s = bm.pontuar(tokenizar_sem_stop(q)); o = np.argsort(-s, kind="stable")[:k]
    return [ids[i] for i in o], [float(s[i]) for i in o]

TIPOS = ["direta", "parafrase", "sem_resposta"]
def por_tipo(tipo): return [c for c in CONSULTAS if c[3] == tipo]

print("1) RECUPERAÇÃO (BM25 do pacote), por tipo de consulta")
print(f"{'tipo':<13} {'n':>3} {'hit@1':>6} {'hit@3':>6} {'recall@3':>9} {'MRR':>6}")
for tp in TIPOS[:2]:
    cs = por_tipo(tp); rk = [recuperar(c[0], 10)[0] for c in cs]
    h1 = np.mean([av.hit_at_k(r, c[1], 1) for r, c in zip(rk, cs)]); h3 = np.mean([av.hit_at_k(r, c[1], 3) for r, c in zip(rk, cs)])
    r3 = np.mean([av.recall_at_k(r, c[1], 3) for r, c in zip(rk, cs)]); mr = np.mean([av.mrr(r, c[1]) for r, c in zip(rk, cs)])
    print(f"{tp:<13} {len(cs):>3} {h1:>6.2f} {h3:>6.2f} {r3:>9.2f} {mr:>6.2f}")

def sentencas(t): return [s.strip() for s in re.split(r"(?<=[.;])\s+", t) if s.strip()]
def gerar(q, ctx_ids, escore_top, tau, ruido=0.0, rng=None):
    """GERADOR EXTRATIVO SIMULADO (não é um LLM). Devolve (resposta, foi_alterada)."""
    if escore_top < tau: return "não consta", False
    qt = set(tokenizar_sem_stop(q)); melhor, bs = "", -1
    for r, i in enumerate(ctx_ids):
        for s in sentencas(por_id[i]):
            sc = len(qt & set(tokenizar_sem_stop(s))) - 0.01 * r
            if sc > bs: melhor, bs = s, sc
    alterada = bool(ruido) and rng.random() < ruido and bool(re.search(r"\d", melhor))
    if alterada:
        melhor = re.sub(r"\d+", lambda m: str(int(m.group()) + 7), melhor, count=1)
    return melhor, alterada
def contem(trecho, resp): return " ".join(tokens(trecho)) in " ".join(tokens(resp))
def numeros_ok(resp, ctx_ids):
    ctx = " ".join(" ".join(tokens(por_id[i])) for i in ctx_ids)
    return all(n in ctx.split() for n in re.findall(r"\d+(?:[.,]\d+)*", " ".join(tokens(resp))))

def avaliar(tau, ruido=0.0, seed=0, k=3):
    rng = np.random.default_rng(seed); linhas = []
    for q, rel, trecho, tp in CONSULTAS:
        ctx, sc = recuperar(q, k)
        resp, alt = gerar(q, ctx, sc[0], tau, ruido, rng)
        if tp == "sem_resposta": ok, rec = resp == "não consta", None
        else: ok, rec = contem(trecho, resp), bool(set(ctx) & set(rel))
        linhas.append(dict(tipo=tp, rec=rec, ok=ok, alt=alt, q=q, trecho=trecho, resp=resp,
                           fiel=numeros_ok(resp, ctx) if resp != "não consta" else True))
    return linhas

L = avaliar(tau=4.5)
print("\n2) GERAÇÃO extrativa SIMULADA com abstenção (limiar 4,5 no escore BM25)")
print(f"{'tipo':<13} {'n':>3} {'recuperou(top3)':>16} {'resposta correta':>17} {'correta | recuperou':>20}")
for tp in TIPOS:
    l = [x for x in L if x["tipo"] == tp]
    if tp == "sem_resposta":
        print(f"{tp:<13} {len(l):>3} {'-':>16} {np.mean([x['ok'] for x in l]):>17.2f} {'(= recusou)':>20}")
    else:
        rec = [x for x in l if x["rec"]]
        print(f"{tp:<13} {len(l):>3} {np.mean([x['rec'] for x in l]):>16.2f} {np.mean([x['ok'] for x in l]):>17.2f} {np.mean([x['ok'] for x in rec]):>20.2f}")
print(f"abstenções indevidas em consultas respondíveis: {sum(x['tipo']!='sem_resposta' and x['resp']=='não consta' for x in L)} de {sum(x['tipo']!='sem_resposta' for x in L)}")

print("\n3) fidelidade: gerador com 'alucinação' simulada (15% das respostas com um número alterado)")
Lr = avaliar(4.5, ruido=0.15, seed=3)
alt = [x for x in Lr if x["alt"]]
print(f"acerto: sem ruído {np.mean([x['ok'] for x in L if x['tipo']!='sem_resposta']):.2f} -> com ruído {np.mean([x['ok'] for x in Lr if x['tipo']!='sem_resposta']):.2f}")
print(f"respostas com número alterado: {len(alt)}; sinalizadas pelo verificador de números: {sum(not x['fiel'] for x in alt)}")
sem_alt = [x for x in Lr if not x["alt"] and x["resp"] != "não consta"]
print(f"falsos alarmes do verificador: {sum(not x['fiel'] for x in sem_alt)} de {len(sem_alt)}")
falha_ger = [x for x in L if x["tipo"] == "direta" and x["rec"] and not x["ok"] and x["resp"] != "não consta"]
print("exemplos de falha de GERAÇÃO com o documento certo recuperado:")
for x in falha_ger[:2]: print("  P:", x["q"], "| esperado:", x["trecho"], "| resposta:", x["resp"][:70])

print("\n4) varredura do limiar de abstenção")
print(f"{'tau':>5} {'recusa correta (sem resp.)':>27} {'recusa indevida (respondíveis)':>31}")
for tau in (0, 2, 3, 4.5, 6, 8, 10):
    Lt = avaliar(tau); sr = [x for x in Lt if x["tipo"] == "sem_resposta"]; rs = [x for x in Lt if x["tipo"] != "sem_resposta"]
    print(f"{tau:>5} {np.mean([x['ok'] for x in sr]):>27.2f} {np.mean([x['resp']=='não consta' for x in rs]):>31.2f}")


# %% [bloco 8]
import numpy as np
from llmdidatico.aval_llm import JuizEnviesado

rng = np.random.default_rng(7)
def resposta(q, palavras=None, estilo=None):
    return {"qualidade": q, "palavras": int(palavras if palavras is not None else np.clip(60 + 12 * q + rng.normal(0, 20), 15, 300)),
            "estilo": estilo or rng.choice(["juiz", "outro"])}

iguais = [(resposta(2, 80, "outro"), resposta(2, 80, "outro")) for _ in range(500)]
print("VIÉS DE POSIÇÃO (juiz SIMULADO; pares de qualidade idêntica; n=500)")
print(f"{'viés de posição':>16} {'prefere a 1ª (sem troca)':>26} {'empates (com troca)':>20}")
for vp in (0.0, 0.3, 0.6, 1.0):
    j = JuizEnviesado(vies_posicao=vp, semente=1)
    primeira = np.mean([j.comparar(a, b) == 1 for a, b in iguais])
    j2 = JuizEnviesado(vies_posicao=vp, semente=2)
    emp = np.mean([j2.comparar_protocolo(a, b, True) == "empate" for a, b in iguais])
    print(f"{vp:>16.1f} {primeira:>26.3f} {emp:>20.3f}")

desiguais = []
while len(desiguais) < 600:
    a, b = resposta(int(rng.integers(0, 4))), resposta(int(rng.integers(0, 4)))
    if a["qualidade"] != b["qualidade"]: desiguais.append((a, b))
humano = np.array(["a" if a["qualidade"] > b["qualidade"] else "b" for a, b in desiguais])
humano = np.where(rng.random(len(humano)) < 0.06, np.where(humano == "a", "b", "a"), humano)
print("\nCONCORDÂNCIA COM HUMANOS (600 pares com qualidades diferentes; viés de posição = 0,8)")
for nome, troca in (("sem troca de ordem", False), ("com troca de ordem", True)):
    j = JuizEnviesado(vies_posicao=0.8, semente=3)
    v = np.array([j.comparar_protocolo(a, b, troca) for a, b in desiguais])
    dec = v != "empate"
    print(f"{nome:<20} decididos {dec.mean():.3f} | concordância com humanos nos decididos {np.mean(v[dec] == humano[dec]):.3f} | "
          f"contando empate como erro {np.mean(v == humano):.3f}")

pares_curto = []
for _ in range(500):
    q = int(rng.integers(1, 4))
    pares_curto.append((resposta(q, 60, "outro"), resposta(q - 1, 200, "outro")))
print("\nVIÉS DE VERBOSIDADE (a melhor resposta é a mais curta; n=500)")
for vv in (0.0, 0.5, 1.0):
    j = JuizEnviesado(vies_verbosidade=vv, semente=4)
    print(f"  viés {vv:.1f}: o juiz escolhe a melhor (curta) em {np.mean([j.comparar(a, b) == 1 for a, b in pares_curto]):.3f} dos pares")

mistos = [(resposta(2, 80, "juiz"), resposta(2, 80, "outro")) for _ in range(500)]
for va in (0.0, 0.5):
    j = JuizEnviesado(vies_autopreferencia=va, vies_posicao=0, semente=5)
    print(f"AUTOPREFERÊNCIA {va:.1f}: preferiu a resposta de estilo 'juiz' em {np.mean([j.comparar(a, b) == 1 for a, b in mistos]):.3f} dos pares de mesma qualidade")


# %% [bloco 9]
import itertools
import numpy as np
from sklearn.metrics import cohen_kappa_score
from llmdidatico.aval_llm import kappa_cohen, alfa_krippendorff

rng = np.random.default_rng(11)
N = 150
verdade = rng.integers(0, 4, N)
def anotador(pexato):
    r = verdade.copy(); err = rng.random(N) > pexato
    r[err] = np.clip(r[err] + rng.choice([-1, 1], err.sum()), 0, 3); return r
H = [anotador(0.80), anotador(0.75), anotador(0.70)]
print("kappa de Cohen entre avaliadores humanos (SIMULADOS): própria x scikit-learn")
for (i, a), (j, b) in itertools.combinations(enumerate(H), 2):
    kq = cohen_kappa_score(a, b, weights="quadratic")
    print(f"  H{i+1}-H{j+1}: kappa {kappa_cohen(a, b):.3f} (sklearn {cohen_kappa_score(a, b):.3f}) | ponderado quadrático {kq:.3f} | concordância bruta {np.mean(a == b):.3f}")
assert all(abs(kappa_cohen(a, b) - cohen_kappa_score(a, b)) < 1e-12 for a, b in itertools.combinations(H, 2))

dados = [list(map(int, h)) for h in H]
print(f"\nalfa de Krippendorff (3 avaliadores, sem faltantes): nominal {alfa_krippendorff(dados):.3f} | ordinal {alfa_krippendorff(dados, 'ordinal'):.3f} | intervalo {alfa_krippendorff(dados, 'intervalo'):.3f}")
faltantes = [[v if rng.random() > 0.2 else None for v in lin] for lin in dados]
print(f"com ~20% de anotações faltantes:  nominal {alfa_krippendorff(faltantes):.3f} | ordinal {alfa_krippendorff(faltantes, 'ordinal'):.3f}")

def alfa_por_definicao(dados, delta):
    us = [[r[u] for r in dados if r[u] is not None] for u in range(len(dados[0]))]; us = [u for u in us if len(u) >= 2]
    n = sum(len(u) for u in us)
    do = sum(delta(x, y) / (len(u) - 1) for u in us for x, y in itertools.permutations(u, 2)) / n
    todos = [v for u in us for v in u]
    de = sum(delta(x, y) for x, y in itertools.permutations(todos, 2)) / (n * (n - 1))
    return 1 - do / de
print(f"conferência por definição (força bruta): nominal {alfa_por_definicao(faltantes, lambda x, y: float(x != y)):.3f} | intervalo {alfa_por_definicao(faltantes, lambda x, y: (x - y) ** 2):.3f} "
      f"(pacote, intervalo: {alfa_krippendorff(faltantes, 'intervalo'):.3f})")

a = np.array([1] * 95 + [0] * 5); b = a.copy(); b[:2] = 0; b[95:97] = 1
print(f"\nparadoxo do kappa: concordância {np.mean(a == b):.2f}, mas kappa {kappa_cohen(a, b):.3f} (prevalência de 95%)")


# %% [bloco 10]
import numpy as np
from llmdidatico.aval_llm import JuizEnviesado, kappa_cohen

rng = np.random.default_rng(5)
M = 1000
q = rng.integers(0, 4, M); palavras = np.clip(60 + 12 * q + rng.normal(0, 20, M), 15, 300).astype(int)
resps = [{"qualidade": int(a), "palavras": int(b), "estilo": "outro"} for a, b in zip(q, palavras)]
juiz = JuizEnviesado(vies_verbosidade=0.6, ruido=0.7, semente=9)
notas = np.array([juiz.nota(r) for r in resps])
verd = (q >= 2).astype(int)
aprov_juiz = (notas >= 1.5).astype(int)
print(f"CALIBRAÇÃO (juiz simulado; {M} respostas, nota de corte 1,5)")
print(f"taxa de aprovação segundo o juiz: {aprov_juiz.mean():.3f} | taxa de aprovação verdadeira (oráculo da simulação): {verd.mean():.3f}")
idx = rng.choice(M, 200, replace=False)
hum = verd[idx]; jz = aprov_juiz[idx]
sens = np.mean(jz[hum == 1]); esp = np.mean(1 - jz[hum == 0])
print(f"nos 200 casos rotulados: sensibilidade {sens:.3f} | especificidade {esp:.3f} | kappa juiz-humano {kappa_cohen(hum, jz):.3f}")
def rogan_gladen(p_obs, sens, esp): return (p_obs + esp - 1) / (sens + esp - 1)
est = rogan_gladen(aprov_juiz.mean(), sens, esp)
bs = []
for _ in range(2000):
    s = rng.choice(len(idx), len(idx)); h, j = hum[s], jz[s]
    if h.min() == h.max(): continue
    se, es = np.mean(j[h == 1]), np.mean(1 - j[h == 0])
    if se + es - 1 > 0.2: bs.append(rogan_gladen(aprov_juiz.mean(), se, es))
print(f"taxa de aprovação corrigida: {est:.3f} (IC95% bootstrap [{np.quantile(bs, .025):.3f}; {np.quantile(bs, .975):.3f}])")


# %% [bloco 11]
import json, re
import numpy as np
from math import comb
from llmdidatico import corpus, avaliacao as av
from llmdidatico.simulador import SimuladorLLM
from llmdidatico.aval_llm import mcnemar, bootstrap_pareado, permutacao_pareada, holm, bonferroni, benjamini_hochberg

X, y = corpus.intencoes(100, seed=31)
def acertos(versao, semente=0):
    sim = SimuladorLLM(versao, semente=semente); out = np.zeros(len(X))
    for i in range(len(X)):
        r = sim.responder(X[i])
        if versao == "v3": pred = json.loads(r)["intencao"]
        else:
            m = re.fullmatch(r"rotulo: (\w+)", r.strip()); pred = m.group(1) if m else None
        out[i] = float(pred == y[i])
    return out
a2, a3 = acertos("v2"), acertos("v3")
print(f"acerto v2 {a2.mean():.3f} | v3 {a3.mean():.3f} | n={len(X)}")
m = mcnemar(a3, a2)
print(f"McNemar v3 vs v2: b={m['b']} (v3 certo, v2 errado), c={m['c']} (v3 errado, v2 certo); p exato {m['p_exato']:.2e}; chi2 {m['chi2']:.1f}, p {m['p_chi2']:.2e}")
d = m['b'] + m['c']; k = min(m['b'], m['c'])
p_manual = min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / 2 ** d)
print(f"verificação manual do p exato: {p_manual:.2e} (confere: {abs(p_manual - m['p_exato']) < 1e-12})")
mu, lo, hi = bootstrap_pareado(a3, a2, seed=0)
print(f"bootstrap pareado da diferença: {mu:+.3f} IC95% [{lo:+.3f}; {hi:+.3f}] | permutação pareada p = {permutacao_pareada(a3, a2):.4f}")
s3 = av.bootstrap_ic(a3, seed=1); s2 = av.bootstrap_ic(a2, seed=2)
print(f"ICs separados: v3 [{s3[1]:.3f}; {s3[2]:.3f}], v2 [{s2[1]:.3f}; {s2[2]:.3f}]")

print("\nMÚLTIPLAS COMPARAÇÕES (simulação): 99 variantes equivalentes à v2 + a v3, realmente melhor")
rng = np.random.default_rng(0); variantes = []
for _ in range(99):
    v = a2.copy(); mask = rng.random(len(v)) < 0.12; v[mask] = rng.random(mask.sum()) < a2.mean(); variantes.append(v)
variantes.append(a3)
ps = np.array([mcnemar(v, a2)["p_exato"] for v in variantes])
for nome, aj in (("sem correção", ps), ("Bonferroni", np.array(bonferroni(ps))), ("Holm", np.array(holm(ps))), ("Benjamini-Hochberg", np.array(benjamini_hochberg(ps)))):
    print(f"  {nome:<19}: {int(np.sum(aj[:99] < 0.05)):>2} falsos positivos entre 99 | a variante realmente melhor (v3) detectada: {bool(aj[99] < 0.05)}")


# %% [bloco 12]
import json, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from llmdidatico import corpus
from llmdidatico.modelos import amostrar
from llmdidatico.simulador import SimuladorLLM

X, y = corpus.intencoes(100, seed=31)
def acerto_v2(semente, ordem):
    sim = SimuladorLLM("v2", semente=semente); ok = np.zeros(len(X))
    for i in ordem:
        m = re.fullmatch(r"rotulo: (\w+)", sim.responder(X[i]).strip()); ok[i] = float(bool(m) and m.group(1) == y[i])
    return ok.mean()
base = np.arange(len(X))
accs = np.array([acerto_v2(s, base) for s in range(20)])
print(f"v2 em 20 sementes: média {accs.mean():.3f}, desvio-padrão {accs.std(ddof=1):.4f}, mín {accs.min():.3f}, máx {accs.max():.3f}")
rng = np.random.default_rng(0)
ords = np.array([acerto_v2(0, rng.permutation(len(X))) for _ in range(20)])
print(f"v2, semente fixa, 20 ordens dos casos: média {ords.mean():.3f}, desvio {ords.std(ddof=1):.4f}, mín {ords.min():.3f}, máx {ords.max():.3f}")
print(f"desvio-padrão do sorteio binomial para n={len(X)}, p=0,8: {np.sqrt(0.8 * 0.2 / len(X)):.4f}")

Xtr, ytr = corpus.intencoes(40, seed=1)
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(Xtr)
clf = LogisticRegression(max_iter=300).fit(vec.transform(Xtr), ytr)
logits = clf.decision_function(vec.transform(X)); cls = list(clf.classes_)
yv = np.array([cls.index(r) for r in y])
print(f"\nclassificador de apoio (NÃO é LLM): acerto com argmax (T→0) = {np.mean(logits.argmax(1) == yv):.3f}")
print(f"{'T':>4} {'média':>7} {'dp':>7} {'mín':>7} {'máx':>7}   (20 execuções, {len(y)} casos)")
for T in (0.25, 0.5, 1.0, 2.0):
    acc = []
    for k in range(20):
        r = np.random.default_rng(100 + k)
        acc.append(np.mean(np.array([amostrar(l, r, T=T) for l in logits]) == yv))
    print(f"{T:>4} {np.mean(acc):>7.3f} {np.std(acc, ddof=1):>7.4f} {np.min(acc):>7.3f} {np.max(acc):>7.3f}")


# %% [bloco 13]
import json
import numpy as np
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM
from llmdidatico.aval_llm import bootstrap_pareado

X, y = corpus.intencoes(100, seed=31)
sim = SimuladorLLM("v3", semente=0)
pred = [json.loads(sim.responder(t))["intencao"] for t in X]
base = np.array([float(p == r) for p, r in zip(pred, y)])

def portao(base_ok, novo_ok, estratos, margem=0.03, queda_estrato=0.10, seed=0):
    """Reprova se (1) o limite inferior do IC da diferença pareada (novo - base) passa de -margem ou (2) algum estrato cai mais que queda_estrato."""
    m, lo, hi = bootstrap_pareado(novo_ok, base_ok, n=2000, seed=seed)
    motivos = []
    if lo < -margem: motivos.append(f"IC da diferença [{lo:+.3f}; {hi:+.3f}] passa de -{margem}")
    for e in sorted(set(estratos)):
        idx = np.array([s == e for s in estratos])
        d = novo_ok[idx].mean() - base_ok[idx].mean()
        if d < -queda_estrato: motivos.append(f"estrato '{e}' caiu {d:+.2f}")
    return (not motivos), m, motivos

rng = np.random.default_rng(3)
def degradar(ok, frac, alvo=None):
    novo = ok.copy(); cand = np.where(novo == 1)[0] if alvo is None else np.array([i for i in np.where(novo == 1)[0] if y[i] == alvo])
    novo[rng.choice(cand, int(frac * len(cand)), replace=False)] = 0; return novo
cenarios = {"idêntico": base.copy(), "queda leve (-1%, difusa)": degradar(base, 0.01), "queda difusa (-5%)": degradar(base, 0.05),
            "queda concentrada em 'elogio'": degradar(base, 0.12, "elogio")}
print(f"{'cenário':<32} {'diferença':>9}  resultado")
for nome, novo in cenarios.items():
    ok, d, mot = portao(base, novo, y)
    print(f"{nome:<32} {d:>+9.3f}  {'APROVADO' if ok else 'REPROVADO'} {'; '.join(mot)}")

rng = np.random.default_rng(0); n_sem = 300
taxa = [0.08] * 8 + [0.15] * 4
contagem = rng.binomial(n_sem, taxa)
p_ref = contagem[:6].sum() / (6 * n_sem); dp = np.sqrt(p_ref * (1 - p_ref) / n_sem)
print(f"\nMONITORAMENTO (dados simulados): abstenção de referência (semanas 1-6) = {p_ref:.3f}; limites 3 desvios [{max(0, p_ref - 3 * dp):.3f}; {p_ref + 3 * dp:.3f}]")
for s, c in enumerate(contagem, 1):
    p = c / n_sem; print(f"  semana {s:>2}: {p:.3f} {'<-- ALERTA' if p > p_ref + 3 * dp or p < p_ref - 3 * dp else ''}")

def amostra_revisao(registros, n_aleatorio=20, limiar_conf=0.6, seed=0):
    rg = np.random.default_rng(seed)
    baixos = [i for i, r in enumerate(registros) if r["conf"] < limiar_conf]
    resto = [i for i in range(len(registros)) if i not in set(baixos)]
    return baixos, sorted(rg.choice(resto, min(n_aleatorio, len(resto)), replace=False).tolist())
registros = [{"conf": float(c)} for c in np.random.default_rng(1).beta(5, 2, 300)]
b, a = amostra_revisao(registros)
print(f"\nrevisão semanal: {len(b)} casos de baixa confiança (todos) + {len(a)} aleatórios = {len(b) + len(a)} de {len(registros)}")


# %% [bloco 14]
import re
import numpy as np
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM
from llmdidatico.aval_llm import ic_wilson

X, y = corpus.intencoes(100, seed=31)
sim = SimuladorLLM("v2", semente=0)
def lab(r):
    m = re.fullmatch(r"rotulo: (\w+)", r.strip()); return m.group(1) if m else None
ok = np.array([lab(sim.responder(t)) == r for t, r in zip(X, y)])
print(f"acerto geral (v2, formato estrito): {ok.mean():.3f}")
comp = np.array([len(t.split()) for t in X]); cortes = np.quantile(comp, [1/3, 2/3])
fatia_c = np.where(comp <= cortes[0], "curta", np.where(comp <= cortes[1], "média", "longa"))
saudacao = np.array([bool(re.match(r"(olá|oi|bom dia|boa tarde|por favor)", t.lower())) for t in X])
fatias = {f"categoria={c}": np.array([r == c for r in y]) for c in sorted(set(y))}
fatias.update({f"tamanho={t}": fatia_c == t for t in ("curta", "média", "longa")})
fatias.update({"com saudação": saudacao, "sem saudação": ~saudacao})
linhas = []
for nome, msk in fatias.items():
    k, n = int(ok[msk].sum()), int(msk.sum()); lo, hi = ic_wilson(k, n); linhas.append((k / n, lo, hi, n, nome))
print(f"{'fatia':<30} {'n':>4} {'acerto':>7} {'IC95% Wilson':>16}")
for p, lo, hi, n, nome in sorted(linhas):
    print(f"{nome:<30} {n:>4} {p:>7.3f} {f'[{lo:.3f}; {hi:.3f}]':>16}{'  <- IC todo abaixo do geral' if hi < ok.mean() else ''}")


# %% [bloco 15]
import hashlib, json, re, datetime
import numpy as np
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM
from llmdidatico.aval_llm import ic_wilson, bootstrap_pareado

X, y = corpus.intencoes(100, seed=31)
def acertos(versao, semente=0):
    sim = SimuladorLLM(versao, semente=semente); out = []
    for t, r in zip(X, y):
        resp = sim.responder(t)
        if versao == "v3": pred = json.loads(resp)["intencao"]
        else:
            m = re.fullmatch(r"rotulo: (\w+)", resp.strip()); pred = m.group(1) if m else None
        out.append(float(pred == r))
    return np.array(out)
a2, a3 = acertos("v2"), acertos("v3")
k = int(a3.sum()); lo, hi = ic_wilson(k, len(a3))
m, dlo, dhi = bootstrap_pareado(a3, a2, seed=0)
cartao = {
    "data": datetime.date.today().isoformat(),
    "sistema": {"nome": "SimuladorLLM v3 (regras; NÃO é um LLM)", "semente": 0, "temperatura": None},
    "linha_de_base": "SimuladorLLM v2 (semente 0)",
    "teste": {"origem": "corpus.intencoes(100, seed=31)", "n": len(X), "sha1_conteudo": hashlib.sha1("\n".join(X).encode("utf-8")).hexdigest()[:12]},
    "metricas": {"acerto": round(k / len(a3), 3), "ic95_wilson": [round(lo, 3), round(hi, 3)],
                 "diferenca_vs_base": round(m, 3), "ic95_diferenca": [round(dlo, 3), round(dhi, 3)]},
    "limitacoes": ["dados sintéticos de poucos modelos de frase", "sem avaliação humana", "uma única semente"],
}
print(json.dumps(cartao, ensure_ascii=False, indent=2))


# %% [bloco 16]
import re
import numpy as np
from sklearn.metrics import cohen_kappa_score
from llmdidatico.aval_llm import tokens

FRASES = {
 "c1": {"literal": ["a frequência mínima é de 75% das aulas", "é preciso comparecer a, no mínimo, 75% das aulas"],
        "parafrase": ["o aluno precisa ir a pelo menos três quartos das aulas"], "errada": ["a frequência mínima é de 50% das aulas"]},
 "c2": {"literal": ["a média final deve ser igual ou superior a 6,0", "é preciso média 6,0 ou mais"],
        "parafrase": ["a nota final precisa chegar a seis pontos"], "errada": ["a média final mínima é 7,0"]},
 "c3": {"literal": ["quem tem média entre 4,0 e 5,9 faz prova final", "com média de 4,0 a 5,9 há prova final"],
        "parafrase": ["os que ficam na faixa intermediária de notas vão para um exame de recuperação"], "errada": ["quem tem média abaixo de 6,0 é reprovado direto"]},
}
rng = np.random.default_rng(21)
respostas, ouro = [], []
for _ in range(120):
    partes, flags = [], []
    for c in ("c1", "c2", "c3"):
        u = rng.random()
        if u < 0.50: partes.append(rng.choice(FRASES[c]["literal"])); flags.append(1)
        elif u < 0.70: partes.append(rng.choice(FRASES[c]["parafrase"])); flags.append(1)
        elif u < 0.85: partes.append(rng.choice(FRASES[c]["errada"])); flags.append(0)
        else: flags.append(0)
    respostas.append(", e ".join(partes).capitalize() + "." if partes else "Não sei explicar."); ouro.append(flags)
ouro = np.array(ouro)
prof2 = ouro.copy(); mask = rng.random(prof2.shape) < 0.06; prof2[mask] = 1 - prof2[mask]

def corretor_regras(txt):
    t = " ".join(tokens(txt))
    c1 = int(bool(re.search(r"\b75\b", t))); c2 = int(bool(re.search(r"\b6,0\b", t)) and "7,0" not in t)
    c3 = int("4,0" in t and "5,9" in t)
    return [c1, c2, c3]
auto = np.array([corretor_regras(r) for r in respostas])
print(f"{'critério':<9} {'prof1-prof2 (kappa)':>20} {'prof1-regras (kappa)':>22} {'concordância bruta':>19}")
for k, c in enumerate(("c1", "c2", "c3")):
    print(f"{c:<9} {cohen_kappa_score(ouro[:, k], prof2[:, k]):>20.3f} {cohen_kappa_score(ouro[:, k], auto[:, k]):>22.3f} {np.mean(ouro[:, k] == auto[:, k]):>19.3f}")
tot_h, tot_2, tot_a = ouro.sum(1), prof2.sum(1), auto.sum(1)
print(f"nota total (0-3): kappa ponderado quadrático prof1-prof2 {cohen_kappa_score(tot_h, tot_2, weights='quadratic'):.3f} | prof1-regras {cohen_kappa_score(tot_h, tot_a, weights='quadratic'):.3f}")
print(f"erro médio absoluto (regras - prof1): {np.mean(np.abs(tot_a - tot_h)):.3f} | viés médio (regras - prof1): {np.mean(tot_a - tot_h):+.3f}")
tem_par = np.array([any(f in r.lower() for k in FRASES for f in FRASES[k]["parafrase"]) for r in respostas])
print(f"respostas com paráfrase: {tem_par.sum()} | divergência total com paráfrase: {np.mean(tot_a[tem_par] != tot_h[tem_par]):.3f}, sem paráfrase: {np.mean(tot_a[~tem_par] != tot_h[~tem_par]):.3f}")

TEMAS = [r"aulas|frequencia", r"media final|nota final|seis", r"prova final|exame|recuperacao|faixa|4,0"]
def revisar(txt, a):
    t = " ".join(tokens(txt)); return any(re.search(TEMAS[k], t) and a[k] == 0 for k in range(3))
rev = np.array([revisar(r, a) for r, a in zip(respostas, auto)])
dif = tot_a != tot_h
print(f"política de revisão: encaminhar {rev.mean():.3f} das respostas; cobre {np.mean(rev[dif]):.3f} das divergências; {np.sum(~rev & dif)} divergências passariam sem revisão")

