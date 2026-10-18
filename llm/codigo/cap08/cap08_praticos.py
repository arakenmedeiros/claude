"""Capítulo 8: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 8.1: Avaliando três versões de prompt com o simulador]
import json, re
import numpy as np
from sklearn.metrics import classification_report
from llmdidatico import corpus, avaliacao as av
from llmdidatico.simulador import SimuladorLLM, INTENCOES

def interpretar(versao, resposta):
    if versao == "v3":
        try:
            o = json.loads(resposta); return (o["intencao"], True)
        except Exception:
            return (None, False)
    if versao == "v2":
        m = re.fullmatch(r"rotulo: (\w+)", resposta.strip())
        return (m.group(1), True) if m else (None, False)
    achados = [c for c in INTENCOES if c.replace("_", " ") in resposta]
    return (achados[0], False) if achados else (None, False)

X, y = corpus.intencoes(60, seed=21)
preds3 = None
print(f"{'versão':<7} {'acerto':>7} {'IC95%':>18} {'formato válido':>15}")
for v in ("v1", "v2", "v3"):
    sim = SimuladorLLM(v, semente=0)
    saidas = [interpretar(v, sim.responder(t)) for t in X]
    acertos = [float(p == r) for (p, _), r in zip(saidas, y)]
    m, lo, hi = av.bootstrap_ic(acertos, n=1000, seed=0)
    print(f"{v:<7} {m:>7.3f} {f'[{lo:.3f}; {hi:.3f}]':>18} {np.mean([ok for _, ok in saidas]):>15.3f}")
    if v == "v3":
        preds3 = [p if p else "indefinido" for p, _ in saidas]
print()
print(classification_report(y, preds3, labels=INTENCOES, zero_division=0, digits=2))


# %% [Exercício 8.2: Quantos casos de teste são suficientes?]
import re
import numpy as np
from llmdidatico import corpus, avaliacao as av
from llmdidatico.simulador import SimuladorLLM

X, y = corpus.intencoes(100, seed=31)
sim = SimuladorLLM("v2", semente=0)
def rotulo(txt):
    m = re.fullmatch(r"rotulo: (\w+)", txt.strip()); return m.group(1) if m else None
acertos = np.array([float(rotulo(sim.responder(t)) == r) for t, r in zip(X, y)])
p = acertos.mean()
print(f"acerto da v2 em {len(X)} mensagens: {p:.3f}")

rng = np.random.default_rng(0)
print(f"{'n':>5} {'largura IC (bootstrap)':>23} {'largura (fórmula)':>18}")
for n in (50, 100, 200, 400, 800):
    larguras = []
    for rep in range(20):
        idx = rng.choice(len(acertos), n, replace=False)
        _, lo, hi = av.bootstrap_ic(acertos[idx], n=500, seed=rep)
        larguras.append(hi - lo)
    aprox = 2 * 1.96 * np.sqrt(p * (1 - p) / n)
    print(f"{n:>5} {np.mean(larguras):>23.3f} {aprox:>18.3f}")
n_necessario = int(np.ceil((1.96 / 0.015) ** 2 * p * (1 - p)))
print(f"casos para uma margem de ±1,5 p.p. (largura 3 p.p.), por candidato: {n_necessario}")


# %% [Exercício 8.3: Tolerância de formato e o que se mede]
import re
import numpy as np
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM

X, y = corpus.intencoes(100, seed=41)
sim = SimuladorLLM("v2", semente=0)
brutas = [sim.responder(t) for t in X]

estrito = lambda s: (re.fullmatch(r"rotulo: (\w+)", s.strip()) or [None, None])[1]
tolerante = lambda s: (re.search(r"rotulo: (\w+)", s) or [None, None])[1]

ae = np.mean([estrito(s) == r for s, r in zip(brutas, y)])
at = np.mean([tolerante(s) == r for s, r in zip(brutas, y)])
rejeitadas = [(s, r) for s, r in zip(brutas, y) if estrito(s) is None]
corretas_rej = sum(tolerante(s) == r for s, r in rejeitadas)
print(f"acerto com analisador estrito   : {ae:.3f}")
print(f"acerto com analisador tolerante : {at:.3f}")
print(f"respostas rejeitadas pelo estrito: {len(rejeitadas)} de {len(brutas)} ({len(rejeitadas) / len(brutas):.1%}); dessas, corretas no conteúdo: {corretas_rej}")
print("exemplos de rejeitadas:", [s for s, _ in rejeitadas[:3]])


# %% [Projeto: Classificador com validação, verificação de evidência e plano B]
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from llmdidatico import corpus, avaliacao as av
from llmdidatico.simulador import SimuladorLLM, INTENCOES

X, y = corpus.intencoes(60, seed=51)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
plano_b = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode"), LogisticRegression(max_iter=1000)).fit(Xtr[:80], ytr[:80])
sim = SimuladorLLM("v3", semente=0)

def chamar_modelo(texto):
    bruto = sim.responder(texto)
    try:
        o = json.loads(bruto)
    except json.JSONDecodeError:
        return None, "json_invalido"
    if o.get("intencao") not in INTENCOES + ["indefinido"]:
        return None, "valor_invalido"
    if o["intencao"] != "indefinido" and o.get("evidencia", "") not in texto:
        return None, "evidencia_inventada"
    return o["intencao"], "ok"

def classificar(texto, usar_plano_b=True):
    rotulo, status = chamar_modelo(texto)
    if rotulo is not None and rotulo != "indefinido":
        return rotulo, "modelo"
    if not usar_plano_b:
        return "indefinido", "sem_plano_b"
    return plano_b.predict([texto])[0], "plano_b_" + ("indefinido" if status == "ok" else status)

res = {}
for nome, f in [("somente simulador v3", lambda t: classificar(t, False)), ("somente plano B", lambda t: (plano_b.predict([t])[0], "plano_b")), ("fluxo completo", classificar)]:
    saidas = [f(t) for t in Xte]
    ac = [float(p == r) for (p, _), r in zip(saidas, yte)]
    m, lo, hi = av.bootstrap_ic(ac, n=1000, seed=0)
    uso_b = np.mean([c.startswith("plano_b") for _, c in saidas])
    res[nome] = (m, lo, hi, uso_b)
print(f"{'alternativa':<22} {'acerto':>7} {'IC95%':>18} {'uso do plano B':>15}")
for nome, (m, lo, hi, uso_b) in res.items():
    print(f"{nome:<22} {m:>7.3f} {f'[{lo:.3f}; {hi:.3f}]':>18} {uso_b:>15.1%}")

inventadas = sum(chamar_modelo(t)[1] == "evidencia_inventada" for t in Xte) / len(Xte)
print(f"taxa de evidências inventadas pelo simulador: {inventadas:.1%} (o simulador copia trechos da mensagem)")

