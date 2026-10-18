"""Capítulo 16: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 16.1: Intervalos de Wilson por estrato e tamanho do teste]
import re
import numpy as np
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM
from llmdidatico.aval_llm import ic_wilson, n_proporcao

X, y = corpus.intencoes(100, seed=41)
sim = SimuladorLLM("v2", semente=0)
def rot(r):
    m = re.fullmatch(r"rotulo: (\w+)", r.strip()); return m.group(1) if m else None
ok = np.array([rot(sim.responder(t)) == r for t, r in zip(X, y)])
lo, hi = ic_wilson(int(ok.sum()), len(ok))
print(f"geral: {ok.mean():.3f} [{lo:.3f}; {hi:.3f}] (largura {hi - lo:.3f}, n={len(ok)})")
linhas = []
for c in sorted(set(y)):
    m = np.array([r == c for r in y]); k, n = int(ok[m].sum()), int(m.sum()); l, h = ic_wilson(k, n)
    linhas.append((k / n, l, h, n, c))
for p, l, h, n, c in sorted(linhas):
    print(f"{c:<20} n={n:>3} acerto {p:.3f} [{l:.3f}; {h:.3f}] largura {h - l:.3f}")
pior = sorted(linhas)[0]
print(f"pior categoria: {pior[4]} ({pior[0]:.3f}); casos necessários para ±5 p.p.: {n_proporcao(pior[0], 0.05)}")


# %% [Exercício 16.2: Viés de posição do juiz simulado e troca de ordem]
import numpy as np
from llmdidatico.aval_llm import JuizEnviesado, kappa_cohen

rng = np.random.default_rng(3)
def resp(q, p=None):
    return {"qualidade": q, "palavras": int(p if p is not None else np.clip(60 + 12 * q + rng.normal(0, 20), 15, 300)), "estilo": "outro"}
iguais = [(resp(2, 80), resp(2, 80)) for _ in range(400)]
dif = []
while len(dif) < 400:
    a, b = resp(int(rng.integers(0, 4))), resp(int(rng.integers(0, 4)))
    if a["qualidade"] != b["qualidade"]: dif.append((a, b))
verdade = np.array(["a" if a["qualidade"] > b["qualidade"] else "b" for a, b in dif])
humano = np.where(rng.random(400) < 0.06, np.where(verdade == "a", "b", "a"), verdade)
print(f"{'viés':>5} {'1ª (pares iguais)':>18} {'conc. sem troca':>16} {'conc. com troca':>16} {'decididos':>10} {'kappa (com troca)':>18}")
for vp in (0.0, 0.5, 1.0):
    j0 = JuizEnviesado(vies_posicao=vp, semente=1)
    pri = np.mean([j0.comparar(a, b) == 1 for a, b in iguais])
    j1 = JuizEnviesado(vies_posicao=vp, semente=2); v1 = np.array([j1.comparar_protocolo(a, b, False) for a, b in dif])
    j2 = JuizEnviesado(vies_posicao=vp, semente=2); v2 = np.array([j2.comparar_protocolo(a, b, True) for a, b in dif]); dec = v2 != "empate"
    print(f"{vp:>5.1f} {pri:>18.3f} {np.mean(v1 == humano):>16.3f} {np.mean(v2[dec] == humano[dec]):>16.3f} {dec.mean():>10.3f} {kappa_cohen(humano[dec], v2[dec]):>18.3f}")


# %% [Exercício 16.3: Comparação pareada e múltiplas comparações]
import json, re
import numpy as np
from llmdidatico import corpus, avaliacao as av
from llmdidatico.simulador import SimuladorLLM
from llmdidatico.aval_llm import mcnemar, bootstrap_pareado, holm, benjamini_hochberg

X, y = corpus.intencoes(100, seed=41)
def acertos(v):
    sim = SimuladorLLM(v, semente=0); out = []
    for t, r in zip(X, y):
        s = sim.responder(t)
        if v == "v3": p = json.loads(s)["intencao"]
        else:
            m = re.fullmatch(r"rotulo: (\w+)", s.strip()); p = m.group(1) if m else None
        out.append(float(p == r))
    return np.array(out)
a2, a3 = acertos("v2"), acertos("v3")
m = mcnemar(a3, a2)
mu, lo, hi = bootstrap_pareado(a3, a2, seed=0)
i3, i2 = av.bootstrap_ic(a3, seed=1), av.bootstrap_ic(a2, seed=2)
print(f"acerto v2 {a2.mean():.3f} | v3 {a3.mean():.3f}")
print(f"McNemar: b={m['b']} c={m['c']} p exato={m['p_exato']:.2e}")
print(f"diferença pareada {mu:+.3f} IC95% [{lo:+.3f}; {hi:+.3f}] (largura {hi - lo:.3f})")
print(f"ICs separados: v3 [{i3[1]:.3f}; {i3[2]:.3f}] v2 [{i2[1]:.3f}; {i2[2]:.3f}]")

rng = np.random.default_rng(5); variantes = []
for _ in range(149):
    v = a2.copy(); mask = rng.random(len(v)) < 0.12; v[mask] = rng.random(mask.sum()) < a2.mean(); variantes.append(v)
variantes.append(a3)
ps = np.array([mcnemar(v, a2)["p_exato"] for v in variantes])
print(f"falsos positivos entre as 149 equivalentes: sem correção {int(np.sum(ps[:149] < 0.05))} | Holm {int(np.sum(np.array(holm(ps))[:149] < 0.05))} | BH {int(np.sum(np.array(benjamini_hochberg(ps))[:149] < 0.05))}")
print(f"v3 detectada: sem correção {bool(ps[149] < 0.05)} | Holm {bool(holm(ps)[149] < 0.05)} | BH {bool(benjamini_hochberg(ps)[149] < 0.05)}")


# %% [Projeto: Bancada de avaliação com relatório automático]
import datetime, hashlib, json, re
import numpy as np
from sklearn.metrics import f1_score
from llmdidatico import corpus
from llmdidatico.simulador import SimuladorLLM, INTENCOES
from llmdidatico.aval_llm import ic_wilson, mcnemar, bootstrap_pareado, holm

def interpretar(v, resp):
    """Devolve (rótulo, formato_válido)."""
    if v == "v3":
        try: return json.loads(resp)["intencao"], True
        except Exception: return None, False
    if v == "v2":
        m = re.fullmatch(r"rotulo: (\w+)", resp.strip()); return (m.group(1), True) if m else (None, False)
    achados = [c for c in INTENCOES if c.replace("_", " ") in resp]
    return (achados[0] if achados else None), False

def executar(v, X, semente):
    sim = SimuladorLLM(v, semente=semente)
    saidas = [interpretar(v, sim.responder(t)) for t in X]
    return [p for p, _ in saidas], [f for _, f in saidas]

def bancada(sistemas, X, y, estratos, base, candidato, K=10, margem=0.02, queda_estrato=0.10):
    L = []; y_arr = np.array(y)
    res = {}
    for s in sistemas:
        accs, preds0, fmt0 = [], None, None
        for k in range(K):
            p, f = executar(s, X, k)
            accs.append(np.mean(np.array(p, dtype=object) == y_arr))
            if k == 0: preds0, fmt0 = p, f
        ok0 = np.array([a == b for a, b in zip(preds0, y)], float)
        f1 = f1_score(y, [p if p else "indefinido" for p in preds0], labels=INTENCOES, average="macro", zero_division=0)
        res[s] = dict(accs=np.array(accs), ok0=ok0, f1=f1, fmt=np.mean(fmt0))
    L.append("# Relatório de avaliação (simulador; não é um LLM)\n")
    L.append(f"- conjunto: n={len(X)}, sha1={hashlib.sha1(chr(10).join(X).encode('utf-8')).hexdigest()[:12]}; K={K} sementes; data {datetime.date.today().isoformat()}\n")
    L.append("| sistema | acerto médio (K) | dp entre sementes | IC95% Wilson (semente 0) | F1 macro | formato válido |\n|---|---|---|---|---|---|")
    for s, r in res.items():
        lo, hi = ic_wilson(int(r["ok0"].sum()), len(X))
        L.append(f"| {s} | {r['accs'].mean():.3f} | {r['accs'].std(ddof=1):.4f} | [{lo:.3f}; {hi:.3f}] | {r['f1']:.3f} | {r['fmt']:.3f} |")
    outros = [s for s in sistemas if s != base]
    ps = [mcnemar(res[s]["ok0"], res[base]["ok0"])["p_exato"] for s in outros]; aj = holm(ps)
    L.append(f"\n## Comparações com a linha de base ({base})\n")
    L.append("| sistema | diferença | IC95% (bootstrap pareado) | b | c | p exato | p Holm |\n|---|---|---|---|---|---|---|")
    for s, p, a in zip(outros, ps, aj):
        d, lo, hi = bootstrap_pareado(res[s]["ok0"], res[base]["ok0"], seed=0); m = mcnemar(res[s]["ok0"], res[base]["ok0"])
        L.append(f"| {s} | {d:+.3f} | [{lo:+.3f}; {hi:+.3f}] | {m['b']} | {m['c']} | {p:.2e} | {a:.2e} |")
    d, lo, hi = bootstrap_pareado(res[candidato]["ok0"], res[base]["ok0"], seed=0)
    motivos = [] if lo >= -margem else [f"IC da diferença [{lo:+.3f}; {hi:+.3f}] passa de -{margem}"]
    linhas = []
    for dim, vals in estratos.items():
        v_arr = np.array(vals)
        for e in sorted(set(vals)):
            msk = v_arr == e; dd = res[candidato]["ok0"][msk].mean() - res[base]["ok0"][msk].mean()
            k, n = int(res[candidato]["ok0"][msk].sum()), int(msk.sum()); l, h = ic_wilson(k, n); linhas.append((k / n, l, h, n, f"{dim}={e}"))
            if dd < -queda_estrato: motivos.append(f"estrato {dim}={e} caiu {dd:+.2f}")
    L.append(f"\n## Portão de regressão: {candidato} vs {base}\n")
    L.append(f"**{'APROVADO' if not motivos else 'REPROVADO'}** (margem {margem}; queda máxima por estrato {queda_estrato}) {'; '.join(motivos)}\n")
    L.append(f"Três piores estratos de {candidato}:\n")
    for p, l, h, n, e in sorted(linhas)[:3]: L.append(f"- {e}: n={n}, acerto {p:.3f} [{l:.3f}; {h:.3f}]")
    cartao = {"sistemas": sistemas, "linha_de_base": base, "candidato": candidato, "sementes": list(range(K)), "temperatura": None,
              "teste": f"corpus.intencoes(100, seed=61), n={len(X)}",
              "limitacoes": ["simulador de regras (não é um LLM)", "dados sintéticos", "sem avaliação humana", "estratos pequenos"]}
    L.append("\n## Cartão de avaliação\n\n" + json.dumps(cartao, ensure_ascii=False, indent=1))
    return "\n".join(L), res

X, y = corpus.intencoes(100, seed=61)
comp = np.array([len(t.split()) for t in X]); c1, c2 = np.quantile(comp, [1/3, 2/3])
tam = ["curta" if c <= c1 else "média" if c <= c2 else "longa" for c in comp]
print(relatorio)
print("\nDecisão: a v3 substitui a v2 neste teste; o que o teste NÃO cobre: modelos reais, usuários reais, texto fora do padrão das frases sintéticas, avaliação humana.")

