"""Capítulo 1: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
from llmdidatico import corpus, modelos

frases = corpus.frases_lm(n=3000, seed=0)       # lista de frases, cada uma uma lista de tokens
treino, teste = frases[:2400], frases[2400:]
print("Exemplo:", " ".join(frases[0]))
print("Tamanho do vocabulário:", len({t for f in frases for t in f}))

lm = modelos.NgramaLM(n=3, k=0.05).ajustar(treino)   # trigramas com suavização aditiva
contexto = ["o", "estudante"]
dist = lm.distribuicao(contexto)                  # probabilidade de cada token do vocabulário
top = sorted(zip(lm.vocab, dist), key=lambda x: -x[1])[:5]
print(f"Próximo token após '{' '.join(contexto)}':")
for tok, p in top:
    print(f"  {tok:<12s} {p:.3f}")
print("Soma das probabilidades:", round(float(dist.sum()), 6))


# %% [bloco 2]
import re
from llmdidatico import DOCS, modelos

def modelo_brinquedo(prompt: str) -> str:
    """MODELO (de brinquedo): devolve a frase do contexto mais parecida com a pergunta."""
    ctx, _, pergunta = prompt.partition("\nPERGUNTA: ")
    frases = [f.strip() for f in re.split(r"(?<=[.!?])\s+", ctx.replace("CONTEXTO:", "")) if f.strip()]
    if not frases:
        return "Não encontrei informação para responder."
    q = set(modelos.tokenizar_sem_stop(pergunta))
    pontos = [len(q & set(modelos.tokenizar_sem_stop(f))) for f in frases]
    return frases[pontos.index(max(pontos))]

def aplicacao(pergunta: str, contexto: str = "") -> str:
    """APLICAÇÃO: monta o prompt, chama o modelo e valida a saída."""
    prompt = f"CONTEXTO: {contexto}\nPERGUNTA: {pergunta}"
    resposta = modelo_brinquedo(prompt)
    return resposta if len(resposta) < 400 else resposta[:400]

def sistema(pergunta: str) -> dict:
    """SISTEMA: acrescenta recuperação de documentos e registro da evidência."""
    bm = modelos.BM25([modelos.tokenizar_sem_stop(d[3]) for d in DOCS])
    s = bm.pontuar(modelos.tokenizar_sem_stop(pergunta))
    melhor = int(s.argmax())
    resposta = aplicacao(pergunta, DOCS[melhor][3])
    return {"resposta": resposta, "fonte": DOCS[melhor][0] + " — " + DOCS[melhor][2]}

pergunta = "Qual o valor da multa por atraso na devolução de livros?"
print("1) só o modelo, sem contexto  :", aplicacao(pergunta))
print("2) aplicação com contexto fixo:", aplicacao(pergunta, "A biblioteca abre às 7h."))
r = sistema(pergunta)
print("3) sistema com recuperação    :", r["resposta"])
print("   fonte registrada           :", r["fonte"])


# %% [bloco 3]
for n in (1, 2, 3):
    m = modelos.NgramaLM(n=n, k=0.05).ajustar(treino)
    print(f"n={n}: perplexidade no teste = {m.perplexidade(teste):.2f}")


# %% [bloco 4]
f1 = "a professora corrige a prova .".split()
f2 = "a prova corrige a professora .".split()
m3 = modelos.NgramaLM(n=3, k=0.05).ajustar(treino)
import math
for f in (f1, f2):
    lp = m3.logprob_frase(f)
    print(" ".join(f), f"-> ln P = {lp:.2f}, P = {math.exp(lp):.2e}")

