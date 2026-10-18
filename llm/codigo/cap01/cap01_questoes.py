"""Capítulo 1: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 14]
import math

def logprob_frase(frase, p_cond, ordem=3):
    total = 0.0
    for i, tok in enumerate(frase):
        ctx = tuple(frase[max(0, i - (ordem - 1)):i])
        total += math.log(p_cond[(ctx, tok)])
    return total

p = {((), "o"): 0.2, (("o",), "aluno"): 0.5, (("o", "aluno"), "lê"): 0.1, (("aluno", "lê"), "."): 0.6}
lp = logprob_frase(["o", "aluno", "lê", "."], p)
assert abs(lp - math.log(0.006)) < 1e-12
print(round(lp, 3), round(math.exp(-lp / 4), 2))

# %% [Questão discursiva 15]
import math

def perplexidade(logprobs):
    return math.exp(-sum(logprobs) / len(logprobs))

assert abs(perplexidade([math.log(0.25)] * 5) - 4.0) < 1e-12
print(perplexidade([math.log(0.25)] * 5))

# %% [Questão discursiva 16]
from collections import Counter, defaultdict

def treinar_bigramas(frases, k=0.1):
    vocab = sorted({t for f in frases for t in f} | {"<s>"})
    cont = defaultdict(Counter)
    for f in frases:
        seq = ["<s>"] + list(f)
        for a, b in zip(seq, seq[1:]):
            cont[a][b] += 1
    V = len(vocab)
    def prob(anterior, token):
        c = cont[anterior]
        return (c[token] + k) / (sum(c.values()) + k * V)
    return prob, vocab

frases = [["o", "aluno", "lê", "."], ["o", "aluno", "escreve", "."], ["a", "aluna", "lê", "."]]
prob, vocab = treinar_bigramas(frases, k=0.1)
assert abs(sum(prob("aluno", t) for t in vocab) - 1.0) < 1e-12
print(round(prob("aluno", "lê"), 4))
