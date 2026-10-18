"""Capítulo 12: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""

# %% [Exercício 12.1: Quanto posto é preciso? Recuperação de uma mudança de posto 5]
import numpy as np
rng = np.random.default_rng(3)
d_in, d_out, alfa = 40, 30, 16
W = rng.normal(0, 0.3, (d_out, d_in))
U = rng.normal(size=(d_out, 5)); V = rng.normal(size=(5, d_in)); D = U @ V / np.sqrt(5) * 0.5
X = rng.normal(size=(600, d_in)); Y = X @ (W + D).T
Xv = rng.normal(size=(300, d_in)); Yv = Xv @ (W + D).T
def treina(r, passos=3000, lr=0.02, seed=0):
    g = np.random.default_rng(seed); A = g.normal(0, 1 / np.sqrt(d_in), (r, d_in)); B = np.zeros((d_out, r)); s = alfa / r
    est = [[A, np.zeros_like(A), np.zeros_like(A)], [B, np.zeros_like(B), np.zeros_like(B)]]
    for t in range(1, passos + 1):
        Z = X @ A.T; E = X @ W.T + s * Z @ B.T - Y; N = len(X)
        gB = s * 2 / N * E.T @ Z; gA = (s * 2 / N * E @ B).T @ X
        for (P, m, v), G in zip(est, (gA, gB)):
            m *= .9; m += .1 * G; v *= .999; v += .001 * G ** 2
            P -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    dW = s * B @ A
    return dW, float(np.mean((Xv @ (W + dW).T - Yv) ** 2))
print(f"posto verdadeiro de ΔW: {np.linalg.matrix_rank(D)} | EQM de validação sem ajuste: {np.mean((Xv @ W.T - Yv) ** 2):.3f}")
print(f"{'r':>3} {'params':>7} {'EQM val.':>10} {'erro rel. ΔW':>13}")
for r in (1, 3, 5, 8):
    dW, e = treina(r); print(f"{r:>3} {r*(d_in+d_out):>7} {e:>10.2e} {np.linalg.norm(dW - D) / np.linalg.norm(D):>13.4f}")
print(f"matriz completa: {d_in*d_out} parâmetros")


# %% [Exercício 12.2: Escala α/r e taxa de aprendizado no MLPLM]
import numpy as np
from llmdidatico import corpus, modelos
from llmdidatico.lora import *
ant = corpus.frases_lm(3000, seed=0); ant_te = corpus.frases_lm(600, seed=99)
nov = frases_dominio_novo(1500, seed=1); nov_te = frases_dominio_novo(600, seed=3)
base = modelos.MLPLM(vocabulario(ant, nov), ctx=3, d=16, h=64, seed=0); base.treinar(ant, epocas=12, lote=64, lr=0.01)
r = 4
print(f"base: ppl nova {base.perplexidade(nov_te):.3f} | original {base.perplexidade(ant_te):.3f}")
print(f"{'alfa/r':>7} {'lr':>7} {'ppl nova':>9} {'ppl original':>13}")
for esc in (0.25, 1, 2, 8):
    for lr in (0.001, 0.005, 0.02):
        res = []
        for s in range(3):
            L = MLPLMLoRA(base, r=r, alfa=esc * r, seed=s); L.treinar(nov, epocas=3, lr=lr); res.append((L.perplexidade(nov_te), L.perplexidade(ant_te)))
        mu = np.mean(res, 0); print(f"{esc:>7} {lr:>7} {mu[0]:>9.3f} {mu[1]:>13.3f}")


# %% [Exercício 12.3: Quanto se pode quantizar? bits e tamanho de bloco]
import numpy as np
from llmdidatico import corpus, modelos
from llmdidatico.lora import *
ant = corpus.frases_lm(3000, seed=0); ant_te = corpus.frases_lm(600, seed=99)
nov = frases_dominio_novo(1500, seed=1)
base = modelos.MLPLM(vocabulario(ant, nov), ctx=3, d=16, h=64, seed=0); base.treinar(ant, epocas=12, lote=64, lr=0.01)
p0 = base.perplexidade(ant_te); print(f"ppl original sem quantizar: {p0:.4f}")
print(f"{'bits':>4} {'bloco':>6} {'ppl':>8} {'variação':>9} {'bits/param':>11}")
ok = []
for bits in (8, 6, 4, 3):
    for bl in (16, 64, 256):
        p = quantizar_modelo(base, bits, bl).perplexidade(ant_te); bpp = bits_por_parametro(bits, bl)
        print(f"{bits:>4} {bl:>6} {p:>8.4f} {p / p0 - 1:>+9.2%} {bpp:>11.3f}")
        if p / p0 - 1 < 0.01: ok.append((bpp, bits, bl))
b = min(ok); print(f"menor custo com degradação < 1%: {b[1]} bits, bloco {b[2]} ({b[0]:.3f} bits/parâmetro)")


# %% [Projeto: Ajuste com mitigação de esquecimento: quanto dado antigo misturar?]
import numpy as np
from llmdidatico import corpus, modelos, avaliacao as av
from llmdidatico.lora import *
ant = corpus.frases_lm(3000, seed=0); ant_te = corpus.frases_lm(600, seed=99)
nov = frases_dominio_novo(1500, seed=1); nov_te = frases_dominio_novo(600, seed=3)
base = modelos.MLPLM(vocabulario(ant, nov), ctx=3, d=16, h=64, seed=0); base.treinar(ant, epocas=12, lote=64, lr=0.01)
p_ant, p_nov = base.perplexidade(ant_te), base.perplexidade(nov_te)
print(f"base: ppl nova {p_nov:.3f} | original {p_ant:.3f}")
rows = []
print(f"{'método':<10} {'antigos/novos':>9} {'ppl nova':>9} {'ppl original':>13} {'(desvio original)':>18}")
for metodo in ("LoRA r=4", "completo"):
    for frac in (0.0, 0.1, 0.25, 0.5):
        res = []
        for s in range(3):
            dados = nov + ant[:int(frac * len(nov))]
            if metodo == "completo":
                m, _ = ajuste_completo(base, dados, epocas=6, lr=0.005, seed=s)
            else:
                m = MLPLMLoRA(base, r=4, alfa=8, seed=s); m.treinar(dados, epocas=6, lr=0.005, seed=s)
            res.append((m.perplexidade(nov_te), m.perplexidade(ant_te)))
        res = np.array(res); mu = res.mean(0); rows.append((metodo, frac, mu[0], mu[1]))
        print(f"{metodo:<10} {frac:>9.0%} {mu[0]:>9.3f} {mu[1]:>13.3f} {res[:, 1].std():>18.3f}")
limite = 1.15 * p_ant
cand = [r for r in rows if r[3] <= limite]
print(f"critério: ppl original <= {limite:.3f} (15% acima do base)")
if cand:
    b = min(cand, key=lambda r: r[2]); print(f"melhor configuração dentro do critério: {b[0]} com {b[1]:.0%} de dados antigos (nova {b[2]:.3f}, original {b[3]:.3f})")
else: print("nenhuma configuração atende ao critério")
L = MLPLMLoRA(base, r=4, alfa=8, seed=0); L.treinar(nov + ant[:int(0.5 * len(nov))], epocas=6, lr=0.005, seed=0)
Xt = L.dados(nov_te)[0]
dif = float(np.abs(L.frente(Xt)[2] - L.mesclar().frente(Xt)[2]).max())
L0 = MLPLMLoRA(base, r=4, alfa=8, seed=0)
print(f"verificação da escolhida: diferença máxima de logits após mesclar = {dif:.1e} | adaptador desligado devolve o base: {np.isclose(L0.perplexidade(ant_te), p_ant)}")

