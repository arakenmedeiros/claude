"""Capítulo 12: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""

# %% [bloco 1]
N_req = 1_000_000
tok_instr, tok_exemplos, tok_entrada, tok_saida = 150, 6 * 60, 80, 20
preco_in, preco_out = 1.0, 3.0              # HIPOTÉTICOS: unidades monetárias por milhão de tokens
def custo(tok_prompt):
    return N_req * (tok_prompt * preco_in + tok_saida * preco_out) / 1e6
c_prompt = custo(tok_instr + tok_exemplos + tok_entrada)
c_ajust = custo(tok_instr // 3 + tok_entrada)
custo_treino = 40.0
print(f"prompt com 6 exemplos: {c_prompt:.1f} | modelo ajustado (prompt curto): {c_ajust:.1f} | diferença por milhão de requisições: {c_prompt - c_ajust:.1f}")
print(f"ponto de equilíbrio do custo de ajuste ({custo_treino:.0f}): {custo_treino / (c_prompt - c_ajust) * N_req:,.0f} requisições")


# %% [bloco 2]
import json, numpy as np
from llmdidatico import corpus, modelos, avaliacao as av
from llmdidatico.simulador import INTENCOES
from llmdidatico.lora import registro_instrucao, taxa_vazamento, divisao_por_hash, detectar_pii, mascarar_pii, normalizar_texto
X, y = corpus.intencoes(120, seed=5)
regs = [registro_instrucao(t, r, INTENCOES) for t, r in zip(X, y)]
print(json.dumps(regs[0], ensure_ascii=False, indent=1))
print("registros:", len(regs), "| textos distintos (normalizados):", len({normalizar_texto(t) for t in X}))
rng = np.random.default_rng(0); perm = rng.permutation(len(X))
tr, te = perm[:int(.8*len(X))], perm[int(.9*len(X)):]
print(f"divisão ALEATÓRIA: vazamento teste->treino = {taxa_vazamento([X[i] for i in tr], [X[i] for i in te]):.1%}")
g = divisao_por_hash(X)
print("divisão POR HASH: tamanhos", [int((g == k).sum()) for k in range(3)],
      f"| vazamento = {taxa_vazamento([X[i] for i in np.where(g==0)[0]], [X[i] for i in np.where(g==2)[0]]):.1%}")


# %% [bloco 3]
amostra = "Cliente Maria (CPF 123.456.789-09, maria@exemplo.com, (11) 91234-5678) pediu segunda via."
print(detectar_pii(amostra)); print(mascarar_pii(amostra))


# %% [bloco 4]
rng = np.random.default_rng(0)
d_in, d_out, r, alfa = 32, 24, 4, 8
W = rng.normal(0, 0.3, (d_out, d_in))
A = rng.normal(0, 1 / np.sqrt(d_in), (r, d_in)); B = np.zeros((d_out, r))
x = rng.normal(size=(5, d_in))
def camada(x, W, A, B, alfa, r): return x @ W.T + (alfa / r) * (x @ A.T) @ B.T
print("B = 0 => saída idêntica à base:", np.allclose(camada(x, W, A, B, alfa, r), x @ W.T))
print(f"parâmetros: W = {W.size}, A+B = {A.size + B.size} ({(A.size + B.size) / W.size:.1%})")
U = rng.normal(size=(d_out, 3)); V = rng.normal(size=(3, d_in)); alvo = W + 0.5 * U @ V / np.sqrt(3)
Xtr = rng.normal(size=(512, d_in)); Ytr = Xtr @ alvo.T
print(f"posto de (alvo - W): {np.linalg.matrix_rank(alvo - W)}")
def treina_lora(r, passos=3000, lr=0.02, seed=0):
    g = np.random.default_rng(seed); A = g.normal(0, 1/np.sqrt(d_in), (r, d_in)); B = np.zeros((d_out, r)); s = alfa / r
    mA = np.zeros_like(A); vA = np.zeros_like(A); mB = np.zeros_like(B); vB = np.zeros_like(B)
    for t in range(1, passos + 1):
        Z = Xtr @ A.T; E = Xtr @ W.T + s * Z @ B.T - Ytr; N = len(Xtr)
        gB = s * 2 / N * E.T @ Z; gA = (s * 2 / N * E @ B).T @ Xtr
        for P, G, m, v in ((A, gA, mA, vA), (B, gB, mB, vB)):
            m *= .9; m += .1 * G; v *= .999; v += .001 * G ** 2
            P -= lr * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
    return s * B @ A, float(np.mean(E ** 2))
print(f"{'r':>3} {'params':>7} {'EQM final':>11} {'erro relativo em ΔW':>20}")
for r_ in (1, 2, 3, 4, 8):
    dW, eqm = treina_lora(r_)
    print(f"{r_:>3} {r_ * (d_in + d_out):>7} {eqm:>11.2e} {np.linalg.norm(dW - (alvo - W)) / np.linalg.norm(alvo - W):>20.4f}")


# %% [bloco 5]
ant = corpus.frases_lm(3000, seed=0); ant_va = corpus.frases_lm(600, seed=98); ant_te = corpus.frases_lm(600, seed=99)
from llmdidatico.lora import *
nov = frases_dominio_novo(1500, seed=1); nov_va = frases_dominio_novo(300, seed=2); nov_te = frases_dominio_novo(600, seed=3)
print("exemplo novo:", " ".join(nov[0]), "| antigo:", " ".join(ant[0]))
voc = vocabulario(ant, nov); print("vocabulário:", len(voc))
base = modelos.MLPLM(voc, ctx=3, d=16, h=64, seed=0); base.treinar(ant, epocas=12, lote=64, lr=0.01)
tot = sum(v.size for v in base.P.values())
print(f"parâmetros do modelo base: {tot}")
print(f"perplexidade do base | tarefa original: {base.perplexidade(ant_te):.3f} | tarefa nova: {base.perplexidade(nov_te):.3f}")


# %% [bloco 6]
lora = MLPLMLoRA(base, r=4, alfa=8, seed=0)
print(f"treináveis LoRA: {lora.n_treinaveis()} de {lora.n_total_base()} ({lora.n_treinaveis()/lora.n_total_base():.1%})")
print("logits iguais ao base antes de treinar:", np.allclose(lora.frente(lora.dados(nov_te)[0])[2], base.frente(base.dados(nov_te)[0])[2]))
h_l = lora.treinar(nov, epocas=10, lr=0.005, val=nov_va)
comp, h_c = ajuste_completo(base, nov, epocas=10, lr=0.005, val=nov_va)
print("época  LoRA(val)  completo(val)")
for e in (0, 1, 2, 4, 9): print(f"{e+1:>5} {h_l['val'][e]:>10.4f} {h_c['val'][e]:>13.4f}")
tab = {}
for nome, m in (("base", base), ("LoRA r=4", lora), ("completo", comp)):
    a = acertos_token(m, nov_te); mm, lo, hi = av.bootstrap_ic(a, n=1000, seed=0)
    tab[nome] = a
    print(f"{nome:<9} ppl nova {m.perplexidade(nov_te):6.3f} | acerto top-1 nova {mm:.3f} [{lo:.3f}; {hi:.3f}] | ppl original {m.perplexidade(ant_te):6.3f}")
dm, dlo, dhi = av.bootstrap_ic(tab["completo"] - tab["LoRA r=4"], n=1000, seed=0)
print(f"completo - LoRA (pareado, nova): {dm:+.4f} IC95% [{dlo:+.4f}; {dhi:+.4f}]")


# %% [bloco 7]
print(f"{'r':>3} {'treináveis':>10} {'ppl nova':>9} {'ppl original':>13}")
for r_ in (1, 2, 4, 8, 16):
    res = []
    for s in range(3):
        L = MLPLMLoRA(base, r=r_, alfa=2 * r_, seed=s); L.treinar(nov, epocas=3, lr=0.005); res.append((L.perplexidade(nov_te), L.perplexidade(ant_te)))
    mu = np.mean(res, 0); print(f"{r_:>3} {L.n_treinaveis():>10} {mu[0]:>9.3f} {mu[1]:>13.3f}")


# %% [bloco 8]
print(f"ppl original: base {base.perplexidade(ant_te):.3f} | LoRA ativo {lora.perplexidade(ant_te):.3f} | completo {comp.perplexidade(ant_te):.3f}")
lora0 = MLPLMLoRA(base, r=4, alfa=8, seed=0)    # equivale a "desligar" os adaptadores
print(f"adaptador desligado (B=0) : ppl original = {lora0.perplexidade(ant_te):.3f}  (idêntico ao base: {np.isclose(lora0.perplexidade(ant_te), base.perplexidade(ant_te))})")
ac_ant = {n: acertos_token(m, ant_te).mean() for n, m in (("base", base), ("LoRA", lora), ("completo", comp))}
print("acerto top-1 original:", {k: round(float(v), 3) for k, v in ac_ant.items()})
mix = MLPLMLoRA(base, r=4, alfa=8, seed=0); mix.treinar(nov + ant[:1500], epocas=10, lr=0.005)
print(f"LoRA com 50% de dados antigos misturados: ppl nova {mix.perplexidade(nov_te):.3f} | ppl original {mix.perplexidade(ant_te):.3f}")


# %% [bloco 9]
fund = lora.mesclar()
Xt = lora.dados(nov_te)[0]
print("max |logits LoRA - logits mesclado|:", float(np.abs(lora.frente(Xt)[2] - fund.frente(Xt)[2]).max()))
print(f"ppl mesclado {fund.perplexidade(nov_te):.6f} | ppl LoRA {lora.perplexidade(nov_te):.6f}")
print("posto de ΔW em W1:", np.linalg.matrix_rank(lora.delta_W("W1"), tol=1e-10), "| W2:", np.linalg.matrix_rank(lora.delta_W("W2"), tol=1e-10))
import copy
for lam in (0.0, 0.5, 1.0):
    Pm = {k: base.P[k] + lam * (lora.pesos_mesclados()[k] - base.P[k]) for k in base.P}
    mm = copy.deepcopy(base); mm.P = Pm
    print(f"interpolação λ={lam}: ppl nova {mm.perplexidade(nov_te):.3f} | original {mm.perplexidade(ant_te):.3f}")


# %% [bloco 10]
W2 = base.P["W2"]
print(f"{'bits':>4} {'bloco':>6} {'RMSE':>8} {'erro rel.':>10} {'SQNR dB':>8} {'ppl original':>13}")
for bits in (8, 4):
    for bl in (16, 64, 256, W2.size):
        e = erro_reconstrucao(W2, desquantizar_blocos(quantizar_blocos(W2, bits, bl)))
        print(f"{bits:>4} {bl:>6} {e['rmse']:>8.4f} {e['erro_relativo']:>10.4f} {e['sqnr_db']:>8.1f} {quantizar_modelo(base, bits, bl).perplexidade(ant_te):>13.4f}")
g = np.random.default_rng(0).normal(size=200_000)
w_out = g.copy(); w_out[::4096] *= 40       # alguns outliers
for bl in (64, 4096):
    print(f"com outliers, bloco {bl:>4}: erro relativo int8 = {erro_reconstrucao(w_out, desquantizar_blocos(quantizar_blocos(w_out, 8, bl)))['erro_relativo']:.4f}")


# %% [bloco 11]
nf = livro_normal(4); print("livro NF4 didático:", np.round(nf, 3))
e = erro_reconstrucao(W2, desquantizar_livro(quantizar_livro(W2, nf, 64)))
print(f"NF4 didático bloco 64: RMSE {e['rmse']:.4f} erro rel. {e['erro_relativo']:.4f} SQNR {e['sqnr_db']:.1f} dB | ppl {quantizar_modelo(base, livro=nf, bloco=64).perplexidade(ant_te):.4f}")
print("pesos gaussianos sintéticos, bloco 64: SQNR int4 =", round(erro_reconstrucao(g, desquantizar_blocos(quantizar_blocos(g, 4, 64)))["sqnr_db"], 1), "dB | NF4 didático =", round(erro_reconstrucao(g, desquantizar_livro(quantizar_livro(g, nf, 64)))["sqnr_db"], 1), "dB")
rec, emax = dupla_quantizacao(quantizar_blocos(W2, 4, 64)["escala"], 256); print(f"dupla quantização: erro relativo máximo nas escalas {emax:.4f}")
print("bits/param (4 bits, bloco 64): simples =", bits_por_parametro(4, 64), "| com dupla =", round(bits_por_parametro(4, 64, dupla=True), 4))


# %% [bloco 12]
for nome, kw in (("base float64", {}), ("base int4 (bloco 64)", dict(bits=4)), ("base NF4 didático", dict(livro=nf))):
    bq = quantizar_modelo(base, bloco=64, **kw) if kw else base
    ql = MLPLMLoRA(bq, r=4, alfa=8, seed=0)
    p0 = ql.perplexidade(nov_te); ql.treinar(nov, epocas=10, lr=0.005)
    print(f"{nome:<22} nova antes {p0:6.3f} -> depois {ql.perplexidade(nov_te):6.3f} | original depois {ql.perplexidade(ant_te):6.3f}")
bq = quantizar_modelo(base, bits=4, bloco=64); ql = MLPLMLoRA(bq, r=4, alfa=8, seed=0); ql.treinar(nov, epocas=10, lr=0.005)
mesc = ql.mesclar()
rq = copy.deepcopy(mesc)
for k in ("W1", "W2", "E"): rq.P[k] = desquantizar_blocos(quantizar_blocos(mesc.P[k], 4, 64))
print(f"mesclado em precisão total: ppl nova {mesc.perplexidade(nov_te):.3f} | mesclado e requantizado em int4: {rq.perplexidade(nov_te):.3f}")


# %% [bloco 13]
from llmdidatico.lora import estimativa_memoria_gb, parametros_lora_transformer, bits_por_parametro
d, L, r = 4096, 32, 16
N = 7e9
n_q = parametros_lora_transformer(d, L, r, alvos=("q","v"))
n_all = parametros_lora_transformer(d, L, r, d_ff=11008, alvos=("q","k","v","o","mlp"))
print(f"adaptadores r={r}: só Wq e Wv = {n_q/1e6:.1f} M ({n_q/N:.3%} de N) | todas as lineares = {n_all/1e6:.1f} M ({n_all/N:.2%} de N)")
for nome, kw in [("ajuste completo", dict(modo="completo")), ("LoRA (base 16 bits)", dict(modo="lora", n_adapt=n_all)), ("QLoRA (base 4 bits)", dict(modo="qlora", n_adapt=n_all, dupla=True))]:
    c = estimativa_memoria_gb(N, **kw); print(f"{nome:<22} {sum(c.values()):7.1f} GB  {({k: round(v, 1) for k, v in c.items()})}")

