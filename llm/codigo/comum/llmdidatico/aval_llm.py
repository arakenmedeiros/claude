"""Ferramentas de avaliação do Capítulo 16 (NumPy/SciPy), escritas para fins didáticos.

Contém: normalização e métricas de texto (correspondência exata, F1 de tokens, ROUGE-1/ROUGE-L e BLEU simplificados),
intervalos e tamanhos de amostra para proporções, testes pareados (McNemar, bootstrap pareado, permutação), correção para
múltiplas comparações, concordância entre avaliadores (kappa de Cohen, alfa de Krippendorff) e um JUIZ SIMULADO por regras
(JuizEnviesado), que NÃO é um LLM: seus vieses são programados para demonstrar como medi-los e mitigá-los.
"""
import math
import re
import unicodedata
from collections import Counter

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ texto
def tokens(texto):
    """Minúsculas, sem acentos; mantém números como '1,50' e '3.100,00' como um único token."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"\d+(?:[.,]\d+)*|[a-z]+", t)


def correspondencia_exata(ref, hip):
    return float(tokens(ref) == tokens(hip))


def f1_tokens(ref, hip):
    r, h = Counter(tokens(ref)), Counter(tokens(hip))
    comum = sum((r & h).values())
    if comum == 0:
        return 0.0
    p, rc = comum / sum(h.values()), comum / sum(r.values())
    return 2 * p * rc / (p + rc)


def _prf(comum, n_hip, n_ref):
    p = comum / n_hip if n_hip else 0.0
    r = comum / n_ref if n_ref else 0.0
    return {"p": p, "r": r, "f": 2 * p * r / (p + r) if p + r else 0.0}


def _ngramas(tok, n):
    return Counter(tuple(tok[i:i + n]) for i in range(len(tok) - n + 1))


def rouge_n(ref, hip, n=1):
    """ROUGE-N simplificado: sobreposição de n-gramas (com contagem clipada) entre referência e hipótese."""
    r, h = _ngramas(tokens(ref), n), _ngramas(tokens(hip), n)
    return _prf(sum((r & h).values()), sum(h.values()), sum(r.values()))


def _lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def rouge_l(ref, hip):
    """ROUGE-L simplificado: usa a maior subsequência comum (LCS) de tokens."""
    a, b = tokens(ref), tokens(hip)
    return _prf(_lcs(a, b), len(b), len(a))


def bleu_simples(ref, hip, max_n=2):
    """BLEU de uma sentença, simplificado: média geométrica das precisões de 1..max_n gramas (com suavização +1 acima de 1)
    e penalidade de brevidade. Serve para ilustrar; para relatos use uma biblioteca consolidada."""
    r, h = tokens(ref), tokens(hip)
    if not h:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        hn, rn = _ngramas(h, n), _ngramas(r, n)
        comum, total = sum((hn & rn).values()), max(sum(hn.values()), 0)
        if n == 1:
            if comum == 0:
                return 0.0
            logs.append(math.log(comum / total))
        else:
            logs.append(math.log((comum + 1) / (total + 1)))
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(sum(logs) / len(logs))


# ------------------------------------------------------------------ proporções e tamanho de amostra
def n_proporcao(p, margem, z=1.96):
    """Casos necessários para estimar uma proporção p com margem (semi-largura) dada: n = z^2 p(1-p)/margem^2."""
    return int(math.ceil(z ** 2 * p * (1 - p) / margem ** 2))


def ic_wilson(k, n, z=1.96):
    """Intervalo de Wilson para uma proporção k/n (melhor que o de Wald quando n é pequeno ou p está perto de 0 ou 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (centro - meia, centro + meia)


def n_pareado(delta, psi, alpha=0.05, poder=0.80):
    """Casos para detectar com o teste de McNemar (aprox. normal) uma diferença delta = P(A certo, B errado) - P(A errado, B certo),
    dada a proporção psi de casos discordantes: n = (z_a sqrt(psi) + z_b sqrt(psi - delta^2))^2 / delta^2."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(poder)
    return int(math.ceil((za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2))


# ------------------------------------------------------------------ comparação pareada
def mcnemar(a_ok, b_ok):
    """Teste de McNemar para dois sistemas avaliados nos mesmos casos (vetores 0/1 de acerto).
    b = A acerta e B erra; c = A erra e B acerta. Devolve dict com b, c, p exato (binomial) e p assintótico (qui-quadrado com correção)."""
    a, b_ = np.asarray(a_ok, int), np.asarray(b_ok, int)
    b = int(np.sum((a == 1) & (b_ == 0)))
    c = int(np.sum((a == 0) & (b_ == 1)))
    d = b + c
    if d == 0:
        return {"b": b, "c": c, "p_exato": 1.0, "chi2": 0.0, "p_chi2": 1.0}
    p_exato = float(stats.binomtest(min(b, c), d, 0.5).pvalue)
    chi2 = (abs(b - c) - 1) ** 2 / d
    return {"b": b, "c": c, "p_exato": p_exato, "chi2": chi2, "p_chi2": float(stats.chi2.sf(chi2, 1))}


def bootstrap_pareado(a, b, n=2000, alpha=0.05, seed=0):
    """IC bootstrap da diferença média (A - B), reamostrando CASOS (os pares permanecem juntos)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    medias = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(medias, alpha / 2)), float(np.quantile(medias, 1 - alpha / 2))


def permutacao_pareada(a, b, n=5000, seed=0):
    """Teste de permutação (inversão de sinais) para a diferença média pareada; p bilateral."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    sinais = rng.choice([-1.0, 1.0], size=(n, len(d)))
    sim = np.abs((sinais * d).mean(axis=1))
    return float((np.sum(sim >= obs - 1e-12) + 1) / (n + 1))


def bonferroni(ps):
    m = len(ps)
    return [min(1.0, p * m) for p in ps]


def holm(ps):
    """Ajuste de Holm (controla a probabilidade de qualquer falso positivo, com mais poder que Bonferroni)."""
    m = len(ps)
    ordem = np.argsort(ps)
    aj = np.zeros(m)
    maximo = 0.0
    for rank, i in enumerate(ordem):
        maximo = max(maximo, (m - rank) * ps[i])
        aj[i] = min(1.0, maximo)
    return aj.tolist()


def benjamini_hochberg(ps):
    """Ajuste de Benjamini-Hochberg (controla a taxa de falsas descobertas)."""
    m = len(ps)
    ordem = np.argsort(ps)[::-1]
    aj = np.zeros(m)
    minimo = 1.0
    for k, i in enumerate(ordem):
        rank = m - k
        minimo = min(minimo, ps[i] * m / rank)
        aj[i] = minimo
    return aj.tolist()


# ------------------------------------------------------------------ concordância entre avaliadores
def kappa_cohen(a, b):
    """Kappa de Cohen para dois avaliadores (rótulos nominais): (po - pe) / (1 - pe)."""
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def alfa_krippendorff(dados, nivel="nominal"):
    """Alfa de Krippendorff. dados: lista de avaliadores, cada um com uma lista de valores por unidade (None = ausente).
    nivel: 'nominal', 'ordinal' ou 'intervalo'. Admite dados faltantes e qualquer número de avaliadores."""
    n_unid = len(dados[0])
    unidades = [[r[u] for r in dados if r[u] is not None] for u in range(n_unid)]
    unidades = [u for u in unidades if len(u) >= 2]
    valores = sorted({v for u in unidades for v in u})
    idx = {v: i for i, v in enumerate(valores)}
    k = len(valores)
    o = np.zeros((k, k))
    for u in unidades:
        m = len(u)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[idx[u[i]], idx[u[j]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if nivel == "nominal":
        delta = 1.0 - np.eye(k)
    elif nivel == "intervalo":
        v = np.array(valores, float)
        delta = (v[:, None] - v[None, :]) ** 2
    elif nivel == "ordinal":
        acum = np.cumsum(n_c)
        delta = np.zeros((k, k))
        for c in range(k):
            for d in range(k):
                if c != d:
                    lo, hi = min(c, d), max(c, d)
                    delta[c, d] = (acum[hi] - (n_c[lo] + n_c[hi]) / 2 - (acum[lo] - n_c[lo])) ** 2
    else:
        raise ValueError(nivel)
    d_obs = (o * delta).sum() / n
    d_esp = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1 - d_obs / d_esp) if d_esp > 0 else 1.0


# ------------------------------------------------------------------ juiz simulado
class JuizEnviesado:
    """SIMULAÇÃO por regras (NÃO é um LLM). Pontua uma resposta como: qualidade + vieses programados + ruído.
    Cada resposta é um dicionário com 'qualidade' (número de fatos corretos), 'palavras' (tamanho) e 'estilo' ('juiz' = o estilo
    que o juiz prefere, imitando a autopreferência)."""

    def __init__(self, vies_posicao=0.0, vies_verbosidade=0.0, vies_autopreferencia=0.0, ruido=0.35, semente=0):
        self.vp, self.vv, self.va, self.ruido = vies_posicao, vies_verbosidade, vies_autopreferencia, ruido
        self.rng = np.random.default_rng(semente)

    def nota(self, resp, posicao=1):
        s = resp["qualidade"] + self.vv * resp["palavras"] / 100.0
        s += self.va * (resp["estilo"] == "juiz")
        s += self.vp * (posicao == 1)
        return s + self.rng.normal(0, self.ruido)

    def comparar(self, primeira, segunda):
        """Devolve 1 se prefere a resposta apresentada primeiro, 2 se a segunda."""
        return 1 if self.nota(primeira, 1) >= self.nota(segunda, 2) else 2

    def comparar_protocolo(self, a, b, trocar_ordem=True):
        """Devolve 'a', 'b' ou 'empate'. Com troca de ordem, só há vencedor se os dois julgamentos concordarem."""
        v1 = "a" if self.comparar(a, b) == 1 else "b"
        if not trocar_ordem:
            return v1
        v2 = "b" if self.comparar(b, a) == 1 else "a"
        return v1 if v1 == v2 else "empate"
