"""Implementações didáticas em NumPy: softmax, atenção, amostragem, modelos de linguagem pequenos, BPE, BM25, LSA."""
import math
from collections import Counter, defaultdict
import numpy as np


def softmax(z, axis=-1, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def entropia_cruzada(p_alvo_idx, logits):
    """Média de -log p(alvo). p_alvo_idx: índices inteiros (N,), logits: (N, V)."""
    lp = log_softmax(logits)
    return float(-lp[np.arange(len(p_alvo_idx)), p_alvo_idx].mean())


def cosseno(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def atencao(Q, K, V, mascara_causal=False):
    """Atenção por produto escalar escalonado. Q:(n,dk) K:(m,dk) V:(m,dv). Retorna (saída, pesos)."""
    dk = Q.shape[-1]
    S = Q @ K.T / math.sqrt(dk)
    if mascara_causal:
        S = np.where(np.tril(np.ones_like(S)) == 1, S, -1e9)
    A = softmax(S, axis=-1)
    return A @ V, A


# ---------------------------------------------------------------- amostragem
def filtrar_top_k(p, k):
    p = p.copy(); idx = np.argsort(p)[::-1][k:]; p[idx] = 0; return p / p.sum()


def filtrar_top_p(p, top_p):
    ordem = np.argsort(p)[::-1]; cum = np.cumsum(p[ordem])
    corte = int(np.searchsorted(cum, top_p) + 1)
    q = np.zeros_like(p); q[ordem[:corte]] = p[ordem[:corte]]; return q / q.sum()


def amostrar(logits, rng, T=1.0, top_k=None, top_p=None):
    p = softmax(logits, T=T)
    if top_k: p = filtrar_top_k(p, top_k)
    if top_p: p = filtrar_top_p(p, top_p)
    return int(rng.choice(len(p), p=p))


# ---------------------------------------------------------------- modelo n-grama
class NgramaLM:
    """Modelo de n-gramas com suavização aditiva (Laplace generalizada)."""
    def __init__(self, n=2, k=0.1):
        self.n, self.k = n, k

    def ajustar(self, frases):
        self.vocab = sorted({t for f in frases for t in f} | {"<s>"})
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        self.V = len(self.vocab)
        self.cont = defaultdict(Counter)
        for f in frases:
            seq = ["<s>"] * (self.n - 1) + list(f)
            for i in range(self.n - 1, len(seq)):
                self.cont[tuple(seq[i - self.n + 1:i])][seq[i]] += 1
        return self

    def prob(self, ctx, tok):
        c = self.cont.get(tuple(ctx), Counter())
        return (c[tok] + self.k) / (sum(c.values()) + self.k * self.V)

    def logprob_frase(self, f):
        seq = ["<s>"] * (self.n - 1) + list(f)
        return sum(math.log(self.prob(seq[i - self.n + 1:i], seq[i])) for i in range(self.n - 1, len(seq)))

    def perplexidade(self, frases):
        lp = sum(self.logprob_frase(f) for f in frases); N = sum(len(f) for f in frases)
        return math.exp(-lp / N)

    def distribuicao(self, ctx):
        ctx = (["<s>"] * (self.n - 1) + list(ctx))[-(self.n - 1):] if self.n > 1 else []
        return np.array([self.prob(ctx, t) for t in self.vocab])


# ---------------------------------------------------------------- modelo neural pequeno (janela de contexto), gradientes manuais
class MLPLM:
    """Embeddings + camada oculta tanh + softmax, treinado com Adam. Prevê o próximo token dado os últimos `ctx` tokens."""
    def __init__(self, vocab, ctx=3, d=16, h=48, seed=0):
        self.vocab = list(vocab); self.idx = {t: i for i, t in enumerate(self.vocab)}
        V = len(self.vocab); self.V, self.ctx, self.d, self.h = V, ctx, d, h
        rng = np.random.default_rng(seed); s = 0.1
        self.P = {"E": rng.normal(0, s, (V, d)), "W1": rng.normal(0, s, (ctx * d, h)), "b1": np.zeros(h), "W2": rng.normal(0, s, (h, V)), "b2": np.zeros(V)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}; self.v = {k: np.zeros_like(v) for k, v in self.P.items()}; self.t = 0

    def dados(self, frases):
        X, y = [], []
        for f in frases:
            seq = ["<s>"] * self.ctx + list(f)
            for i in range(self.ctx, len(seq)):
                X.append([self.idx.get(t, 0) for t in seq[i - self.ctx:i]]); y.append(self.idx.get(seq[i], 0))
        return np.array(X), np.array(y)

    def frente(self, X):
        P = self.P; e = P["E"][X].reshape(len(X), -1)
        a = np.tanh(e @ P["W1"] + P["b1"]); z = a @ P["W2"] + P["b2"]
        return e, a, z

    def perda_grad(self, X, y):
        P = self.P; e, a, z = self.frente(X); N = len(X)
        lp = log_softmax(z); loss = -lp[np.arange(N), y].mean()
        dz = np.exp(lp); dz[np.arange(N), y] -= 1; dz /= N
        g = {"W2": a.T @ dz, "b2": dz.sum(0)}
        da = dz @ P["W2"].T; dpre = da * (1 - a ** 2)
        g["W1"] = e.T @ dpre; g["b1"] = dpre.sum(0)
        de = (dpre @ P["W1"].T).reshape(N, self.ctx, self.d)
        gE = np.zeros_like(P["E"]); np.add.at(gE, X, de); g["E"] = gE
        return loss, g

    def passo(self, g, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.P:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]; self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= lr * mh / (np.sqrt(vh) + eps)

    def treinar(self, frases, epocas=8, lote=128, lr=0.01, seed=0, verbose=False):
        X, y = self.dados(frases); rng = np.random.default_rng(seed); hist = []
        for ep in range(epocas):
            ordem = rng.permutation(len(X)); tot = 0
            for i in range(0, len(X), lote):
                b = ordem[i:i + lote]; loss, g = self.perda_grad(X[b], y[b]); self.passo(g, lr); tot += loss * len(b)
            hist.append(tot / len(X))
            if verbose: print(f"época {ep + 1}: perda {hist[-1]:.4f}")
        return hist

    def logits(self, ctx_tokens):
        ids = [self.idx.get(t, 0) for t in (["<s>"] * self.ctx + list(ctx_tokens))[-self.ctx:]]
        return self.frente(np.array([ids]))[2][0]

    def perplexidade(self, frases):
        X, y = self.dados(frases); _, _, z = self.frente(X)
        return float(math.exp(-log_softmax(z)[np.arange(len(y)), y].mean()))


# ---------------------------------------------------------------- BPE
def treinar_bpe(palavras, n_fusoes):
    """BPE sobre palavras (lista de strings). Retorna lista de fusões [(a,b),...]."""
    vocab = Counter(tuple(w) + ("</w>",) for w in palavras); fusoes = []
    for _ in range(n_fusoes):
        pares = Counter()
        for w, f in vocab.items():
            for a, b in zip(w, w[1:]): pares[(a, b)] += f
        if not pares: break
        melhor = max(pares, key=lambda p: (pares[p], p)); fusoes.append(melhor)
        novo = Counter()
        for w, f in vocab.items():
            out = []; i = 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == melhor: out.append(w[i] + w[i + 1]); i += 2
                else: out.append(w[i]); i += 1
            novo[tuple(out)] += f
        vocab = novo
    return fusoes


def codificar_bpe(palavra, fusoes):
    toks = list(palavra) + ["</w>"]
    for a, b in fusoes:
        i = 0; out = []
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b: out.append(a + b); i += 2
            else: out.append(toks[i]); i += 1
        toks = out
    return toks


# ---------------------------------------------------------------- recuperação
class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b; self.docs = docs_tokens; self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / self.N
        self.df = Counter(t for d in docs_tokens for t in set(d)); self.tf = [Counter(d) for d in docs_tokens]

    def pontuar(self, consulta_tokens):
        s = np.zeros(self.N)
        for t in consulta_tokens:
            if t not in self.df: continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f: s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avgdl))
        return s


def tokenizar(texto):
    import re, unicodedata
    t = unicodedata.normalize("NFD", texto.lower()); t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", t)


STOP = set("a o as os de da do das dos e em no na nos nas um uma uns umas para por com que se ao aos qual quais quanto quantos quantas quando como posso pode ser sao e foi ha tem ter meu minha meus minhas eu".split())


def tokenizar_sem_stop(texto):
    return [t for t in tokenizar(texto) if t not in STOP]


class LSAEmbedder:
    """'Modelo de embeddings' local e acessível: TF-IDF + SVD truncada (análise semântica latente). Não é um LLM."""
    def __init__(self, dim=40, seed=0):
        self.dim, self.seed = dim, seed

    def ajustar(self, textos):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(tokenizer=tokenizar_sem_stop, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = self.vec.fit_transform(textos)
        self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=self.seed).fit(X)
        return self

    def transformar(self, textos):
        Z = self.svd.transform(self.vec.transform(textos))
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
