import heapq
import numpy as np
from fedlib import *
from personalizacao import dividir_local, acuracias_locais


# ============================================================ 12.2 Aprendizado federado assíncrono
def simular_sincrono(partes, X, y, Xte, yte, tempos, T=60.0, m=None, epocas=3, eta=0.1, lote=32, seed=0, classes=10):
    """FedAvg síncrono em tempo virtual: cada rodada dura tanto quanto o cliente mais lento sorteado.
    Devolve o modelo e a curva (tempo, acurácia), com uma medida por unidade de tempo."""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes)); t, curva, prox, r = 0.0, [], 0.0, 0
    m = m or len(partes)
    while t < T:
        esc = rng.choice(len(partes), m, replace=False)
        duracoes = [tempos[c] * rng.uniform(0.8, 1.2) for c in esc]
        locais, tam = [], []
        for c in esc:
            if len(partes[c]) == 0:
                continue
            locais.append(treino_local(W, X[partes[c]], y[partes[c]], epocas, eta, lote, 0.0, seed * 1000 + r * 50 + int(c)))
            tam.append(len(partes[c]))
        p = np.array(tam) / sum(tam)
        W_novo = sum(pi * Wi for pi, Wi in zip(p, locais))
        fim = t + max(duracoes)                                   # a rodada espera pelo mais lento
        while prox < fim:
            curva.append((prox, acuracia(W, Xte, yte))); prox += 1.0
        W, t, r = W_novo, fim, r + 1
    return W, np.array(curva)


def simular_assincrono(partes, X, y, Xte, yte, modo, tempos, T=60.0, K=5, alfa=0.6, a_atraso=0.5, epocas=3,
                       eta=0.1, lote=32, seed=0, classes=10):
    """Aprendizado federado assíncrono em tempo virtual. Cada cliente treina continuamente, a partir da versão do
    modelo que encontrou ao começar, e entrega o resultado quando termina (ele pode estar 'atrasado').
    modo='fedasync': mistura imediata W <- (1 - a) W + a Wk, com a = alfa (1 + atraso)^(-a_atraso)  (Xie et al., 2019)
    modo='fedbuff' : guarda K atualizações (deltas) em um buffer e só então aplica a média, cada delta
                     dividido por sqrt(1 + atraso)  (Nguyen et al., 2022)"""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes)); versao, n_eventos = 0, 0
    duracao = lambda c: tempos[c] * rng.uniform(0.8, 1.2)
    fila = [(duracao(c), c, c, W.copy(), 0) for c in range(len(partes))]      # (fim, desempate, cliente, modelo de partida, versão)
    heapq.heapify(fila)
    buffer, curva, prox, t = [], [], 0.0, 0.0
    while fila and t < T:
        t, _, c, Wp, v0 = heapq.heappop(fila)
        Wc = Wp
        if len(partes[c]) > 0:
            Wc = treino_local(Wp, X[partes[c]], y[partes[c]], epocas, eta, lote, 0.0, seed * 1000 + n_eventos % 997)
        n_eventos += 1
        atraso = versao - v0                                       # quantas versões do servidor passaram enquanto o cliente treinava
        if modo == "fedasync":
            a = alfa * (1 + atraso) ** (-a_atraso)
            W = (1 - a) * W + a * Wc; versao += 1
        else:
            buffer.append((Wc - Wp) / np.sqrt(1 + atraso))
            if len(buffer) == K:
                W = W + np.mean(buffer, axis=0); buffer = []; versao += 1
        while prox <= t:
            curva.append((prox, acuracia(W, Xte, yte))); prox += 1.0
        heapq.heappush(fila, (t + duracao(c), n_eventos + 10_000, c, W.copy(), versao))
    return W, np.array(curva)


# ============================================================ 12.3 Seleção de clientes e viés de participação
def classe_dominante(partes, y):
    """Classe mais frequente em cada cliente (usada para criar disponibilidade correlacionada com os dados)."""
    return np.array([np.bincount(y[p], minlength=10).argmax() if len(p) else -1 for p in partes])


def fedavg_selecao(treino, X, y, Xte, yte, politica, disp, m=4, rodadas=60, epocas=3, eta=0.1, lote=32,
                   seed=0, candidatos=10, classes=10):
    """FedAvg com política de seleção de clientes; 'disp[c]' é a probabilidade de o cliente c estar disponível numa rodada.
    uniforme  : sorteia m clientes entre todos (ideal e irreal)
    disponivel: sorteia m entre os disponíveis (o que acontece na prática)
    ponderado : como 'disponivel', mas divide o peso n_k de cada cliente pela sua disponibilidade (correção por probabilidade)
    poder     : 'power-of-choice' (Cho et al., 2020): sorteia candidatos disponíveis e escolhe os m de maior perda local"""
    rng = np.random.default_rng(seed)
    K = len(treino); W = np.zeros((X.shape[1], classes))
    for r in range(rodadas):
        if politica == "uniforme":
            esc = rng.choice(K, m, replace=False)
        else:
            livres = [c for c in range(K) if rng.random() < disp[c] and len(treino[c]) > 0]
            if len(livres) < m:
                livres = list(range(K))
            if politica == "poder":
                cand = rng.choice(livres, min(candidatos, len(livres)), replace=False)
                perdas = [perda(W, X[treino[c]], y[treino[c]]) for c in cand]
                esc = cand[np.argsort(perdas)[::-1][:m]]
            else:
                esc = rng.choice(livres, min(m, len(livres)), replace=False)
        locais, pesos = [], []
        for c in esc:
            locais.append(treino_local(W, X[treino[c]], y[treino[c]], epocas, eta, lote, 0.0, seed * 1000 + r * 50 + int(c)))
            pesos.append(len(treino[c]) / (disp[c] if politica == "ponderado" else 1.0))
        p = np.array(pesos) / sum(pesos)
        W = sum(pi * Wi for pi, Wi in zip(p, locais))
    return W


# ============================================================ 12.4 Federações com "estilo" próprio e clientes novos
def federacao_com_estilo(Xbruto, partes, seed=0, intensidade=1.0):
    """Dá a cada cliente um 'estilo' (ganho e deslocamento dos pixels, como a letra de um escritor).
    Devolve a matriz com os atributos já 'estilizados' (mesma ordem das linhas de Xbruto) e o estilo de cada cliente."""
    rng = np.random.default_rng(seed)
    K = len(partes)
    ganho = 1 + intensidade * rng.uniform(-0.5, 0.6, K)
    desloc = intensidade * rng.normal(0, 2.0, K)
    Xs = Xbruto.copy().astype(float)
    for c, p in enumerate(partes):
        Xs[p] = np.clip(ganho[c] * Xbruto[p] + desloc[c], 0, None)
    return Xs, ganho, desloc


def ajustar_padronizador(Xbruto, treino):
    """Padronização federada (somas dos clientes de TREINO). Devolve uma função que aplica a mesma transformação
    a qualquer dado novo, inclusive de clientes que nunca participaram do treino, e acrescenta a coluna de viés."""
    n = sum(len(p) for p in treino)
    mu = sum(Xbruto[p].sum(0) for p in treino) / n
    dp = np.sqrt(np.maximum(sum((Xbruto[p] ** 2).sum(0) for p in treino) / n - mu ** 2, 0.0)); dp[dp == 0] = 1.0
    return lambda A: np.hstack([(A - mu) / dp, np.ones((len(A), 1))])


# ============================================================ 12.5 FedBN: normalização que fica em casa
def iniciar_mlp_bn(d_in, d_ocul=32, classes=10, seed=0):
    """Rede com uma camada oculta e normalização em lote: parâmetros W1, g (escala), b (deslocamento), W2, c."""
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_ocul)), "g": np.ones(d_ocul), "b": np.zeros(d_ocul),
            "W2": rng.normal(0, np.sqrt(2 / d_ocul), (d_ocul, classes)), "c": np.zeros(classes)}


def frente_bn(P, X, mu=None, var=None, eps=1e-5):
    """Passagem direta. Sem mu e var, a normalização usa as estatísticas do próprio lote."""
    Z = X @ P["W1"]
    if mu is None:
        mu, var = Z.mean(0), Z.var(0)
    Zh = (Z - mu) / np.sqrt(var + eps)
    A = P["g"] * Zh + P["b"]
    H = np.maximum(A, 0)
    return H @ P["W2"] + P["c"], (Z, Zh, A, H, var, eps)


def passo_bn(P, X, y, eta):
    """Um passo de gradiente com todos os dados do cliente (as estatísticas da normalização são as locais)."""
    L, (Z, Zh, A, H, var, eps) = frente_bn(P, X)
    n = len(y); S = softmax(L); S[np.arange(n), y] -= 1; S /= n
    dA = (S @ P["W2"].T) * (A > 0)
    dZh = dA * P["g"]
    dZ = (dZh - dZh.mean(0) - Zh * (dZh * Zh).mean(0)) / np.sqrt(var + eps)      # retropropagação pela normalização
    grad = {"W1": X.T @ dZ, "g": (dA * Zh).sum(0), "b": dA.sum(0), "W2": H.T @ S, "c": S.sum(0)}
    return {k: P[k] - eta * grad[k] for k in P}


def estatisticas_bn(P, X):
    Z = X @ P["W1"]
    return Z.mean(0), Z.var(0)


def fedbn_treinar(Xc, yc, modo, rodadas=30, passos=10, eta=0.3, seed=0):
    """Treina a rede federada com 'modo' = 'fedavg' (tudo é compartilhado, inclusive g, b e as estatísticas) ou
    'fedbn' (g, b e as estatísticas da normalização ficam em cada cliente). Devolve uma função que prevê para o cliente c."""
    K = len(Xc); n = np.array([len(a) for a in Xc], float); p = n / n.sum()
    G = iniciar_mlp_bn(Xc[0].shape[1], seed=seed)
    local = [dict(G) for _ in range(K)]
    for _ in range(rodadas):
        novos = []
        for c in range(K):
            P = {k: (local[c][k] if modo == "fedbn" and k in ("g", "b") else G[k]).copy() for k in G}
            for _ in range(passos):
                P = passo_bn(P, Xc[c], yc[c], eta)
            novos.append(P)
        for k in G:
            if modo == "fedbn" and k in ("g", "b"):
                continue
            G[k] = sum(p[c] * novos[c][k] for c in range(K))
        if modo == "fedbn":
            local = [{**novos[c], **{k: G[k] for k in ("W1", "W2", "c")}} for c in range(K)]
    def prever(c, Xnovo):
        P = {**G, **{k: local[c][k] for k in ("g", "b")}} if modo == "fedbn" else G
        if modo == "fedbn":
            mu, var = estatisticas_bn(P, Xc[c])                      # estatísticas locais
        else:
            est = [estatisticas_bn(P, a) for a in Xc]                # média das estatísticas, como no FedAvg
            mu = sum(p[j] * est[j][0] for j in range(K)); var = sum(p[j] * est[j][1] for j in range(K))
        return frente_bn(P, Xnovo, mu, var)[0].argmax(1)
    return prever


# ============================================================ 12.6 Desaprendizado federado
def fedavg_historico(partes, X, y, rodadas=30, epocas=5, eta=0.1, lote=32, seed=0, classes=10, W0=None):
    """FedAvg que guarda, rodada a rodada, o modelo global e as atualizações (deltas) de cada cliente."""
    W = np.zeros((X.shape[1], classes)) if W0 is None else W0.copy()
    modelos, deltas = [W.copy()], []
    for r in range(rodadas):
        d_r = {}
        for c, p in enumerate(partes):
            if len(p) > 0:
                d_r[c] = treino_local(W, X[p], y[p], epocas, eta, lote, 0.0, seed * 1000 + r * 50 + c) - W
        tot = sum(len(partes[c]) for c in d_r)
        W = W + sum(len(partes[c]) / tot * d for c, d in d_r.items())
        modelos.append(W.copy()); deltas.append(d_r)
    return W, modelos, deltas


def sem_cliente(partes, removido):
    return [p if c != removido else np.array([], dtype=int) for c, p in enumerate(partes)]


def subtrair_contribuicao(W, partes, removido, deltas):
    """Aproximação barata: tira de W a soma das atualizações (ponderadas) enviadas pelo cliente removido."""
    total = sum(len(p) for p in partes)
    return W - sum(len(partes[removido]) / total * d[removido] for d in deltas if removido in d)


def calibrar_federaser(partes, removido, X, y, modelos, deltas, epocas_cal=1, eta=0.1, lote=32, seed=0):
    """Aproximação no espírito do FedEraser (Liu et al., 2021). Reconstrói o treino a partir do modelo inicial: a cada rodada
    usa a NORMA da atualização antiga dos clientes restantes e, como DIREÇÃO, um treino curto de calibração feito agora."""
    resto = [c for c in range(len(partes)) if c != removido and len(partes[c]) > 0]
    tot = sum(len(partes[c]) for c in resto)
    W = modelos[0].copy()
    for r, d_r in enumerate(deltas):
        antiga = sum(len(partes[c]) / tot * d_r[c] for c in resto if c in d_r)
        nova = sum(len(partes[c]) / tot * (treino_local(W, X[partes[c]], y[partes[c]], epocas_cal, eta, lote, 0.0,
                                                          seed * 1000 + r * 50 + c) - W) for c in resto)
        W = W + np.linalg.norm(antiga) * nova / (np.linalg.norm(nova) + 1e-12)
    return W


# ============================================================ 12.7 Semissupervisionado: pseudorrótulos
def fedavg_pseudorotulos(partes, rotulados, X, y, Xte, yte, limiar=0.9, rodadas=40, aquece=10, epocas=3,
                         eta=0.1, lote=32, seed=0, classes=10):
    """Só os clientes em 'rotulados' têm rótulos. Depois de 'aquece' rodadas, os demais treinam com pseudorrótulos
    (previsões do modelo global com confiança >= limiar). Devolve o modelo, a acurácia e a precisão dos pseudorrótulos
    aceitos na última rodada (medida com rótulos que o sistema não enxerga)."""
    W = np.zeros((X.shape[1], classes)); precisao = float("nan")
    for r in range(rodadas):
        locais, pesos, acertos, aceitos = [], [], 0, 0
        for c, p in enumerate(partes):
            if len(p) == 0:
                continue
            if c in rotulados:
                Xc, yc = X[p], y[p]
            elif r >= aquece:
                P = softmax(X[p] @ W); sel = P.max(1) >= limiar
                if sel.sum() == 0:
                    continue
                Xc, yc = X[p][sel], P.argmax(1)[sel]
                acertos += int((yc == y[p][sel]).sum()); aceitos += int(sel.sum())
            else:
                continue
            locais.append(treino_local(W, Xc, yc, epocas, eta, lote, 0.0, seed * 1000 + r * 50 + c)); pesos.append(len(yc))
        pk = np.array(pesos) / sum(pesos)
        W = sum(a * b for a, b in zip(pk, locais))
        if aceitos:
            precisao = acertos / aceitos
    return W, acuracia(W, Xte, yte), precisao


# ============================================================ 12.8 Multirrótulo e hierarquia
def rotulos_multi(y):
    """Três atributos derivados do dígito: é par, é maior ou igual a 5, é primo."""
    return np.stack([(y % 2 == 0), (y >= 5), np.isin(y, [2, 3, 5, 7])], axis=1).astype(float)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def treino_local_multi(W, X, Y, M, epocas=3, eta=0.1, lote=32, seed=0):
    """Regressão logística multirrótulo (uma saída sigmoide por rótulo). M[i, j] = 1 se o rótulo j do exemplo i foi anotado."""
    rng = np.random.default_rng(seed); W = W.copy()
    for _ in range(epocas):
        ordem = rng.permutation(len(X))
        for i in range(0, len(X), lote):
            b = ordem[i:i + lote]
            W -= eta * X[b].T @ ((sigmoide(X[b] @ W) - Y[b]) * M[b]) / len(b)      # a perda só conta os rótulos anotados
    return W


def fedavg_multi(treino, anota, X, Y, ausente_como_zero=False, agregacao="por_rotulo", rodadas=40, epocas=3,
                 eta=0.1, lote=32, seed=0):
    """FedAvg multirrótulo em que cada cliente anota só alguns rótulos (anota[c] é a máscara dos rótulos que ele anota).
    ausente_como_zero=True: trata o rótulo não anotado como 'ausente' (valor 0), o erro mais comum.
    agregacao='por_rotulo': cada coluna é a média ponderada só dos clientes que anotam aquele rótulo.
    agregacao='global'    : média ponderada por n_k de todos os clientes, ignorando quem anota o quê."""
    L = Y.shape[1]; W = np.zeros((X.shape[1], L))
    for r in range(rodadas):
        locais, ns, masc = [], [], []
        for c, p in enumerate(treino):
            if len(p) == 0:
                continue
            Yc, M = Y[p].copy(), np.tile(anota[c], (len(p), 1)).astype(float)
            if ausente_como_zero:
                Yc, M = Yc * M, np.ones_like(M)
            locais.append(treino_local_multi(W, X[p], Yc, M, epocas, eta, lote, seed * 1000 + r * 50 + c))
            ns.append(len(p)); masc.append(anota[c])
        ns, masc = np.array(ns, float), np.array(masc, float)
        if agregacao == "global":
            W = sum(n / ns.sum() * Wc for n, Wc in zip(ns, locais))
        else:
            for j in range(L):
                w = ns * masc[:, j]
                if w.sum() > 0:
                    W[:, j] = sum(wi / w.sum() * Wc[:, j] for wi, Wc in zip(w, locais))
    return W


def metricas_multi(W, X, Y):
    """Acurácia e F1 de cada rótulo."""
    P = (sigmoide(X @ W) > 0.5).astype(float)
    tp = (P * Y).sum(0)
    return (P == Y).mean(0), 2 * tp / (P.sum(0) + Y.sum(0) + 1e-12)


# ============================================================ 12.9 Deriva de conceito
def fedavg_com_deriva(partes, X, y, Xte, yte, perm, r_deriva=25, drift_clientes=None, estrategia="continuar",
                      rodadas=60, epocas=2, eta=0.1, lote=32, seed=0, classes=10, fator=2.0):
    """FedAvg em fluxo. Na rodada r_deriva, os rótulos dos clientes em 'drift_clientes' (todos, se None) passam a ser perm[y]:
    mudança de conceito. estrategia='continuar' segue treinando o mesmo modelo; 'reiniciar' zera o modelo quando a perda média
    no começo da rodada passa de 'fator' vezes a menor perda recente. Devolve o modelo, a acurácia por rodada (medida no teste
    do conceito ATUAL) e a rodada em que o detector disparou."""
    K = len(partes); drift = set(range(K)) if drift_clientes is None else set(drift_clientes)
    W = np.zeros((X.shape[1], classes)); hist, perdas, disparo = [], [], None
    for r in range(rodadas):
        yr = [perm[y[p]] if (r >= r_deriva and c in drift) else y[p] for c, p in enumerate(partes)]
        tam = np.array([len(p) for p in partes], float)
        perdas.append(sum(t * perda(W, X[p], yc) for t, p, yc in zip(tam, partes, yr) if t > 0) / tam.sum())
        if estrategia == "reiniciar" and r >= 6 and disparo is None and perdas[-1] > fator * min(perdas[-6:-1]) and perdas[-1] > 0.5:
            W = np.zeros_like(W); disparo = r                        # detector de deriva disparou
        locais = [treino_local(W, X[p], yc, epocas, eta, lote, 0.0, seed * 1000 + r * 50 + c)
                  for c, (p, yc) in enumerate(zip(partes, yr)) if len(p) > 0]
        pk = np.array([t for t in tam if t > 0]); pk /= pk.sum()
        W = sum(a * b for a, b in zip(pk, locais))
        hist.append(acuracia(W, Xte, perm[yte] if r >= r_deriva else yte))
    return W, np.array(hist), disparo
