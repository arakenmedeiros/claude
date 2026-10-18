"""Otimizadores didáticos do livro (NumPy).

Todos recebem o vetor inicial theta0 e funções de perda/gradiente (ou resíduo/
Jacobiana, no caso de Levenberg-Marquardt). Backpropagation CALCULA o gradiente;
estes algoritmos o USAM para atualizar theta. São implementações didáticas do
livro, não funções do scikit-learn.

Retorno comum: objeto Resultado com theta, theta_melhor, historico (dict de
listas), n_iter e motivo da parada.
"""
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Resultado:
    theta: np.ndarray
    theta_melhor: np.ndarray
    historico: dict = field(default_factory=dict)
    n_iter: int = 0
    motivo: str = ""
    tempo: float = 0.0


class _Registro:
    """Histórico + critérios de parada + parada antecipada (early stopping)."""

    def __init__(self, theta0, fun, grad, val=None, paciencia=None,
                 tol_grad=1e-8, tol_perda=0.0, max_iter=1000):
        self.fun, self.grad, self.val = fun, grad, val
        self.paciencia, self.tol_grad, self.tol_perda = paciencia, tol_grad, tol_perda
        self.max_iter = max_iter
        self.h = {"perda": [], "norma_grad": [], "perda_val": [], "tempo": []}
        self.t0 = time.perf_counter()
        self.melhor_val = np.inf
        self.theta_melhor = theta0.copy()
        self.sem_melhora = 0
        self.motivo = "max_iter"

    def registrar(self, theta, E=None, g=None):
        E = self.fun(theta) if E is None else E
        self.h["perda"].append(float(E))
        if g is not None:
            self.h["norma_grad"].append(float(np.linalg.norm(g)))
        self.h["tempo"].append(time.perf_counter() - self.t0)
        if self.val is not None:
            ev = float(self.val(theta))
            self.h["perda_val"].append(ev)
            if ev < self.melhor_val - 1e-12:
                self.melhor_val, self.theta_melhor = ev, theta.copy()
                self.sem_melhora = 0
            else:
                self.sem_melhora += 1
        else:
            self.theta_melhor = theta.copy()   # sem validação: o melhor é o último
        return E

    def parar(self, g=None):
        if not np.isfinite(self.h["perda"][-1]):
            self.motivo = "perda_nao_finita"
            return True
        if g is not None and np.linalg.norm(g) < self.tol_grad:
            self.motivo = "gradiente_pequeno"
            return True
        if self.paciencia is not None and self.val is not None and self.sem_melhora >= self.paciencia:
            self.motivo = "parada_antecipada"
            return True
        if self.tol_perda and self.h["perda"][-1] < self.tol_perda:
            self.motivo = "perda_alvo"
            return True
        return False

    def fim(self, theta, n):
        return Resultado(theta, self.theta_melhor, self.h, n, self.motivo,
                         time.perf_counter() - self.t0)


# =========================================================== gradiente descendente
def gradiente_descendente(theta0, fun, grad, taxa=0.1, momento=0.0, max_iter=1000,
                          val=None, paciencia=None, tol_grad=1e-8, tol_perda=0.0):
    """Lote completo: theta <- theta - taxa*g + momento*(passo anterior)."""
    theta = theta0.copy()
    reg = _Registro(theta0, fun, grad, val, paciencia, tol_grad, tol_perda, max_iter)
    passo = np.zeros_like(theta)
    n = 0
    for n in range(1, max_iter + 1):
        g = grad(theta)
        reg.registrar(theta, g=g)
        if reg.parar(g):
            break
        passo = -taxa * g + momento * passo
        theta = theta + passo
    else:
        reg.registrar(theta, g=grad(theta))
    return reg.fim(theta, n)


def gd_estocastico(theta0, grad_amostras, N, fun, taxa=0.05, tamanho_lote=1, epocas=100,
                   semente=0, val=None, paciencia=None, momento=0.0):
    """Padrão a padrão (tamanho_lote=1), minilotes (1<tamanho_lote<N) ou lote
    (tamanho_lote=N). grad_amostras(theta, idx) devolve o gradiente da perda
    SOMADA sobre as amostras idx (sem dividir pelo tamanho do lote)."""
    rng = np.random.default_rng(semente)
    theta = theta0.copy()
    reg = _Registro(theta0, fun, None, val, paciencia, 0.0, 0.0, epocas)
    passo = np.zeros_like(theta)
    n = 0
    for n in range(1, epocas + 1):
        reg.registrar(theta)
        if reg.parar():
            break
        ordem = rng.permutation(N)
        for ini in range(0, N, tamanho_lote):
            idx = ordem[ini:ini + tamanho_lote]
            g = grad_amostras(theta, idx) / len(idx)    # média no lote
            passo = -taxa * g + momento * passo
            theta = theta + passo
    return reg.fim(theta, n)


# ===================================================================== Quickprop
def quickprop(theta0, fun, grad, epsilon=0.1, mu=1.75, decaimento=1e-4, max_iter=1000,
              val=None, paciencia=None, tol_grad=1e-8, tol_perda=0.0, den_min=1e-12, passo_max=None):
    """Quickprop (Fahlman, 1988), por parâmetro, em lote.

    Convenção: s = dE/dw (+ decaimento*w).  Passo quadrático
        dw(t) = s(t) / (s(t-1) - s(t)) * dw(t-1),
    limitado por mu*|dw(t-1)| via fator de encolhimento mu/(1+mu).
    Implementação didática do livro; segue a lógica do código de referência
    de Fahlman descrita na literatura (a conferir no relatório original)."""
    theta = theta0.copy()
    reg = _Registro(theta0, fun, grad, val, paciencia, tol_grad, tol_perda, max_iter)
    dw = np.zeros_like(theta)
    s_ant = np.zeros_like(theta)
    encolhe = mu / (1.0 + mu)
    n = 0
    for n in range(1, max_iter + 1):
        g = grad(theta)
        reg.registrar(theta, g=g)
        if reg.parar(g):
            break
        s = g + decaimento * theta
        passo = np.zeros_like(theta)
        neg, pos, zer = dw < 0, dw > 0, dw == 0
        # termo de gradiente quando a inclinação "pede" continuar no mesmo sentido
        passo[neg & (s > 0)] -= epsilon * s[neg & (s > 0)]
        passo[pos & (s < 0)] -= epsilon * s[pos & (s < 0)]
        # arranque: passo de gradiente descendente quando não há passo anterior
        passo[zer] -= epsilon * s[zer]
        # passo quadrático ou crescimento máximo
        cresce = ((neg & (s >= encolhe * s_ant)) | (pos & (s <= encolhe * s_ant)))
        passo[cresce] += mu * dw[cresce]
        quad = (neg | pos) & ~cresce
        den = s_ant[quad] - s[quad]
        den = np.where(np.abs(den) < den_min, np.sign(den + 1e-300) * den_min, den)
        passo[quad] += dw[quad] * s[quad] / den
        if passo_max is not None:                 # salvaguarda opcional (não faz parte da formulação original)
            passo = np.clip(passo, -passo_max, passo_max)
        s_ant = s
        dw = passo
        theta = theta + dw
    else:
        reg.registrar(theta, g=grad(theta))
    return reg.fim(theta, n)


# ======================================================================== RPROP
VARIANTES_RPROP = ("rprop+", "rprop-", "irprop+", "irprop-")


def rprop(theta0, fun, grad, variante="irprop-", eta_mais=1.2, eta_menos=0.5,
          delta0=0.1, delta_min=1e-6, delta_max=50.0, max_iter=1000,
          val=None, paciencia=None, tol_grad=1e-8, tol_perda=0.0):
    """RPROP em lote, quatro variantes SEPARADAS (Riedmiller & Braun, 1993;
    Igel & Huesken, 2000/2003):

      rprop+  : retrocesso do peso incondicional na mudança de sinal
      rprop-  : sem retrocesso; usa o novo passo reduzido na mesma iteração
      irprop+ : retrocesso apenas se a perda AUMENTOU
      irprop- : sem retrocesso; zera o gradiente na mudança de sinal

    Só o SINAL do gradiente é usado; delta é o passo individual de cada peso."""
    assert variante in VARIANTES_RPROP
    theta = theta0.copy()
    reg = _Registro(theta0, fun, grad, val, paciencia, tol_grad, tol_perda, max_iter)
    delta = np.full_like(theta, delta0)
    g_ant = np.zeros_like(theta)
    dw_ant = np.zeros_like(theta)          # deslocamento efetivamente aplicado antes
    E_ant = np.inf
    n = 0
    for n in range(1, max_iter + 1):
        g = grad(theta)
        E = reg.registrar(theta, g=g)
        if reg.parar(g):
            break
        prod = g_ant * g
        pos, neg = prod > 0, prod < 0
        delta = np.where(pos, np.minimum(delta * eta_mais, delta_max), delta)
        delta = np.where(neg, np.maximum(delta * eta_menos, delta_min), delta)
        dw = np.zeros_like(theta)
        if variante == "rprop+":
            dw = np.where(neg, -dw_ant, -np.sign(g) * delta)       # retrocede sempre
            g = np.where(neg, 0.0, g)
        elif variante == "rprop-":
            dw = -np.sign(g) * delta                                # sem retrocesso
        elif variante == "irprop-":
            g = np.where(neg, 0.0, g)                               # zera g na troca
            dw = -np.sign(g) * delta
        elif variante == "irprop+":
            if E > E_ant:                                           # perda subiu
                dw = np.where(neg, -dw_ant, -np.sign(g) * delta)
            else:                                                   # perda não subiu
                dw = np.where(neg, 0.0, -np.sign(g) * delta)
            g = np.where(neg, 0.0, g)
        theta = theta + dw
        g_ant, dw_ant, E_ant = g, dw, E
    else:
        reg.registrar(theta, g=grad(theta))
    return reg.fim(theta, n)


# ========================================================= Levenberg-Marquardt
def levenberg_marquardt(theta0, res, jac, mu0=1e-3, mu_inc=10.0, mu_dec=0.1,
                        mu_max=1e10, escala="identidade", max_iter=200,
                        tol_grad=1e-8, tol_passo=1e-12, tol_perda=0.0,
                        val=None, paciencia=None, max_rejeicoes=20):
    """LM para minimizar E = 1/2 ||r||^2 (r = y - t).

    Passo: (J^T J + mu D) delta = - J^T r, resolvido por Cholesky/solve
    (nunca por inversa explícita).  D = I ('identidade', Levenberg 1944) ou
    D = diag(J^T J) ('marquardt', Marquardt 1963).  Passo aceito se E diminui;
    então mu *= mu_dec; senão mu *= mu_inc e o passo é recalculado."""
    theta = theta0.copy()
    fun = lambda th: 0.5 * float(res(th) @ res(th))
    reg = _Registro(theta0, fun, None, val, paciencia, tol_grad, tol_perda, max_iter)
    reg.h["mu"] = []
    mu = mu0
    Dpiso = np.zeros_like(theta)
    n = 0
    r = res(theta)
    E = 0.5 * float(r @ r)
    for n in range(1, max_iter + 1):
        J = jac(theta)
        g = J.T @ r
        reg.registrar(theta, E=E, g=g)
        reg.h["mu"].append(mu)
        if reg.parar(g):
            break
        JTJ = J.T @ J
        Dpiso = np.maximum(Dpiso, np.diag(JTJ))
        aceito = False
        for _ in range(max_rejeicoes):
            D = np.eye(len(theta)) if escala == "identidade" else np.diag(np.maximum(Dpiso, 1e-12))
            try:
                delta = np.linalg.solve(JTJ + mu * D, -g)
            except np.linalg.LinAlgError:
                mu *= mu_inc
                continue
            r_novo = res(theta + delta)
            E_novo = 0.5 * float(r_novo @ r_novo)
            if np.isfinite(E_novo) and E_novo < E:
                theta, r, E = theta + delta, r_novo, E_novo
                mu = max(mu * mu_dec, 1e-20)
                aceito = True
                break
            mu *= mu_inc
            if mu > mu_max:
                break
        if not aceito:
            reg.motivo = "mu_maximo" if mu > mu_max else "sem_passo_aceito"
            break
        if np.linalg.norm(delta) < tol_passo:
            reg.motivo = "passo_pequeno"
            break
    else:
        reg.registrar(theta, E=E, g=jac(theta).T @ r)
    return reg.fim(theta, n)


# ================================================================== Delta-Bar-Delta
def delta_delta(theta0, fun, grad, alfa0=0.01, gama=0.01, max_iter=1000,
                tol_grad=1e-8, alfa_min=0.0, val=None, paciencia=None):
    """Regra delta-delta (Jacobs, 1988): alfa(t+1) = alfa(t) + gama*g(t)*g(t-1).
    Sem proteção, alfa pode ficar negativo ou explodir; 'alfa_min' é só um
    recurso didático para evitar taxa negativa (não faz parte da regra)."""
    theta = theta0.copy()
    reg = _Registro(theta0, fun, grad, val, paciencia, tol_grad, 0.0, max_iter)
    alfa = np.full_like(theta, alfa0)
    g_ant = np.zeros_like(theta)
    reg.h["alfa_min"], reg.h["alfa_med"], reg.h["alfa_max"] = [], [], []
    n = 0
    for n in range(1, max_iter + 1):
        g = grad(theta)
        reg.registrar(theta, g=g)
        if reg.parar(g):
            break
        alfa = np.maximum(alfa + gama * g * g_ant, alfa_min)
        reg.h["alfa_min"].append(float(alfa.min())); reg.h["alfa_med"].append(float(np.median(alfa))); reg.h["alfa_max"].append(float(alfa.max()))
        theta = theta - alfa * g
        g_ant = g
    else:
        reg.registrar(theta, g=grad(theta))
    return reg.fim(theta, n)


def delta_bar_delta(theta0, fun, grad, alfa0=0.01, kappa=0.001, phi=0.1, teta=0.7,
                    max_iter=1000, tol_grad=1e-8, val=None, paciencia=None):
    """DBD (Jacobs, 1988). g_bar(t) = (1-teta) g(t) + teta g_bar(t-1).
       se g_bar(t-1)*g(t) > 0: alfa += kappa      (aumento aditivo)
       se g_bar(t-1)*g(t) < 0: alfa -= phi*alfa   (redução multiplicativa)
       caso contrário: alfa inalterado.   Peso: w -= alfa*g."""
    theta = theta0.copy()
    reg = _Registro(theta0, fun, grad, val, paciencia, tol_grad, 0.0, max_iter)
    alfa = np.full_like(theta, alfa0)
    gbar = np.zeros_like(theta)
    reg.h["alfa_min"], reg.h["alfa_med"], reg.h["alfa_max"] = [], [], []
    n = 0
    for n in range(1, max_iter + 1):
        g = grad(theta)
        reg.registrar(theta, g=g)
        if reg.parar(g):
            break
        prod = gbar * g
        alfa = np.where(prod > 0, alfa + kappa, np.where(prod < 0, alfa * (1 - phi), alfa))
        reg.h["alfa_min"].append(float(alfa.min())); reg.h["alfa_med"].append(float(np.median(alfa))); reg.h["alfa_max"].append(float(alfa.max()))
        theta = theta - alfa * g
        gbar = (1 - teta) * g + teta * gbar
    else:
        reg.registrar(theta, g=grad(theta))
    return reg.fim(theta, n)


def delta_bar_delta_estendido(theta0, fun, grad, alfa0=0.01, mu0=0.0,
                              kappa_a=0.01, gama_a=1.0, phi_a=0.1, alfa_max=1.0,
                              kappa_m=0.01, gama_m=1.0, phi_m=0.1, mu_max=0.9,
                              teta=0.7, max_iter=1000, tol_grad=1e-8,
                              val=None, paciencia=None):
    """EDBD (Minai & Williams, 1990), formulação didática: taxa E momento
    individuais; os aumentos decaem exponencialmente com |g_bar|:
       aumento:  kappa*exp(-gama*|g_bar(t)|);  redução: -phi*valor.
    Valores limitados por alfa_max e mu_max. Passo: dw = -alfa*g + mu*dw_ant.
    Detalhes (p.ex. mecanismo de recuperação do artigo) a conferir no original;
    constantes padrão aqui são didáticas, NÃO as do artigo."""
    theta = theta0.copy()
    reg = _Registro(theta0, fun, grad, val, paciencia, tol_grad, 0.0, max_iter)
    alfa = np.full_like(theta, alfa0)
    mom = np.full_like(theta, mu0)
    gbar = np.zeros_like(theta)
    dw = np.zeros_like(theta)
    reg.h["alfa_min"], reg.h["alfa_med"], reg.h["alfa_max"], reg.h["mom_med"] = [], [], [], []
    n = 0
    for n in range(1, max_iter + 1):
        g = grad(theta)
        reg.registrar(theta, g=g)
        if reg.parar(g):
            break
        prod = gbar * g
        ag = np.abs(gbar)
        alfa = np.where(prod > 0, np.minimum(alfa + kappa_a * np.exp(-gama_a * ag), alfa_max),
                        np.where(prod < 0, alfa * (1 - phi_a), alfa))
        mom = np.where(prod > 0, np.minimum(mom + kappa_m * np.exp(-gama_m * ag), mu_max),
                       np.where(prod < 0, mom * (1 - phi_m), mom))
        reg.h["alfa_min"].append(float(alfa.min())); reg.h["alfa_med"].append(float(np.median(alfa))); reg.h["alfa_max"].append(float(alfa.max())); reg.h["mom_med"].append(float(np.median(mom)))
        dw = -alfa * g + mom * dw
        theta = theta + dw
        gbar = (1 - teta) * g + teta * gbar
    else:
        reg.registrar(theta, g=grad(theta))
    return reg.fim(theta, n)


# ============================================================ Gauss-Newton (sem amortecimento)
def gauss_newton(theta0, res, jac, max_iter=100, tol_grad=1e-8, passo_max_norma=None):
    """Gauss-Newton puro: resolve J d = -r por mínimos quadrados lineares (np.linalg.lstsq).
    Sem amortecimento e sem busca em linha: pode divergir ou falhar se J^T J for singular."""
    theta = theta0.copy()
    fun = lambda th: 0.5 * float(res(th) @ res(th))
    reg = _Registro(theta0, fun, None, None, None, tol_grad, 0.0, max_iter)
    n = 0
    for n in range(1, max_iter + 1):
        r, J = res(theta), jac(theta)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(J))):
            reg.motivo = "valores_nao_finitos"          # o passo anterior explodiu
            break
        g = J.T @ r
        reg.registrar(theta, E=0.5 * float(r @ r), g=g)
        if reg.parar(g):
            break
        try:
            d = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            reg.motivo = "falha_numerica_lstsq"
            break
        if passo_max_norma is not None and np.linalg.norm(d) > passo_max_norma:
            d = d * passo_max_norma / np.linalg.norm(d)
        theta = theta + d
    return reg.fim(theta, n)


# ====================================== Levenberg-Marquardt com razão de ganho (Nielsen / Madsen et al.)
def levenberg_marquardt_nielsen(theta0, res, jac, tau=1e-3, max_iter=200, tol_grad=1e-8, tol_passo=1e-12):
    """LM com atualização de mu pela razão de ganho rho = (E - E_novo) / (L(0) - L(d)),
    L(d) = 1/2 ||r + J d||^2  (modelo linearizado). Se rho > 0: aceita e
    mu <- mu * max(1/3, 1 - (2 rho - 1)^3), nu <- 2; senão rejeita e mu <- nu mu, nu <- 2 nu.
    Mu inicial: tau * max(diag(J^T J)).   Implementação didática do livro (esquema de Madsen, Nielsen e Tingleff)."""
    theta = theta0.copy()
    fun = lambda th: 0.5 * float(res(th) @ res(th))
    reg = _Registro(theta0, fun, None, None, None, tol_grad, 0.0, max_iter)
    reg.h["mu"], reg.h["rho"] = [], []
    r = res(theta); E = 0.5 * float(r @ r); J = jac(theta); JTJ = J.T @ J; g = J.T @ r
    mu = tau * np.max(np.diag(JTJ)); nu = 2.0; n = 0
    for n in range(1, max_iter + 1):
        reg.registrar(theta, E=E, g=g); reg.h["mu"].append(mu)
        if reg.parar(g):
            break
        try:
            d = np.linalg.solve(JTJ + mu * np.eye(len(theta)), -g)
        except np.linalg.LinAlgError:
            mu *= nu; nu *= 2; reg.h["rho"].append(np.nan); continue
        if np.linalg.norm(d) < tol_passo * (np.linalg.norm(theta) + tol_passo):
            reg.motivo = "passo_pequeno"; break
        r_novo = res(theta + d); E_novo = 0.5 * float(r_novo @ r_novo)
        ganho_previsto = 0.5 * float(d @ (mu * d - g))
        rho = (E - E_novo) / ganho_previsto if ganho_previsto > 0 else -1.0
        reg.h["rho"].append(rho)
        if rho > 0 and np.isfinite(E_novo):
            theta, r, E = theta + d, r_novo, E_novo
            J = jac(theta); JTJ = J.T @ J; g = J.T @ r
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3); nu = 2.0
        else:
            mu *= nu; nu *= 2.0
    else:
        reg.registrar(theta, E=E, g=g)
    return reg.fim(theta, n)
