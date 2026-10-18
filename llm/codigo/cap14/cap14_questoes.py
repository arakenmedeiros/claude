"""Capítulo 14: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 15]
import ast, operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}

def calcular(expr):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 100:
                raise ValueError("expoente grande demais")
            try:
                return OPS[type(n.op)](a, b)
            except ZeroDivisionError:
                raise ValueError("divisão por zero")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("elemento não permitido: " + type(n).__name__)
    return ev(ast.parse(expr, mode="eval"))

for e in ["2*(3+4)", "2**10", "__import__('os')", "1/0"]:
    try:
        print(e, "->", calcular(e))
    except ValueError as erro:
        print(e, "-> recusado:", erro)
assert calcular("2*(3+4)") == 14 and calcular("2**10") == 1024

# %% [Questão discursiva 16]
def detectar_laco(chamadas, limite=3):
    seq = 1
    for i in range(1, len(chamadas)):
        seq = seq + 1 if chamadas[i] == chamadas[i - 1] else 1
        if seq >= limite:
            return i
    return None

a = ("buscar", '{"q": "multa"}'); b = ("calc", '{"e": "1+1"}')
print(detectar_laco([a, b, a, b, a]))          # alternância: sem laço
print(detectar_laco([a, a, b, a, a, a]))       # laço no índice 5
print(detectar_laco([a, a], limite=2))         # limite 2: índice 1
assert detectar_laco([a, b, a, b, a]) is None and detectar_laco([a, a, b, a, a, a]) == 5 and detectar_laco([a, a], limite=2) == 1

# %% [Questão discursiva 17]
def autorizar(politica, nome, args, confirmador=None):
    if nome not in politica["permitidas"]:
        return False, "ferramenta não autorizada"
    if nome == "enviar_email" and not str(args.get("para", "")).endswith(politica["dominio"]):
        return False, "destinatário fora do domínio permitido"
    if nome in politica["confirmar"]:
        if confirmador is None or not confirmador(nome, args):
            return False, "sem confirmação humana"
    return True, "ok"

pol = {"permitidas": {"buscar", "enviar_email", "protocolar"}, "confirmar": {"protocolar", "enviar_email"}, "dominio": "@ufa.exemplo.br"}
sim = lambda n, a: True
casos = [("apagar", {}, sim), ("enviar_email", {"para": "x@externo.org"}, sim), ("protocolar", {}, None), ("protocolar", {}, sim)]
for nome, args, conf in casos:
    print(nome, args, "->", autorizar(pol, nome, args, conf))
assert [autorizar(pol, n, a, c)[0] for n, a, c in casos] == [False, False, False, True]

# %% [Questão discursiva 18]
import numpy as np

rng = np.random.default_rng(0)
p_tarefa = rng.choice([0.2, 0.5, 0.9, 1.0], size=15)
exec_ = (rng.random((15, 20)) < p_tarefa[:, None]).astype(float)
taxas = exec_.mean(axis=1)

def ic(v, n=2000, seed=1):
    r = np.random.default_rng(seed)
    medias = [r.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return np.quantile(medias, [0.025, 0.975])

lo_t, hi_t = ic(taxas)
lo_e, hi_e = ic(exec_.ravel())
print(f"taxa média: {taxas.mean():.3f}")
print(f"(a) por tarefa    : [{lo_t:.3f}; {hi_t:.3f}] largura {hi_t - lo_t:.3f}")
print(f"(b) por execução  : [{lo_e:.3f}; {hi_e:.3f}] largura {hi_e - lo_e:.3f}")
assert (hi_t - lo_t) > (hi_e - lo_e)
