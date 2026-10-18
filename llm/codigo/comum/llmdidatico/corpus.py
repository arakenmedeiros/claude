"""Corpora sintéticos em português, gerados por regras com semente fixa (nenhum texto vem de fontes externas)."""
import random

# ---------------------------------------------------------------- mini-corpus para modelos de linguagem
_SUJ = [
    ("o estudante", ["lê", "escreve", "entrega", "procura", "empresta", "devolve", "estuda"]),
    ("a estudante", ["lê", "escreve", "entrega", "procura", "empresta", "devolve", "estuda"]),
    ("o professor", ["corrige", "explica", "escreve", "lê", "prepara", "entrega"]),
    ("a professora", ["corrige", "explica", "escreve", "lê", "prepara", "entrega"]),
    ("o bibliotecário", ["empresta", "procura", "organiza", "devolve"]),
    ("a coordenadora", ["assina", "lê", "organiza", "prepara"]),
]
_OBJ = {
    "lê": ["o livro", "o artigo", "o relatório", "a prova"],
    "escreve": ["o relatório", "o artigo", "a prova", "o trabalho"],
    "entrega": ["o trabalho", "o relatório", "a prova"],
    "procura": ["o livro", "o artigo"],
    "empresta": ["o livro"],
    "devolve": ["o livro", "o trabalho"],
    "estuda": ["o artigo", "o livro", "a prova"],
    "corrige": ["a prova", "o trabalho", "o relatório"],
    "explica": ["o artigo", "a prova"],
    "prepara": ["a prova", "a aula"],
    "organiza": ["o livro", "o relatório", "a aula"],
    "assina": ["o relatório", "o trabalho"],
}
_COMP = ["na biblioteca", "no laboratório", "em casa", "na sala de aula", "hoje", "ontem", "com atenção", "antes da prova", "depois da aula", "todos os dias"]


def frases_lm(n=3000, seed=0):
    """Gera n frases curtas (lista de listas de tokens, terminadas em '.')."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        suj, verbos = rnd.choice(_SUJ)
        v = rnd.choice(verbos)
        o = rnd.choice(_OBJ[v])
        toks = suj.split() + [v] + o.split()
        if rnd.random() < 0.8:
            toks += rnd.choice(_COMP).split()
        out.append(toks + ["."])
    return out


# ---------------------------------------------------------------- intenções de atendimento ao cliente
_INT = {
    "segunda_via": ["preciso da segunda via da {coisa}", "como emitir a segunda via da {coisa}?", "perdi minha {coisa}, pode reenviar?", "onde baixo a {coisa} deste mês?", "não recebi a {coisa}, mandem de novo"],
    "cancelamento": ["quero cancelar meu {prod}", "como faço para encerrar o {prod}?", "desejo cancelar o {prod} hoje", "não quero mais o {prod}, cancelem por favor", "preciso interromper o {prod} imediatamente"],
    "suporte_tecnico": ["o {prod} não funciona desde ontem", "meu {prod} está fora do ar", "não consigo conectar o {prod}", "o {prod} apresenta erro ao iniciar", "o {prod} está muito lento e trava"],
    "alteracao_cadastro": ["quero mudar meu endereço cadastrado", "preciso atualizar meu telefone no cadastro", "como altero o e-mail da minha conta?", "corrigir o nome que aparece no meu cadastro", "troquei de endereço e preciso atualizar os dados"],
    "rastreio_pedido": ["onde está meu pedido?", "quero rastrear a entrega do pedido {num}", "meu pedido {num} ainda não chegou", "qual o prazo de entrega do pedido {num}?", "a entrega está atrasada, preciso do código de rastreio"],
    "elogio": ["quero elogiar o atendimento de vocês", "parabéns pela rapidez, fiquei muito satisfeito", "o atendente foi excelente, obrigado", "ótimo serviço, recomendo a todos", "gostei muito do suporte que recebi"],
    "reclamacao": ["estou muito insatisfeito com o atendimento", "é um absurdo o que aconteceu, quero reclamar", "fui cobrado indevidamente e ninguém resolve", "péssimo serviço, vou procurar outra empresa", "já liguei três vezes e o problema continua"],
    "duvida_plano": ["qual a diferença entre os planos básico e premium?", "quanto custa o plano {plano}?", "o plano {plano} inclui suporte aos sábados?", "posso trocar do plano básico para o {plano}?", "quais benefícios tem o plano {plano}?"],
}
_FILL = {"coisa": ["fatura", "conta", "nota fiscal", "boleto"], "prod": ["serviço", "aplicativo", "plano", "contrato", "internet"], "num": ["1042", "8873", "2201", "5560"], "plano": ["premium", "básico", "família", "empresarial"]}
_PRE = ["", "", "olá, ", "bom dia, ", "por favor, ", "oi, ", "boa tarde, "]
_POS = ["", "", "", " obrigado", " urgente", " por favor"]


def _typo(s, rnd):
    if len(s) < 6 or rnd.random() > 0.15:
        return s
    i = rnd.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def intencoes(n_por_classe=120, seed=0):
    """Retorna (textos, rótulos) de mensagens curtas de atendimento, com ruído (prefixos, erros de digitação)."""
    rnd = random.Random(seed)
    X, y = [], []
    for lab, tpls in _INT.items():
        for _ in range(n_por_classe):
            t = rnd.choice(tpls)
            for k, v in _FILL.items():
                t = t.replace("{" + k + "}", rnd.choice(v))
            t = rnd.choice(_PRE) + t + rnd.choice(_POS)
            if rnd.random() < 0.5:
                t = t.lower()
            X.append(_typo(t, rnd)); y.append(lab)
    idx = list(range(len(X))); rnd.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]


def frase_valida(tokens):
    """Verdadeiro se a frase (lista de tokens terminada em '.') obedece à gramática que gera o corpus sintético."""
    s = " ".join(tokens)
    for suj, verbos in _SUJ:
        if not s.startswith(suj + " "):
            continue
        resto = s[len(suj) + 1:].split()
        if not resto or resto[0] not in verbos:
            return False
        v = resto[0]; tail = " ".join(resto[1:])
        for o in _OBJ[v]:
            if tail.startswith(o):
                r = tail[len(o):].strip()
                if r in [".", *[c + " ." for c in _COMP]]:
                    return True
        return False
    return False
