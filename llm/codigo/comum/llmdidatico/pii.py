"""Capítulo 17: detector/mascarador de dados pessoais por regras e gerador de textos SINTÉTICOS (mesmo código do texto do capítulo).
Os CPFs são números sorteados com dígitos verificadores válidos; podem coincidir com CPFs reais por acaso e não devem ser usados para outro fim."""
import random, re
import numpy as np

import random, re

PRIMEIROS = ["Maria", "João", "Ana", "Pedro", "Lucas", "Beatriz", "Carlos", "Fernanda", "Rafael", "Juliana", "Marcos", "Camila", "Paulo", "Larissa", "Gabriel", "Patrícia", "Bruno", "Aline", "Ricardo", "Tatiane"]
PRIMEIROS_FORA = ["Thayná", "Iolanda", "Ravi", "Nayara", "Ubirajara", "Zélia"]      # nomes que o detector NÃO conhece
SOBRENOMES = ["Souza", "Lima", "Oliveira", "Pereira", "Costa", "Almeida", "Ribeiro", "Barbosa", "Cardoso", "Moura", "Teixeira", "Nogueira"]

def dv_cpf(nove):
    d = []
    for k in (10, 11):
        base = nove + "".join(map(str, d))
        r = (sum(int(c) * (k - i) for i, c in enumerate(base)) * 10) % 11
        d.append(0 if r >= 10 else r)
    return "".join(map(str, d))

def cpf_valido(dig):
    return len(dig) == 11 and len(set(dig)) > 1 and dv_cpf(dig[:9]) == dig[9:]

def gerar_cpf(rnd, formatado=True):
    n = "".join(rnd.choice("0123456789") for _ in range(9)); c = n + dv_cpf(n)
    return f"{c[:3]}.{c[3:6]}.{c[6:9]}-{c[9:]}" if formatado else c

TEMPLATES = [
    "Paciente {NOME}, CPF {CPF}, nascido em {NASC}, compareceu ao ambulatório (caso sintético). Contato: {TEL}. Prontuário {PRONT}.",
    "A aluna {NOME} solicitou trancamento. E-mail para resposta: {EMAIL}. Protocolo {PROT}, prazo até {PRAZO}.",
    "Encaminhar a {NOME} o ofício do processo {PROT}, com prazo até {PRAZO}. Endereço: CEP {CEP}, Rua das Acácias, 120.",
    "O servidor {NOME}, telefone {TEL}, deve ir à Avenida Maria Moura até {PRAZO} retirar o documento nº {PROT}.",
    "Relato sintético: {NOME} informa cefaleia há três dias. CPF informado: {CPF}. Pedido nº {PROT}.",
    "O Hospital Santa Clara Norte atende de segunda a sexta. Protocolo {PROT}; o prazo termina em {PRAZO}.",
    "Contato institucional: secretaria, telefone (85) 3000-0000 (linha pública fictícia). Edital {PROT}, inscrições até {PRAZO}.",
    "Prontuário {PRONT}: {NOME} (nascida em {NASC}) retorna em {PRAZO}. Responder para {EMAIL}.",
]
TEMPLATES_ADV = [      # versões "difíceis": formatos incomuns, nomes fora do dicionário, minúsculas, por extenso
    "Paciente {NOME}, cpf {CPF_ESP} (sem pontuação, com espaços), contato {TEL_ESP}.",
    "Responder para {EMAIL_ESC}. Titular: {NOME_MIN}. Protocolo {PROT}.",
    "Sra. {NOME_FORA} informou o CPF {CPF} por mensagem; nascida em {NASC}. Prontuário {PRONT}.",
    "Contato do responsável: {TEL}. Nome: {NOME_FORA}. E-mail {EMAIL}. Prazo {PRAZO}.",
    "Meu nome é {NOME_MIN} e meu cpf é {CPF_EXT}.",
    "A coordenadora {NOME} (e-mail {EMAIL}) pediu o dossiê de {NOME_FORA}, CEP {CEP}.",
]

def gerar_textos(n, seed, adversarial=False):
    rnd = random.Random(seed); out = []
    pal = "zero um dois três quatro cinco seis sete oito nove".split()
    for _ in range(n):
        t = rnd.choice(TEMPLATES_ADV if adversarial else TEMPLATES)
        nome = f"{rnd.choice(PRIMEIROS)} {rnd.choice(SOBRENOMES)}"; fora = f"{rnd.choice(PRIMEIROS_FORA)} {rnd.choice(SOBRENOMES)}"
        c = gerar_cpf(rnd, False); u = f"{rnd.choice(PRIMEIROS).lower()}.{rnd.choice(SOBRENOMES).lower()}"
        tel = lambda: f"({rnd.choice([85, 11, 21, 31, 61])}) 9{rnd.randint(1000, 9999)}-{rnd.randint(1000, 9999)}"
        data = lambda: f"{rnd.randint(1, 28):02d}/{rnd.randint(1, 12):02d}/{rnd.randint(1950, 2005)}"
        mapa = {"NOME": (nome, "NOME"), "NOME_MIN": (nome.lower(), "NOME"), "NOME_FORA": (fora, "NOME"),
                "CPF": (gerar_cpf(rnd), "CPF"), "CPF_ESP": (f"{c[:3]} {c[3:6]} {c[6:9]} {c[9:]}", "CPF"),
                "CPF_EXT": (" ".join(pal[int(ch)] for ch in c), "CPF"),
                "TEL": (tel(), "TEL"), "TEL_ESP": (f"85 9{rnd.randint(1000, 9999)} {rnd.randint(1000, 9999)}", "TEL"),
                "EMAIL": (u + "@exemplo.example", "EMAIL"), "EMAIL_ESC": (u + " arroba exemplo ponto example", "EMAIL"),
                "NASC": (data(), "NASC"), "CEP": (f"{rnd.randint(60000, 69999)}-{rnd.randint(100, 999)}", "CEP"),
                "PRONT": (f"PRT-{rnd.randint(100000, 999999)}", "PRONT"),
                "PROT": (f"{rnd.randint(2024, 2026)}{rnd.randint(0, 9999999):07d}", None),      # parece número, mas NÃO é dado pessoal
                "PRAZO": (f"{rnd.randint(1, 28):02d}/{rnd.randint(1, 12):02d}/2026", None)}     # data de prazo, NÃO é nascimento
        texto, spans, pos = "", [], 0
        for m in re.finditer(r"\{(\w+)\}", t):
            texto += t[pos:m.start()]; v, tipo = mapa[m.group(1)]
            if tipo: spans.append((len(texto), len(texto) + len(v), tipo))
            texto += v; pos = m.end()
        out.append((texto + t[pos:], spans))
    return out


DICIONARIO = {p.lower() for p in PRIMEIROS} | {"clara"}
MAI, MIN = "A-ZÁÉÍÓÚÂÊÔÃÕÇ", "a-záéíóúâêôãõç"
ER = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "TEL": re.compile(r"\(?\b\d{2}\)?[ -]?9\d{4}-?\d{4}\b"),
    "CEP": re.compile(r"\b\d{5}-\d{3}\b"),
    "PRONT": re.compile(r"\bPRT-\d{6}\b"),
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "NASC": re.compile(r"nasc[a-z]{2,4} em (\d{2}/\d{2}/\d{4})"),
    "NOME": re.compile(rf"(?=\b([{MAI}][{MIN}]+) ([{MAI}][{MIN}]+)\b)"),
}

def detectar(texto):
    achados = []
    for tipo, er in ER.items():
        for m in er.finditer(texto):
            if tipo == "NASC": ini, fim = m.start(1), m.end(1)           # só a data, sem o contexto "nascido em"
            elif tipo == "NOME": ini, fim = m.start(), m.end(2)          # nome + sobrenome
            else: ini, fim = m.start(), m.end()
            if tipo == "CPF" and not cpf_valido(re.sub(r"\D", "", m.group())): continue     # validação dos dígitos verificadores
            if tipo == "NOME" and m.group(1).lower() not in DICIONARIO: continue            # só nomes que o dicionário conhece
            achados.append((ini, fim, tipo))
    achados.sort(key=lambda a: (a[0], -(a[1] - a[0]))); final = []
    for a in achados:
        if not final or a[0] >= final[-1][1]: final.append(a)       # remove sobreposições
    return final

def contar(dados):
    """Por tipo: [verdadeiros positivos, falsos positivos, falsos negativos], comparando intervalos exatos."""
    tot, por_doc = {}, []
    for texto, ver in dados:
        pred, ver = set(detectar(texto)), set(ver); d = {}
        for t in {x[2] for x in pred | ver}:
            p, v = {x for x in pred if x[2] == t}, {x for x in ver if x[2] == t}
            d[t] = (len(p & v), len(p - v), len(v - p))
            a = tot.setdefault(t, [0, 0, 0])
            for i in range(3): a[i] += d[t][i]
        por_doc.append(d)
    return tot, por_doc

def prf(tp, fp, fn):
    return (tp / (tp + fp) if tp + fp else float("nan")), (tp / (tp + fn) if tp + fn else float("nan"))

def ic_micro(por_doc, B=500, seed=0):
    """IC95% de precisão e revocação micro, reamostrando DOCUMENTOS."""
    rng = np.random.default_rng(seed); n = len(por_doc)
    M = np.array([[sum(v[i] for v in d.values()) for i in range(3)] for d in por_doc])
    P, R = [], []
    for _ in range(B):
        s = M[rng.integers(0, n, n)].sum(axis=0); P.append(s[0] / (s[0] + s[1])); R.append(s[0] / (s[0] + s[2]))
    return np.quantile(P, [0.025, 0.975]), np.quantile(R, [0.025, 0.975])


import hmac

CHAVE = b"chave-secreta-de-demonstracao"       # em produção: cofre de segredos, nunca no código

def mascarar(texto, achados, modo="rotulo"):
    saida = texto
    for ini, fim, tipo in sorted(achados, reverse=True):
        if modo == "rotulo":
            sub = f"[{tipo}]"
        else:
            sub = f"<{tipo}:" + hmac.new(CHAVE, texto[ini:fim].lower().encode(), "sha256").hexdigest()[:8] + ">"
        saida = saida[:ini] + sub + saida[fim:]
    return saida

