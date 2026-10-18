"""Capítulo 11: códigos de referência das questões discursivas de programação."""

# %% [Questão discursiva 10]
def hit_at_k(ranking, relevantes, k):
    return float(len(set(ranking[:k]) & set(relevantes)) > 0)

def recall_do_documento(rankings, relevantes_por_consulta, k):
    v = [hit_at_k(r, rel, k) for r, rel in zip(rankings, relevantes_por_consulta) if rel]
    return sum(v) / len(v) if v else None

rankings = [["D2", "D1", "D3"], ["D5", "D4", "D9"], ["D7", "D8", "D1"], ["D1", "D2", "D3"]]
relevantes = [["D2"], ["D9"], [], ["D3", "D4"]]
r1 = recall_do_documento(rankings, relevantes, 1)
r3 = recall_do_documento(rankings, relevantes, 3)
assert abs(r1 - 1 / 3) < 1e-9 and r3 == 1.0
print("recall@1 =", round(r1, 3), "| recall@3 =", r3)

# %% [Questão discursiva 11]
def custo(texto):
    return len(texto.split()) + 10

def montar_orcamento(fragmentos, limite):
    incluidos, cortados, usado = [], [], 0
    for fid, texto in fragmentos:
        c = custo(texto)
        if usado + c <= limite:
            incluidos.append(fid); usado += c
        else:
            cortados.append(fid)
    return incluidos, cortados, usado

frs = [("D10", "palavra " * 60), ("D60", "palavra " * 12), ("D59", "palavra " * 5), ("D45", "palavra " * 3)]
inc, cort, usado = montar_orcamento(frs, 50)
assert inc == ["D60", "D59", "D45"] and cort == ["D10"] and usado == 22 + 15 + 13
print(inc, cort, usado)

# %% [Questão discursiva 12]
def fundir_rrf(listas, k, c=60):
    s = {}
    for L in listas:
        for pos, d in enumerate(L, 1):
            s[d] = s.get(d, 0.0) + 1.0 / (c + pos)
    return [d for d, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))][:k]

A, B = ["D1", "D2", "D3"], ["D3", "D1", "D4"]
fund = fundir_rrf([A, B], 4)
assert fund == ["D1", "D3", "D2", "D4"], fund
assert fundir_rrf([["D1"], ["D2", "D1"]], 2)[0] == "D1"
print(fund)

# %% [Questão discursiva 13]
import re

def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()

def verificar_citacoes(resposta, ids_contexto, textos_contexto):
    cit = re.findall(r"\[(D\d+)\]", resposta)
    validas = bool(cit) and all(c in ids_contexto for c in cit)
    afirm = norm(re.sub(r"\s*\[D\d+\]", "", resposta))
    literal = validas and any(afirm in norm(textos_contexto[c]) for c in cit)
    return {"validas": validas, "literal": bool(literal)}

ctx = {"D12": "O atraso na devolução gera multa de R$ 1,50 por dia e por exemplar."}
ok = verificar_citacoes("multa de R$ 1,50  por dia [D12]", list(ctx), ctx)
mau_id = verificar_citacoes("multa de R$ 1,50 por dia [D99]", list(ctx), ctx)
sem = verificar_citacoes("multa de R$ 1,50 por dia", list(ctx), ctx)
assert ok == {"validas": True, "literal": True}
assert mau_id == {"validas": False, "literal": False} and sem == {"validas": False, "literal": False}
print(ok, mau_id, sem)

# %% [Questão discursiva 14]
def filtro_antes(ranking, papel, k):
    return [d for d, papeis in ranking if papel in papeis][:k]

def filtro_depois(ranking, papel, k):
    return [d for d, papeis in ranking[:k] if papel in papeis]

ranking = [("D31", {"pos"}), ("D01", {"grad", "pos"}), ("D33", {"pos"}), ("D02", {"grad", "pos"}), ("D03", {"grad", "pos"})]
a = filtro_antes(ranking, "grad", 3)
d = filtro_depois(ranking, "grad", 3)
assert a == ["D01", "D02", "D03"] and d == ["D01"]
print("antes:", a, "| depois:", d)
