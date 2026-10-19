"""Capítulo 18 (Projeto 5): páginas de DOCUMENTAÇÃO TÉCNICA sintéticas de uma API FICTÍCIA ('API Faróis'), geradas por modelos de frase.

Nada aqui descreve um produto real. Cada página tem um gabarito (`gold`) com campos estruturados e um resumo de referência. Há três
estilos de redação: 'tabela' (campos rotulados), 'prosa' (a mesma informação em frases) e 'ruidoso' (rótulos variantes, erros de
digitação e campos ausentes), para que a extração tenha erros naturais e mensuráveis."""
import random

ACOES = [("criar", "POST", "Cria", ["titulo", "texto", "prazo"]), ("listar", "GET", "Lista", ["pagina", "limite"]),
         ("obter", "GET", "Obtém", ["id"]), ("atualizar", "PUT", "Atualiza", ["id", "titulo", "texto"]),
         ("remover", "DELETE", "Remove", ["id"]), ("arquivar", "POST", "Arquiva", ["id", "motivo"])]
RECURSOS = [("lembrete", "lembretes"), ("etiqueta", "etiquetas"), ("anexo", "anexos"), ("comentario", "comentarios"), ("grupo", "grupos"), ("alerta", "alertas")]
_SUB = {"Cria": "a criação de", "Lista": "a consulta paginada de", "Obtém": "a leitura de", "Atualiza": "a alteração de", "Remove": "a exclusão de", "Arquiva": "o arquivamento de"}


def gerar_paginas(n=36, seed=0):
    """Lista de dicts: id, estilo, texto, gold (nome, metodo, caminho, parametros, versao_minima, autenticacao, limite_por_min), resumo."""
    rnd = random.Random(seed)
    combos = [(a, r) for a in ACOES for r in RECURSOS]
    rnd.shuffle(combos)
    out = []
    for k in range(n):
        (acao, met, verbo, params), (rs, rp) = combos[k % len(combos)]
        nome = f"{acao}_{rs}"
        caminho = f"/v{rnd.choice([1, 2, 3])}/{rp}" + ("/{id}" if "id" in params else "")
        ver = f"{rnd.randint(1, 3)}.{rnd.randint(0, 9)}"
        auth = rnd.random() < 0.8
        lim = rnd.choice([30, 60, 120, 300])
        gold = dict(nome=nome, metodo=met, caminho=caminho, parametros=params, versao_minima=ver, autenticacao=auth, limite_por_min=lim)
        resumo = f"{verbo} {rs}."
        estilo = ["tabela", "prosa", "ruidoso"][k % 3]
        pj = ", ".join(params)
        if estilo == "tabela":
            t = (f"# {nome}\n{verbo} {rs} na API Faróis. Use este recurso para {_SUB[verbo]} {rp}.\n"
                 f"Método: {met}\nCaminho: {caminho}\nParâmetros: {pj}\nVersão mínima: {ver}\nAutenticação: {'obrigatória' if auth else 'dispensada'}\nLimite: {lim} requisições por minuto")
        elif estilo == "prosa":
            t = (f"# {nome}\n{verbo} {rs} na API Faróis. A chamada usa o método {met} no caminho {caminho} e aceita os parâmetros {pj}. "
                 f"Está disponível a partir da versão {ver}. A autenticação é {'obrigatória' if auth else 'dispensada'}, e o limite é de {lim} requisições por minuto.")
        else:
            rot = rnd.choice([("Metodo", "Rota", "Params"), ("Verbo HTTP", "Endpoint", "Argumentos")])
            t = (f"# {nome}\n{verbo} {rs} na API Faróis.\n{rot[0]}: {met if rnd.random() < 0.85 else met.lower()}\n{rot[1]}: {caminho}\n{rot[2]}: {pj}\n"
                 + (f"Versão mínima: {ver}\n" if rnd.random() < 0.7 else "") + f"Autenticação: {'obrigatória' if auth else 'dispensada'}\nLimite: {lim} req/min")
        out.append(dict(id=f"P{k:02d}", estilo=estilo, texto=t, gold=gold, resumo=resumo))
    return out
