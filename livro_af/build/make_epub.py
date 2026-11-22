#!/usr/bin/env python3
"""Gera o EPUB do livro a partir das mesmas fontes (cap/*.md, ex*.txt, pr/*.txt) usadas pelo build.js.
O texto é justificado por CSS. Uso: python3 make_epub.py saida.epub"""
import html, json, os, re, sys, uuid, zipfile, datetime

B = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(B, "..")
CAP, FIGS = os.path.join(ROOT, "cap"), os.path.join(ROOT, "figs")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Aprendizado_Federado.epub")
TITLE, SUBTITLE, AUTHOR, YEAR = "Aprendizado Federado", "Fundamentos, Prática e Fronteiras", "Araken de Medeiros Santos", "2026"
PR_OUT = json.load(open(os.path.join(B, "pr_out.json"), encoding="utf-8")) if os.path.exists(os.path.join(B, "pr_out.json")) else {}

def rd(p): return open(os.path.join(CAP, p), encoding="utf-8").read().replace("\r", "")
E = lambda s: html.escape(s, quote=False)

# ------------------------------------------------------------ inline
INL = re.compile(r"(\*\*(?:[^*]|\*(?!\*)[^*]+\*)+\*\*|\*[^*\s][^*]*\*|`[^`]+`)")
def inline(t):
    out, last = [], 0
    for m in INL.finditer(t):
        out.append(E(t[last:m.start()])); s = m.group(0)
        if s.startswith("**"): out.append("<strong>" + inline(s[2:-2]) + "</strong>")
        elif s.startswith("`"): out.append("<code>" + E(s[1:-1]) + "</code>")
        else: out.append("<em>" + E(s[1:-1]) + "</em>")
        last = m.end()
    out.append(E(t[last:])); return "".join(out)

FR = re.compile(r"_\(([^)]*)\)|_([A-Za-z0-9ÀÿΑ-ωα-ω]+)|\^\(([^)]*)\)|\^([A-Za-z0-9]+)")
def formula_html(t):
    out, last = [], 0
    for m in FR.finditer(t):
        out.append(E(t[last:m.start()]))
        sub = m.group(1) is not None or m.group(2) is not None
        v = next(g for g in m.groups() if g is not None)
        out.append(("<sub>" if sub else "<sup>") + E(v) + ("</sub>" if sub else "</sup>")); last = m.end()
    out.append(E(t[last:])); return "".join(out)

# ------------------------------------------------------------ parser do markdown do livro
def parse_md(text):
    L = text.split("\n"); blocks, meta, i, para = [], {"obj": []}, 0, []
    def flush():
        nonlocal para
        if para: blocks.append(("p", " ".join(para))); para = []
    while i < len(L):
        ln = L[i]
        if ln.startswith("@part:"): meta["part"] = ln[6:].strip(); i += 1; continue
        if ln.startswith("@banner:"): meta["banner"] = int(ln[8:]); i += 1; continue
        if ln.startswith("@obj:"): meta["obj"].append(ln[5:].strip()); i += 1; continue
        if re.match(r"^# ", ln): flush(); meta["title"] = ln[2:].strip(); i += 1; continue
        if ln.startswith("### "): flush(); blocks.append(("h4", ln[4:].strip())); i += 1; continue
        if ln.startswith("## "): flush(); blocks.append(("h3", ln[3:].strip())); i += 1; continue
        if ln.startswith("```"):
            flush(); lang = ln[3:].strip(); code = []; i += 1
            while i < len(L) and not L[i].startswith("```"): code.append(L[i]); i += 1
            i += 1; blocks.append(("code", lang, code)); continue
        if ln.startswith(":::"):
            flush(); m = re.match(r"^:::(\w+)\s*(.*)$", ln); body = []; i += 1
            while i < len(L) and L[i].strip() != ":::": body.append(L[i]); i += 1
            i += 1; blocks.append(("callout", m.group(1), m.group(2), body)); continue
        m = re.match(r"^!\[(\w+)\]\((.*)\)\s*$", ln)
        if m: flush(); blocks.append(("fig", m.group(1), m.group(2))); i += 1; continue
        if ln.startswith("$$"):
            flush(); blocks.append(("formula", re.sub(r"\s*\$\$$", "", re.sub(r"^\$\$\s*", "", ln)))); i += 1; continue
        if ln.startswith("|"):
            flush(); rows = []
            while i < len(L) and L[i].startswith("|"): rows.append(L[i]); i += 1
            rows = [[c.strip() for c in re.sub(r"^\||\|\s*$", "", r).split("|")] for r in rows if not re.match(r"^\|\s*-{2,}", r)]
            blocks.append(("table", rows)); continue
        if re.match(r"^- ", ln):
            flush(); items = []
            while i < len(L) and re.match(r"^- ", L[i]): items.append(L[i][2:]); i += 1
            blocks.append(("ul", items)); continue
        if re.match(r"^\d+\. ", ln):
            flush(); items = []
            while i < len(L) and re.match(r"^\d+\. ", L[i]): items.append(re.sub(r"^\d+\.\s+", "", L[i])); i += 1
            blocks.append(("ol", items)); continue
        if ln.strip() == "": flush(); i += 1; continue
        if ln.startswith("> "): flush(); blocks.append(("quote", ln[2:])); i += 1; continue
        para.append(ln.strip()); i += 1
    flush(); return meta, blocks

# ------------------------------------------------------------ questões e exercícios práticos
def mulberry32(a):
    M = 0xFFFFFFFF; st = [a & M]
    imul = lambda x, y: ((x & M) * (y & M)) & M
    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & M; a_ = st[0]
        t = imul(a_ ^ (a_ >> 15), 1 | a_)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M) ^ t
        return (((t ^ (t >> 14)) & M)) / 4294967296
    return nxt

def parse_ex(text, seed):
    rnd = mulberry32(seed); mc, dc = [], []
    for bl in re.split(r"\n\s*\n", text.strip()):
        L = bl.split("\n")
        if L[0].startswith("MC:"):
            opts = [re.sub(r"^[abcd]\)\s*", "", o) for o in L[1:5]]
            rm = re.match(r"^R:\s*([abcd])\s*\|\s*(.*)$", L[5]); correct = "abcd".index(rm.group(1))
            idx = [0, 1, 2, 3]
            for k in (3, 2, 1):
                j = int(rnd() * (k + 1)); idx[k], idx[j] = idx[j], idx[k]
            mc.append({"stem": L[0][3:].strip(), "opts": [opts[x] for x in idx], "ans": "abcd"[idx.index(correct)], "why": rm.group(2).strip()})
        elif L[0].startswith("DISC:"):
            dc.append({"stem": L[0][5:].strip(), "ans": re.sub(r"^R:\s*", "", L[1]).strip()})
    return mc, dc

def parse_pr(text):
    items, alem, cur, mode = [], [], None, None
    for ln in text.replace("\r", "").split("\n"):
        if ln.startswith("@@EX ") or ln.startswith("@@PROJ "):
            kind = "EX" if ln.startswith("@@EX ") else "PROJ"
            parts = [x.strip() for x in ln[ln.index(" ") + 1:].split("|")]
            cur = ({"kind": kind, "num": int(parts[0]), "title": parts[1], "level": parts[2] if len(parts) > 2 else ""} if kind == "EX"
                   else {"kind": kind, "num": 0, "title": parts[0], "level": parts[1] if len(parts) > 1 else ""})
            cur.update(stmt=[], sol=[], rubrica=[], note=[]); items.append(cur); mode = "stmt"; continue
        if ln.startswith("@@ALEM"): cur = None; mode = "alem"; continue
        if ln.startswith("@@SOL"): mode = "sol"; continue
        if ln.startswith("@@RUBRICA"): mode = "rubrica"; continue
        if ln.startswith("@@NOTE"):
            mode = "note"; rest = ln[6:].strip()
            if rest and cur: cur["note"].append(rest)
            continue
        if mode == "alem":
            if ln.strip(): alem.append(ln)
            continue
        if cur is not None and mode: cur[mode].append(ln)
    return items, alem

def code_of(it):
    m = re.search(r"```python\n([\s\S]*?)```", "\n".join(it["sol"])); return m.group(1).rstrip("\n").split("\n") if m else []
def saida_of(ch, it):
    out = PR_OUT.get(f"{ch}-{it['kind']}{it['num']}")
    if out is None: return None
    L = [l if len(l) <= 118 else l[:117] + "…" for l in out.rstrip().split("\n")]
    return L[:48] + ["… (saída truncada)"] if len(L) > 48 else L

# ------------------------------------------------------------ HTML
LABEL = {"dica": "DICA", "atencao": "ATENÇÃO", "cotidiano": "NO SEU DIA A DIA", "saibamais": "SAIBA MAIS", "exemplo": "EXEMPLO", "memorize": "MEMORIZE"}
used_imgs, nav = set(), []     # nav: (nível, título, arquivo#id)
_id = [0]
def hid(): _id[0] += 1; return f"h{_id[0]}"

def callout_html(kind, title, body):
    label = LABEL.get(kind, "DICA"); norm = lambda t: re.sub(r"[\u0300-\u036f]", "", __import__("unicodedata").normalize("NFD", t)).lower()
    dup = norm(title).startswith(norm(label.split()[0])) or (kind == "cotidiano" and norm(title).startswith("no seu dia")) or (kind == "exemplo" and norm(title).startswith("exemplo"))
    out = [f'<aside class="callout {kind}"><p class="ctitle">' + ("" if dup else f'<span class="clabel">{E(label)}</span> ') + inline(title) + "</p>"]
    para, i = [], 0
    def fl():
        nonlocal para
        if para: out.append("<p>" + inline(" ".join(para)) + "</p>"); para = []
    while i < len(body):
        ln = body[i]
        if re.match(r"^- ", ln):
            fl(); out.append("<ul>")
            while i < len(body) and re.match(r"^- ", body[i]): out.append("<li>" + inline(body[i][2:]) + "</li>"); i += 1
            out.append("</ul>"); continue
        if ln.strip() == "": fl(); i += 1; continue
        para.append(ln.strip()); i += 1
    fl(); out.append("</aside>"); return "".join(out)

def blocks_html(blocks, fname=None, level_off=0):
    o = []
    for b in blocks:
        t = b[0]
        if t in ("h3", "h4"):
            i_ = hid(); lv = 2 if t == "h3" else 3
            o.append(f'<h{lv} id="{i_}">{inline(b[1])}</h{lv}>')
            if fname: nav.append((lv, re.sub(r"[*`]", "", b[1]), f"{fname}#{i_}"))
        elif t == "p": o.append("<p>" + inline(b[1]) + "</p>")
        elif t == "quote": o.append("<blockquote><p>" + inline(b[1]) + "</p></blockquote>")
        elif t == "ul": o.append("<ul>" + "".join("<li>" + inline(x) + "</li>" for x in b[1]) + "</ul>")
        elif t == "ol": o.append("<ol>" + "".join("<li>" + inline(x) + "</li>" for x in b[1]) + "</ol>")
        elif t == "callout": o.append(callout_html(b[1], b[2], b[3]))
        elif t == "code": o.append(code_html(b[1], b[2]))
        elif t == "formula": o.append('<p class="formula">' + formula_html(b[1]) + "</p>")
        elif t == "table":
            rows = b[1]
            o.append("<table><thead><tr>" + "".join("<th>" + inline(c) + "</th>" for c in rows[0]) + "</tr></thead><tbody>" +
                     "".join("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in r) + "</tr>" for r in rows[1:]) + "</tbody></table>")
        elif t == "fig":
            used_imgs.add(b[1]); o.append(f'<figure><img src="images/{b[1]}.png" alt="{html.escape(b[2], quote=True)}"/><figcaption>{inline(b[2])}</figcaption></figure>')
    return "\n".join(o)

def code_html(lang, code):
    return f'<div class="codebox"><p class="clang">{E((lang or "código").upper())}</p><pre><code>{E(chr(10).join(code))}</code></pre></div>'

CSS = """
body { font-family: Georgia, "Times New Roman", serif; line-height: 1.5; text-align: justify; hyphens: auto; -webkit-hyphens: auto; -epub-hyphens: auto; margin: 0 4%; color: #1f2937; }
p, li, blockquote, td, dd { text-align: justify; }
h1, h2, h3 { font-family: Helvetica, Arial, sans-serif; text-align: left; hyphens: none; -webkit-hyphens: none; page-break-after: avoid; color: #312e81; }
h1 { font-size: 1.7em; margin: 1.2em 0 .6em; border-bottom: 3px solid #f59e0b; padding-bottom: .2em; }
h2 { font-size: 1.3em; margin: 1.4em 0 .5em; color: #4f46e5; }
h3 { font-size: 1.1em; margin: 1.1em 0 .4em; }
img { max-width: 100%; height: auto; }
figure { margin: 1em 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: .85em; font-style: italic; color: #6b7280; text-align: center; margin-top: .3em; }
pre { font-family: "Courier New", Courier, monospace; font-size: .72em; line-height: 1.3; white-space: pre-wrap; word-wrap: break-word; text-align: left; margin: 0; hyphens: none; -webkit-hyphens: none; }
code { font-family: "Courier New", Courier, monospace; font-size: .88em; color: #b91c1c; background: #f3f4f6; hyphens: none; -webkit-hyphens: none; }
pre code { color: #1f2937; background: none; font-size: 1em; }
.codebox { background: #f8fafc; border: 1px solid #e5e7eb; border-left: 4px solid #4f46e5; padding: .5em .7em; margin: .8em 0; }
.clang { font-family: Helvetica, Arial, sans-serif; font-size: .65em; font-weight: bold; color: #4f46e5; letter-spacing: .08em; margin: 0 0 .3em; text-align: left; }
.callout { border-left: 5px solid #4f46e5; background: #eef2ff; padding: .5em .9em; margin: 1em 0; page-break-inside: avoid; }
.callout.dica { border-color: #10b981; background: #ecfdf5; } .callout.atencao { border-color: #f59e0b; background: #fffbeb; }
.callout.cotidiano { border-color: #06b6d4; background: #ecfeff; } .callout.saibamais { border-color: #8b5cf6; background: #f5f3ff; }
.callout.memorize { border-color: #ec4899; background: #fdf2f8; }
.ctitle { font-family: Helvetica, Arial, sans-serif; font-weight: bold; margin: 0 0 .3em; text-align: left; }
.clabel { font-size: .75em; letter-spacing: .06em; opacity: .85; }
.callout p, .callout li { margin: .3em 0; }
.formula { text-align: center; background: #eef2ff; border-top: 1px solid #c7d2fe; border-bottom: 1px solid #c7d2fe; padding: .4em; font-family: "Cambria Math", Cambria, Georgia, serif; color: #312e81; }
table { border-collapse: collapse; width: 100%; font-size: .85em; margin: 1em 0; }
th { background: #4f46e5; color: #fff; text-align: left; padding: .35em .5em; border: 1px solid #e5e7eb; }
td { padding: .35em .5em; border: 1px solid #e5e7eb; text-align: left; vertical-align: top; }
.objetivos { border-left: 5px solid #4f46e5; background: #eef2ff; padding: .6em 1em; margin: 1em 0; }
.nivel { font-family: Helvetica, Arial, sans-serif; font-size: .7em; font-weight: bold; color: #4f46e5; letter-spacing: .08em; margin: 0; text-align: left; }
.opt { margin: .1em 0 .1em 1.5em; text-indent: -1.1em; }
.q { margin: .9em 0 .2em; }
.capa { text-align: center; } .capa img { max-height: 98vh; }
.titulo { text-align: center; margin-top: 18%; } .titulo h1 { border: none; text-align: center; color: #312e81; font-size: 2.2em; text-transform: uppercase; }
.titulo .sub { font-style: italic; color: #6b7280; font-size: 1.3em; text-align: center; }
.titulo .autor { font-weight: bold; font-size: 1.4em; margin-top: 2em; text-align: center; }
.ref { margin: .5em 0 .5em 1.5em; text-indent: -1.5em; font-size: .92em; }
.parte { margin-top: 20%; } .parte .pn { font-weight: bold; letter-spacing: .1em; font-family: Helvetica, Arial, sans-serif; text-align: left; }
.banner { width: 100%; margin: 0 0 .6em; }
"""

def xhtml(title, body, cls=""):
    return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="pt-BR" xml:lang="pt-BR">\n'
            f'<head><meta charset="utf-8"/><title>{E(title)}</title><link rel="stylesheet" type="text/css" href="style.css"/></head>\n<body{(" class=%s" % chr(34) + cls + chr(34)) if cls else ""}>\n{body}\n</body>\n</html>\n')

files = []   # (nome, título, xhtml, spine?)
def add(name, title, body, cls=""): files.append((name, title, xhtml(title, body, cls)))

# ------------------------------------------------------------ montagem
PART_DESC = {
    "Parte I — Fundamentos": "Por que treinar modelos sem juntar os dados, como o treino de modelos funciona por dentro e como nasce o algoritmo FedAvg, a base de quase tudo o que vem depois.",
    "Parte II — Desafios do Treino Federado": "O que muda quando os dados de cada participante são diferentes (não-IID), quando a rede é lenta e quando cada pessoa precisa de um modelo só seu.",
    "Parte III — Privacidade e Segurança": "O que ainda pode vazar mesmo sem enviar dados, como se proteger (privacidade diferencial, agregação segura, criptografia) e como resistir a participantes mal-intencionados.",
    "Parte IV — Arquiteturas, Aplicações e Fronteiras": "Outros desenhos de aprendizado federado (vertical, descentralizado, split learning), aprendizado não supervisionado federado, modelos de fundação, ferramentas, aplicações reais, regulação e ética, além de tópicos do mundo real: assincronia, participação, desaprendizado, partições naturais, deriva e rótulos escassos.",
    "Parte V — Projeto Final": "Um estudo completo que integra simulação federada, dados não-IID, comunicação, personalização, privacidade e robustez, da pergunta de negócio ao relatório.",
}
chapters = []
for n in range(1, 14):
    nn = f"{n:02d}"
    if not os.path.exists(os.path.join(CAP, f"ch{nn}.md")): continue
    meta, blocks = parse_md(rd(f"ch{nn}.md")); mc, dc = parse_ex(rd(f"ex{nn}.txt"), 1000 + n)
    prf = os.path.join(CAP, "pr", f"pr{nn}.txt"); pr = parse_pr(rd(f"pr/pr{nn}.txt")) if os.path.exists(prf) else None
    chapters.append({"n": n, "meta": meta, "blocks": blocks, "mc": mc, "dc": dc, "pr": pr})

# capa e folha de rosto
add("capa.xhtml", "Capa", '<div class="capa"><img src="images/capa.png" alt="Capa do livro"/></div>')
add("rosto.xhtml", "Folha de rosto", f'<div class="titulo"><h1>{E(TITLE)}</h1><p class="sub">{E(SUBTITLE)}</p><p class="autor">{E(AUTHOR)}</p><p>{YEAR}</p></div>')
files_toc_index = len(files)       # o sumário entra aqui (depois de gerar a navegação)

# apresentação
meta, blocks = parse_md(rd("apresentacao.md"))
nav.append((1, "Apresentação", "apresentacao.xhtml"))
add("apresentacao.xhtml", "Apresentação", "<h1>Apresentação</h1>\n" + blocks_html(blocks, "apresentacao.xhtml"))

def exercises_html(c):
    o = [f'<h2 id="{(i := hid())}">Exercícios de múltipla escolha</h2>', '<p><em>Escolha <strong>uma</strong> alternativa em cada questão. O gabarito está no Apêndice A.</em></p>']
    nav.append((2, "Exercícios de múltipla escolha", f"cap{c['n']:02d}.xhtml#{i}"))
    for k, q in enumerate(c["mc"], 1):
        o.append(f'<p class="q"><strong>Questão {k}.</strong> {inline(q["stem"])}</p>')
        o += [f'<p class="opt"><strong>{"abcd"[j]})</strong> {inline(op)}</p>' for j, op in enumerate(q["opts"])]
    i = hid(); nav.append((2, "Exercícios discursivos", f"cap{c['n']:02d}.xhtml#{i}"))
    o.append(f'<h2 id="{i}">Exercícios discursivos</h2><p><em>Responda com suas palavras, em poucas linhas. Compare depois com a resposta-modelo do Apêndice A.</em></p>')
    o.append("<ol>" + "".join("<li>" + inline(q["stem"]) + "</li>" for q in c["dc"]) + "</ol>")
    return "\n".join(o)

def praticos_html(c):
    if not c["pr"]: return ""
    items, alem = c["pr"]; exs = [x for x in items if x["kind"] == "EX"]; proj = next((x for x in items if x["kind"] == "PROJ"), None); fn = f"cap{c['n']:02d}.xhtml"; o = []
    if exs:
        i = hid(); nav.append((2, "Exercícios práticos em Python (scikit-learn)", f"{fn}#{i}"))
        o.append(f'<h2 id="{i}">Exercícios práticos em Python (scikit-learn)</h2><p>Resolva os exercícios no seu computador (ou no Google Colab). Eles usam conjuntos de dados da UCI que já vêm no scikit-learn ou dados simulados. Tente resolver <strong>antes</strong> de consultar a solução, no Apêndice D.</p>')
        for it in exs:
            o.append(f'<h3>Exercício {it["num"]} — {inline(it["title"])}</h3><p class="nivel">NÍVEL: {E(it["level"].upper())}</p>')
            o.append(blocks_html(parse_md("\n".join(it["stmt"]))[1]))
    if alem:
        o.append(callout_html("saibamais", "Para ir além: bases públicas da UCI e do Kaggle", alem + ["", "Estes conjuntos não foram usados nem testados nos códigos do livro. Confira, nos sites, a descrição, a licença e as colunas antes de usá-los."]))
    if proj:
        t = "Projeto integrador final" if c["n"] == 13 else "Projeto integrador do capítulo"
        i = hid(); nav.append((2, t, f"{fn}#{i}"))
        o.append(f'<h2 id="{i}">{t}</h2><h3>{inline(proj["title"])}</h3>' + blocks_html(parse_md("\n".join(proj["stmt"]))[1]) + "<p><em>A solução de referência está no Apêndice E.</em></p>")
    return "\n".join(o)

shown = set()
for c in chapters:
    m = c["meta"]
    if m.get("part") and m["part"] not in shown:
        shown.add(m["part"]); pn, pt = m["part"].split(" — ")
        lst = []
        for x in chapters[c["n"] - 1:]:
            if x is not c and x["meta"].get("part"): break
            lst.append(x)
        fnp = f"parte{len(shown)}.xhtml"; nav.append((0, m["part"], fnp))
        add(fnp, m["part"], f'<div class="parte"><p class="pn">{E(pn.upper())}</p><h1>{E(m["part"])}</h1><p><em>{E(PART_DESC[m["part"]])}</em></p><p><strong>Nesta parte:</strong></p><ul>' +
            "".join(f'<li>Cap. {x["n"]} — {E(x["meta"]["title"])}</li>' for x in lst) + "</ul></div>")
    fn = f"cap{c['n']:02d}.xhtml"; used_imgs.add(f"banner{m.get('banner', c['n'])}")
    nav.append((1, f"Capítulo {c['n']} — {m['title']}", fn))
    body = [f'<img class="banner" src="images/banner{m.get("banner", c["n"])}.png" alt="Faixa do capítulo {c["n"]}"/>', f'<h1>Capítulo {c["n"]} — {E(m["title"])}</h1>',
            '<div class="objetivos"><p><strong>OBJETIVOS DE APRENDIZAGEM</strong></p><p><strong>Ao final deste capítulo você será capaz de:</strong></p><ul>' + "".join("<li>" + inline(x) + "</li>" for x in m["obj"]) + "</ul></div>",
            blocks_html(c["blocks"], fn), exercises_html(c), praticos_html(c)]
    add(fn, f"Capítulo {c['n']}", "\n".join(body))

# apêndice A: gabarito
fa = "apendiceA.xhtml"; nav.append((1, "Apêndice A — Gabarito dos exercícios", fa)); o = ["<h1>Apêndice A — Gabarito dos exercícios</h1>",
    "<p>Esta seção traz a resposta de todas as questões. Nas questões de múltipla escolha, aparecem a alternativa correta e uma breve justificativa. Nas discursivas, aparece uma <strong>resposta-modelo</strong>: a sua resposta pode ser redigida de outro modo, desde que contemple as mesmas ideias.</p>"]
for c in chapters:
    i = hid(); nav.append((2, f"Capítulo {c['n']} — {c['meta']['title'].split(':')[0]}", f"{fa}#{i}"))
    o.append(f'<h2 id="{i}">Capítulo {c["n"]} — {E(c["meta"]["title"].split(":")[0])}</h2><h3>Múltipla escolha</h3><table><thead><tr><th>Questão</th><th>Resposta</th><th>Justificativa</th></tr></thead><tbody>' +
             "".join(f"<tr><td>{k}</td><td>{q['ans']})</td><td>{inline(q['why'])}</td></tr>" for k, q in enumerate(c["mc"], 1)) + "</tbody></table><h3>Discursivas (respostas-modelo)</h3><ol>" +
             "".join("<li>" + inline(q["ans"]) + "</li>" for q in c["dc"]) + "</ol>")
add(fa, "Apêndice A", "\n".join(o))
# B: glossário
nav.append((1, "Apêndice B — Glossário", "apendiceB.xhtml")); import locale
def pt_key(s):
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()
gl = sorted([l for l in rd("glossario.txt").strip().split("\n") if l], key=pt_key)
add("apendiceB.xhtml", "Apêndice B", "<h1>Apêndice B — Glossário</h1><p>Termos técnicos usados no livro, explicados de forma simples.</p>" +
    "".join(f'<p><strong>{E(l[:l.index(":")])}</strong> — {inline(l[l.index(":") + 1:].strip())}</p>' for l in gl))
# C: guia
meta, blocks = parse_md(rd("guia.md")); nav.append((1, meta["title"], "apendiceC.xhtml"))
add("apendiceC.xhtml", "Apêndice C", f"<h1>{E(meta['title'])}</h1>" + blocks_html(blocks, "apendiceC.xhtml"))
# D: soluções práticas
fd = "apendiceD.xhtml"; nav.append((1, "Apêndice D — Soluções dos exercícios práticos", fd))
o = ["<h1>Apêndice D — Soluções dos exercícios práticos</h1>", "<p>As soluções abaixo são <strong>caminhos possíveis</strong>, e não os únicos. As saídas foram obtidas com as bibliotecas e as sementes indicadas nos códigos.</p>"]
for c in chapters:
    if not c["pr"]: continue
    exs = [x for x in c["pr"][0] if x["kind"] == "EX"]
    if not exs: continue
    i = hid(); nav.append((2, f"Capítulo {c['n']} — {c['meta']['title'].split(':')[0]}", f"{fd}#{i}")); o.append(f'<h2 id="{i}">Capítulo {c["n"]} — {E(c["meta"]["title"].split(":")[0])}</h2>')
    for it in exs:
        o.append(f'<h3>Exercício {it["num"]} — {inline(it["title"])}</h3>' + code_html("python", code_of(it)))
        sd = saida_of(c["n"], it)
        if sd: o.append(code_html("saída", sd))
        if it["note"]: o.append(callout_html("exemplo", "O que observar", it["note"]))
add(fd, "Apêndice D", "\n".join(o))
# E: projetos
fe = "apendiceE.xhtml"; nav.append((1, "Apêndice E — Soluções de referência dos projetos integradores", fe))
o = ["<h1>Apêndice E — Soluções de referência dos projetos integradores</h1>", "<p>Estas são <strong>soluções de referência</strong>: caminhos possíveis, e não os únicos. A avaliação de um projeto deve considerar também a qualidade da argumentação, e não apenas o código.</p>"]
for c in chapters:
    if not c["pr"]: continue
    proj = next((x for x in c["pr"][0] if x["kind"] == "PROJ"), None)
    if not proj: continue
    i = hid(); nav.append((2, f"Capítulo {c['n']} — {c['meta']['title'].split(':')[0]}", f"{fe}#{i}")); o.append(f'<h2 id="{i}">Capítulo {c["n"]} — {E(c["meta"]["title"].split(":")[0])}</h2><h3>{inline(proj["title"])}</h3>')
    if any(l.strip() for l in proj["rubrica"]): o.append("<p><strong>Critérios de avaliação</strong></p>" + blocks_html(parse_md("\n".join(proj["rubrica"]))[1]))
    o.append(code_html("python", code_of(proj)))
    sd = saida_of(c["n"], proj)
    if sd: o.append(code_html("saída", sd))
    if proj["note"]: o.append(callout_html("exemplo", "Discussão dos resultados", proj["note"]))
add(fe, "Apêndice E", "\n".join(o))
# referências
nav.append((1, "Referências", "referencias.xhtml"))
refs = sorted([l for l in rd("referencias.txt").strip().split("\n") if l], key=pt_key)
add("referencias.xhtml", "Referências", "<h1>Referências</h1><p><em>Lista de artigos de periódicos e conferências de referência, além de livros reconhecidos, citados ao longo do texto.</em></p>" + "".join(f'<p class="ref">{E(r)}</p>' for r in refs))

# sumário (nav) — também como página
def nav_list(entries):
    out, depth = [], 0
    for lv, t, href in entries:
        if lv == 3: continue
        out.append((lv, t, href))
    return out
entries = nav_list(nav)
toc_body = '<nav epub:type="toc" id="toc"><h1>Sumário</h1><ol style="list-style:none;padding-left:0">' + "".join(
    f'<li style="margin-left:{max(0, lv - 1) * 1.2}em;{"font-weight:bold;margin-top:.5em;" if lv <= 1 else ""}"><a href="{href}">{E(t)}</a></li>' for lv, t, href in entries) + "</ol></nav>"
files.insert(files_toc_index, ("nav.xhtml", "Sumário", xhtml("Sumário", toc_body)))

# ------------------------------------------------------------ empacotamento
uid = "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, "aprendizado-federado-araken-santos"))
opf = ['<?xml version="1.0" encoding="utf-8"?>', '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="pt-BR">',
       '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">', f'<dc:identifier id="bookid">{uid}</dc:identifier>', f"<dc:title>{E(TITLE)}: {E(SUBTITLE)}</dc:title>",
       f"<dc:creator>{E(AUTHOR)}</dc:creator>", "<dc:language>pt-BR</dc:language>", f"<dc:date>{YEAR}-01-01</dc:date>",
       "<dc:description>Livro didático sobre aprendizado federado, dos fundamentos às fronteiras, com exercícios e projetos em Python.</dc:description>",
       f'<meta property="dcterms:modified">{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}</meta>', '<meta name="cover" content="img-capa"/>', "</metadata>", "<manifest>",
       '<item id="css" href="style.css" media-type="text/css"/>']
imgs = sorted(used_imgs | {"capa"})
for im in imgs: opf.append(f'<item id="img-{im}" href="images/{im}.png" media-type="image/png"' + (' properties="cover-image"' if im == "capa" else "") + "/>")
opf.append('<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
for name, _, _ in files: opf.append(f'<item id="{name.split(".")[0]}" href="{name}" media-type="application/xhtml+xml"' + (' properties="nav"' if name == "nav.xhtml" else "") + "/>")
opf += ['</manifest>', '<spine toc="ncx">'] + [f'<itemref idref="{name.split(".")[0]}"/>' for name, _, _ in files] + ["</spine>", '<guide><reference type="cover" title="Capa" href="capa.xhtml"/><reference type="toc" title="Sumário" href="nav.xhtml"/></guide>', "</package>"]
ncx = ['<?xml version="1.0" encoding="utf-8"?>', '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">', f'<head><meta name="dtb:uid" content="{uid}"/></head>', f"<docTitle><text>{E(TITLE)}</text></docTitle>", "<navMap>"]
for k, (lv, t, href) in enumerate(entries, 1): ncx.append(f'<navPoint id="np{k}" playOrder="{k}"><navLabel><text>{E(t)}</text></navLabel><content src="{href}"/></navPoint>')
ncx += ["</navMap>", "</ncx>"]
with zipfile.ZipFile(OUT, "w") as z:
    z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip")
    z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>', zipfile.ZIP_DEFLATED)
    z.writestr("OEBPS/content.opf", "\n".join(opf), zipfile.ZIP_DEFLATED); z.writestr("OEBPS/toc.ncx", "\n".join(ncx), zipfile.ZIP_DEFLATED); z.writestr("OEBPS/style.css", CSS, zipfile.ZIP_DEFLATED)
    for name, _, content in files: z.writestr("OEBPS/" + name, content, zipfile.ZIP_DEFLATED)
    for im in imgs: z.write(os.path.join(FIGS, im + ".png"), f"OEBPS/images/{im}.png", zipfile.ZIP_DEFLATED)
print("EPUB gerado:", OUT, f"({os.path.getsize(OUT) / 1e6:.1f} MB,", len(files), "arquivos XHTML,", len(imgs), "imagens)")
