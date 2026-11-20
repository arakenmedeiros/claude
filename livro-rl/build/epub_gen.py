#!/usr/bin/env python3
"""Gera o EPUB a partir de build/book.json (escrito por build.js). Uso: python3 build/epub_gen.py [saida.epub]"""
import json, re, html, subprocess, sys, os, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "livro.epub")
B = json.load(open(os.path.join(ROOT, "build", "book.json"), encoding="utf-8"))
E = html.escape

SUBRE = re.compile(r'(?<![A-Za-z0-9_])([A-ZΑ-Ωγπαβδεθλμστφ])_(\([^)]*\)|[a-z0-9]{1,3})(?![A-Za-z_])')
def subs(t):
    return SUBRE.sub(lambda m: f'{m.group(1)}<sub>{m.group(2).strip("()")}</sub>', t)
TOK = re.compile(r'(\*\*[^*]+\*\*(?:_\([^)]*\)|_[a-z0-9]{1,3})?|\*[^*\s][^*]*\*|`[^`]+`)')
def inl(t):
    out = []; pos = 0
    for m in TOK.finditer(t):
        out.append(subs(E(t[pos:m.start()], quote=False)))
        s = m.group(0)
        if s.startswith("**"):
            mm = re.match(r'^\*\*([^*]+)\*\*(?:_(.+))?$', s)
            sub = f'<sub>{E(mm.group(2).strip("()"))}</sub>' if mm.group(2) else ""
            out.append(f'<strong>{subs(E(mm.group(1), quote=False))}</strong>{sub}')
        elif s.startswith("`"): out.append(f'<code>{E(s[1:-1], quote=False)}</code>')
        else: out.append(f'<em>{subs(E(s[1:-1], quote=False))}</em>')
        pos = m.end()
    out.append(subs(E(t[pos:], quote=False)))
    return "".join(out).replace("", "*")

def fml(t):
    t = E(t, quote=False)
    t = re.sub(r'_\(([^)]*)\)', r'<sub>\1</sub>', t)
    t = re.sub(r'_([A-Za-z0-9ÀÿΑ-ωα-ω]+)', r'<sub>\1</sub>', t)
    t = re.sub(r'\^\(([^)]*)\)', r'<sup>\1</sup>', t)
    t = re.sub(r'\^([A-Za-z0-9]+)', r'<sup>\1</sup>', t)
    return t.replace("", "*")

LABEL = {"dica": "DICA", "atencao": "ATENÇÃO", "cotidiano": "NO SEU DIA A DIA", "saibamais": "SAIBA MAIS", "exemplo": "EXEMPLO", "memorize": "MEMORIZE"}
def norm(t):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn").lower()

def callout_body(lines):
    h = []; para = []; i = 0
    def flush():
        nonlocal para
        if para: h.append(f'<p>{inl(" ".join(para))}</p>'); para = []
    while i < len(lines):
        ln = lines[i]
        if re.match(r'^- ', ln):
            flush(); items = []
            while i < len(lines) and re.match(r'^- ', lines[i]): items.append(lines[i][2:]); i += 1
            h.append("<ul>" + "".join(f"<li>{inl(x)}</li>" for x in items) + "</ul>"); continue
        if re.match(r'^\d+\.\s', ln):
            flush(); items = []
            while i < len(lines) and re.match(r'^\d+\.\s', lines[i]): items.append(re.sub(r'^\d+\.\s+', '', lines[i])); i += 1
            h.append("<ol>" + "".join(f"<li>{inl(x)}</li>" for x in items) + "</ol>"); continue
        if ln.startswith("$$"):
            flush(); h.append(f'<p class="formula">{fml(re.sub(r"^\$\$\s*|\s*\$\$$", "", ln))}</p>'); i += 1; continue
        if not ln.strip(): flush(); i += 1; continue
        para.append(ln.strip()); i += 1
    flush(); return "".join(h)

def blocks_html(blocks):
    h = []
    for b in blocks:
        t = b["t"]
        if t == "h3": h.append(f'<h3>{inl(b["text"])}</h3>')
        elif t == "h4": h.append(f'<h4>{inl(b["text"])}</h4>')
        elif t == "p": h.append(f'<p>{inl(b["text"])}</p>')
        elif t == "quote": h.append(f'<blockquote><p>{inl(b["text"])}</p></blockquote>')
        elif t == "ul": h.append("<ul>" + "".join(f"<li>{inl(x)}</li>" for x in b["items"]) + "</ul>")
        elif t == "ol": h.append("<ol>" + "".join(f"<li>{inl(x)}</li>" for x in b["items"]) + "</ol>")
        elif t == "formula": h.append(f'<p class="formula">{fml(b["text"])}</p>')
        elif t == "code":
            lang = (b.get("lang") or "").replace("-nr", "")
            h.append(f'<pre class="codigo"><code>{E(chr(10).join(b["code"]), quote=False)}</code></pre>')
        elif t == "fig":
            fn = os.path.join(ROOT, "figs", b["id"] + ".png")
            assert os.path.exists(fn), b["id"]
            h.append(f'<figure><img src="figs/{b["id"]}.png" alt="{E(re.sub(r"[*`]", "", b["caption"]))}"/><figcaption>{inl(b["caption"])}</figcaption></figure>')
        elif t == "table":
            rows = b["rows"]
            h.append("<table><thead><tr>" + "".join(f"<th>{inl(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>" +
                     "".join("<tr>" + "".join(f"<td>{inl(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</tbody></table>")
        elif t == "callout":
            lab = LABEL.get(b["kind"], "DICA"); title = b["title"]
            dup = norm(title).startswith(norm(lab.split(" ")[0])) or (b["kind"] == "cotidiano" and norm(title).startswith("no seu dia")) or (b["kind"] == "exemplo" and norm(title).startswith("exemplo"))
            head = (f'<span class="rotulo">{lab}</span> ' if not dup else "") + inl(title)
            h.append(f'<div class="callout {b["kind"]}"><p class="callout-titulo"><strong>{head}</strong></p>{callout_body(b["body"])}</div>')
    return "\n".join(h)

def exercises_html(ex):
    h = ['<h3>Exercícios de múltipla escolha</h3><p class="nota">Escolha <strong>uma</strong> alternativa em cada questão. O gabarito está no Apêndice A.</p>']
    for i, q in enumerate(ex["mc"]):
        h.append(f'<div class="questao"><p><strong>Questão {i+1}.</strong> {inl(q["stem"])}</p><ol type="a">' + "".join(f"<li>{inl(o)}</li>" for o in q["opts"]) + "</ol></div>")
    h.append('<h3>Exercícios discursivos</h3><p class="nota">Responda com suas palavras, em poucas linhas. Compare depois com a resposta-modelo do Apêndice A.</p><ol>')
    h += [f"<li>{inl(q['stem'])}</li>" for q in ex["dc"]]; h.append("</ol>")
    return "\n".join(h)

parts = []
def PART(name, desc, items):
    pnum = name.split(" — ")[0]
    return f'<h1 class="parte">{E(name)}</h1><p class="parte-desc">{E(desc)}</p><p><strong>Nesta parte:</strong></p><ul>' + "".join(f"<li>{E(x)}</li>" for x in items) + "</ul>"

PART_DESC = {
  "Parte I — Fundamentos": "As ideias que sustentam tudo o que vem depois: o que é aprender por tentativa e erro, como equilibrar explorar e aproveitar, como modelar decisões com Processos de Decisão de Markov.",
  "Parte II — Métodos Tabulares": "Como aprender a decidir apenas com a experiência, sem conhecer as regras do ambiente: Monte Carlo, diferença temporal, SARSA e Q-learning.",
  "Parte III — Aproximação de Funções e Aprendizado por Reforço Profundo": "Como lidar com mundos enormes: aproximação de funções, redes neurais, DQN, gradientes de política, atores-críticos e algoritmos modernos.",
  "Parte IV — Tópicos Avançados": "Planejamento com modelos do mundo, aprendizado offline e por imitação, sistemas multiagente, aprendizado a partir de preferências humanas (RLHF) e as fronteiras da área.",
  "Parte V — Projeto Integrador": "Hora de juntar tudo: um projeto completo, do problema ao agente avaliado, que mobiliza os conceitos de todos os capítulos.",
}
body = []
body.append("<h1>Apresentação</h1>" + blocks_html(B["apresentacao"]["blocks"]))
chs = B["chapters"]
for c in chs:
    if c["meta"].get("part"):
        items = []
        for k in range(c["n"] - 1, len(chs)):
            if k > c["n"] - 1 and chs[k]["meta"].get("part"): break
            items.append(f'Cap. {chs[k]["n"]} — {chs[k]["meta"]["title"]}')
        body.append(PART(c["meta"]["part"], PART_DESC.get(c["meta"]["part"], ""), items))
    bn = c["meta"].get("banner") or c["n"]
    obj = "".join(f"<li>{inl(o)}</li>" for o in c["meta"]["obj"])
    body.append(f'<h2>Capítulo {c["n"]} — {inl(c["meta"]["title"])}</h2><p class="banner"><img src="figs/banner{bn}.png" alt="Faixa do capítulo {c["n"]}"/></p>'
                f'<div class="callout objetivos"><p class="callout-titulo"><strong>Neste capítulo você vai aprender a:</strong></p><ul>{obj}</ul></div>'
                + blocks_html(c["blocks"]) + exercises_html(c["ex"]))
ig = B["integrador"]
body.append(PART("Parte V — Projeto Integrador", PART_DESC["Parte V — Projeto Integrador"], ["Projeto Integrador — " + ig["meta"]["title"]]))
body.append(f'<h2>Projeto Integrador — {inl(ig["meta"]["title"])}</h2><p class="banner"><img src="figs/banner14.png" alt="Faixa"/></p>' + blocks_html(ig["blocks"]))
# Apêndice A
g = ['<h1>Apêndice A — Gabarito dos exercícios</h1><p>Nas questões de múltipla escolha, aparecem a alternativa correta e uma breve justificativa. Nas discursivas, aparece uma <strong>resposta-modelo</strong>.</p>']
for c in chs:
    g.append(f'<h2>Capítulo {c["n"]} — {inl(c["meta"]["title"].split(":")[0])}</h2><h3>Múltipla escolha</h3><table><thead><tr><th>Questão</th><th>Resposta</th><th>Justificativa</th></tr></thead><tbody>'
             + "".join(f'<tr><td>{i+1}</td><td>{q["ans"]})</td><td>{inl(q["why"])}</td></tr>' for i, q in enumerate(c["ex"]["mc"])) + '</tbody></table><h3>Discursivas (respostas-modelo)</h3><ol>'
             + "".join(f'<li>{inl(q["ans"])}</li>' for q in c["ex"]["dc"]) + "</ol>")
body.append("\n".join(g))
gl = ['<h1>Apêndice B — Glossário</h1><p>Termos técnicos usados no livro, explicados de forma simples.</p>']
for ln in B["glossario"]:
    k = ln.index(":"); gl.append(f'<p class="verbete"><strong>{inl(ln[:k])}</strong>:{inl(ln[k+1:])}</p>')
body.append("\n".join(gl))
for a in B["apx"]:
    body.append(f'<h1>{inl(a["meta"]["title"])}</h1>' + blocks_html(a["blocks"]))
body.append('<h1>Referências</h1><p class="nota">Artigos de periódicos e conferências de referência, além de livros reconhecidos, citados ao longo do texto (norma ABNT).</p>' + "".join(f'<p class="ref">{inl(r)}</p>' for r in B["refs"]))

work = os.path.join(ROOT, "build", "epub_tmp"); shutil.rmtree(work, ignore_errors=True); os.makedirs(work); os.symlink(os.path.join(ROOT, "figs"), os.path.join(work, "figs"))
open(os.path.join(work, "livro.html"), "w", encoding="utf-8").write(
    '<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"/><title>%s</title></head><body>%s</body></html>' % (E(B["title"]), "\n".join(body)))
css = """
body { font-family: serif; line-height: 1.45; }
p, li, td, th, blockquote { text-align: justify; hyphens: auto; -webkit-hyphens: auto; }
h1, h2, h3, h4 { text-align: left; hyphens: none; page-break-after: avoid; }
h1.parte { text-align: left; margin-top: 3em; }
.parte-desc { font-style: italic; }
pre.codigo { background: #f3f4f6; border: 1px solid #d1d5db; padding: .6em; font-size: .78em; white-space: pre-wrap; word-wrap: break-word; text-align: left; hyphens: none; }
code { font-family: monospace; font-size: .92em; }
p.formula { text-align: center; background: #eef2ff; padding: .5em; font-family: "Cambria Math", serif; margin: .8em 0; hyphens: none; }
figure { margin: 1em 0; text-align: center; } figure img { max-width: 100%; height: auto; }
figcaption { font-size: .85em; font-style: italic; text-align: center; color: #555; }
p.banner img { max-width: 100%; height: auto; }
div.callout { border: 1px solid #c7d2fe; background: #f5f6ff; padding: .5em .9em; margin: 1em 0; border-radius: .4em; }
div.callout.dica { background: #ecfdf5; border-color: #6ee7b7; } div.callout.atencao { background: #fffbeb; border-color: #fcd34d; }
div.callout.cotidiano { background: #ecfeff; border-color: #67e8f9; } div.callout.saibamais { background: #f5f3ff; border-color: #c4b5fd; }
div.callout.memorize { background: #fdf2f8; border-color: #f9a8d4; } div.callout.objetivos { background: #eef2ff; }
.callout-titulo { font-weight: bold; text-align: left; } .rotulo { font-size: .8em; letter-spacing: .05em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .92em; } th, td { border: 1px solid #9ca3af; padding: .25em .4em; vertical-align: top; }
th { background: #1e1b4b; color: #fff; }
.questao { margin-bottom: .8em; } .nota { color: #555; } p.ref { margin-left: 1.5em; text-indent: -1.5em; font-size: .9em; } p.verbete { margin: .4em 0; }
"""
open(os.path.join(work, "estilo.css"), "w").write(css)
meta = f"""<dc:date>{B['year']}</dc:date>"""
open(os.path.join(work, "meta.xml"), "w").write(meta)
cmd = ["pandoc", "-f", "html", "-t", "epub3", "-o", OUT, "livro.html", "--toc", "--toc-depth=3", "--split-level=2",
       "--css", os.path.join(work, "estilo.css"), "--epub-metadata", os.path.join(work, "meta.xml"),
       "--epub-cover-image", os.path.join(ROOT, "figs", "capa.png"), "--metadata", f"title={B['title']} — {B['subtitle']}", "--metadata", f"author={B['author']}",
       "--metadata", "lang=pt-BR", "--resource-path", ROOT]
r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
# a pasta figs precisa estar acessível a partir do diretório de trabalho
print(r.stdout, r.stderr[:2000]); print("epub:", OUT, os.path.getsize(OUT) if os.path.exists(OUT) else "FALHOU")
