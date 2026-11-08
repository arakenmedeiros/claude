#!/usr/bin/env python3
"""Gera o EPUB 3 do livro a partir do DOCX (variante EPUB=1, sem capa e sem sumário de páginas).

Uso: python3 epub.py entrada.docx saida.epub
Lê word/document.xml diretamente (estrutura simples gerada por build.js): títulos 1–4,
parágrafos, listas, tabelas (callouts e atividades) e figuras com texto alternativo.
"""
import sys, zipfile, re, uuid, os, html, datetime
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": W, "r": R, "a": A, "wp": WP}
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

TITLE = "Inteligência Artificial Aplicada à Educação"
SUBTITLE = "da educação infantil à pós-graduação"
SUB2 = "Diretrizes, práticas, oportunidades, benefícios e riscos"
AUTHOR = "Araken de Medeiros Santos"
LANG = "pt-BR"


def q(tag):
    p, t = tag.split(":")
    return "{%s}%s" % (NS[p], t)


def esc(s):
    return html.escape(s, quote=False)


def main(src, out):
    z = zipfile.ZipFile(src)
    doc = etree.fromstring(z.read("word/document.xml"))
    rels = etree.fromstring(z.read("word/_rels/document.xml.rels"))
    relmap = {r.get("Id"): r.get("Target") for r in rels}
    num = etree.fromstring(z.read("word/numbering.xml"))
    absfmt = {}
    for an in num.findall("w:abstractNum", NS):
        lv = an.find("w:lvl", NS)
        absfmt[an.get(q("w:abstractNumId"))] = lv.find("w:numFmt", NS).get(q("w:val")) if lv is not None else "bullet"
    numfmt = {}
    for n in num.findall("w:num", NS):
        numfmt[n.get(q("w:numId"))] = absfmt.get(n.find("w:abstractNumId", NS).get(q("w:val")), "bullet")

    media = {}  # nome no zip -> nome no epub

    def media_name(target):
        key = "word/" + target
        if key not in media:
            media[key] = "img%03d%s" % (len(media) + 1, os.path.splitext(target)[1])
        return media[key]

    def runs_html(p):
        out_ = []
        for r in p.iter(q("w:r")):
            rpr = r.find("w:rPr", NS)
            b = i = False; color = None; sz = None
            if rpr is not None:
                b = rpr.find("w:b", NS) is not None
                i = rpr.find("w:i", NS) is not None
                c = rpr.find("w:color", NS)
                if c is not None: color = c.get(q("w:val"))
                s = rpr.find("w:sz", NS)
                if s is not None: sz = int(s.get(q("w:val")))
            parts = []
            for ch in r:
                if ch.tag == q("w:t"): parts.append(esc(ch.text or ""))
                elif ch.tag == q("w:tab"): parts.append(" ")
                elif ch.tag == q("w:br") and ch.get(q("w:type")) != "page": parts.append("<br/>")
                elif ch.tag == q("w:drawing"):
                    blip = ch.find(".//a:blip", NS)
                    if blip is not None:
                        tgt = relmap[blip.get(q("r:embed"))]
                        dp = ch.find(".//wp:docPr", NS)
                        alt = (dp.get("descr") if dp is not None else "") or ""
                        parts.append('<img src="../images/%s" alt="%s"/>' % (media_name(tgt), html.escape(alt, quote=True)))
            if not parts: continue
            t = "".join(parts)
            if "<img" in t: out_.append(t); continue
            st = []
            if color and color not in ("auto", "000000"): st.append("color:#%s" % color)
            if sz and abs(sz - 21) / 21 > 0.15: st.append("font-size:%.2fem" % (sz / 21))
            if st: t = '<span style="%s">%s</span>' % (";".join(st), t)
            if i: t = "<em>%s</em>" % t
            if b: t = "<strong>%s</strong>" % t
            out_.append(t)
        return "".join(out_)

    def para_text(p):
        return "".join(t.text or "" for t in p.iter(q("w:t")))

    headings = []  # (nivel, texto, id, arquivo)
    files = []     # [{name, title, parts:[html]}]
    cur = {"f": None}
    hid = [0]

    def new_file(title):
        f = {"name": "c%03d.xhtml" % (len(files) + 1), "title": title, "body": []}
        files.append(f); cur["f"] = f
        return f

    def emit(h):
        if cur["f"] is None: new_file("Início")
        cur["f"]["body"].append(h)

    def render_block(el, ctx):
        """Devolve HTML de um w:p (já tratado como não-lista/título) ou w:tbl."""
        if el.tag == q("w:p"):
            ppr = el.find("w:pPr", NS)
            jc = None
            if ppr is not None:
                j = ppr.find("w:jc", NS)
                if j is not None: jc = j.get(q("w:val"))
            inner = runs_html(el)
            if not inner.strip(): return ""
            cls = {"both": "j", "center": "c", "right": "r"}.get(jc, "")
            return '<p%s>%s</p>' % (' class="%s"' % cls if cls else "", inner)
        if el.tag == q("w:tbl"):
            rows = []
            for tr in el.findall("w:tr", NS):
                cells = []
                for tc in tr.findall("w:tc", NS):
                    tcpr = tc.find("w:tcPr", NS)
                    span = 1; fill = None
                    if tcpr is not None:
                        gs = tcpr.find("w:gridSpan", NS)
                        if gs is not None: span = int(gs.get(q("w:val")))
                        sh = tcpr.find("w:shd", NS)
                        if sh is not None and sh.get(q("w:fill")) not in (None, "auto"): fill = sh.get(q("w:fill"))
                    inner = render_container(tc)
                    attrs = ""
                    if span > 1: attrs += ' colspan="%d"' % span
                    if fill: attrs += ' style="background-color:#%s"' % fill
                    cells.append("<td%s>%s</td>" % (attrs, inner or "&#160;"))
                rows.append("<tr>%s</tr>" % "".join(cells))
            return '<table>%s</table>' % "".join(rows)
        return ""

    def render_container(parent):
        """Parágrafos/listas de uma célula."""
        out_ = []
        lst = None
        for el in parent:
            if el.tag == q("w:p"):
                ppr = el.find("w:pPr", NS)
                np_ = ppr.find("w:numPr", NS) if ppr is not None else None
                if np_ is not None:
                    nid = np_.find("w:numId", NS).get(q("w:val"))
                    kind = "ol" if numfmt.get(nid) == "decimal" else "ul"
                    if lst and lst[0] == nid: lst[2].append(runs_html(el))
                    else:
                        if lst: out_.append(close_list(lst))
                        lst = (nid, kind, [runs_html(el)])
                    continue
                if lst: out_.append(close_list(lst)); lst = None
                out_.append(render_block(el, "cell"))
            elif el.tag == q("w:tbl"):
                if lst: out_.append(close_list(lst)); lst = None
                out_.append(render_block(el, "cell"))
        if lst: out_.append(close_list(lst))
        return "".join(out_)

    def close_list(l):
        return "<%s>%s</%s>" % (l[1], "".join("<li>%s</li>" % x for x in l[2]), l[1])

    body = doc.find("w:body", NS)
    lst = None
    for el in body:
        if el.tag == q("w:p"):
            ppr = el.find("w:pPr", NS)
            sty = None
            if ppr is not None:
                ps = ppr.find("w:pStyle", NS)
                if ps is not None: sty = ps.get(q("w:val"))
            m = re.match(r"Heading(\d)$", sty or "")
            if m:
                if lst: emit(close_list(lst)); lst = None
                lvl = int(m.group(1)); txt = para_text(el).strip()
                hid[0] += 1; i_ = "h%04d" % hid[0]
                if lvl <= 2 or not files:
                    # novo arquivo em cada título 1 ou 2
                    new_file(txt)
                headings.append((lvl, txt, i_, cur["f"]["name"]))
                emit('<h%d id="%s">%s</h%d>' % (lvl, i_, esc(txt), lvl))
                continue
            np_ = ppr.find("w:numPr", NS) if ppr is not None else None
            if np_ is not None:
                nid = np_.find("w:numId", NS).get(q("w:val"))
                kind = "ol" if numfmt.get(nid) == "decimal" else "ul"
                if lst and lst[0] == nid: lst[2].append(runs_html(el))
                else:
                    if lst: emit(close_list(lst))
                    lst = (nid, kind, [runs_html(el)])
                continue
            if lst: emit(close_list(lst)); lst = None
            h = render_block(el, "body")
            if h: emit(h)
        elif el.tag == q("w:tbl"):
            if lst: emit(close_list(lst)); lst = None
            emit(render_block(el, "body"))
    if lst: emit(close_list(lst))

    # ------------------------------------------------------------ arquivos do pacote
    book_id = "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, "ia-educacao-araken-2026"))
    today = datetime.date.today().isoformat()
    css = open(os.path.join(HERE, "epub.css"), encoding="utf8").read()

    def xhtml(title, inner):
        return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
                '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="%s" xml:lang="%s">\n'
                '<head><meta charset="utf-8"/><title>%s</title><link rel="stylesheet" type="text/css" href="../css/livro.css"/></head>\n'
                '<body>\n%s\n</body></html>\n') % (LANG, LANG, esc(title), inner)

    # capa e folha de rosto
    cover_x = xhtml("Capa", '<section epub:type="cover" class="capa"><img src="../images/capa.png" alt="Capa do livro: %s, %s, de %s. Ilustração de crianças e jovens de diferentes idades sobre degraus que representam as etapas de ensino, da educação infantil à pós-graduação."/></section>' % (TITLE, SUBTITLE, AUTHOR))
    title_x = xhtml("Folha de rosto", '<section epub:type="titlepage" class="rosto"><h1 class="rosto-t">%s</h1><p class="rosto-s">%s</p><p class="rosto-s2">%s</p><p class="rosto-a">%s</p><p class="rosto-y">2026</p></section>' % (esc(TITLE.upper()), esc(SUBTITLE), esc(SUB2), esc(AUTHOR)))

    # navegação
    # construção recursiva da árvore de navegação
    def build_nav(items, idx=0, lvl=1):
        out_ = ["<ol>"]
        while idx < len(items) and items[idx][0] >= lvl:
            h = items[idx]
            out_.append('<li><a href="text/%s#%s">%s</a>' % (h[3], h[2], esc(h[1])))
            idx += 1
            if idx < len(items) and items[idx][0] > lvl:
                sub, idx = build_nav(items, idx, items[idx][0])
                out_.append(sub)
            out_.append("</li>")
        out_.append("</ol>")
        return "".join(out_), idx

    nav_items = [h for h in headings if h[0] <= 3]
    nav_ol, _ = build_nav(nav_items)
    nav_x = ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
             '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="%s" xml:lang="%s">\n'
             '<head><meta charset="utf-8"/><title>Sumário</title><link rel="stylesheet" type="text/css" href="css/livro.css"/></head>\n<body>\n'
             '<nav epub:type="toc" id="toc"><h1>Sumário</h1>%s</nav>\n'
             '<nav epub:type="landmarks" hidden="hidden"><ol>'
             '<li><a epub:type="cover" href="text/capa.xhtml">Capa</a></li>'
             '<li><a epub:type="titlepage" href="text/rosto.xhtml">Folha de rosto</a></li>'
                          '<li><a epub:type="bodymatter" href="text/%s">Início do texto</a></li></ol></nav>\n</body></html>\n') % (LANG, LANG, nav_ol, files[0]["name"])

    # NCX (compatibilidade EPUB 2)
    play = [0]

    def ncx_tree(items, idx=0, lvl=1):
        out_ = []
        while idx < len(items) and items[idx][0] >= lvl:
            h = items[idx]
            play[0] += 1
            out_.append('<navPoint id="n%d" playOrder="%d"><navLabel><text>%s</text></navLabel><content src="text/%s#%s"/>' % (play[0], play[0], esc(h[1]), h[3], h[2]))
            idx += 1
            if idx < len(items) and items[idx][0] > lvl:
                sub, idx = ncx_tree(items, idx, items[idx][0])
                out_.append(sub)
            out_.append("</navPoint>")
        return "".join(out_), idx
    ncx_body, _ = ncx_tree([h for h in headings if h[0] <= 2])
    ncx_x = ('<?xml version="1.0" encoding="utf-8"?>\n<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="%s"><head>'
             '<meta name="dtb:uid" content="%s"/><meta name="dtb:depth" content="2"/><meta name="dtb:totalPageCount" content="0"/><meta name="dtb:maxPageNumber" content="0"/></head>'
             '<docTitle><text>%s</text></docTitle><navMap>%s</navMap></ncx>\n') % (LANG, book_id, esc(TITLE), ncx_body)

    # manifesto
    man = ['<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
           '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
           '<item id="css" href="css/livro.css" media-type="text/css"/>',
           '<item id="capa-img" href="images/capa.png" media-type="image/png" properties="cover-image"/>',
           '<item id="capa" href="text/capa.xhtml" media-type="application/xhtml+xml"/>',
           '<item id="rosto" href="text/rosto.xhtml" media-type="application/xhtml+xml"/>']
    spine = ['<itemref idref="capa" linear="yes"/>', '<itemref idref="rosto"/>']
    for k, f in enumerate(files):
        man.append('<item id="f%d" href="text/%s" media-type="application/xhtml+xml"/>' % (k, f["name"]))
        spine.append('<itemref idref="f%d"/>' % k)
    for zk, nm in media.items():
        ext = os.path.splitext(nm)[1].lower()
        mt = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "svg": "image/svg+xml"}[ext[1:]]
        man.append('<item id="m_%s" href="images/%s" media-type="%s"/>' % (nm.split(".")[0], nm, mt))
    opf = ('<?xml version="1.0" encoding="utf-8"?>\n'
           '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="%s">\n'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
           '<dc:identifier id="bookid">%s</dc:identifier>\n<dc:title>%s: %s</dc:title>\n<dc:creator>%s</dc:creator>\n<dc:language>%s</dc:language>\n'
           '<dc:subject>Inteligência artificial</dc:subject><dc:subject>Educação</dc:subject><dc:subject>Formação de professores</dc:subject>\n'
           '<dc:description>%s. Livro didático para professores, da educação infantil à pós-graduação.</dc:description>\n'
           '<meta property="dcterms:modified">%sT00:00:00Z</meta>\n<meta name="cover" content="capa-img"/>\n'
           '<meta property="schema:accessMode">textual</meta><meta property="schema:accessMode">visual</meta>\n'
           '<meta property="schema:accessModeSufficient">textual</meta>\n'
           '<meta property="schema:accessibilityFeature">alternativeText</meta><meta property="schema:accessibilityFeature">tableOfContents</meta><meta property="schema:accessibilityFeature">readingOrder</meta>\n'
           '<meta property="schema:accessibilityHazard">none</meta>\n'
           '<meta property="schema:accessibilitySummary">Figuras com texto alternativo e descrição acessível em texto; navegação por sumário com partes, capítulos e seções.</meta>\n'
           '</metadata>\n<manifest>\n%s\n</manifest>\n<spine toc="ncx">\n%s\n</spine>\n'
           '<guide><reference type="cover" title="Capa" href="text/capa.xhtml"/><reference type="toc" title="Sumário" href="nav.xhtml"/></guide>\n</package>\n') % (
        LANG, book_id, esc(TITLE), esc(SUBTITLE), esc(AUTHOR), LANG, esc(SUB2), today, "\n".join(man), "\n".join(spine))

    zo = zipfile.ZipFile(out, "w")
    zo.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
    D = zipfile.ZIP_DEFLATED
    zo.writestr("META-INF/container.xml", '<?xml version="1.0" encoding="UTF-8"?>\n<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="EPUB/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>\n', compress_type=D)
    zo.writestr("EPUB/content.opf", opf, compress_type=D)
    zo.writestr("EPUB/nav.xhtml", nav_x, compress_type=D)
    zo.writestr("EPUB/toc.ncx", ncx_x, compress_type=D)
    zo.writestr("EPUB/css/livro.css", css, compress_type=D)
    zo.writestr("EPUB/text/capa.xhtml", cover_x, compress_type=D)
    zo.writestr("EPUB/text/rosto.xhtml", title_x, compress_type=D)
    for f in files:
        zo.writestr("EPUB/text/" + f["name"], xhtml(f["title"], "\n".join(f["body"])), compress_type=D)
    for zk, nm in media.items():
        zo.writestr("EPUB/images/" + nm, z.read(zk), compress_type=zipfile.ZIP_STORED)
    zo.write(os.path.join(ROOT, "figs", "capa.png"), "EPUB/images/capa.png", compress_type=zipfile.ZIP_STORED)
    zo.close()
    print("ok", out, "|", len(files), "arquivos de texto,", len(media), "imagens,", len(headings), "títulos")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
