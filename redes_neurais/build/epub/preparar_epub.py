"""Gera a fonte do EPUB a partir do DOCX final: remove a folha de rosto e o
sumário (tudo antes do título "Apresentação"), pois o EPUB tem folha de rosto
e sumário navegável próprios.
Uso: python3 preparar_epub.py livro.docx saida.docx"""
import sys, re, zipfile
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == "word/document.xml":
        d = data.decode("utf8")
        ini = d.index("<w:body>") + len("<w:body>")
        paras = list(re.finditer(r"<w:p[ >].*?</w:p>", d[ini:], re.S))
        corte = None
        for m in paras:
            txt = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", m.group(0))).strip()
            if txt == "Apresentação" and "Heading" in m.group(0) or (txt == "Apresentação" and "w:pStyle" in m.group(0) and "TOC" not in m.group(0) and "fldChar" not in m.group(0)):
                corte = ini + m.start(); break
        assert corte is not None, "título Apresentação não encontrado"
        d = d[:ini] + d[corte:]
        data = d.encode("utf8")
    zout.writestr(it, data)
zout.close()
print("ok:", dst)
