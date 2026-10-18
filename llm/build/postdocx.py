"""Transforma as entradas estáticas do sumário em um campo TOC real do Word (com páginas em cache já verificadas)."""
import sys, zipfile, shutil, re, os
p = sys.argv[1]
tmp = p + ".tmp"
with zipfile.ZipFile(p) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for it in zin.infolist():
        data = zin.read(it.filename)
        if it.filename == "word/document.xml":
            x = data.decode("utf8")
            begin = ('<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-4" \\h \\z \\u </w:instrText></w:r>'
                     '<w:r><w:fldChar w:fldCharType="separate"/></w:r>')
            end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            n1 = len(re.findall(r"<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>§TOC_BEGIN§</w:t></w:r>", x))
            x = re.sub(r"<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>§TOC_BEGIN§</w:t></w:r>", lambda m: begin, x)
            x = re.sub(r"<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>§TOC_END§</w:t></w:r>", lambda m: end, x)
            assert n1 == 1, "marcador do sumário não encontrado"
            data = x.encode("utf8")
        zout.writestr(it, data)
shutil.move(tmp, p)
print("campo TOC inserido")
