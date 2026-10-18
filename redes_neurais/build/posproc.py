"""Pós-processamento do DOCX: substitui marcadores ⟦Mn⟧ por equações nativas
(OMML, convertidas do LaTeX via pandoc) e envolve o sumário em um campo TOC."""
import json, re, subprocess, sys, tempfile, zipfile, os, shutil

src, dst = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
eqs = json.load(open(os.path.join(here, "equations.json")))

# 1) LaTeX -> OMML (um parágrafo por equação)
md = []
for e in eqs:
    md.append(("$$%s$$" if e["display"] else "$%s$") % e["tex"].replace("\n", " "))
    md.append("")
tmp = tempfile.mkdtemp()
open(os.path.join(tmp, "eq.md"), "w").write("\n".join(md))
r = subprocess.run(["pandoc", os.path.join(tmp, "eq.md"), "-f", "markdown+tex_math_dollars", "-o", os.path.join(tmp, "eq.docx")], capture_output=True, text=True)
if r.returncode or r.stderr.strip():
    print("pandoc:", r.stderr[:2000])
z = zipfile.ZipFile(os.path.join(tmp, "eq.docx"))
x = z.read("word/document.xml").decode("utf8")
paras = re.findall(r"<w:p>.*?</w:p>|<w:p [^>]*>.*?</w:p>", x, re.S)
omaths = []
for p in paras:
    m = re.search(r"<m:oMath>.*?</m:oMath>", p, re.S)
    if m:
        omaths.append(m.group(0))
    elif "$" in re.sub(r"<[^>]+>", "", p):
        omaths.append(None)  # falhou: pandoc deixou o texto literal
        print("AVISO: pandoc não converteu:", re.sub(r"<[^>]+>", "", p)[:120])
if len(omaths) != len(eqs):
    print("ERRO: %d equações, %d OMML" % (len(eqs), len(omaths)))
    sys.exit(1)

# 2) injeta no documento
zin = zipfile.ZipFile(src)
doc = zin.read("word/document.xml").decode("utf8")
if 'xmlns:m=' not in doc[:3000]:
    doc = doc.replace("<w:document ", '<w:document xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" ', 1)
falhas = 0
def sub_marker(m):
    global falhas
    i = int(m.group(2))
    om = omaths[i]
    if om is None:
        falhas += 1
        return m.group(0)
    if eqs[i]["display"]:
        return "<m:oMathPara>%s</m:oMathPara>" % om
    return om
doc = re.sub(r"<w:r>(<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>\u27e6M(\d+)\u27e7</w:t></w:r>", sub_marker, doc, flags=re.S)
resto = re.findall(r"⟦M\d+⟧", doc)
if resto:
    print("AVISO: marcadores restantes:", len(resto), resto[:5])

# 3) campo TOC
def fld(kind, instr=None):
    if kind == "begin":
        return ('<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-4" \\h \\z \\u </w:instrText></w:r>'
                '<w:r><w:fldChar w:fldCharType="separate"/></w:r>')
    return '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
doc = re.sub(r"<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>⟦TBEGIN⟧</w:t></w:r>", lambda m: fld("begin"), doc, flags=re.S)
doc = re.sub(r"<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>⟦TEND⟧</w:t></w:r>", lambda m: fld("end"), doc, flags=re.S)

zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        data = doc.encode("utf8")
    zout.writestr(item, data)
zout.close()
shutil.rmtree(tmp)
print("posproc ok:", len(eqs), "equações; falhas:", falhas)
