import sys, zipfile, re, shutil, os
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(dst + ".tmp", "w", zipfile.ZIP_DEFLATED)
beg = '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-4" \\h \\z \\u </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
n = 0
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename == "word/document.xml":
        s = data.decode("utf8")
        s, a = re.subn(r'<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>@@TOCBEGIN@@</w:t></w:r>', lambda m: beg, s)
        s, b = re.subn(r'<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>@@TOCEND@@</w:t></w:r>', lambda m: end, s)
        n = (a, b)
        data = s.encode("utf8")
    zout.writestr(it, data)
zout.close(); os.replace(dst + ".tmp", dst)
print("campo TOC inserido:", n)
