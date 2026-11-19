set -e
# Gera DOCX, PDF e EPUB (texto justificado). Requer: node (docx, jszip), LibreOffice e as dependências Python.
cd /home/user/claude/livro/build
export NODE_PATH=/home/user/claude/livro/slides/node_modules
SOFF=$(ls /root/.claude/skills/synced/*/docx/scripts/office/soffice.py | head -1)
LIVRO=/home/user/claude/livro
rm -f pages.json
# passada 1: sumário provisório -> descobre as páginas dos títulos
node build.js /tmp/pass1.docx >/dev/null
rm -rf /tmp/pdf1; python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf1 /tmp/pass1.docx >/dev/null 2>&1
python3 pages.py /tmp/pdf1/pass1.pdf
# passada 2: sumário com as páginas; repete até estabilizar (no máximo 3 vezes)
for i in 1 2 3; do
  EPUB_OUT=$LIVRO/Aprendizado_Nao_Supervisionado.epub node build.js $LIVRO/Aprendizado_Nao_Supervisionado.docx >/dev/null
  rm -rf /tmp/pdf2; python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf2 $LIVRO/Aprendizado_Nao_Supervisionado.docx >/dev/null 2>&1
  cp pages.json pages_pass1.json
  python3 pages.py /tmp/pdf2/Aprendizado_Nao_Supervisionado.pdf
  if cmp -s pages.json pages_pass1.json; then echo "PAGINAS ESTAVEIS"; break; fi
done
cp /tmp/pdf2/Aprendizado_Nao_Supervisionado.pdf $LIVRO/Aprendizado_Nao_Supervisionado.pdf
pdfinfo $LIVRO/Aprendizado_Nao_Supervisionado.pdf | grep Pages
