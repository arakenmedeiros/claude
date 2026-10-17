#!/bin/bash
# Uso: make.sh [saida.docx]   (duas passagens para fixar as páginas do sumário)
set -e
cd "$(dirname "$0")"
OUT=${1:-../saida/livro.docx}
SOFF=$(ls /root/.claude/skills/synced/*/docx/scripts/office/soffice.py | head -1)
rm -f pages.json
node build.js /tmp/ia_pass1.docx >/dev/null
python3 post.py /tmp/ia_pass1.docx /tmp/ia_pass1p.docx >/dev/null
rm -rf /tmp/ia_pdf1; python3 $SOFF --headless --convert-to pdf --outdir /tmp/ia_pdf1 /tmp/ia_pass1p.docx >/dev/null 2>&1
python3 pages.py /tmp/ia_pdf1/ia_pass1p.pdf
node build.js $OUT
python3 post.py $OUT $OUT
rm -rf /tmp/ia_pdf2; python3 $SOFF --headless --convert-to pdf --outdir /tmp/ia_pdf2 $OUT >/dev/null 2>&1
cp pages.json pages_pass1.json
python3 pages.py /tmp/ia_pdf2/$(basename ${OUT%.docx}).pdf
cmp pages.json pages_pass1.json && echo "PAGINAS ESTAVEIS" || echo "PAGINAS DIFEREM (rodar novamente)"
pdfinfo /tmp/ia_pdf2/$(basename ${OUT%.docx}).pdf | grep Pages
