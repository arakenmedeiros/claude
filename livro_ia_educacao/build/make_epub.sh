#!/bin/bash
# Uso: make_epub.sh [saida.epub]  — gera o DOCX da variante EPUB (sem capa e sem sumário de páginas) e o converte em EPUB 3
set -e
cd "$(dirname "$0")"
OUT=${1:-../saida/livro.epub}
EPUB=1 node build.js /tmp/ia_epub_src.docx >/dev/null
python3 epub.py /tmp/ia_epub_src.docx "$OUT"
python3 -m epubcheck "$OUT" && echo "epubcheck: sem erros ou avisos"
