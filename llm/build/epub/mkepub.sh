#!/bin/bash
# Gera o EPUB a partir do DOCX final (requer pandoc >= 3.1). Uso: bash build/epub/mkepub.sh
# 1) remove estilos duplicados do docx-js (pandoc não reconhece os títulos com duplicatas); 2) troca espaços do código por NBSP
#    (preserva a indentação); 3) filtro Lua recompõe títulos (build/headings.json), remove o sumário estático e converte caixas de código.
set -e
R=/home/user/claude/llm; W=$(mktemp -d); cd $W; mkdir d; cd d; unzip -q $R/Grandes_Modelos_de_Linguagem.docx
python3 - <<'PY'
import re
s=open('word/styles.xml',encoding='utf8').read()
sty=list(re.finditer(r'<w:style [^>]*w:styleId="([^"]+)"[^>]*>.*?</w:style>',s,re.S)); last={m.group(1):m.start() for m in sty}
for m in reversed(sty):
    if last[m.group(1)]!=m.start(): s=s[:m.start()]+s[m.end():]
open('word/styles.xml','w',encoding='utf8').write(s)
x=open('word/document.xml',encoding='utf8').read()
def fix(m):
    r=m.group(0)
    return re.sub(r'(<w:t[^>]*>)([^<]*)(</w:t>)',lambda k:k.group(1)+k.group(2).replace(' ',' ')+k.group(3),r) if 'Consolas' in r and 'w:val="15"' in r else r
open('word/document.xml','w',encoding='utf8').write(re.sub(r'<w:r>.*?</w:r>',fix,x,flags=re.S))
PY
zip -q -r ../limpo.docx . -x '.*'; cd ..
python3 -c "
import json;h=json.load(open('$R/build/headings.json'));open('/tmp/epub_heads.tsv','w',encoding='utf8').write('\n'.join(f\"{x['level']}\t{x['text']}\" for x in h))"
sed 's#/tmp/epub/heads.tsv#/tmp/epub_heads.tsv#' $R/build/epub/filtro.lua > f.lua
pandoc limpo.docx -f docx -t epub3 -o $R/Grandes_Modelos_de_Linguagem.epub --lua-filter=f.lua --toc --toc-depth=3 --split-level=1 --css=$R/build/epub/estilo.css --metadata title="Grandes Modelos de Linguagem" --metadata subtitle="Dos fundamentos aos sistemas, com Python" --metadata author="Araken de Medeiros Santos" --metadata lang=pt-BR --metadata date=2026 --epub-cover-image=$R/figs/capa.png
