set -e
cd /home/user/claude/llm/build
SOFF=$(ls /root/.claude/skills/synced/*/docx/scripts/office/soffice.py | head -1)
OUTDOC=/home/user/claude/llm/Grandes_Modelos_de_Linguagem.docx
node eqs.js
rm -f pages.json
node build.js /tmp/pass1.docx >/dev/null
python3 postdocx.py /tmp/pass1.docx
rm -rf /tmp/pdf1; python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf1 /tmp/pass1.docx >/dev/null 2>&1
python3 pages.py /tmp/pdf1/pass1.pdf
node build.js $OUTDOC
python3 postdocx.py $OUTDOC
rm -rf /tmp/pdf2; python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf2 $OUTDOC >/dev/null 2>&1
cp pages.json pages_pass1.json
python3 pages.py /tmp/pdf2/Grandes_Modelos_de_Linguagem.pdf
cmp pages.json pages_pass1.json && echo "PAGINAS ESTAVEIS"
pdfinfo /tmp/pdf2/Grandes_Modelos_de_Linguagem.pdf | grep Pages
