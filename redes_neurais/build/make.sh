set -e
cd /home/user/claude/redes_neurais/build
SOFF=$(ls /root/.claude/skills/synced/*/docx/scripts/office/soffice.py | head -1)
OUT=/home/user/claude/redes_neurais/Backpropagation_e_Redes_Neurais.docx
rm -f pages.json
node build.js /tmp/p1raw.docx
python3 posproc.py /tmp/p1raw.docx /tmp/pass1.docx
python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf1 /tmp/pass1.docx >/dev/null 2>&1
python3 pages.py /tmp/pdf1/pass1.pdf
node build.js /tmp/p2raw.docx >/dev/null
python3 posproc.py /tmp/p2raw.docx $OUT
python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf2 $OUT >/dev/null 2>&1
cp pages.json pages_pass1.json
python3 pages.py /tmp/pdf2/Backpropagation_e_Redes_Neurais.pdf
cmp pages.json pages_pass1.json && echo "PAGINAS ESTAVEIS"
pdfinfo /tmp/pdf2/Backpropagation_e_Redes_Neurais.pdf | grep Pages
