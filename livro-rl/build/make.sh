set -e
cd /home/user/claude/livro-rl/build
export NODE_PATH=$(npm root -g 2>/dev/null)
SOFF=$(ls /root/.claude/skills/synced/*/docx/scripts/office/soffice.py | head -1)
OUT=${OUT:-/home/user/claude/livro-rl/livro.docx}
rm -f pages.json
node build.js /tmp/pass1.docx >/dev/null
python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf1 /tmp/pass1.docx >/dev/null 2>&1
python3 pages.py /tmp/pdf1/pass1.pdf
node build.js $OUT
python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf2 $OUT >/dev/null 2>&1
cp pages.json pages_pass1.json
python3 pages.py /tmp/pdf2/$(basename ${OUT%.docx}).pdf
cmp pages.json pages_pass1.json && echo "PAGINAS ESTAVEIS"
pdfinfo /tmp/pdf2/$(basename ${OUT%.docx}).pdf | grep Pages
