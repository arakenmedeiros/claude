set -e
cd /home/user/claude/mlops/build
SOFF=$(ls /root/.claude/skills/synced/*/docx/scripts/office/soffice.py | head -1)
rm -f pages.json
node build.js /tmp/pass1.docx >/dev/null
python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf1 /tmp/pass1.docx >/dev/null 2>&1
python3 pages.py /tmp/pdf1/pass1.pdf
node build.js /home/user/claude/mlops/MLOps.docx
python3 $SOFF --headless --convert-to pdf --outdir /tmp/pdf2 /home/user/claude/mlops/MLOps.docx >/dev/null 2>&1
cp pages.json pages_pass1.json
python3 pages.py /tmp/pdf2/MLOps.pdf
cmp pages.json pages_pass1.json && echo "PAGINAS ESTAVEIS"
pdfinfo /tmp/pdf2/MLOps.pdf | grep Pages
