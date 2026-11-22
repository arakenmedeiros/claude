set -e
cd /home/user/claude/automl/build
EMIT_HTML=1 node build.js /tmp/x.docx
cd epub
pandoc livro.html -f html -t epub3 -o ../../AutoML_Teoria_e_Pratica.epub --toc --toc-depth=2 --split-level=1 --css epub.css \
  --epub-cover-image=figs/capa.png --no-highlight --metadata title="AutoML: Teoria e Prática" --metadata author="Araken de Medeiros Santos" \
  --metadata lang=pt-BR --metadata date=2026 --metadata rights="Todos os direitos reservados" 2>&1 | tail -5
ls -la ../../AutoML_Teoria_e_Pratica.epub
