# Backpropagation e Redes Neurais (Araken de Medeiros Santos)

- Livro: `Backpropagation_e_Redes_Neurais.docx`, `.pdf` e `.epub` (texto justificado); slides em `slides/` (9 PPTX) e `slides_redes_neurais.zip`; notebooks em `notebooks/`.
- Ambiente: `cd codigo && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 testes_rnbook.py`.
- Scripts por capítulo: `codigo/capNN/`. Reconstruir o livro: `bash build/make.sh` (requer Node com `docx`, pandoc e LibreOffice); slides: `NODE_PATH=<pptxgenjs> node build/slides/gerar_slides.js`. Exercícios: edite `cap/src/exNN.src.txt` e rode `python3 build/gerar_ex.py` antes do `make.sh` (gera `cap/exNN.txt`). O PDF exige o pacote `libreoffice-math` (sem ele as equações somem). EPUB: `cd build/epub && python3 preparar_epub.py ../../Backpropagation_e_Redes_Neurais.docx /tmp/epub_src.docx && pandoc /tmp/epub_src.docx -f docx -t epub3 -o ../../Backpropagation_e_Redes_Neurais.epub --metadata-file=metadata.yaml --css=epub.css --epub-cover-image=../../figs/capa.png --toc --toc-depth=3 --split-level=1 --mathml`.
- Ver `PENDENCIAS.md` para limitações.
