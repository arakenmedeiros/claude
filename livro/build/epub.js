// Gera a versão EPUB 3 do livro a partir da mesma estrutura de dados usada para o DOCX.
// Texto justificado, hifenização automática, imagens, sumário navegável e gabarito idêntico ao do DOCX.
const zlib = require("zlib");

module.exports = function buildEpub(ctx) {
  const { fs, path, chapters, parseMd, readTxt, codeOf, saidaOf, PART_DESC, TITLE, SUBTITLE, AUTHOR, YEAR, FIGS, OUTFILE } = ctx;
  const JSZip = require("jszip");

  // ------------------------------------------------------------ utilidades
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const slug = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

  function math(text) {
    // _x, _(…), _{…} -> <sub>; ^x, ^(…), ^{…} -> <sup>
    const re = /_\(([^)]*)\)|_\{([^}]*)\}|_([A-Za-z0-9ÀÿΑ-ωα-ω]+)|\^\(([^)]*)\)|\^\{([^}]*)\}|\^([A-Za-z0-9]+)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(text)) !== null) {
      out += esc(text.slice(last, m.index));
      const sub = m[1] !== undefined || m[2] !== undefined || m[3] !== undefined;
      out += `<${sub ? "sub" : "sup"}>${esc(m[1] ?? m[2] ?? m[3] ?? m[4] ?? m[5] ?? m[6])}</${sub ? "sub" : "sup"}>`;
      last = m.index + m[0].length;
    }
    return out + esc(text.slice(last));
  }
  function inl(text) {
    const re = /(\*\*(?:[^*]|\*(?!\*)[^*]+\*)+\*\*|\*[^*\s][^*]*\*|`[^`]+`|(?<!R)\$(?=\S)[^$\n]*?\S\$)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(text)) !== null) {
      out += esc(text.slice(last, m.index));
      const t = m[0];
      if (t.startsWith("**")) out += `<strong>${inl(t.slice(2, -2))}</strong>`;
      else if (t.startsWith("`")) out += `<code>${esc(t.slice(1, -1))}</code>`;
      else if (t.startsWith("$")) out += `<span class="math">${math(t.slice(1, -1))}</span>`;
      else out += `<em>${esc(t.slice(1, -1))}</em>`;
      last = m.index + t.length;
    }
    return out + esc(text.slice(last));
  }

  const KW = new Set("def class for in import from as if elif else return with while lambda and or not True False None try except pass break continue yield".split(" "));
  function codeHtml(lines, lang) {
    const plain = lang === "bash" || lang === "saída";
    return lines.map((line) => {
      if (plain) return esc(line);
      const re = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b([A-Za-z_]+)\b|(\b\d+(?:\.\d+)?\b)/g;
      let out = "", last = 0, m;
      while ((m = re.exec(line)) !== null) {
        out += esc(line.slice(last, m.index));
        if (m[1]) out += `<span class="c">${esc(m[1])}</span>`;
        else if (m[2]) out += `<span class="s">${esc(m[2])}</span>`;
        else if (m[3]) out += KW.has(m[3]) ? `<span class="k">${esc(m[3])}</span>` : esc(m[3]);
        else out += `<span class="n">${esc(m[4])}</span>`;
        last = m.index + m[0].length;
      }
      return out + esc(line.slice(last));
    }).join("\n");
  }

  // ------------------------------------------------------------ estado
  const images = new Map(); // nome -> caminho
  const files = []; // {href, id, title, xhtml, props?}
  const nav = []; // árvore do sumário

  function img(name) {
    const file = path.join(FIGS, name + ".png");
    if (!fs.existsSync(file)) throw new Error("figura ausente: " + name);
    images.set(name + ".png", file);
    return `images/${name}.png`;
  }

  const CALL = { dica: "DICA", atencao: "ATENÇÃO", cotidiano: "NO SEU DIA A DIA", saibamais: "SAIBA MAIS", exemplo: "EXEMPLO", memorize: "MEMORIZE" };

  // ------------------------------------------------------------ blocos -> html
  function listHtml(tag, items, cls = "") {
    return `<${tag}${cls ? ` class="${cls}"` : ""}>` + items.map((it) => {
      const text = typeof it === "string" ? it : it.text; const subs = typeof it === "string" ? [] : it.subs;
      return `<li>${inl(text)}${subs.length ? `<ul>${subs.map((s) => `<li>${inl(s)}</li>`).join("")}</ul>` : ""}</li>`;
    }).join("") + `</${tag}>`;
  }
  function calloutBody(lines) {
    let html = "", para = [], i = 0;
    const flush = () => { if (para.length) { html += `<p>${inl(para.join(" "))}</p>`; para = []; } };
    while (i < lines.length) {
      const ln = lines[i];
      if (/^- /.test(ln)) { flush(); const items = []; while (i < lines.length && /^- /.test(lines[i])) { items.push(lines[i].slice(2)); i++; } html += listHtml("ul", items); continue; }
      if (ln.trim() === "") { flush(); i++; continue; }
      para.push(ln.trim()); i++;
    }
    flush(); return html;
  }
  function tableHtml(rows) {
    const [head, ...body] = rows;
    return `<table><thead><tr>${head.map((c) => `<th>${inl(c)}</th>`).join("")}</tr></thead><tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inl(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  function blocksHtml(blocks, ids) {
    let out = "";
    for (const b of blocks) {
      switch (b.t) {
        case "h3": { const id = ids.next(b.text); out += `<h2 id="${id}">${esc(b.text.replace(/\*+/g, "").replace(/`/g, ""))}</h2>`; ids.list.push({ level: 2, id, text: b.text.replace(/\*+/g, "").replace(/`/g, "") }); break; }
        case "h4": { const id = ids.next(b.text); out += `<h3 id="${id}">${esc(b.text.replace(/\*+/g, "").replace(/`/g, ""))}</h3>`; break; }
        case "p": out += `<p>${inl(b.text)}</p>`; break;
        case "quote": out += `<blockquote><p>${inl(b.text)}</p></blockquote>`; break;
        case "ul": out += listHtml("ul", b.items); break;
        case "ol": out += listHtml("ol", b.items); break;
        case "callout": out += `<div class="callout k-${esc(b.kind)}"><p class="callout-title"><span class="tag">${CALL[b.kind] || "DICA"}</span> ${inl(b.title)}</p>${calloutBody(b.body)}</div>`; break;
        case "code": out += `<div class="code"><p class="code-label">${esc((b.lang || "código").toUpperCase())}</p><pre><code>${codeHtml(b.code, b.lang)}</code></pre></div>`; break;
        case "formula": out += `<p class="formula">${math(b.text)}</p>`; break;
        case "table": out += `<div class="table-wrap">${tableHtml(b.table || b.rows)}</div>`; break;
        case "fig": out += `<figure><img src="${img(b.id)}" alt="${esc(b.caption.replace(/\*/g, ""))}"/><figcaption>${inl(b.caption)}</figcaption></figure>`; break;
      }
    }
    return out;
  }
  function makeIds(prefix) {
    const seen = new Set(); const list = [];
    return { list, next(t) { let s = prefix + "-" + (slug(t).slice(0, 40) || "s"); let k = s, n = 2; while (seen.has(k)) k = s + "-" + n++; seen.add(k); return k; } };
  }

  function page(title, body, extraClass = "") {
    return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="pt-BR" lang="pt-BR">
<head><meta charset="utf-8"/><title>${esc(title)}</title><link rel="stylesheet" type="text/css" href="style.css"/></head>
<body class="${extraClass}">${body}</body>
</html>`;
  }
  function addFile(id, title, body, opts = {}) {
    const href = id + ".xhtml";
    files.push({ id, href, title, xhtml: page(title, body, opts.cls || ""), props: opts.props, linear: opts.linear !== false });
    return href;
  }

  // ------------------------------------------------------------ capa e folha de rosto
  images.set("capa.png", path.join(FIGS, "capa.png"));
  addFile("capa", "Capa", `<div class="capa"><img src="images/capa.png" alt="Capa do livro ${esc(TITLE)}"/></div>`, { cls: "cover" });
  addFile("rosto", "Folha de rosto", `<div class="rosto"><h1 class="rosto-titulo">${esc(TITLE.toUpperCase())}</h1><p class="rosto-sub">${esc(SUBTITLE)}</p><p class="rosto-autor">${esc(AUTHOR)}</p><p class="rosto-ano">${YEAR}</p></div>`);
  nav.push({ title: "Capa", href: "capa.xhtml" }, { title: "Folha de rosto", href: "rosto.xhtml" });

  // ------------------------------------------------------------ apresentação
  {
    const md = parseMd(readTxt("apresentacao.md")); const ids = makeIds("ap");
    const body = `<h1 id="ap-top">Apresentação</h1>` + blocksHtml(md.blocks, ids);
    addFile("apresentacao", "Apresentação", body);
    nav.push({ title: "Apresentação", href: "apresentacao.xhtml", children: ids.list.map((h) => ({ title: h.text, href: `apresentacao.xhtml#${h.id}` })) });
  }

  // ------------------------------------------------------------ capítulos
  let partNav = null;
  for (const c of chapters) {
    const nn = String(c.n).padStart(2, "0");
    if (c.meta.part) {
      const [pnum, ptitle] = c.meta.part.split(" — ");
      const list = []; for (let k = c.n - 1; k < chapters.length; k++) { if (k > c.n - 1 && chapters[k].meta.part) break; list.push(chapters[k]); }
      const body = `<div class="parte"><p class="parte-num">${esc(pnum.toUpperCase())}</p><h1 id="parte-${nn}">${esc(c.meta.part)}</h1><p class="parte-desc">${esc(PART_DESC[c.meta.part])}</p><p class="parte-lista-titulo">Nesta parte:</p><ul>${list.map((x) => `<li>Cap. ${x.n} — ${esc(x.meta.title)}</li>`).join("")}</ul></div>`;
      addFile(`parte-${nn}`, c.meta.part, body, { cls: "parte-pag" });
      partNav = { title: c.meta.part, href: `parte-${nn}.xhtml`, children: [] }; nav.push(partNav);
    }
    const ids = makeIds("c" + nn);
    const objs = `<div class="objetivos"><p class="obj-tag">OBJETIVOS DE APRENDIZAGEM</p><p class="obj-lead"><strong>Ao final deste capítulo você será capaz de:</strong></p>${listHtml("ul", c.meta.obj)}</div>`;
    const bannerFile = path.join(FIGS, `banner${c.meta.banner || c.n}.png`); images.set(`banner${c.meta.banner || c.n}.png`, bannerFile);
    let body = `<p class="banner"><img src="images/banner${c.meta.banner || c.n}.png" alt="Faixa decorativa do capítulo ${c.n}"/></p>`;
    body += `<h1 id="cap-${nn}">Capítulo ${c.n} — ${esc(c.meta.title)}</h1>` + objs + blocksHtml(c.blocks, ids);

    // exercícios de múltipla escolha e discursivos
    const exIds = [];
    body += `<h2 id="c${nn}-mc">Exercícios de múltipla escolha</h2><p class="nota">Escolha <strong>uma</strong> alternativa em cada questão. O gabarito está no Apêndice A.</p>`;
    c.ex.mc.forEach((q, i) => { body += `<div class="questao"><p><strong class="qn">Questão ${i + 1}.</strong> ${inl(q.stem)}</p><ol class="alt" type="a">${q.opts.map((o) => `<li>${inl(o)}</li>`).join("")}</ol></div>`; });
    body += `<h2 id="c${nn}-disc">Exercícios discursivos</h2><p class="nota">Responda com suas palavras, em poucas linhas. Compare depois com a resposta-modelo do Apêndice A.</p><ol class="disc">${c.ex.dc.map((q) => `<li>${inl(q.stem)}</li>`).join("")}</ol>`;

    // exercícios práticos e projeto
    if (c.pr) {
      const exs = c.pr.items.filter((x) => x.kind === "EX"); const proj = c.pr.items.find((x) => x.kind === "PROJ");
      if (exs.length) {
        body += `<h2 id="c${nn}-pr">Exercícios práticos em Python (scikit-learn)</h2><p class="nota">Resolva os exercícios no seu computador (ou no Google Colab). Eles usam conjuntos de dados da UCI que já vêm no scikit-learn ou dados simulados. Tente resolver <strong>antes</strong> de consultar a solução, no Apêndice D.</p>`;
        for (const it of exs) body += `<h3>Exercício ${it.num} — ${esc(it.title)}</h3><p class="nivel">NÍVEL: ${esc(it.level.toUpperCase())}</p>` + blocksHtml(parseMd(it.stmt.join("\n")).blocks, makeIds("x"));
      }
      if (c.pr.alem.length) body += `<div class="callout k-saibamais"><p class="callout-title"><span class="tag">SAIBA MAIS</span> Para ir além: bases públicas da UCI e do Kaggle</p>${calloutBody([...c.pr.alem, "", "Estes conjuntos não foram usados nem testados nos códigos do livro. Confira, nos sites, a descrição, a licença e as colunas antes de usá-los."])}</div>`;
      if (proj) body += `<h2 id="c${nn}-proj">${c.n === 14 ? "Projeto integrador final" : "Projeto integrador do capítulo"}</h2><h3>${esc(proj.title)}</h3>` + blocksHtml(parseMd(proj.stmt.join("\n")).blocks, makeIds("p")) + `<p class="nota">A solução de referência está no Apêndice E.</p>`;
    }
    addFile(`cap-${nn}`, `Capítulo ${c.n} — ${c.meta.title}`, body);
    const entry = { title: `Capítulo ${c.n} — ${c.meta.title}`, href: `cap-${nn}.xhtml`, children: [
      ...ids.list.map((h) => ({ title: h.text, href: `cap-${nn}.xhtml#${h.id}` })),
      { title: "Exercícios de múltipla escolha", href: `cap-${nn}.xhtml#c${nn}-mc` }, { title: "Exercícios discursivos", href: `cap-${nn}.xhtml#c${nn}-disc` },
      ...(c.pr && c.pr.items.some((x) => x.kind === "EX") ? [{ title: "Exercícios práticos em Python", href: `cap-${nn}.xhtml#c${nn}-pr` }] : []),
      ...(c.pr && c.pr.items.some((x) => x.kind === "PROJ") ? [{ title: c.n === 14 ? "Projeto integrador final" : "Projeto integrador do capítulo", href: `cap-${nn}.xhtml#c${nn}-proj` }] : []) ] };
    (partNav ? partNav.children : nav).push(entry);
  }

  // ------------------------------------------------------------ apêndices
  const short = (c) => c.meta.title.split(":")[0];
  // A: gabarito
  {
    let body = `<h1 id="apA">Apêndice A — Gabarito dos exercícios</h1><p>Esta seção traz a resposta de todas as questões. Nas questões de múltipla escolha, aparecem a alternativa correta e uma breve justificativa. Nas discursivas, aparece uma <strong>resposta-modelo</strong>: a sua resposta pode ser redigida de outro modo, desde que contemple as mesmas ideias.</p>`;
    const ch = [];
    for (const c of chapters) {
      body += `<h2 id="apA-${c.n}">Capítulo ${c.n} — ${esc(short(c))}</h2><h3>Múltipla escolha</h3><table class="gab"><thead><tr><th>Questão</th><th>Resposta</th><th>Justificativa</th></tr></thead><tbody>${c.ex.mc.map((q, i) => `<tr><td>${i + 1}</td><td><strong>${q.ans})</strong></td><td>${inl(q.why)}</td></tr>`).join("")}</tbody></table><h3>Discursivas (respostas-modelo)</h3><ol class="disc">${c.ex.dc.map((q) => `<li>${inl(q.ans)}</li>`).join("")}</ol>`;
      ch.push({ title: `Capítulo ${c.n} — ${short(c)}`, href: `apendice-a.xhtml#apA-${c.n}` });
    }
    addFile("apendice-a", "Apêndice A — Gabarito", body); nav.push({ title: "Apêndice A — Gabarito dos exercícios", href: "apendice-a.xhtml", children: ch });
  }
  // B: glossário
  {
    const lines = readTxt("glossario.txt").trim().split("\n").filter(Boolean).sort((a, b) => a.localeCompare(b, "pt"));
    const body = `<h1 id="apB">Apêndice B — Glossário</h1><p>Termos técnicos usados no livro, explicados de forma simples.</p><dl>${lines.map((ln) => { const k = ln.indexOf(":"); return `<dt>${esc(ln.slice(0, k))}</dt><dd>${inl(ln.slice(k + 1).trim())}</dd>`; }).join("")}</dl>`;
    addFile("apendice-b", "Apêndice B — Glossário", body); nav.push({ title: "Apêndice B — Glossário", href: "apendice-b.xhtml" });
  }
  // C: guia rápido
  {
    const md = parseMd(readTxt("guia.md")); const ids = makeIds("g");
    const body = `<h1 id="apC">${esc(md.meta.title)}</h1>` + blocksHtml(md.blocks, ids);
    addFile("apendice-c", md.meta.title, body); nav.push({ title: md.meta.title, href: "apendice-c.xhtml", children: ids.list.map((h) => ({ title: h.text, href: `apendice-c.xhtml#${h.id}` })) });
  }
  // D: soluções práticas
  {
    let body = `<h1 id="apD">Apêndice D — Soluções dos exercícios práticos</h1><p>Cada solução traz o código, a <strong>saída obtida</strong> na elaboração do livro e um comentário sobre o que observar. Há várias maneiras corretas de resolver um exercício: o importante é que o seu resultado seja coerente e que você consiga explicá-lo.</p>`;
    const ch = [];
    for (const c of chapters) {
      if (!c.pr) continue; const exs = c.pr.items.filter((x) => x.kind === "EX"); if (!exs.length) continue;
      body += `<h2 id="apD-${c.n}">Capítulo ${c.n} — ${esc(short(c))}</h2>`; ch.push({ title: `Capítulo ${c.n} — ${short(c)}`, href: `apendice-d.xhtml#apD-${c.n}` });
      for (const it of exs) {
        body += `<h3>Exercício ${it.num} — ${esc(it.title)}</h3><div class="code"><p class="code-label">PYTHON</p><pre><code>${codeHtml(codeOf(it), "python")}</code></pre></div>`;
        const sd = saidaOf(c.n, it); if (sd) body += `<div class="code"><p class="code-label">SAÍDA</p><pre><code>${codeHtml(sd, "saída")}</code></pre></div>`;
        if (it.note.length) body += `<div class="callout k-exemplo"><p class="callout-title"><span class="tag">EXEMPLO</span> O que observar</p>${calloutBody(it.note)}</div>`;
      }
    }
    addFile("apendice-d", "Apêndice D — Soluções práticas", body); nav.push({ title: "Apêndice D — Soluções dos exercícios práticos", href: "apendice-d.xhtml", children: ch });
  }
  // E: projetos
  {
    let body = `<h1 id="apE">Apêndice E — Soluções de referência dos projetos integradores</h1><p>Estas são <strong>soluções de referência</strong>: caminhos possíveis, e não os únicos. Compare com o seu trabalho, observe as decisões e discuta as diferenças.</p>`;
    const ch = [];
    for (const c of chapters) {
      if (!c.pr) continue; const proj = c.pr.items.find((x) => x.kind === "PROJ"); if (!proj) continue;
      body += `<h2 id="apE-${c.n}">Capítulo ${c.n} — ${esc(short(c))}</h2><h3>${esc(proj.title)}</h3>`; ch.push({ title: `Capítulo ${c.n} — ${short(c)}`, href: `apendice-e.xhtml#apE-${c.n}` });
      if (proj.rubrica.filter((l) => l.trim()).length) body += `<p><strong>Critérios de avaliação</strong></p>` + blocksHtml(parseMd(proj.rubrica.join("\n")).blocks, makeIds("r"));
      body += `<div class="code"><p class="code-label">PYTHON</p><pre><code>${codeHtml(codeOf(proj), "python")}</code></pre></div>`;
      const sd = saidaOf(c.n, proj); if (sd) body += `<div class="code"><p class="code-label">SAÍDA</p><pre><code>${codeHtml(sd, "saída")}</code></pre></div>`;
      if (proj.note.length) body += `<div class="callout k-exemplo"><p class="callout-title"><span class="tag">EXEMPLO</span> Discussão dos resultados</p>${calloutBody(proj.note)}</div>`;
    }
    addFile("apendice-e", "Apêndice E — Projetos", body); nav.push({ title: "Apêndice E — Soluções de referência dos projetos integradores", href: "apendice-e.xhtml", children: ch });
  }
  // Referências
  {
    const lines = readTxt("referencias.txt").trim().split("\n").filter(Boolean).sort((a, b) => a.localeCompare(b, "pt"));
    const body = `<h1 id="refs">Referências</h1><p class="nota">Lista de artigos de periódicos e conferências de referência, além de livros reconhecidos, citados ao longo do texto.</p>${lines.map((ln) => `<p class="ref">${esc(ln)}</p>`).join("")}`;
    addFile("referencias", "Referências", body); nav.push({ title: "Referências", href: "referencias.xhtml" });
  }

  // ------------------------------------------------------------ sumário (nav.xhtml)
  const navList = (items) => `<ol>${items.map((n) => `<li><a href="${n.href}">${esc(n.title)}</a>${n.children && n.children.length ? navList(n.children) : ""}</li>`).join("")}</ol>`;
  const navXhtml = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="pt-BR" lang="pt-BR">
<head><meta charset="utf-8"/><title>Sumário</title><link rel="stylesheet" type="text/css" href="style.css"/></head>
<body class="sumario"><nav epub:type="toc" id="toc"><h1>Sumário</h1>${navList(nav)}</nav>
<nav epub:type="landmarks" hidden="hidden"><h2>Marcos</h2><ol><li><a epub:type="cover" href="capa.xhtml">Capa</a></li><li><a epub:type="toc" href="nav.xhtml">Sumário</a></li><li><a epub:type="bodymatter" href="apresentacao.xhtml">Início do texto</a></li></ol></nav></body></html>`;

  // NCX (compatibilidade com leitores antigos)
  let playOrder = 0;
  const ncxPoints = (items) => items.map((n) => { const id = ++playOrder; return `<navPoint id="np${id}" playOrder="${id}"><navLabel><text>${esc(n.title)}</text></navLabel><content src="${n.href}"/>${n.children && n.children.length ? ncxPoints(n.children) : ""}</navPoint>`; }).join("");
  const UID = "urn:uuid:3f6c1e4a-2b8d-4c5e-9a71-6d0b8e2f1a47";
  const ncx = `<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head><meta name="dtb:uid" content="${UID}"/><meta name="dtb:depth" content="3"/><meta name="dtb:totalPageCount" content="0"/><meta name="dtb:maxPageNumber" content="0"/></head><docTitle><text>${esc(TITLE)}</text></docTitle><navMap>${ncxPoints(nav)}</navMap></ncx>`;

  // ------------------------------------------------------------ CSS (texto justificado)
  const css = `
@charset "utf-8";
html { font-size: 100%; }
body { font-family: "Georgia", "Times New Roman", serif; line-height: 1.5; color: #1F2937; margin: 0 5%; text-align: justify; -webkit-hyphens: auto; -epub-hyphens: auto; hyphens: auto; orphans: 2; widows: 2; }
p, li, dd, blockquote, figcaption { text-align: justify; -webkit-hyphens: auto; -epub-hyphens: auto; hyphens: auto; }
p { margin: 0 0 0.75em 0; }
h1, h2, h3 { font-family: "Helvetica Neue", Arial, sans-serif; color: #312E81; text-align: left; -webkit-hyphens: none; -epub-hyphens: none; hyphens: none; page-break-after: avoid; break-after: avoid; }
h1 { font-size: 1.8em; margin: 1.2em 0 0.6em; border-bottom: 3px solid #F59E0B; padding-bottom: 0.2em; }
h2 { font-size: 1.35em; margin: 1.6em 0 0.5em; color: #4F46E5; }
h3 { font-size: 1.1em; margin: 1.2em 0 0.4em; color: #4338CA; }
a { color: #4F46E5; text-decoration: none; }
ul, ol { margin: 0.4em 0 0.9em 0; padding-left: 1.4em; }
li { margin-bottom: 0.3em; }
code { font-family: "Courier New", Consolas, monospace; font-size: 0.88em; background: #F3F4F6; color: #B91C1C; padding: 0 0.2em; border-radius: 2px; }
.math { font-family: "Cambria Math", "Times New Roman", serif; font-style: italic; color: #312E81; }
.formula { text-align: center; font-family: "Cambria Math", "Times New Roman", serif; font-size: 1.1em; color: #312E81; background: #EEF2FF; border-top: 1px solid #C7D2FE; border-bottom: 1px solid #C7D2FE; padding: 0.5em 0.4em; margin: 1em 0; -webkit-hyphens: none; -epub-hyphens: none; hyphens: none; }
figure { margin: 1.2em 0; text-align: center; page-break-inside: avoid; break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 0.82em; font-style: italic; color: #6B7280; margin-top: 0.4em; }
.banner { text-align: center; margin: 0 0 1em 0; } .banner img { max-width: 100%; }
.callout { margin: 1.1em 0; padding: 0.7em 0.9em; border-left: 6px solid #4F46E5; background: #EEF2FF; page-break-inside: avoid; break-inside: avoid; }
.callout p { font-size: 0.95em; margin: 0 0 0.5em 0; }
.callout-title { font-family: "Helvetica Neue", Arial, sans-serif; font-weight: bold; margin-bottom: 0.4em !important; text-align: left !important; -webkit-hyphens: none; -epub-hyphens: none; hyphens: none; }
.callout .tag { font-size: 0.72em; letter-spacing: 0.05em; margin-right: 0.4em; }
.k-dica { border-color: #10B981; background: #ECFDF5; } .k-dica .callout-title { color: #047857; }
.k-atencao { border-color: #F59E0B; background: #FFFBEB; } .k-atencao .callout-title { color: #B45309; }
.k-cotidiano { border-color: #06B6D4; background: #ECFEFF; } .k-cotidiano .callout-title { color: #0E7490; }
.k-saibamais { border-color: #8B5CF6; background: #F5F3FF; } .k-saibamais .callout-title { color: #6D28D9; }
.k-exemplo { border-color: #4F46E5; background: #EEF2FF; } .k-exemplo .callout-title { color: #4338CA; }
.k-memorize { border-color: #EC4899; background: #FDF2F8; } .k-memorize .callout-title { color: #BE185D; }
.objetivos { margin: 0 0 1.4em 0; padding: 0.7em 0.9em; border-left: 6px solid #4F46E5; background: #EEF2FF; }
.obj-tag { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 0.72em; font-weight: bold; letter-spacing: 0.06em; color: #4F46E5; margin: 0 0 0.3em 0; text-align: left; }
.code { margin: 1em 0; border: 1px solid #E5E7EB; border-left: 5px solid #4F46E5; background: #F8FAFC; padding: 0.4em 0.7em; }
.code-label { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 0.65em; letter-spacing: 0.08em; font-weight: bold; color: #4F46E5; margin: 0 0 0.3em 0; text-align: left; }
pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere; text-align: left; -webkit-hyphens: none; -epub-hyphens: none; hyphens: none; font-family: "Courier New", Consolas, monospace; font-size: 0.78em; line-height: 1.35; }
pre code { background: none; color: #1F2937; padding: 0; font-size: 1em; }
.c { color: #6B7280; font-style: italic; } .s { color: #B45309; } .k { color: #7C3AED; font-weight: bold; } .n { color: #0369A1; }
.table-wrap { overflow-x: auto; margin: 1em 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 0.8em 0; }
th { background: #4F46E5; color: #FFFFFF; text-align: left; padding: 0.35em 0.5em; font-family: "Helvetica Neue", Arial, sans-serif; border: 1px solid #C7D2FE; }
td { padding: 0.3em 0.5em; border: 1px solid #E5E7EB; vertical-align: top; text-align: left; -webkit-hyphens: none; -epub-hyphens: none; hyphens: none; }
tr:nth-child(even) td { background: #F5F7FF; }
table.gab td:nth-child(2) { text-align: center; color: #4F46E5; }
blockquote { margin: 1em 0; padding: 0 0 0 0.9em; border-left: 4px solid #06B6D4; font-style: italic; }
.questao { margin: 0.9em 0; page-break-inside: avoid; break-inside: avoid; }
.qn { color: #4F46E5; }
ol.alt { list-style-type: lower-alpha; margin: 0.2em 0 0.6em 0; padding-left: 2em; }
ol.alt li { margin-bottom: 0.15em; }
ol.disc li { margin-bottom: 0.7em; }
.nota { color: #6B7280; font-size: 0.92em; }
.nivel { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 0.7em; font-weight: bold; letter-spacing: 0.08em; color: #4F46E5; margin: 0 0 0.4em 0; text-align: left; }
.ref { padding-left: 1.6em; text-indent: -1.6em; font-size: 0.9em; margin-bottom: 0.5em; }
dt { font-weight: bold; color: #312E81; font-family: "Helvetica Neue", Arial, sans-serif; margin-top: 0.7em; text-align: left; }
dd { margin: 0 0 0.3em 1.2em; }
.cover { margin: 0; padding: 0; text-align: center; } .cover img { max-width: 100%; max-height: 100%; }
.rosto { text-align: center; margin-top: 18%; } .rosto-titulo { font-size: 1.9em; border: none; text-align: center !important; color: #312E81; letter-spacing: 0.04em; }
.rosto-sub { font-style: italic; color: #6B7280; text-align: center; font-size: 1.2em; border-bottom: 3px solid #F59E0B; display: inline-block; padding-bottom: 0.4em; margin-bottom: 2em; }
.rosto-autor { font-size: 1.3em; font-weight: bold; text-align: center; margin-top: 2em; } .rosto-ano { color: #6B7280; text-align: center; margin-top: 3em; }
.parte { margin-top: 18%; } .parte-num { font-family: "Helvetica Neue", Arial, sans-serif; font-weight: bold; letter-spacing: 0.15em; font-size: 1.2em; color: #4F46E5; text-align: left; }
.parte-desc { font-style: italic; font-size: 1.1em; border-top: 5px solid #4F46E5; padding-top: 0.6em; margin: 1em 0 1.4em; }
.parte-lista-titulo { font-weight: bold; text-align: left; }
.parte-pag h1 { border: none; }
nav ol { list-style-type: none; padding-left: 1em; } nav > ol { padding-left: 0; } nav li { margin-bottom: 0.15em; text-align: left; } nav a { color: #1F2937; }
`;

  // ------------------------------------------------------------ pacote
  const zip = new JSZip();
  zip.file("mimetype", "application/epub+zip", { compression: "STORE" });
  zip.file("META-INF/container.xml", `<?xml version="1.0" encoding="UTF-8"?>\n<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>`);
  zip.file("OEBPS/style.css", css);
  zip.file("OEBPS/nav.xhtml", navXhtml);
  zip.file("OEBPS/toc.ncx", ncx);
  for (const f of files) zip.file("OEBPS/" + f.href, f.xhtml);
  for (const [name, file] of images) zip.file("OEBPS/images/" + name, fs.readFileSync(file));

  const manifest = [
    `<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>`,
    `<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>`,
    `<item id="css" href="style.css" media-type="text/css"/>`,
    ...files.map((f) => `<item id="${f.id}" href="${f.href}" media-type="application/xhtml+xml"${f.props ? ` properties="${f.props}"` : ""}/>`),
    ...[...images.keys()].map((n, i) => `<item id="img${i}" href="images/${n}" media-type="image/png"${n === "capa.png" ? ` properties="cover-image"` : ""}/>`),
  ].join("\n    ");
  const spine = [`<itemref idref="capa" linear="yes"/>`, `<itemref idref="rosto"/>`, `<itemref idref="nav"/>`, ...files.filter((f) => f.id !== "capa" && f.id !== "rosto").map((f) => `<itemref idref="${f.id}"/>`)].join("\n    ");
  const modified = new Date().toISOString().replace(/\.\d+Z$/, "Z");
  zip.file("OEBPS/content.opf", `<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="pt-BR">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">${UID}</dc:identifier>
    <dc:title>${esc(TITLE)}: ${esc(SUBTITLE)}</dc:title>
    <dc:creator>${esc(AUTHOR)}</dc:creator>
    <dc:language>pt-BR</dc:language>
    <dc:date>${YEAR}</dc:date>
    <dc:description>Livro didático sobre aprendizado não supervisionado: agrupamento, redução de dimensionalidade, regras de associação, detecção de anomalias e aprendizado profundo, com Python e scikit-learn.</dc:description>
    <dc:subject>Aprendizado de máquina</dc:subject>
    <meta property="dcterms:modified">${modified}</meta>
    <meta name="cover" content="img${[...images.keys()].indexOf("capa.png")}"/>
  </metadata>
  <manifest>
    ${manifest}
  </manifest>
  <spine toc="ncx">
    ${spine}
  </spine>
</package>`);

  return zip.generateAsync({ type: "nodebuffer", mimeType: "application/epub+zip", compression: "DEFLATE", compressionOptions: { level: 9 } }).then((buf) => {
    fs.writeFileSync(OUTFILE, buf);
    console.log("epub ok", OUTFILE, files.length, "arquivos,", images.size, "imagens,", (buf.length / 1048576).toFixed(1), "MB");
  });
};
