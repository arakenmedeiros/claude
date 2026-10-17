const fs = require("fs");
const path = require("path");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, HeadingLevel, LevelFormat, TabStopType, LeaderType, Header, Footer, PageNumber, SectionType,
  VerticalAlign, PageBreak, HorizontalPositionRelativeFrom, VerticalPositionRelativeFrom, TextWrappingType,
} = D;

const ROOT = path.join(__dirname, "..");
const CAP = path.join(ROOT, "cap");
const FIGS = path.join(ROOT, "figs");
const OUT = process.argv[2] || path.join(ROOT, "AutoML_Teoria_e_Pratica.docx");
const PAGES = fs.existsSync(path.join(__dirname, "pages.json")) ? JSON.parse(fs.readFileSync(path.join(__dirname, "pages.json"), "utf8")) : {};

const FONT = "Calibri", MONO = "Consolas";
const C = { ind: "4F46E5", indD: "312E81", cya: "06B6D4", grn: "10B981", amb: "F59E0B", red: "EF4444", vio: "8B5CF6", pnk: "EC4899", ink: "1F2937", gry: "6B7280", light: "F3F4F6", mid: "E5E7EB" };
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134; // A4, 2 cm
const CW = PAGE_W - 2 * MARGIN; // 9638

const TITLE = "AutoML";
const SUBTITLE = "Teoria e Prática";
const AUTHOR = "Araken de Medeiros Santos";
const YEAR = "2026";

// ---------------------------------------------------------------- helpers
const readTxt = (f) => fs.readFileSync(path.join(CAP, f), "utf8").replace(/\r/g, "");
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// inline parser: **bold**, *italic*, `code`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*(?:[^*]|\*(?!\*)[^*]+\*)+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(...runs(t.slice(2, -2), { ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: MONO, size: (base.size || 21) - 2, color: "B91C1C", shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" } }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

// ---------------------------------------------------------------- numbering
const numberingConfigs = [];
let numCounter = 0;
function newNumRef() {
  const ref = "num" + (++numCounter);
  numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 320 } }, run: { bold: true, color: C.ind } } }] });
  return ref;
}
numberingConfigs.push({ reference: "bul", levels: [
  { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } }, run: { color: C.ind, bold: true } } },
  { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 280 } } } } ] });

// ---------------------------------------------------------------- md parser -> blocks
function parseMd(text) {
  const lines = text.split("\n");
  const blocks = []; const meta = { obj: [] };
  let i = 0; let para = [];
  const flush = () => { if (para.length) { blocks.push({ t: "p", text: para.join(" ") }); para = []; } };
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.startsWith("@part:")) { meta.part = ln.slice(6).trim(); i++; continue; }
    if (ln.startsWith("@banner:")) { meta.banner = parseInt(ln.slice(8)); i++; continue; }
    if (ln.startsWith("@obj:")) { meta.obj.push(ln.slice(5).trim()); i++; continue; }
    if (/^# /.test(ln)) { flush(); meta.title = ln.slice(2).trim(); i++; continue; }
    if (/^### /.test(ln)) { flush(); blocks.push({ t: "h4", text: ln.slice(4).trim() }); i++; continue; }
    if (/^## /.test(ln)) { flush(); blocks.push({ t: "h3", text: ln.slice(3).trim() }); i++; continue; }
    if (ln.startsWith("```")) { flush(); const lang = ln.slice(3).trim(); const code = []; i++; while (i < lines.length && !lines[i].startsWith("```")) { code.push(lines[i]); i++; } i++; blocks.push({ t: "code", lang, code }); continue; }
    if (ln.startsWith(":::")) { flush(); const m = ln.match(/^:::(\w+)\s*(.*)$/); const body = []; i++; while (i < lines.length && lines[i].trim() !== ":::") { body.push(lines[i]); i++; } i++; blocks.push({ t: "callout", kind: m[1], title: m[2], body }); continue; }
    const fig = ln.match(/^!\[(\w+)\]\((.*)\)\s*$/);
    if (fig) { flush(); blocks.push({ t: "fig", id: fig[1], caption: fig[2] }); i++; continue; }
    if (ln.startsWith("$$")) { flush(); blocks.push({ t: "formula", text: ln.replace(/^\$\$\s*/, "").replace(/\s*\$\$$/, "") }); i++; continue; }
    if (ln.startsWith("|")) { flush(); const rows = []; while (i < lines.length && lines[i].startsWith("|")) { rows.push(lines[i]); i++; } blocks.push({ t: "table", rows: rows.filter((r) => !/^\|\s*-{2,}/.test(r)).map((r) => r.replace(/^\||\|\s*$/g, "").split("|").map((c) => c.trim())) }); continue; }
    if (/^- /.test(ln)) { flush(); const items = []; while (i < lines.length && /^- /.test(lines[i])) { items.push(lines[i].slice(2)); i++; } blocks.push({ t: "ul", items }); continue; }
    if (/^\d+\. /.test(ln)) { flush(); const items = []; while (i < lines.length && /^\d+\. /.test(lines[i])) { items.push(lines[i].replace(/^\d+\.\s+/, "")); i++; } blocks.push({ t: "ol", items }); continue; }
    if (ln.trim() === "") { flush(); i++; continue; }
    if (ln.startsWith("> ")) { flush(); blocks.push({ t: "quote", text: ln.slice(2) }); i++; continue; }
    para.push(ln.trim()); i++;
  }
  flush();
  return { meta, blocks };
}

// ---------------------------------------------------------------- builders
const headings = []; // for TOC: {level, text, id}
let hid = 0;
function heading(level, text, opts = {}) {
  text = text.replace(/\*+/g, "").replace(/`/g, "");
  const id = "h" + (++hid);
  const toc = opts.toc !== false;
  if (toc) headings.push({ level, text, id });
  const lv = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3, 4: HeadingLevel.HEADING_4 }[level];
  return new Paragraph({ heading: lv, pageBreakBefore: !!opts.pageBreak, keepNext: true, children: [new TextRun({ text })] });
}
function P(text, o = {}) {
  return new Paragraph({ spacing: { after: 120, line: 300 }, alignment: o.align || AlignmentType.LEFT, keepNext: o.keepNext, indent: o.indent, children: runs(text, { size: 21, ...(o.run || {}) }) });
}
function noBorder() { const n = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }; return { top: n, bottom: n, left: n, right: n }; }

function calloutParas(body, color) {
  const out = []; let para = [];
  const flush = () => { if (para.length) { out.push(new Paragraph({ spacing: { after: 80, line: 280 }, children: runs(para.join(" "), { size: 20 }) })); para = []; } };
  let i = 0;
  while (i < body.length) {
    const ln = body[i];
    if (/^- /.test(ln)) { flush(); while (i < body.length && /^- /.test(body[i])) { out.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(body[i].slice(2), { size: 20 }) })); i++; } continue; }
    if (ln.trim() === "") { flush(); i++; continue; }
    para.push(ln.trim()); i++;
  }
  flush();
  return out;
}
const CALL = {
  dica: { c: C.grn, f: "ECFDF5", label: "DICA" }, atencao: { c: C.amb, f: "FFFBEB", label: "ATENÇÃO" }, cotidiano: { c: C.cya, f: "ECFEFF", label: "NO SEU DIA A DIA" },
  saibamais: { c: C.vio, f: "F5F3FF", label: "SAIBA MAIS" }, exemplo: { c: C.ind, f: "EEF2FF", label: "EXEMPLO" }, memorize: { c: C.pnk, f: "FDF2F8", label: "MEMORIZE" },
};
function callout(b) {
  const s = CALL[b.kind] || CALL.dica;
  const norm = (t) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  const dup = norm(b.title).startsWith(norm(s.label.split(" ")[0])) || (b.kind === "cotidiano" && norm(b.title).startsWith("no seu dia")) || (b.kind === "exemplo" && norm(b.title).startsWith("exemplo"));
  const kids = [new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [...(dup ? [] : [new TextRun({ text: s.label + "  ", bold: true, size: 17, color: s.c, characterSpacing: 20 })]), new TextRun({ text: b.title, bold: true, size: 22, color: s.c })] }), ...calloutParas(b.body, s.c)];
  const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: s.f, color: "auto" }, margins: { top: 100, bottom: 60, left: 200, right: 160 }, borders: { top: nb, bottom: nb, right: nb, left: { style: BorderStyle.SINGLE, size: 36, color: s.c } }, children: kids })] })] });
}
function spacer(after = 120) { return new Paragraph({ spacing: { after, line: 120 }, children: [] }); }

const KW = new Set("def class for in import from as if elif else return with while lambda and or not True False None try except pass break continue yield".split(" "));
function codeRuns(line) {
  const out = []; const re = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b([A-Za-z_]+)\b|(\b\d+(?:\.\d+)?\b)/g; let last = 0, m;
  const base = { font: MONO, size: 15 };
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: line.slice(last, m.index), ...base, color: C.ink }));
    if (m[1]) out.push(new TextRun({ text: m[1], ...base, color: "6B7280", italics: true }));
    else if (m[2]) out.push(new TextRun({ text: m[2], ...base, color: "B45309" }));
    else if (m[3]) out.push(new TextRun({ text: m[3], ...base, color: KW.has(m[3]) ? "7C3AED" : C.ink, bold: KW.has(m[3]) }));
    else out.push(new TextRun({ text: m[4], ...base, color: "0369A1" }));
    last = m.index + m[0].length;
  }
  if (last < line.length) out.push(new TextRun({ text: line.slice(last), ...base, color: C.ink }));
  if (!out.length) out.push(new TextRun({ text: " ", ...base }));
  return out;
}
function codeBlock(b) {
  const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const kids = b.code.map((ln) => new Paragraph({ spacing: { after: 0, line: 240 }, children: (b.lang === "bash" || b.lang === "saída") ? [new TextRun({ text: ln || " ", font: MONO, size: 15, color: C.ink })] : codeRuns(ln) }));
  const label = b.lang ? b.lang.toUpperCase() : "CÓDIGO";
  return [new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: "F8FAFC", color: "auto" }, margins: { top: 100, bottom: 100, left: 200, right: 120 }, borders: { top: { style: BorderStyle.SINGLE, size: 4, color: C.mid }, bottom: { style: BorderStyle.SINGLE, size: 4, color: C.mid }, right: { style: BorderStyle.SINGLE, size: 4, color: C.mid }, left: { style: BorderStyle.SINGLE, size: 24, color: C.ind } }, children: [new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: label, bold: true, size: 14, color: C.ind, characterSpacing: 30 })] }), ...kids] })] })] }), spacer(140)];
}
function formulaRuns(text) {
  const out = []; const re = /_\(([^)]*)\)|_([A-Za-z0-9ÀÿΑ-ωα-ω]+)|\^\(([^)]*)\)|\^([A-Za-z0-9]+)/g; let last = 0, m;
  const base = { font: "Cambria Math", size: 23, color: C.indD };
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const sub = m[1] !== undefined || m[2] !== undefined;
    out.push(new TextRun({ text: m[1] ?? m[2] ?? m[3] ?? m[4], ...base, subScript: sub, superScript: !sub }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
function formula(b) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 160 }, shading: { type: ShadingType.CLEAR, fill: "EEF2FF", color: "auto" }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: "C7D2FE", space: 4 }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "C7D2FE", space: 4 } }, children: formulaRuns(b.text) });
}
function table(b) {
  const rows = b.rows; const n = rows[0].length;
  const lens = new Array(n).fill(0).map((_, j) => Math.max(...rows.map((r) => Math.min(40, Math.max(6, (r[j] || "").length)))));
  const tot = lens.reduce((a, c) => a + c, 0);
  let widths = lens.map((l) => Math.max(900, Math.floor((l / tot) * CW)));
  const diff = CW - widths.reduce((a, c) => a + c, 0); widths[widths.length - 1] += diff;
  const bd = { style: BorderStyle.SINGLE, size: 4, color: C.mid };
  const trs = rows.map((r, ri) => new TableRow({ tableHeader: ri === 0, cantSplit: true, children: r.map((c, j) => new TableCell({ width: { size: widths[j], type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: ri === 0 ? C.ind : ri % 2 === 0 ? "F5F7FF" : "FFFFFF", color: "auto" }, margins: { top: 60, bottom: 60, left: 100, right: 100 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ spacing: { after: 0, line: 252 }, children: runs(c, { size: 18, ...(ri === 0 ? { bold: true, color: "FFFFFF" } : {}) }) })] })) }));
  return [new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: trs }), spacer(160)];
}
function figure(b) {
  const file = path.join(FIGS, b.id + ".png");
  if (!fs.existsSync(file)) throw new Error("figura ausente: " + b.id);
  const { w, h } = pngSize(file);
  let dw = Math.min(600, (w / 200) * 96 * 1.0), dh = (dw * h) / w;
  if (dh > 400) { dh = 400; dw = (dh * w) / h; }
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(dw), height: Math.round(dh) }, altText: { title: b.id, description: b.caption, name: b.id } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(b.caption, { size: 17, italics: true, color: C.gry }) }),
  ];
}
function banner(n) {
  const file = path.join(FIGS, `banner${n}.png`); const { w, h } = pngSize(file);
  const dw = 642, dh = Math.round((642 * h) / w);
  return new Paragraph({ spacing: { after: 160 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: dw, height: dh }, altText: { title: "banner", description: "Faixa decorativa do capítulo " + n, name: "banner" + n } })] });
}

function renderBlocks(blocks) {
  const out = [];
  for (const b of blocks) {
    switch (b.t) {
      case "h3": out.push(heading(3, b.text)); break;
      case "h4": out.push(heading(4, b.text)); break;
      case "p": out.push(P(b.text)); break;
      case "quote": out.push(new Paragraph({ spacing: { after: 120 }, indent: { left: 400 }, border: { left: { style: BorderStyle.SINGLE, size: 18, color: C.cya, space: 8 } }, children: runs(b.text, { size: 21, italics: true }) })); break;
      case "ul": b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 70, line: 288 }, children: runs(it, { size: 21 }) }))); out.push(spacer(60)); break;
      case "ol": { const ref = newNumRef(); b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 70, line: 288 }, children: runs(it, { size: 21 }) }))); out.push(spacer(60)); break; }
      case "callout": out.push(callout(b)); out.push(spacer(140)); break;
      case "code": out.push(...codeBlock(b)); break;
      case "formula": out.push(formula(b)); break;
      case "table": out.push(...table(b)); break;
      case "fig": out.push(...figure(b)); break;
    }
  }
  return out;
}

// ---------------------------------------------------------------- questions
function parseEx(text, seed) {
  const rnd = mulberry32(seed);
  const blocks = text.trim().split(/\n\s*\n/);
  const mc = [], dc = [];
  for (const bl of blocks) {
    const L = bl.split("\n");
    if (L[0].startsWith("MC:")) {
      const stem = L[0].slice(3).trim();
      let opts = [L[1], L[2], L[3], L[4]].map((o) => o.replace(/^[abcd]\)\s*/, ""));
      const rm = L[5].match(/^R:\s*([abcd])\s*\|\s*(.*)$/);
      const correct = "abcd".indexOf(rm[1]);
      const idx = [0, 1, 2, 3]; for (let k = 3; k > 0; k--) { const j = Math.floor(rnd() * (k + 1)); [idx[k], idx[j]] = [idx[j], idx[k]]; }
      const newOpts = idx.map((ix) => opts[ix]); const newCorrect = idx.indexOf(correct);
      mc.push({ stem, opts: newOpts, ans: "abcd"[newCorrect], why: rm[2].trim() });
    } else if (L[0].startsWith("DISC:")) dc.push({ stem: L[0].slice(5).trim(), ans: L[1].replace(/^R:\s*/, "").trim() });
  }
  return { mc, dc };
}
function exercises(ex) {
  const out = [];
  out.push(heading(3, "Exercícios de múltipla escolha", { pageBreak: true }));
  out.push(P("Escolha **uma** alternativa em cada questão. O gabarito está no Apêndice A.", { run: { color: C.gry } }));
  ex.mc.forEach((q, i) => {
    out.push(new Paragraph({ keepNext: true, spacing: { before: 140, after: 60, line: 276 }, children: [new TextRun({ text: `Questão ${i + 1}. `, bold: true, color: C.ind, size: 21 }), ...runs(q.stem, { size: 21 })] }));
    q.opts.forEach((o, k) => out.push(new Paragraph({ keepNext: k < 3, keepLines: true, spacing: { after: 30, line: 264 }, indent: { left: 500, hanging: 300 }, children: [new TextRun({ text: "abcd"[k] + ")\t", bold: true, color: C.gry, size: 20 }), ...runs(o, { size: 20 })], tabStops: [{ type: TabStopType.LEFT, position: 500 }] })));
  });
  out.push(heading(3, "Exercícios discursivos", { pageBreak: true }));
  out.push(P("Responda com suas palavras, em poucas linhas. Compare depois com a resposta-modelo do Apêndice A.", { run: { color: C.gry } }));
  ex.dc.forEach((q, i) => out.push(new Paragraph({ keepLines: true, spacing: { before: 120, after: 80, line: 288 }, indent: { left: 500, hanging: 500 }, tabStops: [{ type: TabStopType.LEFT, position: 500 }], children: [new TextRun({ text: `${i + 1}.\t`, bold: true, color: C.ind, size: 21 }), ...runs(q.stem, { size: 21 })] })));
  return out;
}


// ---------------------------------------------------------------- exercícios práticos e projetos
const PR_OUT = fs.existsSync(path.join(__dirname, "pr_out.json")) ? JSON.parse(fs.readFileSync(path.join(__dirname, "pr_out.json"), "utf8")) : {};
function parsePr(text) {
  const lines = text.replace(/\r/g, "").split("\n"); const items = []; const alem = []; let cur = null, mode = null;
  for (const ln of lines) {
    if (ln.startsWith("@@EX ") || ln.startsWith("@@PROJ ")) {
      const kind = ln.startsWith("@@EX ") ? "EX" : "PROJ";
      const parts = ln.slice(ln.indexOf(" ") + 1).split("|").map((x) => x.trim());
      cur = kind === "EX" ? { kind, num: parseInt(parts[0]), title: parts[1], level: parts[2] || "", stmt: [], sol: [], rubrica: [], note: [] } : { kind, num: 0, title: parts[0], level: parts[1] || "", stmt: [], sol: [], rubrica: [], note: [] };
      items.push(cur); mode = "stmt"; continue;
    }
    if (ln.startsWith("@@ALEM")) { cur = null; mode = "alem"; continue; }
    if (ln.startsWith("@@SOL")) { mode = "sol"; continue; }
    if (ln.startsWith("@@RUBRICA")) { mode = "rubrica"; continue; }
    if (ln.startsWith("@@NOTE")) { mode = "note"; const rest = ln.slice(6).trim(); if (rest && cur) cur.note.push(rest); continue; }
    if (mode === "alem") { if (ln.trim()) alem.push(ln); continue; }
    if (cur && mode) cur[mode].push(ln);
  }
  return { items, alem };
}
function codeOf(it) { const m = it.sol.join("\n").match(/```python\n([\s\S]*?)```/); return m ? m[1].replace(/\n$/, "").split("\n") : []; }
function saidaOf(ch, it) {
  const key = `${ch}-${it.kind}${it.num}`; let out = PR_OUT[key]; if (out === undefined) return null;
  let L = out.replace(/\s+$/, "").split("\n").map((l) => (l.length > 118 ? l.slice(0, 117) + "…" : l));
  if (L.length > 48) L = L.slice(0, 48).concat(["… (saída truncada)"]);
  return L;
}
function praticos(c) {
  const pr = c.pr; if (!pr) return [];
  const exs = pr.items.filter((x) => x.kind === "EX"); const proj = pr.items.find((x) => x.kind === "PROJ");
  const out = [];
  if (exs.length) {
    out.push(heading(3, "Exercícios práticos em Python (scikit-learn)", { pageBreak: true }));
    out.push(P("Resolva os exercícios no seu computador (ou no Google Colab). Eles usam conjuntos de dados da UCI que já vêm no scikit-learn ou dados simulados. Tente resolver **antes** de consultar a solução, no Apêndice D. As saídas apresentadas lá foram obtidas com as versões de bibliotecas listadas na Apresentação.", { run: { color: C.gry } }));
    for (const it of exs) {
      out.push(heading(4, `Exercício ${it.num} — ${it.title}`));
      out.push(new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: `NÍVEL: ${it.level.toUpperCase()}`, bold: true, size: 15, color: C.ind, characterSpacing: 30 })] }));
      out.push(...renderBlocks(parseMd(it.stmt.join("\n")).blocks));
    }
  }
  if (pr.alem.length) {
    out.push(callout({ kind: "saibamais", title: "Para ir além: bases públicas da UCI e do Kaggle", body: [...pr.alem, "", "Estes conjuntos não foram usados nem testados nos códigos do livro. Confira, nos sites, a descrição, a licença e as colunas antes de usá-los."] }));
    out.push(spacer(140));
  }
  if (proj) {
    out.push(heading(3, c.n === 14 ? "Projeto integrador final" : "Projeto integrador do capítulo", { pageBreak: true }));
    out.push(heading(4, proj.title));
    out.push(...renderBlocks(parseMd(proj.stmt.join("\n")).blocks));
    out.push(P(`A solução de referência está no Apêndice E.`, { run: { color: C.gry, italics: true } }));
  }
  return out;
}

// ---------------------------------------------------------------- headers/footers
function mkHeader(text) {
  return new Header({ children: [new Paragraph({ alignment: AlignmentType.LEFT, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.ind, space: 4 } }, tabStops: [{ type: TabStopType.RIGHT, position: CW }], children: [new TextRun({ text: TITLE, size: 17, color: C.gry }), new TextRun({ text: "\t" + text, size: 17, color: C.ind, bold: true })] })] });
}
function mkFooter() {
  return new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: C.gry })] })] });
}
const secProps = (extra = {}) => ({ page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1300, bottom: 1200, left: MARGIN, right: MARGIN, header: 560, footer: 500 } }, ...extra });

// ---------------------------------------------------------------- content assembly
const sections = [];

// cover (floating image)
{
  const file = path.join(FIGS, "capa.png");
  sections.push({ properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 0, bottom: 0, left: 0, right: 0, header: 0, footer: 0 } } },
    children: [new Paragraph({ spacing: { after: 0 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: 794, height: 1123 }, floating: { horizontalPosition: { relative: HorizontalPositionRelativeFrom.PAGE, offset: 0 }, verticalPosition: { relative: VerticalPositionRelativeFrom.PAGE, offset: 0 }, wrap: { type: TextWrappingType.NONE }, behindDocument: true }, altText: { title: "Capa", description: "Capa do livro", name: "capa" } })] })] });
}
// title page
sections.push({ properties: secProps({ type: SectionType.NEXT_PAGE, }),
  children: [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 3600, after: 120 }, children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, size: 56, color: C.indD, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 12 } }, children: [new TextRun({ text: SUBTITLE, italics: true, size: 30, color: C.gry })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 }, children: [new TextRun({ text: AUTHOR, bold: true, size: 34, color: C.ink })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1800 }, children: [new TextRun({ text: YEAR, size: 24, color: C.gry })] }),
  ] });
function SectionVerticalAlignCenter() { return D.VerticalAlignSection.CENTER; }

// load content
const chapters = [];
for (let n = 1; n <= 14; n++) {
  const nn = String(n).padStart(2, "0");
  if (!fs.existsSync(path.join(CAP, `ch${nn}.md`))) continue;
  const md = parseMd(readTxt(`ch${nn}.md`));
  const ex = parseEx(readTxt(`ex${nn}.txt`), 1000 + n);
  const prf = path.join(CAP, "pr", `pr${nn}.txt`);
  chapters.push({ n, ...md, ex, pr: fs.existsSync(prf) ? parsePr(readTxt("pr/pr" + nn + ".txt")) : null });
}
const PART_DESC = {
  "Parte I — Fundamentos": "O terreno comum: o que é AutoML e por que ele existe, o essencial de aprendizado de máquina para quem começa do zero e como montar pipelines confiáveis, sem vazamento de dados.",
  "Parte II — Otimização de Hiperparâmetros": "O coração do AutoML: como buscar bons ajustes de forma inteligente. Da busca em grade e aleatória à otimização bayesiana, ao Hyperband e aos métodos evolutivos.",
  "Parte III — Automação do Pipeline": "Automatizar o que vem antes e depois do modelo: atributos, meta-aprendizado, ensembles e as principais ferramentas de AutoML em ação.",
  "Parte IV — AutoML Avançado": "Fronteiras do campo: a busca automática de arquiteturas de redes neurais (NAS) e o AutoML para séries temporais, texto e imagens.",
  "Parte V — AutoML Responsável e em Produção": "Como confiar, explicar e colocar em produção: explicabilidade, ética, vieses, custo ambiental, MLOps e um projeto integrador final.",
};
const PART_COL = { "Parte I": C.ind, "Parte II": C.cya, "Parte III": C.grn, "Parte IV": C.amb, "Parte V": C.vio };

// TOC placeholder section is built after we know headings -> build content first into temp arrays
const contentSections = [];

// front matter: Apresentação
{
  const md = parseMd(readTxt("apresentacao.md"));
  const ch = [heading(1, "Apresentação"), ...renderBlocks(md.blocks)];
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apresentação") }, footers: { default: mkFooter() }, children: ch });
}

for (const c of chapters) {
  if (c.meta.part) {
    const pk = c.meta.part.split(" — ")[0]; const col = PART_COL[pk];
    const [pnum, ptitle] = c.meta.part.split(" — ");
    const partChaps = chapters.filter((x) => x.meta.part === c.meta.part || (chapters.indexOf(x) > chapters.indexOf(c) && !x.meta.part && true));
    // list chapters belonging to this part
    const list = []; for (let k = chapters.indexOf(c); k < chapters.length; k++) { if (k > chapters.indexOf(c) && chapters[k].meta.part) break; list.push(chapters[k]); }
    const kids = [
      new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 3200, after: 100 }, children: [new TextRun({ text: pnum.toUpperCase(), bold: true, size: 40, color: col, characterSpacing: 60 })] }),
      heading(1, c.meta.part),
      new Paragraph({ spacing: { before: 200, after: 300 }, border: { top: { style: BorderStyle.SINGLE, size: 24, color: col, space: 10 } }, children: [new TextRun({ text: PART_DESC[c.meta.part], size: 26, color: C.ink, italics: true })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Nesta parte:", bold: true, size: 22, color: col })] }),
      ...list.map((x) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text: `Cap. ${x.n} — ${x.meta.title}`, size: 22 })] })),
    ];
    contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), children: kids });
  }
  const kids = [];
  kids.push(banner(c.meta.banner || c.n));
  kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title}`));
  // objectives
  const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  kids.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: "EEF2FF", color: "auto" }, margins: { top: 120, bottom: 80, left: 220, right: 180 }, borders: { top: nb, bottom: nb, right: nb, left: { style: BorderStyle.SINGLE, size: 36, color: C.ind } }, children: [new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "OBJETIVOS DE APRENDIZAGEM", bold: true, size: 17, color: C.ind, characterSpacing: 20 })] }), new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Ao final deste capítulo você será capaz de:", bold: true, size: 21 })] }), ...c.meta.obj.map((o) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 50, line: 276 }, children: runs(o, { size: 20 }) }))] })] })] }));
  kids.push(spacer(200));
  kids.push(...renderBlocks(c.blocks));
  kids.push(...exercises(c.ex));
  kids.push(...praticos(c));
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(`Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`) }, footers: { default: mkFooter() }, children: kids });
}

// ---------------------------------------------------------------- appendices
// A: gabarito
{
  const kids = [heading(1, "Apêndice A — Gabarito dos exercícios")];
  kids.push(P("Esta seção traz a resposta de todas as questões. Nas questões de múltipla escolha, aparecem a alternativa correta e uma breve justificativa. Nas discursivas, aparece uma **resposta-modelo**: a sua resposta pode ser redigida de outro modo, desde que contemple as mesmas ideias."));
  for (const c of chapters) {
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: c.n > 1 || true }));
    kids.push(heading(3, "Múltipla escolha"));
    const bd = { style: BorderStyle.SINGLE, size: 4, color: C.mid };
    const widths = [800, 900, CW - 1700];
    const mk = (cells, head) => new TableRow({ tableHeader: head, cantSplit: true, children: cells.map((t, j) => new TableCell({ width: { size: widths[j], type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: head ? C.ind : "FFFFFF", color: "auto" }, margins: { top: 40, bottom: 40, left: 90, right: 90 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, children: [new Paragraph({ alignment: j < 2 ? AlignmentType.CENTER : AlignmentType.LEFT, spacing: { after: 0, line: 240 }, children: runs(t, { size: 17, ...(head ? { bold: true, color: "FFFFFF" } : j === 1 ? { bold: true, color: C.ind } : {}) }) })] })) });
    kids.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [mk(["Questão", "Resposta", "Justificativa"], true), ...c.ex.mc.map((q, i) => mk([String(i + 1), q.ans + ")", q.why], false))] }));
    kids.push(spacer(160));
    kids.push(heading(3, "Discursivas (respostas-modelo)"));
    c.ex.dc.forEach((q, i) => kids.push(new Paragraph({ keepLines: true, spacing: { after: 90, line: 270 }, indent: { left: 500, hanging: 500 }, tabStops: [{ type: TabStopType.LEFT, position: 500 }], children: [new TextRun({ text: `${i + 1}.\t`, bold: true, color: C.ind, size: 20 }), ...runs(q.ans, { size: 20 })] })));
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice A — Gabarito") }, footers: { default: mkFooter() }, children: kids });
}
// B: glossário
{
  const kids = [heading(1, "Apêndice B — Glossário")];
  kids.push(P("Termos técnicos usados no livro, explicados de forma simples."));
  const lines = readTxt("glossario.txt").trim().split("\n").filter(Boolean).sort((a, b) => a.localeCompare(b, "pt"));
  for (const ln of lines) { const k = ln.indexOf(":"); kids.push(new Paragraph({ keepLines: true, spacing: { after: 90, line: 276 }, indent: { left: 0 }, children: [new TextRun({ text: ln.slice(0, k), bold: true, color: C.indD, size: 20 }), new TextRun({ text: " — " + ln.slice(k + 1).trim(), size: 20 })] })); }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice B — Glossário") }, footers: { default: mkFooter() }, children: kids });
}
// C: guia rápido
{
  const md = parseMd(readTxt("guia.md"));
  const kids = [heading(1, md.meta.title), ...renderBlocks(md.blocks)];
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice C — Guia rápido") }, footers: { default: mkFooter() }, children: kids });
}

// D: soluções dos exercícios práticos
{
  const kids = [heading(1, "Apêndice D — Soluções dos exercícios práticos")];
  kids.push(P("Cada solução traz o código, a **saída obtida** na elaboração do livro e um comentário sobre o que observar. Há várias maneiras corretas de resolver um exercício: o importante é que o seu resultado seja coerente e que você consiga explicá-lo. Pequenas diferenças numéricas podem ocorrer com outras versões das bibliotecas. Os gráficos aparecem quando você executa o código."));
  for (const c of chapters) {
    if (!c.pr) continue;
    const exs = c.pr.items.filter((x) => x.kind === "EX"); if (!exs.length) continue;
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: true }));
    for (const it of exs) {
      kids.push(heading(3, `Exercício ${it.num} — ${it.title}`));
      kids.push(...codeBlock({ lang: "python", code: codeOf(it) }));
      const sd = saidaOf(c.n, it);
      if (sd) kids.push(...codeBlock({ lang: "saída", code: sd }));
      if (it.note.length) kids.push(callout({ kind: "exemplo", title: "O que observar", body: it.note }), spacer(120));
    }
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice D — Soluções práticas") }, footers: { default: mkFooter() }, children: kids });
}
// E: projetos integradores
{
  const kids = [heading(1, "Apêndice E — Soluções de referência dos projetos integradores")];
  kids.push(P("Estas são **soluções de referência**: caminhos possíveis, e não os únicos. Compare com o seu trabalho, observe as decisões e discuta as diferenças. A avaliação de um projeto deve considerar também a qualidade da argumentação, e não apenas o código. Os critérios de avaliação de cada projeto aparecem antes do código."));
  for (const c of chapters) {
    if (!c.pr) continue;
    const proj = c.pr.items.find((x) => x.kind === "PROJ"); if (!proj) continue;
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: true }));
    kids.push(heading(3, proj.title));
    if (proj.rubrica.filter((l) => l.trim()).length) {
      kids.push(new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: "Critérios de avaliação", bold: true, size: 22, color: C.indD })] }));
      kids.push(...renderBlocks(parseMd(proj.rubrica.join("\n")).blocks));
    }
    kids.push(...codeBlock({ lang: "python", code: codeOf(proj) }));
    const sd = saidaOf(c.n, proj);
    if (sd) kids.push(...codeBlock({ lang: "saída", code: sd }));
    if (proj.note.length) kids.push(callout({ kind: "exemplo", title: "Discussão dos resultados", body: proj.note }), spacer(120));
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice E — Projetos") }, footers: { default: mkFooter() }, children: kids });
}
// Referências
{
  const kids = [heading(1, "Referências")];
  kids.push(P("Lista de artigos de periódicos e conferências de referência, além de livros reconhecidos, citados ao longo do texto.", { run: { color: C.gry } }));
  const lines = readTxt("referencias.txt").trim().split("\n").filter(Boolean).sort((a, b) => a.localeCompare(b, "pt"));
  for (const ln of lines) kids.push(new Paragraph({ keepLines: true, spacing: { after: 90, line: 264 }, indent: { left: 420, hanging: 420 }, children: [new TextRun({ text: ln, size: 19 })] }));
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Referências") }, footers: { default: mkFooter() }, children: kids });
}

// ---------------------------------------------------------------- TOC
function tocParagraphs() {
  const out = [];
  out.push(new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 6 } }, children: [new TextRun({ text: "Sumário", bold: true, size: 44, color: C.indD })] }));
  for (const h of headings) {
    const pg = PAGES[h.id] !== undefined ? String(PAGES[h.id]) : "000";
    const lvl = h.level; const ind = [0, 0, 300, 700, 1100][lvl];
    const style = { 1: { bold: true, size: 22, color: C.indD, before: 160 }, 2: { bold: true, size: 21, color: C.ink, before: 100 }, 3: { bold: false, size: 19, color: C.ink, before: 0 }, 4: { bold: false, size: 18, color: C.gry, before: 0 } }[lvl];
    out.push(new Paragraph({ spacing: { before: style.before, after: 20, line: 250 }, indent: { left: ind, right: 650 }, tabStops: [{ type: TabStopType.RIGHT, position: CW, leader: LeaderType.DOT }], keepLines: true, children: [new TextRun({ text: h.text, bold: style.bold, size: style.size, color: style.color }), new TextRun({ children: ["\t", pg], size: style.size, color: style.color, bold: style.bold })] }));
  }
  return out;
}

// assemble: cover, title, TOC, content
const tocSection = { properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Sumário") }, footers: { default: mkFooter() }, children: tocParagraphs() };

const doc = new Document({
  creator: AUTHOR, title: `${TITLE} — ${SUBTITLE}`, description: "Livro didático sobre AutoML (aprendizado de máquina automatizado)", language: "pt-BR",
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: C.ink } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 44, bold: true, color: C.indD, font: FONT }, paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 40, bold: true, color: C.ind, font: FONT }, paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 1, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: C.amb, space: 6 } } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: C.indD, font: FONT }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, color: "4338CA", font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 3 } },
    ],
  },
  numbering: { config: numberingConfigs },
  sections: [sections[0], sections[1], tocSection, ...contentSections],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  fs.writeFileSync(path.join(__dirname, "headings.json"), JSON.stringify(headings, null, 1));
  console.log("ok", OUT, headings.length, "títulos,", numCounter, "listas numeradas");
});
