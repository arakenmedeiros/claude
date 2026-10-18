// Montagem do livro em DOCX. Equações: marcadores ⟦Mn⟧ substituídos por OMML (pandoc) em posproc.py.
// Uso: node build.js saida.docx
const fs = require("fs");
const path = require("path");
const D = require("/opt/node-tools/node_modules/docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, HeadingLevel, LevelFormat, TabStopType, LeaderType, Header, Footer, PageNumber, SectionType,
  VerticalAlign, HorizontalPositionRelativeFrom, VerticalPositionRelativeFrom, TextWrappingType,
} = D;

const ROOT = path.join(__dirname, "..");
const CAP = process.env.CAPDIR || path.join(ROOT, "cap");
const FIGS = path.join(ROOT, "figs");
const OUT = process.argv[2] || path.join(ROOT, "livro.docx");
const PAGES = fs.existsSync(path.join(__dirname, "pages.json")) ? JSON.parse(fs.readFileSync(path.join(__dirname, "pages.json"), "utf8")) : {};

const FONT = "Calibri", MONO = "Courier New";
const C = { ind: "4F46E5", indD: "312E81", cya: "06B6D4", grn: "10B981", amb: "F59E0B", red: "EF4444", vio: "8B5CF6", pnk: "EC4899", ink: "1F2937", gry: "6B7280", light: "F3F4F6", mid: "E5E7EB" };
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134; // A4, 2 cm
const CW = PAGE_W - 2 * MARGIN; // 9638

const TITLE = "Backpropagation e Redes Neurais";
const SUBTITLE = "Do perceptron ao Levenberg–Marquardt: fundamentos matemáticos e implementações em Python";
const AUTHOR = "Araken de Medeiros Santos";
const YEAR = "2026";

const readTxt = (f) => fs.readFileSync(path.join(CAP, f), "utf8").replace(/\r/g, "");
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

// ------------------------------------------------------------------ equações
const equations = []; // {id, tex, display}
let curChap = 0;
const eqLabels = {}; // label -> "c.k"
const figLabels = {};
function mathMarker(tex, display) { const id = equations.length; equations.push({ id, tex, display }); return `⟦M${id}⟧`; }

// ------------------------------------------------------------------ inline
function xref(t) {
  return t.replace(/\[eq:([\w-]+)\]/g, (m, k) => (eqLabels[k] ? `(${eqLabels[k]})` : "(?)"))
    .replace(/\[fig:([\w-]+)\]/g, (m, k) => (figLabels[k] ? `Figura ${figLabels[k]}` : "Figura ?"));
}
function leaf(text, base) {
  // divide texto em partes comuns e marcadores de equações inline ⟦Mn⟧
  const out = []; const re = /\u27e6(\d+)\u27e7/g; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    out.push(new TextRun({ text: `⟦M${m[1]}⟧`, ...base }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
function runs(text, base = {}) {
  text = xref(text);
  // protege matemática inline $...$ (ids = índice em equations)
  text = text.replace(/\$([^$]+)\$/g, (m, tex) => { const id = equations.length; equations.push({ id, tex, display: false }); return "\u27e6" + id + "\u27e7"; });
  return runs0(text, base);
}
function runs0(text, base) {
  const out = [];
  const re = /(\*\*(?:[^*]|\*(?!\*)[^*]+\*)+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(...leaf(text.slice(last, m.index), base));
    const t = m[0];
    if (t.startsWith("**")) out.push(...runs0(t.slice(2, -2), { ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: MONO, size: (base.size || 21) - 2, color: "B91C1C", shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" } }));
    else out.push(...runs0(t.slice(1, -1), { ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(...leaf(text.slice(last), base));
  return out;
}

// ------------------------------------------------------------------ numeração
const numberingConfigs = []; let numCounter = 0;
function newNumRef() {
  const ref = "num" + (++numCounter);
  numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 320 } }, run: { bold: true, color: C.ind } } }] });
  return ref;
}
numberingConfigs.push({ reference: "bul", levels: [
  { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } }, run: { color: C.ind, bold: true } } },
  { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 280 } } } } ] });

// ------------------------------------------------------------------ parser md
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
    if (ln.startsWith("$$")) {
      flush(); let buf = ln;
      while (!/\$\$\s*(\{eq:[\w-]+\})?\s*$/.test(buf) || buf.trim() === "$$") { i++; buf += "\n" + lines[i]; }
      i++;
      const m = buf.match(/^\$\$([\s\S]*?)\$\$\s*(?:\{eq:([\w-]+)\})?\s*$/);
      blocks.push({ t: "formula", tex: m[1].trim(), label: m[2] || null });
      continue;
    }
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

// ------------------------------------------------------------------ construtores
const headings = []; let hid = 0;
function heading(level, text, opts = {}) {
  text = text.replace(/\*+/g, "").replace(/`/g, "");
  const id = "h" + (++hid);
  if (opts.toc !== false) headings.push({ level, text, id });
  const lv = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3, 4: HeadingLevel.HEADING_4 }[level];
  // títulos com matemática inline: usa runs (mantém marcadores)
  return new Paragraph({ heading: lv, pageBreakBefore: !!opts.pageBreak, keepNext: true, children: [new TextRun({ text })] });
}
function P(text, o = {}) {
  return new Paragraph({ spacing: { after: 120, line: 300 }, alignment: o.align || AlignmentType.LEFT, keepNext: o.keepNext, indent: o.indent, children: runs(text, { size: 21, ...(o.run || {}) }) });
}
function calloutParas(body) {
  const out = []; let para = [];
  const flush = () => { if (para.length) { out.push(new Paragraph({ spacing: { after: 80, line: 280 }, children: runs(para.join(" "), { size: 20 }) })); para = []; } };
  let i = 0;
  while (i < body.length) {
    const ln = body[i];
    if (/^- /.test(ln)) { flush(); while (i < body.length && /^- /.test(body[i])) { out.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(body[i].slice(2), { size: 20 }) })); i++; } continue; }
    if (ln.startsWith("$$")) { // fórmula dentro de callout (sem número)
      let buf = ln; while (!/\$\$\s*(\{eq:[\w-]+\})?\s*$/.test(buf) || buf.trim() === "$$") { i++; if (i >= body.length) throw new Error("fórmula sem fim: " + ln.slice(0, 60)); buf += "\n" + body[i]; } i++;
      const tex = buf.replace(/^\$\$/, "").replace(/\s*\{eq:[\w-]+\}\s*$/, "").replace(/\$\$\s*$/, "").trim();
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 100 }, children: [new TextRun({ text: mathMarker(tex, true), size: 21 })] }));
      continue;
    }
    if (ln.trim() === "") { flush(); i++; continue; }
    para.push(ln.trim()); i++;
  }
  flush();
  return out;
}
const CALL = {
  dica: { c: C.grn, f: "ECFDF5", label: "DICA" }, atencao: { c: C.amb, f: "FFFBEB", label: "ATENÇÃO" }, cotidiano: { c: C.cya, f: "ECFEFF", label: "NO SEU DIA A DIA" },
  saibamais: { c: C.vio, f: "F5F3FF", label: "SAIBA MAIS" }, exemplo: { c: C.ind, f: "EEF2FF", label: "EXEMPLO RESOLVIDO" }, memorize: { c: C.pnk, f: "FDF2F8", label: "MEMORIZE" },
  situacao: { c: C.cya, f: "ECFEFF", label: "SITUAÇÃO INICIAL" }, prereq: { c: C.gry, f: "F3F4F6", label: "PRÉ-REQUISITOS" },
  erro: { c: C.red, f: "FEF2F2", label: "ERROS FREQUENTES" }, sintese: { c: C.grn, f: "ECFDF5", label: "SÍNTESE" },
  derivacao: { c: C.vio, f: "F5F3FF", label: "DERIVAÇÃO" }, limite: { c: C.amb, f: "FFFBEB", label: "LIMITAÇÕES" },
};
const nbd = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function callout(b) {
  const s = CALL[b.kind] || CALL.dica;
  const norm = (t) => (t || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  const dup = !b.title || norm(b.title).startsWith(norm(s.label.split(" ")[0]));
  const head = new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: s.label + (dup ? "" : "  "), bold: true, size: 17, color: s.c, characterSpacing: 20 }), ...(dup ? [] : [new TextRun({ text: b.title, bold: true, size: 22, color: s.c })])] });
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: s.f, color: "auto" }, margins: { top: 100, bottom: 60, left: 200, right: 160 }, borders: { top: nbd, bottom: nbd, right: nbd, left: { style: BorderStyle.SINGLE, size: 36, color: s.c } }, children: [head, ...calloutParas(b.body)] })] })] });
}
function spacer(after = 120) { return new Paragraph({ spacing: { after, line: 120 }, children: [] }); }

const KW = new Set("def class for in import from as if elif else return with while lambda and or not True False None try except pass break continue yield assert raise global".split(" "));
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
  const plain = ["bash", "saída", "algoritmo"].includes(b.lang);
  const kids = b.code.map((ln) => new Paragraph({ spacing: { after: 0, line: 240 }, children: plain ? [new TextRun({ text: ln || " ", font: MONO, size: 15, color: C.ink })] : codeRuns(ln) }));
  const label = b.lang === "algoritmo" ? "PSEUDOCÓDIGO" : b.lang ? b.lang.toUpperCase() : "CÓDIGO";
  const colr = b.lang === "algoritmo" ? C.vio : C.ind;
  const bd = { style: BorderStyle.SINGLE, size: 4, color: C.mid };
  return [new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: b.lang === "algoritmo" ? "FAF5FF" : "F8FAFC", color: "auto" }, margins: { top: 100, bottom: 100, left: 200, right: 120 }, borders: { top: bd, bottom: bd, right: bd, left: { style: BorderStyle.SINGLE, size: 24, color: colr } }, children: [new Paragraph({ spacing: { after: 60 }, keepNext: true, children: [new TextRun({ text: label, bold: true, size: 14, color: colr, characterSpacing: 30 })] }), ...kids] })] })] }), spacer(140)];
}
function formula(b) {
  const lab = b.num || "";
  const wN = 1000;
  const eqCell = new TableCell({ width: { size: CW - wN, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, borders: { top: nbd, bottom: nbd, left: nbd, right: nbd }, shading: { type: ShadingType.CLEAR, fill: "F5F7FF", color: "auto" }, margins: { top: 60, bottom: 60, left: 100, right: 100 }, children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: mathMarker(b.tex, true), size: 22 })] })] });
  const numCell = new TableCell({ width: { size: wN, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, borders: { top: nbd, bottom: nbd, left: nbd, right: nbd }, shading: { type: ShadingType.CLEAR, fill: "F5F7FF", color: "auto" }, children: [new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 0 }, children: [new TextRun({ text: lab ? `(${lab})` : "", size: 20, color: C.gry })] })] });
  return [new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW - wN, wN], rows: [new TableRow({ cantSplit: true, children: [eqCell, numCell] })] }), spacer(100)];
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
  let dw = Math.min(600, (w / 200) * 96), dh = (dw * h) / w;
  if (dh > 420) { dh = 420; dw = (dh * w) / h; }
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(dw), height: Math.round(dh) }, altText: { title: b.id, description: b.caption, name: b.id } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: `Figura ${b.num} — `, bold: true, size: 17, color: C.gry }), ...runs(b.caption, { size: 17, italics: true, color: C.gry })] }),
  ];
}
function banner(n) {
  const file = path.join(FIGS, `banner${n}.png`); if (!fs.existsSync(file)) return spacer(60);
  const { w, h } = pngSize(file); const dw = 642, dh = Math.round((642 * h) / w);
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
      case "formula": out.push(...formula(b)); break;
      case "table": out.push(...table(b)); break;
      case "fig": out.push(...figure(b)); break;
    }
  }
  return out;
}
// numeração de equações e figuras por capítulo (pré-passe)
function numberBlocks(blocks, chap) {
  let e = 0, f = 0;
  for (const b of blocks) {
    if (b.t === "formula") { e++; b.num = `${chap}.${e}`; if (b.label) eqLabels[b.label] = b.num; }
    if (b.t === "fig") { f++; b.num = `${chap}.${f}`; figLabels[b.id] = b.num; }
  }
}

// ------------------------------------------------------------------ questões
function parseEx(text) {
  const entries = []; let cur = null;
  for (const ln of text.split("\n")) {
    if (/^(MC|DISC):/.test(ln)) { cur = [ln]; entries.push(cur); } else if (cur) cur.push(ln);
  }
  const mc = [], dc = [];
  for (const L0 of entries) {
    const L = L0.slice(); while (L.length && !L[L.length - 1].trim()) L.pop();
    if (L[0].startsWith("MC:")) {
      const stem = L[0].slice(3).trim();
      const opts = L.slice(1, 6).map((o) => o.replace(/^[a-e]\)\s*/, ""));
      const rm = L[6].match(/^R:\s*([a-e])\s*\|\s*(.*)$/);
      mc.push({ stem, opts, ans: rm[1], why: rm[2].trim() });
    } else {
      const stem = L[0].slice(5).trim(); const ans = L.slice(1).join("\n").replace(/^\s*R:\s*/, "").trim();
      dc.push({ stem, ans });
    }
  }
  return { mc, dc };
}
function exercises(ex) {
  const out = [];
  out.push(heading(3, "Exercícios de múltipla escolha", { pageBreak: true }));
  out.push(P("Escolha **uma** alternativa em cada questão. O gabarito comentado está no Apêndice A.", { run: { color: C.gry } }));
  ex.mc.forEach((q, i) => {
    out.push(new Paragraph({ keepNext: true, spacing: { before: 140, after: 60, line: 276 }, children: [new TextRun({ text: `Questão ${i + 1}. `, bold: true, color: C.ind, size: 21 }), ...runs(q.stem, { size: 21 })] }));
    q.opts.forEach((o, k) => out.push(new Paragraph({ keepNext: k < 4, keepLines: true, spacing: { after: 30, line: 264 }, indent: { left: 500, hanging: 300 }, children: [new TextRun({ text: "abcde"[k] + ")\t", bold: true, color: C.gry, size: 20 }), ...runs(o, { size: 20 })], tabStops: [{ type: TabStopType.LEFT, position: 500 }] })));
  });
  out.push(heading(3, "Exercícios discursivos", { pageBreak: true }));
  out.push(P("Resolva por escrito, mostrando as etapas. As soluções estão no Apêndice B.", { run: { color: C.gry } }));
  ex.dc.forEach((q, i) => out.push(new Paragraph({ keepLines: true, spacing: { before: 120, after: 80, line: 288 }, indent: { left: 500, hanging: 500 }, tabStops: [{ type: TabStopType.LEFT, position: 500 }], children: [new TextRun({ text: `${i + 1}.\t`, bold: true, color: C.ind, size: 21 }), ...runs(q.stem, { size: 21 })] })));
  return out;
}

// ------------------------------------------------------------------ práticos e projetos
const PR_OUT = fs.existsSync(path.join(__dirname, "pr_out.json")) ? JSON.parse(fs.readFileSync(path.join(__dirname, "pr_out.json"), "utf8")) : {};
function parsePr(text) {
  const lines = text.replace(/\r/g, "").split("\n"); const items = []; let cur = null, mode = null;
  for (const ln of lines) {
    if (ln.startsWith("@@EX ") || ln.startsWith("@@PROJ ")) {
      const kind = ln.startsWith("@@EX ") ? "EX" : "PROJ";
      const parts = ln.slice(ln.indexOf(" ") + 1).split("|").map((x) => x.trim());
      cur = kind === "EX" ? { kind, num: parseInt(parts[0]), title: parts[1], level: parts[2] || "", stmt: [], sol: [], rubrica: [], note: [] } : { kind, num: 0, title: parts[0], level: parts[1] || "", stmt: [], sol: [], rubrica: [], note: [] };
      items.push(cur); mode = "stmt"; continue;
    }
    if (ln.startsWith("@@SOL")) { mode = "sol"; continue; }
    if (ln.startsWith("@@RUBRICA")) { mode = "rubrica"; continue; }
    if (ln.startsWith("@@NOTE")) { mode = "note"; const rest = ln.slice(6).trim(); if (rest && cur) cur.note.push(rest); continue; }
    if (cur && mode) cur[mode].push(ln);
  }
  return { items };
}
function codeOf(it) { const m = it.sol.join("\n").match(/```python\n([\s\S]*?)```/); return m ? m[1].replace(/\n$/, "").split("\n") : []; }
function saidaOf(ch, it) {
  const key = `${ch}-${it.kind}${it.num}`; const out = PR_OUT[key]; if (out === undefined) return null;
  let L = out.replace(/\s+$/, "").split("\n").map((l) => (l.length > 96 ? l.slice(0, 95) + "…" : l));
  if (L.length > 40) L = L.slice(0, 40).concat(["… (saída truncada)"]);
  return L;
}
function praticos(c) {
  const pr = c.pr; if (!pr) return [];
  const exs = pr.items.filter((x) => x.kind === "EX"); const proj = pr.items.find((x) => x.kind === "PROJ");
  const out = [];
  if (exs.length) {
    out.push(heading(3, "Exercícios práticos em Python (scikit-learn)", { pageBreak: true }));
    out.push(P("Resolva no seu computador ou no Google Colab. Tente resolver **antes** de consultar a solução (Apêndice J). Os exercícios usam bases embutidas no scikit-learn, que foram executadas na elaboração do livro. Indicamos também bases públicas da UCI e do Kaggle para ir além; essas bases NÃO foram baixadas nem testadas aqui.", { run: { color: C.gry } }));
    for (const it of exs) {
      out.push(heading(4, `Exercício prático ${c.n}.${it.num} — ${it.title}`));
      out.push(new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: `NÍVEL: ${it.level.toUpperCase()}`, bold: true, size: 15, color: C.ind, characterSpacing: 30 })] }));
      out.push(...renderBlocks(parseMd(it.stmt.join("\n")).blocks));
    }
  }
  if (proj) {
    out.push(heading(3, c.n === 0 ? "Projeto integrador final" : "Projeto do capítulo", { pageBreak: true }));
    out.push(heading(4, proj.title));
    out.push(...renderBlocks(parseMd(proj.stmt.join("\n")).blocks));
    out.push(P(`A solução de referência está no Apêndice K.`, { run: { color: C.gry, italics: true } }));
  }
  return out;
}

// ------------------------------------------------------------------ cabeçalho/rodapé
function mkHeader(text) {
  return new Header({ children: [new Paragraph({ alignment: AlignmentType.LEFT, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.ind, space: 4 } }, tabStops: [{ type: TabStopType.RIGHT, position: CW }], children: [new TextRun({ text: TITLE, size: 17, color: C.gry }), new TextRun({ text: "\t" + text, size: 17, color: C.ind, bold: true })] })] });
}
function mkFooter() { return new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: C.gry })] })] }); }
const secProps = (extra = {}) => ({ page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1300, bottom: 1200, left: MARGIN, right: MARGIN, header: 560, footer: 500 } }, ...extra });

// ------------------------------------------------------------------ montagem
const coverSections = [];
{
  const file = path.join(FIGS, "capa.png");
  coverSections.push({ properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 0, bottom: 0, left: 0, right: 0, header: 0, footer: 0 } } },
    children: [new Paragraph({ spacing: { after: 0 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: 794, height: 1123 }, floating: { horizontalPosition: { relative: HorizontalPositionRelativeFrom.PAGE, offset: 0 }, verticalPosition: { relative: VerticalPositionRelativeFrom.PAGE, offset: 0 }, wrap: { type: TextWrappingType.NONE }, behindDocument: true }, altText: { title: "Capa", description: "Capa do livro", name: "capa" } })] })] });
  coverSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }),
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 3000, after: 120 }, children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, size: 52, color: C.indD, font: FONT })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 12 } }, children: [new TextRun({ text: SUBTITLE, italics: true, size: 28, color: C.gry })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 }, children: [new TextRun({ text: AUTHOR, bold: true, size: 34, color: C.ink })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1800 }, children: [new TextRun({ text: YEAR, size: 24, color: C.gry })] }),
    ] });
}

const CHAPS = 8;
const chapters = [];
for (let n = 1; n <= CHAPS; n++) {
  const nn = String(n).padStart(2, "0");
  const md = parseMd(readTxt(`ch${nn}.md`));
  numberBlocks(md.blocks, n);
  const ex = parseEx(readTxt(`ex${nn}.txt`));
  const prf = path.join(CAP, "pr", `pr${nn}.txt`);
  chapters.push({ n, ...md, ex, pr: fs.existsSync(prf) ? parsePr(readTxt("pr/pr" + nn + ".txt")) : null });
}
const PART_DESC = {
  "Parte 1 — Introdução e fundamentos matemáticos e computacionais": "O vocabulário, a história e as ferramentas: derivadas, gradientes, métodos de otimização, mínimos quadrados e o ambiente Python, para que nada do que vem depois pareça mágica.",
  "Parte 2 — Perceptron e treinamento de redes de múltiplas camadas": "Do neurônio de Rosenblatt ao backpropagation e, em seguida, aos algoritmos que aceleram o treinamento: Quickprop, RPROP e suas variantes, Levenberg–Marquardt e Delta-Bar-Delta.",
};
const PART_COL = { "Parte 1": C.ind, "Parte 2": C.cya };

const contentSections = [];
// Apresentação
{
  const md = parseMd(readTxt("apresentacao.md")); numberBlocks(md.blocks, 0);
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apresentação") }, footers: { default: mkFooter() }, children: [heading(1, "Apresentação"), ...renderBlocks(md.blocks)] });
}
for (const c of chapters) {
  if (c.meta.part) {
    const pk = c.meta.part.split(" — ")[0]; const col = PART_COL[pk]; const [pnum] = c.meta.part.split(" — ");
    const list = []; for (let k = c.n - 1; k < chapters.length; k++) { if (k > c.n - 1 && chapters[k].meta.part) break; list.push(chapters[k]); }
    contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), children: [
      new Paragraph({ spacing: { before: 3200, after: 100 }, children: [new TextRun({ text: pnum.toUpperCase(), bold: true, size: 40, color: col, characterSpacing: 60 })] }),
      heading(1, c.meta.part),
      new Paragraph({ spacing: { before: 200, after: 300 }, border: { top: { style: BorderStyle.SINGLE, size: 24, color: col, space: 10 } }, children: [new TextRun({ text: PART_DESC[c.meta.part], size: 26, color: C.ink, italics: true })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Nesta parte:", bold: true, size: 22, color: col })] }),
      ...list.map((x) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text: `Cap. ${x.n} — ${x.meta.title}`, size: 22 })] })),
    ] });
  }
  const kids = [];
  kids.push(banner(c.meta.banner || c.n));
  kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title}`));
  kids.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: "EEF2FF", color: "auto" }, margins: { top: 120, bottom: 80, left: 220, right: 180 }, borders: { top: nbd, bottom: nbd, right: nbd, left: { style: BorderStyle.SINGLE, size: 36, color: C.ind } }, children: [new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "OBJETIVOS DE APRENDIZAGEM", bold: true, size: 17, color: C.ind, characterSpacing: 20 })] }), new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Ao final deste capítulo você será capaz de:", bold: true, size: 21 })] }), ...c.meta.obj.map((o) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 50, line: 276 }, children: runs(o, { size: 20 }) }))] })] })] }));
  kids.push(spacer(200));
  kids.push(...renderBlocks(c.blocks));
  kids.push(...exercises(c.ex));
  kids.push(...praticos(c));
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(`Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`) }, footers: { default: mkFooter() }, children: kids });
}
// projeto integrador final (após o capítulo 8)
let integrador = null;
{
  const f = path.join(CAP, "pr", "pr_final.txt");
  if (fs.existsSync(f)) {
    integrador = parsePr(fs.readFileSync(f, "utf8"));
    const proj = integrador.items.find((x) => x.kind === "PROJ");
    const kids = [heading(1, "Projeto integrador final"), heading(2, proj.title), ...renderBlocks(parseMd(proj.stmt.join("\n")).blocks), P("A solução de referência está no Apêndice K.", { run: { color: C.gry, italics: true } })];
    contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Projeto integrador final") }, footers: { default: mkFooter() }, children: kids });
  }
}

// apêndices de texto livre (arquivos apNN_*.md)
const apFiles = fs.readdirSync(CAP).filter((f) => /^ap_[a-z]+\.md$/.test(f)).sort();
function apendiceMd(file, headerText) {
  const md = parseMd(readTxt(file)); numberBlocks(md.blocks, 0);
  return { properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(headerText) }, footers: { default: mkFooter() }, children: [heading(1, md.meta.title), ...renderBlocks(md.blocks)] };
}
const apOrder = [["ap_c.md", "Apêndice C"], ["ap_d.md", "Apêndice D"], ["ap_e.md", "Apêndice E"], ["ap_f.md", "Apêndice F"], ["ap_g.md", "Apêndice G"], ["ap_h.md", "Apêndice H"], ["ap_i.md", "Apêndice I"]];

// A: gabarito comentado (MC) ; B: discursivas
{
  const kids = [heading(1, "Apêndice A — Gabaritos comentados")];
  kids.push(P("Para cada questão de múltipla escolha aparecem a alternativa correta e um comentário. Em cada capítulo, as respostas corretas estão distribuídas igualmente entre as letras A, B, C, D e E (seis de cada)."));
  for (const c of chapters) {
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: true }));
    const bd = { style: BorderStyle.SINGLE, size: 4, color: C.mid };
    const widths = [800, 900, CW - 1700];
    const mk = (cells, head) => new TableRow({ tableHeader: head, cantSplit: true, children: cells.map((t, j) => new TableCell({ width: { size: widths[j], type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: head ? C.ind : "FFFFFF", color: "auto" }, margins: { top: 40, bottom: 40, left: 90, right: 90 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, children: [new Paragraph({ alignment: j < 2 ? AlignmentType.CENTER : AlignmentType.LEFT, spacing: { after: 0, line: 240 }, children: runs(t, { size: 17, ...(head ? { bold: true, color: "FFFFFF" } : j === 1 ? { bold: true, color: C.ind } : {}) }) })] })) });
    kids.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [mk(["Questão", "Resposta", "Comentário"], true), ...c.ex.mc.map((q, i) => mk([String(i + 1), q.ans.toUpperCase(), q.why], false))] }));
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice A — Gabaritos") }, footers: { default: mkFooter() }, children: kids });
}
{
  const kids = [heading(1, "Apêndice B — Respostas das questões discursivas")];
  kids.push(P("Resoluções passo a passo. Nas questões de programação, o código é uma solução de referência e a avaliação deve considerar: (i) correção do resultado, (ii) clareza e organização, (iii) verificação do próprio código (por exemplo, por teste numérico) e (iv) interpretação dos resultados."));
  for (const c of chapters) {
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: true }));
    c.ex.dc.forEach((q, i) => {
      kids.push(new Paragraph({ keepNext: true, spacing: { before: 140, after: 50 }, children: [new TextRun({ text: `Questão ${i + 1}.`, bold: true, color: C.ind, size: 20 })] }));
      // resposta pode ter linhas, código ``` e fórmulas $$
      const parsed = parseMd(q.ans);
      kids.push(...renderBlocks(parsed.blocks.map((b) => (b.t === "p" ? { ...b } : b))).map((x) => x));
    });
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice B — Discursivas") }, footers: { default: mkFooter() }, children: kids });
}
for (const [f, h] of apOrder) if (fs.existsSync(path.join(CAP, f))) contentSections.push(apendiceMd(f, h));

// J: soluções práticas; K: projetos
{
  const kids = [heading(1, "Apêndice J — Soluções dos exercícios práticos")];
  kids.push(P("Cada solução traz o código e a **saída obtida na execução** feita na elaboração do livro (versões em requirements.txt). Há várias formas corretas de resolver; pequenas diferenças numéricas podem surgir com outras versões ou sistemas."));
  for (const c of chapters) {
    if (!c.pr) continue;
    const exs = c.pr.items.filter((x) => x.kind === "EX"); if (!exs.length) continue;
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: true }));
    for (const it of exs) {
      kids.push(heading(3, `Exercício prático ${c.n}.${it.num} — ${it.title}`));
      kids.push(...codeBlock({ lang: "python", code: codeOf(it) }));
      const sd = saidaOf(c.n, it); if (sd) kids.push(...codeBlock({ lang: "saída", code: sd }));
      if (it.note.length) kids.push(callout({ kind: "exemplo", title: "O que observar", body: it.note }), spacer(120));
    }
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice J — Soluções práticas") }, footers: { default: mkFooter() }, children: kids });
}
{
  const kids = [heading(1, "Apêndice K — Projetos: soluções de referência")];
  kids.push(P("Soluções de referência (caminhos possíveis, não os únicos). Os critérios de avaliação precedem o código."));
  const lista = chapters.filter((c) => c.pr).map((c) => ({ n: c.n, t: c.meta.title.split(":")[0], pr: c.pr }));
  if (integrador) lista.push({ n: 0, t: "Projeto integrador final", pr: integrador });
  for (const c of lista) {
    const proj = c.pr.items.find((x) => x.kind === "PROJ"); if (!proj) continue;
    kids.push(heading(2, c.n ? `Capítulo ${c.n} — ${c.t}` : c.t, { pageBreak: true }));
    kids.push(heading(3, proj.title));
    if (proj.rubrica.filter((l) => l.trim()).length) { kids.push(new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: "Critérios de avaliação", bold: true, size: 22, color: C.indD })] })); kids.push(...renderBlocks(parseMd(proj.rubrica.join("\n")).blocks)); }
    kids.push(...codeBlock({ lang: "python", code: codeOf(proj) }));
    const key = c.n === 0 ? "final" : c.n; const out = PR_OUT[`${key}-PROJ0`];
    if (out !== undefined) { let L = out.replace(/\s+$/, "").split("\n").map((l) => (l.length > 96 ? l.slice(0, 95) + "…" : l)); if (L.length > 40) L = L.slice(0, 40).concat(["… (saída truncada)"]); kids.push(...codeBlock({ lang: "saída", code: L })); }
    if (proj.note.length) kids.push(callout({ kind: "exemplo", title: "Discussão dos resultados", body: proj.note }), spacer(120));
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice K — Projetos") }, footers: { default: mkFooter() }, children: kids });
}
// Referências
{
  const kids = [heading(1, "Referências")];
  kids.push(P("Fontes citadas ao longo do texto. Nota de verificação: os dados bibliográficos foram registrados a partir do conhecimento do autor e NÃO foram conferidos em bases externas nesta elaboração (a rede do ambiente não permitiu); confira-os antes de citar. Os textos completos dos artigos originais de Fahlman (1988), Jacobs (1988), Riedmiller e Braun (1993), Igel e Hüsken (2000, 2003), Minai e Williams (1990) e Hagan e Menhaj (1994) não puderam ser consultados na elaboração deste livro — ver Apresentação.", { run: { color: C.gry } }));
  const lines = readTxt("referencias.txt").trim().split("\n").filter(Boolean).sort((a, b) => a.localeCompare(b, "pt"));
  for (const ln of lines) kids.push(new Paragraph({ keepLines: true, spacing: { after: 90, line: 264 }, indent: { left: 420, hanging: 420 }, children: runs(ln, { size: 19 }) }));
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Referências") }, footers: { default: mkFooter() }, children: kids });
}

// ------------------------------------------------------------------ sumário (campo TOC com cache verificado)
function tocParagraphs() {
  const out = [];
  out.push(new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 6 } }, children: [new TextRun({ text: "Sumário", bold: true, size: 44, color: C.indD })] }));
  headings.forEach((h, idx) => {
    const pg = PAGES[h.id] !== undefined ? String(PAGES[h.id]) : "000";
    out.push(new Paragraph({ style: "TOC" + h.level, keepLines: true, children: [...(idx === 0 ? [new TextRun({ text: "⟦TBEGIN⟧", size: 2 })] : []), new TextRun({ text: h.text }), new TextRun({ children: ["\t", pg] }), ...(idx === headings.length - 1 ? [new TextRun({ text: "⟦TEND⟧", size: 2 })] : [])] }));
  });
  return out;
}
const tocSection = { properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Sumário") }, footers: { default: mkFooter() }, children: tocParagraphs() };

const tocStyle = (lvl, ind, bold, size, color, before) => ({ id: "TOC" + lvl, name: "toc " + lvl, basedOn: "Normal", next: "Normal", run: { bold, size, color }, paragraph: { spacing: { before, after: 20, line: 250 }, indent: { left: ind, right: 650 }, tabStops: [{ type: TabStopType.RIGHT, position: CW, leader: LeaderType.DOT }] } });
const doc = new Document({
  creator: AUTHOR, title: `${TITLE} — ${SUBTITLE}`, description: "Livro didático sobre backpropagation e redes neurais", language: "pt-BR",
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: C.ink } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 44, bold: true, color: C.indD, font: FONT }, paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 40, bold: true, color: C.ind, font: FONT }, paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 1, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: C.amb, space: 6 } } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: C.indD, font: FONT }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, color: "4338CA", font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 3 } },
      tocStyle(1, 0, true, 22, C.indD, 160), tocStyle(2, 0, true, 21, C.ink, 100), tocStyle(3, 300, false, 19, C.ink, 0), tocStyle(4, 700, false, 18, C.gry, 0),
    ],
  },
  numbering: { config: numberingConfigs },
  sections: [...coverSections, tocSection, ...contentSections],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  fs.writeFileSync(path.join(__dirname, "headings.json"), JSON.stringify(headings, null, 1));
  fs.writeFileSync(path.join(__dirname, "equations.json"), JSON.stringify(equations, null, 1));
  console.log("ok", OUT, headings.length, "títulos,", equations.length, "equações,", numCounter, "listas numeradas");
});
