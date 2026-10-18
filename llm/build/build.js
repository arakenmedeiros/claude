// Monta o livro "Grandes Modelos de Linguagem" em DOCX a partir de cap/*.md, ex*.txt, pr/*.txt etc.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, HeadingLevel, LevelFormat, TabStopType, LeaderType, Header, Footer, PageNumber, SectionType,
  VerticalAlign, HorizontalPositionRelativeFrom, VerticalPositionRelativeFrom, TextWrappingType,
} = D;

const ROOT = path.join(__dirname, "..");
const CAP = path.join(ROOT, "cap");
const FIGS = path.join(ROOT, "figs");
const OUT = process.argv[2] || path.join(ROOT, "livro.docx");
const rj = (f, d) => (fs.existsSync(path.join(__dirname, f)) ? JSON.parse(fs.readFileSync(path.join(__dirname, f), "utf8")) : d);
const PAGES = rj("pages.json", {});
const PR_OUT = rj("pr_out.json", {});
const CODE_OUT = rj("code_out.json", {});
const EX_OK = rj("ex_ok.json", {});
const NCH = parseInt(process.env.NCH || "18");

const FONT = "Calibri", MONO = "Consolas";
const C = { ind: "4F46E5", indD: "312E81", cya: "06B6D4", grn: "10B981", amb: "F59E0B", red: "EF4444", vio: "8B5CF6", pnk: "EC4899", ink: "1F2937", gry: "6B7280", light: "F3F4F6", mid: "E5E7EB" };
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134;
const CW = PAGE_W - 2 * MARGIN;

const TITLE = "Grandes Modelos de Linguagem";
const SUBTITLE = "Dos fundamentos aos sistemas: tokens, Transformers, RAG, ajuste fino e uso responsável";
const AUTHOR = "Araken de Medeiros Santos";
const YEAR = "2026";

// ---------------------------------------------------------------- helpers
const readTxt = (f) => fs.readFileSync(path.join(CAP, f), "utf8").replace(/\r/g, "");
const exists = (f) => fs.existsSync(path.join(CAP, f));
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const sha = (t) => crypto.createHash("sha1").update(t).digest("hex").slice(0, 12);

// TeX-lite -> unicode para matemática em linha
const GREEK = { alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", theta: "θ", lambda: "λ", mu: "μ", pi: "π", rho: "ρ", sigma: "σ", tau: "τ", phi: "φ", psi: "ψ", omega: "ω", eta: "η", kappa: "κ", nu: "ν", xi: "ξ", zeta: "ζ", chi: "χ", Delta: "Δ", Sigma: "Σ", Theta: "Θ", Omega: "Ω", Lambda: "Λ", Pi: "Π", Phi: "Φ", Psi: "Ψ", Gamma: "Γ" };
const SYM = { cdot: "·", times: "×", sum: "∑", prod: "∏", sqrt: "√", top: "ᵀ", in: "∈", le: "≤", leq: "≤", ge: "≥", geq: "≥", approx: "≈", to: "→", rightarrow: "→", infty: "∞", nabla: "∇", partial: "∂", neq: "≠", ne: "≠", log: "log", ln: "ln", exp: "exp", max: "max", min: "min", argmax: "argmax", argmin: "argmin", pm: "±", ldots: "…", dots: "…", propto: "∝", mid: "|", cup: "∪", cap: "∩", forall: "∀", exists: "∃", sim: "∼", odot: "⊙", ell: "ℓ", mathcal: "", mathbf: "", mathrm: "", text: "", hat: "^", bar: "‾", quad: "  ", ",": " ", ";": " ", " ": " ", "{": "{", "}": "}", "|": "‖", lVert: "‖", rVert: "‖", langle: "⟨", rangle: "⟩", lfloor: "⌊", rfloor: "⌋", Rightarrow: "⇒", leftarrow: "←", mathbb: "" };
function texToUni(s) {
  s = s.replace(/\\([A-Za-z]+|[,; {}|])/g, (m, n) => (n in GREEK ? GREEK[n] : n in SYM ? SYM[n] : n));
  return s;
}
function mathRuns(tex, base = {}) {
  tex = tex.replace(/\{,\}/g, ",").replace(/\\(?:text|mathrm|mathbf|mathcal|mathbb|operatorname)\{([^}]*)\}/g, "$1").replace(/\\(?:left|right)/g, "").replace(/\\mid/g, "|");
  const s = texToUni(tex); const out = []; const b = { font: "Cambria Math", ...base };
  const clean = (x) => x.replace(/[{}]/g, "");
  const re = /_\{([^}]*)\}|_(\S)|\^\{([^}]*)\}|\^(\S)/g; let last = 0, m;
  while ((m = re.exec(s)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: clean(s.slice(last, m.index)), ...b, italics: true }));
    const sub = m[1] !== undefined || m[2] !== undefined;
    out.push(new TextRun({ text: clean(m[1] ?? m[2] ?? m[3] ?? m[4]), ...b, subScript: sub, superScript: !sub, italics: true }));
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push(new TextRun({ text: clean(s.slice(last)), ...b, italics: true }));
  return out;
}

// inline: **negrito**, *itálico*, `código`, $matemática$
function runs(text, base = {}) {
  const out = [];
  const re = /(\$[^$\n]+\$|\*\*(?:[^*]|\*(?!\*)[^*]+\*)+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("$")) out.push(...mathRuns(t.slice(1, -1), { ...base, color: base.color || C.indD }));
    else if (t.startsWith("**")) out.push(...runs(t.slice(2, -2), { ...base, bold: true }));
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

// ---------------------------------------------------------------- md parser
function parseMd(text) {
  const lines = text.split("\n");
  const blocks = []; const meta = { obj: [], refs: [] };
  let i = 0; let para = [];
  const flush = () => { if (para.length) { blocks.push({ t: "p", text: para.join(" ") }); para = []; } };
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.startsWith("@part:")) { meta.part = ln.slice(6).trim(); i++; continue; }
    if (ln.startsWith("@banner:")) { meta.banner = parseInt(ln.slice(8)); i++; continue; }
    if (ln.startsWith("@obj:")) { meta.obj.push(ln.slice(5).trim()); i++; continue; }
    if (ln.startsWith("@ref:")) { meta.refs.push(...ln.slice(5).trim().split(/\s+/)); i++; continue; }
    if (ln.startsWith("@@OBJ")) { flush(); blocks.push({ t: "objbox" }); i++; continue; }
    if (/^# /.test(ln)) { flush(); meta.title = ln.slice(2).trim(); i++; continue; }
    if (/^### /.test(ln)) { flush(); blocks.push({ t: "h4", text: ln.slice(4).trim() }); i++; continue; }
    if (/^## /.test(ln)) { flush(); blocks.push({ t: "h3", text: ln.slice(3).trim() }); i++; continue; }
    if (ln.startsWith("```")) { flush(); const lang = ln.slice(3).trim(); const code = []; i++; while (i < lines.length && !lines[i].startsWith("```")) { code.push(lines[i]); i++; } i++; blocks.push({ t: "code", lang, code }); continue; }
    if (ln.startsWith(":::")) { flush(); const m = ln.match(/^:::(\w+)\s*(.*)$/); const body = []; i++; while (i < lines.length && lines[i].trim() !== ":::") { body.push(lines[i]); i++; } i++; blocks.push({ t: "callout", kind: m[1], title: m[2], body }); continue; }
    const fig = ln.match(/^!\[(\w+)\]\((.*)\)\s*$/);
    if (fig) { flush(); blocks.push({ t: "fig", id: fig[1], caption: fig[2] }); i++; continue; }
    if (ln.startsWith("$$")) { flush(); blocks.push({ t: "formula", text: ln.replace(/^\$\$\s*/, "").replace(/\s*\$\$$/, "") }); i++; continue; }
    if (ln.startsWith("|")) { flush(); const rows = []; while (i < lines.length && lines[i].startsWith("|")) { rows.push(lines[i]); i++; } blocks.push({ t: "table", rows: rows.filter((r) => !/^\|\s*:?-{2,}/.test(r)).map((r) => r.replace(/^\||\|\s*$/g, "").split("|").map((c) => c.trim())) }); continue; }
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
const headings = [];
let hid = 0;
function heading(level, text, opts = {}) {
  text = text.replace(/\*+/g, "").replace(/`/g, "").replace(/\$/g, "");
  const id = "h" + (++hid);
  headings.push({ level, text, id });
  const lv = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3, 4: HeadingLevel.HEADING_4 }[level];
  return new Paragraph({ heading: lv, pageBreakBefore: !!opts.pageBreak, keepNext: true, children: [new TextRun({ text })] });
}
function subtitleLike(text) { // título sem entrar no sumário (itens de exercício)
  return new Paragraph({ keepNext: true, spacing: { before: 240, after: 100 }, children: [new TextRun({ text, bold: true, size: 25, color: "4338CA" })] });
}
function P(text, o = {}) {
  return new Paragraph({ spacing: { after: 120, line: 300 }, alignment: o.align || AlignmentType.LEFT, keepNext: o.keepNext, indent: o.indent, children: runs(text, { size: 21, ...(o.run || {}) }) });
}
function spacer(after = 120) { return new Paragraph({ spacing: { after, line: 120 }, children: [] }); }

function calloutParas(body) {
  const out = []; let para = [];
  const flush = () => { if (para.length) { out.push(new Paragraph({ spacing: { after: 80, line: 280 }, children: runs(para.join(" "), { size: 20 }) })); para = []; } };
  let i = 0;
  while (i < body.length) {
    const ln = body[i];
    if (/^- /.test(ln)) { flush(); while (i < body.length && /^- /.test(body[i])) { out.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(body[i].slice(2), { size: 20 }) })); i++; } continue; }
    if (/^\d+\. /.test(ln)) { flush(); while (i < body.length && /^\d+\. /.test(body[i])) { out.push(new Paragraph({ spacing: { after: 60, line: 276 }, indent: { left: 420, hanging: 320 }, children: [new TextRun({ text: body[i].match(/^\d+\./)[0] + " ", bold: true, color: C.ind, size: 20 }), ...runs(body[i].replace(/^\d+\.\s+/, ""), { size: 20 })] })); i++; } continue; }
    if (ln.trim() === "") { flush(); i++; continue; }
    para.push(ln.trim()); i++;
  }
  flush();
  return out;
}
const CALL = {
  dica: { c: C.grn, f: "ECFDF5", label: "DICA" }, atencao: { c: C.amb, f: "FFFBEB", label: "ATENÇÃO" }, cotidiano: { c: C.cya, f: "ECFEFF", label: "NO SEU DIA A DIA" },
  saibamais: { c: C.vio, f: "F5F3FF", label: "SAIBA MAIS" }, exemplo: { c: C.ind, f: "EEF2FF", label: "EXEMPLO" }, memorize: { c: C.pnk, f: "FDF2F8", label: "MEMORIZE" },
  intuicao: { c: C.cya, f: "ECFEFF", label: "INTUIÇÃO" }, passo: { c: C.ind, f: "EEF2FF", label: "PASSO A PASSO" }, pratica: { c: C.grn, f: "ECFDF5", label: "NA PRÁTICA" },
  cuidado: { c: C.red, f: "FEF2F2", label: "CUIDADO COM ESTA INTERPRETAÇÃO" }, lembrar: { c: C.pnk, f: "FDF2F8", label: "O QUE VALE A PENA LEMBRAR" },
  fonte: { c: C.gry, f: "F9FAFB", label: "SOBRE AS FONTES" }, situacao: { c: C.amb, f: "FFFBEB", label: "SITUAÇÃO" },
};
function callout(b) {
  const s = CALL[b.kind] || CALL.dica;
  const norm = (t) => t.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const dup = !b.title || norm(b.title).startsWith(norm(s.label.split(" ")[0]));
  const kids = [new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [...(dup && !b.title ? [new TextRun({ text: s.label, bold: true, size: 20, color: s.c, characterSpacing: 10 })] : [new TextRun({ text: s.label + (b.title ? "  " : ""), bold: true, size: 16, color: s.c, characterSpacing: 20 }), new TextRun({ text: b.title || "", bold: true, size: 22, color: s.c })])] }), ...calloutParas(b.body)];
  const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: b.body.length < 14, children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: s.f, color: "auto" }, margins: { top: 100, bottom: 60, left: 200, right: 160 }, borders: { top: nb, bottom: nb, right: nb, left: { style: BorderStyle.SINGLE, size: 36, color: s.c } }, children: kids })] })] });
}

const KW = new Set("def class for in import from as if elif else return with while lambda and or not True False None try except pass break continue yield assert raise is global".split(" "));
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
const CODE_LABEL = {
  python: ["PYTHON · EXECUTADO", C.ind], "python+": ["PYTHON · EXECUTADO (CONTINUAÇÃO)", C.ind], "python-ni": ["PYTHON · NÃO EXECUTADO NESTE AMBIENTE (ILUSTRATIVO)", C.amb],
  pseudo: ["PSEUDOCÓDIGO", C.vio], ilustrativo: ["EXEMPLO ILUSTRATIVO", C.amb], bash: ["TERMINAL", C.gry], saida: ["SAÍDA OBTIDA NA EXECUÇÃO", C.grn], "saída": ["SAÍDA OBTIDA NA EXECUÇÃO", C.grn], texto: ["TEXTO / PROMPT", C.cya], json: ["JSON", C.gry], yaml: ["YAML", C.gry],
};
function codeBlock(b) {
  const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const hl = b.lang && b.lang.startsWith("python");
  const kids = b.code.map((ln) => new Paragraph({ spacing: { after: 0, line: 240 }, children: hl ? codeRuns(ln) : [new TextRun({ text: ln || " ", font: MONO, size: 15, color: C.ink })] }));
  const [label, col] = CODE_LABEL[b.lang] || [(b.lang || "CÓDIGO").toUpperCase(), C.ind];
  const fill = b.lang === "saida" || b.lang === "saída" ? "F0FDF4" : "F8FAFC";
  return [new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill, color: "auto" }, margins: { top: 100, bottom: 100, left: 200, right: 120 }, borders: { top: { style: BorderStyle.SINGLE, size: 4, color: C.mid }, bottom: { style: BorderStyle.SINGLE, size: 4, color: C.mid }, right: { style: BorderStyle.SINGLE, size: 4, color: C.mid }, left: { style: BorderStyle.SINGLE, size: 24, color: col } }, children: [new Paragraph({ spacing: { after: 60 }, keepNext: true, children: [new TextRun({ text: label, bold: true, size: 14, color: col, characterSpacing: 30 })] }), ...kids] })] })] }), spacer(140)];
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

let curCh = 0, figN = 0, eqN = 0, codeN = 0;
function figure(b) {
  const file = path.join(FIGS, b.id + ".png");
  if (!fs.existsSync(file)) throw new Error("figura ausente: " + b.id);
  const { w, h } = pngSize(file);
  let dw = Math.min(600, (w / 200) * 96), dh = (dw * h) / w;
  if (dh > 400) { dh = 400; dw = (dh * w) / h; }
  figN++;
  let cap = b.caption; let kind = "";
  if (cap.startsWith("[C]")) { kind = "Esquema conceitual. "; cap = cap.slice(3).trim(); }
  else if (cap.startsWith("[D")) { const m = cap.match(/^\[D(?::([^\]]*))?\]\s*/); kind = "Gráfico baseado em dados" + (m[1] ? ` (${m[1]})` : "") + ". "; cap = cap.slice(m[0].length); }
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(dw), height: Math.round(dh) }, altText: { title: b.id, description: cap, name: b.id } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: `Figura ${curCh}.${figN} — `, bold: true, size: 17, color: C.gry }), new TextRun({ text: kind, bold: true, italics: true, size: 17, color: C.gry }), ...runs(cap, { size: 17, italics: true, color: C.gry })] }),
  ];
}
function formulaImg(b) {
  const file = path.join(FIGS, "eq", sha(b.text) + ".png");
  if (!fs.existsSync(file)) throw new Error("equação não renderizada (rode eqs.js): " + b.text);
  const { w, h } = pngSize(file);
  let dw = (w / 3) * 0.78, dh = (h / 3) * 0.78;
  if (dw > 520) { dh = (dh * 520) / dw; dw = 520; }
  eqN++;
  return new Paragraph({ spacing: { before: 100, after: 140 }, keepLines: true, tabStops: [{ type: TabStopType.CENTER, position: CW / 2 }, { type: TabStopType.RIGHT, position: CW }], children: [new TextRun({ text: "\t" }), new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(dw), height: Math.round(dh) }, altText: { title: "equação", description: b.text, name: "eq" + sha(b.text) } }), new TextRun({ text: `\t(${curCh}.${eqN})`, size: 20, color: C.gry })] });
}
function banner(n) {
  const file = path.join(FIGS, `banner${n}.png`); const { w, h } = pngSize(file);
  const dw = 642, dh = Math.round((642 * h) / w);
  return new Paragraph({ spacing: { after: 160 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: dw, height: dh }, altText: { title: "banner", description: "Faixa decorativa do capítulo " + n, name: "banner" + n } })] });
}

function objbox(obj) {
  const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: "EEF2FF", color: "auto" }, margins: { top: 120, bottom: 80, left: 220, right: 180 }, borders: { top: nb, bottom: nb, right: nb, left: { style: BorderStyle.SINGLE, size: 36, color: C.ind } }, children: [new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Ao final deste capítulo você será capaz de:", bold: true, size: 21 })] }), ...obj.map((o) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 50, line: 276 }, children: runs(o, { size: 20 }) }))] })] })] });
}

function renderBlocks(blocks, ctx = {}) {
  const out = []; let pycode = [];
  for (const b of blocks) {
    switch (b.t) {
      case "h3": out.push(heading(3, b.text)); break;
      case "h4": out.push(heading(4, b.text)); break;
      case "p": out.push(P(b.text)); break;
      case "objbox": out.push(objbox(ctx.obj || []), spacer(160)); break;
      case "quote": out.push(new Paragraph({ spacing: { after: 120 }, indent: { left: 400 }, border: { left: { style: BorderStyle.SINGLE, size: 18, color: C.cya, space: 8 } }, children: runs(b.text, { size: 21, italics: true }) })); break;
      case "ul": b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 70, line: 288 }, children: runs(it, { size: 21 }) }))); out.push(spacer(60)); break;
      case "ol": { const ref = newNumRef(); b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 70, line: 288 }, children: runs(it, { size: 21 }) }))); out.push(spacer(60)); break; }
      case "callout": out.push(callout(b)); out.push(spacer(140)); break;
      case "code": {
        out.push(...codeBlock(b));
        if (ctx.codeKeys && (b.lang === "python" || b.lang === "python+")) {
          codeN++; const key = `${curCh}-${codeN}`; const o = CODE_OUT[key];
          if (o && o.trim()) { let L = o.replace(/\s+$/, "").split("\n").map((l) => (l.length > 108 ? l.slice(0, 107) + "…" : l)); if (L.length > 40) L = L.slice(0, 40).concat(["… (saída truncada)"]); out.push(...codeBlock({ lang: "saida", code: L })); }
        }
        break;
      }
      case "formula": out.push(formulaImg(b)); break;
      case "table": out.push(...table(b)); break;
      case "fig": out.push(...figure(b)); break;
    }
  }
  return out;
}

// ---------------------------------------------------------------- questões
function balancedLetters(n, seed) {
  const L = "ABCDE"; const per = n / 5;
  for (let attempt = 0; attempt < 5000; attempt++) {
    const rnd = mulberry32(seed * 7919 + attempt);
    const arr = []; for (const l of L) for (let k = 0; k < per; k++) arr.push(l);
    for (let k = arr.length - 1; k > 0; k--) { const j = Math.floor(rnd() * (k + 1)); [arr[k], arr[j]] = [arr[j], arr[k]]; }
    let ok = true;
    for (let k = 2; k < n; k++) if (arr[k] === arr[k - 1] && arr[k] === arr[k - 2]) ok = false;
    for (let k = 1; k < n; k++) if (arr[k] === arr[k - 1]) { /* repetições simples permitidas */ }
    // evita padrão cíclico óbvio
    let cyc = 0; for (let k = 1; k < n; k++) if (L.indexOf(arr[k]) === (L.indexOf(arr[k - 1]) + 1) % 5) cyc++;
    if (cyc > n * 0.3) ok = false;
    if (ok) return arr;
  }
  throw new Error("não consegui balancear gabarito");
}
function parseEx(text, seed, chn) {
  const lines = text.replace(/\r/g, "").split("\n");
  const mc = [], dc = [];
  let i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.startsWith("MC:")) {
      const q = { stem: ln.slice(3).trim(), correct: null, wrong: [], why: "" }; i++;
      while (i < lines.length && lines[i].trim() !== "") {
        const l = lines[i];
        if (l.startsWith("= ")) q.correct = l.slice(2).trim();
        else if (l.startsWith("- ")) q.wrong.push(l.slice(2).trim());
        else if (l.startsWith("J:")) q.why = l.slice(2).trim();
        else throw new Error(`ex${chn}: linha inesperada: ${l}`);
        i++;
      }
      if (q.correct === null || q.wrong.length !== 4 || !q.why) throw new Error(`ex${chn}: MC malformada: ${q.stem.slice(0, 60)}`);
      mc.push(q);
    } else if (ln.startsWith("DISC")) {
      const m = ln.match(/^DISC\[([MPACD])\]:\s*(.*)$/); if (!m) throw new Error(`ex${chn}: DISC sem tipo: ${ln.slice(0, 50)}`);
      const q = { type: m[1], stem: m[2], ans: [], code: [], crit: [] }; i++;
      let mode = "ans";
      while (i < lines.length && !(lines[i].trim() === "" && mode !== "code")) {
        const l = lines[i];
        if (mode === "code") { if (l.trim() === "ENDCODE") mode = "ans"; else q.code.push(l); }
        else if (l.startsWith("R:")) q.ans.push(l.slice(2).trim());
        else if (l.startsWith("CODE:")) mode = "code";
        else if (l.startsWith("CRIT:")) q.crit.push(l.slice(5).trim());
        else q.ans.push(l.trim());
        i++;
      }
      if (!q.ans.length && q.code.length) q.ans.push("Código de referência:");
      if (!q.ans.length) throw new Error(`ex${chn}: DISC sem resposta: ${q.stem.slice(0, 50)}`);
      dc.push(q);
    }
    i++;
  }
  if (mc.length !== 30) throw new Error(`ex${chn}: ${mc.length} MC (esperado 30)`);
  if (dc.length !== 20) throw new Error(`ex${chn}: ${dc.length} DISC (esperado 20)`);
  const targets = balancedLetters(30, seed);
  mc.forEach((q, k) => {
    const pos = "ABCDE".indexOf(targets[k]); const opts = []; let w = 0;
    for (let p = 0; p < 5; p++) opts.push(p === pos ? q.correct : q.wrong[w++]);
    q.opts = opts; q.ans = targets[k];
  });
  return { mc, dc };
}
const TYPE_NAME = { M: "Matemática", P: "Programação", A: "Análise", C: "Compreensão", D: "Decisão de projeto" };
function exercises(ex) {
  const out = [];
  out.push(heading(3, "Exercícios de múltipla escolha", { pageBreak: true }));
  out.push(P("Escolha **uma** alternativa em cada questão. O gabarito comentado está no Apêndice A.", { run: { color: C.gry } }));
  ex.mc.forEach((q, i) => {
    out.push(new Paragraph({ keepNext: true, spacing: { before: 140, after: 60, line: 276 }, children: [new TextRun({ text: `Questão ${i + 1}. `, bold: true, color: C.ind, size: 21 }), ...runs(q.stem, { size: 21 })] }));
    q.opts.forEach((o, k) => out.push(new Paragraph({ keepNext: k < 4, keepLines: true, spacing: { after: 30, line: 264 }, indent: { left: 500, hanging: 300 }, children: [new TextRun({ text: "ABCDE"[k] + ")\t", bold: true, color: C.gry, size: 20 }), ...runs(o, { size: 20 })], tabStops: [{ type: TabStopType.LEFT, position: 500 }] })));
  });
  out.push(heading(3, "Exercícios discursivos", { pageBreak: true }));
  out.push(P("Responda por escrito. Cada questão indica o tipo entre colchetes: [Compreensão], [Matemática], [Análise], [Programação] ou [Decisão]. As soluções das questões matemáticas estão no Apêndice B; as respostas orientadoras e os critérios, no Apêndice C.", { run: { color: C.gry } }));
  ex.dc.forEach((q, i) => out.push(new Paragraph({ keepLines: true, spacing: { before: 120, after: 80, line: 288 }, indent: { left: 500, hanging: 500 }, tabStops: [{ type: TabStopType.LEFT, position: 500 }], children: [new TextRun({ text: `${i + 1}.\t`, bold: true, color: C.ind, size: 21 }), new TextRun({ text: `[${TYPE_NAME[q.type]}] `, bold: true, size: 17, color: C.gry }), ...runs(q.stem, { size: 21 })] })));
  return out;
}

// ---------------------------------------------------------------- práticos e projetos
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
  const key = `${ch}-${it.kind}${it.num}`; const out = PR_OUT[key]; if (out === undefined) return null;
  let L = out.replace(/\s+$/, "").split("\n").map((l) => (l.length > 108 ? l.slice(0, 107) + "…" : l));
  if (L.length > 44) L = L.slice(0, 44).concat(["… (saída truncada)"]);
  return L;
}
function praticos(c) {
  const pr = c.pr; if (!pr) return [];
  const exs = pr.items.filter((x) => x.kind === "EX"); const proj = pr.items.find((x) => x.kind === "PROJ");
  const out = [];
  if (exs.length) {
    out.push(heading(3, "Exercícios práticos em Python", { pageBreak: true }));
    out.push(P("Resolva no seu computador ou no Google Colab. Os exercícios usam NumPy, scikit-learn e os dados didáticos do pacote `llmdidatico` (Apêndice H), que funcionam em CPU e sem internet. Tente resolver **antes** de consultar a solução, no Apêndice D. As saídas apresentadas lá foram obtidas por execução real, com as versões do Apêndice H.", { run: { color: C.gry } }));
    for (const it of exs) {
      out.push(subtitleLike(`Exercício ${c.n}.${it.num} — ${it.title}`));
      out.push(new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: `NÍVEL: ${it.level.toUpperCase()}`, bold: true, size: 15, color: C.ind, characterSpacing: 30 })] }));
      out.push(...renderBlocks(parseMd(it.stmt.join("\n")).blocks));
    }
  }
  if (pr.alem.length) {
    out.push(callout({ kind: "saibamais", title: "Para ir além: conjuntos de dados públicos", body: [...pr.alem, "", "Estes conjuntos **não** foram usados nem testados neste livro (o ambiente de elaboração não tinha acesso a eles). Confira nos sites a descrição, a licença e o formato antes de usá-los."] }));
    out.push(spacer(140));
  }
  if (proj) {
    out.push(heading(3, c.n === 18 ? "Projetos integradores finais" : "Projeto do capítulo", { pageBreak: true }));
    out.push(subtitleLike(proj.title));
    out.push(...renderBlocks(parseMd(proj.stmt.join("\n")).blocks));
    out.push(P(`A solução de referência está no Apêndice E.`, { run: { color: C.gry, italics: true } }));
  }
  return out;
}

// ---------------------------------------------------------------- referências
const REFS = {};
for (const ln of readTxt("referencias.txt").split("\n")) {
  if (!ln.trim() || ln.startsWith("#")) continue;
  const m = ln.match(/^(\w+)\s*\|\s*([VCP])\s*\|\s*(.*)$/); if (!m) throw new Error("referência malformada: " + ln);
  REFS[m[1]] = { st: m[2], text: m[3] };
}
const refSortKey = (t) => t.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
function refPara(r) {
  return new Paragraph({ keepLines: true, spacing: { after: 80, line: 264 }, indent: { left: 420, hanging: 420 }, children: runs(r.text + (r.st === "V" ? "" : " †"), { size: 19 }) });
}
function chapterRefs(c) {
  const out = [heading(3, "Referências do capítulo")];
  const list = [...new Set(c.meta.refs)].map((k) => { if (!REFS[k]) throw new Error(`cap ${c.n}: referência desconhecida: ${k}`); return REFS[k]; }).sort((a, b) => refSortKey(a.text).localeCompare(refSortKey(b.text), "pt"));
  list.forEach((r) => out.push(refPara(r)));
  return out;
}

// ---------------------------------------------------------------- cabeçalho/rodapé
function mkHeader(text) {
  return new Header({ children: [new Paragraph({ alignment: AlignmentType.LEFT, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.ind, space: 4 } }, tabStops: [{ type: TabStopType.RIGHT, position: CW }], children: [new TextRun({ text: TITLE, size: 17, color: C.gry }), new TextRun({ text: "\t" + text, size: 17, color: C.ind, bold: true })] })] });
}
function mkFooter() { return new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: C.gry })] })] }); }
const secProps = (extra = {}) => ({ page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1300, bottom: 1200, left: MARGIN, right: MARGIN, header: 560, footer: 500 } }, ...extra });

// ---------------------------------------------------------------- montagem
const sections = [];
{
  const file = path.join(FIGS, "capa.png");
  sections.push({ properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 0, bottom: 0, left: 0, right: 0, header: 0, footer: 0 } } },
    children: [new Paragraph({ spacing: { after: 0 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: 794, height: 1123 }, floating: { horizontalPosition: { relative: HorizontalPositionRelativeFrom.PAGE, offset: 0 }, verticalPosition: { relative: VerticalPositionRelativeFrom.PAGE, offset: 0 }, wrap: { type: TextWrappingType.NONE }, behindDocument: true }, altText: { title: "Capa", description: `Capa do livro ${TITLE}, de ${AUTHOR}`, name: "capa" } })] })] });
}
sections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }),
  children: [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 3000, after: 120 }, children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, size: 52, color: C.indD, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 12 } }, children: [new TextRun({ text: SUBTITLE, italics: true, size: 28, color: C.gry })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 }, children: [new TextRun({ text: AUTHOR, bold: true, size: 34, color: C.ink })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1800 }, children: [new TextRun({ text: YEAR, size: 24, color: C.gry })] }),
  ] });

const chapters = [];
for (let n = 1; n <= NCH; n++) {
  const nn = String(n).padStart(2, "0");
  const md = parseMd(readTxt(`ch${nn}.md`));
  const ex = parseEx(readTxt(`ex${nn}.txt`), 1000 + n, n);
  const prf = path.join(CAP, "pr", `pr${nn}.txt`);
  chapters.push({ n, ...md, ex, pr: fs.existsSync(prf) ? parsePr(readTxt("pr/pr" + nn + ".txt")) : null });
}
const PARTS = {};
for (const ln of readTxt("partes.txt").split("\n")) { if (!ln.trim()) continue; const k = ln.indexOf("|"); PARTS[ln.slice(0, k).trim()] = ln.slice(k + 1).trim(); }
const PART_COL = { "Parte I": C.ind, "Parte II": C.cya, "Parte III": C.grn, "Parte IV": C.amb, "Parte V": C.red };

const contentSections = [];
const addSec = (kids, hdr) => contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(hdr) }, footers: { default: mkFooter() }, children: kids });

// Apresentação
{ const md = parseMd(readTxt("apresentacao.md")); addSec([heading(1, "Apresentação"), ...renderBlocks(md.blocks)], "Apresentação"); }

for (const c of chapters) {
  curCh = c.n; figN = 0; eqN = 0; codeN = 0;
  if (c.meta.part) {
    const pk = c.meta.part.split(" — ")[0]; const col = PART_COL[pk];
    const [pnum] = c.meta.part.split(" — ");
    const list = []; for (let k = c.n - 1; k < chapters.length; k++) { if (k > c.n - 1 && chapters[k].meta.part) break; list.push(chapters[k]); }
    contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), children: [
      new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 3200, after: 100 }, children: [new TextRun({ text: pnum.toUpperCase(), bold: true, size: 40, color: col, characterSpacing: 60 })] }),
      heading(1, c.meta.part),
      new Paragraph({ spacing: { before: 200, after: 300 }, border: { top: { style: BorderStyle.SINGLE, size: 24, color: col, space: 10 } }, children: [new TextRun({ text: PARTS[c.meta.part] || "", size: 26, color: C.ink, italics: true })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Nesta parte:", bold: true, size: 22, color: col })] }),
      ...list.map((x) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text: `Cap. ${x.n} — ${x.meta.title}`, size: 22 })] })),
    ] });
  }
  const kids = [];
  kids.push(banner(c.meta.banner || c.n));
  kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title}`));
  kids.push(...renderBlocks(c.blocks, { obj: c.meta.obj, codeKeys: true }));
  kids.push(...exercises(c.ex));
  kids.push(...praticos(c));
  kids.push(...chapterRefs(c));
  addSec(kids, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`);
}
curCh = 0;

const chShort = (c) => `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`;
// Apêndice A — gabarito MC
{
  const kids = [heading(1, "Apêndice A — Gabarito comentado das questões objetivas")];
  kids.push(P("Para cada capítulo, a tabela traz a alternativa correta e uma justificativa. Em cada capítulo, as 30 respostas se distribuem igualmente entre A, B, C, D e E (seis de cada), em ordem sem padrão previsível. Tente responder antes de consultar."));
  for (const c of chapters) {
    kids.push(heading(2, chShort(c), { pageBreak: true }));
    const cnt = { A: 0, B: 0, C: 0, D: 0, E: 0 }; c.ex.mc.forEach((q) => cnt[q.ans]++);
    kids.push(P(`Distribuição das respostas — A: ${cnt.A}, B: ${cnt.B}, C: ${cnt.C}, D: ${cnt.D}, E: ${cnt.E}.`, { run: { color: C.gry, size: 18 } }));
    const bd = { style: BorderStyle.SINGLE, size: 4, color: C.mid }; const widths = [800, 900, CW - 1700];
    const mk = (cells, head) => new TableRow({ tableHeader: head, cantSplit: true, children: cells.map((t, j) => new TableCell({ width: { size: widths[j], type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: head ? C.ind : "FFFFFF", color: "auto" }, margins: { top: 40, bottom: 40, left: 90, right: 90 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, children: [new Paragraph({ alignment: j < 2 ? AlignmentType.CENTER : AlignmentType.LEFT, spacing: { after: 0, line: 240 }, children: runs(t, { size: 17, ...(head ? { bold: true, color: "FFFFFF" } : j === 1 ? { bold: true, color: C.ind } : {}) }) })] })) });
    kids.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [mk(["Questão", "Resposta", "Justificativa"], true), ...c.ex.mc.map((q, i) => mk([String(i + 1), q.ans, q.why], false))] }));
  }
  addSec(kids, "Apêndice A — Gabarito");
}
// Apêndice B — matemática; C — orientadoras
function answerPara(q, i, c) {
  const ps = [new Paragraph({ keepLines: true, spacing: { before: 100, after: 60, line: 270 }, indent: { left: 500, hanging: 500 }, tabStops: [{ type: TabStopType.LEFT, position: 500 }], children: [new TextRun({ text: `${i + 1}.\t`, bold: true, color: C.ind, size: 20 }), new TextRun({ text: `[${TYPE_NAME[q.type]}] `, bold: true, size: 16, color: C.gry }), ...runs(q.stem, { size: 19, bold: true })] })];
  for (const a of q.ans) ps.push(new Paragraph({ keepLines: true, spacing: { after: 60, line: 270 }, indent: { left: 500 }, children: runs(a, { size: 19 }) }));
  if (q.code.length) ps.push(...codeBlock({ lang: EX_OK[`${c.n}-${i + 1}`] ? "python" : "python-ni", code: q.code }).slice(0, 1));
  if (q.crit.length) { ps.push(new Paragraph({ spacing: { after: 40 }, indent: { left: 500 }, children: [new TextRun({ text: "Critérios de avaliação", bold: true, size: 17, color: C.indD })] })); q.crit.forEach((k) => ps.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 30, line: 260 }, children: runs(k, { size: 18 }) }))); }
  return ps;
}
{
  const kids = [heading(1, "Apêndice B — Soluções das questões matemáticas")];
  kids.push(P("Solução passo a passo de todas as questões discursivas do tipo **Matemática**. A numeração é a da lista de discursivas de cada capítulo."));
  for (const c of chapters) {
    kids.push(heading(2, chShort(c), { pageBreak: true }));
    c.ex.dc.forEach((q, i) => { if (q.type === "M") kids.push(...answerPara(q, i, c)); });
  }
  addSec(kids, "Apêndice B — Soluções matemáticas");
}
{
  const kids = [heading(1, "Apêndice C — Respostas orientadoras, códigos de referência e critérios")];
  kids.push(P("Respostas-modelo das questões discursivas de compreensão, análise, programação e decisão de projeto. A sua resposta pode ser redigida de outra forma, desde que contemple as mesmas ideias. Os códigos de referência marcados como **executado** foram rodados com os testes mostrados; os marcados como *não executado* são esboços para consulta."));
  for (const c of chapters) {
    kids.push(heading(2, chShort(c), { pageBreak: true }));
    c.ex.dc.forEach((q, i) => { if (q.type !== "M") kids.push(...answerPara(q, i, c)); });
  }
  addSec(kids, "Apêndice C — Respostas orientadoras");
}
// D — soluções práticas
{
  const kids = [heading(1, "Apêndice D — Soluções dos exercícios práticos")];
  kids.push(P("Cada solução traz o código, a **saída obtida** por execução real e um comentário sobre o que observar. Há várias maneiras corretas de resolver um exercício: importa que o resultado seja coerente e que você consiga explicá-lo. Pequenas diferenças numéricas podem ocorrer com outras versões das bibliotecas."));
  for (const c of chapters) {
    if (!c.pr) continue;
    const exs = c.pr.items.filter((x) => x.kind === "EX"); if (!exs.length) continue;
    kids.push(heading(2, chShort(c), { pageBreak: true }));
    for (const it of exs) {
      kids.push(subtitleLike(`Exercício ${c.n}.${it.num} — ${it.title}`));
      kids.push(...codeBlock({ lang: "python", code: codeOf(it) }));
      const sd = saidaOf(c.n, it); if (sd) kids.push(...codeBlock({ lang: "saida", code: sd }));
      if (it.note.length) kids.push(callout({ kind: "exemplo", title: "O que observar", body: it.note }), spacer(120));
    }
  }
  addSec(kids, "Apêndice D — Soluções práticas");
}
// E — projetos
{
  const kids = [heading(1, "Apêndice E — Soluções de referência dos projetos")];
  kids.push(P("Estas são **soluções de referência**: caminhos possíveis, e não os únicos. Compare com o seu trabalho, observe as decisões e discuta as diferenças. A avaliação deve considerar também a qualidade da argumentação, e não apenas o código. Os critérios aparecem antes do código."));
  for (const c of chapters) {
    if (!c.pr) continue;
    const projs = c.pr.items.filter((x) => x.kind === "PROJ"); if (!projs.length) continue;
    kids.push(heading(2, chShort(c), { pageBreak: true }));
    for (const proj of projs) {
      kids.push(subtitleLike(proj.title));
      if (proj.rubrica.filter((l) => l.trim()).length) {
        kids.push(new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: "Critérios de avaliação", bold: true, size: 22, color: C.indD })] }));
        kids.push(...renderBlocks(parseMd(proj.rubrica.join("\n")).blocks));
      }
      kids.push(...codeBlock({ lang: "python", code: codeOf(proj) }));
      const sd = saidaOf(c.n, proj); if (sd) kids.push(...codeBlock({ lang: "saida", code: sd }));
      if (proj.note.length) kids.push(callout({ kind: "exemplo", title: "Discussão dos resultados", body: proj.note }), spacer(120));
    }
  }
  addSec(kids, "Apêndice E — Projetos");
}
// F glossário, G símbolos, H..L materiais
{
  const kids = [heading(1, "Apêndice F — Glossário")];
  kids.push(P("Termos técnicos usados no livro, explicados de forma simples."));
  const lines = readTxt("glossario.txt").trim().split("\n").filter(Boolean).sort((a, b) => refSortKey(a).localeCompare(refSortKey(b), "pt"));
  for (const ln of lines) { const k = ln.indexOf(":"); kids.push(new Paragraph({ keepLines: true, spacing: { after: 90, line: 276 }, children: [new TextRun({ text: ln.slice(0, k), bold: true, color: C.indD, size: 20 }), ...runs(" — " + ln.slice(k + 1).trim(), { size: 20 })] })); }
  addSec(kids, "Apêndice F — Glossário");
}
{
  const md = parseMd(readTxt("simbolos.md")); addSec([heading(1, md.meta.title), ...renderBlocks(md.blocks)], "Apêndice G — Símbolos");
}
for (const [f, h] of [["instalacao.md", "Apêndice H — Instalação"], ["prompts.md", "Apêndice I — Prompts"], ["checklists.md", "Apêndice J — Checklists"], ["matriz.md", "Apêndice K — Decisão"], ["metodo.md", "Apêndice L — Método"]]) {
  curCh = 0; const md = parseMd(readTxt(f)); addSec([heading(1, md.meta.title), ...renderBlocks(md.blocks)], h);
}
// Referências gerais
{
  const kids = [heading(1, "Referências")];
  kids.push(P("Lista consolidada das obras citadas. O símbolo † indica itens cujos dados bibliográficos ainda devem ser conferidos na fonte original antes da publicação (ver Apêndice L). Normas: ABNT NBR 6023.", { run: { color: C.gry } }));
  const used = new Set(chapters.flatMap((c) => c.meta.refs));
  const list = [...used].map((k) => REFS[k]).sort((a, b) => refSortKey(a.text).localeCompare(refSortKey(b.text), "pt"));
  list.forEach((r) => kids.push(refPara(r)));
  addSec(kids, "Referências");
}

// ---------------------------------------------------------------- sumário
function tocParagraphs() {
  const out = [];
  out.push(new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 6 } }, children: [new TextRun({ text: "Sumário", bold: true, size: 44, color: C.indD })] }));
  headings.forEach((h, idx) => {
    const pg = PAGES[h.id] !== undefined ? String(PAGES[h.id]) : "000";
    const lvl = h.level; const ind = [0, 0, 300, 700, 1100][lvl];
    const style = { 1: { bold: true, size: 22, color: C.indD, before: 160 }, 2: { bold: true, size: 21, color: C.ink, before: 100 }, 3: { bold: false, size: 19, color: C.ink, before: 0 }, 4: { bold: false, size: 18, color: C.gry, before: 0 } }[lvl];
    const kids = [];
    if (idx === 0) kids.push(new TextRun({ text: "§TOC_BEGIN§" }));
    kids.push(new TextRun({ text: h.text, bold: style.bold, size: style.size, color: style.color }), new TextRun({ children: ["\t", pg], size: style.size, color: style.color, bold: style.bold }));
    if (idx === headings.length - 1) kids.push(new TextRun({ text: "§TOC_END§" }));
    out.push(new Paragraph({ spacing: { before: style.before, after: 20, line: 250 }, indent: { left: ind, right: 650 }, tabStops: [{ type: TabStopType.RIGHT, position: CW, leader: LeaderType.DOT }], keepLines: true, children: kids }));
  });
  return out;
}
const tocSection = { properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Sumário") }, footers: { default: mkFooter() }, children: tocParagraphs() };

const doc = new Document({
  creator: AUTHOR, title: `${TITLE} — ${SUBTITLE}`, description: "Livro didático sobre Grandes Modelos de Linguagem", language: "pt-BR",
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
