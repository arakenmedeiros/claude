// Gerador do DOCX — "Inteligência Artificial Aplicada à Educação"
const fs = require("fs");
const path = require("path");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, HeadingLevel, LevelFormat, TabStopType, LeaderType, Header, Footer, PageNumber, SectionType,
  VerticalAlign, HorizontalPositionRelativeFrom, VerticalPositionRelativeFrom, TextWrappingType,
} = D;

const ROOT = path.join(__dirname, "..");
const CAP = path.join(ROOT, "cap");
const FIGS = path.join(ROOT, "figs");
const OUT = process.argv[2] || path.join(ROOT, "saida", "livro.docx");
const ONLY = process.env.ONLY ? process.env.ONLY.split(",").map(Number) : null; // para testes
const PAGES = fs.existsSync(path.join(__dirname, "pages.json")) ? JSON.parse(fs.readFileSync(path.join(__dirname, "pages.json"), "utf8")) : {};

const FONT = "Calibri";
const C = { pri: "D9653B", priD: "5A2E2A", teal: "2A9D8F", amb: "E9A23B", plum: "6D3B6E", blue: "3D7EA6", grn: "6E9A4F", rose: "D95D7F", ink: "3B2A2A", gry: "7A6F69", light: "F7EFE6", mid: "E6D8C8", cream: "FFF8EF" };
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134;
const CW = PAGE_W - 2 * MARGIN;
const TITLE = "Inteligência Artificial Aplicada à Educação";
const SUBTITLE = "da educação infantil à pós-graduação";
const SUB2 = "Diretrizes, práticas, oportunidades, benefícios e riscos";
const AUTHOR = "Araken de Medeiros Santos";
const YEAR = "2026";

// ---------------------------------------------------------------- referências (banco)
const BANK = {};
for (const ln of fs.readFileSync(path.join(CAP, "bank.tsv"), "utf8").split("\n")) {
  if (!ln.trim() || ln.startsWith("#")) continue;
  const [k, st, ref] = ln.split("|");
  BANK[k.trim()] = { st: st.trim(), ref: ref.trim() };
}
const usedByChapter = {};
let CUR_CH = 0;
const missingKeys = new Set();
function narrKey(k) { // "BASTANI et al., 2025" -> "Bastani et al. (2025)"
  const m = k.match(/^(.*),\s*(\d{4}[a-z]?)$/);
  if (!m) return k;
  const ACR = new Set(["UNESCO", "OCDE", "OECD", "CAST", "UNICEF", "CETIC.BR", "CGI.BR", "TEQSA", "ANPD", "CNE", "MEC", "INEP", "CAPES", "CNPQ", "ISTE", "ICMJE", "COPE", "LEARNLM", "TEAM"]);
  const names = m[1].split(";").map((s) => s.trim().split(" ").map((w) => (w === "et" || w === "al." || w === "e" || ACR.has(w) ? w : (w.charAt(0) + w.slice(1).toLowerCase()).replace(/-([a-zà-ÿ])/g, (x, b) => "-" + b.toUpperCase()))).join(" "));
  const who = names.length > 1 ? names.slice(0, -1).join(", ") + " e " + names[names.length - 1] : names[0];
  return `${who.replace(/ et al$/, " et al.")} (${m[2]})`.replace("et al.. ", "et al. ");
}
function cite(text) {
  return text.replace(/\{\{(n:)?([^}]+)\}\}/g, (all, n, body) => {
    const keys = body.split(/(?<=\d{4}[a-z]?);\s*/).map((s) => s.trim());
    for (const k of keys) { if (!BANK[k]) missingKeys.add(k); else { (usedByChapter[CUR_CH] = usedByChapter[CUR_CH] || new Set()).add(k); } }
    if (n) return keys.map(narrKey).join("; ");
    return "(" + keys.join("; ") + ")";
  });
}

// ---------------------------------------------------------------- helpers
const readTxt = (f) => fs.readFileSync(path.join(CAP, f), "utf8").replace(/\r/g, "");
const exists = (f) => fs.existsSync(path.join(CAP, f));
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function runs(text, base = {}) {
  text = cite(text);
  const out = [];
  const re = /(\*\*(?:[^*]|\*(?!\*)[^*]+\*)+\*\*|\*[^*\s][^*]*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(...runs(t.slice(2, -2), { ...base, bold: true }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const numberingConfigs = [];
let numCounter = 0;
function newNumRef() {
  const ref = "num" + (++numCounter);
  numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 320 } }, run: { bold: true, color: C.pri } } }] });
  return ref;
}
numberingConfigs.push({ reference: "bul", levels: [
  { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } }, run: { color: C.pri, bold: true } } },
  { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 280 } } } } ] });

// ---------------------------------------------------------------- parser de capítulo (markdown simplificado)
function parseMd(text) {
  const lines = text.split("\n");
  const blocks = []; const meta = { obj: [] };
  let i = 0; let para = [];
  const flush = () => { if (para.length) { blocks.push({ t: "p", text: para.join(" ") }); para = []; } };
  while (i < lines.length) {
    const ln = lines[i];
    if (ln.startsWith("@part:")) { meta.part = ln.slice(6).trim(); i++; continue; }
    if (ln.startsWith("@obj:")) { meta.obj.push(ln.slice(5).trim()); i++; continue; }
    if (/^# /.test(ln)) { flush(); meta.title = ln.slice(2).trim(); i++; continue; }
    if (/^### /.test(ln)) { flush(); blocks.push({ t: "h4", text: ln.slice(4).trim() }); i++; continue; }
    if (/^## /.test(ln)) { flush(); blocks.push({ t: "h3", text: ln.slice(3).trim() }); i++; continue; }
    if (ln.startsWith(":::")) { flush(); const m = ln.match(/^:::(\w+)\s*(.*)$/); const body = []; i++; while (i < lines.length && lines[i].trim() !== ":::") { body.push(lines[i]); i++; } i++; blocks.push({ t: m[1] === "ativ" ? "ativ" : "callout", kind: m[1], title: m[2], body }); continue; }
    const fig = ln.match(/^!\[(\w+)\]\((.*)\)\s*$/);
    if (fig) { flush(); const [cap, desc] = fig[2].split("||").map((s) => s.trim()); blocks.push({ t: "fig", id: fig[1], caption: cap, desc: desc || "" }); i++; continue; }
    if (ln.startsWith("|")) { flush(); const rows = []; while (i < lines.length && lines[i].startsWith("|")) { rows.push(lines[i]); i++; } blocks.push({ t: "table", rows: rows.filter((r) => !/^\|\s*:?-{2,}/.test(r)).map((r) => r.replace(/^\||\|\s*$/g, "").split("|").map((c) => c.trim())) }); continue; }
    if (/^- /.test(ln)) { flush(); const items = []; while (i < lines.length && /^- /.test(lines[i])) { items.push(lines[i].slice(2)); i++; } blocks.push({ t: "ul", items }); continue; }
    if (/^\d+\. /.test(ln)) { flush(); const items = []; while (i < lines.length && /^\d+\. /.test(lines[i])) { items.push(lines[i].replace(/^\d+\.\s+/, "")); i++; } blocks.push({ t: "ol", items }); continue; }
    if (ln.trim() === "") { flush(); i++; continue; }
    if (ln.startsWith("> ")) { flush(); const q = []; while (i < lines.length && lines[i].startsWith("> ")) { q.push(lines[i].slice(2)); i++; } blocks.push({ t: "quote", text: q.join(" ") }); continue; }
    para.push(ln.trim()); i++;
  }
  flush();
  return { meta, blocks };
}

// ---------------------------------------------------------------- builders
const headings = [];
let hid = 0;
function heading(level, text, opts = {}) {
  text = cite(text).replace(/\*+/g, "");
  const id = "h" + (++hid);
  if (opts.toc !== false) headings.push({ level, text, id });
  const lv = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3, 4: HeadingLevel.HEADING_4 }[level];
  return new Paragraph({ heading: lv, pageBreakBefore: !!opts.pageBreak, keepNext: true, children: [new TextRun({ text })] });
}
function P(text, o = {}) { return new Paragraph({ spacing: { after: 120, line: 300 }, alignment: o.align || AlignmentType.LEFT, keepNext: o.keepNext, indent: o.indent, children: runs(text, { size: 21, ...(o.run || {}) }) }); }
function spacer(after = 120) { return new Paragraph({ spacing: { after, line: 120 }, children: [] }); }
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

const CALL = {
  situacao: { c: C.pri, f: "FDF1EA", label: "SITUAÇÃO INICIAL" },
  pergunta: { c: C.plum, f: "F5EEF6", label: "PERGUNTA MOBILIZADORA" },
  pratica: { c: C.teal, f: "EAF6F4", label: "NA PRÁTICA DOCENTE" },
  experimente: { c: C.amb, f: "FEF4E0", label: "EXPERIMENTE COM SUA TURMA" },
  verifique: { c: C.rose, f: "FCEBF0", label: "PARE E VERIFIQUE" },
  pesquisa: { c: C.blue, f: "EBF3F9", label: "O QUE A PESQUISA PERMITE AFIRMAR" },
  dispensar: { c: C.priD, f: "F3E9E7", label: "QUANDO DISPENSAR A IA" },
  poucos: { c: C.grn, f: "EFF5EA", label: "UMA ALTERNATIVA COM POUCOS RECURSOS" },
  lembrar: { c: C.pri, f: "FDF1EA", label: "O QUE VALE A PENA LEMBRAR" },
  alem: { c: C.plum, f: "F5EEF6", label: "PARA IR ALÉM" },
  reflexao: { c: C.teal, f: "EAF6F4", label: "REFLITA SOBRE A SUA PRÁTICA" },
  hipotese: { c: C.gry, f: "F4F1EE", label: "SITUAÇÃO HIPOTÉTICA" },
  prompt: { c: C.blue, f: "EBF3F9", label: "MODELO DE PROMPT" },
  modelo: { c: C.gry, f: "F4F1EE", label: "MODELO ADAPTÁVEL" },
  cotidiano: { c: C.amb, f: "FEF4E0", label: "NO SEU DIA A DIA" },
};
function bodyParas(body, size = 20) {
  const out = []; let para = []; let i = 0;
  const flush = () => { if (para.length) { out.push(new Paragraph({ spacing: { after: 80, line: 280 }, children: runs(para.join(" "), { size }) })); para = []; } };
  while (i < body.length) {
    const ln = body[i];
    if (/^- /.test(ln)) { flush(); while (i < body.length && /^- /.test(body[i])) { out.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(body[i].slice(2), { size }) })); i++; } continue; }
    if (/^\d+\. /.test(ln)) { flush(); const ref = newNumRef(); while (i < body.length && /^\d+\. /.test(body[i])) { out.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 276 }, children: runs(body[i].replace(/^\d+\.\s+/, ""), { size }) })); i++; } continue; }
    if (ln.startsWith("> ")) { flush(); const q = []; while (i < body.length && body[i].startsWith("> ")) { q.push(body[i].slice(2)); i++; } out.push(new Paragraph({ spacing: { after: 80, line: 276 }, indent: { left: 240, right: 120 }, shading: { type: ShadingType.CLEAR, fill: "FFFFFF", color: "auto" }, border: { left: { style: BorderStyle.SINGLE, size: 12, color: C.blue, space: 6 } }, children: runs(q.join(" "), { size: size - 1, italics: false, color: C.priD }) })); continue; }
    if (ln.trim() === "") { flush(); i++; continue; }
    para.push(ln.trim()); i++;
  }
  flush();
  return out;
}
function callout(b) {
  const s = CALL[b.kind] || CALL.pratica;
  const norm = (t) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z ]/g, "");
  const lab2 = norm(s.label).split(" ").slice(0, 2).join(" ");
  const dup = b.title && norm(b.title).startsWith(lab2);
  const titleRuns = [];
  if (!dup) titleRuns.push(new TextRun({ text: s.label + (b.title ? "  " : ""), bold: true, size: 17, color: s.c, characterSpacing: 20 }));
  if (b.title) titleRuns.push(new TextRun({ text: b.title, bold: true, size: 22, color: s.c }));
  const kids = [new Paragraph({ spacing: { after: 80 }, keepNext: true, children: titleRuns }), ...bodyParas(b.body)];
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: kids.length < 9, children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: s.f, color: "auto" }, margins: { top: 100, bottom: 60, left: 200, right: 160 }, borders: { top: NB, bottom: NB, right: NB, left: { style: BorderStyle.SINGLE, size: 36, color: s.c } }, children: kids })] })] });
}
// Atividade prática: tabela de duas colunas "Campo :: conteúdo"
function ativ(b) {
  const fields = []; let cur = null;
  for (const ln of b.body) {
    const m = ln.match(/^([^:]{2,60})::\s*(.*)$/);
    if (m && !ln.startsWith("- ") && !ln.startsWith("> ")) { cur = { k: m[1].trim(), v: m[2] ? [m[2]] : [] }; fields.push(cur); }
    else if (cur) cur.v.push(ln);
  }
  const W1 = 2300, W2 = CW - W1;
  const bd = { style: BorderStyle.SINGLE, size: 4, color: C.mid };
  const head = new TableRow({ cantSplit: true, children: [new TableCell({ columnSpan: 2, width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: C.teal, color: "auto" }, margins: { top: 90, bottom: 90, left: 160, right: 120 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, children: [new Paragraph({ keepNext: true, children: [new TextRun({ text: "ATIVIDADE PRÁTICA  ", bold: true, size: 16, color: "D5F0EC", characterSpacing: 20 }), new TextRun({ text: b.title, bold: true, size: 23, color: "FFFFFF" })] })] })] });
  const rows = fields.map((f) => new TableRow({ cantSplit: f.v.join("").length < 900, children: [
    new TableCell({ width: { size: W1, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: "EAF6F4", color: "auto" }, margins: { top: 70, bottom: 70, left: 110, right: 90 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, children: [new Paragraph({ spacing: { after: 0, line: 252 }, children: [new TextRun({ text: f.k, bold: true, size: 18, color: C.teal })] })] }),
    new TableCell({ width: { size: W2, type: WidthType.DXA }, margins: { top: 70, bottom: 50, left: 130, right: 110 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, children: bodyParas(f.v, 19).length ? bodyParas(f.v, 19) : [new Paragraph({ children: [] })] }),
  ] }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [W1, W2], rows: [head, ...rows] });
}
function table(b) {
  const rows = b.rows; const n = rows[0].length;
  const lens = new Array(n).fill(0).map((_, j) => Math.max(...rows.map((r) => Math.min(44, Math.max(7, (r[j] || "").length)))));
  const tot = lens.reduce((a, c) => a + c, 0);
  let widths = lens.map((l) => Math.max(1000, Math.floor((l / tot) * CW)));
  widths[widths.length - 1] += CW - widths.reduce((a, c) => a + c, 0);
  const bd = { style: BorderStyle.SINGLE, size: 4, color: C.mid };
  const trs = rows.map((r, ri) => new TableRow({ tableHeader: ri === 0, cantSplit: true, children: r.map((c, j) => new TableCell({ width: { size: widths[j], type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: ri === 0 ? C.pri : ri % 2 === 0 ? "FBF4EC" : "FFFFFF", color: "auto" }, margins: { top: 60, bottom: 60, left: 100, right: 100 }, borders: { top: bd, bottom: bd, left: bd, right: bd }, verticalAlign: VerticalAlign.CENTER, children: [new Paragraph({ spacing: { after: 0, line: 252 }, children: runs(c, { size: 18, ...(ri === 0 ? { bold: true, color: "FFFFFF" } : {}) }) })] })) }));
  return [new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: trs }), spacer(160)];
}
const figNum = {};
function figure(b) {
  const file = path.join(FIGS, b.id + ".png");
  if (!fs.existsSync(file)) { console.warn("FIGURA AUSENTE:", b.id); return [P("[figura ausente: " + b.id + "]")]; }
  const { w, h } = pngSize(file);
  let dw = Math.min(600, (w / 200) * 96), dh = (dw * h) / w;
  if (dh > 430) { dh = 430; dw = (dh * w) / h; }
  const alt = b.caption + (b.desc ? " Descrição: " + b.desc : "");
  const out = [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: Math.round(dw), height: Math.round(dh) }, altText: { title: b.id, description: alt.replace(/[{}]/g, ""), name: b.id } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: !!b.desc, spacing: { after: b.desc ? 40 : 200 }, children: runs(b.caption, { size: 18, bold: false, color: C.priD }) }),
  ];
  if (b.desc) out.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 200 }, indent: { left: 400, right: 400 }, children: [new TextRun({ text: "Descrição acessível: ", bold: true, size: 16, color: C.gry }), ...runs(b.desc, { size: 16, color: C.gry, italics: true })] }));
  return out;
}
function banner(n, col) {
  const file = path.join(FIGS, `banner${String(n).padStart(2, "0")}.png`);
  if (!fs.existsSync(file)) return spacer(60);
  const { w, h } = pngSize(file);
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
      case "quote": out.push(new Paragraph({ spacing: { after: 120 }, indent: { left: 400 }, border: { left: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 8 } }, children: runs(b.text, { size: 21, italics: true }) })); break;
      case "ul": b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 70, line: 288 }, children: runs(it, { size: 21 }) }))); out.push(spacer(60)); break;
      case "ol": { const ref = newNumRef(); b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 70, line: 288 }, children: runs(it, { size: 21 }) }))); out.push(spacer(60)); break; }
      case "callout": out.push(callout(b)); out.push(spacer(140)); break;
      case "ativ": out.push(ativ(b)); out.push(spacer(160)); break;
      case "table": out.push(...table(b)); break;
      case "fig": out.push(...figure(b)); break;
    }
  }
  return out;
}

// ---------------------------------------------------------------- questões
const FORBID = /(todas as (alternativas )?anteriores|nenhuma das (alternativas )?anteriores|alternativas? [a-e]\b|ambas as)/i;
const warnings = [];
function parseEx(text, ch) {
  const blocks = text.trim().split(/\n\s*\n/);
  const mc = [], dc = [];
  for (const bl of blocks) {
    const L = bl.split("\n").filter((x) => x.trim() !== "");
    if (L[0].startsWith("MC:")) {
      const stem = L[0].slice(3).trim();
      const opts = []; let why = "";
      for (const l of L.slice(1)) {
        if (l.startsWith("+ ")) opts.push({ ok: true, t: l.slice(2).trim(), why: "" });
        else if (l.startsWith("- ")) { const [t, w] = l.slice(2).split("||"); opts.push({ ok: false, t: t.trim(), why: (w || "").trim() }); }
        else if (l.startsWith("R:")) why = l.slice(2).trim();
      }
      if (opts.length !== 5 || opts.filter((o) => o.ok).length !== 1) { warnings.push(`cap ${ch}: questão MC mal formada: ${stem.slice(0, 50)}`); continue; }
      opts.forEach((o) => { if (FORBID.test(o.t)) warnings.push(`cap ${ch}: alternativa referencial: ${o.t.slice(0, 50)}`); });
      mc.push({ stem, opts, why });
    } else if (L[0].startsWith("DISC:")) {
      const q = { stem: L[0].slice(5).trim(), ans: "", crit: "" };
      for (const l of L.slice(1)) { if (l.startsWith("R:")) q.ans = l.slice(2).trim(); else if (l.startsWith("C:")) q.crit = l.slice(2).trim(); }
      dc.push(q);
    }
  }
  // gabarito equilibrado: 6 de cada letra, em sequência embaralhada de modo determinístico
  const n = mc.length;
  const rnd = mulberry32(7000 + ch);
  const target = []; for (let k = 0; k < n; k++) target.push(k % 5);
  for (let k = n - 1; k > 0; k--) { const j = Math.floor(rnd() * (k + 1)); [target[k], target[j]] = [target[j], target[k]]; }
  // evita 3 respostas iguais seguidas
  for (let k = 2; k < n; k++) if (target[k] === target[k - 1] && target[k] === target[k - 2]) { const j = (k + 7) % n; [target[k], target[j]] = [target[j], target[k]]; }
  mc.forEach((q, i) => {
    const ci = q.opts.findIndex((o) => o.ok);
    const others = q.opts.filter((o) => !o.ok);
    // embaralha distratores de modo determinístico e insere o correto na posição-alvo
    for (let k = others.length - 1; k > 0; k--) { const j = Math.floor(rnd() * (k + 1)); [others[k], others[j]] = [others[j], others[k]]; }
    const arr = []; let oi = 0;
    for (let p = 0; p < 5; p++) arr.push(p === target[i] ? q.opts[ci] : others[oi++]);
    q.opts = arr; q.ans = "abcde"[target[i]];
  });
  if (n !== 30) warnings.push(`cap ${ch}: ${n} questões MC (esperado 30)`);
  if (dc.length !== 20) warnings.push(`cap ${ch}: ${dc.length} questões discursivas (esperado 20)`);
  const cnt = [0, 0, 0, 0, 0]; mc.forEach((q) => cnt["abcde".indexOf(q.ans)]++);
  if (n === 30 && cnt.some((c) => c !== 6)) warnings.push(`cap ${ch}: gabarito desbalanceado ${cnt}`);
  return { mc, dc };
}
function exercises(ex, chn) {
  const out = [];
  out.push(heading(3, "Exercícios de múltipla escolha", { pageBreak: true }));
  out.push(P(`Escolha **uma** alternativa em cada questão. O gabarito comentado está no Apêndice A (Capítulo ${chn}).`, { run: { color: C.gry } }));
  ex.mc.forEach((q, i) => {
    out.push(new Paragraph({ keepNext: true, spacing: { before: 140, after: 60, line: 276 }, children: [new TextRun({ text: `Questão ${i + 1}. `, bold: true, color: C.pri, size: 21 }), ...runs(q.stem, { size: 21 })] }));
    q.opts.forEach((o, k) => out.push(new Paragraph({ keepNext: k < 4, keepLines: true, spacing: { after: 30, line: 264 }, indent: { left: 500, hanging: 300 }, children: [new TextRun({ text: "abcde"[k] + ")\t", bold: true, color: C.gry, size: 20 }), ...runs(o.t, { size: 20 })], tabStops: [{ type: TabStopType.LEFT, position: 500 }] })));
  });
  out.push(heading(3, "Exercícios discursivos", { pageBreak: true }));
  out.push(P(`Responda com suas palavras, justificando com base no conteúdo do capítulo e na sua realidade de trabalho. Sugestões de resposta e critérios de avaliação estão no Apêndice B (Capítulo ${chn}).`, { run: { color: C.gry } }));
  ex.dc.forEach((q, i) => out.push(new Paragraph({ keepLines: true, spacing: { before: 120, after: 80, line: 288 }, indent: { left: 500, hanging: 500 }, tabStops: [{ type: TabStopType.LEFT, position: 500 }], children: [new TextRun({ text: `${i + 1}.\t`, bold: true, color: C.pri, size: 21 }), ...runs(q.stem, { size: 21 })] })));
  return out;
}
function chapterRefs(chn) {
  const set = usedByChapter[chn]; if (!set || !set.size) return [];
  const list = [...set].map((k) => BANK[k].ref).sort((a, b) => a.localeCompare(b, "pt"));
  const out = [heading(3, "Referências utilizadas neste capítulo")];
  list.forEach((r) => out.push(new Paragraph({ keepLines: true, spacing: { after: 80, line: 260 }, indent: { left: 420, hanging: 420 }, children: [new TextRun({ text: r, size: 18 })] })));
  return out;
}

// ---------------------------------------------------------------- cabeçalho/rodapé
function mkHeader(text) {
  return new Header({ children: [new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: C.pri, space: 4 } }, tabStops: [{ type: TabStopType.RIGHT, position: CW }], children: [new TextRun({ text: TITLE, size: 16, color: C.gry }), new TextRun({ text: "\t" + text, size: 16, color: C.pri, bold: true })] })] });
}
function mkFooter() { return new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: C.gry })] })] }); }
const secProps = (extra = {}) => ({ page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1300, bottom: 1200, left: MARGIN, right: MARGIN, header: 560, footer: 500 } }, ...extra });

// ---------------------------------------------------------------- montagem
const sections = [];
{
  const file = path.join(FIGS, "capa.png");
  sections.push({ properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 0, bottom: 0, left: 0, right: 0, header: 0, footer: 0 } } },
    children: [new Paragraph({ spacing: { after: 0 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: 794, height: 1123 }, floating: { horizontalPosition: { relative: HorizontalPositionRelativeFrom.PAGE, offset: 0 }, verticalPosition: { relative: VerticalPositionRelativeFrom.PAGE, offset: 0 }, wrap: { type: TextWrappingType.NONE }, behindDocument: true }, altText: { title: "Capa", description: "Capa do livro Inteligência Aplicada à Educação: professores e estudantes de diferentes idades e origens, em uma trilha ascendente da infância à pós-graduação, conectados por pontos de luz.", name: "capa" } })] })] });
}
sections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }),
  children: [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 3000, after: 160 }, children: [new TextRun({ text: TITLE.toUpperCase(), bold: true, size: 50, color: C.priD, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: SUBTITLE, italics: true, size: 32, color: C.pri })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 700 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 14 } }, children: [new TextRun({ text: SUB2, size: 26, color: C.gry })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 700, after: 120 }, children: [new TextRun({ text: AUTHOR, bold: true, size: 34, color: C.ink })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1800 }, children: [new TextRun({ text: YEAR, size: 24, color: C.gry })] }),
  ] });

// capítulos
const chapters = [];
for (let n = 1; n <= 32; n++) {
  if (ONLY && !ONLY.includes(n)) continue;
  const nn = String(n).padStart(2, "0");
  if (!exists(`ch${nn}.md`)) continue;
  CUR_CH = n;
  const md = parseMd(readTxt(`ch${nn}.md`));
  const ex = exists(`ex${nn}.txt`) ? parseEx(readTxt(`ex${nn}.txt`), n) : { mc: [], dc: [] };
  chapters.push({ n, ...md, ex });
}
const PART_INFO = JSON.parse(fs.readFileSync(path.join(CAP, "partes.json"), "utf8"));
const PART_COL = [C.pri, C.teal, "C98A10", C.plum, C.blue, C.rose, C.grn];

const contentSections = [];
function frontMatter(file, title) {
  const md = parseMd(readTxt(file));
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(title) }, footers: { default: mkFooter() }, children: [heading(1, md.meta.title || title), ...renderBlocks(md.blocks)] });
}
if (!ONLY) { frontMatter("apresentacao.md", "Apresentação"); frontMatter("como_usar.md", "Como usar este livro"); }

for (const c of chapters) {
  CUR_CH = c.n;
  if (c.meta.part) {
    const pidx = parseInt(c.meta.part.match(/Parte\s+([IVX]+)/)[1].replace(/VII/, "7").replace(/VI/, "6").replace(/IV/, "4").replace(/V/, "5").replace(/III/, "3").replace(/II/, "2").replace(/I/, "1")) - 1;
    const col = PART_COL[pidx];
    const [pnum] = c.meta.part.split(" — ");
    const list = chapters.filter((x) => x.meta.part === c.meta.part || (x.n > c.n && !x.meta.part && x.n < (chapters.find((y) => y.n > c.n && y.meta.part)?.n || 999)));
    const kids = [
      new Paragraph({ spacing: { before: 3000, after: 100 }, children: [new TextRun({ text: pnum.toUpperCase(), bold: true, size: 40, color: col, characterSpacing: 60 })] }),
      heading(1, c.meta.part),
      new Paragraph({ spacing: { before: 200, after: 300 }, border: { top: { style: BorderStyle.SINGLE, size: 24, color: col, space: 10 } }, children: [new TextRun({ text: PART_INFO[c.meta.part] || "", size: 26, color: C.ink, italics: true })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Nesta parte:", bold: true, size: 22, color: col })] }),
      ...list.map((x) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text: `Capítulo ${x.n} — ${x.meta.title}`, size: 22 })] })),
    ];
    contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), children: kids });
  }
  const kids = [];
  kids.push(banner(c.n));
  kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title}`));
  kids.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: [CW], rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: CW, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: "FDF1EA", color: "auto" }, margins: { top: 120, bottom: 80, left: 220, right: 180 }, borders: { top: NB, bottom: NB, right: NB, left: { style: BorderStyle.SINGLE, size: 36, color: C.pri } }, children: [new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "OBJETIVOS DE APRENDIZAGEM", bold: true, size: 17, color: C.pri, characterSpacing: 20 })] }), new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Ao final deste capítulo você será capaz de:", bold: true, size: 21 })] }), ...c.meta.obj.map((o) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 50, line: 276 }, children: runs(o, { size: 20 }) }))] })] })] }));
  kids.push(spacer(200));
  kids.push(...renderBlocks(c.blocks));
  if (c.ex.mc.length) kids.push(...exercises(c.ex, c.n));
  kids.push(...chapterRefs(c.n));
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(`Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`) }, footers: { default: mkFooter() }, children: kids });
}

// Apêndice A e B: gabaritos
function gabaritoA() {
  const kids = [heading(1, "Apêndice A — Gabarito comentado das questões objetivas")];
  kids.push(P("Para cada questão, são indicadas a alternativa correta, a justificativa e o motivo pelo qual cada distrator não é adequado. Use o gabarito para estudar, e não apenas para conferir: se você errou, volte ao trecho do capítulo e releia a explicação."));
  for (const c of chapters) {
    if (!c.ex.mc.length) continue;
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: true }));
    const cnt = [0, 0, 0, 0, 0]; c.ex.mc.forEach((q) => cnt["abcde".indexOf(q.ans)]++);
    kids.push(new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Resumo do gabarito: ", bold: true, size: 19, color: C.pri }), new TextRun({ text: c.ex.mc.map((q, i) => `${i + 1}-${q.ans}`).join("  "), size: 18, font: "Consolas" })] }));
    c.ex.mc.forEach((q, i) => {
      kids.push(new Paragraph({ keepNext: true, spacing: { before: 100, after: 30, line: 264 }, children: [new TextRun({ text: `Questão ${i + 1} — resposta ${q.ans}. `, bold: true, size: 19, color: C.pri }), ...runs(q.why, { size: 19 })] }));
      const demais = q.opts.map((o, k) => ({ o, k })).filter((x) => !x.o.ok).map((x) => `${"abcde"[x.k]}) ${x.o.why}`).join("  ");
      if (demais) kids.push(new Paragraph({ keepLines: true, spacing: { after: 50, line: 252 }, indent: { left: 300 }, children: [new TextRun({ text: "Por que as demais não: ", bold: true, size: 17, color: C.gry }), ...runs(demais, { size: 17, color: C.gry })] }));
    });
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice A — Gabarito comentado") }, footers: { default: mkFooter() }, children: kids });
}
function gabaritoB() {
  const kids = [heading(1, "Apêndice B — Sugestões de resposta e critérios de avaliação das questões discursivas")];
  kids.push(P("As respostas abaixo são **sugestões**: outras formulações podem ser igualmente adequadas se estiverem bem fundamentadas, forem coerentes com o contexto descrito e mostrarem compreensão dos conceitos do capítulo. Em cada questão, os critérios indicam o que observar. Respostas alternativas bem argumentadas devem ser reconhecidas."));
  for (const c of chapters) {
    if (!c.ex.dc.length) continue;
    kids.push(heading(2, `Capítulo ${c.n} — ${c.meta.title.split(":")[0]}`, { pageBreak: true }));
    c.ex.dc.forEach((q, i) => {
      kids.push(new Paragraph({ keepNext: true, keepLines: true, spacing: { before: 100, after: 30, line: 264 }, indent: { left: 400, hanging: 400 }, tabStops: [{ type: TabStopType.LEFT, position: 400 }], children: [new TextRun({ text: `${i + 1}.\t`, bold: true, size: 19, color: C.pri }), ...runs(q.ans, { size: 19 })] }));
      if (q.crit) kids.push(new Paragraph({ keepLines: true, spacing: { after: 60, line: 252 }, indent: { left: 400 }, children: [new TextRun({ text: "Critérios: ", bold: true, size: 17, color: C.gry }), ...runs(q.crit, { size: 17, color: C.gry })] }));
    });
  }
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Apêndice B — Discursivas") }, footers: { default: mkFooter() }, children: kids });
}
if (chapters.length) { gabaritoA(); gabaritoB(); }

// Apêndices de materiais (C a N)
const APS = JSON.parse(fs.readFileSync(path.join(CAP, "apendices.json"), "utf8"));
if (!ONLY) {
  for (const ap of APS) {
    CUR_CH = 900;
    if (ap.file.endsWith(".txt")) { // glossário
      const kids = [heading(1, ap.title)];
      kids.push(P(ap.intro || ""));
      const lines = readTxt(ap.file).trim().split("\n").filter(Boolean).sort((a, b) => a.localeCompare(b, "pt"));
      for (const ln of lines) { const k = ln.indexOf(":"); kids.push(new Paragraph({ keepLines: true, spacing: { after: 90, line: 276 }, children: [new TextRun({ text: ln.slice(0, k), bold: true, color: C.priD, size: 20 }), ...runs(" — " + ln.slice(k + 1).trim(), { size: 20 })] })); }
      contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(ap.short) }, footers: { default: mkFooter() }, children: kids });
    } else {
      const md = parseMd(readTxt(ap.file));
      contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader(ap.short) }, footers: { default: mkFooter() }, children: [heading(1, ap.title), ...renderBlocks(md.blocks)] });
    }
  }
  // Referências consolidadas
  const used = new Set(); Object.values(usedByChapter).forEach((s) => s.forEach((k) => used.add(k)));
  const kids = [heading(1, "Referências")];
  kids.push(P("Lista consolidada das obras citadas nos capítulos e apêndices. Os dados de autoria, título, periódico e ano dos artigos de periódico marcados como verificados foram conferidos em base bibliográfica em 17/10/2026; volume, número e páginas, quando ausentes, devem ser completados na revisão editorial. A legislação e os documentos de política foram citados conforme os dados oficiais conhecidos, com status de vigência a conferir na data de uso. Veja a Nota sobre a verificação das fontes, no início do livro.", { run: { color: C.gry } }));
  [...used].map((k) => BANK[k].ref).sort((a, b) => a.localeCompare(b, "pt")).forEach((r) => kids.push(new Paragraph({ keepLines: true, spacing: { after: 90, line: 264 }, indent: { left: 420, hanging: 420 }, children: [new TextRun({ text: r, size: 19 })] })));
  contentSections.push({ properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Referências") }, footers: { default: mkFooter() }, children: kids });
}

// ---------------------------------------------------------------- sumário (campo TOC com resultado em cache)
function tocParagraphs() {
  const out = [];
  out.push(new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: C.amb, space: 6 } }, children: [new TextRun({ text: "Sumário", bold: true, size: 44, color: C.priD })] }));
  headings.forEach((h, idx) => {
    const pg = PAGES[h.id] !== undefined ? String(PAGES[h.id]) : "000";
    const lvl = h.level; const ind = [0, 0, 280, 620, 960][lvl];
    const style = { 1: { bold: true, size: 22, color: C.priD, before: 180 }, 2: { bold: true, size: 20, color: C.ink, before: 80 }, 3: { bold: false, size: 18, color: C.ink, before: 0 }, 4: { bold: false, size: 17, color: C.gry, before: 0 } }[lvl];
    const kids = [];
    if (idx === 0) kids.push(new TextRun({ text: "@@TOCBEGIN@@" }));
    kids.push(new TextRun({ text: h.text, bold: style.bold, size: style.size, color: style.color }), new TextRun({ children: ["\t", pg], size: style.size, color: style.color, bold: style.bold }));
    if (idx === headings.length - 1) kids.push(new TextRun({ text: "@@TOCEND@@" }));
    out.push(new Paragraph({ spacing: { before: style.before, after: 10, line: 240 }, indent: { left: ind, right: 650 }, tabStops: [{ type: TabStopType.RIGHT, position: CW, leader: LeaderType.DOT }], keepLines: true, children: kids }));
  });
  return out;
}
const tocSection = { properties: secProps({ type: SectionType.NEXT_PAGE }), headers: { default: mkHeader("Sumário") }, footers: { default: mkFooter() }, children: tocParagraphs() };

const doc = new Document({
  creator: AUTHOR, title: `${TITLE}: ${SUBTITLE}`, description: SUB2, language: "pt-BR",
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: C.ink } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 44, bold: true, color: C.priD, font: FONT }, paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 38, bold: true, color: C.pri, font: FONT }, paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 1, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: C.amb, space: 6 } } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 29, bold: true, color: C.priD, font: FONT }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, color: C.teal, font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 3 } },
    ],
  },
  numbering: { config: numberingConfigs },
  sections: [sections[0], sections[1], tocSection, ...contentSections],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  fs.writeFileSync(path.join(__dirname, "headings.json"), JSON.stringify(headings, null, 1));
  console.log("ok", OUT, headings.length, "títulos,", numCounter, "listas;", chapters.length, "capítulos");
  if (missingKeys.size) console.log("CHAVES DE REFERÊNCIA AUSENTES NO BANCO:", [...missingKeys]);
  if (warnings.length) console.log("AVISOS:\n" + warnings.join("\n"));
});
