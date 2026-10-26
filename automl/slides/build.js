// Gera um deck .pptx por capítulo a partir de specs/chNN.json
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const RDS = require("react-dom/server");
const FA = require("react-icons/fa");
const SKILL = require("child_process").execSync("ls -d /root/.claude/skills/synced/*/pptx").toString().trim().split("\n")[0];
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");

const FIGS = "/home/user/claude/automl/figs/";
const THEME = {
  name: "AutoML", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "111827", lt1: "FFFFFF", dk2: "1E1B4B", lt2: "EEF2FF", accent1: "4F46E5", accent2: "06B6D4", accent3: "10B981", accent4: "F59E0B", accent5: "EF4444", accent6: "8B5CF6", hlink: "4F46E5", folHlink: "8B5CF6" },
};
const HEX = { accent1: "4F46E5", accent2: "06B6D4", accent3: "10B981", accent4: "F59E0B", accent5: "EF4444", accent6: "8B5CF6", text2: "1E1B4B", text1: "111827", background1: "FFFFFF", background2: "EEF2FF" };
const PART_COLOR = { "Parte I": "accent1", "Parte II": "accent2", "Parte III": "accent3", "Parte IV": "accent4", "Parte V": "accent6" };
const MUTED = "6B7280";
const CYCLE = ["accent1", "accent2", "accent3", "accent4", "accent6", "accent5"];

const iconCache = {};
async function icon(name, hex) {
  const key = name + hex;
  if (iconCache[key]) return iconCache[key];
  const Comp = FA[name] || FA.FaStar;
  const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: 256 }));
  const png = await sharp(Buffer.from(svg)).resize(256, 256, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } }).png().toBuffer();
  return (iconCache[key] = "image/png;base64," + png.toString("base64"));
}
const sizeCache = {};
async function imgSize(name) {
  if (!sizeCache[name]) { const m = await sharp(FIGS + name + ".png").metadata(); sizeCache[name] = { w: m.width, h: m.height }; }
  return sizeCache[name];
}
function fit(box, s) {
  const r = Math.min(box.w / s.w, box.h / s.h); const w = s.w * r, h = s.h * r;
  return { x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h };
}

async function buildDeck(specPath, outPath) {
  const S = JSON.parse(fs.readFileSync(specPath, "utf8"));
  const partKey = S.parte.split(" — ")[0];
  const PC = PART_COLOR[partKey] || "accent1";
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title = `AutoML: Teoria e Prática — Capítulo ${S.num}: ${S.titulo}`;
  pres.author = "Araken de Medeiros Santos";
  pres.subject = "Slides de sala de aula";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const footer = `AutoML: Teoria e Prática  ·  Cap. ${S.num} — ${S.curto || S.titulo}`;

  pres.defineSlideMaster({
    title: "LIGHT", background: { color: C.background1 },
    objects: [
      { text: { text: footer, options: { x: 0.5, y: 5.28, w: 7.5, h: 0.25, fontSize: 10, color: MUTED, margin: 0, isTextBox: true } } },
    ],
    slideNumber: { x: 9.0, y: 5.28, w: 0.5, h: 0.25, fontSize: 10, color: MUTED, align: "right" },
  });
  pres.defineSlideMaster({
    title: "DARK", background: { color: C.text2 },
    objects: [{ text: { text: footer, options: { x: 0.5, y: 5.28, w: 7.5, h: 0.25, fontSize: 10, color: "A5B4FC", margin: 0, isTextBox: true } } }],
    slideNumber: { x: 9.0, y: 5.28, w: 0.5, h: 0.25, fontSize: 10, color: "A5B4FC", align: "right" },
  });

  const title = (s, t, color) => s.addText(t, { x: 0.5, y: 0.28, w: 9, h: 0.8, fontSize: 30, bold: true, color: color || C.text2, fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true, objectName: "Título" });
  const notes = (s, n) => { if (n) s.addNotes(n); };
  const circle = async (s, x, y, d, col, ic, name) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: C[col] }, line: { color: C[col] }, objectName: "Círculo " + name });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: x + d * 0.25, y: y + d * 0.25, w: d * 0.5, h: d * 0.5, objectName: "Ícone " + name, altText: name });
  };
  const bulletRuns = (items, size, col) => items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1, paraSpaceAfter: 10, fontSize: size, color: col || C.text1 } }));
  const boldRuns = (items, size, col) => {
    // item pode ser "Termo: explicação" -> termo em negrito
    const out = [];
    items.forEach((t, i) => {
      const k = t.indexOf(": ");
      const last = i === items.length - 1;
      if (k > 0 && k < 32) { out.push({ text: t.slice(0, k + 1) + " ", options: { bold: true, fontSize: size, color: col || C.text2 } }); out.push({ text: t.slice(k + 2), options: { breakLine: !last, paraSpaceAfter: 10, fontSize: size, color: C.text1 } }); }
      else out.push({ text: t, options: { breakLine: !last, paraSpaceAfter: 10, fontSize: size, color: C.text1 } });
    });
    return out;
  };

  const T = {};
  T.capa = async (d) => {
    const s = pres.addSlide({ masterName: "DARK" });
    s.addText(`CAPÍTULO ${S.num}`, { x: 0.6, y: 0.5, w: 6, h: 0.4, fontSize: 16, bold: true, color: C[PC], charSpacing: 6, margin: 0, isTextBox: true });
    s.addText(S.parte, { x: 0.6, y: 0.9, w: 8, h: 0.35, fontSize: 14, color: "A5B4FC", margin: 0, isTextBox: true });
    s.addText(S.titulo, { x: 0.6, y: 1.35, w: 8.8, h: 1.5, fontSize: 38, bold: true, color: C.background1, fontFace: "Cambria", valign: "top", margin: 0, isTextBox: true, objectName: "Título" });
    s.addText(S.subtitulo, { x: 0.6, y: 2.9, w: 8.8, h: 0.6, fontSize: 18, italic: true, color: "C7D2FE", valign: "top", margin: 0, isTextBox: true });
    s.addText("AutoML: Teoria e Prática  ·  Araken de Medeiros Santos", { x: 0.6, y: 3.45, w: 8.8, h: 0.3, fontSize: 12, color: "A5B4FC", margin: 0, isTextBox: true });
    const bs = await imgSize("banner" + S.num); const bw = 10, bh = bw * bs.h / bs.w;
    s.addImage({ path: FIGS + "banner" + S.num + ".png", x: 0, y: 5.625 - bh - 0.3, w: bw, h: bh, objectName: "Banner do capítulo", altText: "Ilustração do capítulo " + S.num });
    notes(s, d.notas);
  };
  T.objetivos = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo || "O que você vai aprender");
    const n = d.itens.length; const cols = 2; const rows = Math.ceil(n / cols);
    const cw = 4.4, ch = Math.min(0.95, 3.7 / rows - 0.12);
    for (let i = 0; i < n; i++) {
      const x = 0.5 + (i % cols) * (cw + 0.2), y = 1.3 + Math.floor(i / cols) * (ch + 0.12);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: "Cartão " + (i + 1) });
      s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + (ch - 0.5) / 2, w: 0.5, h: 0.5, fill: { color: C[CYCLE[i % 6]] }, line: { color: C[CYCLE[i % 6]] } });
      s.addText(String(i + 1), { x: x + 0.15, y: y + (ch - 0.5) / 2, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(d.itens[i], { x: x + 0.8, y, w: cw - 0.95, h: ch, fontSize: 15, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.roteiro = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo || "Roteiro da aula");
    const n = d.itens.length; const ch = Math.min(0.62, 3.8 / n - 0.1);
    for (let i = 0; i < n; i++) {
      const y = 1.3 + i * (ch + 0.1);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 1.0, h: ch, fill: { color: C[PC] }, line: { color: C[PC] }, rectRadius: 0.08, objectName: "Número " + (i + 1) });
      s.addText(d.itens[i][0], { x: 0.5, y, w: 1.0, h: ch, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.6, y, w: 7.9, h: ch, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08, objectName: "Faixa " + (i + 1) });
      s.addText(d.itens[i][1], { x: 1.8, y, w: 7.6, h: ch, fontSize: 17, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.conceito = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo);
    const hasFig = !!d.fig; const tw = hasFig ? (d.largura || 4.4) : 5.2;
    const nch = d.itens.join("").length; const fz = d.fonte || (nch < 200 ? 20 : nch < 320 ? 18 : 16);
    s.addText(boldRuns(d.itens, fz), { x: 0.5, y: 1.3, w: tw, h: 3.75, valign: "top", margin: 0, isTextBox: true, objectName: "Tópicos" });
    const bx = { x: 0.5 + tw + 0.3, y: 1.3, w: 9.5 - (0.5 + tw + 0.3), h: 3.75 };
    if (hasFig) {
      const sz = await imgSize(d.fig); const p = fit({ x: bx.x, y: bx.y, w: bx.w, h: d.legenda ? bx.h - 0.55 : bx.h }, sz);
      s.addImage({ path: FIGS + d.fig + ".png", ...p, objectName: "Figura", altText: d.alt || d.titulo });
      if (d.legenda) s.addText(d.legenda, { x: bx.x, y: bx.y + bx.h - 0.5, w: bx.w, h: 0.5, fontSize: 12, italic: true, color: MUTED, align: "center", valign: "middle", margin: 0, isTextBox: true });
    } else {
      const col = d.cor || PC;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx.x, y: bx.y, w: bx.w, h: bx.h, fill: { color: C[col] }, line: { color: C[col] }, rectRadius: 0.15, objectName: "Cartão de destaque" });
      await circle(s, bx.x + bx.w / 2 - 0.4, bx.y + 0.35, 0.8, "text2", d.icone || "FaLightbulb", "destaque");
      s.addText(d.frase || "", { x: bx.x + 0.25, y: bx.y + 1.3, w: bx.w - 0.5, h: bx.h - 1.5, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.figura = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo);
    const sz = await imgSize(d.fig); const p = fit({ x: 0.5, y: 1.2, w: 9, h: d.frase ? 3.1 : 3.8 }, sz);
    s.addImage({ path: FIGS + d.fig + ".png", ...p, objectName: "Figura", altText: d.alt || d.titulo });
    if (d.frase) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.0, y: 4.45, w: 8.0, h: 0.62, fill: { color: C[PC] }, line: { color: C[PC] }, rectRadius: 0.3, objectName: "Frase de impacto" });
      s.addText(d.frase, { x: 1.2, y: 4.45, w: 7.6, h: 0.62, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.analogia = async (d) => {
    const s = pres.addSlide({ masterName: "DARK" }); title(s, d.titulo || "Pense assim...", C.background1);
    await circle(s, 0.6, 1.5, 1.5, "accent4", d.icone || "FaLightbulb", "analogia");
    s.addText("ANALOGIA", { x: 2.4, y: 1.4, w: 4, h: 0.3, fontSize: 12, bold: true, color: C.accent4, charSpacing: 4, margin: 0, isTextBox: true });
    s.addText(d.texto, { x: 2.4, y: 1.75, w: 7.1, h: 1.5, fontSize: 21, italic: true, color: C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "Texto da analogia" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 3.55, w: 8.9, h: 1.35, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.12, objectName: "Tradução técnica" });
    s.addText([{ text: "Em termos técnicos: ", options: { bold: true, color: C.background1 } }, { text: d.tecnico, options: { color: C.background1 } }], { x: 0.85, y: 3.55, w: 8.4, h: 1.35, fontSize: 16, valign: "middle", margin: 0, isTextBox: true });
    notes(s, d.notas);
  };
  T.numeros = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo);
    const n = d.itens.length; const gap = 0.25; const cw = (9 - gap * (n - 1)) / n;
    d.itens.forEach((it, i) => {
      const x = 0.5 + i * (cw + gap); const col = CYCLE[i % 6];
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: cw, h: 2.5, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12, shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 }, objectName: "Cartão de número " + (i + 1) });
      s.addText(it.n, { x, y: 1.4, w: cw, h: 1.3, fontSize: n > 3 ? 40 : 52, bold: true, color: C[col], align: "center", valign: "middle", margin: 0, isTextBox: true, fontFace: "Cambria" });
      s.addText(it.r, { x: x + 0.15, y: 2.7, w: cw - 0.3, h: 1.0, fontSize: 14, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true });
    });
    if (d.frase) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 4.05, w: 9, h: 0.95, fill: { color: C[PC] }, line: { color: C[PC] }, rectRadius: 0.15, objectName: "Conclusão" });
      s.addText(d.frase, { x: 0.75, y: 4.05, w: 8.5, h: 0.95, fontSize: 16, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.comparar = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo);
    const sides = [d.esq, d.dir];
    sides.forEach((sd, i) => {
      const x = 0.5 + i * 4.6; const col = sd.cor || (i === 0 ? "accent1" : "accent3");
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.3, w: 4.4, h: 0.7, fill: { color: C[col] }, line: { color: C[col] }, rectRadius: 0.1, objectName: "Cabeçalho " + (i + 1) });
      s.addText(sd.t, { x, y: 1.3, w: 4.4, h: 0.7, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.05, w: 4.4, h: d.frase ? 2.15 : 2.9, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: "Corpo " + (i + 1) });
      s.addText(boldRuns(sd.itens, 15, C[col]), { x: x + 0.2, y: 2.15, w: 4.0, h: d.frase ? 1.95 : 2.7, valign: "top", margin: 0, isTextBox: true });
    });
    if (d.frase) {
      s.addText(d.frase, { x: 0.5, y: 4.35, w: 9, h: 0.7, fontSize: 17, bold: true, italic: true, color: C.text2, align: "center", valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.passos = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo);
    const n = d.passos.length; const gap = 0.2; const cw = (9 - gap * (n - 1)) / n;
    d.passos.forEach((p, i) => {
      const x = 0.5 + i * (cw + gap); const col = CYCLE[i % 6];
      s.addShape(pres.shapes.OVAL, { x: x + cw / 2 - 0.3, y: 1.3, w: 0.6, h: 0.6, fill: { color: C[col] }, line: { color: C[col] }, objectName: "Passo " + (i + 1) });
      s.addText(String(i + 1), { x: x + cw / 2 - 0.3, y: 1.3, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.05, w: cw, h: 2.2, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: "Cartão do passo " + (i + 1) });
      s.addText(p.t, { x: x + 0.1, y: 2.1, w: cw - 0.2, h: 0.6, fontSize: 15, bold: true, color: C[col], align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(p.d, { x: x + 0.12, y: 2.7, w: cw - 0.24, h: 1.5, fontSize: 13, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true });
    });
    if (d.frase) s.addText(d.frase, { x: 0.5, y: 4.4, w: 9, h: 0.65, fontSize: 17, bold: true, italic: true, color: C.text2, align: "center", valign: "middle", margin: 0, isTextBox: true });
    notes(s, d.notas);
  };
  T.codigo = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo);
    const cw = d.observe ? 5.6 : 9;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.3, w: cw, h: 3.75, fill: { color: C.text2 }, line: { color: C.text2 }, rectRadius: 0.08, objectName: "Quadro de código" });
    s.addText(d.codigo, { x: 0.7, y: 1.4, w: cw - 0.4, h: 3.55, fontFace: "Courier New", fontSize: d.fonte || 12, color: "E0E7FF", valign: "top", margin: 0, isTextBox: true, objectName: "Código" });
    if (d.observe) {
      s.addText("Observe", { x: 6.4, y: 1.3, w: 3.1, h: 0.4, fontSize: 18, bold: true, color: C[PC], margin: 0, isTextBox: true });
      s.addText(boldRuns(d.observe, 15), { x: 6.4, y: 1.75, w: 3.1, h: 3.3, valign: "top", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.saida = async (d) => { // código curto + saída real
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo);
    const nl = d.saida.split("\n").length; const bh = Math.min(d.frase ? 2.55 : 3.4, 0.4 + nl * 0.28);
    s.addText("SAÍDA REAL (executada na elaboração do livro)", { x: 0.5, y: 1.25, w: 9, h: 0.3, fontSize: 12, bold: true, color: C[PC], charSpacing: 2, margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.6, w: 9, h: bh, fill: { color: C.text2 }, line: { color: C.text2 }, rectRadius: 0.08, objectName: "Quadro de saída" });
    s.addText(d.saida, { x: 0.7, y: 1.7, w: 8.6, h: bh - 0.2, fontFace: "Courier New", fontSize: d.fonte || 14, color: "A7F3D0", valign: "top", margin: 0, isTextBox: true, objectName: "Saída" });
    if (d.frase) s.addText(d.frase, { x: 0.5, y: 1.6 + bh + 0.2, w: 9, h: 0.75, fontSize: 16, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
    notes(s, d.notas);
  };
  T.quiz = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo || "Quiz-relâmpago");
    s.addText(d.q, { x: 0.5, y: 1.2, w: 9, h: 1.0, fontSize: 20, bold: true, color: C.text1, valign: "top", margin: 0, isTextBox: true });
    d.op.forEach((o, i) => {
      const x = 0.5 + (i % 2) * 4.6, y = 2.35 + Math.floor(i / 2) * 1.2; const col = CYCLE[i];
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.4, h: 1.05, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: "Alternativa " + "ABCD"[i] });
      s.addShape(pres.shapes.OVAL, { x: x + 0.12, y: y + 0.27, w: 0.5, h: 0.5, fill: { color: C[col] }, line: { color: C[col] } });
      s.addText("ABCD"[i], { x: x + 0.12, y: y + 0.27, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(o, { x: x + 0.75, y, w: 3.55, h: 1.05, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    });
    notes(s, "RESPOSTA: " + "ABCD"[d.certa] + ". " + (d.expl || "") + " (Dê 30 segundos para a turma pensar antes de revelar.)");
  };
  T.memorize = async (d) => {
    const s = pres.addSlide({ masterName: "DARK" }); title(s, d.titulo || "Para levar na mochila", C.background1);
    const n = d.itens.length; const cols = 2; const rows = Math.ceil(n / cols); const ch = Math.min(0.9, 3.75 / rows - 0.1);
    for (let i = 0; i < n; i++) {
      const x = 0.5 + (i % cols) * 4.6, y = 1.25 + Math.floor(i / cols) * (ch + 0.1);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.4, h: ch, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.1, objectName: "Cartão " + (i + 1) });
      s.addImage({ data: await icon("FaCheckCircle", "FFFFFF"), x: x + 0.15, y: y + ch / 2 - 0.17, w: 0.34, h: 0.34, objectName: "Ícone " + (i + 1), altText: "marcador" });
      s.addText(d.itens[i], { x: x + 0.62, y, w: 3.65, h: ch, fontSize: 13, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.pratica = async (d) => {
    const s = pres.addSlide({ masterName: "LIGHT" }); title(s, d.titulo || "Mão na massa");
    const n = d.itens.length; const ch = Math.min(0.95, 3.75 / n - 0.1);
    for (let i = 0; i < n; i++) {
      const y = 1.25 + i * (ch + 0.1); const col = CYCLE[i % 6];
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 9, h: ch, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1, objectName: "Faixa " + (i + 1) });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.65, y: y + ch / 2 - 0.2, w: 1.25, h: 0.4, fill: { color: C[col] }, line: { color: C[col] }, rectRadius: 0.2 });
      s.addText(d.itens[i].tag, { x: 0.65, y: y + ch / 2 - 0.2, w: 1.25, h: 0.4, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText([{ text: d.itens[i].h + "  ", options: { bold: true, color: C.text2 } }, { text: d.itens[i].d, options: { color: C.text1 } }], { x: 2.05, y, w: 7.3, h: ch, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };
  T.fim = async (d) => {
    const s = pres.addSlide({ masterName: "DARK" });
    await circle(s, 4.5, 0.7, 1.0, "accent4", "FaFlagCheckered", "fim");
    s.addText(d.frase, { x: 0.8, y: 1.9, w: 8.4, h: 1.7, fontSize: 30, bold: true, color: C.background1, fontFace: "Cambria", align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Frase final" });
    if (d.proximo) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.5, y: 3.95, w: 7.0, h: 0.85, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.15, objectName: "Próximo capítulo" });
      s.addText([{ text: "A seguir: ", options: { bold: true } }, { text: d.proximo }], { x: 1.7, y: 3.95, w: 6.6, h: 0.85, fontSize: 16, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    }
    notes(s, d.notas);
  };

  for (const sl of S.slides) { if (!T[sl.tipo]) throw new Error("tipo desconhecido: " + sl.tipo); await T[sl.tipo](sl); }
  await pres.writeFile({ fileName: outPath });
  await applyTheme(outPath, THEME);
  // remove pPr "buNone" que o pptxgenjs insere no meio de um parágrafo (após uma run) e que apaga o marcador
  const JSZip = require("jszip"); const zip = await JSZip.loadAsync(fs.readFileSync(outPath));
  for (const f of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    const x = await zip.file(f).async("string"); const y = x.replace(/<\/a:r><a:pPr[^>]*><a:buNone\/><\/a:pPr>/g, "</a:r>");
    if (x !== y) zip.file(f, y);
  }
  fs.writeFileSync(outPath, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  const [spec, out] = process.argv.slice(2);
  await buildDeck(spec, out);
  console.log("ok", out);
})().catch((e) => { console.error(e); process.exit(1); });
