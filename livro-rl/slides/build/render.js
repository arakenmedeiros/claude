const pptxgen = require("pptxgenjs");
const fs = require("fs");
const { applyTheme } = require("/root/.claude/skills/synced/4cfcf78b-3b5e-47b6-8db8-09e9433975ff_d303de2b-5e35-4489-a93a-f157e923a1fb/pptx/scripts/apply_theme.js");
const ROOT = "/home/user/claude/livro-rl";
const THEME = { name: "Aprendizado por Reforço", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B1F3B", lt1: "FFFFFF", dk2: "14173F", lt2: "F2F3FB", accent1: "F59E0B", accent2: "12A89A", accent3: "6C5CE7", accent4: "E84A8A", accent5: "3B82F6", accent6: "94A3B8", hlink: "3B82F6", folHlink: "6C5CE7" } };
const HEX = { navy: "14173F", ink: "1B1F3B", soft: "F2F3FB", orange: "F59E0B", teal: "12A89A", violet: "6C5CE7", pink: "E84A8A", muted: "5B6180" };
const AUTHOR = "Araken de Medeiros Santos";

function est(text, w, fs) { const cpl = w / (fs * 0.0057); return Math.max(1, Math.ceil(text.length / cpl)); }

async function build(num) {
  const slides = JSON.parse(fs.readFileSync(`${__dirname}/ch${String(num).padStart(2, "0")}.json`, "utf8"));
  const pres = new pptxgen(); pres.layout = "LAYOUT_16x9"; pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
  pres.author = AUTHOR; pres.title = `Aprendizado por Reforço — Capítulo ${num}`;
  const C = pres.SchemeColor;
  const foot = (c) => ({ text: { text: `Aprendizado por Reforço · ${AUTHOR}`, options: { x: 0.5, y: 5.25, w: 6, h: 0.3, fontSize: 10, color: c, fontFace: "Calibri", margin: 0 } } });
  pres.defineSlideMaster({ title: "CLARO", background: { color: C.background1 }, objects: [foot(C.accent6)], slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" } });
  pres.defineSlideMaster({ title: "ESCURO", background: { color: C.text2 }, objects: [foot(C.accent6)], slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" } });
  const accents = { teal: C.accent2, orange: C.accent1, violet: C.accent3, pink: C.accent4 };

  const seen = new Set();
  function newSlide(dark, sec, notes) {
    if (sec && !seen.has(sec)) { seen.add(sec); pres.addSection({ title: sec }); }
    const s = pres.addSlide({ masterName: dark ? "ESCURO" : "CLARO", sectionTitle: sec });
    if (notes) s.addNotes(notes);
    return s;
  }
  function header(s, title, chip, chipColor) {
    s.addText(title, { x: 0.5, y: 0.3, w: chip ? 7.3 : 9, h: 0.85, fontFace: "Cambria", bold: true, fontSize: title.length > 48 ? 24 : 30, color: C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: "Título", fit: "shrink" });
    if (chip) s.addText(chip, { x: 7.9, y: 0.45, w: 1.6, h: 0.4, shape: pres.ShapeType.roundRect, rectRadius: 0.2, fill: { color: chipColor || C.accent3 }, color: C.background1, bold: true, fontSize: 12, align: "center", valign: "middle", margin: 0, objectName: "Etiqueta" });
  }
  function cards(s, items, color, y0 = 1.3, y1 = 5.1, fsz0) {
    const fsz = fsz0 || (items.length <= 3 ? 18 : 16);
    const w = 9.0, gap = 0.12; const hs = items.map(t => { const sub = t.startsWith("– "); return Math.max(0.52, est(t, (sub ? w - 1.0 : w - 0.95), fsz) * 0.26 + 0.24); });
    let tot = hs.reduce((a, b) => a + b, 0) + gap * (items.length - 1), sc = Math.min(1, (y1 - y0) / tot);
    let y = y0; let n = 0;
    items.forEach((t, i) => {
      const sub = t.startsWith("– "); const txt = sub ? t.slice(2) : t; const h = hs[i] * (sc < 1 ? sc : 1);
      const x = sub ? 0.95 : 0.5; const cw = w - (sub ? 0.45 : 0);
      s.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h, fill: { color: sub ? C.background1 : C.background2 }, line: { color: C.background2, width: 1 }, rectRadius: 0.12, objectName: "Cartão" });
      if (!sub) { n++; s.addText(String(n), { x: x + 0.15, y: y + h / 2 - 0.2, w: 0.4, h: 0.4, shape: pres.ShapeType.ellipse, fill: { color: color }, color: C.background1, bold: true, fontSize: 13, align: "center", valign: "middle", margin: 0, objectName: "Número" }); }
      else s.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: y + h / 2 - 0.06, w: 0.12, h: 0.12, fill: { color: color }, line: { color: color, width: 0 }, objectName: "Marcador" });
      s.addText(txt, { x: x + 0.7, y, w: cw - 0.85, h, fontSize: fsz, color: C.text1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink", objectName: "Texto" });
      y += h + gap;
    });
  }
  const chap = `Cap. ${num}`;
  let sec = null;
  for (const d of slides) {
    if (d.t === "title") {
      const s = newSlide(true, "Abertura", `Abertura do Capítulo ${num}. Apresente a pergunta central do capítulo e o roteiro da aula.`);
      s.addImage({ path: d.banner, x: 0, y: 0.35, w: 10, h: 10 * 248 / 1317, objectName: "Banner" });
      s.addText(d.title, { x: 0.6, y: 2.55, w: 8.8, h: 1.5, fontFace: "Cambria", bold: true, fontSize: 38, color: C.background1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink", objectName: "Título" });
      s.addText("Aprendizado por Reforço", { x: 0.6, y: 4.15, w: 6, h: 0.4, fontSize: 18, color: C.accent1, bold: true, margin: 0, isTextBox: true });
      s.addText(AUTHOR, { x: 0.6, y: 4.55, w: 6, h: 0.4, fontSize: 16, color: C.accent6, margin: 0, isTextBox: true });
    } else if (d.t === "obj") {
      sec = "Início";
      const s = newSlide(false, sec, "Leia os objetivos com a turma e pergunte quem já conhece algum deles.");
      header(s, "O que você vai aprender", chap, C.accent2); cards(s, d.items, C.accent2);
    } else if (d.t === "roteiro") {
      const s = newSlide(false, sec, "Mostre o mapa da aula: cada item é uma estação da viagem.");
      header(s, "Roteiro da aula", chap, C.accent2);
      d.items.forEach((t, i) => { const col = i % 2, row = Math.floor(i / 2); const x = 0.5 + col * 4.6, y = 1.35 + row * 0.62;
        s.addShape(pres.ShapeType.roundRect, { x, y, w: 4.4, h: 0.52, fill: { color: C.background2 }, line: { color: C.background2, width: 1 }, rectRadius: 0.1, objectName: "Cartão" });
        s.addText(String(i + 1), { x: x + 0.1, y: y + 0.08, w: 0.36, h: 0.36, shape: pres.ShapeType.ellipse, fill: { color: [C.accent3, C.accent2, C.accent1, C.accent4][i % 4] }, color: C.background1, bold: true, fontSize: 12, align: "center", valign: "middle", margin: 0 });
        s.addText(t, { x: x + 0.6, y, w: 3.7, h: 0.52, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink" }); });
    } else if (d.t === "section") {
      sec = d.title; 
      const s = newSlide(true, sec, "");
      s.addShape(pres.ShapeType.ellipse, { x: 7.2, y: 0.9, w: 2.6, h: 2.6, fill: { color: C.accent3, transparency: 30 }, line: { color: C.accent3, width: 0 }, objectName: "Círculo" });
      s.addShape(pres.ShapeType.ellipse, { x: 6.5, y: 2.6, w: 1.4, h: 1.4, fill: { color: C.accent1, transparency: 20 }, line: { color: C.accent1, width: 0 }, objectName: "Círculo" });
      s.addText(d.title, { x: 0.6, y: 1.6, w: 6.2, h: 1.6, fontFace: "Cambria", bold: true, fontSize: 38, color: C.background1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink", objectName: "Título" });
      if (d.sub) s.addText(d.sub, { x: 0.6, y: 3.3, w: 6.2, h: 0.8, fontSize: 18, color: C.accent1, margin: 0, isTextBox: true, valign: "top", fit: "shrink" });
    } else if (d.t === "bullets" || d.t === "resumo") {
      const resumo = d.t === "resumo";
      if (resumo && sec !== "Resumo") { sec = "Resumo" }
      const s = newSlide(false, sec, d.notes ? "Roteiro do professor: " + d.notes : "");
      header(s, d.title, d.num ? `Seção ${d.num}` : chap, resumo ? C.accent1 : C.accent3);
      cards(s, d.items, resumo ? C.accent1 : C.accent3);
    } else if (d.t === "cards") {
      const s = newSlide(false, sec, d.notes); header(s, d.title, chap, C.accent2);
      const n = d.items.length, w = (9 - 0.3 * (n - 1)) / n;
      d.items.forEach((t, i) => { const x = 0.5 + i * (w + 0.3); const col = [C.accent3, C.accent2, C.accent1, C.accent4][i % 4];
        s.addShape(pres.ShapeType.roundRect, { x, y: 1.5, w, h: 3.2, fill: { color: C.background2 }, line: { color: C.background2, width: 1 }, rectRadius: 0.15, objectName: "Cartão" });
        s.addText(String(i + 1), { x: x + w / 2 - 0.35, y: 1.75, w: 0.7, h: 0.7, shape: pres.ShapeType.ellipse, fill: { color: col }, color: C.background1, bold: true, fontSize: 22, align: "center", valign: "middle", margin: 0 });
        s.addText(t, { x: x + 0.2, y: 2.65, w: w - 0.4, h: 1.8, fontSize: 16, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true, fit: "shrink" }); });
    } else if (d.t === "fig") {
      const s = newSlide(false, sec, d.notes ? "Descreva a figura: " + d.notes : "");
      header(s, d.title, "Figura", C.accent4);
      const bw = 9, bh = 3.15, r = d.fig.w / d.fig.h; let w = bw, h = w / r; if (h > bh) { h = bh; w = h * r; }
      s.addImage({ path: d.fig.path, x: 0.5 + (bw - w) / 2, y: 1.25 + (bh - h) / 2, w, h, altText: d.cap, objectName: "Figura" });
      s.addText(d.cap, { x: 0.5, y: 4.5, w: 9, h: 0.65, fontSize: 14, italic: true, color: C.accent6 === "" ? C.text1 : C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, fit: "shrink", objectName: "Legenda" });
    } else if (d.t === "formula") {
      const s = newSlide(false, sec, "Leia a fórmula em voz alta, termo a termo, e traduza com a frase \"em palavras\".");
      header(s, d.title, "Fórmula", C.accent1);
      s.addText(d.formula, { x: 0.8, y: 1.45, w: 8.4, h: 1.4, shape: pres.ShapeType.roundRect, rectRadius: 0.2, fill: { color: C.text2 }, color: C.background1, fontFace: "Cambria", fontSize: d.formula.length > 55 ? 22 : 30, align: "center", valign: "middle", margin: 0.15, fit: "shrink", objectName: "Fórmula" });
      s.addText("EM PALAVRAS", { x: 0.8, y: 3.1, w: 1.6, h: 0.32, shape: pres.ShapeType.roundRect, rectRadius: 0.16, fill: { color: C.accent2 }, color: C.background1, bold: true, fontSize: 11, align: "center", valign: "middle", margin: 0, objectName: "Etiqueta" });
      s.addText(d.words.replace(/^Em palavras:?\s*/, ""), { x: 0.8, y: 3.5, w: 8.4, h: 1.5, fontSize: 16, color: C.text1, valign: "top", margin: 0, isTextBox: true, fit: "shrink", objectName: "Texto" });
    } else if (d.t === "cotidiano") {
      const col = d.head === "Atenção" ? C.accent4 : d.head === "Exemplo" ? C.accent3 : C.accent1;
      const s = newSlide(false, sec, "Peça aos alunos que citem outro exemplo do próprio cotidiano.");
      header(s, d.title, d.head, col); cards(s, d.items, col);
    } else if (d.t === "quote") {
      const s = newSlide(true, sec, "Peça que a turma repita a frase em voz alta e com as próprias palavras.");
      s.addText("MEMORIZE", { x: 0.6, y: 0.7, w: 1.6, h: 0.4, shape: pres.ShapeType.roundRect, rectRadius: 0.2, fill: { color: C.accent1 }, color: C.text2, bold: true, fontSize: 13, align: "center", valign: "middle", margin: 0, objectName: "Etiqueta" });
      s.addText(d.text, { x: 0.8, y: 1.4, w: 8.4, h: 3.2, fontFace: "Cambria", italic: true, fontSize: 28, color: C.background1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink", objectName: "Frase" });
    } else if (d.t === "table") {
      const s = newSlide(false, sec, "Compare linha a linha; peça que a turma justifique cada \"cuidado\".");
      header(s, d.title, "Quadro", C.accent5);
      const rows = d.rows.map((r, i) => r.map(c => ({ text: c, options: i === 0 ? { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } : { color: C.text1, fill: { color: i % 2 ? C.background2 : C.background1 }, fontSize: 13 } })));
      s.addTable(rows, { x: 0.5, y: 1.35, w: 9, colW: Array(d.rows[0].length).fill(9 / d.rows[0].length), border: { type: "solid", pt: 0.5, color: C.accent6 }, valign: "middle", margin: 0.08, objectName: "Tabela" });
    } else if (d.t === "quiz") {
      const s = newSlide(false, "Quiz", `Resposta: ${d.ans}. ${d.exp}`);
      header(s, `Pergunta ${d.n}`, "Quiz", C.accent4);
      s.addText(d.q, { x: 0.5, y: 1.15, w: 9, h: 0.95, fontSize: 18, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true, fit: "shrink", objectName: "Pergunta" });
      const L = "abcd"; const oy = 2.15, oh = 0.7;
      d.opts.forEach((o, i) => { const y = oy + i * (oh + 0.08);
        s.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 9, h: oh, fill: { color: C.background2 }, line: { color: C.background2, width: 1 }, rectRadius: 0.1, objectName: "Cartão" });
        s.addText(L[i].toUpperCase(), { x: 0.65, y: y + 0.15, w: 0.4, h: 0.4, shape: pres.ShapeType.ellipse, fill: { color: [C.accent3, C.accent2, C.accent1, C.accent4][i] }, color: C.background1, bold: true, fontSize: 13, align: "center", valign: "middle", margin: 0 });
        s.addText(o, { x: 1.2, y, w: 8.1, h: oh, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true, fit: "shrink" }); });
    } else if (d.t === "gabarito") {
      const s = newSlide(false, "Quiz", ""); header(s, "Respostas do quiz", "Quiz", C.accent4); cards(s, d.items, C.accent4);
    } else if (d.t === "discuss") {
      const s = newSlide(false, "Quiz", "Divida a turma em grupos de 3 a 4 pessoas; 5 minutos de discussão e depois compartilhamento.");
      header(s, "Para discutir em grupo", "Debate", C.accent2); cards(s, d.items, C.accent2);
    } else if (d.t === "fim") {
      const s = newSlide(true, "Encerramento", "Encerre retomando a pergunta central e antecipando o próximo capítulo.");
      s.addShape(pres.ShapeType.ellipse, { x: 6.8, y: 0.6, w: 3.0, h: 3.0, fill: { color: C.accent3, transparency: 35 }, line: { color: C.accent3, width: 0 }, objectName: "Círculo" });
      s.addText("Até a próxima!", { x: 0.6, y: 1.4, w: 6.2, h: 1.0, fontFace: "Cambria", bold: true, fontSize: 40, color: C.background1, margin: 0, isTextBox: true });
      s.addText(`Capítulo ${num}: ${d.title}`, { x: 0.6, y: 2.5, w: 6.2, h: 1.0, fontSize: 18, color: C.accent1, margin: 0, isTextBox: true, valign: "top", fit: "shrink" });
      s.addText("Pratique: refaça os códigos, resolva as questões e faça o projeto do capítulo.", { x: 0.6, y: 3.6, w: 6.2, h: 0.8, fontSize: 16, color: C.accent6, margin: 0, isTextBox: true, valign: "top", fit: "shrink" });
    }
  }
  const out = `${ROOT}/slides/cap${String(num).padStart(2, "0")}_slides.pptx`;
  await pres.writeFile({ fileName: out }); await applyTheme(out, THEME); return out;
}
(async () => { const ns = process.argv.slice(2).map(Number); for (const n of ns) console.log(await build(n)); })();
