// Gera um conjunto de slides (pptx) por capítulo do livro "Aprendizado Federado".
// Uso: NODE_PATH=<node_modules com pptxgenjs, react-icons, sharp> node build_slides.js [n ...]
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const SKILL = fs.readdirSync("/root/.claude/skills/synced").map((d) => path.join("/root/.claude/skills/synced", d, "pptx")).find((p) => fs.existsSync(p));
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const ROOT = path.join(__dirname, "..");
const CAP = path.join(ROOT, "cap");
const FIGS = path.join(ROOT, "figs");
const OUT = __dirname;

const THEME = {
  name: "Aprendizado Federado",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: { dk1: "1F2937", lt1: "FFFFFF", dk2: "111827", lt2: "F3F4F6", accent1: "4F46E5", accent2: "06B6D4", accent3: "10B981", accent4: "F59E0B", accent5: "EF4444", accent6: "8B5CF6", hlink: "4F46E5", folHlink: "8B5CF6" },
};
const HEX = { ind: "4F46E5", indD: "312E81", navy: "111827", cya: "06B6D4", grn: "10B981", amb: "F59E0B", red: "EF4444", vio: "8B5CF6", ink: "1F2937", gry: "6B7280", light: "F3F4F6" };
const PART_COL = [HEX.ind, HEX.ind, HEX.ind, HEX.cya, HEX.cya, HEX.cya, HEX.red, HEX.red, HEX.red, HEX.amb, HEX.amb, HEX.vio];
const PART_NOME = ["Parte I — Fundamentos", "Parte I — Fundamentos", "Parte I — Fundamentos", "Parte II — Desafios do treino", "Parte II — Desafios do treino", "Parte II — Desafios do treino", "Parte III — Privacidade e segurança", "Parte III — Privacidade e segurança", "Parte III — Privacidade e segurança", "Parte IV — Arquiteturas e fronteiras", "Parte IV — Arquiteturas e fronteiras", "Parte V — Projeto final"];

// ---------------------------------------------------------------- utilidades
const clean = (s) => s.replace(/\*\*/g, "").replace(/\*/g, "").replace(/`/g, "").replace(/\s+/g, " ").trim();
const nn = (n) => String(n).padStart(2, "0");
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
async function icon(name, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function parseChapter(n) {
  const md = fs.readFileSync(path.join(CAP, `ch${nn(n)}.md`), "utf8");
  const titulo = (md.match(/^# (.+)$/m) || [])[1];
  const objetivos = [...md.matchAll(/^@obj: (.+)$/gm)].map((m) => clean(m[1]));
  const mem = md.match(/:::memorize (?:O essencial|Para levar)[^\n]*\n([\s\S]*?)\n:::/);
  const resumo = mem ? mem[1].split("\n").filter((l) => l.startsWith("- ")).map((l) => clean(l.slice(2))) : [];
  const ex = fs.readFileSync(path.join(CAP, `ex${nn(n)}.txt`), "utf8").trim().split(/\n\s*\n/);
  const mc = ex.filter((b) => b.startsWith("MC:")).map((b) => {
    const L = b.split("\n");
    const r = L[5].match(/^R:\s*([abcd])\s*\|\s*(.*)$/);
    return { q: clean(L[0].slice(3)), ops: [1, 2, 3, 4].map((i) => clean(L[i].replace(/^[abcd]\)\s*/, ""))), ok: "abcd".indexOf(r[1]), why: clean(r[2]) };
  });
  let exs = [], proj = null;
  const pf = path.join(CAP, "pr", `pr${nn(n)}.txt`);
  if (fs.existsSync(pf)) {
    for (const ln of fs.readFileSync(pf, "utf8").split("\n")) {
      if (ln.startsWith("@@EX ")) { const p = ln.slice(5).split("|").map((x) => x.trim()); exs.push({ t: p[1], nivel: p[2] }); }
      if (ln.startsWith("@@PROJ ")) { proj = ln.slice(7).split("|")[0].trim(); }
    }
  }
  return { titulo: clean(titulo), objetivos, resumo, mc, exs, proj };
}

// ---------------------------------------------------------------- construção de um deck
async function buildDeck(n) {
  const spec = require(path.join(__dirname, "content", `ch${nn(n)}.js`));
  const info = parseChapter(n);
  const col = PART_COL[n - 1];
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.title = `Cap. ${n} — ${info.titulo}`;
  pres.author = "Araken de Medeiros Santos";
  pres.subject = "Aprendizado Federado: Fundamentos, Prática e Fronteiras";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const rodape = `Aprendizado Federado · Cap. ${n}`;

  pres.defineSlideMaster({
    title: "CONTEUDO", background: { color: "FFFFFF" },
    objects: [
      { text: { text: rodape, options: { x: 0.5, y: 5.25, w: 5, h: 0.25, fontSize: 10, color: HEX.gry, margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.85, fontSize: 32, bold: true, color: C.text2, fontFace: "Calibri", align: "left", valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: HEX.gry, align: "right" },
  });
  pres.defineSlideMaster({ title: "CAPA", background: { color: HEX.navy }, objects: [] });
  pres.defineSlideMaster({
    title: "ESCURO", background: { color: HEX.navy },
    objects: [
      { text: { text: rodape, options: { x: 0.5, y: 5.25, w: 5, h: 0.25, fontSize: 10, color: "9CA3AF", margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.85, fontSize: 32, bold: true, color: C.background1, fontFace: "Calibri", align: "left", valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: "9CA3AF", align: "right" },
  });

  const iBulb = await icon("FaLightbulb", HEX.navy);
  const iCheck = await icon("FaCheckCircle", "FFFFFF");
  const iPlay = await icon("FaLaptopCode", "FFFFFF");
  const iQ = await icon("FaQuestion", "FFFFFF");
  const iBook = await icon("FaBookOpen", "FFFFFF");
  const iFlag = await icon("FaFlagCheckered", "FFFFFF");

  const nova = (secao, escuro = false) => pres.addSlide({ masterName: escuro ? "ESCURO" : "CONTEUDO", sectionTitle: secao });
  const titulo = (s, t) => s.addText(t, { placeholder: "title" });
  const circulo = (s, x, y, d, cor, img) => { s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: cor }, line: { color: cor, width: 0 }, objectName: "icone-fundo" }); if (img) s.addImage({ data: img, x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, objectName: "icone" }); };
  const figura = (s, id, x, y, w, h) => {
    const f = path.join(FIGS, id + ".png");
    if (!fs.existsSync(f)) throw new Error("figura ausente: " + id);
    const { w: pw, h: ph } = pngSize(f);
    let dw = w, dh = (w * ph) / pw;
    if (dh > h) { dh = h; dw = (h * pw) / ph; }
    s.addImage({ path: f, x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh, objectName: "figura " + id, altText: id });
  };
  const cartaoAnalogia = (s, texto, x, y, w, h) => {
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: "FEF3C7" }, line: { color: "FEF3C7", width: 0 }, rectRadius: 0.12, objectName: "cartao analogia" });
    circulo(s, x + 0.15, y + (h - 0.5) / 2, 0.5, HEX.amb, iBulb);
    s.addText([{ text: "Pense assim: ", options: { bold: true, color: HEX.navy } }, { text: texto, options: { color: HEX.navy, italic: true } }], { x: x + 0.8, y: y + 0.04, w: w - 0.95, h: h - 0.08, fontSize: 14, valign: "middle", margin: 0, isTextBox: true, objectName: "analogia" });
  };
  const bullets = (s, itens, x, y, w, h, tam = 16) => {
    const runs = itens.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < itens.length - 1, paraSpaceAfter: 8 } }));
    s.addText(runs, { x, y, w, h, fontSize: tam, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "topicos" });
  };

  // ---- CAPA
  pres.addSection({ title: "Abertura" });
  {
    const s = pres.addSlide({ masterName: "CAPA", sectionTitle: "Abertura" });
    const banner = path.join(FIGS, `banner${n}.png`);
    const { w: bw, h: bh } = pngSize(banner);
    s.addImage({ path: banner, x: 0, y: 0, w: 10, h: (10 * bh) / bw, objectName: "faixa do capitulo", altText: "Faixa do capítulo " + n });
    s.addText(PART_NOME[n - 1].toUpperCase(), { x: 0.6, y: 2.2, w: 8.8, h: 0.4, fontSize: 14, bold: true, color: col === HEX.ind ? "A5B4FC" : "FDE68A", charSpacing: 3, margin: 0, isTextBox: true, objectName: "parte" });
    s.addText(info.titulo, { x: 0.6, y: 2.65, w: 8.8, h: 1.6, fontSize: 34, bold: true, color: "FFFFFF", valign: "top", margin: 0, isTextBox: true, objectName: "titulo do capitulo" });
    s.addText(spec.gancho, { x: 0.6, y: 4.3, w: 8.8, h: 0.6, fontSize: 18, italic: true, color: "67E8F9", margin: 0, isTextBox: true, objectName: "gancho" });
    s.addText("Araken de Medeiros Santos  ·  Aprendizado Federado: Fundamentos, Prática e Fronteiras", { x: 0.6, y: 5.1, w: 8.8, h: 0.3, fontSize: 11, color: "9CA3AF", margin: 0, isTextBox: true, objectName: "autor" });
    s.addNotes(spec.notaCapa || "Abertura. Leia o gancho em voz alta e pergunte à turma o que esperam aprender neste capítulo.");
  }
  // ---- OBJETIVOS
  {
    const s = nova("Abertura");
    titulo(s, "O que você vai conseguir fazer");
    info.objetivos.slice(0, 6).forEach((o, i) => {
      const y = 1.4 + i * 0.62;
      circulo(s, 0.5, y, 0.44, col, iCheck);
      s.addText(o, { x: 1.15, y: y - 0.04, w: 8.3, h: 0.52, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "objetivo " + (i + 1) });
    });
    s.addNotes("Peça que a turma leia os objetivos em silêncio e marque o que já sabe. No fim da aula, volte a esta lista.");
  }
  // ---- ROTEIRO (agenda)
  if (spec.roteiro) {
    const s = nova("Abertura");
    titulo(s, "Roteiro da aula");
    const itens = spec.roteiro;
    const w = 9 / itens.length;
    itens.forEach((it, i) => {
      const x = 0.5 + i * w;
      s.addShape(pres.ShapeType.roundRect, { x: x + 0.05, y: 1.7, w: w - 0.1, h: 2.0, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.1, shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 90, opacity: 0.15 }, objectName: "etapa " + (i + 1) });
      circulo(s, x + w / 2 - 0.25, 1.9, 0.5, col);
      s.addText(String(i + 1), { x: x + w / 2 - 0.25, y: 1.9, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "numero " + (i + 1) });
      s.addText(it, { x: x + 0.15, y: 2.55, w: w - 0.3, h: 1.0, fontSize: 15, bold: true, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "titulo etapa " + (i + 1) });
    });
    s.addNotes("Mostre o caminho da aula. Cada bloco leva de 8 a 12 minutos.");
  }

  // ---- CONTEÚDO
  pres.addSection({ title: "Conteúdo" });
  for (const sl of spec.slides) {
    const escuro = sl.k === "impacto";
    const s = nova("Conteúdo", escuro);
    if (sl.k === "c") {
      titulo(s, sl.t);
      if (sl.f) {
        bullets(s, sl.b, 0.5, 1.4, 4.7, sl.a ? 2.9 : 3.6, 17);
        figura(s, sl.f, 5.4, 1.35, 4.1, 3.0);
      } else {
        bullets(s, sl.b, 0.5, 1.5, 9, sl.a ? 2.8 : 3.5, 22);
      }
      if (sl.a) cartaoAnalogia(s, sl.a, 0.5, 4.42, 9, 0.75);
    } else if (sl.k === "f") {
      titulo(s, sl.t);
      figura(s, sl.f, 0.5, 1.3, 9, 3.3);
      s.addText(sl.leg, { x: 0.5, y: 4.65, w: 9, h: 0.5, fontSize: 14, italic: true, color: HEX.ind, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "legenda" });
    } else if (sl.k === "t") {
      titulo(s, sl.t);
      const head = sl.h.map((x) => ({ text: x, options: { bold: true, color: "FFFFFF", fill: { color: col }, fontSize: 14 } }));
      const rows = [head, ...sl.r.map((r, i) => r.map((c, j) => ({ text: c, options: { fontSize: 14, color: C.text1, bold: j === 0, fill: { color: i % 2 ? "FFFFFF" : "F3F4F6" } } })))];
      s.addTable(rows, { x: 0.5, y: 1.4, w: 9, colW: sl.cw || sl.h.map(() => 9 / sl.h.length), border: { type: "solid", color: "E5E7EB", pt: 1 }, margin: [0.06, 0.1, 0.06, 0.1], valign: "middle", autoPage: false, objectName: "tabela" });
      if (sl.a) cartaoAnalogia(s, sl.a, 0.5, 4.42, 9, 0.75);
    } else if (sl.k === "code") {
      titulo(s, sl.t);
      s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.35, w: sl.saida ? 5.6 : 9, h: 3.1, fill: { color: "0F172A" }, line: { color: "0F172A", width: 0 }, rectRadius: 0.08, objectName: "fundo codigo" });
      s.addText(sl.code, { x: 0.65, y: 1.45, w: sl.saida ? 5.3 : 8.7, h: 2.9, fontFace: "Courier New", fontSize: 12, color: "E2E8F0", valign: "top", margin: 0, isTextBox: true, objectName: "codigo" });
      if (sl.saida) {
        s.addShape(pres.ShapeType.roundRect, { x: 6.3, y: 1.35, w: 3.2, h: 3.1, fill: { color: "ECFDF5" }, line: { color: "ECFDF5", width: 0 }, rectRadius: 0.08, objectName: "fundo saida" });
        s.addText([{ text: "O que aparece", options: { bold: true, color: HEX.grn, fontFace: "Calibri", fontSize: 12, breakLine: true } }, { text: sl.saida, options: { fontFace: "Courier New", fontSize: 11, color: HEX.navy } }], { x: 6.4, y: 1.42, w: 3.0, h: 2.95, valign: "top", margin: 0, isTextBox: true, objectName: "saida" });
      }
      if (sl.nota) s.addText(sl.nota, { x: 0.5, y: 4.55, w: 9, h: 0.6, fontSize: 14, italic: true, color: HEX.ind, valign: "middle", margin: 0, isTextBox: true, objectName: "nota codigo" });
    } else if (sl.k === "num") {
      titulo(s, sl.t);
      const m = sl.itens.length, w = (9 - 0.3 * (m - 1)) / m;
      sl.itens.forEach(([v, rot], i) => {
        const x = 0.5 + i * (w + 0.3);
        s.addShape(pres.ShapeType.roundRect, { x, y: 1.5, w, h: 2.5, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.1, shadow: { type: "outer", color: "000000", blur: 4, offset: 1, angle: 90, opacity: 0.15 }, objectName: "cartao numero " + (i + 1) });
        s.addText(v, { x, y: 1.7, w, h: 1.1, fontSize: 48, bold: true, color: col, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "numero " + (i + 1) });
        s.addText(rot, { x: x + 0.15, y: 2.85, w: w - 0.3, h: 1.0, fontSize: 14, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "rotulo " + (i + 1) });
      });
      if (sl.a) cartaoAnalogia(s, sl.a, 0.5, 4.3, 9, 0.8);
    } else if (sl.k === "fase") {
      titulo(s, sl.t);
      const m = sl.passos.length, w = (9 - 0.35 * (m - 1)) / m;
      sl.passos.forEach(([ti, de], i) => {
        const x = 0.5 + i * (w + 0.35);
        s.addShape(pres.ShapeType.roundRect, { x, y: 1.45, w, h: 2.75, fill: { color: "FFFFFF" }, line: { color: col, width: 1.5 }, rectRadius: 0.1, objectName: "passo " + (i + 1) });
        circulo(s, x + w / 2 - 0.25, 1.6, 0.5, col);
        s.addText(String(i + 1), { x: x + w / 2 - 0.25, y: 1.6, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "num passo " + (i + 1) });
        s.addText(ti, { x: x + 0.1, y: 2.2, w: w - 0.2, h: 0.55, fontSize: 16, bold: true, color: C.text2, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "titulo passo " + (i + 1) });
        s.addText(de, { x: x + 0.12, y: 2.8, w: w - 0.24, h: 1.3, fontSize: 14, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: "texto passo " + (i + 1) });
        if (i < m - 1) s.addText("➜", { x: x + w, y: 2.55, w: 0.35, h: 0.4, fontSize: 18, bold: true, color: col, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "seta " + (i + 1) });
      });
      if (sl.a) cartaoAnalogia(s, sl.a, 0.5, 4.42, 9, 0.75);
    } else if (sl.k === "cmp") {
      titulo(s, sl.t);
      [[sl.esq, 0.5, HEX.ind], [sl.dir, 5.2, HEX.grn]].forEach(([blk, x, cor]) => {
        s.addShape(pres.ShapeType.roundRect, { x, y: 1.4, w: 4.3, h: 3.0, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.1, objectName: "coluna " + blk[0] });
        s.addText(blk[0], { x: x + 0.2, y: 1.5, w: 3.9, h: 0.5, fontSize: 20, bold: true, color: cor, margin: 0, valign: "middle", isTextBox: true, objectName: "cabecalho " + blk[0] });
        bullets(s, blk[1], x + 0.2, 2.1, 3.9, 2.2, 15);
      });
      if (sl.a) cartaoAnalogia(s, sl.a, 0.5, 4.5, 9, 0.7);
    } else if (sl.k === "impacto") {
      s.addText(sl.frase, { x: 0.8, y: 1.3, w: 8.4, h: 2.4, fontSize: 38, bold: true, color: "FFFFFF", align: "left", valign: "middle", margin: 0, isTextBox: true, objectName: "frase de impacto" });
      if (sl.sub) s.addText(sl.sub, { x: 0.8, y: 3.8, w: 8.4, h: 0.9, fontSize: 18, italic: true, color: "67E8F9", valign: "top", margin: 0, isTextBox: true, objectName: "complemento" });
    }
    s.addNotes(sl.n || "Explique com suas palavras e peça um exemplo à turma antes de avançar.");
  }

  // ---- QUIZ RELÂMPAGO
  pres.addSection({ title: "Prática e fechamento" });
  const quizIdx = spec.quiz || [3, 13, 23];
  quizIdx.forEach((qi, k) => {
    const q = info.mc[qi];
    if (!q) return;
    const s = nova("Prática e fechamento");
    titulo(s, `Quiz relâmpago ${k + 1}/${quizIdx.length}`);
    circulo(s, 8.9, 0.45, 0.55, HEX.amb, iQ);
    s.addText(q.q, { x: 0.5, y: 1.25, w: 9, h: 1.0, fontSize: 18, bold: true, color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: "pergunta" });
    q.ops.forEach((o, i) => {
      const x = 0.5 + (i % 2) * 4.6, y = 2.4 + Math.floor(i / 2) * 1.35;
      s.addShape(pres.ShapeType.roundRect, { x, y, w: 4.4, h: 1.2, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.1, objectName: "opcao " + "abcd"[i] });
      circulo(s, x + 0.12, y + 0.35, 0.5, col);
      s.addText("abcd"[i].toUpperCase(), { x: x + 0.12, y: y + 0.35, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "letra " + "abcd"[i] });
      s.addText(o, { x: x + 0.75, y: y + 0.05, w: 3.55, h: 1.1, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "texto opcao " + "abcd"[i] });
    });
    s.addNotes(`RESPOSTA: ${"ABCD"[q.ok]}. ${q.why} Dê 30 segundos para a turma votar levantando a mão e peça que alguém justifique.`);
  });

  // ---- MÃO NA MASSA
  if (info.exs.length || info.proj) {
    const s = nova("Prática e fechamento");
    titulo(s, "Mão na massa: Python + scikit-learn");
    circulo(s, 8.9, 0.45, 0.55, col, iPlay);
    const lista = info.exs.slice(0, 4);
    lista.forEach((e, i) => {
      const y = 1.4 + i * 0.62;
      s.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 9, h: 0.52, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.08, objectName: "exercicio " + (i + 1) });
      s.addText([{ text: `Exercício ${i + 1}  `, options: { bold: true, color: col } }, { text: e.t, options: { color: C.text1 } }, { text: `   (${e.nivel})`, options: { color: HEX.gry, italic: true } }], { x: 0.7, y, w: 8.6, h: 0.52, fontSize: 14, valign: "middle", margin: 0, isTextBox: true, objectName: "texto exercicio " + (i + 1) });
    });
    if (info.proj) {
      const y = 1.4 + lista.length * 0.62 + 0.05;
      s.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 9, h: 0.75, fill: { color: col }, line: { color: col, width: 0 }, rectRadius: 0.1, objectName: "projeto" });
      s.addText([{ text: "PROJETO  ", options: { bold: true, fontSize: 12, color: "FDE68A" } }, { text: info.proj, options: { bold: true, fontSize: 16, color: "FFFFFF" } }], { x: 0.75, y, w: 8.5, h: 0.75, valign: "middle", margin: 0, isTextBox: true, objectName: "titulo projeto" });
    }
    s.addNotes("Os enunciados completos e as soluções estão nos Apêndices D e E do livro. Sugestão: resolver o exercício 1 em dupla, em sala.");
  }

  // ---- RESUMO
  {
    const s = nova("Prática e fechamento", true);
    titulo(s, "Para levar na mochila");
    circulo(s, 8.9, 0.45, 0.55, HEX.pnk || "EC4899", iBook);
    const itens = info.resumo.slice(0, 7);
    const h = Math.min(0.62, 3.7 / Math.max(itens.length, 1));
    itens.forEach((r, i) => {
      const y = 1.35 + i * (h + 0.04);
      s.addText([{ text: "✔  ", options: { color: "34D399", bold: true } }, { text: r, options: { color: "FFFFFF" } }], { x: 0.5, y, w: 9, h, fontSize: 14, valign: "middle", margin: 0, isTextBox: true, objectName: "resumo " + (i + 1) });
    });
    s.addNotes("Revisão em voz alta. Peça a um aluno que explique cada item com uma analogia diferente da usada em aula.");
  }
  // ---- FECHAMENTO
  {
    const s = pres.addSlide({ masterName: "ESCURO", sectionTitle: "Prática e fechamento" });
    circulo(s, 0.8, 1.0, 0.8, col, iFlag);
    s.addText(spec.fechamento, { x: 0.8, y: 2.0, w: 8.4, h: 1.7, fontSize: 32, bold: true, color: "FFFFFF", valign: "top", margin: 0, isTextBox: true, objectName: "fechamento" });
    s.addText(spec.proximo, { x: 0.8, y: 3.9, w: 8.4, h: 1.0, fontSize: 18, color: "67E8F9", italic: true, valign: "top", margin: 0, isTextBox: true, objectName: "proximo capitulo" });
    s.addNotes("Encerramento. Anuncie a leitura recomendada e o próximo tema.");
  }

  const fname = path.join(OUT, `Cap${nn(n)}_${spec.arquivo}.pptx`);
  await pres.writeFile({ fileName: fname });
  await applyTheme(fname, THEME);
  return fname;
}

(async () => {
  const args = process.argv.slice(2).map(Number).filter(Boolean);
  const lista = args.length ? args : Array.from({ length: 12 }, (_, i) => i + 1).filter((n) => fs.existsSync(path.join(__dirname, "content", `ch${nn(n)}.js`)));
  for (const n of lista) console.log("ok", await buildDeck(n));
})().catch((e) => { console.error(e); process.exit(1); });
