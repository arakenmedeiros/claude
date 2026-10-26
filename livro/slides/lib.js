// Biblioteca de geração de slides didáticos (pptxgenjs) para o livro "Aprendizado Não Supervisionado"
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const SKILL = fs.readdirSync("/root/.claude/skills/synced").map((d) => path.join("/root/.claude/skills/synced", d, "pptx", "scripts", "apply_theme.js")).find((p) => fs.existsSync(p));
const { applyTheme } = require(SKILL);

const FIGS = path.join(__dirname, "..", "figs");
const HEAD = "Cambria", BODY = "Calibri", MONO = "Courier New";

const THEME = {
  name: "Clusters",
  headFontFace: HEAD,
  bodyFontFace: BODY,
  colors: { dk1: "1F2937", lt1: "FFFFFF", dk2: "111827", lt2: "F3F4F6", accent1: "4F46E5", accent2: "F59E0B", accent3: "10B981", accent4: "06B6D4", accent5: "8B5CF6", accent6: "6B7280", hlink: "4F46E5", folHlink: "8B5CF6" },
};
const HEX = { accent1: "4F46E5", accent2: "F59E0B", accent3: "10B981", accent4: "06B6D4", accent5: "8B5CF6", accent6: "6B7280", text1: "1F2937", background1: "FFFFFF" };

// pptxgenjs escreve <a:pPr> em cada run; mantém só o primeiro de cada parágrafo
async function limparPPr(arquivo) {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(arquivo));
  for (const nome of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    let x = await zip.file(nome).async("string");
    x = x.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, corpo) => {
      let n = 0;
      return "<a:p>" + corpo.replace(/<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g, (p) => (n++ === 0 ? p : "")) + "</a:p>";
    });
    zip.file(nome, x);
  }
  fs.writeFileSync(arquivo, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

// ------------------------------------------------------------------ ícones
const iconCache = new Map();
async function icon(name, hex) {
  const key = name + hex;
  if (iconCache.has(key)) return iconCache.get(key);
  const comp = fa[name] || fa.FaStar;
  const svg = RDS.renderToStaticMarkup(React.createElement(comp, { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } }).png().toBuffer();
  const data = "image/png;base64," + buf.toString("base64");
  iconCache.set(key, data);
  return data;
}
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

// ------------------------------------------------------------------ verificação aproximada de ajuste de texto
const avisos = [];
function medir(texto, wIn, pt, bold = false) {
  const larg = pt * (bold ? 0.56 : 0.51); // pontos por caractere (estimativa conservadora)
  const porLinha = Math.max(1, Math.floor((wIn * 72) / larg));
  return texto.split("\n").reduce((n, ln) => n + Math.max(1, Math.ceil(ln.length / porLinha)), 0);
}
function checar(rotulo, texto, wIn, hIn, pt, opts = {}) {
  const linhas = medir(texto, wIn, pt, opts.bold);
  const alt = (linhas * pt * 1.2) / 72 + (opts.extra || 0);
  if (alt > hIn + 0.02) avisos.push(`${rotulo}: precisa de ${alt.toFixed(2)}" mas há ${hIn.toFixed(2)}" -> "${texto.slice(0, 50)}..."`);
}

// maior tamanho de fonte (pt) em que os parágrafos cabem na caixa
function ajusta(itens, wIn, hIn, { max = 22, min = 14, extraPar = 0.1, recuo = 0 } = {}) {
  for (let pt = max; pt >= min; pt--) {
    let alt = 0;
    for (const t of itens) alt += (medir(plano(t), wIn - recuo, pt) * pt * 1.2) / 72 + extraPar;
    if (alt <= hIn) return pt;
  }
  return min;
}
// "**negrito**" -> runs
function runs(texto, base = {}) {
  const partes = texto.split("**");
  const out = [];
  partes.forEach((p, i) => { if (p) out.push({ text: p, options: { ...base, bold: i % 2 === 1 ? true : base.bold } }); });
  return out.length ? out : [{ text: "", options: base }];
}
const plano = (t) => t.replace(/\*\*/g, "");
// parágrafos: propriedades de parágrafo na primeira run; quebra de linha na última run de cada item
function paras(itens, base = {}, perPara = {}) {
  const out = [];
  itens.forEach((it, i) => {
    const partes = runs(it, base);
    partes.forEach((r, j) => {
      if (j === 0) r.options = { ...r.options, ...perPara };
      if (j === partes.length - 1 && i < itens.length - 1) r.options = { ...r.options, breakLine: true };
    });
    out.push(...partes);
  });
  return out;
}

// ------------------------------------------------------------------ deck
async function criarDeck({ capNum, capTitulo, subtitulo, slides, arquivo, autor }) {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.author = autor; pres.title = `Cap. ${capNum} — ${capTitulo}`; pres.subject = "Aprendizado Não Supervisionado";
  pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };
  const C = pres.SchemeColor;
  const rodape = `Aprendizado Não Supervisionado  ·  Capítulo ${capNum}`;
  const tituloPh = (cor) => ({ placeholder: { options: { name: "title", type: "title", x: 1.3, y: 0.3, w: 8.2, h: 0.8, fontFace: HEAD, fontSize: 34, bold: true, color: cor, align: "left", valign: "middle", margin: 0 }, text: "" } });

  pres.defineSlideMaster({ title: "CONTEUDO", background: { color: C.background1 }, objects: [tituloPh(C.text1), { text: { text: rodape, options: { x: 0.5, y: 5.25, w: 6, h: 0.22, fontSize: 10, fontFace: BODY, color: C.accent6, margin: 0 } } }], slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.22, fontSize: 10, fontFace: BODY, color: C.accent6, align: "right", margin: 0 } });
  pres.defineSlideMaster({ title: "ESCURO", background: { color: C.text2 }, objects: [tituloPh(C.background1), { text: { text: rodape, options: { x: 0.5, y: 5.25, w: 6, h: 0.22, fontSize: 10, fontFace: BODY, color: C.background2, margin: 0 } } }], slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.22, fontSize: 10, fontFace: BODY, color: C.background2, align: "right", margin: 0 } });
  pres.defineSlideMaster({ title: "CAPA", background: { color: C.text2 }, objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 5.6, h: 1.9, fontFace: HEAD, fontSize: 38, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.6, w: 5.4, h: 1.0, fontFace: BODY, fontSize: 18, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ] });

  pres.addSection({ title: "Abertura" });
  let secaoAtual = "Abertura";
  const sec = (s) => { if (s && s !== secaoAtual) { pres.addSection({ title: s }); secaoAtual = s; } };
  const nova = (master, s, titulo) => {
    const sl = pres.addSlide({ masterName: master, sectionTitle: secaoAtual });
    if (titulo) { checar(`slide "${titulo}" título`, titulo, 8.2, 0.8, 34, { bold: true }); sl.addText(titulo, { placeholder: "title" }); }
    if (s.nota) sl.addNotes(s.nota);
    return sl;
  };
  const circ = async (sl, x, y, d, cor, ic, nome) => {
    sl.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: cor }, line: { color: cor, width: 0 }, objectName: nome || "circulo" });
    sl.addImage({ data: await icon(ic, "FFFFFF"), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, altText: "ícone", objectName: "icone" });
  };
  const cabecalho = async (sl, s, ic, cor) => { await circ(sl, 0.5, 0.4, 0.6, cor || C.accent1, ic || "FaBook", "icone-titulo"); };
  const frase = async (sl, texto, y = 4.55) => {
    checar("frase de impacto", texto, 8.2, 0.5, 16, { bold: true });
    sl.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 9, h: 0.52, rectRadius: 0.1, fill: { color: C.accent2, transparency: 82 }, line: { color: C.accent2, width: 0 }, objectName: "faixa-frase" });
    sl.addImage({ data: await icon("FaBolt", HEX.accent2), x: 0.65, y: y + 0.12, w: 0.28, h: 0.28, altText: "destaque", objectName: "icone-frase" });
    sl.addText(texto, { x: 1.1, y, w: 8.3, h: 0.52, fontSize: 16, bold: true, italic: true, fontFace: BODY, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "frase" });
  };
  const figura = (sl, fig, x, y, w, h, nome) => {
    const f = path.join(FIGS, fig + ".png");
    if (!fs.existsSync(f)) throw new Error("figura ausente: " + fig);
    const d = pngSize(f); const r = Math.min(w / d.w, h / d.h);
    const fw = d.w * r, fh = d.h * r;
    sl.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.background2, width: 1.25 }, objectName: "moldura-figura" });
    sl.addImage({ path: f, x: x + (w - fw) / 2, y: y + (h - fh) / 2, w: fw, h: fh, altText: nome || fig, objectName: "figura-" + fig });
  };
  const CORES = [C.accent1, C.accent3, C.accent4, C.accent5, C.accent2];

  // ---------------------------------------------------------------- tipos
  const tipos = {
    async capa(s) {
      const sl = pres.addSlide({ masterName: "CAPA", sectionTitle: "Abertura" });
      // motivo: aglomerados de pontos
      let seed = capNum * 97 + 13; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
      const centros = [[7.2, 1.3, C.accent1], [8.6, 3.2, C.accent4], [6.6, 3.9, C.accent5], [8.9, 1.0, C.accent3], [7.6, 2.5, C.accent2]];
      centros.forEach(([cx, cy, cor], k) => {
        sl.addShape(pres.ShapeType.ellipse, { x: cx - 0.9, y: cy - 0.9, w: 1.8, h: 1.8, fill: { color: cor, transparency: 88 }, line: { color: cor, width: 0 }, objectName: `halo-${k}` });
        for (let i = 0; i < 16; i++) {
          const a = rnd() * 6.283, r = Math.sqrt(rnd()) * 0.8, d = 0.07 + rnd() * 0.12;
          sl.addShape(pres.ShapeType.ellipse, { x: cx + r * Math.cos(a) - d / 2, y: cy + r * Math.sin(a) - d / 2, w: d, h: d, fill: { color: cor }, line: { color: cor, width: 0 }, objectName: `ponto-${k}-${i}` });
        }
      });
      sl.addText(`CAPÍTULO ${capNum}`, { x: 0.6, y: 0.95, w: 5, h: 0.4, fontSize: 18, bold: true, charSpacing: 6, fontFace: BODY, color: C.accent2, margin: 0, isTextBox: true, objectName: "rotulo-capitulo" });
      checar("capa título", capTitulo, 5.6, 1.9, 38, { bold: true });
      sl.addText(capTitulo, { placeholder: "title" });
      checar("capa subtítulo", subtitulo, 5.4, 1.0, 18);
      sl.addText(subtitulo, { placeholder: "body" });
      sl.addText(`${autor}  ·  Aprendizado Não Supervisionado`, { x: 0.6, y: 5.0, w: 8, h: 0.3, fontSize: 12, fontFace: BODY, color: C.background2, margin: 0, isTextBox: true, objectName: "autor" });
      if (s.nota) sl.addNotes(s.nota);
    },
    async objetivos(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo || "O que você vai aprender");
      await cabecalho(sl, s, "FaBullseye", C.accent1);
      const n = s.itens.length, h = Math.min(0.85, 3.7 / n);
      const pto = Math.min(...s.itens.map((t) => ajusta([t], 8.3, h - 0.05, { max: 22, extraPar: 0 })));
      for (let i = 0; i < n; i++) {
        const y = 1.35 + i * (h + 0.02);
        checar("objetivo", s.itens[i], 8.0, h, pto);
        sl.addImage({ data: await icon("FaCheckCircle", HEX.accent3), x: 0.55, y: y + h / 2 - 0.17, w: 0.34, h: 0.34, altText: "marca", objectName: "marca" });
        sl.addText(runs(s.itens[i]), { x: 1.1, y, w: 8.3, h, fontSize: pto, fontFace: BODY, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: `objetivo-${i + 1}` });
      }
    },
    async divisor(s) {
      sec(s.secao);
      const sl = nova("ESCURO", s, null);
      sl.addText(s.numero || "", { x: 0.6, y: 1.2, w: 3, h: 1.6, fontSize: 96, bold: true, fontFace: HEAD, color: C.accent2, margin: 0, isTextBox: true, objectName: "numero" });
      checar("divisor", s.titulo, 8.5, 1.2, 40, { bold: true });
      sl.addText(s.titulo, { x: 0.6, y: 2.9, w: 8.8, h: 1.1, fontSize: 40, bold: true, fontFace: HEAD, color: C.background1, margin: 0, isTextBox: true, valign: "top", objectName: "titulo-divisor" });
      if (s.sub) sl.addText(s.sub, { x: 0.6, y: 4.1, w: 8.8, h: 0.7, fontSize: 18, fontFace: BODY, color: C.background2, margin: 0, isTextBox: true, valign: "top", objectName: "subtitulo-divisor" });
    },
    async analogia(s) {
      sec(s.secao);
      const sl = nova("ESCURO", s, s.titulo);
      await circ(sl, 0.5, 0.4, 0.6, C.accent2, s.icone || "FaLightbulb", "icone-titulo");
      const temFig = !!s.fig; const w = temFig ? 4.6 : 9;
      const pta = ajusta([s.historia], w, 2.3, { max: 24, extraPar: 0 });
      checar("analogia texto", plano(s.historia), w, 2.3, pta);
      sl.addText(runs(s.historia), { x: 0.5, y: 1.4, w, h: 2.3, fontSize: pta, fontFace: BODY, color: C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "historia", paraSpaceAfter: 6 });
      if (temFig) figura(sl, s.fig, 5.3, 1.35, 4.2, 2.6, s.figAlt);
      checar("analogia frase", s.frase, 9, 1.0, 26, { bold: true });
      sl.addText(s.frase, { x: 0.5, y: 4.0, w: 9, h: 1.05, fontSize: 26, bold: true, italic: true, fontFace: HEAD, color: C.accent2, valign: "middle", margin: 0, isTextBox: true, objectName: "frase-central" });
    },
    async conceito(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone, s.cor || C.accent1);
      const temFrase = !!s.frase; const yMax = temFrase ? 4.4 : 5.0;
      if (s.fig) {
        const ptc = ajusta(s.pontos, 4.5, yMax - 1.4, { max: 20, extraPar: 0.14, recuo: 0.3 });
        checar("conceito texto", s.pontos.map(plano).join("\n"), 4.2, yMax - 1.35, ptc, { extra: 0.14 * s.pontos.length });
        sl.addText(paras(s.pontos, {}, { bullet: true, paraSpaceAfter: 8 }), { x: 0.5, y: 1.35, w: 4.5, h: yMax - 1.35, fontSize: ptc, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "pontos" });
        figura(sl, s.fig, 5.2, 1.35, 4.3, yMax - 1.35, s.figAlt);
      } else {
        const n = s.pontos.length; const cols = n <= 3 ? n : n === 4 ? 2 : 3; const lin = Math.ceil(n / cols);
        const gap = 0.25, cw = (9 - gap * (cols - 1)) / cols, ch = (yMax - 1.35 - gap * (lin - 1)) / lin;
        const ptk = Math.min(...s.pontos.map((t) => ajusta([t], cw - 0.4, ch - 0.85, { max: 20, extraPar: 0 })));
        for (let i = 0; i < n; i++) {
          const x = 0.5 + (i % cols) * (cw + gap), y = 1.35 + Math.floor(i / cols) * (ch + gap);
          checar("cartão", plano(s.pontos[i]), cw - 0.4, ch - 0.7, ptk);
          sl.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, rectRadius: 0.1, fill: { color: CORES[i % 5], transparency: 90 }, line: { color: CORES[i % 5], width: 0 }, objectName: `cartao-${i + 1}` });
          sl.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: y + 0.18, w: 0.42, h: 0.42, fill: { color: CORES[i % 5] }, line: { color: CORES[i % 5], width: 0 }, objectName: `numero-${i + 1}` });
          sl.addText(String(i + 1), { x: x + 0.2, y: y + 0.18, w: 0.42, h: 0.42, fontSize: 16, bold: true, fontFace: BODY, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `n-${i + 1}` });
          sl.addText(runs(s.pontos[i]), { x: x + 0.2, y: y + 0.7, w: cw - 0.4, h: ch - 0.8, fontSize: ptk, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: `texto-${i + 1}` });
        }
      }
      if (temFrase) await frase(sl, s.frase);
    },
    async figura(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone || "FaImage", s.cor || C.accent4);
      const yMax = s.frase ? 4.4 : 5.0;
      figura(sl, s.fig, 0.9, 1.3, 8.2, yMax - 1.3, s.figAlt);
      if (s.frase) await frase(sl, s.frase);
    },
    async passos(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone || "FaSortAmountDown", C.accent1);
      const n = s.passos.length; const yMax = s.frase ? 4.4 : 5.05; const h = Math.min(0.95, (yMax - 1.3) / n);
      const ptp = Math.min(...s.passos.map((p) => ajusta([p.t + (p.d ? "  " + p.d : "")], 8.3, h - 0.04, { max: 20, extraPar: 0 })));
      for (let i = 0; i < n; i++) {
        const y = 1.3 + i * h; const p = s.passos[i];
        checar(`passo ${i + 1}`, plano(p.t + " " + (p.d || "")), 8.0, h, ptp, { bold: false });
        sl.addShape(pres.ShapeType.ellipse, { x: 0.5, y: y + h / 2 - 0.23, w: 0.46, h: 0.46, fill: { color: CORES[i % 5] }, line: { color: CORES[i % 5], width: 0 }, objectName: `passo-circulo-${i + 1}` });
        sl.addText(String(i + 1), { x: 0.5, y: y + h / 2 - 0.23, w: 0.46, h: 0.46, fontSize: 16, bold: true, fontFace: BODY, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `passo-n-${i + 1}` });
        sl.addText([{ text: p.t, options: { bold: true, color: C.text1, breakLine: !!p.d && p.quebra } }, ...(p.d ? [{ text: (p.quebra ? "" : "  ") + p.d, options: { color: C.accent6 } }] : [])], { x: 1.2, y, w: 8.3, h, fontSize: ptp, fontFace: BODY, valign: "middle", margin: 0, isTextBox: true, objectName: `passo-${i + 1}` });
      }
      if (s.frase) await frase(sl, s.frase);
    },
    async comparar(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone || "FaBalanceScale", C.accent5);
      const n = s.colunas.length, gap = 0.25, cw = (9 - gap * (n - 1)) / n, yMax = s.frase ? 4.4 : 5.05;
      const ptq = Math.min(...s.colunas.map((c) => ajusta(c.itens, cw - 0.4, yMax - 1.3 - 0.85, { max: 18, extraPar: 0.09, recuo: 0.3 })));
      for (let i = 0; i < n; i++) {
        const c = s.colunas[i], x = 0.5 + i * (cw + gap), cor = CORES[i % 5];
        sl.addShape(pres.ShapeType.roundRect, { x, y: 1.3, w: cw, h: yMax - 1.3, rectRadius: 0.1, fill: { color: cor, transparency: 90 }, line: { color: cor, width: 0 }, objectName: `coluna-${i + 1}` });
        sl.addShape(pres.ShapeType.roundRect, { x, y: 1.3, w: cw, h: 0.5, rectRadius: 0.1, fill: { color: cor }, line: { color: cor, width: 0 }, objectName: `cabeca-${i + 1}` });
        checar("cabeça coluna", c.t, cw - 0.2, 0.5, 16, { bold: true });
        sl.addText(c.t, { x: x + 0.1, y: 1.3, w: cw - 0.2, h: 0.5, fontSize: 16, bold: true, fontFace: BODY, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `titulo-coluna-${i + 1}` });
        checar("itens coluna", c.itens.map(plano).join("\n"), cw - 0.4, yMax - 1.3 - 0.7, ptq, { extra: 0.09 * c.itens.length });
        sl.addText(paras(c.itens, {}, { bullet: true, paraSpaceAfter: 6 }), { x: x + 0.2, y: 1.95, w: cw - 0.4, h: yMax - 1.3 - 0.75, fontSize: ptq, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: `itens-${i + 1}` });
      }
      if (s.frase) await frase(sl, s.frase);
    },
    async tabela(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone || "FaThList", C.accent3);
      const nc = s.cab.length; const larg = s.larg || new Array(nc).fill(9 / nc);
      const altDisp = s.frase ? 3.0 : 3.6; const altLinha0 = Math.min(0.75, altDisp / (s.linhas.length + 1));
      let linhaPt = 14;
      for (const pt of [20, 18, 16, 15, 14]) {
        const cabs = s.cab.every((c, j) => medir(c, larg[j] - 0.2, pt, true) * pt * 1.2 / 72 <= altLinha0 - 0.06);
        const corpos = s.linhas.every((l) => l.every((c, j) => medir(plano(c), larg[j] - 0.25, pt) * pt * 1.2 / 72 <= altLinha0 - 0.06));
        if (cabs && corpos) { linhaPt = pt; break; }
      }
      const cabecalhoRow = s.cab.map((c) => ({ text: c, options: { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: linhaPt, fontFace: BODY, valign: "middle" } }));
      const corpo = s.linhas.map((l, i) => l.map((c, j) => ({ text: plano(c), options: { bold: j === 0, color: C.text1, fill: { color: i % 2 ? C.background2 : C.background1 }, fontSize: linhaPt, fontFace: BODY, valign: "middle" } })));
      const altLinha = altLinha0;
      s.linhas.forEach((l) => l.forEach((c, j) => checar("célula", plano(c), larg[j] - 0.2, altLinha * 1.0, linhaPt)));
      sl.addTable([cabecalhoRow, ...corpo], { x: 0.5, y: 1.35, w: 9, colW: larg, rowH: altLinha, border: { type: "solid", pt: 0.75, color: "E5E7EB" }, margin: [0.04, 0.1, 0.04, 0.1], objectName: "tabela" });
      if (s.frase) await frase(sl, s.frase);
    },
    async formula(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone || "FaRuler", C.accent5);
      sl.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.3, w: 9, h: 1.15, rectRadius: 0.1, fill: { color: C.accent1, transparency: 90 }, line: { color: C.accent1, width: 0 }, objectName: "painel-formula" });
      checar("fórmula", s.formula, 8.6, 1.1, 26);
      sl.addText(s.formula, { x: 0.7, y: 1.3, w: 8.6, h: 1.15, fontSize: 26, italic: true, fontFace: HEAD, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "formula" });
      const itens = s.explicacao; const yMax = s.frase ? 4.4 : 5.05;
      const ptf = ajusta(itens, 9, yMax - 2.65, { max: 20, extraPar: 0.14, recuo: 0.3 });
      checar("explicação", itens.map(plano).join("\n"), 8.7, yMax - 2.65, ptf, { extra: 0.14 * itens.length });
      sl.addText(paras(itens, {}, { bullet: true, paraSpaceAfter: 8 }), { x: 0.5, y: 2.65, w: 9, h: yMax - 2.65, fontSize: ptf, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "explicacao" });
      if (s.frase) await frase(sl, s.frase);
    },
    async codigo(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, "FaCode", C.accent4);
      const linhas = s.codigo.split("\n"); const maxc = Math.max(...linhas.map((l) => l.length));
      const wMax = s.dicas ? 5.5 : 8.6;
      let pt = 11;
      for (const p of [18, 16, 14, 13, 12, 11]) { if (maxc * p * 0.6 / 72 <= wMax && 0.3 + linhas.length * p * 1.25 / 72 <= 3.65) { pt = p; break; } }
      const wCod = s.dicas ? 5.9 : 9; const hCod = Math.min(3.65, 0.3 + linhas.length * (pt * 1.25) / 72);
      if (maxc * pt * 0.6 / 72 > wCod - 0.4) avisos.push(`código "${s.titulo}": linha de ${maxc} caracteres não cabe em ${wCod}"`);
      sl.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.35, w: wCod, h: hCod, rectRadius: 0.08, fill: { color: C.text2 }, line: { color: C.text2, width: 0 }, objectName: "painel-codigo" });
      sl.addText(linhas.map((l, i) => ({ text: l || " ", options: { breakLine: i < linhas.length - 1 } })), { x: 0.7, y: 1.45, w: wCod - 0.4, h: hCod - 0.2, fontSize: pt, fontFace: MONO, color: C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "codigo" });
      if (s.dicas) {
        sl.addText([{ text: "O que cada parte faz", options: { bold: true, color: C.accent1, breakLine: true, fontSize: 16, paraSpaceAfter: 6 } }, ...paras(s.dicas, { fontSize: 14 }, { bullet: true, paraSpaceAfter: 6 })], { x: 6.65, y: 1.35, w: 2.85, h: 3.2, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "dicas-codigo" });
        checar("dicas código", s.dicas.map(plano).join("\n"), 2.85, 2.7, 14, { extra: 0.06 * s.dicas.length });
      }
      if (s.frase) await frase(sl, s.frase);
    },
    async numeros(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone || "FaChartLine", C.accent2);
      const n = s.stats.length, gap = 0.3, cw = (9 - gap * (n - 1)) / n, yMax = s.frase ? 4.4 : 5.0;
      for (let i = 0; i < n; i++) {
        const x = 0.5 + i * (cw + gap), st = s.stats[i], cor = CORES[i % 5];
        sl.addShape(pres.ShapeType.roundRect, { x, y: 1.4, w: cw, h: yMax - 1.4, rectRadius: 0.1, fill: { color: cor, transparency: 90 }, line: { color: cor, width: 0 }, objectName: `estatistica-${i + 1}` });
        checar("número grande", st.n, cw - 0.2, 1.1, 48, { bold: true });
        sl.addText(st.n, { x: x + 0.1, y: 1.55, w: cw - 0.2, h: 1.1, fontSize: 48, bold: true, fontFace: HEAD, color: cor, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `numero-${i + 1}` });
        checar("rótulo número", st.r, cw - 0.4, yMax - 2.8, 15);
        sl.addText(st.r, { x: x + 0.2, y: 2.75, w: cw - 0.4, h: yMax - 2.85, fontSize: 15, fontFace: BODY, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: `rotulo-${i + 1}` });
      }
      if (s.frase) await frase(sl, s.frase);
    },
    async memorize(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo || "Cartões de memória");
      await cabecalho(sl, s, "FaBrain", C.accent5);
      const n = s.cartoes.length, cols = n <= 2 ? n : 2, lin = Math.ceil(n / cols), gap = 0.25;
      const cw = (9 - gap * (cols - 1)) / cols, ch = (3.75 - gap * (lin - 1)) / lin;
      const ptm = Math.min(...s.cartoes.map((c) => ajusta([c.r], cw - 0.3, ch - 0.7, { max: 20, extraPar: 0 })));
      for (let i = 0; i < n; i++) {
        const x = 0.5 + (i % cols) * (cw + gap), y = 1.3 + Math.floor(i / cols) * (ch + gap), c = s.cartoes[i], cor = CORES[i % 5];
        sl.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: ch, rectRadius: 0.1, fill: { color: cor, transparency: 90 }, line: { color: cor, width: 0 }, objectName: `cartao-${i + 1}` });
        sl.addShape(pres.ShapeType.roundRect, { x, y, w: cw, h: 0.5, rectRadius: 0.1, fill: { color: cor }, line: { color: cor, width: 0 }, objectName: `pergunta-fundo-${i + 1}` });
        checar("cartão pergunta", c.p, cw - 0.3, 0.5, 15, { bold: true });
        sl.addText(c.p, { x: x + 0.15, y, w: cw - 0.3, h: 0.5, fontSize: 15, bold: true, fontFace: BODY, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: `pergunta-${i + 1}` });
        checar("cartão resposta", plano(c.r), cw - 0.3, ch - 0.65, ptm);
        sl.addText(runs(c.r), { x: x + 0.15, y: y + 0.58, w: cw - 0.3, h: ch - 0.65, fontSize: ptm, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: `resposta-${i + 1}` });
      }
    },
    async quiz(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo || "Pergunta relâmpago");
      await cabecalho(sl, s, "FaQuestionCircle", C.accent2);
      checar("pergunta", s.p, 9, 0.9, 22, { bold: true });
      sl.addText(s.p, { x: 0.5, y: 1.3, w: 9, h: 0.9, fontSize: 22, bold: true, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "pergunta" });
      const rot = ["A", "B", "C", "D"]; const n = s.ops.length; const h = 0.55;
      for (let i = 0; i < n; i++) {
        const y = 2.25 + i * (h + 0.1), certa = i === s.certa;
        checar("opção", s.ops[i], 7.8, h, 18);
        sl.addShape(pres.ShapeType.roundRect, { x: 0.5, y, w: 9, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: `opcao-fundo-${i + 1}` });
        sl.addShape(pres.ShapeType.ellipse, { x: 0.62, y: y + 0.1, w: 0.35, h: 0.35, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: `opcao-letra-fundo-${i + 1}` });
        sl.addText(rot[i], { x: 0.62, y: y + 0.1, w: 0.35, h: 0.35, fontSize: 14, bold: true, fontFace: BODY, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `opcao-letra-${i + 1}` });
        sl.addText(s.ops[i], { x: 1.15, y, w: 8.2, h, fontSize: 18, fontFace: BODY, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: `opcao-${i + 1}` });
      }
      sl.addNotes(`${s.nota ? s.nota + "\n\n" : ""}RESPOSTA: ${rot[s.certa]}. ${s.expl}`);
    },
    async pratica(s) {
      sec(s.secao);
      const sl = nova("CONTEUDO", s, s.titulo);
      await cabecalho(sl, s, s.icone || "FaFlask", C.accent3);
      sl.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.3, w: 5.6, h: 3.75, rectRadius: 0.1, fill: { color: C.accent3, transparency: 90 }, line: { color: C.accent3, width: 0 }, objectName: "painel-missao" });
      sl.addText("Missão", { x: 0.7, y: 1.4, w: 5.2, h: 0.4, fontSize: 18, bold: true, fontFace: BODY, color: C.text1, margin: 0, isTextBox: true, objectName: "rotulo-missao" });
      const ptM = ajusta(s.missao, 5.2, 3.05, { max: 20, extraPar: 0.12, recuo: 0.3 });
      checar("missão", s.missao.map(plano).join("\n"), 4.9, 3.1, ptM, { extra: 0.12 * s.missao.length });
      sl.addText(paras(s.missao, {}, { bullet: { type: "number" }, paraSpaceAfter: 6 }), { x: 0.7, y: 1.85, w: 5.2, h: 3.1, fontSize: ptM, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "passos-missao" });
      sl.addShape(pres.ShapeType.roundRect, { x: 6.35, y: 1.3, w: 3.15, h: 3.75, rectRadius: 0.1, fill: { color: C.accent2, transparency: 85 }, line: { color: C.accent2, width: 0 }, objectName: "painel-entrega" });
      sl.addText(s.entrega.titulo || "Entrega", { x: 6.55, y: 1.4, w: 2.8, h: 0.4, fontSize: 18, bold: true, fontFace: BODY, color: C.text1, margin: 0, isTextBox: true, objectName: "rotulo-entrega" });
      const ptE = ajusta(s.entrega.itens, 2.8, 3.05, { max: 18, extraPar: 0.12, recuo: 0.3 });
      checar("entrega", s.entrega.itens.map(plano).join("\n"), 2.5, 3.1, ptE, { extra: 0.12 * s.entrega.itens.length });
      sl.addText(paras(s.entrega.itens, {}, { bullet: true, paraSpaceAfter: 6 }), { x: 6.55, y: 1.85, w: 2.8, h: 3.1, fontSize: ptE, fontFace: BODY, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "itens-entrega" });
    },
    async fim(s) {
      sec(s.secao);
      const sl = nova("ESCURO", s, s.titulo || "Para guardar");
      await circ(sl, 0.5, 0.4, 0.6, C.accent2, "FaFlagCheckered", "icone-titulo");
      const n = s.pontos.length, h = Math.min(0.75, 3.25 / n);
      const ptz = Math.min(...s.pontos.map((t) => ajusta([t], 8.3, h - 0.04, { max: 20, extraPar: 0 })));
      for (let i = 0; i < n; i++) {
        const y = 1.35 + i * (h + 0.04);
        checar("ponto final", plano(s.pontos[i]), 8.2, h, ptz);
        sl.addImage({ data: await icon("FaCheckCircle", HEX.accent3), x: 0.55, y: y + h / 2 - 0.15, w: 0.3, h: 0.3, altText: "marca", objectName: "marca" });
        sl.addText(runs(s.pontos[i]), { x: 1.05, y, w: 8.4, h, fontSize: ptz, fontFace: BODY, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: `ponto-${i + 1}` });
      }
      if (s.proximo) {
        checar("próximo", s.proximo, 8.8, 0.45, 15);
        sl.addText(s.proximo, { x: 0.5, y: 4.7, w: 9, h: 0.45, fontSize: 15, italic: true, fontFace: BODY, color: C.accent2, valign: "middle", margin: 0, isTextBox: true, objectName: "proximo" });
      }
    },
  };

  for (const s of slides) {
    if (!tipos[s.tipo]) throw new Error("tipo desconhecido: " + s.tipo);
    await tipos[s.tipo](s);
  }
  await pres.writeFile({ fileName: arquivo });
  await limparPPr(arquivo);
  await applyTheme(arquivo, THEME);
  return { n: slides.length, avisos: avisos.splice(0) };
}

module.exports = { criarDeck };
