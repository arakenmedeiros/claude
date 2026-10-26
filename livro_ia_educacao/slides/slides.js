// Gerador de PPTX por capítulo: node slides.js N [N...]   (lê json/chNN.json)
const pptxgen = require("pptxgenjs");
const fs = require("fs"), path = require("path");
const sharp = require("sharp");
const { applyTheme } = require("/root/.claude/skills/synced/4cfcf78b-3b5e-47b6-8db8-09e9433975ff_d303de2b-5e35-4489-a93a-f157e923a1fb/pptx/scripts/apply_theme.js");
const ROOT = path.join(__dirname, ".."), FIGS = path.join(ROOT, "figs");
const THEME = { name: "IA na Educação", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "3B2A2A", lt1: "FFFFFF", dk2: "5A2E2A", lt2: "F7EFE6", accent1: "D9653B", accent2: "2A9D8F", accent3: "E9A23B", accent4: "6D3B6E", accent5: "3D7EA6", accent6: "6E9A4F", hlink: "3D7EA6", folHlink: "6D3B6E" } };
const PARTE_COR = { 1: "D9653B", 2: "2A9D8F", 3: "E9A23B", 4: "6D3B6E", 5: "3D7EA6", 6: "D95D7F", 7: "6E9A4F" };
const INK = "3B2A2A", DARK = "5A2E2A", LIGHT = "F7EFE6", GRY = "6B5F59", W = 13.333, H = 7.5;
const PARTE_DE = (n) => (n <= 5 ? 1 : n <= 10 ? 2 : n <= 17 ? 3 : n <= 23 ? 4 : n <= 26 ? 5 : n <= 30 ? 6 : 7);

async function build(n) {
  const nn = String(n).padStart(2, "0");
  const spec = JSON.parse(fs.readFileSync(path.join(__dirname, "json", `ch${nn}.json`), "utf8"));
  const COR = PARTE_COR[PARTE_DE(n)];
  const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };
  pres.author = "Araken de Medeiros Santos"; pres.title = `Capítulo ${n} — ${spec.titulo}`;
  const S = pres.SchemeColor;
  let secOpen = false;
  const addS = (bg, notes) => { const s = pres.addSlide(); s.background = { color: bg }; if (notes) s.addNotes(notes); return s; };
  const foot = (s, dark) => s.addText(`IA aplicada à Educação · Cap. ${n}`, { x: 0.6, y: 7.05, w: 6, h: 0.3, fontSize: 10, color: dark ? "E9D9C8" : GRY, fontFace: "Calibri", margin: 0, isTextBox: true });
  const title = (s, t, dark) => s.addText(t, { x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: t.length > 55 ? 28 : 34, bold: true, fontFace: "Cambria", color: dark ? "FFFFFF" : DARK, margin: 0, valign: "top", isTextBox: true });
  const fs_ = (txt, big, mid, small, l1, l2) => (txt.length > l2 ? small : txt.length > l1 ? mid : big);
  const circle = (s, x, y, d, label, col) => { s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: col }, line: { color: col } }); s.addText(String(label), { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, color: "FFFFFF", fontSize: d > 0.7 ? 22 : 16, fontFace: "Cambria", margin: 0, isTextBox: true }); };

  // ---- capa
  let s = addS(DARK, `Abertura do capítulo ${n}. Apresente o tema e peça que a turma pense em uma situação do dia a dia relacionada.`);
  s.addShape(pres.ShapeType.ellipse, { x: 9.2, y: -1.6, w: 6.5, h: 6.5, fill: { color: COR, transparency: 25 }, line: { color: COR, transparency: 25 } });
  s.addShape(pres.ShapeType.ellipse, { x: 10.6, y: 4.2, w: 4.2, h: 4.2, fill: { color: "E9A23B", transparency: 35 }, line: { color: "E9A23B", transparency: 35 } });
  const ban = path.join(FIGS, `banner${nn}.png`);
  s.addText(`CAPÍTULO ${n}`, { x: 0.8, y: 1.3, w: 8, h: 0.5, fontSize: 18, bold: true, color: "F4B860", charSpacing: 4, margin: 0, isTextBox: true });
  s.addText(spec.titulo, { x: 0.8, y: 1.9, w: 9.2, h: 2.6, fontSize: spec.titulo.length > 70 ? 34 : 40, bold: true, color: "FFFFFF", fontFace: "Cambria", margin: 0, valign: "top", isTextBox: true });
  s.addText(spec.parte || "", { x: 0.8, y: 4.7, w: 9, h: 0.5, fontSize: 18, color: "F7EFE6", italic: true, margin: 0, isTextBox: true });
  s.addText("Inteligência Artificial Aplicada à Educação · Araken de Medeiros Santos", { x: 0.8, y: 6.6, w: 10, h: 0.4, fontSize: 13, color: "E9D9C8", margin: 0, isTextBox: true });

  for (const sl of spec.slides) {
    const notes = sl.notas || "";
    if (sl.t === "objetivos") {
      s = addS("FFFFFF", notes); title(s, sl.titulo || "O que vamos aprender", false);
      const it = sl.itens.slice(0, 5), h = Math.min(1.0, 5.2 / it.length);
      it.forEach((t, i) => { const y = 1.6 + i * (h + 0.1); s.addShape(pres.ShapeType.roundRect, { x: 0.6, y, w: 12.1, h, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.12 });
        circle(s, 0.8, y + (h - 0.6) / 2, 0.6, i + 1, COR);
        s.addText(t, { x: 1.7, y, w: 10.8, h, fontSize: fs_(t, 20, 18, 16, 80, 120), color: INK, valign: "middle", margin: 0, isTextBox: true }); });
      foot(s);
    } else if (sl.t === "pergunta") {
      s = addS(COR, notes);
      s.addText(sl.rotulo || "PENSE UM POUCO", { x: 0.9, y: 1.0, w: 11, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", charSpacing: 4, margin: 0, isTextBox: true });
      s.addText(sl.texto, { x: 0.9, y: 1.8, w: 11.5, h: 3.6, fontSize: fs_(sl.texto, 38, 32, 26, 90, 160), bold: true, fontFace: "Cambria", color: "FFFFFF", margin: 0, valign: "middle", isTextBox: true });
      if (sl.apoio) s.addText(sl.apoio, { x: 0.9, y: 5.6, w: 11.5, h: 1.0, fontSize: 18, color: "FFF3E6", italic: true, margin: 0, isTextBox: true });
      foot(s, true);
    } else if (sl.t === "conceito") {
      s = addS("FFFFFF", notes); title(s, sl.titulo);
      s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.6, w: 7.3, h: 3.2, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.15 });
      s.addText("O QUE É", { x: 0.9, y: 1.75, w: 6.5, h: 0.4, fontSize: 13, bold: true, color: COR, charSpacing: 3, margin: 0, isTextBox: true });
      s.addText(sl.definicao, { x: 0.9, y: 2.2, w: 6.7, h: 2.5, fontSize: fs_(sl.definicao, 22, 20, 18, 120, 200), color: INK, valign: "top", margin: 0, isTextBox: true });
      s.addShape(pres.ShapeType.roundRect, { x: 8.2, y: 1.6, w: 4.5, h: 3.2, fill: { color: COR }, line: { color: COR }, rectRadius: 0.15 });
      s.addText("É COMO…", { x: 8.5, y: 1.75, w: 4, h: 0.4, fontSize: 13, bold: true, color: "FFFFFF", charSpacing: 3, margin: 0, isTextBox: true });
      s.addText(sl.analogia, { x: 8.5, y: 2.2, w: 3.95, h: 2.5, fontSize: fs_(sl.analogia, 20, 18, 16, 100, 170), color: "FFFFFF", valign: "top", margin: 0, isTextBox: true });
      if (sl.destaque) { s.addText("“" + sl.destaque + "”", { x: 0.6, y: 5.2, w: 12.1, h: 1.4, fontSize: fs_(sl.destaque, 28, 24, 20, 70, 110), italic: true, bold: true, fontFace: "Cambria", color: DARK, align: "center", valign: "middle", margin: 0, isTextBox: true }); }
      foot(s);
    } else if (sl.t === "cartoes") {
      s = addS("FFFFFF", notes); title(s, sl.titulo);
      const cs = sl.cartoes.slice(0, 4), k = cs.length, gap = 0.3, cw = (12.1 - gap * (k - 1)) / k;
      const cols = [COR, "2A9D8F", "6D3B6E", "3D7EA6"];
      cs.forEach((c, i) => { const x = 0.6 + i * (cw + gap);
        s.addShape(pres.ShapeType.roundRect, { x, y: 1.6, w: cw, h: 4.9, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.15, shadow: { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 } });
        circle(s, x + 0.25, 1.85, 0.7, i + 1, cols[(i + PARTE_DE(n)) % 4]);
        s.addText(c.cab, { x: x + 0.25, y: 2.7, w: cw - 0.5, h: 0.9, fontSize: k > 3 ? 18 : 20, bold: true, color: DARK, fontFace: "Cambria", margin: 0, valign: "top", isTextBox: true });
        s.addText(c.txt, { x: x + 0.25, y: 3.65, w: cw - 0.5, h: 2.7, fontSize: fs_(c.txt, k > 3 ? 15 : 17, 15, 14, 90, 140), color: INK, margin: 0, valign: "top", isTextBox: true }); });
      foot(s);
    } else if (sl.t === "passos") {
      s = addS("FFFFFF", notes); title(s, sl.titulo);
      const ps = sl.passos.slice(0, 5), k = ps.length, gap = 0.25, cw = (12.1 - gap * (k - 1)) / k;
      ps.forEach((p, i) => { const x = 0.6 + i * (cw + gap);
        s.addShape(pres.ShapeType.roundRect, { x, y: 2.1, w: cw, h: 3.6, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.15 });
        circle(s, x + cw / 2 - 0.35, 1.75, 0.7, i + 1, COR);
        s.addText(p, { x: x + 0.15, y: 2.6, w: cw - 0.3, h: 3.0, fontSize: fs_(p, k > 4 ? 15 : 17, 15, 14, 70, 110), color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true });
        if (i < k - 1) s.addText("›", { x: x + cw - 0.02, y: 3.5, w: gap + 0.04, h: 0.6, fontSize: 28, bold: true, color: COR, align: "center", margin: 0, isTextBox: true }); });
      if (sl.rodape) s.addText(sl.rodape, { x: 0.6, y: 6.0, w: 12.1, h: 0.8, fontSize: 18, italic: true, color: DARK, align: "center", valign: "middle", margin: 0, isTextBox: true });
      foot(s);
    } else if (sl.t === "comparar") {
      s = addS("FFFFFF", notes); title(s, sl.titulo);
      [[sl.esq, 0.6, "2A9D8F"], [sl.dir, 6.85, COR]].forEach(([c, x, col]) => {
        s.addShape(pres.ShapeType.roundRect, { x, y: 1.6, w: 5.9, h: 5.0, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.15 });
        s.addShape(pres.ShapeType.roundRect, { x, y: 1.6, w: 5.9, h: 0.8, fill: { color: col }, line: { color: col }, rectRadius: 0.15 });
        s.addText(c.cab, { x: x + 0.25, y: 1.6, w: 5.4, h: 0.8, fontSize: 22, bold: true, color: "FFFFFF", fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true });
        const it = c.itens.slice(0, 5);
        s.addText(it.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < it.length - 1, paraSpaceAfter: 8 } })), { x: x + 0.3, y: 2.6, w: 5.3, h: 3.9, fontSize: it.join("").length > 220 ? 15 : 17, color: INK, valign: "top", margin: 0, isTextBox: true }); });
      foot(s);
    } else if (sl.t === "figura") {
      s = addS("FFFFFF", notes); title(s, sl.titulo);
      const f = path.join(FIGS, sl.fig + ".png");
      if (fs.existsSync(f)) { const m = await sharp(f).metadata(); const maxW = 12.1, maxH = 4.85; let w = maxW, h = w * m.height / m.width; if (h > maxH) { h = maxH; w = h * m.width / m.height; }
        s.addImage({ path: f, x: 0.6 + (maxW - w) / 2, y: 1.45, w, h, altText: sl.alt || sl.legenda || sl.titulo }); }
      else console.error("FIGURA AUSENTE", sl.fig);
      if (sl.legenda) s.addText(sl.legenda, { x: 0.6, y: 6.4, w: 12.1, h: 0.6, fontSize: 14, italic: true, color: GRY, align: "center", margin: 0, isTextBox: true });
      foot(s);
    } else if (sl.t === "frase") {
      s = addS(DARK, notes);
      s.addText("“" + sl.texto + "”", { x: 1.0, y: 1.3, w: 11.3, h: 4.2, fontSize: fs_(sl.texto, 40, 34, 28, 80, 130), bold: true, italic: true, fontFace: "Cambria", color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
      if (sl.apoio) s.addText(sl.apoio, { x: 1.0, y: 5.7, w: 11.3, h: 0.9, fontSize: 18, color: "F4B860", align: "center", margin: 0, isTextBox: true });
      foot(s, true);
    } else if (sl.t === "atividade") {
      s = addS("FFFFFF", notes); title(s, sl.titulo);
      s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.55, w: 7.6, h: 5.1, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.15 });
      s.addText("PASSO A PASSO", { x: 0.9, y: 1.7, w: 6, h: 0.4, fontSize: 13, bold: true, color: COR, charSpacing: 3, margin: 0, isTextBox: true });
      const ps = sl.passos.slice(0, 6);
      s.addText(ps.map((t, i) => ({ text: t, options: { bullet: { type: "number" }, breakLine: i < ps.length - 1, paraSpaceAfter: 6 } })), { x: 0.9, y: 2.15, w: 7.0, h: 4.4, fontSize: ps.join("").length > 330 ? 14 : 16, color: INK, valign: "top", margin: 0, isTextBox: true });
      const side = [["TEMPO", sl.tempo, COR], ["PRODUTO", sl.produto, "2A9D8F"], ["POUCOS RECURSOS", sl.poucos, "6D3B6E"]].filter((r) => r[1]);
      const sh = 5.1 / side.length - 0.12;
      side.forEach(([lab, txt, col], i) => { const y = 1.55 + i * (sh + 0.18);
        s.addShape(pres.ShapeType.roundRect, { x: 8.5, y, w: 4.2, h: sh, fill: { color: col }, line: { color: col }, rectRadius: 0.15 });
        s.addText(lab, { x: 8.75, y: y + 0.12, w: 3.7, h: 0.35, fontSize: 12, bold: true, color: "FFFFFF", charSpacing: 3, margin: 0, isTextBox: true });
        s.addText(txt, { x: 8.75, y: y + 0.5, w: 3.7, h: sh - 0.6, fontSize: fs_(txt, 16, 14, 14, 80, 130), color: "FFFFFF", valign: "top", margin: 0, isTextBox: true }); });
      foot(s);
    } else if (sl.t === "cuidado") {
      s = addS("FFFFFF", notes); title(s, sl.titulo);
      const it = sl.itens.slice(0, 4), k = it.length, h = Math.min(1.15, 5.0 / k);
      it.forEach((t, i) => { const y = 1.6 + i * (h + 0.15);
        s.addShape(pres.ShapeType.roundRect, { x: 0.6, y, w: 12.1, h, fill: { color: "FDECE4" }, line: { color: "F2B8A0" }, rectRadius: 0.12 });
        s.addShape(pres.ShapeType.ellipse, { x: 0.85, y: y + (h - 0.6) / 2, w: 0.6, h: 0.6, fill: { color: "C0392B" }, line: { color: "C0392B" } });
        s.addText("!", { x: 0.85, y: y + (h - 0.6) / 2, w: 0.6, h: 0.6, align: "center", valign: "middle", bold: true, fontSize: 22, color: "FFFFFF", margin: 0, isTextBox: true });
        s.addText(t, { x: 1.75, y, w: 10.7, h, fontSize: fs_(t, 18, 16, 14, 90, 140), color: INK, valign: "middle", margin: 0, isTextBox: true }); });
      foot(s);
    } else if (sl.t === "lembrar") {
      s = addS(COR, notes); title(s, sl.titulo || "O que vale a pena lembrar", true);
      const it = sl.itens.slice(0, 6), k = it.length, h = Math.min(0.9, 5.2 / k - 0.12);
      it.forEach((t, i) => { const y = 1.55 + i * (h + 0.12);
        s.addShape(pres.ShapeType.roundRect, { x: 0.6, y, w: 12.1, h, fill: { color: "FFFFFF", transparency: 12 }, line: { color: "FFFFFF", transparency: 12 }, rectRadius: 0.12 });
        s.addText("✓", { x: 0.8, y, w: 0.6, h, align: "center", valign: "middle", bold: true, fontSize: 24, color: COR, margin: 0, isTextBox: true });
        s.addText(t, { x: 1.6, y, w: 10.9, h, fontSize: fs_(t, 18, 16, 14, 80, 130), color: INK, valign: "middle", bold: true, margin: 0, isTextBox: true }); });
      foot(s, true);
    } else if (sl.t === "reflexao") {
      s = addS("FFFFFF", notes); title(s, sl.titulo || "Para pensar na sua prática");
      const it = sl.perguntas.slice(0, 4), k = it.length, h = Math.min(1.2, 5.0 / k);
      it.forEach((t, i) => { const y = 1.6 + i * (h + 0.15);
        s.addShape(pres.ShapeType.roundRect, { x: 0.6, y, w: 12.1, h, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.12 });
        circle(s, 0.85, y + (h - 0.6) / 2, 0.6, "?", "6D3B6E");
        s.addText(t, { x: 1.75, y, w: 10.7, h, fontSize: fs_(t, 20, 18, 16, 90, 140), color: INK, valign: "middle", margin: 0, isTextBox: true }); });
      foot(s);
    } else console.error("tipo desconhecido", sl.t);
  }
  // encerramento
  s = addS(DARK, "Encerramento. Convide a turma a escolher uma ideia para experimentar nesta semana.");
  s.addShape(pres.ShapeType.ellipse, { x: -1.5, y: 4.2, w: 5, h: 5, fill: { color: COR, transparency: 30 }, line: { color: COR, transparency: 30 } });
  s.addText(spec.fecho || "Você decide. A IA apoia.", { x: 1.2, y: 2.0, w: 11, h: 2.2, fontSize: 44, bold: true, fontFace: "Cambria", color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText("A docência é insubstituível.", { x: 1.2, y: 4.5, w: 11, h: 0.7, fontSize: 22, italic: true, color: "F4B860", align: "center", margin: 0, isTextBox: true });
  s.addText("Obrigado(a)!", { x: 1.2, y: 5.5, w: 11, h: 0.6, fontSize: 18, color: "E9D9C8", align: "center", margin: 0, isTextBox: true });
  const out = path.join(ROOT, "saida", "pptx", `Cap${nn}_slides.pptx`);
  await pres.writeFile({ fileName: out }); await applyTheme(out, THEME);
  console.log("ok", out, spec.slides.length + 2, "slides");
}
(async () => { for (const a of process.argv.slice(2)) await build(+a); })();
